"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from rlpd.agents.agent import Agent
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal, Normal
from rlpd.networks import (
    MLP,
    Ensemble,
    MLPResNetV2,
    StateActionValue,
    subsample_ensemble,
)


# From https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=ap-zaOyKJDXM
def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.traverse_util.unflatten_dict(flat_mask)
    # return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


class SACLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)
    exterior_linear_c: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        actor_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        gradient_clipping_norm: Optional[float] = None,
        use_tanh_normal: bool = True,
        state_dependent_std: bool = False,
        exterior_linear_c: float = 0.0,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm
        )
        if use_tanh_normal:
            actor_def = TanhNormal(actor_base_cls, action_dim, state_dependent_std=state_dependent_std)
        else:
            actor_def = Normal(
                actor_base_cls,
                action_dim,
                state_dependent_std=state_dependent_std,
                squash_tanh=use_tanh_normal  ## Should be false 
            )
        actor_params = actor_def.init(actor_key, observations)["params"]
        if actor_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=actor_lr,
                weight_decay=actor_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=actor_lr)
        if gradient_clipping_norm:
            actor_optim = optax.chain(
                optax.clip_by_global_norm(gradient_clipping_norm),
                tx
            )
        else:
            actor_optim = tx
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=actor_optim,
        )

        if use_critic_resnet:
            critic_base_cls = partial(
                MLPResNetV2,
                num_blocks=1,
            )
        else:
            critic_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
        if gradient_clipping_norm:
            critic_optim = optax.chain(
                optax.clip_by_global_norm(gradient_clipping_norm),
                tx
            )
        else:
            critic_optim = tx
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=critic_optim,
        )
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        if gradient_clipping_norm:
            temp_optim = optax.chain(
                optax.clip_by_global_norm(gradient_clipping_norm),
                optax.adam(learning_rate=temp_lr)
            )
        else:
            temp_optim = optax.adam(learning_rate=temp_lr)
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=temp_optim,
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            exterior_linear_c=exterior_linear_c,
        )
    
    def reset_actor(self,
                    seed: int,
                    observation_space: gym.Space,
                    action_space: gym.Space,
                    actor_lr: float = 3e-4,
                    critic_lr: float = 3e-4,
                    temp_lr: float = 3e-4,
                    hidden_dims: Sequence[int] = (256, 256),
                    discount: float = 0.99,
                    tau: float = 0.005,
                    num_qs: int = 2,
                    num_min_qs: Optional[int] = None,
                    critic_dropout_rate: Optional[float] = None,
                    critic_weight_decay: Optional[float] = None,
                    actor_weight_decay: Optional[float] = None,
                    critic_layer_norm: bool = False,
                    target_entropy: Optional[float] = None,
                    init_temperature: float = 1.0,
                    backup_entropy: bool = True,
                    use_pnorm: bool = False,
                    use_critic_resnet: bool = False,
                    gradient_clipping_norm: Optional[float] = None,
                    use_tanh_normal: bool = True,
                    state_dependent_std: bool = False,
                    exterior_linear_c: float = 0.0,):
        action_dim = action_space.shape[-1]
        key, rng = jax.random.split(self.rng)

        # Get a fresh set of random actor parameters
        actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm
        )
        ## Replace
        # actor_def = TanhNormal(actor_base_cls, action_dim)

        if use_tanh_normal:
            actor_def = TanhNormal(actor_base_cls, action_dim, state_dependent_std=state_dependent_std)
        else:
            actor_def = Normal(
                actor_base_cls,
                action_dim,
                state_dependent_std=state_dependent_std,
                squash_tanh=use_tanh_normal  ## Should be false 
            )
        ## Done replacing

        ## Replace
        # actor_params = actor_def.init(key, self.batch["observations"])["params"]

        observations = observation_space.sample()
        actions = action_space.sample()
        actor_params = actor_def.init(key, observations)["params"]
        ## Done replacing

        ## Replace
        # actor = TrainState.create(
        #     apply_fn=actor_def.apply,
        #     params=actor_params,
        #     tx=optax.adam(learning_rate=actor_lr),
        # )

        if actor_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=actor_lr,
                weight_decay=actor_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=actor_lr)
        if gradient_clipping_norm:
            actor_optim = optax.chain(
                optax.clip_by_global_norm(gradient_clipping_norm),
                tx
            )
        else:
            actor_optim = tx

        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=actor_optim,
        )
        ## Done replacing

        # Replace the actor with the new random parameters
        new_agent = self.replace(actor=actor, rng=rng)
        return new_agent
    
    
    def reset_critic(self,
                    seed: int,
                    observation_space: gym.Space,
                    action_space: gym.Space,
                    actor_lr: float = 3e-4,
                    critic_lr: float = 3e-4,
                    temp_lr: float = 3e-4,
                    hidden_dims: Sequence[int] = (256, 256),
                    discount: float = 0.99,
                    tau: float = 0.005,
                    num_qs: int = 2,
                    num_min_qs: Optional[int] = None,
                    critic_dropout_rate: Optional[float] = None,
                    critic_weight_decay: Optional[float] = None,
                    actor_weight_decay: Optional[float] = None,
                    critic_layer_norm: bool = False,
                    target_entropy: Optional[float] = None,
                    init_temperature: float = 1.0,
                    backup_entropy: bool = True,
                    use_pnorm: bool = False,
                    use_critic_resnet: bool = False,
                    gradient_clipping_norm: Optional[float] = None,
                    use_tanh_normal: bool = True,
                    state_dependent_std: bool = False,
                    exterior_linear_c: float = 0.0,):
        key, rng = jax.random.split(self.rng)

        if use_critic_resnet:
            critic_base_cls = partial(
                MLPResNetV2,
                num_blocks=1,
            )
        else:
            critic_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )

        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=self.num_qs)

        ## Replace
        # critic_params = critic_def.init(key, self.batch["observations"], self.batch["actions"])["params"]
        
        observations = observation_space.sample()
        actions = action_space.sample()
        critic_params = critic_def.init(key, observations, actions)["params"]
        ## Done replacing

        ## Replace
        # if critic_weight_decay is not None:
        #     if max_gradient_norm is not None:
        #         tx = optax.chain(
        #             optax.clip_by_global_norm(max_gradient_norm),
        #             optax.adamw(
        #                 learning_rate=critic_lr,
        #                 weight_decay=critic_weight_decay,
        #                 mask=decay_mask_fn,
        #             )
        #         )
        #     else:
        #         tx = optax.adamw(
        #             learning_rate=critic_lr,
        #             weight_decay=critic_weight_decay,
        #             mask=decay_mask_fn,
        #         )
        # else:
        #     if max_gradient_norm is not None:
        #         tx = optax.chain(
        #                 optax.clip_by_global_norm(max_gradient_norm),
        #                 optax.adam(learning_rate=critic_lr)
        #         )
        #     else:
        #         tx = optax.adam(learning_rate=critic_lr)


        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
        if gradient_clipping_norm:
            critic_optim = optax.chain(
                optax.clip_by_global_norm(gradient_clipping_norm),
                tx
            )
        else:
            critic_optim = tx
        ## Done replacing

        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=critic_optim,
        )

        target_critic_def = Ensemble(critic_cls, num=self.num_min_qs or self.num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # Replace the critic and target critic with the new random parameters
        new_agent = self.replace(critic=critic, target_critic=target_critic, rng=rng)
        return new_agent

    def initialize_pretrained_params(self, actor_params, critic_params):
        new_agent = self
        new_actor = self.actor.replace(params=actor_params)
        new_critic = self.critic.replace(params=critic_params)
        new_target_critic = self.target_critic.replace(params=critic_params)
        new_agent = self.replace(actor=new_actor, critic=new_critic, target_critic=new_target_critic)
        return new_agent

    def update_actor(self, batch: DatasetDict, output_range: Optional[Tuple[jnp.ndarray, jnp.ndarray]]) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)

            ## Added APRL-like loss

            # create a mask if the action is out of range (0 if out of range, 1 if in range)
            if output_range is not None:
                mask = 1 - jnp.logical_or(jnp.any(actions < output_range[0], axis=-1), jnp.any(actions > output_range[1], axis=-1))
                oor_actions = jnp.mean(1-mask)

                exterior_actions = jnp.where(mask[:, None], jnp.zeros_like(actions), actions)
                interior_actions = jnp.where(mask[:, None], actions, jnp.zeros_like(actions))
                
                exterior_l1_penalty = jnp.sum(jnp.abs(exterior_actions), axis=-1)
                exterior_l2_penalty = jnp.sum(exterior_actions ** 2, axis=-1)
                exterior_l2_penalty = jnp.sqrt(exterior_l2_penalty)
                
                interior_linear_penalty = jnp.sum(jnp.abs(interior_actions), axis=-1)
                interior_quadratic_penalty = jnp.sum(interior_actions ** 4, axis=-1)
                
                # penalty_function = self.interior_quadratic_c * (self.exterior_linear_c / (3 * output_range[1] ** 3)) * interior_quadratic_penalty + self.exterior_linear_c * exterior_l1_penalty + self.exterior_quadratic_c * exterior_l2_penalty
                # penalty_function = self.ctrl_weight * oob_penalty + (self.ctrl_weight / (2 * output_range[1])) * ib_penalty
                penalty_function = self.exterior_linear_c * exterior_l1_penalty
                penalty_function_mean = penalty_function.mean()
            else:
                penalty_function = 0.0
                penalty_function_mean = 0.0
                oor_actions = 0.0
            ## End of added APRL-like loss


            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q + penalty_function
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean(), "oor_actions": oor_actions, "penalty_function": penalty_function_mean}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:

        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int, output_range: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None):
        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)
        
        new_agent, actor_info = new_agent.update_actor(mini_batch, output_range=output_range)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **temp_info}
