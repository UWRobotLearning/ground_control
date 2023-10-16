import torch
import torch.nn as nn

def orthogonal_init(module: nn.Linear, gain: float = 1.):
    """
    For the given linear module, performs an orthogonal initialization of the
    weight matrix and zeros out the bias. Based on code from the following:
    https://github.com/implementation-matters/code-for-paper/blob/master/src/policy_gradients/torch_utils.py#L494
    """
    weight = module.weight
    if weight.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = weight.size(0)
    cols = weight[0].numel()
    flattened = weight.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, _, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        weight.data.view_as(q).copy_(q)
        weight.data.mul_(gain)

    if module.bias is not None:
        module.bias.data.zero_()
