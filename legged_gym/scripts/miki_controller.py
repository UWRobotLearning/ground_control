import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PMTGEnv:
    def __init__(self, control_dt: float=1./50., max_height: float = 1.0):
        '''
        Args:
        - control_dt: float, the dt at which the system is controlled. If there is a simulation frequency and an action repeat, then control frequency = sim_freq/action_repeat. Additionally, control_dt = 1/control_frequency
        - max_height: float, the maximum height for the foot position.

        Defines the following quantities:
        - phi: np.ndarray of size (4,) ~ U[0, 2*pi), the initial phase per leg. This will be sampled from the uniform distribution between 0 and 2 pi, so as to make the policy robust to any initial phase.
        - phi_step: float, the nominal value by which the phase gets increased at every timestep, i.e., phi_{t+1} = (phi_{t} + delta_phi_0) (mod 2*pi). Note that the agent can also modify how much the phase changes at each timestep, giving the more general internal state update phi_{t+1} = (phi_{t} + delta_phi_0 + delta_phi_l_action) (mod 2*pi)
        - initial_foot_positions: Union[np.ndarray, torch.Tensor] of shape (4, 3), the initial positions in 3D in the world frame of the 4 feet. The order is (front_right, front_left, rear_right, rear_left).
        '''
        self.phi = np.array((0., np.pi, np.pi, 0.)) #np.random.uniform(low=0.0, high=2*np.pi, size=(4,))  # np.zeros((4,))
        self.phi_step = 1.25*control_dt  ## 1.25 comes from f_0 in https://arxiv.org/pdf/2010.11251, control_dt is assumed to be 50 Hz, giving us phi_step = 0.025
        self.initial_foot_positions = np.zeros((4, 3))  ## For now these are all collapsed into the same position (0, 0, 0), but eventually these should come from the initial position of the robot as measured by the simulator
        self.max_height= max_height

    def step(self, action: np.ndarray = np.zeros((16,))):
        '''
        Args:
        - action: (16,) numpy array with the following action space
            Action space
            | Num | Action                             | Min | Max |
            | --- | ---------------------------------- | --- | --- |
            |  0  | frequency offset for FR leg        |     |     |
            |  1  | frequency offset for FL leg        |     |     |
            |  2  | frequency offset for RR leg        |     |     |
            |  3  | frequency offset for RL leg        |     |     |
            |  4  | FR hip position target residual    |     |     |
            |  5  | FR thigh position target residual  |     |     |
            |  6  | FR calf position target residual   |     |     |
            |  7  | FL hip position target residual    |     |     |
            |  8  | FL thigh position target residual  |     |     |
            |  9  | FL calf position target residual   |     |     |
            | 10  | RR hip position target residual    |     |     |
            | 11  | RR thigh position target residual  |     |     |
            | 12  | RR calf position target residual   |     |     |
            | 13  | RL hip position target residual    |     |     |
            | 14  | RL thigh position target residual  |     |     |
            | 15  | RL calf position target residual   |     |     |
        '''

        phi_offsets = action[0:4]
        self.phi = np.mod(self.phi + phi_offsets + self.phi_step, 2*np.pi)

        t_l_up = (2./np.pi) * self.phi
        t_l_down = (2./np.pi) * self.phi - 1

        z_offset = np.zeros_like(self.phi)
        z_offset = np.where(np.logical_and(self.phi >= 0.0, self.phi <= np.pi/2),
                     self.max_height * (-2*t_l_up**3 + 3*t_l_up**2),
                     z_offset)
        z_offset = np.where(np.logical_and(self.phi > np.pi/2, self.phi <= np.pi),
                     self.max_height * (2*t_l_down**3 - 3*t_l_down**2 + 1),
                     z_offset)
        
        foot_offsets = np.zeros_like(self.initial_foot_positions)
        foot_offsets[:, -1] = z_offset

        foot_positions = self.initial_foot_positions + foot_offsets

        return z_offset



class MikiController:
    def __init__(self, base_freq: float = 2., delta_phi_0: np.ndarray = np.array([0.0, 0.5, 0.5, 0.0]), initial_foot_positions = np.zeros((4, 3))):
        self.base_freq = base_freq
        self.delta_phi_0 = delta_phi_0
        self.dt = 1./50.

    def control(self, cos_phi_l: np.ndarray, sin_phi_l: np.ndarray, delta_phi_l: np.ndarray):
        pass

    def single_leg_controller(self, cos_phi_l, sin_phi_l, delta_phi_l, initial_foot_pos):
        phi_l = np.arctan2(sin_phi_l, cos_phi_l)

        phase = np.mod(phi_l+delta_phi_l+self.base_freq, 2*np.pi)

        if np.logical_and(phase >= 0.0, phase <= np.pi/2):
            t_l = (2./np.pi) * phase
            z_offset = 0.2 * (-2*t_l**3 + 3*t_l**2)
        elif np.logical_and(phase > np.pi/2, phase <= np.pi):
            t_l = (2./np.pi) * phase - 1
            z_offset = 0.2 * (2*t_l**3 - 3*t_l**2 + 1)
        elif np.logical_and(phase > np.pi, phase <= 2*np.pi):
            z_offset = 0
        
        foot_pos = initial_foot_pos + np.array([0., 0., z_offset])

        # return foot_pos
        return foot_pos[-1]

    def loop(self):
        heights = []
        phases = np.arange(0., 5*2*np.pi, self.dt)
        for phase in phases:
            z_height = self.single_leg_controller(np.cos(phase), np.sin(phase), 0.0, np.array([0.0, 0.0, 0.0]))
            heights.append(z_height)
        plt.plot(heights)
        plt.xlabel("Phase mod 2*pi")
        plt.ylabel("Foot height")
        plt.title("Foot height vs phase for Miki policy")
        plt.show()

if __name__ == "__main__":

    # Create the PMTGEnv instance
    pmtg = PMTGEnv()

    # Set up the figure and axes
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("PMTGEnv φ and Z Position Visualization")

    titles = ["Front Right Leg", "Front Left Leg", "Rear Right Leg", "Rear Left Leg"]

    for i, title in enumerate(titles):
        # Phase plot
        ax_phase = axs[0, i]
        ax_phase.set_xlim(-1.5, 1.5)
        ax_phase.set_ylim(-1.5, 1.5)
        ax_phase.set_aspect('equal')
        ax_phase.set_xticks([])
        ax_phase.set_yticks([])
        ax_phase.set_title(title)
        circle = plt.Circle((0, 0), 1, fill=False)
        ax_phase.add_artist(circle)

        # Z position plot
        ax_z = axs[1, i]
        ax_z.set_ylim(0, pmtg.max_height)
        ax_z.set_xlim(0, 1)
        ax_z.set_xticks([])
        ax_z.set_ylabel('Z Position')
        ax_z.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_z.axhline(y=pmtg.max_height, color='gray', linestyle='--', alpha=0.5)

    # Initialize the lines and points
    phi = pmtg.phi
    x = np.sin(phi)
    y = np.cos(phi)
    lines = [axs[0, i].plot([0, x[i]], [0, y[i]], 'r-')[0] for i in range(4)]
    phi_texts = [axs[0, i].text(0.05, 0.95, '', transform=axs[0, i].transAxes) for i in range(4)]

    z_positions = np.zeros(4)
    z_points = [axs[1, i].plot(0.5, z_positions[i], 'ro')[0] for i in range(4)]
    z_lines = [axs[1, i].plot([0.5, 0.5], [0, z_positions[i]], 'r-', alpha=0.3)[0] for i in range(4)]

    # Update function for the animation
    def update(frame):
        action = np.zeros((16,))
        # action[1] = -1.25*1./50
        z_positions = pmtg.step(action)
        phi = pmtg.phi
        x = np.sin(phi)
        y = np.cos(phi)
        
        for i in range(4):
            # Update phase plot
            lines[i].set_xdata([0, x[i]])
            lines[i].set_ydata([0, y[i]])
            phi_texts[i].set_text(f'φ{i} = {phi[i]:.2f}')
            
            # Update z position plot
            z_points[i].set_ydata(z_positions[i])
            z_lines[i].set_ydata([0, z_positions[i]])
        
        return lines + phi_texts + z_points + z_lines

    # Create the animation
    anim = FuncAnimation(fig, update, frames=200, interval=20, blit=True)

    plt.tight_layout()
    plt.show()
