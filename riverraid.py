"""
==============================================================================.
riverraid.py
@author: atenagm
==============================================================================.
"""
# -----------------------------------------------------------------------------
# The following 2 lines are only to use the modified version of BindsNet.
# You can find it on https://github.com/atenagm1375/bindsnet/tree/atena.
# Without it, you may face errors.
import sys

sys.path.append('../bindsnet/')
# -----------------------------------------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
from bindsnet.environment import GymEnvironment
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.learning.reward import AbstractReward, MovingAvgRPE
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_weights
from interactive_env.agents import RiverraidAgent


def _compute_spikes(
    datum: torch.Tensor,
    time: int,
    low: float,
    high: float,
    device: str
) -> torch.Tensor:
    times = torch.linspace(low, high, time, device=device)
    spike_times = torch.argmin(torch.abs(datum - times))
    spikes = (np.array(spike_times.to('cpu')).astype(int) ==
              range(0, time)).astype(int)
    reverse_spikes = np.flip(spikes).copy()
    return torch.stack([
        torch.from_numpy(spikes).to(device),
        torch.from_numpy(reverse_spikes).to(device)
    ]).byte()


def ram_observation_encoder(
        datum: torch.Tensor,
        time: int,
        **kwargs,
) -> torch.Tensor:
    """
    Encode observation vector. It encodes a value and its complement in time.
    So there are two neurons per value.
    Parameters
    ----------
    datum : torch.Tensor
        Observation tensor.
    time : int
        Length of spike train per observation.
    Keyword Arguments
    -----------------
    Returns
    -------
    torch.Tensor
        The tensor of encoded data per input population.
    """
    device = "cpu" if datum.get_device() < 0 else 'cuda'
    datum = datum.squeeze()

    spikes = []
    for d in datum:
        val = (255 - d) / 255
        spikes.append(_compute_spikes(val, time, 0., 1., device))

    spikes = torch.stack(spikes).T

    return spikes.unsqueeze(1).byte().to(device)


class RiverraidReward(AbstractReward):
    """
    Computes the reward for Zaxxon environment.
    Parameters
    ----------
    Keyword Arguments
    -----------------
    """

    def __init__(self, **kwargs):
        self.alpha = 0.01
        self.penalty = 0

    def compute(self, **kwargs):
        """
        Compute the reward.
        Keyword Arguments
        -----------------
        reward : float
            The reward value returned from the environment
        """
        reward = kwargs["reward"] + self.penalty
        self.penalty *= self.alpha
        # return is_alive * (reward * self.alpha)
        return reward

    def update(self, **kwargs):
        """
        Update internal attributes.
        Keyword Arguments
        -----------------
        accumulated_reward : float
            The value of accumulated reward in the episode.
        """
        accumulated_reward = kwargs['accumulated_reward']
        steps = kwargs['steps']
        self.penalty = -steps / accumulated_reward


def select_action(pipeline, output, **kwargs) -> int:
    """
    Choose the proper action based on observation.

    Keyword Arguments
    -----------------

    Returns
    -------
    action : int
        The action to be taken.

    """
    if pipeline.network.monitors[output].recording['s'] != []:
        spikes = (pipeline.network.monitors[output].get("s").float())
        spikes = spikes.squeeze().squeeze().nonzero()

        if spikes.shape[0] == 0:
            return pipeline.env.action_space.sample()
        else:
            return spikes[0, 1]
    return pipeline.env.action_space.sample()


# Define the environment
environment = GymEnvironment('Riverraid-ram-v0')

# Define observer agent, acting on first spike
observer = RiverraidAgent(environment, dt=1.0, method='first_spike',
                          reward_fn=RiverraidReward)
observer.build_network()

pipeline = EnvironmentPipeline(
    network=observer.network,
    environment=environment,
    action_function=select_action,
    encoding=ram_observation_encoder,
    device=observer.device,
    output="PM",
    time=64,
    num_episodes=5000,
)

w1 = pipeline.network.connections[("S2", "PM")].w
# plot_weights(w1)
print(w1)

pipeline.train()
print("Training Finished")
#
w1 = pipeline.network.connections[("S2", "PM")].w
# plot_weights(w1)
print(w1)
