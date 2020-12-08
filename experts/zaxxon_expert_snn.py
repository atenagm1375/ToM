"""
==============================================================================.

zaxxon_expert_snn.py

@author: atenagm

==============================================================================.
"""
# -----------------------------------------------------------------------------
# The following 2 lines are only to use the modified version of BindsNet.
# You can find it on https://github.com/atenagm1375/bindsnet/tree/atena.
# Without it, you may face errors.
import sys

sys.path.append('../../bindsnet/')
sys.path.append('../')
# -----------------------------------------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
from bindsnet.environment import GymEnvironment
from bindsnet.learning.reward import AbstractReward, MovingAvgRPE
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_weights
from ToM.agents import ZaxxonAgent
from ToM.pipelines import AgentPipeline


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
) -> dict:
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
    dict
        The tensor of encoded data per input population.

    """
    device = "cpu" if datum.get_device() < 0 else 'cuda'
    datum = datum.squeeze()

    spikes = []
    for d in datum:
        val = (255 - d) / 255
        spikes.append(_compute_spikes(val, time, 0., 1., device))

    spikes = torch.stack(spikes).T

    return {"S2": spikes.unsqueeze(1).byte().to(device)}


class ZaxxonReward(AbstractReward):
    """
    Computes the reward for Zaxxon environment.

    Parameters
    ----------

    Keyword Arguments
    -----------------

    """

    def __init__(self, **kwargs):
        pass

    def compute(self, **kwargs):
        """
        Compute the reward.

        Keyword Arguments
        -----------------
        reward : float
            The reward value returned from the environment

        """
        reward = kwargs["reward"]
        return reward

    def update(self, **kwargs):
        """
        Update internal attributes.

        Keyword Arguments
        -----------------
        accumulated_reward : float
            The value of accumulated reward in the episode.

        """
        pass


# Define the environment
environment = GymEnvironment('Zaxxon-ram-v0')

# Define observer agent, acting on first spike
observer = ZaxxonAgent(environment, dt=1.0, method='first_spike',
                       reward_fn=ZaxxonReward)

# Define the pipeline by which the agents interact
pipeline = AgentPipeline(
    observer_agent=observer,
    expert_agent=None,
    encoding=ram_observation_encoder,
    time=15,
    num_episodes=10000
)

w1 = pipeline.network.connections[("S2", "PM")].w
# plot_weights(w1)
print(w1)

pipeline.train_by_babbling(test_interval=1000, num_tests=1)
print("Observation Finished")
#
w1 = pipeline.network.connections[("S2", "PM")].w
# plot_weights(w1)
print(w1)

print(pipeline.test_rewards)
