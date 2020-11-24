"""
==============================================================================.

cartpole.py

@author: atenagm

==============================================================================.
"""
import sys

sys.path.append('../bindsnet/')


import torch
import numpy as np

from bindsnet.environment import GymEnvironment
from bindsnet.learning.reward import MovingAvgRPE
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_weights

from agents import ObserverAgent, ExpertAgent
from pipelines import AgentPipeline


def tuning_curve(
        value: float,
        amp: int,
        means: torch.Tensor,
        sigma: float = 1.0
) -> torch.Tensor:
    """
    Compute the tuning curve vector for the value.

    Parameters
    ----------
    value : float
        The value to compute its vector code.
    amp : int
        The amplitude of tuning curves.
    means : torch.Tensor
        The mean values of the curves.
    sigma : float, optional
        The standard deviations. The default is 1.0.

    Returns
    -------
    torch.Tensor
        The vector code.

    """
    device = 'cpu' if value.get_device() < 0 else 'cuda'
    return amp * torch.exp(-(1 / 2) * ((value - means.to(device)) / sigma) ** 2)


def population_coding(
        value: float,
        time: int,
        n_neurons: int,
        low: float,
        high: float) -> torch.Tensor:
    """
    Apply population coding to the value.

    Parameters
    ----------
    value : float
        The value.
    time : int
        Length of spike train.
    n_neurons : int
        Number of neurons in the coding population.
    low : float
        Lowest possible value.
    high : float
        Highest possible value.

    Returns
    -------
    torch.Tensor
        The population coded tensor.

    """
    sigma = (high - low) / (2 * n_neurons)
    means = torch.linspace(low, high, n_neurons)
    spike_times = tuning_curve(value, time - 1, means, sigma)
    spikes = (np.array(spike_times[:, None].to('cpu')).astype(int) ==
              range(1, time+1)).astype(int)
    return torch.from_numpy(spikes.T)


def cartpole_observation_encoder(
        datum: torch.Tensor,
        time: int,
        **kwargs,
) -> torch.Tensor:
    """
    Encode observation vector.

    Parameters
    ----------
    datum : torch.Tensor
        Observation tensor.
    time : int
        Length of spike train per observation.

    Keyword Arguments
    -----------------
    n_neurons : int
        Specifies number of neurons which encode each value.

    Returns
    -------
    torch.Tensor
        The encoded data.

    """
    n_neurons = 30
    device = "cpu" if datum.get_device() < 0 else 'cuda'
    datum = datum[0]
    # cart_position = population_coding(datum[0], time,
    #                                   n_neurons,
    #                                   low=-4.8, high=4.8)
    # cart_velocity = population_coding(datum[1], time,
    #                                   n_neurons,
    #                                   low=-10, high=10)
    pole_angle = population_coding(datum[2], time,
                                   n_neurons,
                                   low=-0.418, high=0.418).unsqueeze(1)
    # pole_agular_velocity = population_coding(datum[3], time,
    #                                          n_neurons,
    #                                          low=-10, high=10)
    #
    # encoded_datum = torch.stack([cart_position,
    #                              # cart_velocity,
    #                              pole_angle,
    #                              # pole_agular_velocity
    #                              ], dim=1)
    #
    # return encoded_datum.unsqueeze(1).to(device)
    return pole_angle.unsqueeze(1).to(device)


def noise_policy(episode, num_episodes, **kwargs):
    return np.exp(-4 * episode/num_episodes)


environment = GymEnvironment('CartPole-v0')
environment.reset()

observer = ObserverAgent(environment, dt=1.0, method='softmax')
expert = ExpertAgent(environment, method='from_weight',
                     noise_policy=noise_policy)

observer.network.add_monitor(
    Monitor(observer.network.layers["S2"], ["s"]), "S2"
)
observer.network.add_monitor(
    Monitor(observer.network.layers["PM"], ["s", "v"]), "PM"
)
observer.network.add_monitor(
    Monitor(observer.network.connections[("S2", "PM")], ["w"]), "S2-PM"
)

pipeline = AgentPipeline(
    observer_agent=observer,
    expert_agent=expert,
    encoding=cartpole_observation_encoder,
    time=15,
    num_episodes=100,
    # plot_interval=1,
    # render_interval=1
)

w = pipeline.network.connections[("S2", "PM")].w
# plot_weights(w)
print(w)

pipeline.train_by_observation(weight='/home/atenagm/hill_climbing.pt')
print("Observation Finished")
#
w = pipeline.network.connections[("S2", "PM")].w
# plot_weights(w)
print(w)

for i in range(100):
    pipeline.test()

test_rewards = pipeline.reward_list[-100:]
print("min:", np.min(test_rewards), "max:", np.max(test_rewards), "average:",
       np.mean(test_rewards))
