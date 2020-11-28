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
import matplotlib.pyplot as plt

from bindsnet.environment import GymEnvironment
from bindsnet.learning.reward import AbstractReward
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_weights

from agents import CartPoleObserverAgent, ExpertAgent
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
    return amp * torch.exp(-(1 / 2) * ((value - means.to(device)) / sigma.to(device)) ** 2)


def population_coding(
        value: float,
        time: int,
        means: torch.Tensor,
        sigma: torch.Tensor) -> torch.Tensor:
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
    n_neurons = 21
    device = "cpu" if datum.get_device() < 0 else 'cuda'
    datum = datum.squeeze()
    # cart_position = population_coding(datum[0], time,
    #                                   n_neurons,
    #                                   low=-4.8, high=4.8)
    # cart_velocity = population_coding(datum[1], time,
    #                                   n_neurons,
    #                                   low=-10, high=10)
    low, high = -0.418, 0.418
    sigma = ((high - low) / n_neurons) * torch.ones(n_neurons)
    means = torch.linspace(low, high, n_neurons)
    pole_angle = population_coding(datum[2], time,
                                   means, sigma).unsqueeze(1)

    low, high = -2, 2
    sigma = ((high - low) / (2 * n_neurons)) * torch.ones(n_neurons)
    sigma[:n_neurons * 3 // 8] *= 2
    sigma[n_neurons * 5 // 8:] *= 2
    means = torch.linspace(low, high, n_neurons)
    pole_agular_velocity = population_coding(datum[3], time,
                                             means, sigma).unsqueeze(1)

    # encoded_datum = torch.stack([
    #                              # cart_position,
    #                              # cart_velocity,
    #                              pole_angle,
    #                              pole_agular_velocity
    #                              ], dim=1)
    #
    # return encoded_datum.unsqueeze(1).to(device)
    return {"S2": pole_angle.unsqueeze(1).to(device),
            "MT": pole_agular_velocity.unsqueeze(1).to(device)}


def noise_policy(episode, num_episodes, **kwargs):
    return (1 - episode / num_episodes) ** 2
    # return np.exp(-4 * episode/num_episodes)


class CartPoleReward(AbstractReward):
    def __init__(self, **kwargs):
        pass

    def compute(self, **kwargs):
        reward = kwargs["reward"]
        # last_state = kwargs["last_state"]
        # curr_state = kwargs["curr_state"]
        #
        # if reward > 0:
        #     if torch.abs(curr_state[2]) <= torch.abs(last_state[2]):
        #         return 1
        #     elif torch.allclose(curr_state[2], last_state[2], 1e-4, 1e-5):
        #         return 0.5
        #     else:
        #         return -0.5
        # return -1
        return reward if reward > 0 else -1

    def update(self, **kwargs):
        pass


environment = GymEnvironment('CartPole-v0')
environment.reset()

observer = CartPoleObserverAgent(environment, dt=1.0, method='softmax',
                                 reward_fn=CartPoleReward)

expert = ExpertAgent(environment, method='from_weight',
                     noise_policy=noise_policy)

pipeline = AgentPipeline(
    observer_agent=observer,
    expert_agent=expert,
    encoding=cartpole_observation_encoder,
    time=15,
    num_episodes=100,
    # plot_interval=1,
    # render_interval=1
)

w1 = pipeline.network.connections[("S2", "PM")].w
# plot_weights(w1)
print(w1)

w2 = pipeline.network.connections[("MT", "PM")].w
# plot_weights(w2)
print(w2)

pipeline.train_by_observation(weight='/home/atenagm/hill_climbing.pt',
                              test_interval=10, num_tests=5)
print("Observation Finished")
#
w1 = pipeline.network.connections[("S2", "PM")].w
# plot_weights(w1)
print(w1)

w2 = pipeline.network.connections[("MT", "PM")].w
# plot_weights(w2)
print(w2)

# for i in range(100):
#     pipeline.test()

# test_rewards = pipeline.test_rewards[-100:]
# print("min:", np.min(test_rewards), "max:", np.max(test_rewards), "average:",
#        np.mean(test_rewards))
# plt.ioff()
# plt.plot(pipeline.test_rewards)
# plt.show()
