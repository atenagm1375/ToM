"""
==============================================================================.

cartpole.py

@author: atenagm

==============================================================================.
"""
import torch
import numpy as np

from bindsnet.environment import GymEnvironment
from bindsnet.learning.reward import MovingAvgRPE

from agents import ObserverAgent, CartPoleExpertAgent


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
    return amp * torch.exp(-(1/2) * ((value - means) / sigma) ** 2)


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
    means = torch.linspace(low, high, n_neurons)
    sigma = (high - low) / n_neurons
    spike_times = tuning_curve(value, time - 1, means, sigma)
    spikes = (np.array(spike_times[:, None]) == range(time)).astype(int)
    return torch.from_numpy(spikes)


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
    device = datum.get_device()
    cart_position = population_coding(datum[0], time,
                                      kwargs["n_neurons"],
                                      low=-4.8, high=4.8)
    cart_velocity = population_coding(datum[1], time,
                                      kwargs["n_neurons"],
                                      low=-10, high=10)
    pole_angle = population_coding(datum[2], time,
                                   kwargs["n_neurons"],
                                   low=-0.418, high=0.418)
    pole_agular_velocity = population_coding(datum[3], time,
                                             kwargs["n_neurons"],
                                             low=-10, high=10)
    encoded_datum = torch.stack([cart_position,
                                 cart_velocity,
                                 pole_angle,
                                 pole_agular_velocity
                                 ], dim=1)
    return encoded_datum.to(device)


environment = GymEnvironment('CartPole-v1')
environment.reset()

observer = ObserverAgent(environment, dt=1.0, reward_fn=MovingAvgRPE)
expert = CartPoleExpertAgent(environment)
