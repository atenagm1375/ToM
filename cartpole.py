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
from bindsnet.network.monitors import Monitor

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
    return amp * torch.exp(-(1/2) * ((value - means.to(device)) / sigma) ** 2)


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
    spikes = (np.array(spike_times[:, None].to('cpu')).astype(int) ==
              range(time)).astype(int)
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
    if kwargs.get("n_neurons", -1) == -1:
        kwargs["n_neurons"] = 10
    device = "cpu" if datum.get_device() < 0 else 'cuda'
    datum = datum[0]
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

    return encoded_datum.unsqueeze(1).to(device)


environment = GymEnvironment('CartPole-v0')
environment.reset()

observer = ObserverAgent(environment, dt=1.0, reward_fn=MovingAvgRPE)
expert = ExpertAgent(environment, method='from_weight')

observer.network.add_monitor(
    Monitor(observer.network.layers["S2"], ["s"]), "S2"
    )

pipeline = AgentPipeline(
    observer_agent=observer,
    expert_agent=expert,
    encoding=cartpole_observation_encoder,
    time=15,
    num_episodes=100,
    )

pipeline.train_by_observation(weight='hill_climbing.pt')
pipeline.test()
