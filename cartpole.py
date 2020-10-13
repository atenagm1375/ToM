"""
==============================================================================.

cartpole.py

@author: atenagm

==============================================================================.
"""
import torch

from bindsnet.environment import GymEnvironment
from bindsnet.learning.reward import MovingAvgRPE

from agents import ObserverAgent


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

    Returns
    -------
    None.

    """
    # TODO fill the body
    pass


environment = GymEnvironment('CartPole-v1')
environment.reset()

observer = ObserverAgent(environment, dt=1.0, reward_fn=MovingAvgRPE)
