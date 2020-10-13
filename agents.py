"""
==============================================================================.

agents.py

@author: atenagm

==============================================================================.
"""
import torch
import gym

from abc import ABC, abstractmethod

from bindsnet.environment import GymEnvironment
from bindsnet.network import Network
from bindsnet.learning.reward import AbstractReward
from bindsnet.network.nodes import Input, DiehlAndCookNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import WeightDependentPostPre, MSTDPET


class Agent(ABC):
    """
    Abstract base class for agents.

    Parameters
    ----------
    environment : GymEnvironment
        The environment of the agent.
    allow_gpu : bool, optional
        Allows automatic transfer to the GPU. The default is True.

    """

    @abstractmethod
    def __init__(
            self,
            environment: GymEnvironment,
            allow_gpu: bool = True,
            ) -> None:

        super().__init__()

        self.environment = environment

        if allow_gpu and torch.cuda.is_available():
            self.allow_gpu = True
            self.device = torch.device("cuda")
        else:
            self.allow_gpu = False
            self.device = torch.device("cpu")

    @abstractmethod
    def select_action(self,
                      action_space: gym.spaces.Space,
                      **kwargs) -> int:
        """
        Abstract method to select an action.

        Parameters
        ----------
        action_space : gym.spaces.Space
            The action space to choose from.

        Keyword Arguments
        -----------------

        Returns
        -------
        action : int
            The action to be taken.

        """
        action = -1
        return action


class ObserverAgent(Agent):
    """
    Observer agent in Gym environment.

    Parameters
    ----------
    environment : GymEnvironment
        The environment of the observer agent.
    dt : float, optional
        Network simulation timestep. The default is 1.0.
    learning : bool, optional
        Whether to allow network connection updates. The default is True.
    reward_fn : AbstractReward, optional
        Optional class allowing for modification of reward in case of
        reward-modulated learning. The default is None.
    allow_gpu : bool, optional
        Allows automatic transfer to the GPU. The default is True.

    """

    def __init__(
            self,
            environment: GymEnvironment,
            dt: float = 1.0,
            learning: bool = True,
            reward_fn: AbstractReward = None,
            allow_gpu: bool = True,
            ) -> None:

        super().__init__(environment, allow_gpu)

        input_shape = self.environment.env.observation_space.shape
        output_shape = self.environment.env.action_space.shape

        self.network = Network(dt=dt, learning=learning, reward_fn=reward_fn)

        # TODO Consider network structure
        s2 = Input(shape=[1, *input_shape, 10], traces=True)
        pfc = Input(n=1000, traces=True)
        sts = DiehlAndCookNodes(n=500, traces=True,
                                thresh=-52.0,
                                rest=-65.0,
                                reset=-65.0,
                                refrac=5,
                                tc_decay=100.0,
                                theta_plus=0.05,
                                tc_theta_decay=1e7)
        pm = DiehlAndCookNodes(shape=[*output_shape, 20], traces=True,
                               thresh=-52.0,
                               rest=-65.0,
                               reset=-65.0,
                               refrac=5,
                               tc_decay=100.0,
                               theta_plus=0.05,
                               tc_theta_decay=1e7)

        s2_sts = Connection(s2, sts,
                            nu=[0.05, 0.04],
                            update_rule=WeightDependentPostPre,
                            wmin=0.0,
                            wmax=0.2)
        sts_pm = Connection(sts, pm,
                            nu=[0.05, 0.04],
                            update_rule=MSTDPET,
                            wmin=0.0,
                            wmax=1.0,
                            norm=0.5 * sts.n)
        pfc_pm = Connection(pfc, pm,
                            nu=[0.05, 0.04],
                            update_rule=MSTDPET,
                            wmin=0.0,
                            wmax=1.0,
                            norm=0.25 * pfc.n)
        pm_pm = Connection(pm, pm,
                           nu=[0.05, 0.04],
                           wmin=-0.1,
                           wmax=0.)

        self.network.add_layer(s2, "S2")
        self.network.add_layer(sts, "STS")
        self.network.add_layer(pfc, "PFC")
        self.network.add_layer(pm, "PM")

        self.network.add_connection(s2_sts, "S2", "STS")
        self.network.add_connection(sts_pm, "STS", "PM")
        self.network.add_connection(pfc_pm, "PFC", "PM")
        self.network.add_connection(pm_pm, "PM", "PM")

        self.network.to(self.device)

    def select_action(self,
                      action_space: gym.spaces.Space,
                      **kwargs) -> int:
        """
        Choose the proper action based on observation.

        Parameters
        ----------
        action_space : gym.spaces.Space
            The action space to choose from.

        Keyword Arguments
        -----------------

        Returns
        -------
        action : int
            The action to be taken.

        """
        # TODO fill the method body (return winner of output population)
        pass


class ExpertAgent(Agent):
    """
    Expert agent in Gym environment.

    Parameters
    ----------
    environment : GymEnvironment
        Environment of the expert agent.
    allow_gpu : bool, optional
        Allows automatic transfer to the GPU. The default is True.

    """

    def __init__(self,
                 environment: GymEnvironment,
                 allow_gpu: bool = True,
                 ) -> None:

        super().__init__(environment, allow_gpu)


class CartPoleExpertAgent(ExpertAgent):
    """
    Expert agent in CartPole Gym environment.

    Parameters
    ----------
    environment : GymEnvironment
        Environment of the expert agent.
    allow_gpu : bool, optional
        Allows automatic transfer to the GPU. The default is True.

    """

    def __init__(self,
                 environment: GymEnvironment,
                 allow_gpu: bool = True,
                 ) -> None:

        super().__init__(environment, allow_gpu)

    def select_action(self,
                      action_space: gym.spaces.Space,
                      **kwargs) -> int:
        """
        Choose the proper action based on observation.

        Parameters
        ----------
        action_space : gym.spaces.Space
            The action space to choose from.

        Keyword Arguments
        -----------------

        Returns
        -------
        action : int
            The action to be taken.

        """
        # TODO ENHANCEMENT
        cart_pos, cart_v, pole_angle, pole_angular_v = \
            self.environment.env.state
        if cart_pos > 1:
            return 0
        if cart_pos < -1:
            return 1
        return -0.209 < pole_angle < 0.209
