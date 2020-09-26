"""
==============================================================================.

agents.py

@author: atenagm

==============================================================================.
"""
import torch

from abc import ABC, abstractmethod

from bindsnet.environment import GymEnvironment
from bindsnet.network import Network
from bindsnet.learning.reward import AbstractReward
from bindsnet.network.nodes import Input, DiehlAndCookNodes
from bindsnet.network.topology import Connection, SparseConnection
from bindsnet.learning import WeightDependentPostPre, MSTDPET


class Agent(ABC):
    """
    Abstract base class for agents.
    """

    @abstractmethod
    def __init__(
            self,
            environment: GymEnvironment,
            **kwargs,
            ) -> None:
        """
        Abstract base class constructor.

        Parameters
        ----------
        environment : GymEnvironment
            The environment of the agent.

        """
        super().__init__()

        self.environment = environment

    @abstractmethod
    def select_action(self, **kwargs):
        """
        Abstract method to select an action.

        Returns
        -------
        action : int
            The action to be taken.

        """
        action = -1
        return action


class ObserverAgent(Agent):
    """
    Observer agent in CartPole Gym environment.
    """

    def __init__(
            self,
            environment: GymEnvironment,
            dt: float = 1.0,
            learning: bool = True,
            reward_fn: AbstractReward = None,
            ) -> None:
        """
        Observer class constructor.

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

        """
        super().__init__(environment)

        self.network = Network(dt=dt, learning=learning, reward_fn=reward_fn)

        s2 = Input(n=4, shape=[1, 1, 1, 4], traces=True)
        sts = DiehlAndCookNodes(n=100, traces=True,
                                thresh=-52.0,
                                rest=-65.0,
                                reset=-65.0,
                                refrac=5,
                                tc_decay=100.0,
                                theta_plus=0.05,
                                tc_theta_decay=1e7)
        pm = DiehlAndCookNodes(n=2, traces=True,
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
        sts_sts = SparseConnection(sts, sts, sparsity=0.25)
        pm_sts = Connection(pm, sts,
                            nu=[0.05, 0.04],
                            update_rule=WeightDependentPostPre,
                            wmin=-1.0,
                            wmax=0.01,
                            norm=0.25 * pm.n)
        pm_pm = Connection(pm, pm,
                           nu=[0.05, 0.04],
                           wmin=-0.1,
                           wmax=0.1)

        self.network.add_layer(s2, "S2")
        self.network.add_layer(sts, "STS")
        self.network.add_layer(pm, "PM")

        self.network.add_connection(s2_sts, "S2", "STS")
        self.network.add_connection(sts_pm, "STS", "PM")
        self.network.add_connection(sts_sts, "STS", "STS")
        self.network.add_connection(pm_sts, "PM", "STS")
        self.network.add_connection(pm_pm, "PM", "PM")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(device)

    def select_action(self, **kwargs):
        # TODO fillt the method body
        pass


class ExpertAgent(Agent):
    """
    Expert agent in CartPole Gym environment.
    """

    def __init__(self,
                 environment: GymEnvironment,
                 ) -> None:
        """
        Expert class constructor.

        Parameters
        ----------
        environment : GymEnvironment
            Environment of the expert agent.

        """
        super().__init__(environment)

    def select_action(self, **kwargs):
        # TODO fill the method body
        pass
