"""
==============================================================================.

pipelines.py

@author: atenagm

==============================================================================.
"""

import torch

from bindsnet.pipeline.environment_pipeline import EnvironmentPipeline

from agents import ObserverAgent, ExpertAgent


class ObserverExpertPipeline(EnvironmentPipeline):
    def __init__(
            self,
            observer_agent: ObserverAgent,
            expert_agent: ExpertAgent,
            **kwargs,
            ) -> None:
        assert (observer_agent.environment == expert_agent.environment), \
            "Observer and Expert must be located in the same environment."
        assert (observer_agent.device == expert_agent.device), \
            "Observer and Expert objects must be on same device."

        super().__init__(
            observer_agent.network,
            observer_agent.environment,
            allow_gpu=observer_agent.allow_gpu,
            **kwargs,
            )
        self.observer_agent = observer_agent
        self.expert_agent = expert_agent

    def encoding(
            self,
            datum: torch.Tensor,
            time: int
            ) -> torch.Tensor:
        # TODO fill the body
        pass

    def train_by_observation(self, **kwargs) -> None:
        # TODO fill the body
        pass

    def self_train(self, **kwargs) -> None:
        # TODO fill the body
        pass

    def test(self, **kwargs) -> None:
        # TODO fill the body
        pass
