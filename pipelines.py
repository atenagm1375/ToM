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
    """
    Abstracts the interaction between agents in the environment.
    """

    def __init__(
            self,
            observer_agent: ObserverAgent,
            expert_agent: ExpertAgent,
            **kwargs,
            ) -> None:
        """
        Pipeline class costructor.

        Parameters
        ----------
        observer_agent : ObserverAgent
            The oberserver agent in the environment.
        expert_agent : ExpertAgent
            The expert agent in the environment.

        Keyword Arguments
        -----------------
        num_episodes : int
            Number of episodes to train for. The default is 100.
        output : str
            String name of the layer from which to take output.
        render_interval : int
            Interval to render the environment.
        reward_delay : int
            How many iterations to delay delivery of reward.
        time : int
            Time for which to run the network.
        overlay_input : int
            Overlay the last X previous input.

        Returns
        -------
        None

        """
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

    def observation_encoder(
            self,
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

    def train_by_observation(self, **kwargs) -> None:
        """
        Train observer agent's network by observing the expert.

        Returns
        -------
        None

        """
        while self.episode < self.num_episodes:
            self.reset_state_variables()

            done = False
            while not done:
                obs, reward, done, info = self.env_step()

                self.step((obs, reward, done, info), **kwargs)

            print(
                f"Episode: {self.episode} - "
                f"accumulated reward: {self.accumulated_reward:.2f}"
            )
            self.episode += 1

    def step_(
            self,
            gym_batch: tuple[torch.Tensor, float, bool, dict],
            **kwargs
            ) -> None:
        """
        Run one step of the oberserver's network.

        Parameters
        ----------
        gym_batch : tuple[torch.Tensor, float, bool, dict]
            An OpenAI gym compatible tuple.

        Returns
        -------
        None

        """
        self.encoding = self.observation_encoder

        obs, reward, done, info = gym_batch

        obs.unsqueeze(0)
        inputs = {
            k: self.encoding(obs, self.time)
            for k in self.inputs
            }

        # TODO edit reward

        self.network.run(inputs=inputs, time=self.time, reward=reward,
                         **kwargs)

        if done:
            if self.network.reward_fn is not None:
                self.network.reward_fn.update(
                    accumulated_reward=self.accumulated_reward,
                    steps=self.step_count,
                    **kwargs,
                )
            self.reward_list.append(self.accumulated_reward)

    def self_train(self, **kwargs) -> None:
        """
        Train observer agent's network by acting in the environment.

        Returns
        -------
        None

        """
        # TODO fill the body
        pass

    def test(self, **kwargs) -> None:
        """
        Test the observer agent in the environment.

        Returns
        -------
        None

        """
        # TODO fill the body
        pass
