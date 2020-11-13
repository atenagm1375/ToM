"""
==============================================================================.

pipelines.py

@author: atenagm

==============================================================================.
"""

import torch
from tqdm import tqdm

from bindsnet.pipeline.environment_pipeline import EnvironmentPipeline

from agents import ObserverAgent, ExpertAgent


class AgentPipeline(EnvironmentPipeline):
    """
    Abstracts the interaction between agents in the environment.

    Parameters
    ----------
    observer_agent : ObserverAgent
        The oberserver agent in the environment.
    expert_agent : ExpertAgent
        The expert agent in the environment.
    encoding : callable, optional
        The observation encoder function. The default is None.

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

    """

    def __init__(
            self,
            observer_agent: ObserverAgent,
            expert_agent: ExpertAgent,
            encoding: callable = None,
            **kwargs,
    ) -> None:

        assert (observer_agent.environment == expert_agent.environment), \
            "Observer and Expert must be located in the same environment."
        assert (observer_agent.device == expert_agent.device), \
            "Observer and Expert objects must be on same device."

        super().__init__(
            observer_agent.network,
            observer_agent.environment,
            encoding=encoding,
            allow_gpu=observer_agent.allow_gpu,
            **kwargs,
        )
        self.observer_agent = observer_agent
        self.expert_agent = expert_agent

        self.plot_config = {
            "data_step": True,
            # "obs_step": True,
            # "reward_eps": 1,
            "data_length": 200,
        }

    def env_step(self, **kwargs) -> tuple:
        """
        Perform single step of the environment.

        Includes rendering, getting and performing the action, and
        accumulating/delaying rewards.

        Returns
        -------
        obs : torch.Tensor
            The observation tensor.
        reward : float
            The reward value.
        done : bool
            Indicates if the the episode is terminated.
        info : dict
            The information dictionary for verbose.

        KeywordArguments
        ----------------

        """
        # Render game.
        if (
            self.render_interval is not None
            and self.step_count % self.render_interval == 0
        ):
            self.env.render()

        # Choose action based on output neuron spiking.
        if self.action_function is not None:
            # self.last_action = self.action
            # if torch.rand(1) < self.percent_of_random_action:
            #     self.action = torch.randint(
            #         low=0, high=self.env.action_space.n, size=(1,)
            #     )[0]
            # elif self.action_counter > self.random_action_after:
            #     if self.last_action == 0:  # last action was start b
            #         self.action = 1  # next action will be fire b
            #         tqdm.write(f"Fire -> too many times {self.last_action} ")
            #     else:
            #         self.action = torch.randint(
            #             low=0, high=self.env.action_space.n, size=(1,)
            #         )[0]
            #         tqdm.write(f"too many times {self.last_action} ")
            # else:
            self.action = self.action_function(episode=self.episode,
                                               num_episodes=self.num_episodes,
                                               **kwargs)

            # if self.last_action == self.action:
            #     self.action_counter += 1
            # else:
            #     self.action_counter = 0

        # Run a step of the environment.
        if not isinstance(self.action, int):
            self.action = self.action.to('cpu').numpy()
        obs, reward, done, info = self.env.step(self.action)

        # Set reward in case of delay.
        if self.reward_delay is not None:
            self.rewards = torch.tensor([reward, *self.rewards[1:]]).float()
            reward = self.rewards[-1]

        # Accumulate reward.
        self.accumulated_reward += reward

        info["accumulated_reward"] = self.accumulated_reward

        return obs, reward, done, info

    def step_(
            self,
            gym_batch: tuple,
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
        obs, reward, done, info = gym_batch

        inputs = {
            k: self.encoding(obs, self.time)
            for k in self.inputs if k != "PFC"
        }
        # inputs["PFC"] = torch.poisson(torch.rand(self.time,
        #                                          self.network.layers["PFC"].n)
        #                               ).byte().to(self.device)
        pm_n = self.network.layers["PM"].n
        n_action = self.env.action_space.n
        pm_v = torch.zeros(pm_n)
        pm_v[pm_n//n_action * self.action:pm_n//n_action * (self.action + 1)] = 1
        # injects_v = {
        #     "PM": pm_v.view(self.time, self.env.action_space.n, -1).to(self.device)
        # }
        pm_v = pm_v.byte()
        clamp = {
            "PM": pm_v.to(self.device)
        } if self.network.learning else {}

        # TODO define keyword arguments for reward function
        reward = reward if reward > 0 else -1

        self.network.run(inputs=inputs, clamp=clamp, time=self.time,
                         reward=reward, **kwargs)

        if done:
            if self.network.reward_fn is not None:
                self.network.reward_fn.update(
                    accumulated_reward=self.accumulated_reward,
                    steps=self.step_count,
                    **kwargs,
                )

            # Update reward list for plotting purposes.
            self.reward_list.append(self.accumulated_reward)

    def train_by_observation(self, **kwargs) -> None:
        """
        Train observer agent's network by observing the expert.

        Returns
        -------
        None

        """
        self.observer_agent.network.train(True)

        # Expert acts in the environment.
        self.action_function = self.expert_agent.select_action

        while self.episode < self.num_episodes:
            self.reset_state_variables()

            done = False
            while not done:
                # The result of expert's action.
                obs, reward, done, info = self.env_step(**kwargs)

                # The observer watches the result of expert's action and how
                # it modified the environment.
                self.step((obs, reward, done, info), **kwargs)

            print(
                f"Episode: {self.episode} - "
                f"accumulated reward: {self.accumulated_reward:.2f}"
            )
            self.episode += 1

    def train_by_observation_action(
            self,
            action_interval: int = 10,
            num_repeated_actions: int = 1,
            **kwargs
    ) -> None:
        """
        Train observer agent by frequent observation-action trials.

        Parameters
        ----------
        action_interval: int, optional
            Number of observation episodes before an action taking episode.
            The default is 10.
        num_repeated_actions: int, optional
            Number of actions to be taken after observation. The default is 1.

        Keyword Arguments
        -----------------

        Returns
        -------
        None

        """
        self.observer_agent.network.train(True)

        while self.episode < self.num_episodes:
            self.action_function = self.expert_agent.select_action
            self.reset_state_variables()
            done = False
            while not done:
                obs, reward, done, info = self.env_step(**kwargs)

                self.step((obs, reward, done, info), **kwargs)

            print("Observing...\n"
                f"Episode: {self.episode} - "
                f"accumulated reward: {self.accumulated_reward:.2f}"
            )

            if (self.episode + 1) % action_interval == 0:
                self.action_function = self.observer_agent.select_action
                for repeated_action_counts in range(num_repeated_actions):
                    self.env_reset()
                    done = False
                    while not done:
                        obs, reward, done, info = self.env_step(**kwargs)

                        self.step((obs, reward, done, info), **kwargs)

                    print("Taking action...\n"
                        f"Episode: {repeated_action_counts} - "
                        f"accumulated reward: {self.accumulated_reward:.2f}"
                    )

            self.episode += 1

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
        self.observer_agent.network.train(False)

        self.reset_state_variables()

        self.action_function = None
        self.action = self.env.action_space.sample()
        obs, reward, done, info = self.env_step(**kwargs)
        self.step((obs, reward, done, info), **kwargs)

        self.action_function = self.observer_agent.select_action

        while not done:
            # The result of observer's action.
            obs, reward, done, info = self.env_step(**kwargs)

            self.step((obs, reward, done, info), **kwargs)

    def env_reset(self) -> None:
        self.env.reset()
        self.accumulated_reward = 0.0
        self.step_count = 0
        self.overlay_start = True
        self.action = torch.tensor(-1)
        self.last_action = torch.tensor(-1)
        self.action_counter = 0
