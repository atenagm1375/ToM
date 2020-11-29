"""
==============================================================================.

pipelines.py

@author: atenagm

==============================================================================.
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

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

        self.observer_agent = observer_agent
        self.observer_agent.build_network()

        super().__init__(
            observer_agent.network,
            observer_agent.environment,
            encoding=encoding,
            allow_gpu=observer_agent.allow_gpu,
            **kwargs,
        )

        self.expert_agent = expert_agent

        self.representation_time = kwargs.get('representation_time', -1)

        self.plot_config = {
            "data_step": True,
        }

        self.test_rewards = []
        self.time_recorder = 0

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
            self.action = self.action_function(episode=self.episode,
                                               num_episodes=self.num_episodes,
                                               **kwargs)

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

        inputs = self.encoding(obs, self.time, **kwargs)

        pm_n = self.network.layers["PM"].n
        n_action = self.env.action_space.n
        pm_v = torch.zeros(self.time, pm_n)
        pm_v[self.representation_time, pm_n//n_action * self.action: \
                                    pm_n//n_action * (self.action + 1)] = 1

        pm_v = pm_v.view(self.time, n_action, -1).byte()
        clamp = {
            "PM": pm_v.to(self.device)
        } if self.network.learning else {}

        self.network.run(inputs=inputs, clamp=clamp, time=self.time,
                         reward=reward, curr_state=obs.squeeze(), **kwargs)

        if kwargs.get("log_path") is not None and not self.observer_agent.network.learning:
            self._log_info(kwargs["log_path"], obs.squeeze(), inputs)

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

        test_interval = kwargs.get("test_interval", None)
        num_tests = kwargs.get("num_tests", 1)
        log_count = 0
        while self.episode < self.num_episodes:
            self.observer_agent.network.train(True)
            # Expert acts in the environment.
            self.action_function = self.expert_agent.select_action
            self.reset_state_variables()

            last_state = torch.Tensor(self.env.env.state)

            prev_obs, prev_reward, prev_done, info = self.env_step(**kwargs)

            new_done = False
            while not prev_done:
                self.network.reset_state_variables()
                prev_done = new_done
                # The result of expert's action.
                if not prev_done:
                    new_obs, new_reward, new_done, info = self.env_step(**kwargs)

                # The observer watches the result of expert's action and how
                # it modified the environment.
                self.step((prev_obs, prev_reward, prev_done, info),
                          last_state=last_state, **kwargs)

                self._save_simulation_info(**kwargs)

                last_state = prev_obs.squeeze()
                prev_obs = new_obs
                prev_reward = new_reward

            print(
                f"Episode: {self.episode} - "
                f"accumulated reward: {self.accumulated_reward:.2f}"
            )

            if test_interval is not None:
                if (self.episode + 1) % test_interval == 0:
                    for nt in range(num_tests):
                        self.test(num=(nt + 1) + (log_count * 5), **kwargs)
                    log_count += 1

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
        # TODO reconsider
        self.observer_agent.network.train(True)

        while self.episode < self.num_episodes:
            self.action_function = self.expert_agent.select_action
            self.reset_state_variables()
            done = False
            while not done:
                self.network.reset_state_variables()
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
                        self.network.reset_state_variables()
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

        self.action_function = self.observer_agent.select_action
        obs = torch.Tensor(self.env.env.state).to(self.observer_agent.device)
        self.step((obs, 1.0, False, {}), **kwargs)

        self._save_simulation_info(**kwargs)

        # obs, reward, done, info = self.env_step(**kwargs)
        # self.step((obs, reward, done, info), **kwargs)
        done = False
        while not done:
            # The result of observer's action.
            obs, reward, done, info = self.env_step(**kwargs)

            self.network.reset_state_variables()
            self.step((obs, reward, done, info), **kwargs)

            self._save_simulation_info(**kwargs)

        self.test_rewards.append(self.reward_list.pop())
        print("Test - accumulated reward:", self.accumulated_reward)

    def env_reset(self) -> None:
        self.env.reset()
        self.accumulated_reward = 0.0
        self.step_count = 0
        self.overlay_start = True
        self.action = torch.tensor(-1)
        self.last_action = torch.tensor(-1)
        self.action_counter = 0

    def _save_simulation_info(self, **kwargs):
        spikes = torch.cat([
            self.network.monitors["S2"].get("s").squeeze(),
            self.network.monitors["MT"].get("s").squeeze(),
            self.network.monitors["PM"].get("s").squeeze(),
        ], dim=1).nonzero()

        if self.observer_agent.network.learning:
            with open(f"train_spikes{self.episode}.txt", 'a') as tr:
                for spike in spikes:
                    tr.write(f"{spike[0] + self.time_recorder} {spike[1]}\n")
            with open(f"train_weights{self.episode}.txt", 'a') as tr:
                w = torch.cat([
                    self.network.monitors["S2-PM"].get("w"),
                    self.network.monitors["MT-PM"].get("w"),
                ], dim=1)
                for wt in range(len(w)):
                    tr.write(f"{self.time_recorder + wt} {w[wt]}\n")
        else:
            num = kwargs["num"]
            with open(f"test_spikes{num}.txt", 'a') as tr:
                for spike in spikes:
                    tr.write(f"{spike[0] + self.time_recorder} {spike[1]}\n")

        self.time_recorder += self.time

    def _log_info(self, path, obs, encoded_input):
        # TODO change it later for inputs of shape n*m
        plt.ioff()
        fig, axes = plt.subplots(len(encoded_input.keys()), 2)
        for idx, k in enumerate(encoded_input.keys()):
            ss = encoded_input[k].squeeze().to("cpu")
            if len(ss.shape) == 2:
                ss = ss.nonzero()
                axes[idx, 0].scatter(ss[:, 0], ss[:, 1])
                axes[idx, 0].set(xlim=[-1, self.time + 1],
                                 ylim=[-1, encoded_input[k].shape[-1]])
                axes[idx, 0].set_title(k)
            else:
                for i in range(ss.shape[1]):
                    s = ss[:, i, :].nonzero()
                    axes[idx * i, 0].scatter(s[:, 0], s[:, 1])
                    plt.xlim([-1, self.time + 1])
                    plt.ylim([-1, ss.shape[-1]])

        v = self.network.monitors["PM"].get("v").squeeze().to("cpu")
        axes[0, 1].plot(v[:, 0], c='r', label="0")
        axes[0, 1].plot(v[:, 1], c='b', label="1")
        axes[0, 1].set(ylim=[self.network.layers["PM"].rest.to("cpu"),
                       self.network.layers["PM"].thresh.to("cpu")])
        axes[0, 1].legend()

        s = self.network.monitors["PM"].get("s").squeeze().nonzero().to("cpu")
        axes[1, 1].scatter(s[:, 0], s[:, 1])
        axes[1, 1].set(xlim=[-1, self.time + 1], ylim=[-1, 2])
        fig.savefig(path + f"/{self.episode}_{len(self.test_rewards)}_"
                    f"{self.step_count}_{obs}_{self.action}.png")
        fig.clf()
