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
from bindsnet.learning.reward import AbstractReward, MovingAvgRPE
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_weights

from agents import CartPoleObserverAgent, ExpertAgent
from pipelines import AgentPipeline


def compute_spikes(datum, time, low, high, device):
    times = torch.linspace(low, high, time, device=device)
    spike_times = torch.argmin(torch.abs(datum - times))
    spikes = (np.array(spike_times.to('cpu')).astype(int) ==
                    range(0, time)).astype(int)
    reverse_spikes = np.flip(spikes).copy()
    return torch.stack([
                torch.from_numpy(spikes).to(device),
                torch.from_numpy(reverse_spikes).to(device)
            ]).byte()


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
    device = "cpu" if datum.get_device() < 0 else 'cuda'
    datum = datum.squeeze()
    angle, velocity = datum[2:4]
    min_angle, max_angle = -0.418, 0.418
    min_velocity, max_velocity = -2, 2

    angle_spikes = compute_spikes(angle, time, min_angle, max_angle, device)
    velocity_spikes = compute_spikes(velocity, time, min_velocity,
                                     max_velocity, device)

    spikes = torch.stack([angle_spikes, velocity_spikes]).T

    return {"S2": spikes.unsqueeze(1).byte().to(device)}

def noise_policy(episode, num_episodes, **kwargs):
    return (1 - episode / num_episodes) ** 2
    # return np.exp(-4 * episode/num_episodes)


class CartPoleReward(AbstractReward):
    def __init__(self, **kwargs):
        self.alpha = 1.0
        self.accumulated_rewards = []

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
        return reward * self.alpha if reward > 0 else -self.alpha

    def update(self, **kwargs):
        episode = kwargs.get("episode", 0) + 1
        accumulated_reward = kwargs.get("accumulated_reward")
        self.accumulated_rewards.append(accumulated_reward)
        if np.mean(self.accumulated_rewards[-10:]) >= 195:
            self.alpha *= 0.1
        # self.alpha *= np.exp(-1 / accumulated_reward)
        # if episode % 10 == 0:
        #     self.alpha *= 0.8


environment = GymEnvironment('CartPole-v0')
environment.reset()

observer = CartPoleObserverAgent(environment, dt=1.0, method='first_spike',
                                 reward_fn=CartPoleReward)

expert = ExpertAgent(environment, method='from_weight',
                     noise_policy=noise_policy)

pipeline = AgentPipeline(
    observer_agent=observer,
    expert_agent=expert,
    encoding=cartpole_observation_encoder,
    time=15,
    num_episodes=100,
    representation_time=7
    # log_writer=True,
    # plot_interval=1,
    # render_interval=1
)

w1 = pipeline.network.connections[("S2", "PM")].w
# plot_weights(w1)
print(w1)

pipeline.train_by_observation(weight='/home/atenagm/hill_climbing.pt',
                              test_interval=10, num_tests=5)
print("Observation Finished")
#
w1 = pipeline.network.connections[("S2", "PM")].w
# plot_weights(w1)
print(w1)

print(pipeline.test_rewards)
test_rewards = np.array(pipeline.test_rewards).reshape(-1, 5)
print(list(map(np.mean, test_rewards)))

# for i in range(100):
#     pipeline.test()

# test_rewards = pipeline.test_rewards[-100:]
# print("min:", np.min(test_rewards), "max:", np.max(test_rewards), "average:",
#        np.mean(test_rewards))
# plt.ioff()
# plt.plot(pipeline.test_rewards)
# plt.show()
