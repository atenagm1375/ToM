"""
==============================================================================.

cartpole.py

@author: atenagm

==============================================================================.
"""
from bindsnet.environment import GymEnvironment
from bindsnet.learning.reward import MovingAvgRPE

from agents import ObserverAgent


environment = GymEnvironment('CartPole-v1')
environment.reset()

observer = ObserverAgent(environment, dt=1.0, reward_fn=MovingAvgRPE)

# environment_pipeline = EnvironmentPipeline(
#     agent,
#     environment,
#     encoding=cartpole_observation_encoder,
#     action_function=expert_agent_action,
#     output="Output Layer",
#     time=100,
#     history_length=5,
#     delta=1,
#     plot_interval=1,
#     render_interval=1,
#     percent_of_random_action=0.01
# )
