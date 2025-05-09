import gymnasium as gym
import numpy as np
import torch   
from PPO_cart_pole import PPO, Config

import typing as typ
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

def make_env():
    # Create environment with image observations
    env_name = "cartpole-balance"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        # self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.env = make_env()
        self.cfg = Config(self.env)
        self.agent = PPO(
            state_dim=self.cfg.state_dim,
            hidden_layers_dim=self.cfg.hidden_layers_dim,
            action_dim=self.cfg.action_dim,
            actor_lr=self.cfg.actor_lr,
            critic_lr=self.cfg.critic_lr,
            gamma=self.cfg.gamma,
            PPO_kwargs=self.cfg.PPO_kwargs,
            device=self.cfg.device
        )
        self.agent.actor.load_state_dict(torch.load(self.cfg.save_path))

    def act(self, observation):
        action = self.agent.policy(observation)
        return action