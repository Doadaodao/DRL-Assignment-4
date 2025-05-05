import gymnasium as gym
import numpy as np
import torch   
from PPO import PPO, Config

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.env = gym.make('Pendulum-v1')
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