import gymnasium as gym
import torch
import numpy as np
from DDPG import DDPG, make_env

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        env = make_env()
        self.agent = DDPG(state_dim=env.observation_space.shape[0],
                 hidden_dim=256,
                 action_dim=env.action_space.shape[0],
                 actor_lr=3e-4,
                 critic_lr=3e-3,
                 gamma=0.99,
                 action_bound=env.action_space.high[0],
                 sigma=0.01,
                 tau=0.005,
                 buffer_size=10000,
                 minimal_size=1000,
                 batch_size=64,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                 env=env,
                 save_dir="./checkpoints",
                 save_interval=1000
                 )
        
        self.agent.load_model("./checkpoints/2025-05-09T09-08-58/mario_net_11.chkpt")

    def act(self, observation):
        return self.agent.act(observation)
        # return self.action_space.sample()
