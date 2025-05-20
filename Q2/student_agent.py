import torch
import numpy as np
from DDPG import DDPG

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        def make_env():
            # Create environment with state observations
            env_name = "cartpole-balance"
            env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
            return env
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
                 save_dir=None,
                 save_interval=1000
                 )
        
        self.agent.load_model("./cart_pole_25.chkpt")

    def act(self, observation):
        return self.agent.act(observation)
