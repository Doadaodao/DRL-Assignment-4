import torch
import numpy as np
from SAC_agent import SAC

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
            env_name = "humanoid-walk"
            env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
            return env
        env = make_env()
        
        self.agent = SAC(state_dim = env.observation_space.shape[0], 
                action_space=env.action_space,
                gamma=0.99,
                tau=0.005, 
                alpha=0.2, 
                automatic_entropy_tuning=True, 
                device="cuda" if torch.cuda.is_available() else "cpu",
                hidden_dim=256,
                lr=0.0003,
                buffer_size = 1000000, 
                batch_size = 256, 
                save_dir = None, 
                save_interval = 5e3)

        self.agent.load_model("./sac_walk_1759.ckpth")

    def act(self, observation):
        return self.agent.select_action(observation, evaluate=False)
