import torch
import numpy as np
from SAC import SAC

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
        
        self.agent = SAC(num_inputs = env.observation_space.shape[0], 
                         action_space=env.action_space,
                         gamma=0.99,
                         tau=0.005, 
                         alpha=0.2, 
                         target_update_interval=1, 
                         automatic_entropy_tuning=True, 
                         device="cuda" if torch.cuda.is_available() else "cpu",
                         hidden_size=256,
                         lr=0.0003)

        self.agent.load_checkpoint("./sac_checkpoint")

    def act(self, observation):
        return self.agent.select_action(observation, evaluate=False)
