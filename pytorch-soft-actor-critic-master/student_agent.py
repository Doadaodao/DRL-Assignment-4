import gymnasium as gym
import torch
import numpy as np
from sac import SAC
import argparse

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

        parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
        parser.add_argument('--env-name', default="HalfCheetah-v2",
                            help='Mujoco Gym environment (default: HalfCheetah-v2)')
        parser.add_argument('--policy', default="Gaussian",
                            help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
        parser.add_argument('--eval', type=bool, default=True,
                            help='Evaluates a policy a policy every 10 episode (default: True)')
        parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                            help='discount factor for reward (default: 0.99)')
        parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                            help='target smoothing coefficient(τ) (default: 0.005)')
        parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                            help='learning rate (default: 0.0003)')
        parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                            help='Temperature parameter α determines the relative importance of the entropy\
                                    term against the reward (default: 0.2)')
        parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                            help='Automaically adjust α (default: False)')
        parser.add_argument('--seed', type=int, default=123456, metavar='N',
                            help='random seed (default: 123456)')
        parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                            help='batch size (default: 256)')
        parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                            help='maximum number of steps (default: 1000000)')
        parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                            help='hidden size (default: 256)')
        parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                            help='model updates per simulator step (default: 1)')
        parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                            help='Steps sampling random actions (default: 0)')
        parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                            help='Value target update per no. of updates per step (default: 1)')
        parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                            help='size of replay buffer (default: 10000000)')
        parser.add_argument('--cuda', action="store_true",
                            help='run on CUDA (default: False)')
        args = parser.parse_args()
        
        self.agent = SAC(env.observation_space.shape[0], env.action_space, args)
        
        self.agent.load_model("./checkpoints/sac_checkpoint_humanoid-walk_999")

    def act(self, observation):
        return self.agent.select_action(observation, evaluate=True)
