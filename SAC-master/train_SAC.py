import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys
import os
import datetime
import random
from collections import deque
from pathlib import Path
from tensordict import TensorDict # type: ignore
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage # type: ignore

from agent import SAC
from logger import MetricLogger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env


def make_env():
	# Create environment with state observations
	env_name = "humanoid-walk"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env


if __name__ == "__main__":

    env = make_env()
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]

    print(f"Number of states:{n_states}\n"
          f"Number of actions:{n_actions}\n"
          f"Action boundaries:{action_bounds}")
    
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    ENV_NAME = "humanoid-walk"
    memory_size = 1e+6
    batch_size = 256
    gamma = 0.99
    alpha = 1
    lr = 3e-4
    reward_scale = 20


    agent = SAC(env_name=ENV_NAME,
                n_states=n_states,
                n_actions=n_actions,
                memory_size=memory_size,
                batch_size=batch_size,
                gamma=gamma,
                alpha=alpha,
                lr=lr,
                action_bounds=action_bounds,
                reward_scale=reward_scale,
                save_dir = save_dir, 
                save_interval = 2e5)

    agent.load_model("./checkpoints/2025-05-10T15-26-02/mario_net_34.chkpt")

    logger = MetricLogger(save_dir)
    episodes = 20000

    for e in range(episodes):

        state, _ = env.reset()
        terminated = False
        truncated = False

        # Play the game!
        while not (terminated or truncated):

            # Run agent on the state
            action = agent.act(state)

            # Agent performs action
            next_state, reward, terminated, truncated, info = env.step(action)

            # Remember
            agent.cache(state, next_state, action, reward, terminated, truncated)

            # Learn
            q, critic_loss, _ = agent.update()

            # Logging
            logger.log_step(reward, critic_loss, q)

            # Update state
            state = next_state

        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, step=agent.curr_step)
