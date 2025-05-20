import datetime
import numpy as np
import torch
from SAC_agent import SAC
from logger import MetricLogger
from pathlib import Path

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

def make_env():
	# Create environment with state observations
	env_name = "humanoid-walk"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env

def main():
    env = make_env()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    agent = SAC(num_inputs = env.observation_space.shape[0], 
                            action_space=env.action_space,
                            gamma=0.99,
                            tau=0.005, 
                            alpha=0.2, 
                            target_update_interval=1, 
                            automatic_entropy_tuning=True, 
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            hidden_size=256,
                            lr=0.0003,
                            buffer_size = 1000000, 
                            batch_size = 256, 
                            save_dir = save_dir, 
                            save_interval = 5e3)

    logger = MetricLogger(save_dir)

    episodes = 400000

    for e in range(episodes):

        state, _ = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Run agent on the state
            action = agent.act(state)

            # Agent performs action
            next_state, reward, terminated, truncated, info = env.step(action)

            # Remember
            agent.cache(state, next_state, action, reward, terminated, truncated)

            # Learn
            q, policy_loss, _ = agent.update()

            # Logging
            logger.log_step(reward, policy_loss, q)

            # Update state
            state = next_state

        logger.log_episode()
        
        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, step=agent.curr_step)

    env.close()

if __name__ == "__main__":
    main()