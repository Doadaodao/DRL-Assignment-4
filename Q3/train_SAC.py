import datetime
import numpy as np
import torch
from SAC import SAC, ReplayBuffer
from logger import MetricLogger
from pathlib import Path

import os
import sys
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

def make_env():
	# Create environment with state observations
	env_name = "humanoid-walk"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env

def main():
    env = make_env()

    agent = SAC(num_inputs = env.observation_space.shape[0], 
                            action_space=env.action_space,
                            gamma=0.99,
                            tau=0.005, 
                            alpha=0.2, 
                            target_update_interval=1, 
                            automatic_entropy_tuning=True, 
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            hidden_size=256,
                            lr=0.0003)

    buffer_size = 1000000
    batch_size = 256
    memory = ReplayBuffer(buffer_size)

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    logger = MetricLogger(save_dir)

    # Training Loop
    updates = 0

    episodes = 400000

    for e in range(episodes):
        
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)  # Sample action from policy

            critic_1_loss, policy_loss = 0, 0
            
            if len(memory) > batch_size:
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, batch_size, updates)
                updates += 1

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Logging
            logger.log_step(reward, critic_1_loss, policy_loss)

            mask = 1 if terminated or truncated else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

        logger.log_episode()
        
        if ((e + 1) % 20 == 0):
            logger.record(episode=e, step=e*1000)
            agent.save_checkpoint("humanoid-walk", e)

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(e, total_numsteps, episode_steps, round(episode_reward, 2)))


    env.close()

if __name__ == "__main__":
    main()