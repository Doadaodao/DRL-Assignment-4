import numpy as np
import torch
import sys
import os
import datetime
from pathlib import Path
from DDPG import DDPG
from logger import MetricLogger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

def make_env():
    # Create environment with image observations
    env_name = "cartpole-balance"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env

def main():
    env = make_env()
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    agent = DDPG(state_dim=env.observation_space.shape[0],
                 hidden_dim=256,
                 action_dim=env.action_space.shape[0],
                 actor_lr=2.5e-4,
                 critic_lr=2.5e-4,
                 gamma=0.99,
                 action_bound=env.action_space.high[0],
                 sigma=0.2,
                 tau=0.005,
                 buffer_size=1000000,
                 minimal_size=1000,
                 batch_size=64,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                 env=env,
                 save_dir = save_dir, 
                 save_interval = 5e3)

    logger = MetricLogger(save_dir)

    episodes = 400000

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

if __name__ == '__main__':
    main()