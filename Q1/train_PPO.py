import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from PPO import PPO, replayBuffer


def main():
    env = gym.make('Pendulum-v1')

    # save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # save_dir.mkdir(parents=True)

    agent = PPO(
        state_dim=env.observation_space.shape[0],
        hidden_layers_dim=[64, 64, 64],
        action_dim=env.action_space.shape[0],
        actor_lr=1e-4,
        critic_lr=5e-3,
        gamma=0.9,
        PPO_kwargs={'lmbda': 0.9, 'eps': 0.2, 'ppo_epochs': 10},
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    buffer_size = 100000

    episodes = 1200
    max_steps = 260
    
    rewards = []
    avg_reward = 0

    tq_bar = tqdm(range(episodes))
    for i in tq_bar:
        buffer = replayBuffer(buffer_size)
        tq_bar.set_description(f'Episode [ {i+1} / {episodes} ]')    
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action = agent.policy(state)
            next_state, reward, done, _, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1
            if steps >= max_steps:
                break

        agent.update(buffer.buffer)
        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-10:])

        if (i % 20 == 0) or (i == episodes - 1):
            torch.save(agent.actor.state_dict(), "PPO_ckpt.pth")
        tq_bar.set_postfix({'MeanRewards': f'{avg_reward:.2f}'})
    env.close()

if __name__ == '__main__':
    main()