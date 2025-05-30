import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
# import util
import sys
import os
import random
from collections import deque
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # tanh activation for action output
        x = x * self.action_bound
        return x
    
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        cat = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma,
                action_bound, sigma, tau, buffer_size, minimal_size, batch_size, device, numOfEpisodes, env):
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.gamma = gamma
        self.sigma = sigma 
        self.tau = tau

        self.device = device
        self.env = env
        self.numOfEpisodes = numOfEpisodes
        self.buffer_size = buffer_size
        self.minimal_size = minimal_size
        self.batch_size = batch_size

    def take_action(self, state):
        state = torch.FloatTensor(np.array([state])).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]  
        action = action + self.sigma * np.random.randn(self.action_dim)
        # action = np.clip(action, -self.action_bound, self.action_bound)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = (torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)) 
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        terminateds = torch.tensor(transition_dict['terminateds'], dtype=torch.float).view(-1, 1).to(self.device)
        truncateds = torch.tensor(transition_dict['truncateds'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = (terminateds.bool() | truncateds.bool()).float()

        q_targets = rewards + self.gamma * self.target_critic(next_states, self.target_actor(next_states)) * (1.0 - dones)
        critic_loss = torch.mean(F.mse_loss(q_targets, self.critic(states, actions)))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  
        self.soft_update(self.critic, self.target_critic)

    def evaluate_policy(self, episodes):
        """Evaluate the agent's performance with state observations"""
        episode_rewards = []
        env = make_env()
        for episode in range(episodes):
            state, _ = env.reset(seed=np.random.randint(0, 1000000))
            state, info = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.take_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                episode_reward += reward
                done = terminated or truncated
            episode_rewards.append(episode_reward)
        return np.mean(episode_rewards)

    def DDPGtrain(self):
        replay_buffer = replayBuffer(self.buffer_size)
        returnList = []

        global_step   = 0
        save_interval = 50_000          # env steps
        max_to_keep   = 10              # rolling buffer
        best_mean     = -float("inf")
        ckpt_dir      = Path("checkpoints_stand"); ckpt_dir.mkdir(exist_ok=True)

        episodes_per_train = 100

        for i in range(episodes_per_train):
            with tqdm(total=int(self.numOfEpisodes / episodes_per_train), desc='Iteration %d' % i) as pbar:
                for episode in range(int(self.numOfEpisodes / episodes_per_train)):
                    # initialize state
                    state, info = self.env.reset()
                    terminated = False
                    truncated = False
                    episodeReward = 0
                    # Loop for each step of episode:
                    while not (terminated or truncated):
                        global_step += 1

                        if global_step % save_interval == 0:
                            fname = ckpt_dir / f"ddpg_step_{global_step//1000:06d}k.pt"
                            agent.save_model(fname)
                            rolling = sorted(ckpt_dir.glob("ddpg_step_*.pt"))
                            for old in rolling[:-max_to_keep]:
                                old.unlink()     # drop oldest

                        action = self.take_action(state)
                        next_state, reward, terminated, truncated, info = self.env.step(action)
                        replay_buffer.add(state, action, reward, next_state, terminated, truncated)
                        state = next_state
                        episodeReward += reward
                        
                        if len(replay_buffer) > self.minimal_size:
                            b_s, b_a, b_r, b_ns, b_te, b_tr = replay_buffer.sample(self.batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'terminateds': b_te,
                                'truncateds': b_tr
                            }
                            self.update(transition_dict)
                        if terminated or truncated:
                            break
                    returnList.append(episodeReward)

                    if (episode + 1) % episodes_per_train == 0: 
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.numOfEpisodes / episodes_per_train * i + episode + 1),
                            'return':
                                '%.3f' % np.mean(returnList[-episodes_per_train:])
                        })
                    pbar.update(1)

                    if (episode + 1) % episodes_per_train == 0:
                        mean_return = self.evaluate_policy(episodes=20)
                        if mean_return > best_mean:
                            best_mean = mean_return
                            agent.save_model(ckpt_dir / "best.pt")
                            # print(f"★ new best {best_mean:.1f} at step {global_step}")
        return returnList
    
    def save_model(self, path):
        torch.save({'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict()}, path)
        
    def load_model(self, path, map_location=None):
        """
        Load a checkpoint that was written by self.save_model(path).

        Args
        ----
        path : str
            File name of the checkpoint (e.g. "checkpoints/ddpg_humanoid.pt").
        map_location : str or torch.device or None
            Passed straight to torch.load().  Use None to load on the current
            device, or "cpu" if you saved on GPU but want to inspect on CPU.

        Notes
        -----
        • Restores online *and* target networks so training can continue
        seamlessly.  
        • Optimiser states are **not** stored; add them if you want exact
        training continuation.
        """
        ckpt = torch.load(path, map_location=map_location or self.device)

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        # keep target nets in sync
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        print(f"[DDPG] checkpoint loaded from «{path}»")


class replayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, terminated, truncated):
        self.buffer.append( (state, action, reward, next_state, terminated, truncated) )
    
    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        # Random sample and organize the data
        states, actions, rewards, next_states, terminateds, truncateds = [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        for state, action, reward, next_state, terminated, truncated in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)
            truncateds.append(truncated)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminateds), np.array(truncateds)
    
def make_env():
	# Create environment with state observations
	env_name = "humanoid-stand"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env

if __name__ == '__main__':

    env = make_env()
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    agent = DDPG(state_dim=env.observation_space.shape[0],
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
                 numOfEpisodes=200_000,
                 env=env)
    agent.DDPGtrain()
