import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys
import os
import random
from collections import deque
from pathlib import Path
from tensordict import TensorDict # type: ignore #
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

def make_env():
	# Create environment with state observations
	env_name = "humanoid-walk"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env

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
                action_bound, sigma, tau, buffer_size, minimal_size, batch_size, device, env, save_dir, save_interval):
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
        self.action_bound = action_bound
        self.sigma = sigma 
        self.tau = tau

        self.device = device
        self.env = env
        self.buffer_size = buffer_size
        self.minimal_size = minimal_size
        self.batch_size = batch_size
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(buffer_size, device=torch.device("cpu")))

        self.curr_step = 0
        self.save_dir = save_dir
        self.save_interval = save_interval

    def act(self, state):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().detach().numpy().flatten()
        noise = np.random.normal(0, self.sigma, size=self.action_dim)
        self.curr_step += 1
        return np.clip(action + noise, -self.action_bound, self.action_bound)
    
    def act_without_noise(self, state):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().detach().numpy().flatten()
        self.curr_step += 1
        return np.clip(action, -self.action_bound, self.action_bound)

    def cache(self, state, next_state, action, reward, terminated, truncated):

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float).to(self.device)
        terminated = torch.tensor([terminated], dtype=torch.float).to(self.device)
        truncated = torch.tensor([truncated], dtype=torch.float).to(self.device)
        
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "terminated": terminated, "truncated": truncated}, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, terminated, truncated = (batch.get(key) for key in ("state", "next_state", "action", "reward", "terminated", "truncated"))
        return state, next_state, action, reward, terminated, truncated
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self):
        if self.curr_step % self.save_interval == 0:
            save_path = (self.save_dir / f"mario_net_{int(self.curr_step // self.save_interval)}.chkpt")
            self.save_model(save_path)

        state, next_state, action, reward, terminated, truncated = self.recall()
        done = (terminated.bool() | truncated.bool()).float()

        # print(state.shape, next_state.shape, action.shape, reward.shape, terminated.shape, truncated.shape, done.shape)
        
        q_targets = reward + self.gamma * self.target_critic(next_state, self.target_actor(next_state)) * (1.0 - done)
        critic_loss = torch.mean(F.mse_loss(q_targets, self.critic(state, action)))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(state, self.actor(state)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  
        self.soft_update(self.critic, self.target_critic)

        return q_targets.mean().item(), critic_loss.item(), actor_loss.item()
    
    def save_model(self, path):
        torch.save({'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'target_actor': self.target_actor.state_dict(),
                'target_critic': self.target_critic.state_dict()}, path)
        print(f"[DDPG] checkpoint saved to «{path}»")


    def load_model(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        if "target_actor" in ckpt and "target_critic" in ckpt:
            self.target_actor.load_state_dict(ckpt["target_actor"])
            self.target_critic.load_state_dict(ckpt["target_critic"])
        else:      
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
