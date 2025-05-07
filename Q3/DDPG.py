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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        # action_bound是环境可以接受的动作最大值
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound

class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        # 拼接状态和动作
        cat = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma,
                action_bound, sigma, tau, buffer_size, minimal_size, batch_size, device, numOfEpisodes, env):
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.device = device
        self.env = env
        self.numOfEpisodes = numOfEpisodes
        self.buffer_size = buffer_size
        self.minimal_size = minimal_size
        self.batch_size = batch_size

    def take_action(self, state):
        state = torch.FloatTensor(np.array([state])).to(self.device)
        action = self.actor(state).item()
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        terminateds = torch.tensor(transition_dict['terminateds'], dtype=torch.float).view(-1, 1).to(self.device)
        truncateds = torch.tensor(transition_dict['truncateds'], dtype=torch.float).view(-1, 1).to(self.device)
        q_targets = rewards + self.gamma * (self.target_critic(next_states, self.target_actor(next_states))) * (1 - terminateds + truncateds)
        critic_loss = torch.mean(F.mse_loss(q_targets, self.critic(states, actions)))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    def DDPGtrain(self):
        replay_buffer = replayBuffer(self.buffer_size)
        returnList = []
        for i in range(10):
            with tqdm(total=int(self.numOfEpisodes / 10), desc='Iteration %d' % i) as pbar:
                for episode in range(int(self.numOfEpisodes / 10)):
                    # initialize state
                    state, info = self.env.reset()
                    terminated = False
                    truncated = False
                    episodeReward = 0
                    # Loop for each step of episode:
                    while (not terminated) or (not truncated):
                        action = self.take_action(state)
                        next_state, reward, terminated, truncated, info = self.env.step(action)
                        replay_buffer.add(state, action, reward, next_state, terminated, truncated)
                        state = next_state
                        episodeReward += reward
                        # 当buffer数据的数量超过一定值后,才进行Q网络训练
                        if replay_buffer.size() > self.minimal_size:
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
                    if (episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.numOfEpisodes / 10 * i + episode + 1),
                            'return':
                                '%.3f' % np.mean(returnList[-10:])
                        })
                    pbar.update(1)
        return returnList

class replayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append( (state, action, reward, next_state, done) )
    
    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
def make_env():
	# Create environment with state observations
	env_name = "humanoid-walk"
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
                 numOfEpisodes=200,
                 env=env)
    agent.DDPGtrain()
    agent.save_model('ddpg_model.pth')
