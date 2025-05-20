import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from tensordict import TensorDict # type: ignore
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

def make_env():
	# Create environment with state observations
	env_name = "humanoid-walk"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.linear1 = torch.nn.Linear(state_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QValueNet, self).__init__()

        self.linear1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1)

        self.linear4 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class GaussianPolicyNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicyNet, self).__init__()
        
        self.linear1 = torch.nn.Linear(num_inputs, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = torch.nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = torch.nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal_dist = Normal(mean, std)
        x_t = normal_dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        log_prob = normal_dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyNet, self).to(device)

class SAC(object):
    def __init__(self, state_dim, action_space, gamma, tau, alpha, 
                 automatic_entropy_tuning, device,
                 hidden_dim, lr, buffer_size, minimal_size, batch_size, save_dir, save_interval):
        self.device = device

        self.critic = QValueNet(state_dim, action_space.shape[0], hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.critic_target = QValueNet(state_dim, action_space.shape[0], hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)

        self.policy = GaussianPolicyNet(state_dim, action_space.shape[0], hidden_dim, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(buffer_size, device=torch.device("cpu")))

        self.curr_step = 0
        self.save_dir = save_dir
        self.save_interval = save_interval

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
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
            save_path = (self.save_dir / f"net_{int(self.curr_step // self.save_interval)}.chkpt")
            self.save_model(save_path)

        state, next_state, action, reward, terminated, truncated = self.recall()
        done = (terminated.bool() | truncated.bool()).float()

        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi 
            next_q_value = reward + done * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state, action)  
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value) 
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state)

        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            
        else:
            alpha_loss = torch.tensor(0.).to(self.device)


        self.soft_update(self.critic, self.critic_target)

        return qf1.mean().item(), policy_loss.item(), alpha_loss.item()

    # Save model parameters
    def save_model(self, path):
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, 
                    path)
        
    # Load model parameters
    def load_model(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)

        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
        self.critic_optim.load_state_dict(ckpt["critic_optimizer_state_dict"])
        self.policy_optim.load_state_dict(ckpt["policy_optimizer_state_dict"])

        print(f"[SAC] checkpoint loaded from «{path}»")
