import numpy as np
from model import PolicyNetwork, QvalueNetwork, ValueNetwork
import torch
from replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import random
from collections import deque

from tensordict import TensorDict # type: ignore
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage # type: ignore



def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class ValueNetwork(nn.Module):
    def __init__(self, n_states, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        return self.value(x)


class QvalueNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.q_value(x)


class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
        self.action_bounds = action_bounds

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    def sample_or_likelihood(self, states):
        dist = self(states)
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob



class SAC:
    def __init__(self, env_name, n_states, n_actions, memory_size, batch_size, gamma, alpha, lr, action_bounds,
                 reward_scale, save_dir, save_interval):
        self.env_name = env_name
        self.n_states = n_states
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.action_bounds = action_bounds
        self.reward_scale = reward_scale
        # self.memory = Memory(memory_size=self.memory_size)
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(memory_size, device=torch.device("cpu")))

        self.curr_step = 0
        self.save_dir = save_dir
        self.save_interval = save_interval

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy_network = PolicyNetwork(n_states=self.n_states, n_actions=self.n_actions,
                                            action_bounds=self.action_bounds).to(self.device)
        self.q_value_network1 = QvalueNetwork(n_states=self.n_states, n_actions=self.n_actions).to(self.device)
        self.q_value_network2 = QvalueNetwork(n_states=self.n_states, n_actions=self.n_actions).to(self.device)
        self.value_network = ValueNetwork(n_states=self.n_states).to(self.device)
        self.value_target_network = ValueNetwork(n_states=self.n_states).to(self.device)
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

        self.value_loss = torch.nn.MSELoss()
        self.q_value_loss = torch.nn.MSELoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.lr)
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.lr)
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.lr)

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


    # def store(self, state, reward, done, action, next_state):
    #     def first_if_tuple(x):
    #         return x[0] if isinstance(x, tuple) else x
    #     state = first_if_tuple(state).__array__()
    #     next_state = first_if_tuple(next_state).__array__()
        
    #     state = from_numpy(state).float().to("cpu")
    #     reward = torch.Tensor([reward]).to("cpu")
    #     done = torch.Tensor([done]).to("cpu")
    #     action = torch.Tensor([action]).to("cpu")
    #     next_state = from_numpy(next_state).float().to("cpu")
    #     self.memory.add(state, reward, done, action, next_state)

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, terminated, truncated = (batch.get(key) for key in ("state", "next_state", "action", "reward", "terminated", "truncated"))
        return state, next_state, action, reward, terminated, truncated
    
    # def unpack(self, batch):
    #     batch = Transition(*zip(*batch))

    #     states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)
    #     rewards = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)
    #     dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
    #     actions = torch.cat(batch.action).view(-1, self.n_actions).to(self.device)
    #     next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states).to(self.device)

    #     return states, rewards, dones, actions, next_states

    def update(self):
        if self.curr_step % self.save_interval == 0:
            save_path = (self.save_dir / f"net_{int(self.curr_step // self.save_interval)}.chkpt")
            self.save_model(save_path)

        if len(self.memory) < self.batch_size:
            return 0, 0, 0
        else:
            states, next_states, actions, rewards, terminated, truncated = self.recall()
            dones = (terminated.bool() | truncated.bool()).float()

            # batch = self.memory.sample(self.batch_size)
            # states, rewards, dones, actions, next_states = self.unpack(batch)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.alpha * log_probs.detach()

            value = self.value_network(states)
            value_loss = self.value_loss(value, target_value)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.reward_scale * rewards + \
                           self.gamma * self.value_target_network(next_states) * (1 - dones)
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.q_value_loss(q1, target_q)
            q2_loss = self.q_value_loss(q2, target_q)

            policy_loss = (self.alpha * log_probs - q).mean()
            
            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()


            self.soft_update_target_network(self.value_network, self.value_target_network)

            return value_loss.item(), 0.5 * (q1_loss + q2_loss).item(), policy_loss.item()

    def act(self, state):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(state)
        return np.clip(action, self.action_bounds[0], self.action_bounds[1])
    
    def save_model(self, path):
        torch.save({'policy_network': self.policy_network.state_dict(),
                    'q_value_network1': self.q_value_network1.state_dict(),
                    'q_value_network2': self.q_value_network2.state_dict(),
                    'value_network': self.value_network.state_dict(),
                    'value_target_network': self.value_target_network.state_dict()}, path)
        # torch.save({'actor': self.actor.state_dict(),
        #         'critic': self.critic.state_dict(),
        #         'target_actor': self.target_actor.state_dict(),
        #         'target_critic': self.target_critic.state_dict()}, path)
        print(f"[SAC] checkpoint saved to «{path}»")

    def load_model(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)

        self.policy_network.load_state_dict(ckpt["policy_network"])
        self.q_value_network1.load_state_dict(ckpt["q_value_network1"])
        self.q_value_network2.load_state_dict(ckpt["q_value_network2"])
        self.value_network.load_state_dict(ckpt["value_network"])
        self.value_target_network.load_state_dict(ckpt["value_target_network"])
        
        print(f"[SAC] checkpoint loaded from «{path}»")


    @staticmethod
    def soft_update_target_network(local_network, target_network, tau=0.005):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def save_weights(self):
        torch.save(self.policy_network.state_dict(), self.env_name + "_weights.pth")

    def load_weights(self):
        self.policy_network.load_state_dict(torch.load(self.env_name + "_weights.pth"))

    def set_to_eval_mode(self):
        self.policy_network.eval()

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
    