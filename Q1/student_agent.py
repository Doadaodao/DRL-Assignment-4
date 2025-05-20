import gymnasium as gym
import torch   
from PPO import PPO

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        env = gym.make('Pendulum-v1')
        
        self.agent = PPO(
            state_dim=env.observation_space.shape[0],
            hidden_layers_dim=[64, 64, 64],
            action_dim=env.action_space.shape[0],
            actor_lr=1e-4,
            critic_lr=5e-3,
            gamma=0.9,
            lmbda=0.9,
            eps=0.2,
            ppo_epochs=10,
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.agent.actor.load_state_dict(torch.load("./PPO_Pendulum.pth"))

    def act(self, observation):
        action = self.agent.policy(observation)
        return action