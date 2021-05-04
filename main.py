import copy
import numpy as np 
import torch 
import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter 


from utils import *
from td3 import Actor, Critic
import argparse


class Agent(object): 
	def __init__(self, state_dim, act_dim, max_action
				 gamma=0.99, tau=0.05, policy_noise=0.2, noise_clip=0.5, freq=2): 
		
		self.state_dim = state_dim
		self.act_dim = act_dim 
		self.max_action = max_action
		self.gamma = gamma # discount factor 
		self.tau = tau # exponential averaging factor
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.freq = freq

		self.actor = Actor(self.state_dim, self.act_dim, self.max_action)
		self.critic1 = Critic(self.state_dim, self.act_dim)
		self.critic2 = Critic(self.state_dim, self.act_dim)

		self.actor_target = copy.deepcopy(self.actor)
		self.critic1_target = copy.deepcopy(self.critic1)
		self.critic2_target = copy.deepcopy(self.critic2)

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)



	def train(self, buffer, batch_size=256): 
		state, action, next_state, reward, done = buffer.sample(batch_size)




if __name__ == "__main__": 
	parser = argparse.ArgumentParser('--env', default='HalfCheetah-v2')
	parser = argparse.ArgumentParser('--mode', default='train')
	args.parser.parse_args()



	env = gym.make(args.env)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n
	max_action = env.action_space.high[0]


