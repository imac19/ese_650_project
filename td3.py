import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 	
import os 



class Critic(nn.Module): 
	def __init__(self, state_dim, act_dim): 
		super(Critic, self).__init__()

		
		self.net = nn.Sequential(
				nn.Linear(state_dim+act_dim, 256),
				nn.ReLU(), 
				nn.Linear(256, 256), 
				nn.ReLU(), 
				nn.Linear(256, 1)
			)

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(device)


	def forward(self, state, action): 
		# state action pair 
		sa = torch.cat([state, action], dim=1)
		q = self.net(sa)
		return q 


	def save(self, filename): 
		''' save with .pt or .pth file extension'''
		torch.save(self.state_dict(), filename)


	def load(self, filename): 
		self.load_state_dict(torch.load(filename))



class Actor(nn.Module): 
	def __init__(self, state_dim, act_dim, max_action): 
		super(Actor, self).__init__()

		self.max_action = max_action

		self.net = nn.Sequential(
				nn.Linear(state_dim, 256),
				nn.ReLU(), 
				nn.Linear(256, 256), 
				nn.ReLU(), 
				nn.Linear(256, act_dim)
			)

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(device)

	def forward(self, state): 
		a = self.net(state)
		return self.max_action * torch.tanh(a)


	def save(self, filename): 
		torch.save(self.state_dict(), filename)


	def load(self, filename): 
		self.load_state_dict(torch.load(filename))

