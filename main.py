import copy
import numpy as np 
import torch 
import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter 
import torch.nn.functional as F


from utils import *
from td3 import Actor, Critic 
import argparse
import gym 
import matplotlib.pyplot as plt 
import tqdm 

class Agent(object): 
	def __init__(self, state_dim, act_dim, max_action,
				 gamma=0.99, tau=0.05, policy_noise=0.2, noise_clip=0.5, freq=2): 
		
		self.state_dim = state_dim
		self.act_dim = act_dim 
		self.max_action = max_action
		self.gamma = gamma # discount factor 
		self.tau = tau # exponential averaging factor
		self.policy_noise = policy_noise * max_action
		self.noise_clip = noise_clip * max_action
		self.freq = freq

		self.actor = Actor(self.state_dim, self.act_dim, self.max_action)
		self.critic1 = Critic(self.state_dim, self.act_dim)
		self.critic2 = Critic(self.state_dim, self.act_dim)

		self.actor_target = copy.deepcopy(self.actor)
		self.critic1_target = copy.deepcopy(self.critic1)
		self.critic2_target = copy.deepcopy(self.critic2)

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
		self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=1e-3)
		self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=1e-3)

#         self.writer = SummaryWriter()
		self.iters = 0 
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
		
	def train(self, buffer, batch_size=256): 
		state, action, reward, next_state, done = buffer.sample(10)
		reward = reward.view((-1,1))
		done = done.view((-1,1))
		with torch.no_grad(): 
			# compute target actions 
			noise = torch.clip(torch.randn_like(action)*self.policy_noise, -self.noise_clip, self.noise_clip)
			next_action = torch.clip(self.actor_target(next_state) + noise, -self.max_action, self.max_action)
			
			# compute target q 
			target_q1 = self.critic1_target(next_state, next_action)
			target_q2 = self.critic2_target(next_state, next_action)
			target_q = reward + self.gamma*(1-done)*torch.min(target_q1, target_q2)


		# current q value 
		q1 = self.critic1(state, action)
		q2 = self.critic2(state, action)


#         loss = torch.mean((q1-target_q1)** 2 + (q2 - target_q2)**2)
		loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

		self.critic1_optimizer.zero_grad()
		self.critic2_optimizer.zero_grad()
		loss.backward()
		# self.writer.add_scalar('Loss/train', loss, self.iters)
		self.critic1_optimizer.step()
		self.critic2_optimizer.step()

		self.iters += 1 

		# delay target policy update 
		if self.iters % self.freq == 0: 

			loss2 = -self.critic1(state, self.actor(state)).mean()
			self.actor_optimizer.zero_grad()
			loss2.backward()
			self.actor_optimizer.step()

			self.update_target_weights()

	def get_best_action(self, state):
		# print('State:')
		# print(state)
		state = torch.FloatTensor(state['observation']).to(self.device)
		action = self.actor(state).detach().cpu().numpy()
		# print('Action:')
		# print(action)
		return action

	def update_target_weights(self): 
		critic1_params = self.critic1.parameters()
		critic1_target_params = self.critic1_target.parameters()
		for p, tp in zip(critic1_params, critic1_target_params): 
			tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)

		critic2_params = self.critic2.parameters()
		critic2_target_params = self.critic2_target.parameters()
		for p, tp in zip(critic2_params, critic2_target_params): 
			tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)


		actor_params = self.actor.parameters()
		actor_target_params = self.actor_target.parameters()
		for p, tp in zip(actor_params, actor_target_params): 
			tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)

		
	def save(self, file_path='saved', checkpoint=0):
		torch.save(self.actor.state_dict(), file_path + '_actor_checkpoint_{}'.format(checkpoint))
		torch.save(self.critic1.state_dict(), file_path + '_critic1_checkpoint_{}'.format(checkpoint))
		torch.save(self.critic2.state_dict(), file_path + '_critic2_checkpoint_{}'.format(checkpoint))
		
		torch.save(self.actor_target.state_dict(), file_path + '_actor_target_checkpoint_{}'.format(checkpoint))
		torch.save(self.critic1_target.state_dict(), file_path + '_critic1_target_checkpoint_{}'.format(checkpoint))
		torch.save(self.critic2_target.state_dict(), file_path + '_critic2_target_checkpoint_{}'.format(checkpoint))
		
		torch.save(self.actor_optimizer.state_dict(), file_path + '_actor_optimizer_{}'.format(checkpoint))
		torch.save(self.critic1_optimizer.state_dict(), file_path + '_critic_optimizer_checkpoint_{}'.format(checkpoint))
		torch.save(self.critic2_optimizer.state_dict(), file_path + '_critic_optimizer_checkpoint_{}'.format(checkpoint))


	def load(self, file_path='saved', checkpoint=0):
		self.actor.load_state_dict(torch.load(file_path + '_actor_checkpoint_{}'.format(checkpoint)))
		self.critic1.load_state_dict(torch.load(file_path + '_critic1_checkpoint_{}'.format(checkpoint)))
		self.critic2.load_state_dict(torch.load(file_path + '_critic2_checkpoint_{}'.format(checkpoint)))
		
		self.actor_target.load_state_dict(torch.load(file_path + '_actor_target_checkpoint_{}'.format(checkpoint)))
		self.critic1_target.load_state_dict(torch.load(file_path + '_critic1_target_checkpoint_{}'.format(checkpoint)))
		self.critic2_target.load_state_dict(torch.load(file_path + '_critic2_target_checkpoint_{}'.format(checkpoint)))
		
		self.actor_optimizer.load_state_dict(torch.load(file_path + '_actor_optimizer_{}'.format(checkpoint)))
		self.critic_optimizer.load_state_dict(torch.load(file_path + '_critic_optimizer_checkpoint_{}'.format(checkpoint)))






if __name__ == "__main__": 
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', default='FetchReach-v1')
	parser.add_argument('--mode', default='train')
	args = parser.parse_args()
	
	env = gym.make(args.env)
	state_dim = env.observation_space['observation'].shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])
	
	agent = Agent(state_dim, action_dim, max_action)
	if args.mode == 'train': 
		replay_buffer = ReplayBuffer(state_dim, action_dim)
		n_epochs = 100 
		n_episodes = 100
		demonstrations = 2000
	
		reward_list = []

		populate_buffer(env, replay_buffer)

		for epoch in tqdm.tqdm(range(n_epochs)):
			for e in range(n_episodes): 
				transitions, trajectory, startTraining = rollout(env, agent)
				replay_buffer.add(transitions, trajectory)

				tqdm.tqdm.write('Training')
				r = sum([trajectory[i]['reward']] for i in range(len(trajectory)))
				reward_list.append(r)
				agent.train(replay_buffer, batch_size = 2000); 


			replay_buffer.fit()

	plt.figure(1)
	plt.plot(list(range(len(reward_list))), reward_list)
	plt.title('{} Environment'.format(args.env))
	plt.ylabel('Reward')
	plt.xlabel('Episodes')
	plt.savefig('{}.png'.format(args.env))
	plt.show()