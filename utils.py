import numpy as np 
import matplotlib.pyplot as plt
import torch 
import gym



class ReplayBuffer(object): 
	def __init__(self, state_dim, act_dim, max_size=int(1e6)): 
		self.max_size = max_size 
		self.size = 0 
		self.counter = 0 

		self.state = np.zeros((max_size, state_dim))		
		self.action = np.zeros((max_size, act_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


	def add(self, state, action, next_state, reward, done): 
		self.state[self.counter] = state
		self.action[self.counter] = action
		self.next_state[self.counter] = next_state
		self.reward[self.counter] = reward
		self.done[self.counter] = done

		self.counter = self.counter % self.max_size
		self.size = min(self.size+1, self.max_size)


	def sample(self, batch_size=1000): 
		ind = np.random.randint(0, self.size, batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device)
			)


def evaluate(agent, args):
	e = gym.make(args)
	iterations = 100
	all_rs = []
	
	for i in range(0, iterations):
		total_r = 0
		state = e.reset()
		done = False
		
		while not done:
			action = agent.get_best_action(np.array(state))
			state, reward, done, is_success = e.step(action)
			total_r += reward
		all_rs.append(total_r)
	
	r = np.mean(all_rs)
	
	return r
