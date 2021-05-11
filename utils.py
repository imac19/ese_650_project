import numpy as np 
import matplotlib.pyplot as plt
import torch 
import gym
from sklearn.mixture import GaussianMixture
import operator 



class ReplayBuffer(object): 
	def __init__(self, state_dim, act_dim, max_size=int(1e6)): 
		self.max_size = max_size 
		self.size = 0 
		self.counter = 0 

		# self.state = np.zeros((max_size, state_dim))		
		# self.action = np.zeros((max_size, act_dim))
		# self.next_state = np.zeros((max_size, state_dim))
		# self.reward = np.zeros((max_size, 1))
		# self.done = np.zeros((max_size, 1))

		self.states = None 
		self.trajectories = [] 
		self.gmm = GaussianMixture(n_components=1)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


	def add(self, states, trajectory): 
		# self.state[self.counter] = state['observation']
		# self.action[self.counter] = action
		# self.next_state[self.counter] = next_state['observation']
		# self.reward[self.counter] = reward
		# self.done[self.counter] = done

		s = np.array(states)
		if self.counter == 0: 
			self.states = s
		else: 
			self.states = np.vstack((self.states, s)) # num_of_trajectories x num_states

		self.trajectories.append(trajectory)
		self.counter = (self.counter + 1) % self.max_size
		self.size = min(self.size+1, self.max_size)

	def sample(self, batch_size=1000):
		self.update_densities()
		sorted_traj = sorted(self.trajectories, key=operator.itemgetter(-1))
		batch = sorted_traj[:batch_size]
		rand_trans = np.random.randint(0, len(self.trajectories), size=batch_size)
		state = [batch[i][j]['state'] for i,j in enumerate(rand_trans)]
		action = [batch[i][j]['action'] for i,j in enumerate(rand_trans)]
		reward = [batch[i][j]['reward'] for i,j in enumerate(rand_trans)]
		next_state = [batch[i][j]['next_state'] for i,j in enumerate(rand_trans)]
		done = [batch[i][j]['done'] for i,j in enumerate(rand_trans)]

		return ( 
			torch.FloatTensor(state).to(self.device), 
			torch.FloatTensor(action).to(self.device), 
			torch.FloatTensor(reward).to(self.device), 
			torch.FloatTensor(next_state).to(self.device), 
			torch.FloatTensor(done).to(self.device)
			)


	def update_densities(self): 
		self.p = np.sum(self.gmm.predict_proba(self.states))

		for traj in self.trajectories: 
			p_tot = 1
			for t in traj[:50]:
				s = t['state']
				p_tot *= self.gmm.predict_proba(s.reshape((1,-1)))
			traj.append(p_tot/self.p)



	def fit(self): 
		self.gmm.fit(self.states)


def rollout(env, policy): 
	state = env.reset()
	traj = []
	s = []
	max_timesteps = env._max_episode_steps	 
	for _ in range(max_timesteps):
		action = policy.get_best_action(state) + np.random.normal(0,0.1,size=env.action_space.shape[0])
		next_state, reward, done, _ = env.step(action)
		state = next_state
		traj.append({'state':state['observation'], 'action': action, 'next_state': next_state['observation'], 'reward': reward, 'done': done})
		s.append(state['observation'])
	return s, traj



def populate_buffer(env, buffer): 
	init_buffer_size = 10000


	traj = []
	s = []
	# cntr = 0 
	for i in range(init_buffer_size):
		state = env.reset()
		for _ in range(env._max_episode_steps):
			action = env.action_space.sample()
			next_state, reward, done, _ = env.step(action)
			state = next_state
			traj.append({'state':state['observation'], 'action': action, 'next_state': next_state['observation'], 'reward': reward, 'done': done})
			s.append(state['observation'])

		buffer.add(s, traj)
		if i % 1000 == 0: 
			print('iterations {}'.format(i))
# def evaluate(agent, args):
# 	e = gym.make(args)
# 	iterations = 100
# 	all_rs = []
	
# 	for i in range(0, iterations):
# 		total_r = 0
# 		state = e.reset()
# 		done = False
		
# 		while not done:
# 			action = agent.get_best_action(np.array(state))
# 			state, reward, done, is_success = e.step(action)
# 			total_r += reward
# 		all_rs.append(total_r)
	
# 	r = np.mean(all_rs)
	
# 	return r



if __name__ == '__main__':
	env = gym.make('FetchReach-v1')
	state_dim = env.observation_space['observation'].shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])
	buffer = ReplayBuffer(state_dim, action_dim)
	for t in range(2):
		states, traj = rollout(env, None, t)
		# print(states)
		buffer.add(states, traj)
		if t == 1: 
			buffer.fit()


	buffer.update_densities()