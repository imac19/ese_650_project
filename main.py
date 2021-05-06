import copy
import numpy as np 
import torch 
import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter 


from utils import *
<<<<<<< HEAD
from td3 import *
=======
from td3 import Actor, Critic
>>>>>>> 3b2033d166eee404b26746cb8080866f52d903e3
import argparse
import gym 


class Agent(object): 
    def __init__(self, state_dim, act_dim, max_action,
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
        
        self.device = 'cuda'
        
    def train(self, buffer, batch_size=256): 
        state, action, next_state, reward, done = buffer.sample(batch_size)
        
    def get_best_action(self, state):
        state = torch.FloatTensor(state.reshape(-1,1)).to(self.device)
        action = self.actor(state).detach().cpu().numpy()
        return action
        
    def save(self, file_path='saved', checkpoint=0):
        torch.save(self.actor.state_dict(), file_path + '_actor_checkpoint_{}'.format(checkpoint))
        torch.save(self.critic1.state_dict(), file_path + '_critic1_checkpoint_{}'.format(checkpoint))
        torch.save(self.critic2.state_dict(), file_path + '_critic2_checkpoint_{}'.format(checkpoint))
        
        torch.save(self.actor_target.state_dict(), file_path + '_actor_target_checkpoint_{}'.format(checkpoint))
        torch.save(self.critic1_target.state_dict(), file_path + '_critic1_target_checkpoint_{}'.format(checkpoint))
        torch.save(self.critic2_target.state_dict(), file_path + '_critic2_target_checkpoint_{}'.format(checkpoint))
        
        torch.save(self.actor_optimizer.state_dict(), file_path + '_actor_optimizer_{}'.format(checkpoint))
        torch.save(self.critic_optimizer.state_dict(), file_path + '_critic_optimizer_checkpoint_{}'.format(checkpoint))

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
    parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()
    
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_action = env.action_space.high[0]
    
    agent = Agent(state_dim, act_dim, max_action)
    
    replay_buffer = utils.ReplayBuffer()
    replay_init_size = 1000
    episode_timesteps = 0
    state, done = env.reset(), False
    
    for i in range(replay_init_size):
        episode_timesteps+=1
        random_action = env.action_space.sample()
        next_state, reward, done, is_success = env.step(random_action)
        done = float(done) if episode_timesteps < env.max_episode_steps else 0
        
        replay_buffer.add(state, action, next_state, reward, done)
        state = next_state
        
        if done:
            state, done = env.reset(), False
            episode_timesteps = 0 
            
    training_iterations = 1000
    episode_timesteps = 0
    train_episodes = 0
    train_reward = 0
    train_rewards_plotting = []
    eval_rewards_plotting = []
    eval_every = 10
    save_every = 50
    
    for i in range(training_iterations):
        episode_timesteps += 1
        action = 
        next_state, reward, done, is_success = env.step(action)
        done = float(done) if episode_timesteps < env.max_episode_steps else 0 
        
        replay_buffer.add(state, action, next_state, reward, done)
        state = next_state
        train_reward += reward
        
        agent.train(replay_buffer)
        
        if done:
            print('Training Episode: {}, Training Reward: {}, Training Timesteps: {}'
                  .format(train_episodes, train_reward, episode_timesteps))
            state, done = env.reset(), False
            train_rewards_plotting.append(train_reward)
            episode_timesteps = 0 
            train_episodes += 1
            train_reward = 0
            
        if train_episodes % eval_every == 0:
            eval_reward = utils.evaluate(agent)
            eval_rewards_plotting.append(eval_reward)

        if trian_episodes % save_every == 0:
            agent.save(checkpoint=train_episodes)
