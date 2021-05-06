import copy
import numpy as np 
import torch 
import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter 


from utils import *
from td3 import Actor, Critic 
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
        

        self.writer = SummaryWriter()
        self.iters = 0 
        
    def train(self, buffer, batch_size=256): 
        state, action, next_state, reward, done = buffer.sample(batch_size)
        
        with torch.no_grad(): 
            # compute target actions 
            noise = torch.clip(torch.randn_like(actions)*self.policy_noise, -self.noise_clip, self.noise_clip)
            next_action = torch.clip(self.actor_target(next_state), -self.max_action, self.max_action)

            # compute target q 
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = reward + self.gamma*(1-done)*torch.min(target_q1, target_q2)

        # current q value 
        q1 = self.critic(state, action)
        q2 = self.critic(state, action)


        loss = torch.mean((q1-target_q1)** 2 + (q2 - target_q2)**2)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.writer.add_scalar('Loss/train', loss, self.iters)
        self.critic_optimizer.step()

        # delay target policy update 
        if self.iters % self.freq == 0: 

            loss2 = -self.critic(state, self.actor(state)).mean()
            loss2.backward()


            self.update_target_weights()

    def get_best_action(self, state):
        state = torch.FloatTensor(state.reshape(-1,1)).to(self.device)
        action = self.actor(state).detach().cpu().numpy()
        return action

    def update_target_weights(self): 
        critic1_params = self.critic1.parameters()
        critic1_target_params = self.critic1_target.parameters()
        for p, tp in zip(critic1_params, critic1_target_params): 
            tp.copy_(self.tau*p.data + (1-self.tau)*tp.data)

        critic2_params = self.critic2.parameters()
        critic2_target_params = self.critic2_target.parameters()
        for p, tp in zip(critic2_params, critic2_target_params): 
            tp.copy_(self.tau*p.data + (1-self.tau)*tp.data)


        actor_params = self.actor.parameters()
        actor_target_params = self.actor_target.parameters()
        for p, tp in zip(actor_params, actor_target_params): 
            tp.copy_(self.tau*p.data + (1-self.tau)*tp.data)

        
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
    
    if args.mode == 'train': 
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
                
        training_iterations = 100000
        episode_timesteps = 0
        train_episodes = 0
        train_reward = 0
        # train_rewards_plotting = []
        # eval_rewards_plotting = []
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
                state = env.reset()
                # train_rewards_plotting.append(train_reward)
                episode_timesteps = 0 
                train_episodes += 1
                train_reward = 0
                
            if train_episodes % eval_every == 0:
                eval_reward = utils.evaluate(agent)
                # eval_rewards_plotting.append(eval_reward)

            if train_episodes % save_every == 0:
                agent.save(checkpoint=train_episodes)
