import torch
import torch.nn as nn
import numpy as np
from base_agent import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.CarRacingEnv import CarRacingEnvironment
import random
from base_agent import OUNoiseGenerator, GaussianNoise

class CarRacingTD3Agent(TD3BaseAgent):
	def __init__(self, config):
		super(CarRacingTD3Agent, self).__init__(config)
		# initialize environment
		self.env = CarRacingEnvironment(N_frame=4, test=False, self_reward_function=self.self_reward_function, self_reward_function2=self.self_reward_function2)
		self.test_env = CarRacingEnvironment(N_frame=4, test=True, self_reward_function=self.self_reward_function, self_reward_function2=self.self_reward_function2)

		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		if not self.single_critic:
			self.critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.actor_net.to(self.device)
		self.critic_net1.to(self.device)
		if not self.single_critic:
			self.critic_net2.to(self.device)
		# target network
		self.target_actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		if not self.single_critic:
			self.target_critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_actor_net.to(self.device)
		self.target_critic_net1.to(self.device)
		if not self.single_critic:
			self.target_critic_net2.to(self.device)
		self.target_actor_net.load_state_dict(self.actor_net.state_dict())
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		if not self.single_critic:
			self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())

		# load model
		if len(self.load_model) > 0:
			print("Loading model from:", self.load_model)
			self.load(self.load_model)
		
		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		
		self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
		if not self.single_critic:
			self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)

		self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.actor_opt,mode='max', factor=0.5, patience=30, min_lr=1e-6)
		self.critic_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.critic_opt1,mode='max', factor=0.5, patience=30, min_lr=1e-6)
		if not self.single_critic:
			self.critic_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.critic_opt2,mode='max', factor=0.5, patience=30, min_lr=1e-6)

		# choose Gaussian noise or OU noise

		noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
		# noise_std = np.full(self.env.action_space.shape[0], 1.0, np.float32)
		noise_std = np.full(self.env.action_space.shape[0], 0.5, np.float32)
		self.noise = OUNoiseGenerator(noise_mean, noise_std)

		# self.noise = GaussianNoise(self.env.action_space.shape[0], 0.0, 1.0)
		
	
	def decide_agent_actions(self, state, sigma=0.0, brake_rate=0.015):
		### TODO ###
		# based on the behavior (actor) network and exploration noise
		# with torch.no_grad():
		# 	state = ???
		# 	action = actor_net(state) + sigma * noise

		# actor shape (batch_size, action_dim)
		# print("state shape:", state.shape)
		if len(state.shape) == 3:
			state = np.expand_dims(state, axis=0)
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device)
			action = self.actor_net(state, brake_rate)
			noise = sigma * self.noise.generate()
			noise = torch.FloatTensor(noise).to(self.device)
			action = action + noise
			action = action.cpu().numpy()
			action = action.clip(self.env.action_space.low, self.env.action_space.high)
		return action

		# return NotImplementedError
		

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		### TD3 ###
		# 1. Clipped Double Q-Learning for Actor-Critic
		# 2. Delayed Policy Updates
		# 3. Target Policy Smoothing Regularization

		## Update Critic ##
		# critic loss
		# q_value1 = ???
		# q_value2 = ???
		# with torch.no_grad():
		# 	# select action a_next from target actor network and add noise for smoothing
		# 	a_next = ??? + noise
		q_value1 = self.critic_net1(state, action)
		if not self.single_critic:
			q_value2 = self.critic_net2(state, action)
		with torch.no_grad():
			a_next = self.target_actor_net(next_state)
			if not self.disable_smoothing:
				noise = (torch.randn_like(a_next) * 0.2).clamp(-0.5, 0.5)
				a_next = a_next + noise
			low = torch.tensor(self.env.action_space.low).to(self.device)
			high = torch.tensor(self.env.action_space.high).to(self.device)
			a_next = torch.clamp(a_next, low, high)

			q_next1 = self.target_critic_net1(next_state, a_next)
			if not self.single_critic:
				q_next2 = self.target_critic_net2(next_state, a_next)
				q_target = reward + (1 - done) * self.gamma * torch.min(q_next1, q_next2)
			else:
				q_target = reward + (1 - done) * self.gamma * q_next1

		# 	q_next1 = ???
		# 	q_next2 = ???
		# 	# select min q value from q_next1 and q_next2 (double Q learning)
		# 	q_target = ???
		
		# critic loss function
		criterion = nn.MSELoss()
		critic_loss1 = criterion(q_value1, q_target)
		if not self.single_critic:
			critic_loss2 = criterion(q_value2, q_target)

		# optimize critic
		self.critic_net1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		if not self.single_critic:
			self.critic_net2.zero_grad()
			critic_loss2.backward()
			self.critic_opt2.step()

		## Delayed Actor(Policy) Updates ##
		# if self.total_time_step % self.update_freq == 0:
		# 	## update actor ##
		# 	# actor loss
		# 	# select action a from behavior actor network (a is different from sample transition's action)
		# 	# get Q from behavior critic network, mean Q value -> objective function
		# 	# maximize (objective function) = minimize -1 * (objective function)
		# 	action = ???
		# 	actor_loss = -1 * (???)
		# 	# optimize actor
		# 	self.actor_net.zero_grad()
		# 	actor_loss.backward()
		# 	self.actor_opt.step()
		if self.total_time_step % self.update_freq == 0:
			action_pred = self.actor_net(state)
			actor_loss = -1 * self.critic_net1(state, action_pred).mean()
			# optimize actor
			self.actor_net.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()
		
