import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from envs_base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
from models.dueling_atari_model import AtariNetDQN as DuelingAtariNetDQN
import gymnasium as gym
import random
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import ale_py
gym.register_envs(ale_py)

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env_name = 'ALE/MsPacman-v5' if self.game == 'Pacman' else 'ALE/Enduro-v5'
		self.num_envs = config["num_envs"]
		self.env = AsyncVectorEnv([lambda: gym.make(self.env_name) for _ in range(self.num_envs)])
		self.parallel_gpu = config["parallel_gpu"]
		### TODO ###
		# initialize test_env
		self.test_env = gym.make(self.env_name, render_mode="rgb_array")

		# initialize behavior network and target network
		if self.network == "Dueling":
			self.behavior_net = DuelingAtariNetDQN(num_classes=self.env.single_action_space.n, channel=self.frame_stack)
			self.target_net = DuelingAtariNetDQN(num_classes=self.env.single_action_space.n, channel=self.frame_stack)
		else:
			self.behavior_net = AtariNetDQN(num_classes=self.env.single_action_space.n, channel=self.frame_stack)
			self.target_net = AtariNetDQN(num_classes=self.env.single_action_space.n, channel=self.frame_stack)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		if self.parallel_gpu and torch.cuda.device_count() > 1:
			print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
			self.behavior_net = nn.DataParallel(self.behavior_net)
			self.target_net = nn.DataParallel(self.target_net)
		self.target_net.to(self.device)
		self.behavior_net.to(self.device)
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		# print(observation.shape)
		observation = np.array(observation)
		if random.random() < epsilon:
			action = action_space.sample()
			if len(action.shape) == 0:
				action = np.array([action])
		else:
			if len(observation.shape) == 3:
				observation = np.expand_dims(observation, axis=0)
			observation = torch.tensor(observation,device=self.device)
			with torch.no_grad():
				action = self.behavior_net(observation).argmax(1).long().cpu()
				# change into scalar
				action = np.array([int(a) for a in action])

		# return action

		return action

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		if self.network == "DQN":
			# DQN
			q_value = self.behavior_net(state).gather(1, action.long())
			with torch.no_grad():
				q_next = self.target_net(next_state)
				q_target = reward + self.gamma * q_next.max(1)[0].unsqueeze(1) * (1 - done)
		elif self.network == "DDQN" or self.network == "Dueling":
			# DDQN
			q_value = self.behavior_net(state).gather(1, action.long())
			with torch.no_grad():
				best_action = self.behavior_net(next_state).argmax(dim=-1)
				q_target = reward + self.gamma * self.target_net(next_state).gather(1, best_action.unsqueeze(1)) * (1-done)
		else:
			raise Exception("network error!!!")

		criterion = nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar(f'{self.network}/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
	