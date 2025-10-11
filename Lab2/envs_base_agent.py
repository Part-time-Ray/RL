import torch
import torch.nn as nn
import numpy as np
import shutil, os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
import imageio
import cv2

class AtariPreprocessor:
	"""
	Preprocesing the state input of DQN for Atari
	"""    
	def __init__(self, frame_stack=4):
		self.frame_stack = frame_stack
		self.frames = deque(maxlen=frame_stack)

	def preprocess(self, obs):
		gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
		resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
		return resized / 255.
		# return resized

	def reset(self, obs):
		frame = self.preprocess(obs)
		self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
		return np.stack(self.frames, axis=0)

	def step(self, obs):
		frame = self.preprocess(obs)
		self.frames.append(frame)
		return np.stack(self.frames, axis=0)

class DQNBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.epsilon = 1.0
		self.eps_min = config["eps_min"]
		self.eps_decay = config["eps_decay"]
		self.eval_epsilon = config["eval_epsilon"]
		self.warmup_steps = config["warmup_steps"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.gamma = config["gamma"]
		self.update_freq = config["update_freq"]
		self.update_target_freq = config["update_target_freq"]
		self.frame_stack = config["frame_stack"]
		self.video = config["video"]
		self.network = config["network"]
		self.game = config['game']
	
		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
		self.preprocessor = [AtariPreprocessor(frame_stack=self.frame_stack) for _ in range(config["num_envs"])]
		self.eval_preprocessor = AtariPreprocessor(frame_stack=self.frame_stack)
		self.logdir = os.path.join(config["logdir"], self.game, "envs_" +self.network)
		if os.path.exists(self.logdir):
			shutil.rmtree(self.logdir, ignore_errors=True)
		os.makedirs(self.logdir, exist_ok=True)
		self.writer = SummaryWriter(self.logdir)
		self.best_eval_reward = -1

	@abstractmethod
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		action = None
		return action
	
	def update(self):
		# print(self.total_time_step, self.update_freq, self.update_target_freq)
		if self.total_time_step % self.update_freq == 0:
			self.update_behavior_network()
		if self.total_time_step % self.update_target_freq == 0:
			# print("Update target network")
			self.update_target_network()

	@abstractmethod
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		# calculate the loss and update the behavior network
		

	def update_target_network(self):
		self.target_net.load_state_dict(self.behavior_net.state_dict())
	
	def epsilon_decay(self):
		self.epsilon -= (1 - self.eps_min) / self.eps_decay
		self.epsilon = max(self.epsilon, self.eps_min)

	def train(self):
		episode_idx = 0
		while self.total_time_step <= self.training_steps:
			observation, info = self.env.reset()
			state = [self.preprocessor[i].reset(observation[i]) for i in range(self.num_envs)]
			episode_reward = [0 for _ in range(self.num_envs)]
			episode_len = [0 for _ in range(self.num_envs)]
			done = [False for _ in range(self.num_envs)]
			episode_idx += 1
			while True:
				if self.total_time_step < self.warmup_steps:
					action = self.decide_agent_actions(state, 1.0, self.env.action_space)
				else:
					action = self.decide_agent_actions(state, self.epsilon, self.env.action_space)
					self.epsilon_decay()
				next_observation, reward, terminate, truncate, info = self.env.step(action)
				next_state = [self.preprocessor[i].step(next_observation[i]) for i in range(self.num_envs)]

				# self.replay_buffer.append(state, action, reward, next_state, terminate.astype(int))
				for i in range(self.num_envs):
					self.replay_buffer.append(state[i], [action[i]], [reward[i]], next_state[i], [int(terminate[i])])
				# self.replay_buffer.append(state, [action], [reward], next_state, [int(terminate)])

				if self.total_time_step >= self.warmup_steps:
					self.update()

				episode_reward = [episode_reward[i] + reward[i] for i in range(self.num_envs)]
				episode_len = [episode_len[i] + 1 for i in range(self.num_envs)]
				done = [done[i] or terminate[i] or truncate[i] for i in range(self.num_envs)]
				if all(done):
					self.writer.add_scalar('Train/Episode Reward', int(np.mean(episode_reward)), self.total_time_step)
					self.writer.add_scalar('Train/Episode Len', int(np.mean(episode_len)), self.total_time_step)
					print(f"[{self.total_time_step}/{self.training_steps} ({self.total_time_step/self.training_steps*100:.1f}%)]  episode: {episode_idx}  episode reward: {int(np.mean(episode_reward))}  episode len: {int(np.mean(episode_len))}  epsilon: {self.epsilon}  learning_rate: {self.lr}")
					break
				# observation = next_observation
				state = next_state
				self.total_time_step += 1

			if episode_idx % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				# self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for i in range(self.eval_episode):
			frames = []
			observation, info = self.test_env.reset()
			state = self.eval_preprocessor.reset(observation)
			total_reward = 0
			while True:
				frame = self.test_env.render()
				frames.append(frame)
				action = self.decide_agent_actions(state, self.eval_epsilon, self.test_env.action_space)[0]

				next_observation, reward, terminate, truncate, info = self.test_env.step(action)
				next_state = self.eval_preprocessor.step(next_observation)
				total_reward += reward
				if terminate or truncate:
					print(f"episode {i+1} reward: {int(total_reward)}")
					all_rewards.append(total_reward)
					break

				state = next_state
				# observation = next_observation
			if self.video:
				video_path = os.path.join("video", f"{self.game}", f"envs_{self.network}")
				os.makedirs(video_path, exist_ok=True)
				with imageio.get_writer(os.path.join(video_path, f"eval_{i+1}.mp4"), fps=30) as video:
					for f in frames:
						height, width = f.shape[:2]
						new_width = (width // 16) * 16
						new_height = (height // 16) * 16
						video.append_data(cv2.resize(f, (new_width, new_height)))

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		if avg > self.best_eval_reward:
			self.best_eval_reward = avg
			save_path = os.path.join("saved_model", f"{self.game}")
			os.makedirs(save_path, exist_ok=True)
			self.save(os.path.join(save_path, f"envs_{self.network}_model.pth"))
		return avg
	
	# save model
	def save(self, save_path):
		if self.parallel_gpu and torch.cuda.device_count() > 1:
			torch.save(self.behavior_net.module.state_dict(), save_path)
		else:
			torch.save(self.behavior_net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.behavior_net.load_state_dict(torch.load(load_path))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()




