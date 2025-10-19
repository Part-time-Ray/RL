import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
import cv2
import imageio
import random
import time

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

class PPOBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.update_sample_count = int(config["update_sample_count"])
		self.discount_factor_gamma = config["discount_factor_gamma"]
		self.discount_factor_lambda = config["discount_factor_lambda"]
		self.clip_epsilon = config["clip_epsilon"]
		self.max_gradient_norm = config["max_gradient_norm"]
		self.batch_size = int(config["batch_size"])
		self.value_coefficient = config["value_coefficient"]
		self.entropy_coefficient = config["entropy_coefficient"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.video = config["video"]

		self.preprocessor = AtariPreprocessor(frame_stack=4)
		self.gae_replay_buffer = GaeSampleMemory({
			"horizon" : config["horizon"],
			"use_return_as_advantage": False,
			"agent_count": 1,
			})

		self.today = time.strftime("%m-%d-%H-%M-%S", time.localtime())
		logdir = os.path.join(config["logdir"], self.today)
		os.makedirs(logdir, exist_ok=True)
		with open(os.path.join(logdir, "config.cfg"), 'w') as f:
			f.write("{\n")
			for key in config:
				f.write(f"{key}: {config[key]},\n")
			f.write("}\n")
		self.writer = SummaryWriter(logdir)
		self.best_eval_reward = -float('inf')

	@abstractmethod
	def decide_agent_actions(self, observation):
		# add batch dimension in observation
		# get action, value, logp from net

		return NotImplementedError

	@abstractmethod
	def update(self):
		# sample a minibatch of transitions
		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		# calculate the loss and update the behavior network

		return NotImplementedError


	def train(self):
		episode_idx = 0
		while self.total_time_step <= self.training_steps:
			observation, info = self.env.reset()
			state = self.preprocessor.reset(observation)
			episode_reward = 0
			episode_len = 0
			episode_idx += 1
			while True:
				### TODO ###
				# get action from net and get next information from env
				action, logp, value = self.decide_agent_actions(state)
				next_observation, reward, terminate, truncate, info = self.env.step(action)
				next_state = self.preprocessor.step(next_observation)

				# observation must be dict before storing into gae_replay_buffer
				# dimension of reward, value, logp_pi, done must be the same
				obs = {}
				obs["observation_2d"] = np.asarray(state, dtype=np.float32)
				self.gae_replay_buffer.append(0, {
						### TODO ###
						# store the transition into gae_replay_buffer
						"observation": obs,
						"action": np.asarray(action, dtype=np.int32),
						"reward": np.asarray(reward, dtype=np.float32),
						"value": np.asarray(value, dtype=np.float32),
						"logp_pi": np.asarray(logp, dtype=np.float32),
						"done": np.asarray(terminate or truncate, dtype=np.float32),
					})

				if len(self.gae_replay_buffer) >= self.update_sample_count:
					self.update()
					self.gae_replay_buffer.clear_buffer()

				episode_reward += reward
				episode_len += 1
				
				if terminate or truncate:
					self.writer.add_scalar('Train/Episode Reward', episode_reward, self.total_time_step)
					self.writer.add_scalar('Train/Episode Len', episode_len, self.total_time_step)
					print(f"[{len(self.gae_replay_buffer)}/{self.update_sample_count}][{self.total_time_step}/{self.training_steps}]  episode: {episode_idx}  episode reward: {episode_reward}  episode len: {episode_len}")
					break
					
				# observation = next_observation
				state = next_state
				self.total_time_step += 1
				
			if episode_idx % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print(f"Evaluating... ({self.today})")
		all_rewards = []
		frames = []
		for i in range(self.eval_episode):
			observation, info = self.test_env.reset(seed=random.randint(0, 10000))
			state = self.preprocessor.reset(observation)
			total_reward = 0
			while True:
				render = self.test_env.render()
				frames.append(render)
				action, _, _ = self.decide_agent_actions(state, eval=True)
				next_observation, reward, terminate, truncate, info = self.test_env.step(action)
				next_state = self.preprocessor.step(next_observation)
				total_reward += reward
				if terminate or truncate:
					print(f"episode {i+1} reward: {total_reward}")
					all_rewards.append(total_reward)
					break

				# observation = next_observation
				state = next_state
			if self.video:
				video_path = os.path.join("video", self.today)
				os.makedirs(video_path, exist_ok=True)
				with imageio.get_writer(os.path.join(video_path, f"eval_{i+1}.mp4"), fps=30) as video:
					for f in frames:
						height, width = f.shape[:2]
						new_width = (width // 16) * 16
						new_height = (height // 16) * 16
						video.append_data(cv2.resize(f, (new_width, new_height)))
			

		avg = sum(all_rewards) / self.eval_episode
		if avg > self.best_eval_reward:
			self.best_eval_reward = avg
			save_path = os.path.join("saved_model", self.today)
			os.makedirs(save_path, exist_ok=True)
			self.save(os.path.join(save_path, f"model.pth"))

		print(f"average score: {avg}")
		print(f"best eval score: {self.best_eval_reward}")
		print("==============================================")
		return avg

	def inference(self, load_model_path, seed=0):
		self.load(load_model_path)
		print("==============================================")
		print(f"Evaluating...")
		all_rewards = []
		frames = []
		for i in range(5):
			observation, info = self.test_env.reset(seed=seed+i)
			state = self.preprocessor.reset(observation)
			total_reward = 0
			length = 0
			while True:
				render = self.test_env.render()
				frames.append(render)
				action, _, _ = self.decide_agent_actions(state, eval=True)
				next_observation, reward, terminate, truncate, info = self.test_env.step(action)
				next_state = self.preprocessor.step(next_observation)
				total_reward += reward
				length += 1
				if terminate or truncate:
					print(f"Episode {i+1}, Length: {length}, Total reward: {int(total_reward)}")
					all_rewards.append(total_reward)
					break

				# observation = next_observation
				state = next_state
			

		avg = sum(all_rewards) / 5

		print(f"average score: {avg}")
		print("==============================================")
		return avg
	def demo(self, seed=0):
		frames = []
		observation, info = self.test_env.reset(seed=seed)
		state = self.preprocessor.reset(observation)
		total_reward = 0
		while True:
			render = self.test_env.render()
			frames.append(render)
			action, _, _ = self.decide_agent_actions(state, eval=True)
			next_observation, reward, terminate, truncate, info = self.test_env.step(action)
			next_state = self.preprocessor.step(next_observation)
			total_reward += reward
			if terminate or truncate:
				break

			# observation = next_observation
			state = next_state
		if self.video:
			video_path = os.path.join("video", 'demo')
			os.makedirs(video_path, exist_ok=True)
			with imageio.get_writer(os.path.join(video_path, f"demo.mp4"), fps=30) as video:
				for f in frames:
					height, width = f.shape[:2]
					new_width = (width // 16) * 16
					new_height = (height // 16) * 16
					video.append_data(cv2.resize(f, (new_width, new_height)))

		return int(total_reward)
	
	# save model
	def save(self, save_path):
		torch.save(self.net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.net.load_state_dict(torch.load(load_path))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()


	

