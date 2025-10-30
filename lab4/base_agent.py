import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
import cv2
import imageio


class GaussianNoise:
	def __init__(self, dim, mu=None, std=None):
		self.mu = mu if mu else np.zeros(dim)
		self.std = np.ones(dim) * std if std else np.ones(dim) * .1
	
	def reset(self):
		pass

	def generate(self):
		return np.random.normal(self.mu, self.std)

class OUNoiseGenerator:
    def __init__(self, mean, std_dev, theta=0.3, dt=5e-2):
        self.theta = theta
        self.dt = dt
        self.mean = mean
        self.std_dev = std_dev

        self.x = None

        self.reset()

    def reset(self):
        self.x = np.zeros_like(self.mean.shape)

    def generate(self):
        self.x = (self.x
                  + self.theta * (self.mean - self.x) * self.dt
                  + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))

        return self.x

class TD3BaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.warmup_steps = config["warmup_steps"]
		self.total_episode = config["total_episode"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.gamma = config["gamma"]
		self.tau = config["tau"]
		self.update_freq = config["update_freq"]
		self.video = config["video"]
		self.seed = config["seed"]
		self.single_critic = config["single_critic"]
		self.disable_smoothing = config["disable_smoothing"]
		self.big_action_noise = config["big_action_noise"]
		self.self_reward_function = config["self_reward_function"]
		self.self_reward_function2 = config["self_reward_function2"]
		self.ns = config["ns"] if len(config["ns"]) > 0 else time.strftime("%Y%m%d-%H%M%S")
		self.load_model = config["load_model"]

		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
		self.dir = os.path.join("store", self.ns)
		os.makedirs(os.path.join(self.dir, "log"), exist_ok=True)
		self.writer = SummaryWriter(os.path.join(self.dir, "log"))

		self.best_score = -float('inf')

	@abstractmethod
	def decide_agent_actions(self, state, sigma=0.0):
		### TODO ###
		# based on the behavior (actor) network and exploration noise
		
		return NotImplementedError
		
	
	def update(self):
		# update the behavior networks
		self.update_behavior_network()
		# update the target networks
		if self.total_time_step % self.update_freq == 0:
			self.update_target_network(self.target_actor_net, self.actor_net, self.tau)
			self.update_target_network(self.target_critic_net1, self.critic_net1, self.tau)
			if not self.single_critic:
				self.update_target_network(self.target_critic_net2, self.critic_net2, self.tau)

	@abstractmethod
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		
		### TODO ###
		# calculate the loss and update the behavior network

		return NotImplementedError
		

	@staticmethod
	def update_target_network(target_net, net, tau):
		# update target network by "soft" copying from behavior network
		for target, behavior in zip(target_net.parameters(), net.parameters()):
			target.data.copy_((1 - tau) * target.data + tau * behavior.data)
	
	def train(self):
		for episode in range(self.total_episode):
			total_reward = 0
			state, infos = self.env.reset() if len(self.seed) == 0 else self.env.reset(seed=self.seed[episode % len(self.seed)])
			self.noise.reset()
			for t in range(10000):
				if self.total_time_step < self.warmup_steps:
					action = self.env.action_space.sample()
				else:
					# exploration degree
					if not self.disable_smoothing:
						sigma = max(0.1*(1-episode/self.total_episode), 0.01)
						if self.big_action_noise:
							sigma *= 2.0
					else:
						sigma = 0.0
					action = self.decide_agent_actions(state, sigma=sigma)[0]
				
				next_state, reward, terminates, truncates, _ = self.env.step(action)
				self.replay_buffer.append(state, action, [reward/10], next_state, [int(terminates)])
				if self.total_time_step >= self.warmup_steps:
					self.update()
				self.total_time_step += 1
				total_reward += reward
				state = next_state
				if terminates or truncates:
					self.writer.add_scalar('Train/Episode Reward', total_reward, self.total_time_step)
					print(
						'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
						.format(self.total_time_step, episode+1, t, total_reward))
				
					break
			
			if (episode+1) % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for episode in range(self.eval_episode):
			frames = []
			total_reward = 0
			state, infos = self.test_env.reset() if len(self.seed) == 0 else self.test_env.reset(seed=self.seed[episode % len(self.seed)])
			for t in range(10000):
				frame = self.test_env.render()
				frames.append(frame)
				action = self.decide_agent_actions(state)[0]
				next_state, reward, terminates, truncates, _ = self.test_env.step(action)
				total_reward += reward
				state = next_state
				if terminates or truncates:
					print((f'Seed: {self.seed[episode % len(self.seed)]}\t' if len(self.seed) else '') +
						f'Episode: {episode+1}\tLength: {t:3d}\tTotal reward: {total_reward:.2f}')
					all_rewards.append(total_reward)
					break
			if self.video:
				video_path = os.path.join(self.dir, "video")
				os.makedirs(video_path, exist_ok=True)
				with imageio.get_writer(os.path.join(video_path, f"eval_{episode+1}.mp4"), fps=30, macro_block_size=1) as video:
					for f in frames:
						video.append_data(f)

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")

		if avg > self.best_score:
			self.best_score = avg
			print(f"New best score: {self.best_score}, saving model...")
			self.save(os.path.join(self.dir, "best_model.pth"))
		return avg
	
	# save model
	def save(self, save_path):
		if self.single_critic:
			torch.save(
				{
					'actor': self.actor_net.state_dict(),
					'critic1': self.critic_net1.state_dict(),
				}, save_path)
		else:
			torch.save(
				{
					'actor': self.actor_net.state_dict(),
					'critic1': self.critic_net1.state_dict(),
					'critic2': self.critic_net2.state_dict(),
				}, save_path)

	# load model
	def load(self, load_path):
		checkpoint = torch.load(load_path)
		self.actor_net.load_state_dict(checkpoint['actor'])
		self.critic_net1.load_state_dict(checkpoint['critic1'])
		if not self.single_critic:
			self.critic_net2.load_state_dict(checkpoint['critic2'])

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()

