import math
import gymnasium as gym
import numpy as np
from racecar_gym.env import RaceEnv
from collections import deque
from gymnasium.spaces import Dict, Box
import cv2
import random

class FrameStackObs:
    def __init__(self, num_frames=12):
        # deque to hold frames
        self.frames = deque(maxlen=num_frames)
    def preprocess(self, obs):
        # Convert to grayscale
        r = obs[0,:,:]
        g = obs[1,:,:]
        b = obs[2,:,:]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)  # downsampling
        return gray

    def reset(self, frame):
        assert frame.shape == (3, 128, 128), f"Frame shape must be (3, 128, 128), got {frame.shape}"
        processed_frame = self.preprocess(frame)
        for _ in range(self.frames.maxlen):
            self.frames.append(processed_frame)
    def add_frame(self, frame):
        assert frame.shape == (3, 128, 128), f"Frame shape must be (3, 128, 128), got {frame.shape}"
        frame = self.preprocess(frame)
        self.frames.append(frame)
    def get_stack(self):
        return np.array(self.frames)

class environment(gym.Env):
    def __init__(self, frame_stack=12, frame_skip=1):
        super().__init__()
        self.scenarios = ['austria_competition', 'circle_cw_competition_collisionStop']
        self.action_space = Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = Box(
            low=0, high=255,
            shape=(frame_stack, 84, 84),
            dtype=np.uint8
        )
        self.obs = FrameStackObs(num_frames=frame_stack)
        self.color_obs = None
        self.prev_info = None
        self.prev_reward = 0.0
        self.frame_skip = frame_skip
        self.circle_env = RaceEnv(
            scenario=self.scenarios[1],
            render_mode='rgb_array_birds_eye',
            reset_when_collision=False,
        )
        self.austria_env = RaceEnv(
            scenario=self.scenarios[0],
            render_mode='rgb_array_birds_eye',
            reset_when_collision=True
        )



    def reset(self, seed=None, **options):
        if random.randint(0, 1) == 0:
            self.env = self.austria_env
        else:
            self.env = self.circle_env

        kwargs = dict()
        kwargs['options'] = {'mode': 'random'}
        obs, info = self.env.reset(seed=seed, **kwargs)
    
        self.obs.reset(obs)
        self.color_obs = obs
        self.prev_info = None
        self.prev_reward = 0.0
        return self.obs.get_stack(), info
        # return {'image': self.obs.get_stack(), 'pose': info['pose'], 'acceleration': info['acceleration'], 'velocity': info['velocity']}, info

    def step(self, action):
        motor_action, steering_action = action
        # add some noise to the action average [0, 0] std [0.005, 0.01]
        if random.randint(0, 100):
            action = np.clip(action + np.random.normal(0, [0.003, 0.03], size=2), -1.0, 1.0)
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        self.obs.add_frame(obs)
        self.color_obs = obs
        self.prev_reward = reward

        reward += motor_action * 0.1

        if self.prev_info == None:
            self.prev_info = info
            self.prev_info['motor'] = 1
            self.prev_info['steering'] = 0
        if info['progress'] > self.prev_info['progress']:
            # reward += 2000 * (info['progress'] - self.prev_info['progress']) 
            reward += 2000 * (info['progress'] - self.prev_info['progress'])
        if info['checkpoint'] > self.prev_info['checkpoint']:
            reward += 10
            # reward += 5
        if info['lap'] > self.prev_info['lap']:
            reward += 100
        # if info['wall_collision'] or info['wrong_way']:
        if info['wall_collision']:
            reward -= 10
        reward -= 0.2
        # reward -= 0.1
        # print(f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        return self.obs.get_stack(), reward, terminated, truncated, info
        # return {'image': self.obs.get_stack(), 'pose': info['pose'], 'acceleration': info['acceleration'], 'velocity': info['velocity']}, reward, terminated, truncated, info

    def get_obs(self):
        return {'image': self.obs.get_stack(), 'pose': self.prev_info['pose'], 'acceleration': self.prev_info['acceleration'], 'velocity': self.prev_info['velocity']}

    def get_color_obs(self):
        return self.color_obs
    
    def get_reward(self):
        return self.prev_reward

    def render(self):
        render = self.env.render()
        return render

    def close(self):
        self.env.close()