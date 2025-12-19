import math
import gymnasium as gym
import numpy as np
from racecar_gym.env import RaceEnv
from collections import deque
from gymnasium.spaces import Dict, Box
import cv2
import random

class FrameStackObs:
    def __init__(self, num_frames=12, flip=False):
        self.frames = deque(maxlen=num_frames)
        self.flip = flip
    def preprocess(self, obs):
        r = obs[0,:,:]
        g = obs[1,:,:]
        b = obs[2,:,:]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)  # downsampling
        if self.flip:
            gray = cv2.flip(gray, 1)
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
    def __init__(self, random_op=False, scenario='austria_competition', frame_stack=12, frame_skip=1, flip=False, reward_shaping=True):
        super().__init__()
        assert scenario in ['austria_competition', 'circle_cw_competition_collisionStop'], "Invalid scenario"
        self.scenario = scenario
        self.flip = flip
        self.env = RaceEnv(
            scenario=self.scenario,
            render_mode='rgb_array_birds_eye',
            # reset_when_collision=(self.scenario=='austria_competition')
            reset_when_collision='austria' in self.scenario
            # reset_when_collision=False
            # Only work for 'austria_competition' and 'austria_competition_collisionStop'
        )
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
        # self.observation_space = Dict({
        #     'image': Box(low=0, high=255, shape=(frame_stack, 84, 84), dtype=np.uint8),
        #     'pose':   Box(low=-100.0, high=100.0, shape=(6,), dtype=np.float32),
        #     'acceleration': Box(low=-100.0, high=100.0, shape=(6,), dtype=np.float32),
        #     'velocity': Box(low=-100.0, high=100.0, shape=(6,), dtype=np.float32),
        # })
        self.obs = FrameStackObs(num_frames=frame_stack, flip=flip)
        self.color_obs = None
        self.prev_info = None
        self.prev_reward = 0.0
        self.random_op = random_op
        self.frame_skip = frame_skip
        self.reward_shaping = reward_shaping


    def reset(self, seed=None, **options):
        kwargs = dict()
        if self.random_op:
            kwargs['options'] = {'mode': 'random'}
            obs, info = self.env.reset(seed=seed, **kwargs)
        else:
            obs, info = self.env.reset(seed=seed)
        self.obs.reset(obs)
        self.color_obs = obs
        self.prev_info = None
        self.prev_reward = 0.0
        return self.obs.get_stack(), info
        # return {'image': self.obs.get_stack(), 'pose': info['pose'], 'acceleration': info['acceleration'], 'velocity': info['velocity']}, info

    def step(self, action):
        if self.flip:
            action = np.array([action[0], -action[1]])
        motor_action, steering_action = action
        # add some noise to the action average [0, 0] std [0.005, 0.01]
        if random.randint(0, 100):
            action = np.clip(action + np.random.normal(0, [0.002, 0.02], size=2), -1.0, 1.0)
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        self.obs.add_frame(obs)
        self.color_obs = obs
        self.prev_reward = reward
        if not self.reward_shaping:
            return self.obs.get_stack(), reward, terminated, truncated, info
        if 'austria' in self.scenario:
            if self.prev_info == None:
                self.prev_info = info
            if info['progress'] > self.prev_info['progress']:
                reward += 500 * (info['progress'] - self.prev_info['progress']) 
            else:
                reward -= 0.2
            if info['checkpoint'] > self.prev_info['checkpoint']:
                reward += 10
            if info['lap'] > self.prev_info['lap']:
                reward += 100
            if info['wall_collision']:
                reward -= 20
        else:
            if info['wall_collision']:
                reward -= 10 * motor_action
                terminated = True
        # reward -= 0.1
        # reward /= 10
        self.prev_info = info.copy()
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