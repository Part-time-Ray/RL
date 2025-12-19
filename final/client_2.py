import argparse
import json
import numpy as np
import requests


def connect(agent, url: str = 'http://localhost:5000'):
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.act(obs)  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return
from stable_baselines3 import PPO
from Training.wrapper import FrameStackObs
from Training.wrapper import environment
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    args = parser.parse_args()

    class PPOAgent:
        def __init__(self, frame_stack, frame_skip, model_path_1, model_path_2):
            self.frame_stack = frame_stack
            self.frame_skip = frame_skip
            self.model_1 = PPO.load(model_path_1)
            self.model_2 = PPO.load(model_path_2)
            self.obs = [FrameStackObs(num_frames=frame_stack) for _ in range(frame_skip)]
            self.index = 0
            self.length = 0

            
        def act(self, observation):
            self.length += 1
            if len(self.obs[self.index].frames) == 0:
                self.obs[self.index].reset(observation)
            else:
                self.obs[self.index].add_frame(observation)
            action_1, _ = self.model_1.predict(self.obs[self.index].get_stack(), deterministic=True)
            action_2, _ = self.model_2.predict(self.obs[self.index].get_stack(), deterministic=True)
            weight_1 = 1 - min(self.length / 500, 1)
            weight_2 = min(self.length / 500, 1)
            action = weight_1 * action_1 + weight_2 * action_2
            self.index = (self.index + 1) % self.frame_skip
            # action = np.clip(action + np.random.normal([0.0, 0.0], [0.1, 0.05], size=action.shape), -1.0, 1.0)
            return action
    ppo_agent = PPOAgent(frame_stack=8, frame_skip=4, model_path_1='./aa.zip', model_path_2='./bb.zip')
    connect(ppo_agent, url=args.url)

    # Initialize the RL Agent
    # import gymnasium as gym

    # rand_agent = RandomAgent(
    #     action_space=gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32))

    # connect(rand_agent, url=args.url)
