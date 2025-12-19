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
        def __init__(self, frame_stack, frame_skip, model_path):
            self.frame_stack = frame_stack
            self.frame_skip = frame_skip
            self.model = PPO.load(model_path)
            self.obs = FrameStackObs(num_frames=frame_stack)
            self.length = -1
            self.action = None

            
        def act(self, observation):
            self.length += 1
            if self.length % self.frame_skip:
                return self.action
            if len(self.obs.frames) == 0:
                self.obs.reset(observation)
            else:
                self.obs.add_frame(observation)
            action, _ = self.model.predict(self.obs.get_stack(), deterministic=True)
            self.action = action
            action = np.clip(action + np.random.normal([0.0, 0.0], [0.005, 0.05], size=action.shape), -1.0, 1.0)
            return action
        
    # model_path = '/mnt/nfs/work/ray/RL/final/Training/austria_competition/checkpoints/ppo_continue_3/ppo_continue_3_austria_best_15816000_steps.zip'
    model_path = '/mnt/nfs/work/ray/RL/final/Training/austria_competition/checkpoints/ppo_continue_3/ppo_continue_3_austria_best_15756000_steps.zip'
    ppo_agent = PPOAgent(frame_stack=8, frame_skip=4, model_path=model_path)
    connect(ppo_agent, url=args.url)

    # Initialize the RL Agent
    # import gymnasium as gym

    # rand_agent = RandomAgent(
    #     action_space=gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32))

    # connect(rand_agent, url=args.url)
