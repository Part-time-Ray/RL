import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv   # 改這裡！超重要
from stable_baselines3.common.callbacks import CheckpointCallback
from Reward_callback import VideoEvalCallback
from wrapper import environment
import torch
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='circle_cw_competition_collisionStop', help='Scenario to train on')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load model from')
    parser.add_argument('--random_op', type=bool, default=True, help='Whether to use random operation in environment')
    parser.add_argument('--flip', type=bool, default=False, help='Whether to flip the environment')
    parser.add_argument('--reward_shaping', type=bool, default=True, help='Whether to use reward shaping')
    args = parser.parse_args()

    load_model_path = args.load_model
    scenario = args.scenario
    random_op = args.random_op
    frame_stack = 1
    frame_skip = 1
    # folder_name = 'ppo'
    folder_name = 'ppo_tensorboard__without_all'
    # frame_skip = 2
    # folder_name = 'ppo_2'
    def make_env():
        def f():
            env = environment(random_op=random_op, scenario=scenario, frame_stack=frame_stack, frame_skip=frame_skip, flip=args.flip, reward_shaping=args.reward_shaping)
            # circle_cw_competition_collisionStop, austria_competition
            return env
        return f

    # 強烈建議改用 SubprocVecEnv，PPO 也能快 2~3 倍（尤其在 32 核機器上）
    num_cpu = 60
    vec_env = SubprocVecEnv([make_env() for i in range(num_cpu)])   # 24~32 隨你機器，越多越快

    os.makedirs(f'./{scenario}/checkpoints/{folder_name}/', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f'./{scenario}/checkpoints/{folder_name}/',
        name_prefix=f'{folder_name}_austria_best'
    )
    os.makedirs(f"./{scenario}/training_eval_videos/{folder_name}/", exist_ok=True)
    video_eval_callback = VideoEvalCallback(
        eval_freq=300000//num_cpu,                        # 每 50 萬步評測一次
        video_folder=f"./{scenario}/training_eval_videos/{folder_name}/",
        scenario=scenario,
        algorithm='ppo',
        frame_stack=frame_stack,
        frame_skip=frame_skip,
        flip=args.flip,
    )

    tensorboard_log_dir = f"./{scenario}/tensorboard_logs/"
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    if load_model_path is not None and os.path.isfile(load_model_path):
        print(f"Loading model from {load_model_path}")
        model = PPO.load(load_model_path, env=vec_env, device="cuda")
        reset_timesteps = False
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            learning_rate=3e-4,
            clip_range=0.2,
            verbose=1,
            use_sde=True,
            ent_coef=0.01,
            tensorboard_log=tensorboard_log_dir,
            device="cuda"
        )
        reset_timesteps = True


    model.learn(
        total_timesteps=1e7,
        callback=[checkpoint_callback, video_eval_callback],
        tb_log_name=f"ppo",
        progress_bar=True,
        reset_num_timesteps=reset_timesteps,
    )

    model.save(f"ppo_{scenario}_final")