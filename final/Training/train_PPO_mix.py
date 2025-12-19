import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv   # 改這裡！超重要
from stable_baselines3.common.callbacks import CheckpointCallback
from Reward_callback import VideoEvalCallback
from wrapper_mix import environment
import torch
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default=None, help='Path to load model from')
    args = parser.parse_args()

    load_model_path = args.load_model
    scenario = 'mix'
    frame_stack = 8
    frame_skip = 4
    folder_name = 'ppo'
    # frame_skip = 2
    # folder_name = 'ppo_2'
    def make_env():
        def f():
            env = environment(frame_stack=frame_stack, frame_skip=frame_skip)
            return env
        return f

    num_cpu = 30
    vec_env = SubprocVecEnv([make_env() for i in range(num_cpu)])   # 24~32 隨你機器，越多越快

    os.makedirs(f'./{scenario}/checkpoints/{folder_name}/', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
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
            n_steps=2048,              # 4096 比 2048 更穩定（rollout 更長，優勢估計更好）
            batch_size=256,           # 越大越穩（我試過 2048 也行，但吃更多顯存）
            n_epochs=10,                # 8~10 是目前 austria 最好的（4 太少容易過擬合）
            gamma=0.99,                # 改高！讓它更重視進度而不是只衝速度
            clip_range=0.2,
            learning_rate=3e-4,        # 2e-4 是甜蜜點（3e-4 會抖，1e-4 太慢）
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