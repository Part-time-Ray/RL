import os
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from Reward_callback import VideoEvalCallback
import numpy as np
from wrapper import environment
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='circle_cw_competition_collisionStop', help='Scenario to train on')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load model from')
    args = parser.parse_args()
    
    scenario = args.scenario
    load_model_path = args.load_model
    def make_env():
        def f():
            env = environment(random_op=True, scenario=scenario)
            # circle_cw_competition_collisionStop, austria_competition
            return env
        return f

    # 強烈建議改用 SubprocVecEnv，TD3 是 off-policy，多進程採樣速度快 5~8 倍
    vec_env = SubprocVecEnv([make_env() for i in range(8)])   # 24~32 個都行，看你 GPU 記憶體

    os.makedirs(f'./{scenario}/checkpoints/td3/', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=6250,                     # 每 50 萬步存一次（TD3 收斂快，可以存密一點）
        save_path=f'./{scenario}/checkpoints/td3/',
        name_prefix=f'td3_{scenario}'
    )
    os.makedirs(f"./{scenario}/training_eval_videos/td3/", exist_ok=True)
    video_eval_callback = VideoEvalCallback(
        video_folder=f"./{scenario}/training_eval_videos/td3/",
        scenario=scenario,
        algorithm='td3',
    )

    tensorboard_log_dir = f"./{scenario}/tensorboard_logs/"
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    if load_model_path is not None and os.path.isfile(load_model_path):
        print(f"Loading model from {load_model_path}")
        model = TD3.load(load_model_path, env=vec_env, device="cuda")
        reset_timesteps = False
    else:
        model = TD3(
            "CnnPolicy",
            vec_env,
            buffer_size=1_000_000,          # 1e6 就夠了，太多吃記憶體
            learning_rate=5e-5,           # 1e-4 比 3e-4 更穩定（很多人用 3e-4 會爆炸）
            gamma=0.99,                   # TD3 建議用 0.98~0.99（比 PPO 的 0.95 更高）
            train_freq=(1, "step"),        # 每步都 train
            gradient_steps=1,
            action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2)),  # 動作噪音
            policy_kwargs=dict(net_arch=[512, 512, 512, 512]),  # 稍微大一點的網路
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
            device="cuda"
        )
        reset_timesteps = True


    model.learn(
        total_timesteps=1e7,
        callback=[checkpoint_callback, video_eval_callback],
        tb_log_name=f"td3",
        progress_bar=True,
        reset_num_timesteps=reset_timesteps,
    )
    model.save(f"td3_{scenario}_final")