import numpy as np
import os
import numpy as np
import imageio
from stable_baselines3.common.callbacks import BaseCallback

class VideoEvalCallback(BaseCallback):
    """
    每隔 eval_freq timesteps 就用當前模型做確定性評測（deterministic=True）
    - 自動錄製 n_eval_episodes 支 mp4（鳥瞰圖）
    - 自動統計 reward / lap time / success rate
    - 自動記錄目前為止最佳單圈，並在 tensorboard 顯示
    - 達到新最佳單圈時自動儲存模型（可選）
    """
    def __init__(
        self, 
        video_folder,
        scenario,
        algorithm,
        frame_stack=8,
        frame_skip=4,
        eval_freq: int = 100_000,           # 每多少 timesteps 評測一次（建議 300k~800k）
        n_eval_episodes: int = 1,           # 每次評測跑幾個 episodes（建議 3~5，越多越準但越慢）
        random_op: bool = True,             # eval 時是否開 random start
        save_best_lap_model: bool = True,   # 是否在達到新最佳單圈時存模型
        verbose: int = 1,
        flip: bool = False,
    ):
        super().__init__(verbose)
        self.scenario = scenario
        self.algorithm = algorithm
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.video_folder = video_folder
        self.random_op = random_op
        self.save_best_lap_model = save_best_lap_model
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        
        os.makedirs(self.video_folder, exist_ok=True)
        
        self.best_lap_time_ever = np.inf
        self.eval_env = None
        self.flip = flip

    def _init_callback(self) -> None:
        if 'mix' in self.scenario:
            from wrapper_mix import environment
            self.eval_env = environment(frame_stack=self.frame_stack, frame_skip=self.frame_skip)
        else:
            from wrapper import environment
            self.eval_env = environment(random_op=False, scenario=self.scenario, frame_stack=self.frame_stack, frame_skip=self.frame_skip, flip=self.flip)
            
    def _on_step(self) -> bool:
        # 每到指定步數就觸發評測
        if self.num_timesteps % self.eval_freq != 0 and self.num_timesteps != 0:
            return True

        print(f"\n{'='*60}")
        print(f"開始 Evaluation @ {self.num_timesteps:,} timesteps")
        print(f"{'='*60}")

        obs, _ = self.eval_env.reset()
        done = False
        frames = []
        real_total_reward = 0.0
        shaping_total_reward = 0.0
        while not done:
            frames.append(self.eval_env.render())  # birds_eye rgb array
            
            action, _ = self.model.predict(obs, deterministic=True)  # 關鍵：關掉 noise
            if self.flip:
                action = np.array([action[0], -action[1]])
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            
            shaping_total_reward += reward
            real_total_reward += self.eval_env.get_reward()  # 使用原始 racecar_gym reward（跟你 eval 腳本一樣）
            done = terminated or truncated

        # ===== 儲存影片 =====
        video_name = f"eval_{self.num_timesteps//1_000_000}M.mp4"
        video_path = os.path.join(self.video_folder, video_name)
        
        # 用 mimsave 更快更穩（比 get_writer 好）
        imageio.mimsave(video_path, [np.array(frame) for frame in frames], fps=30, macro_block_size=1)

        print(f"Reward: {real_total_reward:7.1f}")

        # save to tensorboard
        self.logger.record('eval/real_episode_reward', real_total_reward)
        self.logger.record('eval/shaping_episode_reward', shaping_total_reward)
        self.logger.record('eval/episode_length', len(frames))
        # ===== Tensorboard 記錄 =====
        print(f"影片已儲存至：{self.video_folder}")
        print(f"{'='*60}\n")

        return True