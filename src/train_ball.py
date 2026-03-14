"""
训练脚本：train_ball.py

用法示例：
  python3 train_ball.py --steps 200000 --model ../models/ppo_miaoji_ball
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from miaoji_env import MiaoJiBallEnv


class MiaoJiGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, gui: bool = False, max_steps: int = 1500):
        super().__init__()
        self.core = MiaoJiBallEnv(gui=gui, real_time=gui, max_steps=max_steps)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.core.reset().astype(np.float32)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.core.step(action)
        terminated = bool(done)
        truncated = False
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self):
        self.core.close()


def make_env(max_steps: int):
    return lambda: MiaoJiGymEnv(gui=False, max_steps=max_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000, help="训练步数")
    parser.add_argument("--max-steps", type=int, default=1500, help="每个episode最大步数")
    default_model_path = str(
        (Path(__file__).resolve().parent.parent / "models" / "ppo_miaoji_ball")
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model_path,
        help="模型保存路径（不带.zip也可以）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="训练设备，Jetson 默认建议 cpu（更稳，不易OOM）",
    )
    parser.add_argument("--n-steps", type=int, default=512, help="PPO rollout 步数")
    parser.add_argument("--batch-size", type=int, default=32, help="PPO batch size")
    args = parser.parse_args()

    vec_env = DummyVecEnv([make_env(args.max_steps)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=3e-4,
        gamma=0.99,
        device=args.device,
    )

    model.learn(total_timesteps=args.steps)
    model.save(args.model)
    vec_env.close()
    print(f"训练完成，模型已保存: {args.model}.zip")


if __name__ == "__main__":
    main()
