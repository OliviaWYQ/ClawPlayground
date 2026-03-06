"""
演示脚本：play_ball.py

用法示例：
  python3 play_ball.py --model ../models/ppo_miaoji_ball --episodes 5
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from miaoji_env import MiaoJiBallEnv


class MiaoJiGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, gui: bool = True, max_steps: int = 1500):
        super().__init__()
        self.core = MiaoJiBallEnv(gui=gui, real_time=gui, max_steps=max_steps)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        obs = self.core.reset().astype(np.float32)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.core.step(action)
        terminated = bool(done)
        truncated = False
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self):
        self.core.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../models/ppo_miaoji_ball", help="模型路径")
    parser.add_argument("--episodes", type=int, default=3, help="演示回合数")
    parser.add_argument("--max-steps", type=int, default=1500, help="每回合最大步数")
    args = parser.parse_args()

    env = MiaoJiGymEnv(gui=True, max_steps=args.max_steps)
    model = PPO.load(args.model)

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            for _ in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            print(f"episode={ep + 1}, reward={ep_reward:.2f}, hit={info.get('hit')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
