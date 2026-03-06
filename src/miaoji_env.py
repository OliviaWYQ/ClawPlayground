"""
喵姬物理AI原型（阶段1：虚拟世界避障）

目标：
1) 小球主动避障（“怕疼”）
2) 通过新颖性奖励实现基础“好奇心”
3) 预留情绪层与意志层变量，便于后续迁移到Jetson Orin Nano

依赖：
- pybullet>=3.2.5
- numpy>=1.21.0
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class EmotionState:
    """情绪变量层（The Emotion）"""

    energy: float = 1.0       # 饱腹/能量 [0,1]
    anxiety: float = 0.1      # 焦虑 [0,1]
    pain: float = 0.0         # 疼痛 [0,1]
    curiosity: float = 0.6    # 好奇心 [0,1]


@dataclass
class WillState:
    """意志变量层（The Will）"""

    beta: float = 0.85        # 博弈权重（越大越“坚持自我”）
    conflict: float = 0.0     # 目标冲突度


class MiaoJiBallEnv:
    """
    三层变量体系简化版环境：
    - Body: 位置、速度、动作约束
    - Emotion: 疼痛/焦虑/好奇心/能量
    - Will: beta + 冲突度

    奖励思路：
    - 撞障碍：大惩罚（怕疼）
    - 靠太近：小惩罚（焦虑）
    - 探索新网格：好奇奖励
    - 生存每步：微小正奖励
    """

    def __init__(
        self,
        gui: bool = True,
        real_time: bool = True,
        max_steps: int = 1500,
        world_size: float = 2.0,
        seed: int | None = 42,
    ) -> None:
        self.gui = gui
        self.real_time = real_time
        self.max_steps = max_steps
        self.world_size = world_size

        # 尺寸参数（放大后更易观察）
        self.ball_scale = 0.75
        self.obstacle_scale = 1.35
        self.wall_height = 0.45
        self.wall_thickness = 0.12

        self.rng = np.random.default_rng(seed)

        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 240.0)

        self.plane_id = p.loadURDF("plane.urdf")

        self.wall_ids: List[int] = []
        self.obstacle_ids: List[int] = []
        self._build_walls()
        self._build_obstacles()

        self.ball_id = p.loadURDF(
            "sphere_small.urdf",
            basePosition=[0.0, 0.0, 0.35],
            globalScaling=self.ball_scale,
        )

        # Body层约束
        self.max_force = 15.0
        self.max_speed = 2.5

        # Emotion/Will
        self.emotion = EmotionState()
        self.will = WillState()

        # 观察相关
        self.last_min_dist = 999.0
        self.visited_cells: set[Tuple[int, int]] = set()
        self.step_count = 0

    # ---------- 构建场景 ----------
    def _build_walls(self) -> None:
        """创建边界墙"""
        w = self.world_size
        h = self.wall_height
        t = self.wall_thickness
        wall_cfg = [
            ([0, +w, h], [w, t, h]),
            ([0, -w, h], [w, t, h]),
            ([+w, 0, h], [t, w, h]),
            ([-w, 0, h], [t, w, h]),
        ]

        for pos, half_ext in wall_cfg:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_ext,
                rgbaColor=[0.6, 0.6, 0.7, 1.0],
            )
            bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
            self.wall_ids.append(bid)

    def _build_obstacles(self) -> None:
        """创建固定障碍（阶段1避障）"""
        obstacle_xy = [
            (0.6, 0.4),
            (-0.5, 0.8),
            (0.2, -0.7),
            (-0.8, -0.3),
        ]
        for x, y in obstacle_xy:
            oid = p.loadURDF(
                "cube_small.urdf",
                basePosition=[x, y, 0.30],
                globalScaling=self.obstacle_scale,
            )
            self.obstacle_ids.append(oid)

    # ---------- 环境接口 ----------
    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.visited_cells.clear()

        # 重置球状态
        p.resetBasePositionAndOrientation(self.ball_id, [0.0, 0.0, 0.35], [0, 0, 0, 1])
        p.resetBaseVelocity(self.ball_id, [0, 0, 0], [0, 0, 0])

        # 重置情绪与意志
        self.emotion = EmotionState()
        self.will = WillState()

        pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        self.last_min_dist = self._min_distance_to_obstacles(np.array(pos[:2]))
        self._mark_visited(np.array(pos[:2]))

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1

        # 动作裁剪
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0) * self.max_force

        # 意志层：beta越大越“保守”地执行动作（更趋向自我安全）
        obedience = 1.0 - self.will.beta * 0.35
        applied = action * obedience

        p.applyExternalForce(
            self.ball_id,
            -1,
            forceObj=[float(applied[0]), float(applied[1]), 0.0],
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
        )

        p.stepSimulation()
        if self.gui and self.real_time:
            time.sleep(1.0 / 240.0)

        # 速度限制（Body层安全约束）
        lv, av = p.getBaseVelocity(self.ball_id)
        vxy = np.array(lv[:2], dtype=np.float32)
        speed = np.linalg.norm(vxy)
        if speed > self.max_speed:
            scale = self.max_speed / (speed + 1e-6)
            new_lv = [lv[0] * scale, lv[1] * scale, lv[2]]
            p.resetBaseVelocity(self.ball_id, linearVelocity=new_lv, angularVelocity=av)

        obs = self._get_obs()
        pos = obs[0:2]

        reward, done, info = self._compute_reward_done(pos)
        self._update_internal_states(pos, reward, info)

        return obs, reward, done, info

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    # ---------- 内部逻辑 ----------
    def _get_obs(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        lv, _ = p.getBaseVelocity(self.ball_id)
        x, y = pos[0], pos[1]
        vx, vy = lv[0], lv[1]

        min_dist = self._min_distance_to_obstacles(np.array([x, y], dtype=np.float32))

        # 观测：物理 + 情绪 + 意志
        obs = np.array(
            [
                x,
                y,
                vx,
                vy,
                min_dist,
                self.emotion.energy,
                self.emotion.anxiety,
                self.emotion.pain,
                self.emotion.curiosity,
                self.will.beta,
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_reward_done(self, pos_xy: np.ndarray) -> Tuple[float, bool, Dict]:
        # 1) 撞击惩罚
        hit = False
        for oid in self.obstacle_ids + self.wall_ids:
            if p.getContactPoints(self.ball_id, oid):
                hit = True
                break

        min_dist = self._min_distance_to_obstacles(pos_xy)

        # 2) 生存奖励
        reward = 0.03

        # 3) 近距离焦虑惩罚
        if min_dist < 0.22:
            reward -= 0.8
        elif min_dist < 0.35:
            reward -= 0.2

        # 4) 好奇心（探索新网格）
        cell = self._cell_of(pos_xy)
        is_new = cell not in self.visited_cells
        if is_new:
            self.visited_cells.add(cell)
            reward += 0.22 * self.emotion.curiosity

        # 5) 距离改善奖励（远离危险）
        improvement = self.last_min_dist - min_dist
        reward += float(np.clip(improvement, -0.2, 0.2) * 0.5)
        self.last_min_dist = min_dist

        done = False
        if hit:
            reward -= 10.0
            done = True

        if self.step_count >= self.max_steps:
            done = True

        info = {
            "hit": hit,
            "min_dist": float(min_dist),
            "new_cell": is_new,
        }
        return float(reward), done, info

    def _update_internal_states(self, pos_xy: np.ndarray, reward: float, info: Dict) -> None:
        # 疼痛
        if info["hit"]:
            self.emotion.pain = min(1.0, self.emotion.pain + 0.6)
        else:
            self.emotion.pain = max(0.0, self.emotion.pain - 0.015)

        # 焦虑：离障碍近时上升
        if info["min_dist"] < 0.30:
            self.emotion.anxiety = min(1.0, self.emotion.anxiety + 0.03)
        else:
            self.emotion.anxiety = max(0.0, self.emotion.anxiety - 0.01)

        # 能量随时间衰减，奖励好时回一点
        self.emotion.energy = np.clip(self.emotion.energy - 0.0008 + max(0.0, reward) * 0.0005, 0.0, 1.0)

        # 好奇心：新探索略升，长期无新探索会降
        if info.get("new_cell", False):
            self.emotion.curiosity = min(1.0, self.emotion.curiosity + 0.003)
        else:
            self.emotion.curiosity = max(0.1, self.emotion.curiosity - 0.001)

        # 意志：痛苦/焦虑高 -> 更保守（beta上升）；状态好 -> 更愿意尝试（beta下降）
        stress = 0.6 * self.emotion.pain + 0.4 * self.emotion.anxiety
        self.will.beta = float(np.clip(0.65 + 0.35 * stress, 0.5, 0.98))

        # 冲突度（示例）：把“探索倾向”与“避险倾向”差异当作冲突
        explore_drive = self.emotion.curiosity
        safety_drive = 1.0 - self.will.beta
        self.will.conflict = float(abs(explore_drive - safety_drive))

    def _min_distance_to_obstacles(self, pos_xy: np.ndarray) -> float:
        min_dist = 999.0
        for oid in self.obstacle_ids:
            op, _ = p.getBasePositionAndOrientation(oid)
            d = float(np.linalg.norm(pos_xy - np.array(op[:2], dtype=np.float32)))
            min_dist = min(min_dist, d)

        # 与边界墙距离（近似）
        w = self.world_size
        wall_dist = min(w - abs(float(pos_xy[0])), w - abs(float(pos_xy[1])))
        min_dist = min(min_dist, wall_dist)

        return max(0.0, min_dist)

    def _cell_of(self, pos_xy: np.ndarray, grid: float = 0.25) -> Tuple[int, int]:
        return int(math.floor(pos_xy[0] / grid)), int(math.floor(pos_xy[1] / grid))

    def _mark_visited(self, pos_xy: np.ndarray) -> None:
        self.visited_cells.add(self._cell_of(pos_xy))


# -------------------------
# 演示运行（随机策略 + 轻微避险偏置）
# -------------------------
if __name__ == "__main__":
    print("启动喵姬阶段1环境：虚拟世界避障")

    env = MiaoJiBallEnv(gui=True, real_time=True, max_steps=1200, world_size=2.0)
    obs = env.reset()

    try:
        for i in range(8000):
            # 简单策略：随机 + 轻微远离最近障碍趋势（便于观察）
            rand_action = np.random.uniform(-1, 1, size=(2,))

            # 读取当前位置，估计最近障碍方向
            pos_xy = obs[:2]
            nearest_vec = np.array([0.0, 0.0])
            nearest_d = 1e9
            for oid in env.obstacle_ids:
                op, _ = p.getBasePositionAndOrientation(oid)
                vec = pos_xy - np.array(op[:2])
                d = np.linalg.norm(vec)
                if d < nearest_d:
                    nearest_d = d
                    nearest_vec = vec

            avoid = nearest_vec / (np.linalg.norm(nearest_vec) + 1e-6)
            action = 0.65 * rand_action + 0.35 * avoid
            action = np.clip(action, -1, 1)

            obs, reward, done, info = env.step(action)

            if i % 120 == 0:
                print(
                    f"step={i:4d} reward={reward:+.3f} dist={info['min_dist']:.3f} "
                    f"pain={env.emotion.pain:.2f} anx={env.emotion.anxiety:.2f} "
                    f"cur={env.emotion.curiosity:.2f} beta={env.will.beta:.2f}"
                )

            if done:
                print(f"episode done at step={i}, hit={info['hit']}; reset")
                obs = env.reset()

            keys = p.getKeyboardEvents()
            if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
                break
    finally:
        env.close()
        print("环境关闭。")
