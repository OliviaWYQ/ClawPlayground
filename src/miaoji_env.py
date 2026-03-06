"""
喵姬物理AI原型（阶段1：虚拟世界避障）

更新点：
1) 球体放大（更易观察）
2) 墙体恢复为早期厚度/高度
3) 障碍物数量与位置每回合随机
4) 增加“局部感知”观测：只看附近障碍，感知范围会随运动状态变化
5) 增大活动空间，并在较大范围内随机出生
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
    energy: float = 1.0
    anxiety: float = 0.1
    pain: float = 0.0
    curiosity: float = 0.6


@dataclass
class WillState:
    beta: float = 0.85
    conflict: float = 0.0


class MiaoJiBallEnv:
    def __init__(
        self,
        gui: bool = True,
        real_time: bool = True,
        max_steps: int = 1500,
        world_size: float = 3.5,
        seed: int | None = 42,
    ) -> None:
        self.gui = gui
        self.real_time = real_time
        self.max_steps = max_steps
        self.world_size = world_size

        # 尺寸参数
        self.ball_scale = 0.75
        self.obstacle_scale = 1.05

        # 墙体恢复“之前”视觉参数
        self.wall_height = 0.3
        self.wall_thickness = 0.05

        # 动态障碍参数
        self.min_obstacles = 3
        self.max_obstacles = 7
        self.obstacle_clearance = 0.7  # 障碍之间最小间距

        # 局部感知参数（范围会动态变化）
        self.base_sensor_range = 1.0
        self.max_sensor_range = 2.4
        self.sensor_speed_gain = 0.45
        self.sensor_turn_gain = 0.6
        self.prev_heading = np.array([1.0, 0.0], dtype=np.float32)
        self.sensor_range = self.base_sensor_range
        self.max_nearby_obs = 3

        self.rng = np.random.default_rng(seed)

        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 240.0)

        self.plane_id = p.loadURDF("plane.urdf")

        self.wall_ids: List[int] = []
        self.obstacle_ids: List[int] = []
        self._build_walls()

        self.ball_id = p.loadURDF(
            "sphere_small.urdf",
            basePosition=[0.0, 0.0, 0.35],
            globalScaling=self.ball_scale,
        )

        # Body层约束
        self.max_force = 15.0
        self.max_speed = 3.0

        # Emotion/Will
        self.emotion = EmotionState()
        self.will = WillState()

        # 观察相关
        self.last_min_dist = 999.0
        self.visited_cells: set[Tuple[int, int]] = set()
        self.step_count = 0

        self._spawn_random_obstacles()

    # ---------- 构建场景 ----------
    def _build_walls(self) -> None:
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
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
            )
            self.wall_ids.append(bid)

    def _clear_obstacles(self) -> None:
        for oid in self.obstacle_ids:
            p.removeBody(oid)
        self.obstacle_ids = []

    def _spawn_random_obstacles(self) -> None:
        self._clear_obstacles()
        count = int(self.rng.integers(self.min_obstacles, self.max_obstacles + 1))

        placed: List[np.ndarray] = []
        bound = self.world_size - 0.7

        for _ in range(count):
            for _try in range(120):
                x = float(self.rng.uniform(-bound, bound))
                y = float(self.rng.uniform(-bound, bound))
                pt = np.array([x, y], dtype=np.float32)

                # 避免在出生区域太近
                if np.linalg.norm(pt) < 0.9:
                    continue

                ok = True
                for prev in placed:
                    if np.linalg.norm(pt - prev) < self.obstacle_clearance:
                        ok = False
                        break
                if not ok:
                    continue

                oid = p.loadURDF(
                    "cube_small.urdf",
                    basePosition=[x, y, 0.22],
                    globalScaling=self.obstacle_scale,
                )
                self.obstacle_ids.append(oid)
                placed.append(pt)
                break

    # ---------- 环境接口 ----------
    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.visited_cells.clear()
        self.prev_heading = np.array([1.0, 0.0], dtype=np.float32)
        self.sensor_range = self.base_sensor_range

        # 每回合随机障碍布局
        self._spawn_random_obstacles()

        # 更大范围随机出生，避免总在中间
        spawn_bound = self.world_size - 1.0
        sx = float(self.rng.uniform(-spawn_bound, spawn_bound))
        sy = float(self.rng.uniform(-spawn_bound, spawn_bound))
        p.resetBasePositionAndOrientation(self.ball_id, [sx, sy, 0.35], [0, 0, 0, 1])
        p.resetBaseVelocity(self.ball_id, [0, 0, 0], [0, 0, 0])

        self.emotion = EmotionState()
        self.will = WillState()

        pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        self.last_min_dist = self._min_distance_to_obstacles(np.array(pos[:2]))
        self._mark_visited(np.array(pos[:2]))

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0) * self.max_force

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

        lv, av = p.getBaseVelocity(self.ball_id)
        vxy = np.array(lv[:2], dtype=np.float32)
        speed = np.linalg.norm(vxy)
        if speed > self.max_speed:
            scale = self.max_speed / (speed + 1e-6)
            new_lv = [lv[0] * scale, lv[1] * scale, lv[2]]
            p.resetBaseVelocity(self.ball_id, linearVelocity=new_lv, angularVelocity=av)

        self._update_sensor_range(vxy)

        obs = self._get_obs()
        pos = obs[0:2]

        reward, done, info = self._compute_reward_done(pos)
        self._update_internal_states(pos, reward, info)

        return obs, reward, done, info

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    # ---------- 内部逻辑 ----------
    def _update_sensor_range(self, vxy: np.ndarray) -> None:
        speed = float(np.linalg.norm(vxy))
        if speed > 1e-4:
            heading = vxy / (speed + 1e-8)
        else:
            heading = self.prev_heading

        turn_amount = float(np.linalg.norm(heading - self.prev_heading))
        self.prev_heading = heading

        dynamic = (
            self.base_sensor_range
            + self.sensor_speed_gain * speed
            + self.sensor_turn_gain * turn_amount
        )
        self.sensor_range = float(np.clip(dynamic, self.base_sensor_range, self.max_sensor_range))

    def _get_nearby_obstacles(self, pos_xy: np.ndarray) -> Tuple[List[np.ndarray], int]:
        nearby: List[Tuple[float, np.ndarray]] = []
        for oid in self.obstacle_ids:
            op, _ = p.getBasePositionAndOrientation(oid)
            rel = np.array([op[0] - pos_xy[0], op[1] - pos_xy[1]], dtype=np.float32)
            d = float(np.linalg.norm(rel))
            if d <= self.sensor_range:
                nearby.append((d, rel))

        nearby.sort(key=lambda x: x[0])
        rels = [x[1] for x in nearby[: self.max_nearby_obs]]
        count = len(nearby)
        return rels, count

    def _get_obs(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        lv, _ = p.getBaseVelocity(self.ball_id)
        x, y = pos[0], pos[1]
        vx, vy = lv[0], lv[1]

        pos_xy = np.array([x, y], dtype=np.float32)
        min_dist = self._min_distance_to_obstacles(pos_xy)
        nearby_rels, nearby_count = self._get_nearby_obstacles(pos_xy)

        local_feats = []
        for i in range(self.max_nearby_obs):
            if i < len(nearby_rels):
                rel = nearby_rels[i] / (self.sensor_range + 1e-6)
                local_feats.extend([float(np.clip(rel[0], -1, 1)), float(np.clip(rel[1], -1, 1))])
            else:
                local_feats.extend([0.0, 0.0])

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
                self.sensor_range / self.max_sensor_range,
                min(1.0, nearby_count / 6.0),
                *local_feats,  # 3个障碍 * (rel_x, rel_y)
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_reward_done(self, pos_xy: np.ndarray) -> Tuple[float, bool, Dict]:
        hit = False
        for oid in self.obstacle_ids + self.wall_ids:
            if p.getContactPoints(self.ball_id, oid):
                hit = True
                break

        min_dist = self._min_distance_to_obstacles(pos_xy)

        reward = 0.03

        if min_dist < 0.22:
            reward -= 0.8
        elif min_dist < 0.35:
            reward -= 0.2

        cell = self._cell_of(pos_xy)
        is_new = cell not in self.visited_cells
        if is_new:
            self.visited_cells.add(cell)
            reward += 0.22 * self.emotion.curiosity

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
            "obstacles": len(self.obstacle_ids),
            "sensor_range": self.sensor_range,
        }
        return float(reward), done, info

    def _update_internal_states(self, pos_xy: np.ndarray, reward: float, info: Dict) -> None:
        if info["hit"]:
            self.emotion.pain = min(1.0, self.emotion.pain + 0.6)
        else:
            self.emotion.pain = max(0.0, self.emotion.pain - 0.015)

        if info["min_dist"] < 0.30:
            self.emotion.anxiety = min(1.0, self.emotion.anxiety + 0.03)
        else:
            self.emotion.anxiety = max(0.0, self.emotion.anxiety - 0.01)

        self.emotion.energy = np.clip(self.emotion.energy - 0.0008 + max(0.0, reward) * 0.0005, 0.0, 1.0)

        if info.get("new_cell", False):
            self.emotion.curiosity = min(1.0, self.emotion.curiosity + 0.003)
        else:
            self.emotion.curiosity = max(0.1, self.emotion.curiosity - 0.001)

        stress = 0.6 * self.emotion.pain + 0.4 * self.emotion.anxiety
        self.will.beta = float(np.clip(0.65 + 0.35 * stress, 0.5, 0.98))

        explore_drive = self.emotion.curiosity
        safety_drive = 1.0 - self.will.beta
        self.will.conflict = float(abs(explore_drive - safety_drive))

    def _min_distance_to_obstacles(self, pos_xy: np.ndarray) -> float:
        min_dist = 999.0
        for oid in self.obstacle_ids:
            op, _ = p.getBasePositionAndOrientation(oid)
            d = float(np.linalg.norm(pos_xy - np.array(op[:2], dtype=np.float32)))
            min_dist = min(min_dist, d)

        w = self.world_size
        wall_dist = min(w - abs(float(pos_xy[0])), w - abs(float(pos_xy[1])))
        min_dist = min(min_dist, wall_dist)

        return max(0.0, min_dist)

    def _cell_of(self, pos_xy: np.ndarray, grid: float = 0.28) -> Tuple[int, int]:
        return int(math.floor(pos_xy[0] / grid)), int(math.floor(pos_xy[1] / grid))

    def _mark_visited(self, pos_xy: np.ndarray) -> None:
        self.visited_cells.add(self._cell_of(pos_xy))


if __name__ == "__main__":
    print("启动喵姬阶段1环境：随机障碍 + 动态感知")

    env = MiaoJiBallEnv(gui=True, real_time=True, max_steps=1200, world_size=3.5)
    obs = env.reset()

    try:
        for i in range(9000):
            rand_action = np.random.uniform(-1, 1, size=(2,))

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
                    f"obsN={info['obstacles']} sense={info['sensor_range']:.2f} "
                    f"pain={env.emotion.pain:.2f} anx={env.emotion.anxiety:.2f} beta={env.will.beta:.2f}"
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
