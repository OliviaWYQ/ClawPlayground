import time
import pybullet as p
import pybullet_data

# 1. 连接物理引擎 (打开窗口)
physicsClient = p.connect(p.GUI)

# 2. 添加资源搜索路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. 设置重力
p.setGravity(0, 0, -10)

# 4. 加载地面和一个球
planeId = p.loadURDF("plane.urdf")
ballId = p.loadURDF("sphere_small.urdf", [0, 0, 2])

print("窗口已启动：按 q 退出")

# 5. 持续模拟，直到按 q
while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

    keys = p.getKeyboardEvents()
    if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
        break

# 6. 关闭
p.disconnect()
print("已退出")
