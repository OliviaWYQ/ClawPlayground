#!/usr/bin/env bash
set -euo pipefail

# ===== 可改参数 =====
PYTHON_BIN="/home/nvidia/.venv/bin/python"
MODEL_PT="/home/nvidia/Desktop/test/yolo26n.pt"
ENGINE_OUT="/home/nvidia/Desktop/test/yolo26n.engine"
CAMERA_INDEX="0"
CONF="0.25"
IMGSZ="320"   # Orin 显存紧张时用 320，更稳

TORCH_WHL="/home/nvidia/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl"
TV_WHL="/home/nvidia/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl"

MODE="${1:-all}" # all | export | run

# 让 venv 能看到系统安装的 TensorRT Python 包（Jetson 常见）
VENV_SP="/home/nvidia/.venv/lib/python3.10/site-packages"
for pkg in tensorrt tensorrt_dispatch tensorrt_lean; do
  if [[ -d "/usr/lib/python3.10/dist-packages/$pkg" && ! -e "$VENV_SP/$pkg" ]]; then
    ln -s "/usr/lib/python3.10/dist-packages/$pkg" "$VENV_SP/$pkg"
  fi
done

echo "[1/5] 基础检查"
[[ -x "$PYTHON_BIN" ]] || { echo "❌ Python 不存在: $PYTHON_BIN"; exit 1; }
[[ -f "$MODEL_PT" ]] || { echo "❌ 模型不存在: $MODEL_PT"; exit 1; }
[[ -f "$TORCH_WHL" ]] || { echo "❌ 缺少 torch wheel: $TORCH_WHL"; exit 1; }
[[ -f "$TV_WHL" ]] || { echo "❌ 缺少 torchvision wheel: $TV_WHL"; exit 1; }

echo "[2/5] 恢复并固定 NVIDIA Torch 版本"
uv pip install --python "$PYTHON_BIN" --no-deps --force-reinstall "$TORCH_WHL" "$TV_WHL"

echo "[3/5] 安装导出依赖（禁用 YOLO 自动改依赖）"
# 关键：ultralytics 用 --no-deps，避免解析器把 torch 升级为 CPU 版
YOLO_AUTOINSTALL=False uv pip install --python "$PYTHON_BIN" \
  "numpy==1.26.4" "onnx>=1.12,<2" "opencv-python<4.12"
YOLO_AUTOINSTALL=False uv pip install --python "$PYTHON_BIN" --no-deps ultralytics
# Jetson 上 onnxruntime 常与 torchvision 冲突，确保移除
uv pip uninstall --python "$PYTHON_BIN" onnxruntime onnxruntime-gpu || true

echo "[4/5] 验证 CUDA"
"$PYTHON_BIN" - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('❌ CUDA 不可用，停止')
print('device:', torch.cuda.get_device_name(0))
PY

if [[ "$MODE" == "all" || "$MODE" == "export" ]]; then
  echo "[5/5] 导出 TensorRT 引擎"
  YOLO_AUTOINSTALL=False "$PYTHON_BIN" - <<PY
from ultralytics import YOLO
m = YOLO("$MODEL_PT")
m.export(format="engine", device=0, half=True, simplify=False, imgsz=int("$IMGSZ"), batch=1, workspace=1)
print("✅ 导出完成")
PY

  [[ -f "$ENGINE_OUT" ]] || { echo "❌ 未找到导出的 engine: $ENGINE_OUT"; exit 1; }
  ls -lh "$ENGINE_OUT"
fi

if [[ "$MODE" == "all" || "$MODE" == "run" ]]; then
  echo "启动摄像头实时检测（按 q 退出）"
  "$PYTHON_BIN" - <<PY
import cv2
from ultralytics import YOLO

engine = "$ENGINE_OUT"
model = YOLO(engine)
cap = cv2.VideoCapture(int("$CAMERA_INDEX"))
if not cap.isOpened():
    raise RuntimeError(f"无法打开摄像头 index={$CAMERA_INDEX}")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    results = model.predict(frame, conf=float("$CONF"), imgsz=int("$IMGSZ"), verbose=False)
    vis = results[0].plot()
    cv2.imshow("YOLO TensorRT Webcam", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
PY
fi

echo "✅ 完成（mode=$MODE）"
