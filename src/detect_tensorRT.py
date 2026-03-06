import os
import sys
import time
import argparse
import subprocess
import shutil
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd_stream(cmd):
    log("执行命令: " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in p.stdout:
            print(line.rstrip(), flush=True)
    finally:
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"命令失败，退出码={rc}: {' '.join(cmd)}")


def export_engine(model_pt: Path, onnx_path: Path, engine_path: Path, imgsz: int, force: bool):
    if engine_path.exists() and not force:
        log(f"已存在 engine，跳过导出: {engine_path}")
        return

    if force and engine_path.exists():
        log(f"FORCE_EXPORT=1，删除旧 engine: {engine_path}")
        engine_path.unlink()

    log(f"开始导出 ONNX: {model_pt} -> {onnx_path}")
    m = YOLO(str(model_pt), task="detect")
    m.export(format="onnx", imgsz=imgsz, simplify=False, opset=12)

    if not onnx_path.exists():
        raise RuntimeError(f"ONNX 导出失败，未找到: {onnx_path}")

    trtexec = os.getenv("TRTEXEC") or shutil.which("trtexec")
    if not trtexec:
        for c in ("/usr/src/tensorrt/bin/trtexec", "/usr/local/tensorrt/bin/trtexec"):
            if os.path.exists(c):
                trtexec = c
                break

    if trtexec:
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
            "--verbose",
            f"--minShapes=images:1x3x{imgsz}x{imgsz}",
            f"--optShapes=images:1x3x{imgsz}x{imgsz}",
            f"--maxShapes=images:1x3x{imgsz}x{imgsz}",
        ]

        log("开始构建 TensorRT engine（trtexec verbose 模式）")
        try:
            run_cmd_stream(cmd)
        except Exception as e:
            log(f"trtexec 失败: {e}")
            log("回退到 Ultralytics m.export(format='engine')")
            m = YOLO(str(model_pt), task="detect")
            m.export(format="engine", device=0, half=True, simplify=False, imgsz=imgsz, batch=1, workspace=1)
    else:
        log("未找到 trtexec，回退到 Ultralytics m.export(format='engine')")
        m = YOLO(str(model_pt), task="detect")
        m.export(format="engine", device=0, half=True, simplify=False, imgsz=imgsz, batch=1, workspace=1)

    if not engine_path.exists():
        raise RuntimeError(f"TensorRT 导出失败，未找到: {engine_path}")

    log(f"✅ 导出完成: {engine_path} ({engine_path.stat().st_size / (1024*1024):.2f} MiB)")


def run_infer(engine_path: Path, camera_index: int, conf: float, imgsz: int):
    log(f"加载 TensorRT engine: {engine_path}")
    model = YOLO(str(engine_path), task="detect")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={camera_index}")

    log("开始实时推理，按 q 退出")
    frame_id = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            log("读取摄像头帧失败，退出")
            break

        results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        vis = results[0].plot()
        cv2.imshow("YOLO TensorRT Webcam", vis)

        frame_id += 1
        if frame_id % 60 == 0:
            fps = frame_id / max(time.time() - t0, 1e-6)
            log(f"推理中... frame={frame_id}, avg_fps={fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO TensorRT 导出+推理脚本（Jetson）")
    parser.add_argument("--mode", choices=["all", "export", "run"], default=os.getenv("MODE", "all"))
    parser.add_argument("--force-export", action="store_true", default=os.getenv("FORCE_EXPORT", "0") == "1")
    parser.add_argument("--model-pt", default=os.getenv("MODEL_PT", "/home/nvidia/Desktop/test/yolo26n.pt"))
    parser.add_argument("--onnx", default=os.getenv("ONNX_OUT", "/home/nvidia/Desktop/test/yolo26n.onnx"))
    parser.add_argument("--engine", default=os.getenv("ENGINE_OUT", "/home/nvidia/Desktop/test/yolo26n.engine"))
    parser.add_argument("--camera-index", type=int, default=int(os.getenv("CAMERA_INDEX", "0")))
    parser.add_argument("--conf", type=float, default=float(os.getenv("CONF", "0.25")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("IMGSZ", "320")))
    args = parser.parse_args()

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA 不可用，停止")
    print("device:", torch.cuda.get_device_name(0))

    model_pt = Path(args.model_pt)
    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)

    if not model_pt.exists():
        raise FileNotFoundError(f"未找到模型: {model_pt}")

    if args.mode in ("all", "export"):
        export_engine(model_pt, onnx_path, engine_path, args.imgsz, args.force_export)

    if args.mode in ("all", "run"):
        if not engine_path.exists():
            raise FileNotFoundError(f"未找到 engine: {engine_path}，请先导出")
        run_infer(engine_path, args.camera_index, args.conf, args.imgsz)


if __name__ == "__main__":
    main()
