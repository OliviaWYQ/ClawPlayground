import cv2
from ultralytics import YOLO

# 使用你已有的模型
MODEL_PATH = "/home/nvidia/Desktop/test/yolo26n.pt"
CAMERA_INDEX = 0  # 如无画面可改成 1/2
CONF = 0.25


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={CAMERA_INDEX}")

    print("按 q 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取摄像头帧失败")
            break

        # 直接用 .pt 推理，不做 TensorRT 导出
        results = model.predict(frame, conf=CONF, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("YOLO Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
