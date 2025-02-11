import cv2
from ultralytics import YOLO
import os

# 加载预训练的YOLOv8模型
model = YOLO(r"E:\code\ultralytics\runs\detect\train13\weights\best.pt")  # 这里以yolov8n为例，你可以根据需要更换成其他尺寸的模型，如yolov8s、yolov8m等

# 读取要进行推理的图像
# img_dir = r"E:\BaiduNetdiskDownload\20241024-iVision-Scanner\no_det"
image_path = r"E:\code\ultralytics\\1.jpg"  # 替换为你实际的图像路径
image = cv2.imread(image_path)

# 进行推理
results = model(image)

# 可视化结果
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = box.conf[0]
        cls = box.cls[0]
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示结果图像
cv2.namedWindow('YOLOv8 Inference', cv2.WINDOW_NORMAL)
cv2.imshow("YOLOv8 Inference", image)
cv2.waitKey(0)
cv2.destroyAllWindows()