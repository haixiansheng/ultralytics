from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
# model = YOLO("yolov8n.pt")

# # Export the model
# model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO(r"E:\code\ultralytics\runs\detect\train15\weights\last_openvino_model")

# Run inference
results = ov_model(r"E:\code\bar-detect.jpg")
print(results)