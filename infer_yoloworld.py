from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO(r"E:\code\ultralytics\runs\pose\train7\weights\last_openvino_model")  # or select yolov8m/l-world.pt for different sizes

# Execute inference with the YOLOv8s-world model on the specified image
results = model.predict(r"E:\code\cow_pose_estimation\test.png")

# Show results
results[0].show()