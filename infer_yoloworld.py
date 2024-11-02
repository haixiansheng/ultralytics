from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld(r"E:\code\model\yolov8m-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes

# Execute inference with the YOLOv8s-world model on the specified image
results = model.predict(r"E:\code\data\barcode_img\202410241426349758_S.jpg")

# Show results
results[0].show()