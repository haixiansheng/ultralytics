from ultralytics import YOLO

# 加载预训练的YOLOv8模型，这里以yolov8n为例，你可以根据需要更换成其他尺寸的模型，如yolov8s、yolov8m等
model = YOLO(r'E:\code\ultralytics\runs\detect\train9\weights\last.pt')

# 设置导出ONNX模型的参数，包括简化模型、动态输入尺寸等（可根据需求调整）
export_params = {
    # "simplify": True,  # 简化模型，可减小模型文件大小并可能提高推理速度
    "dynamic": False,  # 允许动态输入尺寸，使得模型在不同尺寸输入下能正常工作
    "opset": 12,  # ONNX操作集版本，根据实际情况和目标平台兼容性选择合适的值
}

# 执行导出操作，将模型导出为ONNX格式，指定导出路径和文件名
model.export(format='onnx', **export_params)