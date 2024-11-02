from ultralytics import YOLO
import os 
import cv2


# 导出模型必须参数
NEED_TEST = True
INT8 = False
HALF = True
EXPORT_PATH = r"E:\code\ultralytics\runs\detect\train10\weights"

# 测试图像必须参数
TEST_INPUT_DIR = r"E:\code\data\test_custom\\7"
TEST_OUTPUT_DIR = r"E:\code\data\tmp\test_result5"
CONF = 0.7

def process_images_in_folder(input_folder, output_folder,ov_model):
    class_names = ["barcode"]
    for root, dirs, files in os.walk(input_folder):
        # 在输出文件夹中创建对应的子文件夹结构
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                input_image_path = os.path.join(root, file)
                output_image_path = os.path.join(output_subfolder, file)

                results = ov_model(input_image_path)

                try:
                    # 读取图像    
                    img = cv2.imread(input_image_path)                
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            confidence = box.conf[0]
                            class_id = int(box.cls[0])
                            class_name = class_names[class_id]
                            label = f"{class_name}: {confidence:.2f}"
                            if confidence < CONF:
                                continue
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    cv2.imwrite(output_image_path, img)
                except Exception as e:
                    print(f"处理图像 {input_image_path} 时出错: {e}")

# Load a YOLOv8n PyTorch model
pt_model_path = os.path.join(EXPORT_PATH,"last.pt")
model = YOLO(pt_model_path)

# Export the model
if INT8:
    model.export(format="openvino",int8=True)  # creates 'yolov8n_openvino_model/'
elif HALF:
    model.export(format="openvino",half=True)  # creates 'yolov8n_openvino_model/'
else:
    model.export(format="openvino")

if NEED_TEST:

    # ov_model_path = os.path.join(EXPORT_PATH,"last_int8_openvino_model")
    if INT8:
        ov_model_path = os.path.join(EXPORT_PATH,"last_int8_openvino_model")
    else:
        ov_model_path = os.path.join(EXPORT_PATH,"last_openvino_model")

    ov_model = YOLO(ov_model_path)
    process_images_in_folder(TEST_INPUT_DIR,TEST_OUTPUT_DIR,ov_model)    
    # Run inference
    