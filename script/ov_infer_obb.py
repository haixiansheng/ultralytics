from ultralytics import YOLO
import os 
import cv2,torch
import numpy as np

def xywhr2xyxyxyxy(center):
    # reference: https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/utils/ops.py#L545
    is_numpy = isinstance(center, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)

    ctr = center[..., :2]
    w, h, angle = (center[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)

def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)

def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)

TEST_INPUT_DIR = r"E:\code\data\gen_barcode_data\train_obb\20241206\images"
TEST_OUTPUT_DIR = r"E:\code\data\tmp\test_result_20241212"
CONF = 0.7
ov_model_path = r"E:\code\ultralytics\runs\obb\train2\weights\last_openvino_model"

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

                results = ov_model(input_image_path)[0]
                

                try:
                    # 读取图像    
                    img = cv2.imread(input_image_path)                
                    names   = results.names
                    boxes   = results.obb.data.cpu()
                    confs   = boxes[..., 5].tolist()
                    classes = list(map(int, boxes[..., 6].tolist()))
                    boxes   = xywhr2xyxyxyxy(boxes[..., :5])

                    for i, box in enumerate(boxes):
                        confidence = confs[i]
                        label = classes[i]
                        color = random_color(label)
                        cv2.polylines(img, [np.asarray(box, dtype=int)], True, color, 2)
                        caption = f"{names[label]} {confidence:.2f}"
                        w, h = cv2.getTextSize(caption, 0 ,1, 2)[0]
                        left, top = [int(b) for b in box[0]]
                        cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
                        cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

                    cv2.imwrite(output_image_path, img)
                except Exception as e:
                    print(f"处理图像 {input_image_path} 时出错: {e}")
    print("[INFO] save done")

# Load a YOLOv8n PyTorch model

ov_model = YOLO(ov_model_path)
process_images_in_folder(TEST_INPUT_DIR,TEST_OUTPUT_DIR,ov_model)    
# Run inference