import json
import os
from PIL import Image
import shutil
import numpy as np
import cv2

label_dict = {"box":0,"fp":1}

def sort_corners(corners):
    """
    对给定的方形角点进行顺时针排序
    :param corners: 角点坐标列表，格式为[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    :return: 排序后的角点坐标列表
    """
    # 将角点坐标转换为numpy数组方便计算
    corners_np = np.array(corners)
    # 计算质心坐标
    centroid = np.mean(corners_np, axis=0)
    # 计算每个角点相对于质心的角度（使用反正切函数）
    angles = np.arctan2(corners_np[:, 1] - centroid[1], corners_np[:, 0] - centroid[0])
    # 根据角度进行排序（获取排序后的索引）
    sorted_indices = np.argsort(angles)
    # 根据排序后的索引返回排序好的角点坐标
    return corners_np[sorted_indices].tolist()

def labelme_to_yolo(json_dir, image_dir, output_dir):
    """
    将LabelMe标注的JSON文件转换为YOLO格式的文本文件

    :param json_dir: LabelMe标注的JSON文件所在目录
    :param image_dir: 与JSON文件对应的原始图像所在目录
    :param output_dir: 转换后YOLO格式文本文件输出目录
    """

    # 获取JSON文件列表
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r') as f:
            labelme_data = json.load(f)

        # 获取图像文件名和尺寸
        image_file = labelme_data['imagePath']
        image_path = os.path.join(image_dir, image_file)
        save_img_dir = os.path.join(output_dir,"images")
        os.makedirs(save_img_dir,exist_ok=True)
        save_img_path = os.path.join(save_img_dir,image_file)
        shutil.copy(image_path,save_img_path)
        image = Image.open(image_path)
        image_width, image_height = image.size

        # 确定类别列表，并为每个类别分配一个索引
        # class_names = list(set([obj['category_id'] for obj in labelme_data['shapes']]))
        # class_dict = {name: idx for idx, name in enumerate(class_names)}

        # 创建输出文件路径
        labels_dir =  os.path.join(output_dir,"labels")
        os.makedirs(labels_dir,exist_ok=True)
        output_file = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')

        save_json_dir = os.path.join(output_dir,"jsons")
        os.makedirs(save_json_dir,exist_ok=True)
        shutil.copy(json_path,save_json_dir)
        with open(output_file, 'w') as f:

            for obj in labelme_data['shapes']:
                # category_id = obj['category_id']
                # class_idx = class_dict[category_id]
                try:
                # 获取边界框坐标（LabelMe格式为左上角和右下角坐标）
                    x1, y1, x2, y2, x3,y3, x4, y4 = obj['points'][0][0], obj['points'][0][1], obj['points'][1][0], obj['points'][1][1],\
                                                        obj['points'][2][0], obj['points'][2][1], obj['points'][3][0], obj['points'][3][1]
                    
                    points_np = np.array(obj['points']).reshape((-1, 1, 2)).astype(np.float32)

                    (cx, cy), (w, h), angle = cv2.minAreaRect(points_np)
                    
                    cx,cy,w,h = cx/image_width,cy/image_height,w/image_width,h/image_height

                    # corners = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    # (x1, y1), (x2, y2), (x3,y3), (x4, y4) = sort_corners(corners)

                    # 计算中心坐标和归一化后的宽度、高度
                    # center_x = ((x1 + x2) / 2) / image_width
                    # center_y = ((y1 + y2) / 2) / image_height
                    # width = (x2 - x1) / image_width
                    # height = (y2 - y1) / image_height

                    gx1,gx2,gx3,gx4 = x1/image_width,x2/image_width,x3/image_width,x4/image_width
                    gy1,gy2,gy3,gy4 = y1/image_height,y2/image_height,y3/image_height,y4/image_height,

                    # 将数据按照YOLO格式写入文件
                    # f.write(f"0 {center_x} {center_y} {width} {height}\n")
                    label = obj["label"]
                    if label == "fp":
                        continue
                    label_indx = label_dict[label]
                    f.write(f"{label_indx} {gx1} {gy1} {gx2} {gy2} {gx3} {gy3} {gx4} {gy4}\n")
                    # f.write(f"{label_indx} {cx} {cy} {w} {h} {angle}\n")
                except Exception as e:
                    print(json_file)
                    print(e)
                    

# 设置LabelMe标注的JSON文件所在目录
json_dir = r'E:\code\data\tmp\to_train_obb'
# 设置与JSON文件对应的原始图像所在目录
image_dir = r'E:\code\data\tmp\to_train_obb'
# 设置转换后YOLO格式文本文件输出目录
output_dir = r'E:\code\data\gen_barcode_data\train_obb\20250102'
os.makedirs(output_dir,exist_ok=True)
labelme_to_yolo(json_dir, image_dir, output_dir)