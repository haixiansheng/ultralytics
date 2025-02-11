import json
import os
from PIL import Image
import shutil
import numpy as np
import cv2

label_dict = {"neck":0,"left_front_leg":1, "right_front_leg":2, "left_rear_leg":3, "right_rear_leg":4, "middle":5, "left_pigu":6, "middle_pigu":7, "right_pigu":8}

"""  0: neck
  1: left_front_leg
  2: right_front_leg
  3: left_rear_leg
  4: right_rear_leg
  5: middle
  6: left_pigu
  7: middle_pigu
  8: right_pigu
"""


def order_box_corners(corners):
    """
    对给定的方盒四个角点坐标进行顺序整理
    :param corners: 包含四个角点坐标的列表，每个角点坐标为[x, y]形式的列表
    :return: 按照特定顺序整理后的角点坐标列表
    """
    # 将角点坐标转换为numpy数组方便计算
    corners_np = np.array(corners)
    # 计算每个角点到原点（这里简单以第一个角点所在位置假设为临时原点）的距离
    origin = corners_np[0]
    distances = np.linalg.norm(corners_np - origin, axis=1)
    # 根据距离从小到大排序，获取排序后的索引
    sorted_indices = np.argsort(distances)
    # 按照排序后的索引重新排列角点坐标
    ordered_corners = corners_np[sorted_indices].tolist()
    return ordered_corners

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
                if obj["shape_type"] == "rectangle":
                        x1, y1, x2, y2 = obj['points'][0][0], obj['points'][0][1], obj['points'][2][0], obj['points'][2][1]

                        center_x = ((x1 + x2) / 2) / image_width
                        center_y = ((y1 + y2) / 2) / image_height
                        width = (x2 - x1) / image_width
                        height = (y2 - y1) / image_height
                        result = f"0 {center_x} {center_y} {width} {height}"

            for l in label_dict.keys():
                for obj in labelme_data['shapes']:
                    # category_id = obj['category_id']
                    # class_idx = class_dict[category_id] 
                    if obj["label"]==l:
                        x1,y1 = obj['points'][0][0],obj['points'][0][1]
                        x = x1  / image_width
                        y = y1 / image_height
                        result += f" {x} {y}"
            result += "\n"
            f.write(result)
                    

# 设置LabelMe标注的JSON文件所在目录
json_dir = r'E:\code\data\tmp\to_train_cow'
# 设置与JSON文件对应的原始图像所在目录
image_dir = r'E:\code\data\tmp\to_train_cow'
# 设置转换后YOLO格式文本文件输出目录
output_dir = r'E:\code\data\training_dataset\cow_pose\20241231'
os.makedirs(output_dir,exist_ok=True)
labelme_to_yolo(json_dir, image_dir, output_dir)