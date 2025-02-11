import torch
import cv2
import numpy as np
import time
from ultralytics import YOLO

# 定义类别名称列表（需根据你自己的数据集类别进行修改）
class_names = ["box"]  # 示例类别，按实际情况替换


# 加载模型

model = YOLO(r"E:\code\ultralytics\runs\pose\train\weights\last.pt") 

model.conf = 0.5  # 设置置信度阈值，可按需调整
model.iou = 0.45  # 设置交并比阈值，可按需调整
model.agnostic = False  # 是否启用类别不可知模式，可按需调整
model.multi_label = True  # 是否启用多标签模式，可按需调整
model.max_det = 1000  # 最大检测数量，可按需调整


def detect_obb(image_path):
    """
    使用YOLOv5进行obb推理的函数
    :param image_path: 待检测图像的路径
    :return: 检测结果（包含类别、置信度、旋转框等信息）以及绘制了检测结果的图像
    """
    # 读取图像
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    # 开始计时
    start_time = time.time()
    # 进行推理
    results = model(img)
    end_time = time.time()
    print(f"推理耗时: {end_time - start_time} 秒")


    for result in results:
        # print(result.obb)  # Print detection boxes
        result.show()  # Display the annotated image
        # result.save(filename="result.jpg")  # Save annotated image

    # return obb_results, img


if __name__ == "__main__":
    # image_path = r"C:\Users\Administrator\Desktop\to-do-10\box\202412021442353401_S.jpg"  # angle=0
    image_path = r"E:\code\data\tmp\to_train\202412061412499346_S_[7,7].jpg"  # angle=45
    detect_obb(image_path)
    # cv2.imshow("Detection Results", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()