import os
import json


def modify_json_files_in_folder(folder_path):
    """
    遍历文件夹下所有json文件，将其中shape_type字段的值修改为polygon
    :param folder_path: 文件夹路径
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for i in data["shapes"]:
                        if 'shape_type' in i.keys():
                            i['shape_type'] = 'polygon'
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)


if __name__ == "__main__":
    target_folder = r"C:\Users\Administrator\Desktop\to-do\box"
    modify_json_files_in_folder(target_folder)
    print("修改完成！")
    