import os
import xml.etree.ElementTree as ET
from PIL import Image

# 定义PascalVOC数据集和YOLO数据集的路径
pascal_voc_dir = "./"
yolo_dataset_dir = "./GENSHIN/"

# 创建YOLO数据集的子目录(如果不存在)
os.makedirs(yolo_dataset_dir + "labels", exist_ok=True)
os.makedirs(yolo_dataset_dir + "images", exist_ok=True)

# 读取预定义的类别列表
with open(pascal_voc_dir + "label/predefined_classes.txt", "r") as f:
    classes = f.read().split("\n")

# 遍历PascalVOC数据集的XML标签文件
for xml_file in os.listdir(pascal_voc_dir + "label"):
    if not xml_file.endswith(".xml"):
        continue

    # 解析XML文件
    tree = ET.parse(pascal_voc_dir + "label/" + xml_file)
    root = tree.getroot()

    # 获取图像文件名和尺寸
    image_file = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    # 打开对应的图像文件
    with Image.open(pascal_voc_dir + "image/" + image_file) as img:
        # 将图像文件复制到YOLO数据集的images目录
        img.save(yolo_dataset_dir + "images/" + image_file)

    # 创建YOLO格式的标签文件
    with open(yolo_dataset_dir + "labels/" + os.path.splitext(xml_file)[0] + ".txt", "w") as f:
        for obj in root.findall("object"):
            print(obj)

            # 获取边界框坐标
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            # 转换为YOLO格式的坐标和尺寸
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            # 获取类别ID
            class_id = classes.index(obj.find("name").text)

            # 写入YOLO格式的标签
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

print("转换完成!")