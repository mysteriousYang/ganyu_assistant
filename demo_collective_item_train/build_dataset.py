import os
import random
import shutil

# 定义数据集路径和划分比例
dataset_dir = "./GENSHIN/"
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 创建训练、测试和验证集的子目录
os.makedirs(dataset_dir + "train/images", exist_ok=True)
os.makedirs(dataset_dir + "train/labels", exist_ok=True)
os.makedirs(dataset_dir + "val/images", exist_ok=True)
os.makedirs(dataset_dir + "val/labels", exist_ok=True)
os.makedirs(dataset_dir + "test/images", exist_ok=True)
os.makedirs(dataset_dir + "test/labels", exist_ok=True)

# 获取所有图像文件名
image_files = os.listdir(dataset_dir + "images")

# 随机划分数据集
random.shuffle(image_files)
train_split = int(len(image_files) * train_ratio)
val_split = int(len(image_files) * (train_ratio + val_ratio))

# 移动图像和标签文件到对应的子目录
for i, image_file in enumerate(image_files):
    label_file = os.path.splitext(image_file)[0] + ".txt"
    if i < train_split:
        shutil.move(dataset_dir + "images/" + image_file, dataset_dir + "train/images/" + image_file)
        shutil.move(dataset_dir + "labels/" + label_file, dataset_dir + "train/labels/" + label_file)
    elif i < val_split:
        shutil.move(dataset_dir + "images/" + image_file, dataset_dir + "val/images/" + image_file)
        shutil.move(dataset_dir + "labels/" + label_file, dataset_dir + "val/labels/" + label_file)
    else:
        shutil.move(dataset_dir + "images/" + image_file, dataset_dir + "test/images/" + image_file)
        shutil.move(dataset_dir + "labels/" + label_file, dataset_dir + "test/labels/" + label_file)

# 读取类别列表
with open(dataset_dir + "classes.txt", "r") as f:
    classes = f.read().split("\n")

# 生成YOLO格式的数据集配置文件(data.yaml)
with open(dataset_dir + "GENSHIN.yaml", "w") as f:
    f.write(f"train: {dataset_dir}train/images\n")
    f.write(f"val: {dataset_dir}val/images\n")
    f.write(f"test: {dataset_dir}test/images\n")
    f.write(f"nc: {len(classes)}\n")
    f.write(f"names: {classes}\n")

print("数据集划分和配置文件生成完成!")