#-*- encoding utf-8 -*-
import os
import random
import shutil

train_rate = 0.8

sample_list = os.listdir(".\\labels")
sample_list.remove("predefined_classes.txt")
sample_list.remove("classes.txt")

sample_count = len(sample_list)
random.shuffle(sample_list)

train_end = int(train_rate*sample_count)
step = int(sample_count * (1 - train_rate) / 2)
train_sample = sample_list[:train_end]
test_sample = sample_list[train_end:train_end+step]
val_sample = sample_list[train_end+step:]

current_path = os.path.dirname(os.path.abspath(__file__))
# current_path = os.getcwd()
# print(os.path.join(current_path,"build_dataset.py"))

if not(os.path.exists(".\\dataset")):
    os.mkdir(".\\dataset")
    print("Made dir: dataset")

for dir in ["train","test","val"]:
    path = os.path.join(".\\dataset",dir)
    if not (os.path.exists(path)):
        os.mkdir(path)
        print("Made dir: ",path)
    if not (os.path.exists(os.path.join(path,"images"))):
        os.mkdir(os.path.join(path,"images"))
        print("Made dir: images")
    if not (os.path.exists(os.path.join(path,"labels"))):
        os.mkdir(os.path.join(path,"labels"))
        print("Made dir: images")

# copy train samples
print("Building train dir")
for sample in train_sample:
    shutil.copyfile(
        os.path.join(".\\images",sample.replace(".txt",".jpg")),
        os.path.join(".\\dataset\\train\\images",sample.replace(".txt",".jpg"))
    )

    shutil.copyfile(
        os.path.join(".\\labels",sample),
        os.path.join(".\\dataset\\train\\labels",sample)
    )


# copy test samples
print("Building test dir")
for sample in test_sample:
    shutil.copyfile(
        os.path.join(".\\images",sample.replace(".txt",".jpg")),
        os.path.join(".\\dataset\\test\\images",sample.replace(".txt",".jpg"))
    )

    shutil.copyfile(
        os.path.join(".\\labels",sample),
        os.path.join(".\\dataset\\test\\labels",sample)
    )


# copy val samples
print("Building val dir")
for sample in val_sample:
    shutil.copyfile(
        os.path.join(".\\images",sample.replace(".txt",".jpg")),
        os.path.join(".\\dataset\\val\\images",sample.replace(".txt",".jpg"))
    )

    shutil.copyfile(
        os.path.join(".\\labels",sample),
        os.path.join(".\\dataset\\val\\labels",sample)
    )

#build yaml
print("Building yaml")
with open(os.path.join(".\\dataset","GENSHIN.yaml"),mode="w",encoding="utf-8") as fout:
    fout.write("train: ")
    fout.write(current_path + "\\dataset\\train\\images")
    fout.write('\n')

    fout.write("test: ")
    fout.write(current_path + "\\dataset\\test\\images")
    fout.write('\n')

    fout.write("val: ")
    fout.write(current_path + "\\dataset\\val\\images")
    fout.write('\n')

    with open(".\\predefined_classes.txt","r") as class_fp:
        classes = class_fp.read()
    classes = classes.splitlines()

    fout.write('\n')
    fout.write("nc: ")
    fout.write(str(len(classes)))
    fout.write('\n')

    fout.write('\n')
    fout.write("names: ")
    fout.write(str(classes))

print("Complete")