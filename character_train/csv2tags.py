#-*- coding utf-8 -*-
import os

with open(".\\label\\predefined_classes.txt",encoding="utf-8",mode="w") as txt_fp:
    with open(".\\character_tag.csv",mode="r") as csv_fp:
        for line in csv_fp.readlines():
            txt_fp.write(line)
    with open(".\\party_tag.csv",mode="r") as csv_fp:
        for line in csv_fp.readlines():
            txt_fp.write(line)