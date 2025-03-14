# -*- coding: utf-8 -*-
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO(r".\runs\detect\train6\weights\best.pt")
# model = YOLO("yolo11m.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("ganyu3.png", save=True, imgsz=320, conf=0.5, show=True)
# input()