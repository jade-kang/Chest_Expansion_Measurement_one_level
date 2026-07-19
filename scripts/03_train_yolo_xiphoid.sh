#!/bin/bash

cd /Users/kangjun2000/Desktop/CE_Xiphoid_Project

yolo detect train \
  model=yolov8n.pt \
  data=yolo_dataset/xiphoid_roi.yaml \
  imgsz=640 \
  epochs=100 \
  batch=8 \
  project=models \
  name=xiphoid_yolov8n_old2roi