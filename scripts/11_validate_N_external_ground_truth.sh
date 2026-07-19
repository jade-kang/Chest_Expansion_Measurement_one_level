#!/bin/bash

cd /Users/kangjun2000/Desktop/CE_Xiphoid_Project

yolo detect val \
  model=runs/detect/models/xiphoid_yolov8n_old2roi/weights/best.pt \
  data=external_test_dataset/external_test.yaml \
  split=test \
  imgsz=640 \
  conf=0.001 \
  iou=0.7 \
  plots=True \
  save_json=True \
  project=results \
  name=N_external_ground_truth_validation