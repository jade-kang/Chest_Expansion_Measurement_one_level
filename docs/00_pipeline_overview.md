# Pipeline Overview

## Project

CE_Xiphoid_Project

## Goal

Detect xiphoid-level ROI and height ROI from smartphone front/side images, extract pixel width/depth, convert pixels to centimeters, estimate xiphoid-level chest circumference and expansion, and compare with tape measurement data.

## Folder Structure

- raw_videos/: original videos
- frames_raw/: raw captured images by subject/view
- frames_all/: renamed unified images
- metadata/: frame, subject, and tape measurement CSV files
- labelstudio_project/: Label Studio task files and interface settings
- exports/: Label Studio exported JSON files
- yolo_dataset/: YOLO training dataset
- ssd_dataset/: MobileNet SSD/COCO dataset
- models/: trained model weights
- results/: prediction and analysis outputs
- scripts/: Python scripts
- docs/: project logs and notes

## Class Definition

0: xiphoid_roi  
1: height_roi
