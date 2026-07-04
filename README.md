# CE Xiphoid Project

## Purpose

This project develops an AI-assisted image analysis pipeline for estimating chest expansion at the xiphoid process level using smartphone front and side images.

The object detection model detects:

- xiphoid_roi
- height_roi

The model does not directly predict chest circumference or chest expansion. After ROI detection, computer vision post-processing is used to extract pixel width/depth, convert pixels to centimeters using height ROI, estimate circumference, and compare the result with tape measurement data.

## Dataset

New subjects:

- front images: 12 per subject
- side images: 12 per subject
- total images: 24 per subject

Each view includes:

- rep1 insp 2 frames
- rep1 exp 2 frames
- rep2 insp 2 frames
- rep2 exp 2 frames
- rep3 insp 2 frames
- rep3 exp 2 frames

## Classes

0: xiphoid_roi  
1: height_roi

## Main Pipeline

1. Organize captured images
2. Generate frame_metadata.csv
3. Label xiphoid_roi and height_roi in Label Studio
4. Export Label Studio JSON
5. Convert annotations to YOLO and COCO datasets
6. Train YOLO lightweight model
7. Train MobileNet SSD comparison model
8. Run inference on test subjects
9. Extract pixel width/depth from ROI
10. Convert pixels to centimeters using height ROI
11. Estimate chest circumference
12. Compare AI-based results with tape measurements
