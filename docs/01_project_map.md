# CE_Xiphoid_Project Project Map

## Project Goal

This project aims to build an AI-assisted image analysis pipeline for estimating chest expansion at the xiphoid process level using smartphone front and side images.

The object detection model detects only two ROI classes:

```text
0: xiphoid_roi
1: height_roi
```

The model does not directly predict chest circumference or chest expansion. After ROI detection, computer vision post-processing is used to extract pixel width and depth, convert pixels to centimeters using height ROI, estimate xiphoid-level circumference, and calculate chest expansion.

---

## Main Folder Structure

### `exports/`

This folder stores Label Studio export files.

Important file:

```text
old_4lev_filtered_2roi.json
```

This file was created from the previous 4-level ROI dataset by keeping only:

```text
xiphoid_roi
height_roi
```

The axillary and costal level ROI labels were removed.

---

### `frames_all/`

This folder stores all renamed image files.

It includes:

```text
S001–S019 legacy CE_Pilot images
N001–N009 newly collected xiphoid validation images
```

The filename rule is:

```text
subjectID_view_repetition_phase_frameNo.jpg
```

Example:

```text
S001_front_rep1_insp_01.jpg
N001_side_rep3_exp_02.jpg
```

The `S` subjects come from the previous CE_Pilot dataset.
The `N` subjects are newly collected external validation subjects.

---

### `scripts/`

This folder stores reusable pipeline scripts.

The goal is to avoid one-time terminal commands and make the whole workflow repeatable.

#### `01_validate_old_export.py`

This script validates the filtered Label Studio JSON file:

```text
exports/old_4lev_filtered_2roi.json
```

It checks:

```text
total number of tasks
label counts
whether each image has one xiphoid_roi and one height_roi
whether image files exist in frames_all
duplicate filenames
bad or incomplete tasks
```

This script should be run before converting annotations to YOLO format.

Command:

```bash
python scripts/01_validate_old_export.py
```

---

#### `02_convert_old_export_to_yolo.py`

This script converts the filtered Label Studio JSON annotation file into YOLO format.

Input:

```text
exports/old_4lev_filtered_2roi.json
frames_all/
```

Output:

```text
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── xiphoid_roi.yaml
```

The YOLO class definition is:

```text
0: xiphoid_roi
1: height_roi
```

Command:

```bash
python scripts/02_convert_old_export_to_yolo.py
```

---

#### `03_train_yolo_xiphoid.sh`

This shell script runs YOLO training using the converted YOLO dataset.

It trains a lightweight YOLO model, such as YOLOv8n, for xiphoid ROI and height ROI detection.

Input:

```text
yolo_dataset/xiphoid_roi.yaml
```

Output:

```text
models/xiphoid_yolov8n_old2roi/
```

Important model weights:

```text
models/xiphoid_yolov8n_old2roi/weights/best.pt
models/xiphoid_yolov8n_old2roi/weights/last.pt
```

Command:

```bash
bash scripts/03_train_yolo_xiphoid.sh
```

---

#### `04_prepare_external_N_images.py`

This script prepares the newly collected N001–N009 images for external validation.

It copies N-subject images from:

```text
frames_all/
```

to:

```text
external_validation/N_images/
```

The N001–N009 dataset is not used for model training. It is reserved for external validation against tape measurement data.

Command:

```bash
python scripts/04_prepare_external_N_images.py
```

---

#### `05_predict_external_N.sh`

This shell script runs the trained YOLO model on the N001–N009 external validation images.

Input:

```text
external_validation/N_images/
models/xiphoid_yolov8n_old2roi/weights/best.pt
```

Output:

```text
results/N_external_predictions_old2roi/
```

The output includes predicted images and YOLO text labels.

Command:

```bash
bash scripts/05_predict_external_N.sh
```

---

## Dataset Strategy

### Training Dataset

The model is trained using the filtered legacy dataset:

```text
old_4lev_filtered_2roi.json
```

This dataset contains only:

```text
xiphoid_roi
height_roi
```

The current plan is to train the object detection model using the legacy S-subject dataset.

---

### External Validation Dataset

The N001–N009 dataset is reserved for external validation.

These images are not used for model training.

The trained model will automatically predict:

```text
xiphoid_roi
height_roi
```

on the N001–N009 images.

The predicted ROIs will be used for post-processing to estimate xiphoid-level chest expansion. The AI-derived expansion values will then be compared with tape-measured expansion values.

---

## Tape Measurement Data

The N001–N009 subjects have tape measurement values at the xiphoid level.

Each subject has three repeated maximal inspiration and maximal expiration measurements.

Recommended tape data structure:

```csv
subject_id,level,repetition,phase,tape_cm,note
N001,xiphoid,rep1,insp,,
N001,xiphoid,rep1,exp,,
N001,xiphoid,rep2,insp,,
N001,xiphoid,rep2,exp,,
N001,xiphoid,rep3,insp,,
N001,xiphoid,rep3,exp,,
```

For each repetition:

```text
tape_expansion = tape_inspiration_cm - tape_expiration_cm
```

---

## Final Analysis Plan

For each N subject, the AI pipeline will calculate chest expansion from the predicted ROIs.

For each repetition:

```text
AI expansion = AI circumference at inspiration - AI circumference at expiration
Tape expansion = tape inspiration - tape expiration
```

Analysis will be performed at two levels:

```text
1. repetition-level analysis: 9 subjects × 3 repetitions = 27 paired values
2. subject-level mean analysis: 9 subjects = 9 paired values
```

Primary analysis:

```text
subject-level mean AI expansion vs subject-level mean tape expansion
```

Secondary analysis:

```text
repetition-level AI expansion vs repetition-level tape expansion
```

Planned statistics:

```text
mean difference
MAE
RMSE
Pearson correlation
Spearman correlation
Bland-Altman analysis
ICC, exploratory due to small sample size
```

---

## Important Notes

The N001–N009 dataset currently does not have ROI ground truth labels.

Therefore, detection mAP, precision, recall, and IoU cannot be calculated for the N dataset unless manual ROI ground truth is later created.

However, clinical agreement analysis can still be performed by comparing:

```text
AI-derived chest expansion
vs
tape-measured chest expansion
```

The N dataset serves as an external validation dataset for the full AI measurement pipeline.
