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

## Dataset Strategy

This project uses two sources of image data.

### 1. Legacy CE_Pilot dataset

The legacy dataset comes from the previous CE_Pilot project. The original images were already organized in `frames_all` using the following filename rule:

```text
subjectID_view_repetition_phase_frameNo.jpg
```

Example:

```text
S001_front_rep1_insp_01.jpg
```

The legacy dataset includes subjects `S001` to `S019`. However, `S019` has incomplete height information and will be excluded from final centimeter-based analysis. The main reusable legacy dataset is therefore defined as:

```text
S001–S018
```

For the current xiphoid-level project, only the xiphoid-level ROI and height ROI are needed. If the previous `lower_thorax_roi` is judged to correspond sufficiently to the xiphoid process level, it may be converted and reused as:

```text
lower_thorax_roi → xiphoid_roi
height_roi → height_roi
```

The previous upper and middle thorax ROI labels are not used in this one-level xiphoid project.

### 2. New xiphoid dataset

The new dataset consists of 9 newly collected subjects. Each subject has 24 images:

```text
front images: 12
side images: 12
total: 24 images per subject
```

Each view includes three repeated breathing trials:

```text
rep1 insp: 2 frames
rep1 exp: 2 frames
rep2 insp: 2 frames
rep2 exp: 2 frames
rep3 insp: 2 frames
rep3 exp: 2 frames
```

The new dataset is organized using the same filename rule:

```text
subjectID_view_repetition_phase_frameNo.jpg
```

Example:

```text
N001_front_rep1_insp_01.jpg
N001_side_rep3_exp_02.jpg
```

The new dataset is the primary dataset for comparison with tape measurements because it contains three repeated inspiration and expiration measurements at the xiphoid process level.

## Recommended Train / Validation / Test Strategy

The main goal of this side project is not only to train an object detection model, but also to test whether AI-derived xiphoid-level chest expansion values are similar to tape measurement values.

Therefore, the dataset should be split by subject, not by image. Images from the same subject must not be divided across training and validation/test sets.

The recommended practical strategy is:

```text
Model training / fine-tuning:
S001–S018
N001–N004

Final paper validation:
N005–N009

Excluded from final cm-based analysis:
S019
```

This structure allows the model to learn from the legacy dataset and a small portion of the new xiphoid dataset, while preserving five new subjects as an independent validation set for the final comparison with tape measurements.

An alternative, stricter strategy is:

```text
Model training:
S001–S018

External validation:
N001–N009
```

This stricter strategy is preferred if the model detects the xiphoid ROI and height ROI well on the new dataset without additional fine-tuning. If detection performance is unstable on the new data, `N001–N004` may be used for additional fine-tuning, while `N005–N009` should remain locked as the final validation set.

## Analysis Strategy

For each new subject, the AI pipeline will estimate xiphoid-level chest expansion using front and side images.

For each repetition:

```text
front image → W pixel at xiphoid level
side image → D pixel at xiphoid level
height ROI → cm/pixel conversion factor
W and D → ellipse-based circumference estimate
AI expansion = inspiration circumference - expiration circumference
```

The same repetition structure is used for tape measurements:

```text
Tape expansion rep1 = tape inspiration rep1 - tape expiration rep1
Tape expansion rep2 = tape inspiration rep2 - tape expiration rep2
Tape expansion rep3 = tape inspiration rep3 - tape expiration rep3
```

The final subject-level values are calculated as the mean of three repetitions:

```text
AI mean expansion = mean(AI rep1, AI rep2, AI rep3)
Tape mean expansion = mean(Tape rep1, Tape rep2, Tape rep3)
```

The agreement between AI-derived values and tape measurement values will be evaluated using descriptive error metrics and agreement analyses such as:

```text
mean difference
mean absolute error
RMSE
correlation
Bland-Altman analysis
ICC, if sample size is appropriate
```

## Important Rule

The object detection model predicts only the following ROI classes:

```text
0: xiphoid_roi
1: height_roi
```

The model does not directly predict chest circumference or chest expansion. Circumference and expansion are calculated later using computer vision post-processing and height-based pixel-to-centimeter conversion.


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

## Current Dataset Plan

The current `frames_all` folder contains 520 images.

```text
Total images: 520
```

The dataset consists of two sources.

```text
Legacy CE_Pilot dataset:
S001–S019

New xiphoid dataset:
N001–N009
```

The previous CE_Pilot dataset is not reused through the old `lower_thorax_roi` annotations. Instead, all images used in the current project will be newly labeled in Label Studio using the same two-class definition:

```text
0: xiphoid_roi
1: height_roi
```

This decision was made to keep the labeling criteria consistent across the legacy and new datasets.

## Labeling Plan

All images except `S019` will be labeled using the new xiphoid-level labeling rule.

```text
Labeling target:
S001–S018
N001–N009

Excluded:
S019
```

The total number of images for new labeling is:

```text
S001–S018: 288 images
N001–N009: 216 images
Total: 504 images
```

Each image will have exactly two bounding boxes:

```text
xiphoid_roi
height_roi
```

## Train and Final Validation Split

The practical dataset split for this project is:

```text
Training / fine-tuning:
S001–S018
N001–N004

Final paper validation:
N005–N009

Excluded from final centimeter-based analysis:
S019
```

This split keeps five newly collected subjects completely separate from model training. The final validation set will be used to compare AI-derived xiphoid-level chest expansion values with tape measurement values.

The split is subject-wise. Images from the same subject are not divided across training and validation sets.
