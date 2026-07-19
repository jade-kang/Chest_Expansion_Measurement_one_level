# Xiphoid ROI Measurement Pipeline

## Goal

This pipeline estimates xiphoid-level chest circumference and chest expansion from front and side smartphone images.

## Main Steps

1. Train ROI detection model using legacy S dataset
2. Validate ROI detection on N external ground truth dataset
3. Predict xiphoid and height ROI on N images
4. Use ground-truth fallback when predicted ROI is missing
5. Apply computer vision segmentation inside xiphoid ROI
6. Extract front width W_px and side depth D_px
7. Convert pixels to centimeters using subject-view height scale
8. Approximate chest cross-section as an ellipse
9. Calculate inspiration and expiration circumference
10. Calculate chest expansion

## Key Outputs

- `N_image_level_pixel_measurements.csv`
- `N_subject_view_scales.csv`
- `N_repetition_level_ai_expansion.csv`
- `N_AI_repetition_detailed.csv`
- `N_AI_tape_format.csv`

## Final AI Table Format

`N_AI_tape_format.csv` has the same structure as the tape measurement table:

```csv
Participant_ID,Inhale_cm_1,Exhale_cm_1,Inhale_cm_2,Exhale_cm_2,Inhale_cm_3,Exhale_cm_3,CE_1,CE_2,CE_3