from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.home() / "Desktop" / "CE_Xiphoid_Project"

IMAGE_LEVEL_CSV = (
    PROJECT_ROOT
    / "results"
    / "N_final_measurements"
    / "csv"
    / "N_image_level_cm_measurements.csv"
)

CONDITION_CSV = (
    PROJECT_ROOT
    / "results"
    / "N_final_measurements"
    / "csv"
    / "N_condition_level_measurements.csv"
)

PHASE_CSV = (
    PROJECT_ROOT
    / "results"
    / "N_final_measurements"
    / "csv"
    / "N_phase_level_circumference.csv"
)

REPETITION_CSV = (
    PROJECT_ROOT
    / "results"
    / "N_final_measurements"
    / "csv"
    / "N_repetition_level_ai_expansion.csv"
)

def show(title, df):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    if df.empty:
        print("EMPTY")
    else:
        print(df.to_string(index=False))

def main():
    subject_id = "N001"
    repetition = "rep2"

    image_df = pd.read_csv(IMAGE_LEVEL_CSV)
    condition_df = pd.read_csv(CONDITION_CSV)
    phase_df = pd.read_csv(PHASE_CSV)
    repetition_df = pd.read_csv(REPETITION_CSV)

    image_sub = image_df[
        (image_df["subject_id"] == subject_id)
        & (image_df["repetition"] == repetition)
    ].copy()

    condition_sub = condition_df[
        (condition_df["subject_id"] == subject_id)
        & (condition_df["repetition"] == repetition)
    ].copy()

    phase_sub = phase_df[
        (phase_df["subject_id"] == subject_id)
        & (phase_df["repetition"] == repetition)
    ].copy()

    repetition_sub = repetition_df[
        (repetition_df["subject_id"] == subject_id)
        & (repetition_df["repetition"] == repetition)
    ].copy()

    image_columns = [
        "filename",
        "view",
        "phase",
        "frame_no",
        "measurement_status",
        "measurement_px",
        "measurement_cm",
        "W_px",
        "D_px",
        "W_cm",
        "D_cm",
        "xiphoid_roi_source",
        "height_roi_source",
        "image_conversion_status",
    ]

    image_columns = [c for c in image_columns if c in image_sub.columns]

    condition_columns = [
        "subject_id",
        "view",
        "repetition",
        "phase",
        "n_valid_frames",
        "measurement_px_median",
        "measurement_cm_median",
        "condition_measurement_px",
        "condition_measurement_cm",
        "predicted_roi_frames",
        "ground_truth_fallback_frames",
        "condition_status",
    ]

    condition_columns = [c for c in condition_columns if c in condition_sub.columns]

    phase_columns = [
        "subject_id",
        "repetition",
        "phase",
        "W_px",
        "D_px",
        "W_cm",
        "D_cm",
        "ellipse_circumference_cm",
        "phase_status",
    ]

    phase_columns = [c for c in phase_columns if c in phase_sub.columns]

    show(
        "IMAGE LEVEL: N001 rep2",
        image_sub[image_columns],
    )

    show(
        "CONDITION LEVEL: N001 rep2",
        condition_sub[condition_columns],
    )

    show(
        "PHASE LEVEL: N001 rep2",
        phase_sub[phase_columns],
    )

    show(
        "REPETITION LEVEL: N001 rep2",
        repetition_sub,
    )

if __name__ == "__main__":
    main()