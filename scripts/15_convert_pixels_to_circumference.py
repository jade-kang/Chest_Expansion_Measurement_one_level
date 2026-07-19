from pathlib import Path
import math

import numpy as np
import pandas as pd


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = (
    Path.home()
    / "Desktop"
    / "CE_Xiphoid_Project"
)

IMAGE_MEASUREMENT_CSV = (
    PROJECT_ROOT
    / "results"
    / "N_cv_measurements"
    / "csv"
    / "N_image_level_pixel_measurements.csv"
)

SUBJECT_METADATA_CSV = (
    PROJECT_ROOT
    / "metadata"
    / "N_subject_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "N_final_measurements"
    / "csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 2. ANALYSIS SETTINGS
# =========================================================

# 주 분석에는 subject × view별 height pixel 중앙값 사용
HEIGHT_SCALE_STATISTIC = "median"

# 같은 조건에 프레임이 여러 장 있으면 중앙값 사용
FRAME_AGGREGATION = "median"


# =========================================================
# 3. ELLIPSE CIRCUMFERENCE
# =========================================================

def ellipse_circumference(
    width_cm,
    depth_cm,
):
    """
    Ramanujan approximation.

    width_cm:
        front image에서 얻은 전체 좌우 폭

    depth_cm:
        side image에서 얻은 전체 전후 깊이

    a:
        반장축 = width / 2

    b:
        반단축 = depth / 2
    """

    if pd.isna(width_cm):
        return np.nan

    if pd.isna(depth_cm):
        return np.nan

    if width_cm <= 0:
        return np.nan

    if depth_cm <= 0:
        return np.nan

    a = width_cm / 2.0
    b = depth_cm / 2.0

    h = (
        (a - b) ** 2
        /
        (a + b) ** 2
    )

    circumference = (
        math.pi
        *
        (a + b)
        *
        (
            1
            +
            (
                3 * h
                /
                (
                    10
                    +
                    math.sqrt(
                        4
                        -
                        3 * h
                    )
                )
            )
        )
    )

    return circumference


# =========================================================
# 4. LOAD DATA
# =========================================================

def load_measurement_data():

    if not IMAGE_MEASUREMENT_CSV.exists():

        raise FileNotFoundError(
            f"\nMeasurement CSV not found:\n"
            f"{IMAGE_MEASUREMENT_CSV}"
        )

    df = pd.read_csv(
        IMAGE_MEASUREMENT_CSV
    )

    print(
        f"Image measurement rows: "
        f"{len(df)}"
    )

    if (
        "measurement_status"
        in
        df.columns
    ):

        print()
        print(
            "Measurement status:"
        )

        print(
            df[
                "measurement_status"
            ]
            .value_counts(
                dropna=False
            )
        )

    return df


def load_subject_metadata():

    if not SUBJECT_METADATA_CSV.exists():

        raise FileNotFoundError(
            f"\nSubject metadata CSV "
            f"not found:\n"
            f"{SUBJECT_METADATA_CSV}"
        )

    metadata = pd.read_csv(
        SUBJECT_METADATA_CSV
    )

    required_columns = {
        "subject_id",
        "height_cm",
    }

    missing_columns = (
        required_columns
        -
        set(
            metadata.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "Missing metadata columns: "
            f"{missing_columns}"
        )

    metadata[
        "height_cm"
    ] = pd.to_numeric(
        metadata[
            "height_cm"
        ],
        errors="coerce",
    )

    return metadata


# =========================================================
# 5. SUBJECT × VIEW HEIGHT SCALE
# =========================================================

def make_subject_view_scale_table(
    image_df,
    metadata_df,
):

    height_rows = (
        image_df
        .copy()
    )

    height_rows[
        "height_px_raw"
    ] = pd.to_numeric(
        height_rows[
            "height_px_raw"
        ],
        errors="coerce",
    )

    height_rows = (
        height_rows[
            height_rows[
                "height_px_raw"
            ]
            .notna()
            &
            (
                height_rows[
                    "height_px_raw"
                ]
                >
                0
            )
        ]
        .copy()
    )

    scale_table = (
        height_rows
        .groupby(
            [
                "subject_id",
                "view",
            ],
            as_index=False,
        )
        .agg(
            height_px_median=(
                "height_px_raw",
                "median",
            ),

            height_px_mean=(
                "height_px_raw",
                "mean",
            ),

            height_px_std=(
                "height_px_raw",
                "std",
            ),

            height_px_min=(
                "height_px_raw",
                "min",
            ),

            height_px_max=(
                "height_px_raw",
                "max",
            ),

            n_valid_height_images=(
                "height_px_raw",
                "count",
            ),
        )
    )

    scale_table = (
        scale_table
        .merge(
            metadata_df[
                [
                    "subject_id",
                    "height_cm",
                ]
            ],
            on="subject_id",
            how="left",
        )
    )

    scale_table[
        "front_or_side"
    ] = scale_table[
        "view"
    ]

    scale_table[
        "cm_per_px_recommended"
    ] = (
        scale_table[
            "height_cm"
        ]
        /
        scale_table[
            "height_px_median"
        ]
    )

    scale_table[
        "scale_method"
    ] = (
        "subject_view_"
        "median_height_px"
    )

    scale_table[
        "scale_qc_status"
    ] = np.where(
        scale_table[
            "height_cm"
        ].isna(),

        "missing_height_cm",

        np.where(
            scale_table[
                "n_valid_height_images"
            ]
            <
            2,

            "limited_height_images",

            "pass",
        ),
    )

    return scale_table


# =========================================================
# 6. IMAGE-LEVEL CM CONVERSION
# =========================================================

def make_image_level_cm_table(
    image_df,
    scale_table,
):

    df = image_df.copy()

    df[
        "width_median_px"
    ] = pd.to_numeric(
        df[
            "width_median_px"
        ],
        errors="coerce",
    )

    merge_scale = (
        scale_table[
            [
                "subject_id",
                "view",
                "height_cm",
                "height_px_median",
                "cm_per_px_recommended",
                "n_valid_height_images",
                "scale_method",
                "scale_qc_status",
            ]
        ]
    )

    df = df.merge(
        merge_scale,
        on=[
            "subject_id",
            "view",
        ],
        how="left",
    )

    df[
        "measurement_px"
    ] = df[
        "width_median_px"
    ]

    df[
        "measurement_cm"
    ] = (
        df[
            "measurement_px"
        ]
        *
        df[
            "cm_per_px_recommended"
        ]
    )

    df[
        "W_px"
    ] = np.where(
        df[
            "view"
        ]
        ==
        "front",

        df[
            "measurement_px"
        ],

        np.nan,
    )

    df[
        "D_px"
    ] = np.where(
        df[
            "view"
        ]
        ==
        "side",

        df[
            "measurement_px"
        ],

        np.nan,
    )

    df[
        "front_cm_per_px"
    ] = np.where(
        df[
            "view"
        ]
        ==
        "front",

        df[
            "cm_per_px_recommended"
        ],

        np.nan,
    )

    df[
        "side_cm_per_px"
    ] = np.where(
        df[
            "view"
        ]
        ==
        "side",

        df[
            "cm_per_px_recommended"
        ],

        np.nan,
    )

    df[
        "W_cm"
    ] = np.where(
        df[
            "view"
        ]
        ==
        "front",

        df[
            "measurement_cm"
        ],

        np.nan,
    )

    df[
        "D_cm"
    ] = np.where(
        df[
            "view"
        ]
        ==
        "side",

        df[
            "measurement_cm"
        ],

        np.nan,
    )

    df[
        "image_conversion_status"
    ] = np.select(

        [

            df[
                "measurement_status"
            ]
            !=
            "success",

            df[
                "measurement_px"
            ]
            .isna(),

            df[
                "cm_per_px_recommended"
            ]
            .isna(),

        ],

        [

            "measurement_failed",

            "pixel_value_missing",

            "scale_missing",

        ],

        default="success",
    )

    return df


# =========================================================
# 7. CONDITION-LEVEL FRAME AGGREGATION
# =========================================================

def make_condition_level_table(
    image_cm_df,
):

    usable = (
        image_cm_df[
            (
                image_cm_df[
                    "image_conversion_status"
                ]
                ==
                "success"
            )
            &
            image_cm_df[
                "measurement_cm"
            ]
            .notna()
        ]
        .copy()
    )

    condition_table = (
        usable
        .groupby(
            [
                "subject_id",
                "view",
                "repetition",
                "phase",
            ],
            as_index=False,
        )
        .agg(

            n_valid_frames=(
                "filename",
                "nunique",
            ),

            measurement_px_median=(
                "measurement_px",
                "median",
            ),

            measurement_px_mean=(
                "measurement_px",
                "mean",
            ),

            measurement_px_std=(
                "measurement_px",
                "std",
            ),

            measurement_cm_median=(
                "measurement_cm",
                "median",
            ),

            measurement_cm_mean=(
                "measurement_cm",
                "mean",
            ),

            measurement_cm_std=(
                "measurement_cm",
                "std",
            ),

            cm_per_px=(
                "cm_per_px_recommended",
                "median",
            ),

            height_px_median=(
                "height_px_median",
                "median",
            ),

            height_cm=(
                "height_cm",
                "median",
            ),

            predicted_roi_frames=(
                "xiphoid_roi_source",
                lambda values:
                int(
                    (
                        values
                        ==
                        "predicted"
                    )
                    .sum()
                ),
            ),

            ground_truth_fallback_frames=(
                "xiphoid_roi_source",
                lambda values:
                int(
                    (
                        values
                        ==
                        "ground_truth_fallback"
                    )
                    .sum()
                ),
            ),

        )
    )

    condition_table[
        "measurement_symbol"
    ] = np.where(
        condition_table[
            "view"
        ]
        ==
        "front",

        "W",

        "D",
    )

    condition_table[
        "condition_measurement_px"
    ] = condition_table[
        "measurement_px_median"
    ]

    condition_table[
        "condition_measurement_cm"
    ] = condition_table[
        "measurement_cm_median"
    ]

    condition_table[
        "condition_status"
    ] = np.where(
        condition_table[
            "n_valid_frames"
        ]
        >=
        2,

        "complete_2_frames",

        "single_frame_only",
    )

    return condition_table


# =========================================================
# 8. PHASE-LEVEL FRONT + SIDE MERGE
# =========================================================

def make_phase_level_table(
    condition_table,
):

    front = (
        condition_table[
            condition_table[
                "view"
            ]
            ==
            "front"
        ]
        .copy()
    )

    side = (
        condition_table[
            condition_table[
                "view"
            ]
            ==
            "side"
        ]
        .copy()
    )

    front = front.rename(
        columns={

            "n_valid_frames":
                "front_valid_frames",

            "condition_measurement_px":
                "W_px",

            "condition_measurement_cm":
                "W_cm",

            "cm_per_px":
                "front_cm_per_px",

            "height_px_median":
                "front_height_px_median",

            "predicted_roi_frames":
                "front_predicted_roi_frames",

            "ground_truth_fallback_frames":
                "front_gt_fallback_frames",

            "condition_status":
                "front_condition_status",
        }
    )

    side = side.rename(
        columns={

            "n_valid_frames":
                "side_valid_frames",

            "condition_measurement_px":
                "D_px",

            "condition_measurement_cm":
                "D_cm",

            "cm_per_px":
                "side_cm_per_px",

            "height_px_median":
                "side_height_px_median",

            "predicted_roi_frames":
                "side_predicted_roi_frames",

            "ground_truth_fallback_frames":
                "side_gt_fallback_frames",

            "condition_status":
                "side_condition_status",
        }
    )

    front_keep = [

        "subject_id",
        "repetition",
        "phase",

        "front_valid_frames",

        "W_px",
        "W_cm",

        "front_cm_per_px",
        "front_height_px_median",

        "front_predicted_roi_frames",
        "front_gt_fallback_frames",

        "front_condition_status",
    ]

    side_keep = [

        "subject_id",
        "repetition",
        "phase",

        "side_valid_frames",

        "D_px",
        "D_cm",

        "side_cm_per_px",
        "side_height_px_median",

        "side_predicted_roi_frames",
        "side_gt_fallback_frames",

        "side_condition_status",
    ]

    phase_table = (
        front[
            front_keep
        ]
        .merge(
            side[
                side_keep
            ],
            on=[
                "subject_id",
                "repetition",
                "phase",
            ],
            how="outer",
        )
    )

    phase_table[
        "ellipse_circumference_cm"
    ] = phase_table.apply(

        lambda row:
        ellipse_circumference(
            width_cm=row[
                "W_cm"
            ],
            depth_cm=row[
                "D_cm"
            ],
        ),

        axis=1,
    )

    phase_table[
        "phase_status"
    ] = np.where(

        phase_table[
            "W_cm"
        ]
        .notna()
        &
        phase_table[
            "D_cm"
        ]
        .notna(),

        "circumference_available",

        "missing_front_or_side",
    )

    return phase_table


# =========================================================
# 9. REPETITION-LEVEL EXPANSION
# =========================================================

def make_repetition_level_table(
    phase_table,
):

    insp = (
        phase_table[
            phase_table[
                "phase"
            ]
            ==
            "insp"
        ]
        .copy()
    )

    exp = (
        phase_table[
            phase_table[
                "phase"
            ]
            ==
            "exp"
        ]
        .copy()
    )

    insp = insp.rename(
        columns={

            "W_px":
                "W_insp_px",

            "D_px":
                "D_insp_px",

            "W_cm":
                "W_insp_cm",

            "D_cm":
                "D_insp_cm",

            "ellipse_circumference_cm":
                "circumference_insp_cm",

            "front_valid_frames":
                "front_insp_valid_frames",

            "side_valid_frames":
                "side_insp_valid_frames",

            "phase_status":
                "insp_status",
        }
    )

    exp = exp.rename(
        columns={

            "W_px":
                "W_exp_px",

            "D_px":
                "D_exp_px",

            "W_cm":
                "W_exp_cm",

            "D_cm":
                "D_exp_cm",

            "ellipse_circumference_cm":
                "circumference_exp_cm",

            "front_valid_frames":
                "front_exp_valid_frames",

            "side_valid_frames":
                "side_exp_valid_frames",

            "phase_status":
                "exp_status",
        }
    )

    insp_keep = [

        "subject_id",
        "repetition",

        "W_insp_px",
        "D_insp_px",

        "W_insp_cm",
        "D_insp_cm",

        "circumference_insp_cm",

        "front_insp_valid_frames",
        "side_insp_valid_frames",

        "insp_status",
    ]

    exp_keep = [

        "subject_id",
        "repetition",

        "W_exp_px",
        "D_exp_px",

        "W_exp_cm",
        "D_exp_cm",

        "circumference_exp_cm",

        "front_exp_valid_frames",
        "side_exp_valid_frames",

        "exp_status",
    ]

    repetition_table = (
        insp[
            insp_keep
        ]
        .merge(
            exp[
                exp_keep
            ],
            on=[
                "subject_id",
                "repetition",
            ],
            how="outer",
        )
    )

    repetition_table[
        "ai_expansion_cm"
    ] = (

        repetition_table[
            "circumference_insp_cm"
        ]

        -

        repetition_table[
            "circumference_exp_cm"
        ]
    )

    repetition_table[
        "repetition_status"
    ] = np.where(

        repetition_table[
            "circumference_insp_cm"
        ]
        .notna()

        &

        repetition_table[
            "circumference_exp_cm"
        ]
        .notna(),

        "complete",

        "incomplete",
    )

    return repetition_table


# =========================================================
# 10. SUBJECT-LEVEL SUMMARY
# =========================================================

def make_subject_level_table(
    repetition_table,
):

    complete = (
        repetition_table[
            repetition_table[
                "repetition_status"
            ]
            ==
            "complete"
        ]
        .copy()
    )

    if complete.empty:

        return pd.DataFrame(
            columns=[

                "subject_id",

                "valid_repetitions",

                "ai_mean_expansion_cm",

                "ai_sd_expansion_cm",

                "ai_median_expansion_cm",

                "ai_min_expansion_cm",

                "ai_max_expansion_cm",

                "analysis_status",
            ]
        )

    subject_table = (
        complete
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(

            valid_repetitions=(
                "ai_expansion_cm",
                "count",
            ),

            ai_mean_expansion_cm=(
                "ai_expansion_cm",
                "mean",
            ),

            ai_sd_expansion_cm=(
                "ai_expansion_cm",
                "std",
            ),

            ai_median_expansion_cm=(
                "ai_expansion_cm",
                "median",
            ),

            ai_min_expansion_cm=(
                "ai_expansion_cm",
                "min",
            ),

            ai_max_expansion_cm=(
                "ai_expansion_cm",
                "max",
            ),
        )
    )

    subject_table[
        "analysis_status"
    ] = subject_table[
        "valid_repetitions"
    ].map(

        lambda count:

        "complete_3reps"
        if count >= 3

        else (

            "usable_2reps"
            if count == 2

            else
            "limited_1rep"
        )
    )

    return subject_table


# =========================================================
# 11. MAIN
# =========================================================

def main():

    print("=" * 76)
    print("PIXEL → CM → ELLIPSE CIRCUMFERENCE")
    print("=" * 76)

    image_df = (
        load_measurement_data()
    )

    metadata_df = (
        load_subject_metadata()
    )

    print()
    print(
        "Creating subject-view scale table..."
    )

    scale_table = (
        make_subject_view_scale_table(
            image_df=image_df,
            metadata_df=metadata_df,
        )
    )

    print()
    print(
        scale_table[
            [
                "subject_id",
                "view",
                "height_cm",
                "height_px_median",
                "cm_per_px_recommended",
                "n_valid_height_images",
                "scale_qc_status",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Converting image-level pixels "
        "to centimeters..."
    )

    image_cm_table = (
        make_image_level_cm_table(
            image_df=image_df,
            scale_table=scale_table,
        )
    )

    print()
    print(
        "Aggregating frames..."
    )

    condition_table = (
        make_condition_level_table(
            image_cm_df=image_cm_table,
        )
    )

    print()
    print(
        "Combining front W and side D..."
    )

    phase_table = (
        make_phase_level_table(
            condition_table=
                condition_table,
        )
    )

    print()
    print(
        "Calculating inspiration-expiration "
        "expansion..."
    )

    repetition_table = (
        make_repetition_level_table(
            phase_table=
                phase_table,
        )
    )

    subject_table = (
        make_subject_level_table(
            repetition_table=
                repetition_table,
        )
    )

    # =====================================================
    # SAVE
    # =====================================================

    scale_path = (
        OUTPUT_DIR
        / "N_subject_view_scales.csv"
    )

    image_path = (
        OUTPUT_DIR
        / "N_image_level_cm_measurements.csv"
    )

    condition_path = (
        OUTPUT_DIR
        / "N_condition_level_measurements.csv"
    )

    phase_path = (
        OUTPUT_DIR
        / "N_phase_level_circumference.csv"
    )

    repetition_path = (
        OUTPUT_DIR
        / "N_repetition_level_ai_expansion.csv"
    )

    subject_path = (
        OUTPUT_DIR
        / "N_subject_level_ai_expansion.csv"
    )

    scale_table.to_csv(
        scale_path,
        index=False,
        encoding="utf-8-sig",
    )

    image_cm_table.to_csv(
        image_path,
        index=False,
        encoding="utf-8-sig",
    )

    condition_table.to_csv(
        condition_path,
        index=False,
        encoding="utf-8-sig",
    )

    phase_table.to_csv(
        phase_path,
        index=False,
        encoding="utf-8-sig",
    )

    repetition_table.to_csv(
        repetition_path,
        index=False,
        encoding="utf-8-sig",
    )

    subject_table.to_csv(
        subject_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 76)
    print("FINAL OUTPUT SUMMARY")
    print("=" * 76)

    print()
    print(
        "Repetition results:"
    )

    if repetition_table.empty:

        print(
            "No repetition-level "
            "results."
        )

    else:

        display_columns = [

            "subject_id",
            "repetition",

            "W_insp_px",
            "D_insp_px",

            "W_exp_px",
            "D_exp_px",

            "W_insp_cm",
            "D_insp_cm",

            "W_exp_cm",
            "D_exp_cm",

            "circumference_insp_cm",
            "circumference_exp_cm",

            "ai_expansion_cm",

            "repetition_status",
        ]

        print(
            repetition_table[
                display_columns
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Subject summary:"
    )

    if subject_table.empty:

        print(
            "No subject-level "
            "results."
        )

    else:

        print(
            subject_table
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Saved files:"
    )

    print(
        f"1. Scale table:\n"
        f"   {scale_path}"
    )

    print(
        f"\n2. Image-level cm:\n"
        f"   {image_path}"
    )

    print(
        f"\n3. Condition-level:\n"
        f"   {condition_path}"
    )

    print(
        f"\n4. Phase circumference:\n"
        f"   {phase_path}"
    )

    print(
        f"\n5. Repetition expansion:\n"
        f"   {repetition_path}"
    )

    print(
        f"\n6. Subject expansion:\n"
        f"   {subject_path}"
    )

    print()
    print("=" * 76)


if __name__ == "__main__":
    main()