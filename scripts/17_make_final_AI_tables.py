from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = (
    Path.home()
    / "Desktop"
    / "CE_Xiphoid_Project"
)

REPETITION_CSV = (
    PROJECT_ROOT
    / "results"
    / "N_final_measurements"
    / "csv"
    / "N_repetition_level_ai_expansion.csv"
)

PHASE_CSV = (
    PROJECT_ROOT
    / "results"
    / "N_final_measurements"
    / "csv"
    / "N_phase_level_circumference.csv"
)

SCALE_CSV = (
    PROJECT_ROOT
    / "results"
    / "N_final_measurements"
    / "csv"
    / "N_subject_view_scales.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "N_final_measurements"
    / "final_tables"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 2. FIXED SUBJECT ORDER
# =========================================================

SUBJECTS = [
    f"N{i:03d}"
    for i in range(1, 10)
]

REPETITIONS = [
    "rep1",
    "rep2",
    "rep3",
]


# =========================================================
# 3. LOAD
# =========================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

    return pd.read_csv(path)


# =========================================================
# 4. MAKE DETAILED REPETITION TABLE
# =========================================================

def make_detailed_table(
    repetition_df,
    phase_df,
    scale_df,
):

    # -----------------------------------------
    # front / side scale
    # -----------------------------------------

    scale_wide = (
        scale_df
        .pivot_table(
            index="subject_id",
            columns="view",
            values="cm_per_px_recommended",
            aggfunc="first",
        )
        .reset_index()
    )

    scale_wide = scale_wide.rename(
        columns={
            "front":
                "front_cm_per_px",

            "side":
                "side_cm_per_px",
        }
    )

    # -----------------------------------------
    # front / side height pixel
    # -----------------------------------------

    height_wide = (
        scale_df
        .pivot_table(
            index="subject_id",
            columns="view",
            values="height_px_median",
            aggfunc="first",
        )
        .reset_index()
    )

    height_wide = height_wide.rename(
        columns={
            "front":
                "front_height_px_median",

            "side":
                "side_height_px_median",
        }
    )

    # -----------------------------------------
    # base repetition table
    # -----------------------------------------

    detailed = (
        repetition_df
        .merge(
            scale_wide,
            on="subject_id",
            how="left",
        )
        .merge(
            height_wide,
            on="subject_id",
            how="left",
        )
    )

    # -----------------------------------------
    # rename to publication-friendly names
    # -----------------------------------------

    detailed = detailed.rename(
        columns={

            "circumference_insp_cm":
                "Inhale_cm",

            "circumference_exp_cm":
                "Exhale_cm",

            "ai_expansion_cm":
                "CE_cm",
        }
    )

    # -----------------------------------------
    # AI source classification
    # -----------------------------------------

    if (
        "front_gt_fallback_frames"
        in
        phase_df.columns
    ):

        phase_source = (
            phase_df
            .copy()
        )

        phase_source[
            "used_gt_fallback"
        ] = (

            phase_source[
                "front_gt_fallback_frames"
            ]
            .fillna(0)

            +

            phase_source[
                "side_gt_fallback_frames"
            ]
            .fillna(0)

        ) > 0

        rep_source = (
            phase_source
            .groupby(
                [
                    "subject_id",
                    "repetition",
                ],
                as_index=False,
            )
            .agg(
                used_gt_fallback=(
                    "used_gt_fallback",
                    "max",
                )
            )
        )

        detailed = (
            detailed
            .merge(
                rep_source,
                on=[
                    "subject_id",
                    "repetition",
                ],
                how="left",
            )
        )

        detailed[
            "ROI_source_summary"
        ] = np.where(

            detailed[
                "used_gt_fallback"
            ]
            .fillna(False),

            "predicted_plus_GT_fallback",

            "predicted_only",
        )

    else:

        detailed[
            "ROI_source_summary"
        ] = "not_available"

    # -----------------------------------------
    # fixed order
    # -----------------------------------------

    preferred_columns = [

        "subject_id",
        "repetition",

        # pixel values
        "W_insp_px",
        "D_insp_px",

        "W_exp_px",
        "D_exp_px",

        # height calibration
        "front_height_px_median",
        "side_height_px_median",

        "front_cm_per_px",
        "side_cm_per_px",

        # converted width/depth
        "W_insp_cm",
        "D_insp_cm",

        "W_exp_cm",
        "D_exp_cm",

        # final circumference
        "Inhale_cm",
        "Exhale_cm",

        # chest expansion
        "CE_cm",

        # QC
        "ROI_source_summary",
        "repetition_status",
    ]

    existing_columns = [

        column
        for column
        in preferred_columns
        if column
        in detailed.columns
    ]

    detailed = (
        detailed[
            existing_columns
        ]
        .sort_values(
            [
                "subject_id",
                "repetition",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return detailed


# =========================================================
# 5. MAKE TAPE-SHAPED AI TABLE
# =========================================================

def make_wide_ai_table(
    detailed,
):

    base = pd.DataFrame({
        "Participant_ID":
            SUBJECTS
    })

    # -----------------------------------------
    # repetition별 들숨
    # -----------------------------------------

    inhale = (
        detailed
        .pivot_table(
            index="subject_id",
            columns="repetition",
            values="Inhale_cm",
            aggfunc="first",
        )
        .reset_index()
    )

    inhale = inhale.rename(
        columns={

            "subject_id":
                "Participant_ID",

            "rep1":
                "Inhale_cm_1",

            "rep2":
                "Inhale_cm_2",

            "rep3":
                "Inhale_cm_3",
        }
    )

    # -----------------------------------------
    # repetition별 날숨
    # -----------------------------------------

    exhale = (
        detailed
        .pivot_table(
            index="subject_id",
            columns="repetition",
            values="Exhale_cm",
            aggfunc="first",
        )
        .reset_index()
    )

    exhale = exhale.rename(
        columns={

            "subject_id":
                "Participant_ID",

            "rep1":
                "Exhale_cm_1",

            "rep2":
                "Exhale_cm_2",

            "rep3":
                "Exhale_cm_3",
        }
    )

    # -----------------------------------------
    # repetition별 CE
    # -----------------------------------------

    ce = (
        detailed
        .pivot_table(
            index="subject_id",
            columns="repetition",
            values="CE_cm",
            aggfunc="first",
        )
        .reset_index()
    )

    ce = ce.rename(
        columns={

            "subject_id":
                "Participant_ID",

            "rep1":
                "CE_1",

            "rep2":
                "CE_2",

            "rep3":
                "CE_3",
        }
    )

    # -----------------------------------------
    # merge
    # -----------------------------------------

    wide = (
        base
        .merge(
            inhale,
            on="Participant_ID",
            how="left",
        )
        .merge(
            exhale,
            on="Participant_ID",
            how="left",
        )
        .merge(
            ce,
            on="Participant_ID",
            how="left",
        )
    )

    # -----------------------------------------
    # exact column order requested
    # -----------------------------------------

    final_columns = [

        "Participant_ID",

        "Inhale_cm_1",
        "Exhale_cm_1",

        "Inhale_cm_2",
        "Exhale_cm_2",

        "Inhale_cm_3",
        "Exhale_cm_3",

        "CE_1",
        "CE_2",
        "CE_3",
    ]

    for column in final_columns:

        if column not in wide.columns:

            wide[column] = np.nan

    wide = wide[
        final_columns
    ]

    # -----------------------------------------
    # round
    # -----------------------------------------

    numeric_columns = [

        column
        for column
        in wide.columns
        if column
        !=
        "Participant_ID"
    ]

    wide[
        numeric_columns
    ] = (
        wide[
            numeric_columns
        ]
        .round(3)
    )

    return wide


# =========================================================
# 6. MAKE SUBJECT SUMMARY
# =========================================================

def make_subject_summary(
    wide,
):

    summary = (
        wide.copy()
    )

    summary[
        "Mean_Inhale_cm"
    ] = summary[
        [
            "Inhale_cm_1",
            "Inhale_cm_2",
            "Inhale_cm_3",
        ]
    ].mean(
        axis=1,
        skipna=True,
    )

    summary[
        "Mean_Exhale_cm"
    ] = summary[
        [
            "Exhale_cm_1",
            "Exhale_cm_2",
            "Exhale_cm_3",
        ]
    ].mean(
        axis=1,
        skipna=True,
    )

    summary[
        "Mean_CE_cm"
    ] = summary[
        [
            "CE_1",
            "CE_2",
            "CE_3",
        ]
    ].mean(
        axis=1,
        skipna=True,
    )

    summary[
        "Valid_Repetitions"
    ] = summary[
        [
            "CE_1",
            "CE_2",
            "CE_3",
        ]
    ].notna().sum(
        axis=1
    )

    summary[
        "Analysis_Status"
    ] = summary[
        "Valid_Repetitions"
    ].map(

        lambda value:

        "complete_3reps"
        if value == 3

        else (

            "usable_2reps"
            if value == 2

            else (

                "limited_1rep"
                if value == 1

                else
                "no_valid_rep"
            )
        )
    )

    return summary


# =========================================================
# 7. MAIN
# =========================================================

def main():

    print("=" * 76)
    print("FINAL AI TABLE GENERATION")
    print("=" * 76)

    repetition_df = (
        load_csv(
            REPETITION_CSV
        )
    )

    phase_df = (
        load_csv(
            PHASE_CSV
        )
    )

    scale_df = (
        load_csv(
            SCALE_CSV
        )
    )

    detailed = (
        make_detailed_table(
            repetition_df=
                repetition_df,

            phase_df=
                phase_df,

            scale_df=
                scale_df,
        )
    )

    wide = (
        make_wide_ai_table(
            detailed=
                detailed
        )
    )

    subject_summary = (
        make_subject_summary(
            wide=
                wide
        )
    )

    # -----------------------------------------
    # output paths
    # -----------------------------------------

    detailed_path = (
        OUTPUT_DIR
        /
        "N_AI_repetition_detailed.csv"
    )

    wide_path = (
        OUTPUT_DIR
        /
        "N_AI_tape_format.csv"
    )

    summary_path = (
        OUTPUT_DIR
        /
        "N_AI_subject_summary.csv"
    )

    # -----------------------------------------
    # save
    # -----------------------------------------

    detailed.to_csv(
        detailed_path,
        index=False,
        encoding="utf-8-sig",
    )

    wide.to_csv(
        wide_path,
        index=False,
        encoding="utf-8-sig",
    )

    subject_summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------
    # display
    # -----------------------------------------

    print()
    print(
        "AI result table "
        "(same format as tape):"
    )

    print()
    print(
        wide.to_string(
            index=False
        )
    )

    print()
    print(
        "Subject summary:"
    )

    print()

    print(
        subject_summary[
            [
                "Participant_ID",

                "Mean_Inhale_cm",

                "Mean_Exhale_cm",

                "Mean_CE_cm",

                "Valid_Repetitions",

                "Analysis_Status",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 76)

    print(
        "1. Detailed repetition table:\n"
        f"{detailed_path}"
    )

    print()

    print(
        "2. Tape-format AI table:\n"
        f"{wide_path}"
    )

    print()

    print(
        "3. Subject summary:\n"
        f"{summary_path}"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()