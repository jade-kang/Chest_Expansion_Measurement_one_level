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

AI_REPETITION_CSV = (
    PROJECT_ROOT
    / "results"
    / "N_final_measurements"
    / "csv"
    / "N_repetition_level_ai_expansion.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "N_final_measurements"
    / "comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 2. SETTINGS
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
# 3. MAIN
# =========================================================

def main():

    if not AI_REPETITION_CSV.exists():

        raise FileNotFoundError(
            "\nAI repetition result not found:\n"
            f"{AI_REPETITION_CSV}"
        )

    ai_df = pd.read_csv(
        AI_REPETITION_CSV
    )

    print("=" * 76)
    print("AI–TAPE COMPARISON TABLE")
    print("=" * 76)

    print(
        f"Input:\n{AI_REPETITION_CSV}"
    )

    # -----------------------------------------------------
    # 모든 대상자 × 3회 반복 기본 틀
    # -----------------------------------------------------

    base_rows = []

    for subject_id in SUBJECTS:

        for repetition in REPETITIONS:

            base_rows.append({

                "subject_id":
                    subject_id,

                "repetition":
                    repetition,
            })

    comparison_df = pd.DataFrame(
        base_rows
    )

    # -----------------------------------------------------
    # AI 결과 열 선택 및 이름 변경
    # -----------------------------------------------------

    ai_columns = [

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

    existing_ai_columns = [

        column
        for column
        in ai_columns
        if column
        in ai_df.columns
    ]

    ai_subset = (
        ai_df[
            existing_ai_columns
        ]
        .copy()
    )

    ai_subset = ai_subset.rename(
        columns={

            "circumference_insp_cm":
                "ai_insp_circumference_cm",

            "circumference_exp_cm":
                "ai_exp_circumference_cm",

            "repetition_status":
                "ai_repetition_status",
        }
    )

    comparison_df = (
        comparison_df
        .merge(
            ai_subset,
            on=[
                "subject_id",
                "repetition",
            ],
            how="left",
        )
    )

    # -----------------------------------------------------
    # 줄자 입력용 빈 열
    # -----------------------------------------------------

    comparison_df[
        "tape_insp_circumference_cm"
    ] = np.nan

    comparison_df[
        "tape_exp_circumference_cm"
    ] = np.nan

    comparison_df[
        "tape_expansion_cm"
    ] = np.nan

    comparison_df[
        "difference_ai_minus_tape_cm"
    ] = np.nan

    comparison_df[
        "absolute_error_cm"
    ] = np.nan

    comparison_df[
        "include_comparison"
    ] = 1

    comparison_df[
        "note"
    ] = ""

    # -----------------------------------------------------
    # 열 순서
    # -----------------------------------------------------

    preferred_columns = [

        "subject_id",
        "repetition",

        # Pixel
        "W_insp_px",
        "D_insp_px",

        "W_exp_px",
        "D_exp_px",

        # Converted cm
        "W_insp_cm",
        "D_insp_cm",

        "W_exp_cm",
        "D_exp_cm",

        # AI circumference
        "ai_insp_circumference_cm",
        "ai_exp_circumference_cm",

        # AI chest expansion
        "ai_expansion_cm",

        # Tape input
        "tape_insp_circumference_cm",
        "tape_exp_circumference_cm",
        "tape_expansion_cm",

        # Comparison
        "difference_ai_minus_tape_cm",
        "absolute_error_cm",

        # QC
        "ai_repetition_status",
        "include_comparison",
        "note",
    ]

    final_columns = [

        column
        for column
        in preferred_columns
        if column
        in comparison_df.columns
    ]

    comparison_df = (
        comparison_df[
            final_columns
        ]
    )

    # -----------------------------------------------------
    # 저장
    # -----------------------------------------------------

    comparison_path = (
        OUTPUT_DIR
        / "N_AI_tape_repetition_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # AI 대상자별 요약
    # -----------------------------------------------------

    usable_ai = (
        comparison_df[
            comparison_df[
                "ai_expansion_cm"
            ]
            .notna()
        ]
        .copy()
    )

    if usable_ai.empty:

        subject_summary = pd.DataFrame(
            {
                "subject_id":
                    SUBJECTS,
            }
        )

    else:

        subject_summary = (
            usable_ai
            .groupby(
                "subject_id",
                as_index=False,
            )
            .agg(

                valid_ai_repetitions=(
                    "ai_expansion_cm",
                    "count",
                ),

                mean_ai_insp_circumference_cm=(
                    "ai_insp_circumference_cm",
                    "mean",
                ),

                mean_ai_exp_circumference_cm=(
                    "ai_exp_circumference_cm",
                    "mean",
                ),

                mean_ai_expansion_cm=(
                    "ai_expansion_cm",
                    "mean",
                ),

                sd_ai_expansion_cm=(
                    "ai_expansion_cm",
                    "std",
                ),

                median_ai_expansion_cm=(
                    "ai_expansion_cm",
                    "median",
                ),
            )
        )

        all_subjects = pd.DataFrame({
            "subject_id":
                SUBJECTS,
        })

        subject_summary = (
            all_subjects
            .merge(
                subject_summary,
                on="subject_id",
                how="left",
            )
        )

    # 줄자 평균 입력 및 비교용 빈 열
    subject_summary[
        "mean_tape_insp_circumference_cm"
    ] = np.nan

    subject_summary[
        "mean_tape_exp_circumference_cm"
    ] = np.nan

    subject_summary[
        "mean_tape_expansion_cm"
    ] = np.nan

    subject_summary[
        "mean_difference_ai_minus_tape_cm"
    ] = np.nan

    subject_summary[
        "note"
    ] = ""

    subject_path = (
        OUTPUT_DIR
        / "N_AI_tape_subject_summary.csv"
    )

    subject_summary.to_csv(
        subject_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 터미널 출력
    # -----------------------------------------------------

    print()
    print(
        "Repetition-level comparison table:"
    )

    print(
        comparison_df[
            [
                "subject_id",
                "repetition",

                "ai_insp_circumference_cm",
                "ai_exp_circumference_cm",

                "ai_expansion_cm",

                "ai_repetition_status",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Subject-level AI summary:"
    )

    print(
        subject_summary[
            [
                column
                for column
                in [

                    "subject_id",

                    "valid_ai_repetitions",

                    "mean_ai_insp_circumference_cm",

                    "mean_ai_exp_circumference_cm",

                    "mean_ai_expansion_cm",
                ]
                if column
                in subject_summary.columns
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 76)

    print(
        "Saved repetition comparison:\n"
        f"{comparison_path}"
    )

    print()

    print(
        "Saved subject summary:\n"
        f"{subject_path}"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()