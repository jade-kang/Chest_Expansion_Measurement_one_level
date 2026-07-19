from pathlib import Path
import re
import pandas as pd


PROJECT_ROOT = (
    Path.home()
    / "Desktop"
    / "CE_Xiphoid_Project"
)

ROI_CSV = (
    PROJECT_ROOT
    / "results"
    / "N_roi_predictions"
    / "csv"
    / "N_predicted_roi_coordinates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "N_roi_predictions"
    / "csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def parse_filename(filename):
    """
    예:
    N001_front_rep1_insp_01.jpg
    """

    pattern = re.compile(
        r"^(N\d{3})_"
        r"(front|side)_"
        r"(rep\d+)_"
        r"(insp|exp)_"
        r"(\d+)"
    )

    match = pattern.match(filename)

    if not match:
        return None

    return {
        "subject_id": match.group(1),
        "view": match.group(2),
        "repetition": match.group(3),
        "phase": match.group(4),
        "frame_no": match.group(5),
    }


def main():

    if not ROI_CSV.exists():
        raise FileNotFoundError(
            f"ROI CSV not found:\n{ROI_CSV}"
        )

    df = pd.read_csv(ROI_CSV)

    parsed_rows = []

    for _, row in df.iterrows():

        parsed = parse_filename(
            row["filename"]
        )

        if parsed is None:
            print(
                "[WARNING] "
                f"Could not parse: "
                f'{row["filename"]}'
            )
            continue

        parsed_rows.append({
            **parsed,
            "filename": row["filename"],
            "class_name": row["class_name"],
            "confidence": row["confidence"],
            "prediction_stage": row["prediction_stage"],
        })

    long_df = pd.DataFrame(
        parsed_rows
    )

    # -----------------------------------------
    # 이미지별 class 존재 여부
    # -----------------------------------------

    image_table = (
        long_df
        .assign(
            detected=1
        )
        .pivot_table(
            index=[
                "filename",
                "subject_id",
                "view",
                "repetition",
                "phase",
                "frame_no",
            ],
            columns="class_name",
            values="detected",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    if "xiphoid_roi" not in image_table:
        image_table["xiphoid_roi"] = 0

    if "height_roi" not in image_table:
        image_table["height_roi"] = 0

    image_table[
        "measurement_usable"
    ] = (
        image_table[
            "xiphoid_roi"
        ]
        ==
        1
    ).astype(int)

    # -----------------------------------------
    # 조건별 유효 프레임 개수
    # -----------------------------------------

    condition_table = (
        image_table
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
            total_detected_rows=(
                "filename",
                "nunique",
            ),
            xiphoid_valid_frames=(
                "xiphoid_roi",
                "sum",
            ),
            height_valid_frames=(
                "height_roi",
                "sum",
            ),
            measurement_usable_frames=(
                "measurement_usable",
                "sum",
            ),
        )
    )

    condition_table[
        "condition_usable"
    ] = (
        condition_table[
            "measurement_usable_frames"
        ]
        >
        0
    ).astype(int)

    # -----------------------------------------
    # rep별 네 조건 완성 여부
    # -----------------------------------------

    condition_key = (
        condition_table[
            "view"
        ]
        +
        "_"
        +
        condition_table[
            "phase"
        ]
    )

    condition_table[
        "condition_key"
    ] = condition_key

    rep_table = (
        condition_table
        .pivot_table(
            index=[
                "subject_id",
                "repetition",
            ],
            columns="condition_key",
            values="condition_usable",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    required_columns = [
        "front_insp",
        "front_exp",
        "side_insp",
        "side_exp",
    ]

    for column in required_columns:

        if column not in rep_table:
            rep_table[column] = 0

    rep_table[
        "all_four_conditions_available"
    ] = (
        rep_table[
            required_columns
        ]
        .sum(
            axis=1
        )
        ==
        4
    ).astype(int)

    # -----------------------------------------
    # subject별 유효 repetition 개수
    # -----------------------------------------

    subject_table = (
        rep_table
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            valid_repetitions=(
                "all_four_conditions_available",
                "sum",
            ),
        )
    )

    subject_table[
        "analysis_status"
    ] = subject_table[
        "valid_repetitions"
    ].map(
        lambda x:
        "complete_3reps"
        if x == 3
        else (
            "usable_2reps"
            if x == 2
            else (
                "limited_1rep"
                if x == 1
                else
                "exclude_no_complete_rep"
            )
        )
    )

    # -----------------------------------------
    # 저장
    # -----------------------------------------

    image_path = (
        OUTPUT_DIR
        / "N_image_roi_availability.csv"
    )

    condition_path = (
        OUTPUT_DIR
        / "N_condition_roi_availability.csv"
    )

    rep_path = (
        OUTPUT_DIR
        / "N_repetition_roi_availability.csv"
    )

    subject_path = (
        OUTPUT_DIR
        / "N_subject_roi_availability.csv"
    )

    image_table.to_csv(
        image_path,
        index=False,
        encoding="utf-8-sig",
    )

    condition_table.to_csv(
        condition_path,
        index=False,
        encoding="utf-8-sig",
    )

    rep_table.to_csv(
        rep_path,
        index=False,
        encoding="utf-8-sig",
    )

    subject_table.to_csv(
        subject_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 70)
    print("N ROI AVAILABILITY AUDIT")
    print("=" * 70)

    print()
    print("Subject summary:")
    print(subject_table.to_string(
        index=False
    ))

    print()
    print("Saved:")
    print(image_path)
    print(condition_path)
    print(rep_path)
    print(subject_path)

    print("=" * 70)


if __name__ == "__main__":
    main()