from pathlib import Path
import re

import cv2
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

IMAGE_DIR = (
    PROJECT_ROOT
    / "frames_all"
)

PREDICTED_ROI_CSV = (
    PROJECT_ROOT
    / "results"
    / "N_roi_predictions"
    / "csv"
    / "N_predicted_roi_coordinates.csv"
)

GROUND_TRUTH_LABEL_DIR = (
    PROJECT_ROOT
    / "external_test_dataset"
    / "labels"
    / "test"
)

SUBJECT_METADATA_CSV = (
    PROJECT_ROOT
    / "metadata"
    / "N_subject_metadata.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "N_cv_measurements"
)

OVERLAY_DIR = (
    OUTPUT_ROOT
    / "measurement_overlays"
)

MASK_DIR = (
    OUTPUT_ROOT
    / "body_masks"
)

CSV_DIR = (
    OUTPUT_ROOT
    / "csv"
)


# =========================================================
# 2. ANALYSIS SETTINGS
# =========================================================

# True:
# 예측 xiphoid ROI가 없으면 GT ROI 사용
#
# False:
# 예측 ROI만 사용
USE_GROUND_TRUTH_FALLBACK = True

# xiphoid ROI 중앙의 몇 %를 폭 측정에 사용할지
CENTRAL_BAND_START = 0.30
CENTRAL_BAND_END = 0.70

# 너무 작은 foreground component 제거 기준
MIN_COMPONENT_AREA_RATIO = 0.05

# class
XIPHOID_CLASS_ID = 0
HEIGHT_CLASS_ID = 1


# =========================================================
# 3. BASIC HELPERS
# =========================================================

def prepare_output_folders():

    for folder in [
        OVERLAY_DIR,
        MASK_DIR,
        CSV_DIR,
    ]:
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )


def parse_filename(filename):

    pattern = re.compile(
        r"^(N\d{3})_"
        r"(front|side)_"
        r"(rep\d+)_"
        r"(insp|exp)_"
        r"(\d+)"
    )

    match = pattern.match(filename)

    if match is None:
        return None

    return {
        "subject_id": match.group(1),
        "view": match.group(2),
        "repetition": match.group(3),
        "phase": match.group(4),
        "frame_no": match.group(5),
    }


def clip_box(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
):

    x1 = int(
        max(
            0,
            min(
                x1,
                image_width - 1,
            ),
        )
    )

    y1 = int(
        max(
            0,
            min(
                y1,
                image_height - 1,
            ),
        )
    )

    x2 = int(
        max(
            x1 + 1,
            min(
                x2,
                image_width,
            ),
        )
    )

    y2 = int(
        max(
            y1 + 1,
            min(
                y2,
                image_height,
            ),
        )
    )

    return x1, y1, x2, y2


# =========================================================
# 4. READ PREDICTED ROI
# =========================================================

def load_predicted_roi():

    if not PREDICTED_ROI_CSV.exists():

        raise FileNotFoundError(
            f"Predicted ROI CSV not found:\n"
            f"{PREDICTED_ROI_CSV}"
        )

    df = pd.read_csv(
        PREDICTED_ROI_CSV
    )

    predicted = {}

    for _, row in df.iterrows():

        filename = str(
            row["filename"]
        )

        class_name = str(
            row["class_name"]
        )

        key = (
            filename,
            class_name,
        )

        candidate = {
            "x1": int(row["x1_px"]),
            "y1": int(row["y1_px"]),
            "x2": int(row["x2_px"]),
            "y2": int(row["y2_px"]),
            "confidence": float(
                row["confidence"]
            ),
            "roi_source": "predicted",
        }

        previous = predicted.get(
            key
        )

        if (
            previous is None
            or
            candidate["confidence"]
            >
            previous["confidence"]
        ):

            predicted[key] = candidate

    return predicted


# =========================================================
# 5. READ YOLO GROUND TRUTH ROI
# =========================================================

def read_ground_truth_box(
    filename,
    class_id,
    image_width,
    image_height,
):

    label_path = (
        GROUND_TRUTH_LABEL_DIR
        /
        f"{Path(filename).stem}.txt"
    )

    if not label_path.exists():
        return None

    candidates = []

    with label_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            parts = (
                line
                .strip()
                .split()
            )

            if len(parts) < 5:
                continue

            current_class_id = int(
                float(
                    parts[0]
                )
            )

            if (
                current_class_id
                !=
                class_id
            ):
                continue

            x_center = float(
                parts[1]
            )

            y_center = float(
                parts[2]
            )

            width = float(
                parts[3]
            )

            height = float(
                parts[4]
            )

            x1 = (
                x_center
                -
                width / 2
            ) * image_width

            y1 = (
                y_center
                -
                height / 2
            ) * image_height

            x2 = (
                x_center
                +
                width / 2
            ) * image_width

            y2 = (
                y_center
                +
                height / 2
            ) * image_height

            candidates.append({
                "x1": int(
                    round(x1)
                ),
                "y1": int(
                    round(y1)
                ),
                "x2": int(
                    round(x2)
                ),
                "y2": int(
                    round(y2)
                ),
                "confidence": np.nan,
                "roi_source":
                    "ground_truth_fallback",
            })

    if not candidates:
        return None

    return candidates[0]


# =========================================================
# 6. SELECT ROI
# =========================================================

def select_roi(
    predicted,
    filename,
    class_name,
    class_id,
    image_width,
    image_height,
):

    predicted_box = predicted.get(
        (
            filename,
            class_name,
        )
    )

    if predicted_box is not None:
        return predicted_box.copy()

    if not USE_GROUND_TRUTH_FALLBACK:
        return None

    return read_ground_truth_box(
        filename=filename,
        class_id=class_id,
        image_width=image_width,
        image_height=image_height,
    )


# =========================================================
# 7. BODY MASK
# =========================================================

def component_score(
    component_mask,
):

    height, width = (
        component_mask.shape
    )

    area = int(
        np.count_nonzero(
            component_mask
        )
    )

    if area == 0:
        return -1

    center_x = (
        width // 2
    )

    center_band_half = max(
        1,
        int(
            width * 0.10
        ),
    )

    center_band = (
        component_mask[
            :,
            max(
                0,
                center_x
                -
                center_band_half,
            ):
            min(
                width,
                center_x
                +
                center_band_half,
            ),
        ]
    )

    center_overlap = (
        np.count_nonzero(
            center_band
        )
    )

    score = (
        area
        +
        center_overlap * 3
    )

    return score


def choose_best_component(
    binary_mask,
):

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    image_area = (
        binary_mask.shape[0]
        *
        binary_mask.shape[1]
    )

    minimum_area = (
        image_area
        *
        MIN_COMPONENT_AREA_RATIO
    )

    best_mask = None
    best_score = -1

    for label_id in range(
        1,
        num_labels,
    ):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA,
        ]

        if area < minimum_area:
            continue

        component = (
            labels
            ==
            label_id
        ).astype(
            np.uint8
        ) * 255

        score = component_score(
            component
        )

        if score > best_score:

            best_score = score
            best_mask = component

    return best_mask


def create_body_mask(
    roi_bgr,
):

    gray = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(
        gray
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0,
    )

    _, binary_normal = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY
        +
        cv2.THRESH_OTSU,
    )

    binary_inverse = (
        cv2.bitwise_not(
            binary_normal
        )
    )

    kernel_close = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11),
        )
    )

    kernel_open = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )
    )

    candidate_masks = []

    for binary in [
        binary_normal,
        binary_inverse,
    ]:

        cleaned = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel_close,
            iterations=2,
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel_open,
            iterations=1,
        )

        component = (
            choose_best_component(
                cleaned
            )
        )

        if component is None:
            continue

        score = component_score(
            component
        )

        candidate_masks.append(
            (
                score,
                component,
            )
        )

    if not candidate_masks:

        return None

    candidate_masks.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidate_masks[0][1]


# =========================================================
# 8. WIDTH PROFILE
# =========================================================

def measure_horizontal_width(
    mask,
):

    roi_height, roi_width = (
        mask.shape
    )

    band_y1 = int(
        round(
            roi_height
            *
            CENTRAL_BAND_START
        )
    )

    band_y2 = int(
        round(
            roi_height
            *
            CENTRAL_BAND_END
        )
    )

    band_y1 = max(
        0,
        min(
            band_y1,
            roi_height - 1,
        ),
    )

    band_y2 = max(
        band_y1 + 1,
        min(
            band_y2,
            roi_height,
        ),
    )

    measurements = []

    for y in range(
        band_y1,
        band_y2,
    ):

        foreground_x = np.where(
            mask[y] > 0
        )[0]

        if len(
            foreground_x
        ) < 2:
            continue

        left_x = int(
            foreground_x.min()
        )

        right_x = int(
            foreground_x.max()
        )

        width = (
            right_x
            -
            left_x
            +
            1
        )

        measurements.append({
            "y": y,
            "left_x": left_x,
            "right_x": right_x,
            "width": width,
        })

    if not measurements:
        return None

    widths = np.array(
        [
            row["width"]
            for row
            in measurements
        ],
        dtype=float,
    )

    median_width = float(
        np.median(
            widths
        )
    )

    mean_width = float(
        np.mean(
            widths
        )
    )

    std_width = float(
        np.std(
            widths,
            ddof=0,
        )
    )

    q1 = float(
        np.percentile(
            widths,
            25,
        )
    )

    q3 = float(
        np.percentile(
            widths,
            75,
        )
    )

    iqr = (
        q3
        -
        q1
    )

    representative_row = min(
        measurements,
        key=lambda row:
        abs(
            row["width"]
            -
            median_width
        ),
    )

    return {
        "width_median_px":
            median_width,

        "width_mean_px":
            mean_width,

        "width_std_px":
            std_width,

        "width_iqr_px":
            iqr,

        "width_min_px":
            float(
                widths.min()
            ),

        "width_max_px":
            float(
                widths.max()
            ),

        "valid_row_count":
            len(
                measurements
            ),

        "band_y1":
            band_y1,

        "band_y2":
            band_y2,

        "representative_y":
            representative_row["y"],

        "representative_left_x":
            representative_row[
                "left_x"
            ],

        "representative_right_x":
            representative_row[
                "right_x"
            ],
    }


# =========================================================
# 9. OVERLAY
# =========================================================

def create_measurement_overlay(
    image,
    roi_box,
    measurement,
    parsed,
    roi_source,
    confidence,
):

    overlay = image.copy()

    x1 = roi_box["x1"]
    y1 = roi_box["y1"]
    x2 = roi_box["x2"]
    y2 = roi_box["y2"]

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        3,
    )

    band_global_y1 = (
        y1
        +
        measurement["band_y1"]
    )

    band_global_y2 = (
        y1
        +
        measurement["band_y2"]
    )

    cv2.rectangle(
        overlay,
        (
            x1,
            band_global_y1,
        ),
        (
            x2,
            band_global_y2,
        ),
        (180, 180, 180),
        2,
    )

    line_y = (
        y1
        +
        measurement[
            "representative_y"
        ]
    )

    line_x1 = (
        x1
        +
        measurement[
            "representative_left_x"
        ]
    )

    line_x2 = (
        x1
        +
        measurement[
            "representative_right_x"
        ]
    )

    cv2.line(
        overlay,
        (
            line_x1,
            line_y,
        ),
        (
            line_x2,
            line_y,
        ),
        (255, 255, 255),
        4,
    )

    cv2.circle(
        overlay,
        (
            line_x1,
            line_y,
        ),
        7,
        (255, 255, 255),
        -1,
    )

    cv2.circle(
        overlay,
        (
            line_x2,
            line_y,
        ),
        7,
        (255, 255, 255),
        -1,
    )

    measure_name = (
        "W"
        if
        parsed["view"]
        ==
        "front"
        else
        "D"
    )

    confidence_text = (
        "NA"
        if pd.isna(
            confidence
        )
        else
        f"{confidence:.3f}"
    )

    text_lines = [

        (
            f"{measure_name}_px median: "
            f'{measurement["width_median_px"]:.1f}'
        ),

        (
            f"ROI source: "
            f"{roi_source}"
        ),

        (
            f"ROI conf: "
            f"{confidence_text}"
        ),

        (
            f"Rows: "
            f'{measurement["valid_row_count"]}'
        ),
    ]

    text_y = max(
        35,
        y1 - 105,
    )

    for text in text_lines:

        cv2.putText(
            overlay,
            text,
            (
                max(
                    10,
                    x1,
                ),
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        text_y += 30

    return overlay


# =========================================================
# 10. MAIN
# =========================================================

def main():

    prepare_output_folders()

    predicted = load_predicted_roi()

    image_paths = sorted(
        [
            path
            for path
            in IMAGE_DIR.iterdir()
            if (
                path.is_file()
                and
                path.name.startswith(
                    "N"
                )
                and
                path.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }
            )
        ]
    )

    rows = []

    print("=" * 72)
    print("N COMPUTER VISION WIDTH / DEPTH")
    print("=" * 72)

    print(
        f"Images found: "
        f"{len(image_paths)}"
    )

    print(
        f"GT fallback: "
        f"{USE_GROUND_TRUTH_FALLBACK}"
    )

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        parsed = parse_filename(
            image_path.name
        )

        if parsed is None:

            print(
                "  [SKIP] "
                "filename parse failed"
            )

            continue

        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is None:

            print(
                "  [SKIP] "
                "image read failed"
            )

            continue

        image_height, image_width = (
            image.shape[:2]
        )

        xiphoid_box = select_roi(
            predicted=predicted,
            filename=image_path.name,
            class_name="xiphoid_roi",
            class_id=XIPHOID_CLASS_ID,
            image_width=image_width,
            image_height=image_height,
        )

        height_box = select_roi(
            predicted=predicted,
            filename=image_path.name,
            class_name="height_roi",
            class_id=HEIGHT_CLASS_ID,
            image_width=image_width,
            image_height=image_height,
        )

        if xiphoid_box is None:

            rows.append({
                **parsed,

                "filename":
                    image_path.name,

                "measurement_status":
                    "xiphoid_roi_missing",

                "xiphoid_roi_source":
                    "missing",

                "height_roi_source":
                    (
                        "missing"
                        if height_box is None
                        else
                        height_box[
                            "roi_source"
                        ]
                    ),

                "note":
                    "",
            })

            continue

        x1, y1, x2, y2 = clip_box(
            xiphoid_box["x1"],
            xiphoid_box["y1"],
            xiphoid_box["x2"],
            xiphoid_box["y2"],
            image_width,
            image_height,
        )

        xiphoid_box[
            "x1"
        ] = x1

        xiphoid_box[
            "y1"
        ] = y1

        xiphoid_box[
            "x2"
        ] = x2

        xiphoid_box[
            "y2"
        ] = y2

        roi = image[
            y1:y2,
            x1:x2,
        ]

        if roi.size == 0:

            rows.append({
                **parsed,

                "filename":
                    image_path.name,

                "measurement_status":
                    "empty_roi_crop",

                "xiphoid_roi_source":
                    xiphoid_box[
                        "roi_source"
                    ],

                "note":
                    "",
            })

            continue

        body_mask = create_body_mask(
            roi
        )

        if body_mask is None:

            rows.append({
                **parsed,

                "filename":
                    image_path.name,

                "measurement_status":
                    "body_mask_failed",

                "xiphoid_roi_source":
                    xiphoid_box[
                        "roi_source"
                    ],

                "note":
                    "",
            })

            continue

        measurement = (
            measure_horizontal_width(
                body_mask
            )
        )

        if measurement is None:

            rows.append({
                **parsed,

                "filename":
                    image_path.name,

                "measurement_status":
                    "width_measurement_failed",

                "xiphoid_roi_source":
                    xiphoid_box[
                        "roi_source"
                    ],

                "note":
                    "",
            })

            continue

        mask_path = (
            MASK_DIR
            /
            image_path.name
        )

        cv2.imwrite(
            str(
                mask_path
            ),
            body_mask,
        )

        overlay = (
            create_measurement_overlay(
                image=image,
                roi_box=xiphoid_box,
                measurement=measurement,
                parsed=parsed,
                roi_source=xiphoid_box[
                    "roi_source"
                ],
                confidence=xiphoid_box[
                    "confidence"
                ],
            )
        )

        overlay_path = (
            OVERLAY_DIR
            /
            image_path.name
        )

        cv2.imwrite(
            str(
                overlay_path
            ),
            overlay,
        )

        height_px_raw = np.nan
        height_roi_source = "missing"
        height_confidence = np.nan

        if height_box is not None:

            height_px_raw = (
                height_box["y2"]
                -
                height_box["y1"]
            )

            height_roi_source = (
                height_box[
                    "roi_source"
                ]
            )

            height_confidence = (
                height_box[
                    "confidence"
                ]
            )

        rows.append({

            **parsed,

            "filename":
                image_path.name,

            "measurement_status":
                "success",

            "measurement_type":
                (
                    "W"
                    if
                    parsed["view"]
                    ==
                    "front"
                    else
                    "D"
                ),

            "xiphoid_roi_source":
                xiphoid_box[
                    "roi_source"
                ],

            "xiphoid_confidence":
                xiphoid_box[
                    "confidence"
                ],

            "height_roi_source":
                height_roi_source,

            "height_confidence":
                height_confidence,

            "xiphoid_x1_px":
                x1,

            "xiphoid_y1_px":
                y1,

            "xiphoid_x2_px":
                x2,

            "xiphoid_y2_px":
                y2,

            "width_median_px":
                measurement[
                    "width_median_px"
                ],

            "width_mean_px":
                measurement[
                    "width_mean_px"
                ],

            "width_std_px":
                measurement[
                    "width_std_px"
                ],

            "width_iqr_px":
                measurement[
                    "width_iqr_px"
                ],

            "width_min_px":
                measurement[
                    "width_min_px"
                ],

            "width_max_px":
                measurement[
                    "width_max_px"
                ],

            "valid_row_count":
                measurement[
                    "valid_row_count"
                ],

            "height_px_raw":
                height_px_raw,

            "note":
                "",
        })

    result_df = pd.DataFrame(
        rows
    )

    image_csv_path = (
        CSV_DIR
        /
        "N_image_level_pixel_measurements.csv"
    )

    result_df.to_csv(
        image_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print("MEASUREMENT FINISHED")
    print("=" * 72)

    print(
        f"Total rows: "
        f"{len(result_df)}"
    )

    if (
        "measurement_status"
        in
        result_df.columns
    ):

        print()
        print(
            result_df[
                "measurement_status"
            ]
            .value_counts(
                dropna=False
            )
        )

    print()
    print(
        f"CSV:\n"
        f"{image_csv_path}"
    )

    print()
    print(
        f"Overlays:\n"
        f"{OVERLAY_DIR}"
    )

    print()
    print(
        f"Masks:\n"
        f"{MASK_DIR}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()