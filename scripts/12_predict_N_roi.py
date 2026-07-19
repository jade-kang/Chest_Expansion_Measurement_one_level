from pathlib import Path
import csv
import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# 1. PATH SETTINGS
# =========================================================

PROJECT_ROOT = (
    Path.home()
    / "Desktop"
    / "CE_Xiphoid_Project"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "models"
    / "xiphoid_yolov8n_old2roi"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "frames_all"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "N_roi_predictions"
)

ANNOTATED_DIR = (
    OUTPUT_ROOT
    / "annotated_images"
)

XIPHOID_CROP_DIR = (
    OUTPUT_ROOT
    / "crops"
    / "xiphoid_roi"
)

HEIGHT_CROP_DIR = (
    OUTPUT_ROOT
    / "crops"
    / "height_roi"
)

CSV_DIR = (
    OUTPUT_ROOT
    / "csv"
)


# =========================================================
# 2. MODEL CLASS SETTINGS
# =========================================================

CLASS_NAMES = {
    0: "xiphoid_roi",
    1: "height_roi",
}

REQUIRED_CLASS_IDS = {
    0,
    1,
}


# =========================================================
# 3. PRIMARY AND RETRY SETTINGS
# =========================================================

# 1차 예측
PRIMARY_CONF = 0.25
PRIMARY_IMGSZ = 640

# ROI 누락 시 2차 예측
RETRY_CONF = 0.10
RETRY_IMGSZ = 960

IOU_THRESHOLD = 0.70


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def prepare_output_folders():
    """
    결과 저장 폴더를 만든다.
    """

    for folder in [
        ANNOTATED_DIR,
        XIPHOID_CROP_DIR,
        HEIGHT_CROP_DIR,
        CSV_DIR,
    ]:
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )


def find_n_images():
    """
    frames_all에서 N001~N009 이미지만 찾는다.
    """

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    images = []

    for path in IMAGE_DIR.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in valid_extensions:
            continue

        if not path.name.startswith("N"):
            continue

        images.append(path)

    return sorted(images)


def run_prediction(
    model,
    image_path,
    conf,
    imgsz,
):
    """
    이미지 한 장에 YOLO 예측을 수행한다.
    """

    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=IOU_THRESHOLD,
        imgsz=imgsz,
        verbose=False,
        save=False,
    )

    return results[0]


def select_best_box_per_class(result):
    """
    같은 class가 여러 개 검출되면
    confidence가 가장 높은 박스 1개만 선택한다.
    """

    selected = {}

    boxes = result.boxes

    if boxes is None:
        return selected

    if len(boxes) == 0:
        return selected

    xyxy_array = (
        boxes.xyxy
        .cpu()
        .numpy()
    )

    class_array = (
        boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )

    confidence_array = (
        boxes.conf
        .cpu()
        .numpy()
    )

    for xyxy, class_id, confidence in zip(
        xyxy_array,
        class_array,
        confidence_array,
    ):

        if class_id not in REQUIRED_CLASS_IDS:
            continue

        x1, y1, x2, y2 = xyxy

        candidate = {
            "class_id": int(class_id),
            "class_name": CLASS_NAMES[int(class_id)],
            "confidence": float(confidence),
            "x1_px": int(round(x1)),
            "y1_px": int(round(y1)),
            "x2_px": int(round(x2)),
            "y2_px": int(round(y2)),
        }

        previous = selected.get(class_id)

        if previous is None:
            selected[class_id] = candidate

        elif (
            candidate["confidence"]
            >
            previous["confidence"]
        ):
            selected[class_id] = candidate

    return selected


def clip_box_to_image(
    box,
    image_width,
    image_height,
):
    """
    박스 좌표가 이미지 범위를 벗어나지 않도록 제한한다.
    """

    x1 = max(
        0,
        min(
            box["x1_px"],
            image_width - 1,
        ),
    )

    y1 = max(
        0,
        min(
            box["y1_px"],
            image_height - 1,
        ),
    )

    x2 = max(
        1,
        min(
            box["x2_px"],
            image_width,
        ),
    )

    y2 = max(
        1,
        min(
            box["y2_px"],
            image_height,
        ),
    )

    return x1, y1, x2, y2


def draw_selected_boxes(
    image,
    selected_boxes,
):
    """
    최종 선택된 ROI를 원본 이미지에 그린다.
    """

    annotated = image.copy()

    for class_id in sorted(
        selected_boxes.keys()
    ):

        box = selected_boxes[class_id]

        x1 = box["x1_px"]
        y1 = box["y1_px"]
        x2 = box["x2_px"]
        y2 = box["y2_px"]

        label_text = (
            f'{box["class_name"]} '
            f'{box["confidence"]:.3f}'
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3,
        )

        cv2.putText(
            annotated,
            label_text,
            (
                x1,
                max(
                    25,
                    y1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def save_roi_crop(
    image,
    box,
    output_path,
):
    """
    원본 이미지에서 ROI를 잘라 저장한다.
    """

    image_height, image_width = (
        image.shape[:2]
    )

    x1, y1, x2, y2 = (
        clip_box_to_image(
            box,
            image_width,
            image_height,
        )
    )

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return False

    cv2.imwrite(
        str(output_path),
        crop,
    )

    return True


# =========================================================
# 5. MAIN
# =========================================================

def main():

    prepare_output_folders()

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nModel not found:\n"
            f"{MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            "\nImage folder not found:\n"
            f"{IMAGE_DIR}"
        )

    image_paths = find_n_images()

    print("=" * 70)
    print("N DATASET ROI PREDICTION")
    print("=" * 70)

    print(
        f"Model:\n{MODEL_PATH}"
    )

    print(
        f"\nInput folder:\n{IMAGE_DIR}"
    )

    print(
        f"\nN images found: "
        f"{len(image_paths)}"
    )

    if len(image_paths) == 0:

        raise RuntimeError(
            "No N images were found."
        )

    print(
        "\nLoading YOLO model..."
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    roi_rows = []
    image_summary_rows = []

    primary_success_count = 0
    retry_success_count = 0
    final_failure_count = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "  [ERROR] "
                "Could not read image"
            )

            image_summary_rows.append({
                "filename": image_path.name,
                "prediction_status": "image_read_error",
                "prediction_stage": "none",
                "xiphoid_detected": 0,
                "height_detected": 0,
                "all_required_roi_detected": 0,
                "note": "cv2.imread failed",
            })

            final_failure_count += 1

            continue

        image_height, image_width = (
            image.shape[:2]
        )

        # -----------------------------------------
        # 1차 예측
        # -----------------------------------------

        primary_result = run_prediction(
            model=model,
            image_path=image_path,
            conf=PRIMARY_CONF,
            imgsz=PRIMARY_IMGSZ,
        )

        selected_boxes = (
            select_best_box_per_class(
                primary_result
            )
        )

        missing_classes = (
            REQUIRED_CLASS_IDS
            -
            set(
                selected_boxes.keys()
            )
        )

        prediction_stage = "primary"

        # -----------------------------------------
        # ROI가 하나라도 누락되면 2차 재예측
        # -----------------------------------------

        if missing_classes:

            print(
                "  [RETRY] "
                "Missing ROI after primary prediction"
            )

            retry_result = run_prediction(
                model=model,
                image_path=image_path,
                conf=RETRY_CONF,
                imgsz=RETRY_IMGSZ,
            )

            retry_boxes = (
                select_best_box_per_class(
                    retry_result
                )
            )

            # 재예측 결과가 있으면
            # 해당 class의 박스를 사용한다.
            for class_id, retry_box in (
                retry_boxes.items()
            ):

                current_box = (
                    selected_boxes.get(
                        class_id
                    )
                )

                if current_box is None:

                    selected_boxes[
                        class_id
                    ] = retry_box

                elif (
                    retry_box["confidence"]
                    >
                    current_box["confidence"]
                ):

                    selected_boxes[
                        class_id
                    ] = retry_box

            prediction_stage = "retry"

        # -----------------------------------------
        # 최종 검출 상태
        # -----------------------------------------

        detected_classes = set(
            selected_boxes.keys()
        )

        final_missing_classes = (
            REQUIRED_CLASS_IDS
            -
            detected_classes
        )

        all_detected = (
            len(
                final_missing_classes
            )
            ==
            0
        )

        if all_detected:

            prediction_status = "pass"

            if (
                prediction_stage
                ==
                "primary"
            ):

                primary_success_count += 1

            else:

                retry_success_count += 1

        else:

            prediction_status = (
                "roi_missing"
            )

            final_failure_count += 1

            print(
                "  [FAIL] "
                f"Missing class IDs: "
                f"{sorted(final_missing_classes)}"
            )

        # -----------------------------------------
        # ROI 좌표 CSV 행 생성
        # -----------------------------------------

        for class_id in sorted(
            selected_boxes.keys()
        ):

            box = selected_boxes[
                class_id
            ]

            box_width = (
                box["x2_px"]
                -
                box["x1_px"]
            )

            box_height = (
                box["y2_px"]
                -
                box["y1_px"]
            )

            roi_rows.append({

                "filename":
                    image_path.name,

                "subject_id":
                    image_path.name.split(
                        "_"
                    )[0],

                "class_id":
                    class_id,

                "class_name":
                    box["class_name"],

                "confidence":
                    round(
                        box["confidence"],
                        6,
                    ),

                "x1_px":
                    box["x1_px"],

                "y1_px":
                    box["y1_px"],

                "x2_px":
                    box["x2_px"],

                "y2_px":
                    box["y2_px"],

                "box_width_px":
                    box_width,

                "box_height_px":
                    box_height,

                "image_width_px":
                    image_width,

                "image_height_px":
                    image_height,

                "prediction_stage":
                    prediction_stage,

                "image_qc_status":
                    prediction_status,
            })

        # -----------------------------------------
        # ROI crop 저장
        # -----------------------------------------

        if 0 in selected_boxes:

            xiphoid_output = (
                XIPHOID_CROP_DIR
                /
                image_path.name
            )

            save_roi_crop(
                image=image,
                box=selected_boxes[0],
                output_path=xiphoid_output,
            )

        if 1 in selected_boxes:

            height_output = (
                HEIGHT_CROP_DIR
                /
                image_path.name
            )

            save_roi_crop(
                image=image,
                box=selected_boxes[1],
                output_path=height_output,
            )

        # -----------------------------------------
        # 확인용 annotated image 저장
        # -----------------------------------------

        annotated = draw_selected_boxes(
            image=image,
            selected_boxes=selected_boxes,
        )

        annotated_output = (
            ANNOTATED_DIR
            /
            image_path.name
        )

        cv2.imwrite(
            str(annotated_output),
            annotated,
        )

        # -----------------------------------------
        # 이미지 단위 QC 결과
        # -----------------------------------------

        missing_names = [
            CLASS_NAMES[class_id]
            for class_id
            in sorted(
                final_missing_classes
            )
        ]

        image_summary_rows.append({

            "filename":
                image_path.name,

            "subject_id":
                image_path.name.split(
                    "_"
                )[0],

            "prediction_status":
                prediction_status,

            "prediction_stage":
                prediction_stage,

            "xiphoid_detected":
                int(
                    0
                    in
                    selected_boxes
                ),

            "height_detected":
                int(
                    1
                    in
                    selected_boxes
                ),

            "all_required_roi_detected":
                int(
                    all_detected
                ),

            "missing_roi":
                ";".join(
                    missing_names
                ),

            "note":
                "",
        })

    # =====================================================
    # CSV SAVE
    # =====================================================

    roi_csv_path = (
        CSV_DIR
        /
        "N_predicted_roi_coordinates.csv"
    )

    summary_csv_path = (
        CSV_DIR
        /
        "N_prediction_qc_summary.csv"
    )

    roi_fieldnames = [

        "filename",
        "subject_id",

        "class_id",
        "class_name",

        "confidence",

        "x1_px",
        "y1_px",
        "x2_px",
        "y2_px",

        "box_width_px",
        "box_height_px",

        "image_width_px",
        "image_height_px",

        "prediction_stage",
        "image_qc_status",
    ]

    with roi_csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=roi_fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            roi_rows
        )

    summary_fieldnames = [

        "filename",
        "subject_id",

        "prediction_status",
        "prediction_stage",

        "xiphoid_detected",
        "height_detected",

        "all_required_roi_detected",

        "missing_roi",

        "note",
    ]

    with summary_csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            image_summary_rows
        )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("PREDICTION FINISHED")
    print("=" * 70)

    print(
        f"Total N images: "
        f"{len(image_paths)}"
    )

    print(
        f"Primary success: "
        f"{primary_success_count}"
    )

    print(
        f"Retry success: "
        f"{retry_success_count}"
    )

    print(
        f"Final ROI failure: "
        f"{final_failure_count}"
    )

    print()
    print(
        "ROI coordinate CSV:"
    )

    print(
        roi_csv_path
    )

    print()
    print(
        "QC summary CSV:"
    )

    print(
        summary_csv_path
    )

    print()
    print(
        "Annotated images:"
    )

    print(
        ANNOTATED_DIR
    )

    print()
    print(
        "ROI crops:"
    )

    print(
        OUTPUT_ROOT
        /
        "crops"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()