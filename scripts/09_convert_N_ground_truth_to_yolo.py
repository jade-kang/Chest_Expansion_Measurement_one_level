import json
import re
import shutil
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path.home() / "Desktop" / "CE_Xiphoid_Project"

EXPORT_JSON = (
    PROJECT_ROOT
    / "exports"
    / "N_external_ground_truth_2roi.json"
)

FRAMES_DIR = PROJECT_ROOT / "frames_all"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "external_test_dataset"
)

IMAGE_OUTPUT = OUTPUT_ROOT / "images" / "test"
LABEL_OUTPUT = OUTPUT_ROOT / "labels" / "test"

CLASS_MAP = {
    "xiphoid_roi": 0,
    "height_roi": 1,
}

EXPECTED_TASKS = 214


def recover_original_filename(task):
    """
    Label Studio의 data 필드에서
    N001_front_rep1_insp_01.jpg 같은 원래 파일명을 찾는다.

    업로드 파일에 UUID prefix가 붙어 있어도
    N001_ 이후 부분을 복구한다.
    """

    data = task.get("data", {})

    candidate_values = []

    for key in [
        "filename",
        "image",
        "file_upload",
    ]:
        value = data.get(key)

        if value:
            candidate_values.append(str(value))

    pattern = re.compile(
        r"(N\d{3}_.+?\.(?:jpg|jpeg|png))",
        re.IGNORECASE,
    )

    for value in candidate_values:
        match = pattern.search(value)

        if match:
            return match.group(1)

    return None


def get_annotation_results(task):
    annotations = task.get("annotations", [])

    if not annotations:
        return []

    # 현재 프로젝트에서 최종 annotation 1개 사용
    return annotations[0].get("result", [])


def convert_bbox(value):
    """
    Label Studio percentage 좌표를
    YOLO normalized 좌표로 변환한다.
    """

    x = float(value["x"]) / 100.0
    y = float(value["y"]) / 100.0
    width = float(value["width"]) / 100.0
    height = float(value["height"]) / 100.0

    x_center = x + width / 2.0
    y_center = y + height / 2.0

    return (
        x_center,
        y_center,
        width,
        height,
    )


def reset_output():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    IMAGE_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LABEL_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_yaml():
    yaml_path = OUTPUT_ROOT / "external_test.yaml"

    yaml_text = f"""path: {OUTPUT_ROOT}

train: images/test
val: images/test
test: images/test

names:
  0: xiphoid_roi
  1: height_roi
"""

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8",
    )


def main():
    if not EXPORT_JSON.exists():
        raise FileNotFoundError(
            f"JSON not found:\n{EXPORT_JSON}"
        )

    reset_output()

    with EXPORT_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        tasks = json.load(file)

    print("=" * 60)
    print("N EXTERNAL GROUND TRUTH → YOLO")
    print("=" * 60)

    print(f"JSON tasks: {len(tasks)}")

    if len(tasks) != EXPECTED_TASKS:
        print(
            f"[WARNING] Expected "
            f"{EXPECTED_TASKS} tasks, "
            f"but found {len(tasks)}"
        )

    label_counter = Counter()

    converted_images = 0
    bad_tasks = []

    for task in tasks:
        filename = recover_original_filename(task)

        if filename is None:
            bad_tasks.append(
                ("filename_not_found", task.get("id"))
            )
            continue

        source_image = FRAMES_DIR / filename

        if not source_image.exists():
            bad_tasks.append(
                ("image_not_found", filename)
            )
            continue

        results = get_annotation_results(task)

        yolo_lines = []

        task_label_counter = Counter()

        for result in results:
            value = result.get("value", {})

            labels = value.get(
                "rectanglelabels",
                [],
            )

            if not labels:
                continue

            label = labels[0]

            if label not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[label]

            (
                x_center,
                y_center,
                width,
                height,
            ) = convert_bbox(value)

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

            label_counter[label] += 1
            task_label_counter[label] += 1

        expected_ok = (
            task_label_counter["xiphoid_roi"] == 1
            and task_label_counter["height_roi"] == 1
        )

        if not expected_ok:
            bad_tasks.append(
                (
                    filename,
                    dict(task_label_counter),
                )
            )
            continue

        destination_image = (
            IMAGE_OUTPUT
            / filename
        )

        destination_label = (
            LABEL_OUTPUT
            / f"{Path(filename).stem}.txt"
        )

        shutil.copy2(
            source_image,
            destination_image,
        )

        destination_label.write_text(
            "\n".join(yolo_lines) + "\n",
            encoding="utf-8",
        )

        converted_images += 1

    write_yaml()

    print()
    print(f"Converted images: {converted_images}")

    print()
    print("Label counts:")

    for label, count in label_counter.items():
        print(f"  {label}: {count}")

    print()
    print(f"Bad tasks: {len(bad_tasks)}")

    if bad_tasks:
        print("First bad tasks:")

        for item in bad_tasks[:20]:
            print(f"  {item}")

    print()
    print(
        f"Output:\n{OUTPUT_ROOT}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()