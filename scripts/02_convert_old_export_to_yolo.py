import json
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image

PROJECT_ROOT = Path("/Users/kangjun2000/Desktop/CE_Xiphoid_Project")

EXPORT_JSON = PROJECT_ROOT / "exports" / "old_4lev_filtered_2roi.json"
FRAMES_DIR = PROJECT_ROOT / "frames_all"
YOLO_DIR = PROJECT_ROOT / "yolo_dataset"

CLASS_MAP = {
    "xiphoid_roi": 0,
    "height_roi": 1,
}

TRAIN_SUBJECTS = {f"S{i:03d}" for i in range(1, 15)}   # S001~S014
VAL_SUBJECTS = {f"S{i:03d}" for i in range(15, 19)}    # S015~S018
EXCLUDED_SUBJECTS = {"S019"}


def recover_filename(task):
    data = task.get("data", {})

    candidates = []
    for key in ["filename", "image", "file_upload"]:
        value = data.get(key)
        if value:
            candidates.append(str(value))

    for value in candidates:
        name = Path(value).name

        if "-" in name:
            maybe = name.split("-", 1)[1]
            if maybe.startswith(("S", "N")):
                return maybe

        if name.startswith(("S", "N")):
            return name

    raise ValueError(f"Could not recover filename from task data: {data}")


def get_split(subject_id):
    if subject_id in TRAIN_SUBJECTS:
        return "train"
    if subject_id in VAL_SUBJECTS:
        return "val"
    if subject_id in EXCLUDED_SUBJECTS:
        return "exclude"
    return "exclude"


def ls_bbox_to_yolo(value):
    # Label Studio rectangle 값은 보통 퍼센트 단위
    x = value["x"] / 100.0
    y = value["y"] / 100.0
    w = value["width"] / 100.0
    h = value["height"] / 100.0

    x_center = x + w / 2.0
    y_center = y + h / 2.0

    return x_center, y_center, w, h


def prepare_dirs():
    if YOLO_DIR.exists():
        print(f"[INFO] Removing existing YOLO dataset: {YOLO_DIR}")
        shutil.rmtree(YOLO_DIR)

    for split in ["train", "val"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml():
    yaml_path = YOLO_DIR / "xiphoid_roi.yaml"
    content = f"""path: {YOLO_DIR}
train: images/train
val: images/val

names:
  0: xiphoid_roi
  1: height_roi
"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[INFO] Saved YAML: {yaml_path}")


def main():
    prepare_dirs()

    with open(EXPORT_JSON, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    split_counter = Counter()
    label_counter = Counter()
    skipped = []

    for task in tasks:
        filename = recover_filename(task)
        subject_id = filename.split("_")[0]
        split = get_split(subject_id)

        if split == "exclude":
            skipped.append(filename)
            continue

        img_path = FRAMES_DIR / filename
        if not img_path.exists():
            print(f"[WARNING] image missing: {img_path}")
            skipped.append(filename)
            continue

        # 이미지가 깨졌는지 확인
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            print(f"[WARNING] invalid image: {img_path}, {e}")
            skipped.append(filename)
            continue

        yolo_lines = []

        anns = task.get("annotations", [])
        for ann in anns:
            for r in ann.get("result", []):
                value = r.get("value", {})
                labels = value.get("rectanglelabels", [])

                if not labels:
                    continue

                label = labels[0]

                if label not in CLASS_MAP:
                    continue

                cls_id = CLASS_MAP[label]
                x_c, y_c, w, h = ls_bbox_to_yolo(value)

                yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                label_counter[label] += 1

        if len(yolo_lines) != 2:
            print(f"[WARNING] {filename}: expected 2 labels, got {len(yolo_lines)}")

        dst_img = YOLO_DIR / "images" / split / filename
        dst_label = YOLO_DIR / "labels" / split / f"{Path(filename).stem}.txt"

        shutil.copy2(img_path, dst_img)
        dst_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        split_counter[split] += 1

    write_yaml()

    print("\nDone.")
    print("Images by split:")
    for k, v in split_counter.items():
        print(f"  {k}: {v}")

    print("\nLabels:")
    for k, v in label_counter.items():
        print(f"  {k}: {v}")

    print(f"\nSkipped: {len(skipped)}")


if __name__ == "__main__":
    main()