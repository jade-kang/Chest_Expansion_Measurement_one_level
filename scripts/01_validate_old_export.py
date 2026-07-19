import json
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path.home() / "Desktop" / "CE_Xiphoid_Project"
EXPORT_JSON = PROJECT_ROOT / "exports" / "old_4lev_filtered_2roi.json"
FRAMES_DIR = PROJECT_ROOT / "frames_all"

EXPECTED_LABELS = {"xiphoid_roi", "height_roi"}


def recover_filename(task):
    data = task.get("data", {})

    candidates = []

    for key in ["filename", "image", "file_upload"]:
        value = data.get(key)
        if value:
            candidates.append(str(value))

    for value in candidates:
        name = Path(value).name

        # Label Studio upload prefix 제거
        # 예: 8c78478d-S001_front_rep1_insp_01.jpg
        # → S001_front_rep1_insp_01.jpg
        if "-" in name:
            maybe = name.split("-", 1)[1]
            if maybe.startswith(("S", "N")):
                return maybe

        if name.startswith(("S", "N")):
            return name

    return None


def get_labels(task):
    labels = []

    for ann in task.get("annotations", []):
        for result in ann.get("result", []):
            value = result.get("value", {})
            rect_labels = value.get("rectanglelabels", [])
            labels.extend(rect_labels)

    return labels


def main():
    if not EXPORT_JSON.exists():
        raise FileNotFoundError(f"Export JSON not found: {EXPORT_JSON}")

    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames folder not found: {FRAMES_DIR}")

    with open(EXPORT_JSON, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    print("=" * 60)
    print("OLD EXPORT VALIDATION")
    print("=" * 60)
    print(f"Export JSON: {EXPORT_JSON}")
    print(f"Frames dir : {FRAMES_DIR}")
    print(f"Total tasks: {len(tasks)}")

    label_counter = Counter()
    missing_files = []
    bad_tasks = []
    filename_counter = Counter()

    for task in tasks:
        filename = recover_filename(task)
        labels = get_labels(task)

        if filename is None:
            bad_tasks.append(("NO_FILENAME", task.get("id")))
            continue

        filename_counter[filename] += 1

        for label in labels:
            label_counter[label] += 1

        image_path = FRAMES_DIR / filename
        if not image_path.exists():
            missing_files.append(filename)

        counts = Counter(labels)

        unexpected = set(counts.keys()) - EXPECTED_LABELS
        if unexpected:
            bad_tasks.append((filename, f"unexpected labels: {unexpected}"))

        for expected_label in EXPECTED_LABELS:
            if counts[expected_label] != 1:
                bad_tasks.append((filename, f"{expected_label} count = {counts[expected_label]}"))

    duplicate_files = [name for name, count in filename_counter.items() if count > 1]

    print("\nLabel counts:")
    for label, count in label_counter.items():
        print(f"  {label}: {count}")

    print("\nFile check:")
    print(f"  Missing image files: {len(missing_files)}")
    if missing_files:
        print("  First missing examples:")
        for name in missing_files[:10]:
            print(f"    {name}")

    print("\nDuplicate filename check:")
    print(f"  Duplicate filenames: {len(duplicate_files)}")
    if duplicate_files:
        print("  First duplicate examples:")
        for name in duplicate_files[:10]:
            print(f"    {name}")

    print("\nTask quality check:")
    print(f"  Bad tasks: {len(bad_tasks)}")
    if bad_tasks:
        print("  First bad examples:")
        for item in bad_tasks[:20]:
            print(f"    {item}")

    print("\nExpected result:")
    print("  Each image should have exactly:")
    print("    1 xiphoid_roi")
    print("    1 height_roi")
    print("=" * 60)


if __name__ == "__main__":
    main()