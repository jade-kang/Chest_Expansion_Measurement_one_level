from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path.home() / "Desktop" / "CE_Xiphoid_Project"

RAW_DIR = PROJECT_ROOT / "frames_raw"
ALL_DIR = PROJECT_ROOT / "frames_all"
META_DIR = PROJECT_ROOT / "metadata"

ALL_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# 각 front/side 폴더 안 12장 순서
ORDER = [
    ("rep1", "insp", "01"),
    ("rep1", "insp", "02"),
    ("rep1", "exp", "01"),
    ("rep1", "exp", "02"),
    ("rep2", "insp", "01"),
    ("rep2", "insp", "02"),
    ("rep2", "exp", "01"),
    ("rep2", "exp", "02"),
    ("rep3", "insp", "01"),
    ("rep3", "insp", "02"),
    ("rep3", "exp", "01"),
    ("rep3", "exp", "02"),
]

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".heic", ".HEIC"]

records = []

subjects = [f"X{i:03d}" for i in range(1, 10)]

for subject_id in subjects:
    for view in ["front", "side"]:
        view_dir = RAW_DIR / subject_id / view

        if not view_dir.exists():
            print(f"[WARNING] Missing folder: {view_dir}")
            continue

        image_files = sorted([
            p for p in view_dir.iterdir()
            if p.suffix in IMAGE_EXTS
        ])

        print(f"{subject_id} {view}: {len(image_files)} images")

        if len(image_files) != 12:
            print(f"[WARNING] {subject_id} {view}: expected 12 images, found {len(image_files)}")

        for idx, img_path in enumerate(image_files):
            if idx >= 12:
                print(f"[WARNING] Extra image skipped: {img_path}")
                continue

            repetition, phase, frame_no = ORDER[idx]

            new_filename = f"{subject_id}_{view}_{repetition}_{phase}_{frame_no}.jpg"
            new_path = ALL_DIR / new_filename

            if new_path.exists():
                print(f"[WARNING] Overwriting existing file: {new_path}")

            shutil.copy2(img_path, new_path)

            records.append({
                "filename": new_filename,
                "subject_id": subject_id,
                "view": view,
                "repetition": repetition,
                "phase": phase,
                "frame_no": frame_no,
                "source_dataset": "xiphoid_new_3rep",
                "original_filename": img_path.name,
                "original_path": str(img_path),
                "new_path": str(new_path),
                "include_for_training": True,
                "include_for_analysis": True,
                "note": "",
            })

df = pd.DataFrame(records)

out_csv = META_DIR / "new_xiphoid_frame_metadata.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("\nDone.")
print(f"Copied new images: {len(df)}")
print(f"Saved metadata: {out_csv}")

if len(df) > 0:
    print("\nCounts by subject/view:")
    print(df.groupby(["subject_id", "view"]).size())

    print("\nCounts by view/phase:")
    print(df.groupby(["view", "phase"]).size())
