from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.home() / "Desktop" / "CE_Xiphoid_Project"

FRAMES_DIR = PROJECT_ROOT / "frames_all"
META_DIR = PROJECT_ROOT / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = META_DIR / "frame_metadata.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

records = []

for img_path in sorted(FRAMES_DIR.iterdir()):
    if img_path.suffix.lower() not in IMAGE_EXTS:
        continue

    filename = img_path.name
    stem = img_path.stem
    parts = stem.split("_")

    if len(parts) != 5:
        print(f"[WARNING] filename parse failed: {filename}")
        continue

    subject_id, view, repetition, phase, frame_no = parts

    # source 구분
    if subject_id.startswith("S"):
        source_dataset = "legacy_CE_Pilot"
    elif subject_id.startswith("N"):
        source_dataset = "new_xiphoid_dataset"
    elif subject_id.startswith("X"):
        source_dataset = "new_xiphoid_dataset"
    else:
        source_dataset = "unknown"

    # split 기본값
    if subject_id in [f"S{i:03d}" for i in range(1, 19)]:
        split_plan = "train_tune"
        include_for_training = True
        include_for_final_validation = False
    elif subject_id in [f"N{i:03d}" for i in range(1, 5)] or subject_id in [f"X{i:03d}" for i in range(1, 5)]:
        split_plan = "train_tune"
        include_for_training = True
        include_for_final_validation = False
    elif subject_id in [f"N{i:03d}" for i in range(5, 10)] or subject_id in [f"X{i:03d}" for i in range(5, 10)]:
        split_plan = "final_validation"
        include_for_training = False
        include_for_final_validation = True
    elif subject_id == "S019":
        split_plan = "excluded"
        include_for_training = False
        include_for_final_validation = False
    else:
        split_plan = "check"
        include_for_training = False
        include_for_final_validation = False

    records.append({
        "filename": filename,
        "subject_id": subject_id,
        "view": view,
        "repetition": repetition,
        "phase": phase,
        "frame_no": frame_no,
        "source_dataset": source_dataset,
        "split_plan": split_plan,
        "include_for_training": include_for_training,
        "include_for_final_validation": include_for_final_validation,
        "image_path": str(img_path),
        "note": "",
    })

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"Saved: {OUT_CSV}")
print(f"Total images: {len(df)}")
print("\nCounts by source:")
print(df.groupby("source_dataset").size())
print("\nCounts by split:")
print(df.groupby("split_plan").size())
print("\nCounts by subject:")
print(df.groupby("subject_id").size())
