import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.home() / "Desktop" / "CE_Xiphoid_Project"

META_CSV = PROJECT_ROOT / "metadata" / "frame_metadata.csv"
OUT_DIR = PROJECT_ROOT / "labelstudio_project"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "xiphoid_labelstudio_tasks.json"

df = pd.read_csv(META_CSV)

# S019 제외
df = df[df["split_plan"] != "excluded"].copy()

tasks = []

for _, row in df.iterrows():
    filename = row["filename"]

    tasks.append({
        "data": {
            "image": f"/data/local-files/?d=frames_all/{filename}",
            "filename": filename,
            "subject_id": row["subject_id"],
            "view": row["view"],
            "repetition": row["repetition"],
            "phase": row["phase"],
            "frame_no": str(row["frame_no"]),
            "split_plan": row["split_plan"],
        }
    })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"Saved: {OUT_JSON}")
print(f"Total tasks: {len(tasks)}")
