#!/bin/bash

set -e

cd /Users/kangjun2000/Desktop/CE_Xiphoid_Project

source xiphoid_env/bin/activate

RUN_ID=$(date +"run_%Y%m%d_%H%M%S")
ARCHIVE_DIR="runs_measurement/${RUN_ID}"

echo "=================================================="
echo "N MEASUREMENT PIPELINE"
echo "RUN_ID: ${RUN_ID}"
echo "ARCHIVE_DIR: ${ARCHIVE_DIR}"
echo "=================================================="

echo ""
echo "Checking required inputs..."

if [ ! -d "frames_all" ]; then
  echo "ERROR: frames_all folder not found"
  exit 1
fi

if [ ! -f "runs/detect/models/xiphoid_yolov8n_old2roi/weights/best.pt" ]; then
  echo "ERROR: best.pt not found"
  exit 1
fi

if [ ! -d "external_test_dataset/labels/test" ]; then
  echo "ERROR: external_test_dataset/labels/test not found"
  exit 1
fi

if [ ! -f "metadata/N_subject_metadata.csv" ]; then
  echo "ERROR: metadata/N_subject_metadata.csv not found"
  exit 1
fi

echo "Required inputs found."

echo ""
echo "Cleaning previous working results..."

rm -rf results/N_roi_predictions
rm -rf results/N_cv_measurements
rm -rf results/N_final_measurements

echo ""
echo "STEP 1. Predict N ROI"
python scripts/12_predict_N_roi.py

echo ""
echo "STEP 2. Audit ROI availability"
python scripts/13_audit_N_roi_availability.py

echo ""
echo "STEP 3. Computer vision width/depth with GT fallback"
python scripts/14_measure_N_width_depth.py

echo ""
echo "STEP 4. Pixel to cm, circumference, expansion"
python scripts/15_convert_pixels_to_circumference.py

echo ""
echo "STEP 5. Final AI tables"
python scripts/17_make_final_AI_tables.py

echo ""
echo "Archiving this run..."

mkdir -p "${ARCHIVE_DIR}"

cp -R results/N_roi_predictions "${ARCHIVE_DIR}/"
cp -R results/N_cv_measurements "${ARCHIVE_DIR}/"
cp -R results/N_final_measurements "${ARCHIVE_DIR}/"

cat > "${ARCHIVE_DIR}/run_info.txt" << EOF
RUN_ID: ${RUN_ID}
DATE: $(date)
PROJECT: CE_Xiphoid_Project

Input model:
runs/detect/models/xiphoid_yolov8n_old2roi/weights/best.pt

Input images:
frames_all/N*.jpg

Ground truth fallback labels:
external_test_dataset/labels/test/*.txt

Subject metadata:
metadata/N_subject_metadata.csv

Pipeline steps:
12_predict_N_roi.py
13_audit_N_roi_availability.py
14_measure_N_width_depth.py
15_convert_pixels_to_circumference.py
17_make_final_AI_tables.py

Final AI tape-format table:
${ARCHIVE_DIR}/N_final_measurements/final_tables/N_AI_tape_format.csv

Detailed repetition table:
${ARCHIVE_DIR}/N_final_measurements/final_tables/N_AI_repetition_detailed.csv

Measurement overlays:
${ARCHIVE_DIR}/N_cv_measurements/measurement_overlays

Body masks:
${ARCHIVE_DIR}/N_cv_measurements/body_masks
EOF

echo ""
echo "=================================================="
echo "PIPELINE DONE"
echo "Saved run:"
echo "${ARCHIVE_DIR}"
echo ""
echo "Final table:"
echo "${ARCHIVE_DIR}/N_final_measurements/final_tables/N_AI_tape_format.csv"
echo ""
echo "Detailed table:"
echo "${ARCHIVE_DIR}/N_final_measurements/final_tables/N_AI_repetition_detailed.csv"
echo "=================================================="