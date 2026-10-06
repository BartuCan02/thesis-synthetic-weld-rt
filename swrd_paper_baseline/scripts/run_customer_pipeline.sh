#!/usr/bin/env bash
# Customer films -> paper-style tiles -> ClearML child dataset of SWRD v1.0 (train split only).
# Steps 1-4 need only this project's venv and the CPU box. Step 0 (export_customer_films.py) must have run
# first in the deeplify mlops env — see that script's docstring.
#
#   bash scripts/run_customer_pipeline.sh ~/swrd_paper_baseline/data/raw_customer customer_v1
#
# Tiling parameters are the ones recorded in the v1.0 dataset's metadata (de772ad9…): edge drop,
# D7 = 4 px / 10 % / 0.5 %, D5 = P0.5–P99.5 stretch, D6 = CLAHE 2.0 / 8x8.
set -euo pipefail
RAW=${1:?raw customer dir (output of export_customer_films.py)}
TAG=${2:-customer_v1}
DATA=$(dirname "$RAW")
WORK=$DATA/work_$TAG
YOLO=$DATA/yolo_$TAG
PARENT=${PARENT_ID:-de772ad9363c4067bed5835e13a9be81}
cd "$(dirname "$0")/.."

echo "== 1/4 inventory"
uv run python scripts/00_inventory.py --raw-dir "$RAW" --work-dir "$WORK"
echo "== 2/4 tile (other_defect tiles are never negatives)"
uv run python scripts/01_tile.py --raw-dir "$RAW" --work-dir "$WORK" --edge drop \
  --min-side-px 4 --min-visible-frac 0.10 --min-tile-frac 0.005 --exclude-from-negatives other_defect
echo "== 3/4 select: all positives + equal negatives, everything to train"
uv run python scripts/02_select_split.py --work-dir "$WORK" --seed 0 --val-ratio 0
echo "== 4/4 render"
uv run python scripts/03_render.py --raw-dir "$RAW" --work-dir "$WORK" --split split_tiles.json \
  --out-dir "$YOLO" --p-lo 0.5 --p-hi 99.5 --clahe-clip 2.0 --clahe-grid 8
echo "== done. Next (needs S3 write):"
echo "AWS_PROFILE=data-rw uv run python scripts/07_merge_upload.py --parent-id $PARENT --customer-yolo-dir $YOLO --customer-work-dir $WORK --raw-customer-dir $RAW"
