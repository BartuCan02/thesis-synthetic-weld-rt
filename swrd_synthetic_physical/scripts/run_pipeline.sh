#!/usr/bin/env bash
# Synthetic films -> paper-style tiles -> ready for a ClearML child dataset of SWRD v1.0 (train only).
# Runs on the data box, CPU only. Tiling and rendering use the baseline's own scripts with the parameters
# recorded for the v1.0 dataset (de772ad9...): edge drop, D7 = 4 px / 10 % / 0.5 %, P0.5-P99.5 stretch,
# CLAHE 2.0 / 8x8, single channel.
#
#   bash swrd_synthetic_physical/scripts/run_pipeline.sh physical v1
#   LIMIT=40 bash swrd_synthetic_physical/scripts/run_pipeline.sh physical smoke   # quick check
set -euo pipefail
ARM=${1:-physical}
TAG=${2:-v1}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BASE=$REPO/swrd_paper_baseline/scripts
PY=${PY:-$HOME/thesis/swrd_paper_baseline/.venv/bin/python}
SWRD_RAW=${SWRD_RAW:-$HOME/swrd_paper_baseline/data/raw}
SWRD_WORK=${SWRD_WORK:-$HOME/swrd_paper_baseline/data/work}
DATA=${DATA:-$HOME/swrd_synthetic_physical/data}
RAW=$DATA/raw_${ARM}_$TAG
WORK=$DATA/work_${ARM}_$TAG
YOLO=$DATA/yolo_${ARM}_$TAG
BUDGET=$REPO/swrd_synthetic_physical/results/budget.json
LIMIT=${LIMIT:-0}

echo "== 0/5 grey steps (find 8-bit films stored as 16-bit) + budget (matched to RFS t=0.1, film split)"
"$PY" "$HERE/00_grey_steps.py" --raw "$SWRD_RAW" --work-dir "$SWRD_WORK" --out "$(dirname "$BUDGET")"
"$PY" "$HERE/01_budget.py" --work-dir "$SWRD_WORK" --out "$(dirname "$BUDGET")"

echo "== 1/5 synthetic films ($ARM) -> $RAW"
rm -rf "$RAW" "$WORK" "$YOLO"
"$PY" "$HERE/02_make_films.py" --raw "$SWRD_RAW" --work-dir "$SWRD_WORK" --budget "$BUDGET" \
  --out "$RAW" --arm "$ARM" --limit "$LIMIT" --workers 4

echo "== 2/5 inventory + tiles (baseline scripts, v1.0 parameters)"
(cd "$BASE" && "$PY" 00_inventory.py --raw-dir "$RAW" --work-dir "$WORK" --workers 4)
(cd "$BASE" && "$PY" 01_tile.py --raw-dir "$RAW" --work-dir "$WORK" --edge drop \
  --min-side-px 4 --min-visible-frac 0.10 --min-tile-frac 0.005 \
  --exclude-from-negatives 焊瘤 内凹 --workers 4)

echo "== 3/5 keep the tiles that hold an inserted defect"
"$PY" "$HERE/03_select_tiles.py" --raw "$RAW" --work-dir "$WORK" --budget "$BUDGET"

echo "== 4/5 render (baseline script, v1.0 parameters)"
(cd "$BASE" && "$PY" 03_render.py --raw-dir "$RAW" --work-dir "$WORK" --split split_tiles.json \
  --out-dir "$YOLO" --p-lo 0.5 --p-hi 99.5 --clahe-clip 2.0 --clahe-grid 8 --workers 4)

echo "== 5/5 dry run of the upload (needs S3 read)"
AWS_PROFILE=${AWS_PROFILE:-data-rw} "$PY" "$HERE/04_merge_upload.py" --parent-id de772ad9363c4067bed5835e13a9be81 \
  --yolo-dir "$YOLO" --work-dir "$WORK" --raw-dir "$RAW" --budget "$BUDGET" --arm "$ARM" --dry-run
echo "== done. Upload with the same command without --dry-run."
