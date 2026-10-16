#!/usr/bin/env bash
# Synthetic films -> paper-style tiles -> ClearML child dataset of SWRD v1.0 (train only).
# Runs on the data box, CPU only. Tiling and rendering use the baseline's own scripts with the parameters
# recorded for the v1.0 dataset (de772ad9...): edge drop, D7 = 4 px / 10 % / 0.5 %, P0.5-P99.5 stretch,
# CLAHE 2.0 / 8x8, single channel.
#
#   bash run_pipeline.sh physical v2                      # run C's pipeline (version 2, RFS-matched budget)
#   TOPUP=1 bash run_pipeline.sh physical v2              # fill a tile shortfall of a v2 build
#   bash run_pipeline.sh physical v3                      # version 3 (2026-10-16): 3x budget, geometry placement,
#                                                         #   varied defects, null implants; top-up is automatic;
#                                                         #   UPLOAD=1 registers the ClearML dataset at the end
#   LIMIT=60 bash run_pipeline.sh physical v3smoke        # quick check of the v3 path on a few films
#
# A tag starting with "v3" selects the version-3 options. Environment overrides: PY, SWRD_RAW, SWRD_WORK, DATA,
# BUDGET_SCALE (default 3 for v3), PER_FILM, NULL_PROB, UPLOAD, DATASET_VERSION, WORKERS.
set -euo pipefail
ARM=${1:-physical}
TAG=${2:-v2}
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
RESULTS=$REPO/swrd_synthetic_physical/results
LIMIT=${LIMIT:-0}
WORKERS=${WORKERS:-4}
PARENT=de772ad9363c4067bed5835e13a9be81

V3=0
case "$TAG" in v3*) V3=1 ;; esac
if [ "$V3" = 1 ]; then
  BUDGET_SCALE=${BUDGET_SCALE:-3}
  BUDGET=$RESULTS/budget_v3.json
  POSITIONS=$DATA/positions/positions_real.json
  MAKE_OPTS=(--version 3 --positions "$POSITIONS" --per-film "${PER_FILM:-5}" --null-prob "${NULL_PROB:-0.5}")
  DATASET_VERSION=${DATASET_VERSION:-3.0.0}
else
  BUDGET_SCALE=1
  BUDGET=$RESULTS/budget.json
  MAKE_OPTS=(--version 2)
  DATASET_VERSION=${DATASET_VERSION:-2.0.1}
fi
echo "== pipeline $ARM $TAG (version $((V3 + 2)), budget x$BUDGET_SCALE) started $(date)"

echo "== 0/6 grey steps + budget"
"$PY" "$HERE/00_grey_steps.py" --raw "$SWRD_RAW" --work-dir "$SWRD_WORK" --out "$RESULTS"
"$PY" "$HERE/01_budget.py" --work-dir "$SWRD_WORK" --out "$RESULTS" --scale "$BUDGET_SCALE" --name "$(basename "$BUDGET")"
if [ "$V3" = 1 ] && [ ! -f "$POSITIONS" ]; then
  echo "== 0b/6 across-weld position of every real defect on the training films"
  "$PY" "$HERE/05_measure_positions.py" --raw "$SWRD_RAW" --work-dir "$SWRD_WORK" --grey-steps "$RESULTS/grey_steps.json" \
    --out "$DATA/positions" --name positions_real --workers "$WORKERS"
fi

tile_select_render() {
  rm -rf "$WORK" "$YOLO"
  echo "== 2/6 inventory + tiles (baseline scripts, v1.0 parameters)"
  (cd "$BASE" && "$PY" 00_inventory.py --raw-dir "$RAW" --work-dir "$WORK" --workers "$WORKERS")
  (cd "$BASE" && "$PY" 01_tile.py --raw-dir "$RAW" --work-dir "$WORK" --edge drop \
    --min-side-px 4 --min-visible-frac 0.10 --min-tile-frac 0.005 \
    --exclude-from-negatives 焊瘤 内凹 --workers "$WORKERS")
  echo "== 3/6 keep the tiles that hold an inserted defect (and the null-implant tiles)"
  "$PY" "$HERE/03_select_tiles.py" --raw "$RAW" --work-dir "$WORK" --budget "$BUDGET" \
    --split-file "$SWRD_WORK/split_films.json"
  echo "== 4/6 render (baseline script, v1.0 parameters)"
  (cd "$BASE" && "$PY" 03_render.py --raw-dir "$RAW" --work-dir "$WORK" --split split_tiles.json \
    --out-dir "$YOLO" --p-lo 0.5 --p-hi 99.5 --clahe-clip 2.0 --clahe-grid 8 --workers "$WORKERS")
}

if [ "${TOPUP:-0}" = 1 ]; then
  echo "== 1/6 top-up: more synthetic films ($ARM) until every class reaches its tile target"
  "$PY" "$HERE/02_make_films.py" --raw "$SWRD_RAW" --work-dir "$SWRD_WORK" --budget "$BUDGET" \
    --out "$RAW" --arm "$ARM" --workers "$WORKERS" --seed 1 --topup-from "$WORK/selection_report.json" "${MAKE_OPTS[@]}"
  tile_select_render
else
  echo "== 1/6 synthetic films ($ARM) -> $RAW"
  rm -rf "$RAW"
  "$PY" "$HERE/02_make_films.py" --raw "$SWRD_RAW" --work-dir "$SWRD_WORK" --budget "$BUDGET" \
    --out "$RAW" --arm "$ARM" --limit "$LIMIT" --workers "$WORKERS" "${MAKE_OPTS[@]}"
  tile_select_render
  if [ "$V3" = 1 ] && [ "$LIMIT" = 0 ]; then
    # automatic top-up when a class is short of its tile target (run C needed one pass, 97-102 % after it)
    SHORT=$("$PY" - "$WORK/selection_report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))["per_class"]
print(sum(1 for c, s in r.items() if s["target_extra_tiles"] and s["tiles_with_inserted"] < 0.97 * s["target_extra_tiles"]))
EOF
)
    if [ "$SHORT" != 0 ]; then
      echo "== 1b/6 top-up ($SHORT classes short of 97 % of their tile target)"
      "$PY" "$HERE/02_make_films.py" --raw "$SWRD_RAW" --work-dir "$SWRD_WORK" --budget "$BUDGET" \
        --out "$RAW" --arm "$ARM" --workers "$WORKERS" --seed 1 --topup-from "$WORK/selection_report.json" "${MAKE_OPTS[@]}"
      tile_select_render
    fi
  fi
fi

if [ "$V3" = 1 ]; then
  echo "== 5/6 checks: positions of the inserted defects, comparison sheet"
  "$PY" "$HERE/05_measure_positions.py" --raw "$RAW" --synthetic --out "$RAW" --name positions_synthetic --workers "$WORKERS"
  SYN_ARGS=(--synthetic "$TAG=$RAW")
  if [ -f "$DATA/raw_${ARM}_v2/inserted.jsonl" ]; then
    [ -f "$DATA/raw_${ARM}_v2/positions_synthetic.jsonl" ] || \
      "$PY" "$HERE/05_measure_positions.py" --raw "$DATA/raw_${ARM}_v2" --synthetic --out "$DATA/raw_${ARM}_v2" --name positions_synthetic --workers "$WORKERS"
    SYN_ARGS=(--synthetic "run C (v2)=$DATA/raw_${ARM}_v2" "${SYN_ARGS[@]}")
    PLOT_ARGS=(--synthetic "run C (v2)=$DATA/raw_${ARM}_v2/positions_synthetic.jsonl" --synthetic "$TAG=$RAW/positions_synthetic.jsonl")
  else
    PLOT_ARGS=(--synthetic "$TAG=$RAW/positions_synthetic.jsonl")
  fi
  "$PY" "$HERE/07_position_plot.py" --real "$DATA/positions/positions_real.jsonl" "${PLOT_ARGS[@]}" --out "$RAW/position_check.png"
  "$PY" "$HERE/06_compare_sheet.py" --raw "$SWRD_RAW" --work-dir "$SWRD_WORK" --grey-steps "$RESULTS/grey_steps.json" \
    "${SYN_ARGS[@]}" --n 4 --out "$RAW/compare_sheet.png"
  "$PY" "$HERE/06_compare_sheet.py" --raw "$SWRD_RAW" --work-dir "$SWRD_WORK" --grey-steps "$RESULTS/grey_steps.json" \
    "${SYN_ARGS[@]}" --n 4 --outline --out "$RAW/compare_sheet_labels.png"
fi

if [ "${UPLOAD:-0}" = 1 ]; then
  echo "== 6/6 upload: ClearML child dataset of $PARENT, version $DATASET_VERSION"
  AWS_PROFILE=${AWS_PROFILE:-data-rw} "$PY" "$HERE/04_merge_upload.py" --parent-id $PARENT \
    --yolo-dir "$YOLO" --work-dir "$WORK" --raw-dir "$RAW" --budget "$BUDGET" --arm "$ARM" --version "$DATASET_VERSION"
else
  echo "== 6/6 dry run of the upload (needs S3 read)"
  AWS_PROFILE=${AWS_PROFILE:-data-rw} "$PY" "$HERE/04_merge_upload.py" --parent-id $PARENT \
    --yolo-dir "$YOLO" --work-dir "$WORK" --raw-dir "$RAW" --budget "$BUDGET" --arm "$ARM" --version "$DATASET_VERSION" --dry-run
  echo "== done. Upload with UPLOAD=1, or run the same command without --dry-run."
fi
echo "== pipeline $ARM $TAG finished $(date)"
