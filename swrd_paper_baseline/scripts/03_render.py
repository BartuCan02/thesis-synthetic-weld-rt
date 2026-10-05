"""Step 3b — cut the selected tiles out of the 16-bit TIFFs, preprocess them (Sect. 3.3), write YOLO files.

For every tile in the chosen split file: crop the window from the raw image, apply
  contrast stretch (percentiles p_lo..p_hi of the tile)  ->  8-bit  ->  CLAHE(clip, grid)
and save it as PNG under the Ultralytics layout, with one label .txt per tile (empty file for negatives):

  <out-dir>/images/{train,val}/<tile_id>.png
  <out-dir>/labels/{train,val}/<tile_id>.txt
  <out-dir>/swrd6.yaml                 data file with absolute paths
  <out-dir>/render_params.json         every parameter used, for the ClearML metadata

Paper saves "24-bit three-channel" images. A single-channel PNG loads as three identical channels in
Ultralytics (cv2.imread colour mode), so the tensors are identical; --three-channel writes the literal
format at 3x the size.

Run (one split file at a time; the film split re-uses the same PNGs through a second label/list layout):
  uv run python scripts/03_render.py --raw-dir ~/swrd_paper_baseline/data/raw --work-dir ~/swrd_paper_baseline/data/work \
      --split split_tiles.json --out-dir ~/swrd_paper_baseline/data/yolo_tilesplit --p-lo 0.5 --p-hi 99.5 --clahe-clip 2.0 --clahe-grid 8
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml
from _common import CLASS_NAMES, preprocess_tile, read_tif, write_json
from tqdm import tqdm


def render_image(args: tuple) -> int:
    """All tiles of one source image, so the big TIFF is decoded once."""
    image_rel, tiles, raw_dir, out_dir, p_lo, p_hi, clip, grid, three = args
    img16 = read_tif(Path(raw_dir) / image_rel)
    n = 0
    for t in tiles:
        x, y, s = t["x"], t["y"], t["side"]
        tile16 = img16[y : y + s, x : x + s]
        if tile16.shape != (s, s):
            raise RuntimeError(
                f"{t['tile_id']}: window {tile16.shape} does not fit image {img16.shape}"
            )
        tile8 = preprocess_tile(tile16, p_lo, p_hi, clip, grid)
        if three:
            tile8 = cv2.merge([tile8, tile8, tile8])
        part = t["part"]
        img_path = Path(out_dir) / "images" / part / f"{t['tile_id']}.png"
        lab_path = Path(out_dir) / "labels" / part / f"{t['tile_id']}.txt"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        lab_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(img_path), tile8):
            raise RuntimeError(f"could not write {img_path}")
        lab_path.write_text("\n".join(t["boxes"]) + ("\n" if t["boxes"] else ""), encoding="utf-8")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--split", default="split_tiles.json", help="split file name inside work-dir")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--p-lo",
        type=float,
        default=0.5,
        help="D5 lower percentile of the contrast stretch (0 = min)",
    )
    ap.add_argument("--p-hi", type=float, default=99.5, help="D5 upper percentile (100 = max)")
    ap.add_argument("--clahe-clip", type=float, default=2.0, help="D6 CLAHE clip limit")
    ap.add_argument("--clahe-grid", type=int, default=8, help="D6 CLAHE tile grid (n x n)")
    ap.add_argument("--three-channel", action="store_true", help="write literal 24-bit PNGs")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    split = json.loads((args.work_dir / args.split).read_text(encoding="utf-8"))
    part_of = {i: "train" for i in split["train"]} | {i: "val" for i in split["val"]}

    by_image: dict[str, list[dict]] = defaultdict(list)
    with open(args.work_dir / "tiles.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t["tile_id"] in part_of:
                t["part"] = part_of[t["tile_id"]]
                by_image[t["image"]].append(t)
    n_sel = sum(len(v) for v in by_image.values())
    assert n_sel == len(part_of), f"{len(part_of) - n_sel} split tiles missing from tiles.jsonl"
    print(f"[render] {n_sel:,} tiles from {len(by_image)} images -> {args.out_dir}")

    jobs = [
        (
            img,
            tiles,
            str(args.raw_dir),
            str(args.out_dir),
            args.p_lo,
            args.p_hi,
            args.clahe_clip,
            args.clahe_grid,
            args.three_channel,
        )
        for img, tiles in sorted(by_image.items())
    ]
    done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for n in tqdm(ex.map(render_image, jobs, chunksize=2), total=len(jobs), unit="img"):
            done += n

    data_yaml = {
        "path": str(args.out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    (args.out_dir / "swrd6.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True)
    )
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    params["split_counts"] = split.get("counts")
    params["split_params"] = split.get("params")
    write_json(args.out_dir / "render_params.json", params)

    sizes = np.array([t["side"] for v in by_image.values() for t in v])
    print(
        f"[render] wrote {done:,} tiles; side px median/min/max {int(np.median(sizes))}/{sizes.min()}/{sizes.max()}"
    )
    print(f"[render] data file: {args.out_dir / 'swrd6.yaml'}")


if __name__ == "__main__":
    main()
