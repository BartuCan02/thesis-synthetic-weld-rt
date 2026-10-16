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

Labels (--labels, 2026-10-16, segmentation arm of WP1):
  box       the D7 boxes stored in tiles.jsonl by 01_tile.py (the paper's detection task; default)
  polygon   Ultralytics segment lines "cls x1 y1 x2 y2 ..." for the SAME instances: every defect polygon is
            rasterised into the window again, the D7 keep rule (read from tiles_summary.json) decides exactly as
            in 01_tile.py, and the box of each kept instance must equal the stored one byte for byte, otherwise
            the script stops. The written polygon is the LabelMe polygon clipped to the window (Sutherland-
            Hodgman, optional Douglas-Peucker with --simplify-px), so the instance set and its boxes are those of
            the detection dataset and only the label geometry differs. Degenerate shapes (1-2 points, or nothing
            left after clipping) fall back to the D7 box as a 4-point polygon and are counted.
--reuse-images DIR  hardlink (or copy) the PNGs of an existing render instead of decoding the TIFFs, so a second
            label format sits on byte-identical images. Only the labels are computed.

Run (one split file at a time; the film split re-uses the same PNGs through a second label/list layout):
  uv run python scripts/03_render.py --raw-dir ~/swrd_paper_baseline/data/raw --work-dir ~/swrd_paper_baseline/data/work \
      --split split_tiles.json --out-dir ~/swrd_paper_baseline/data/yolo_tilesplit --p-lo 0.5 --p-hi 99.5 --clahe-clip 2.0 --clahe-grid 8
  uv run python scripts/03_render.py --raw-dir ~/swrd_paper_baseline/data/raw --work-dir ~/swrd_paper_baseline/data/work \
      --split split_tiles.json --labels polygon --reuse-images ~/swrd_paper_baseline/data/yolo_v1.0_papergrid \
      --out-dir ~/swrd_paper_baseline/data/yoloseg_v1.0_papergrid --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml
from _common import (
    CLASS_NAMES,
    Polygon,
    box_from_visible,
    clip_polygon_to_tile,
    load_polygons,
    polygon_to_yolo_seg,
    polygon_visible_mask,
    preprocess_tile,
    read_tif,
    simplify_polygon,
    write_json,
)
from tqdm import tqdm

# Instances whose written polygon covers less than this fraction (IoU) of the rasterised D7 piece are counted
# in label_stats as "low_mask_iou"; a few are expected from rounding on 4 px pieces, many would mean a bug.
MASK_IOU_WARN = 0.8


def _place_image(src: Path, dst: Path) -> None:
    """Hardlink the existing PNG (same filesystem), else copy it."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def polygon_lines_for_tile(
    polys: list[Polygon], t: dict, d7: dict, simplify_px: float, stats: Counter
) -> list[str]:
    """Segment lines of one tile, for exactly the instances 01_tile.py kept (checked against t['boxes'])."""
    x, y, side = t["x"], t["y"], t["side"]
    lines: list[str] = []
    boxes: list[str] = []
    for p in polys:
        mask = polygon_visible_mask(p, x, y, side)
        if mask is None or not mask.any():
            continue
        b = box_from_visible(
            p, mask, d7["min_side_px"], d7["min_visible_frac"], d7["min_tile_frac"]
        )
        if b is None:
            continue
        boxes.append(b.to_yolo(side))
        stats["instances"] += 1
        pts = clip_polygon_to_tile(p.points, x, y, side) if len(p.points) >= 3 else np.empty((0, 2))
        pts = simplify_polygon(pts, simplify_px) if len(pts) >= 3 else pts
        if (
            len(pts) < 3
        ):  # 1-2 point shapes, or clipping left nothing usable: the D7 box as a polygon
            stats["fallback_box_polygon"] += 1
            pts = np.array(
                [[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]], dtype=np.float32
            )
        else:
            canvas = np.zeros((side, side), dtype=np.uint8)
            cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
            inter = int(np.logical_and(canvas, mask).sum())
            union = int(np.logical_or(canvas, mask).sum())
            if union and inter / union < MASK_IOU_WARN:
                stats["low_mask_iou"] += 1
            px0, py0 = pts.min(axis=0)
            px1, py1 = pts.max(axis=0)
            dev = max(abs(px0 - b.x0), abs(py0 - b.y0), abs(px1 - b.x1), abs(py1 - b.y1))
            stats["max_box_deviation_px"] = max(stats["max_box_deviation_px"], float(dev))
        stats["vertices"] += len(pts)
        lines.append(polygon_to_yolo_seg(p.class_id, pts, side))
    if boxes != t["boxes"]:
        raise RuntimeError(
            f"{t['tile_id']}: D7 boxes recomputed here differ from tiles.jsonl\n"
            f"  here:   {boxes}\n  stored: {t['boxes']}"
        )
    return lines


def render_image(args: tuple) -> tuple[int, Counter]:
    """All tiles of one source image, so the big TIFF is decoded once."""
    (
        image_rel,
        json_rel,
        tiles,
        raw_dir,
        out_dir,
        p_lo,
        p_hi,
        clip,
        grid,
        three,
        labels,
        reuse_dir,
        d7,
        simplify_px,
    ) = args
    img16 = None if reuse_dir else read_tif(Path(raw_dir) / image_rel)
    polys: list[Polygon] = []
    if labels == "polygon":
        all_polys, _ = load_polygons(Path(raw_dir) / json_rel)
        polys = [p for p in all_polys if p.class_id is not None]
    stats: Counter = Counter()
    n = 0
    for t in tiles:
        x, y, s = t["x"], t["y"], t["side"]
        part = t["part"]
        img_path = Path(out_dir) / "images" / part / f"{t['tile_id']}.png"
        lab_path = Path(out_dir) / "labels" / part / f"{t['tile_id']}.txt"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        lab_path.parent.mkdir(parents=True, exist_ok=True)
        if reuse_dir:
            src = Path(reuse_dir) / "images" / part / f"{t['tile_id']}.png"
            if not src.is_file():
                raise FileNotFoundError(f"{t['tile_id']}: no image to reuse at {src}")
            _place_image(src, img_path)
        else:
            tile16 = img16[y : y + s, x : x + s]
            if tile16.shape != (s, s):
                raise RuntimeError(
                    f"{t['tile_id']}: window {tile16.shape} does not fit image {img16.shape}"
                )
            tile8 = preprocess_tile(tile16, p_lo, p_hi, clip, grid)
            if three:
                tile8 = cv2.merge([tile8, tile8, tile8])
            if not cv2.imwrite(str(img_path), tile8):
                raise RuntimeError(f"could not write {img_path}")
        if labels == "polygon":
            lines = polygon_lines_for_tile(polys, t, d7, simplify_px, stats)
        else:
            lines = t["boxes"]
        lab_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n += 1
    return n, stats


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
    ap.add_argument(
        "--labels",
        default="box",
        choices=["box", "polygon"],
        help="box = D7 boxes from tiles.jsonl (detection); polygon = Ultralytics segments of the same instances",
    )
    ap.add_argument(
        "--reuse-images",
        type=Path,
        default=None,
        help="an existing render whose PNGs are hardlinked instead of decoding the TIFFs",
    )
    ap.add_argument(
        "--simplify-px",
        type=float,
        default=0.5,
        help="Douglas-Peucker tolerance in pixels for polygon labels (0 = keep every vertex)",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.reuse_images and args.three_channel:
        ap.error("--three-channel has no effect with --reuse-images")

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

    json_of: dict[str, str] = {}
    d7: dict = {}
    if args.labels == "polygon":
        inv = json.loads((args.work_dir / "inventory.json").read_text(encoding="utf-8"))
        json_of = {r["image"]: r["json"] for r in inv["records"] if r["json"]}
        missing = [img for img in by_image if img not in json_of]
        if missing:
            raise SystemExit(f"{len(missing)} images without a label file, e.g. {missing[:3]}")
        tiling = json.loads((args.work_dir / "tiles_summary.json").read_text(encoding="utf-8"))
        d7 = {k: tiling["params"][k] for k in ("min_side_px", "min_visible_frac", "min_tile_frac")}
        print(f"[render] polygon labels; D7 rule from tiles_summary.json: {d7}")
    if args.reuse_images:
        print(f"[render] images hardlinked from {args.reuse_images}")

    jobs = [
        (
            img,
            json_of.get(img),
            tiles,
            str(args.raw_dir),
            str(args.out_dir),
            args.p_lo,
            args.p_hi,
            args.clahe_clip,
            args.clahe_grid,
            args.three_channel,
            args.labels,
            str(args.reuse_images) if args.reuse_images else None,
            d7,
            args.simplify_px,
        )
        for img, tiles in sorted(by_image.items())
    ]
    done = 0
    stats: Counter = Counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for n, st in tqdm(ex.map(render_image, jobs, chunksize=2), total=len(jobs), unit="img"):
            done += n
            dev = max(stats["max_box_deviation_px"], st.pop("max_box_deviation_px", 0.0))
            stats.update(st)
            stats["max_box_deviation_px"] = dev

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
    if args.labels == "polygon":
        params["d7_params"] = d7
        params["label_stats"] = {
            "instances": stats["instances"],
            "fallback_box_polygon": stats["fallback_box_polygon"],
            "low_mask_iou": stats["low_mask_iou"],
            "mask_iou_warn_threshold": MASK_IOU_WARN,
            "max_box_deviation_px": round(stats["max_box_deviation_px"], 3),
            "mean_vertices_per_instance": round(stats["vertices"] / max(stats["instances"], 1), 2),
        }
        print(f"[render] label stats: {params['label_stats']}")
    write_json(args.out_dir / "render_params.json", params)

    sizes = np.array([t["side"] for v in by_image.values() for t in v])
    print(
        f"[render] wrote {done:,} tiles; side px median/min/max {int(np.median(sizes))}/{sizes.min()}/{sizes.max()}"
    )
    print(f"[render] data file: {args.out_dir / 'swrd6.yaml'}")


if __name__ == "__main__":
    main()
