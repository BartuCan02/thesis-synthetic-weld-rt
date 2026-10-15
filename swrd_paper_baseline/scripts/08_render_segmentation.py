"""Paper-preprocessed tiles with 9-class masks, in Deeplify's segmentation-framework layout, on v2.13's film split.

Bartu, 2026-10-15: compare Deeplify's SegFormer under the paper's preprocessing with the same SegFormer under
Deeplify's own preprocessing (run `weld_defect_all_v2.13_pullv5`, task c3df0377). Same model, same films, same
split, same 9 classes, same metric; only the preprocessing differs.

What is reproduced from the paper pipeline (01_tile / 02_select_split / 03_render): windows of half the short
side with 50 % overlap (edge drop), per-tile P0.5-99.5 stretch -> 8-bit -> CLAHE 2.0/8x8, all positive tiles plus
an equal number of random clean tiles per split (seed 0). What differs from the YOLO sets: every tile gets a mask,
"positive" means any of the 9-class values 1..8 is present, and the train/val assignment is v2.13's per-film split
(SWRD 3,701 / 305, customer 2,081 / 532 films), not the thesis film split.

Mask values, identical to the framework's use case `weld_defect_all`:
  0 background | 1 porosity | 2 inclusion | 3 crack | 4 undercut | 5 lack_of_fusion | 6 lack_of_penetration
  7 other_defect (export label "other_defect"; SWRD 焊瘤 weld overlap and 内凹 root concavity)
  8 excess_material (export label "excess_material")
Painting order (the builder's rule): excess_material, then other_defect, then the six in index order, so a named
class wins where polygons overlap. Pseudo-defect (伪缺陷) and the seam polygon are ignored.

Inputs
  --swrd-raw-dir       the SWRD release mirror (crop_weld_images/, crop_weld_jsons/)
  --swrd-work-dir      its work dir with inventory.json (stem -> files)
  --customer-raw-dir   export_customer_films.py output with other_defect / excess_material labels (raw_customer_v213_9cls)
  --v213-dir           the v2.13 build folder: split.json (film key -> split) and patch_index.json (swrd film -> stem)

Output <out>/: train/images, train/masks, val/images, val/masks (PNG), split.json, patch_index.json,
build_stats.json (counts, class presence, measured normalisation mean/std of the train images).

Run on the box:
  uv run python scripts/08_render_segmentation.py --swrd-raw-dir ~/swrd_paper_baseline/data/raw \\
      --swrd-work-dir ~/swrd_paper_baseline/data/work --customer-raw-dir ~/swrd_paper_baseline/data/raw_customer_v213_9cls \\
      --v213-dir ~/v213_mcall_resid2ch --out ~/swrd_paper_baseline/data/seg_v213split_9cls --workers 8
  then  AWS_PROFILE=data-rw uv run python scripts/08_render_segmentation.py ... --upload-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    EXCESS_MATERIAL_LABEL,
    LABEL_TO_CLASS,
    OTHER_DEFECT_LABEL,
    SWRD_OTHER_DEFECT_LABELS,
    load_polygons,
    preprocess_tile,
    read_tif,
    tif_shape,
    tile_positions,
    tile_side,
    write_json,
)
from tqdm import tqdm

CLASS_NAMES = [
    "background",
    "porosity",
    "inclusion",
    "crack",
    "undercut",
    "lack_of_fusion",
    "lack_of_penetration",
    "other_defect",
    "excess_material",
]
OTHER_INDEX, EXCESS_INDEX = 7, 8
PROJECT = "weld_defect_segmentation"
OUTPUT_URI = "s3://clearml-data-deeplify/clearml-datasets"


def mask_value(label: str) -> int | None:
    """9-class mask value of a LabelMe label string, None if it is not a training class."""
    if label in LABEL_TO_CLASS:
        return LABEL_TO_CLASS[label] + 1
    if label == OTHER_DEFECT_LABEL or label in SWRD_OTHER_DEFECT_LABELS:
        return OTHER_INDEX
    if label == EXCESS_MATERIAL_LABEL:
        return EXCESS_INDEX
    return None


def rasterise(polys, height: int, width: int) -> np.ndarray:
    """Full-crop 9-class mask; excess first, other second, the six in index order (named classes win)."""
    mask = np.zeros((height, width), np.uint8)
    order = sorted(
        ((mask_value(p.label), p) for p in polys if mask_value(p.label) is not None),
        key=lambda t: {EXCESS_INDEX: -2, OTHER_INDEX: -1}.get(t[0], t[0]),
    )
    for value, p in order:
        pts = np.round(p.points).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], int(value))
        else:
            x0, y0 = np.floor(pts.min(axis=0)).astype(int)
            x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 1
            mask[max(0, y0) : y1, max(0, x0) : x1] = value
    return mask


def index_film(job: tuple) -> dict:
    """Pass 1: every window of one film with its mask classes, without touching the TIFF pixels."""
    key, image_path, json_path, side_fraction, overlap, edge = job
    h, w, _ = tif_shape(Path(image_path))
    polys, _ = load_polygons(Path(json_path))
    mask = rasterise(polys, h, w)
    side = tile_side(h, w, side_fraction)
    tiles = []
    for y in tile_positions(h, side, overlap, edge):
        for x in tile_positions(w, side, overlap, edge):
            present = np.unique(mask[y : y + side, x : x + side])
            classes = [int(v) for v in present if v > 0]
            tiles.append({"x": x, "y": y, "side": side, "classes": classes})
    return {"key": key, "height": h, "width": w, "tiles": tiles}


def render_film(job: tuple) -> dict:
    """Pass 2: cut and preprocess the selected windows of one film, write image and mask PNGs."""
    key, image_path, json_path, tiles, out_dir, split, p_lo, p_hi, clip, grid = job
    img16 = read_tif(Path(image_path))
    polys, _ = load_polygons(Path(json_path))
    mask = rasterise(polys, img16.shape[0], img16.shape[1])
    stem = Path(image_path).stem
    s = s2 = 0.0
    n = 0
    for t in tiles:
        x, y, side = t["x"], t["y"], t["side"]
        tile8 = preprocess_tile(img16[y : y + side, x : x + side], p_lo, p_hi, clip, grid)
        tile_id = f"{stem}__x{x}__y{y}__s{side}"
        cv2.imwrite(str(Path(out_dir) / split / "images" / f"{tile_id}.png"), tile8)
        cv2.imwrite(
            str(Path(out_dir) / split / "masks" / f"{tile_id}.png"),
            mask[y : y + side, x : x + side],
        )
        f = tile8.astype(np.float64) / 255.0
        s += f.sum()
        s2 += (f * f).sum()
        n += f.size
    return {"key": key, "sum": s, "sumsq": s2, "n": n, "written": len(tiles)}


def collect_films(args) -> tuple[list[dict], dict]:
    """The films of v2.13's split, each with its TIFF and JSON in the thesis exports."""
    split = json.loads((args.v213_dir / "split.json").read_text())
    pi = json.loads((args.v213_dir / "patch_index.json").read_text())
    swrd_stem = {rec["film"]: rec["external_id"] for rec in pi.values() if rec["source"] == "swrd"}
    inv = json.loads((args.swrd_work_dir / "inventory.json").read_text())
    swrd_by_stem = {r["stem"]: r for r in inv["records"] if r["json"]}
    films, missing = [], Counter()
    for key, part in sorted(split.items()):
        source, sid = key.split("/", 1)
        if source == "swrd":
            stem = swrd_stem.get(key)
            rec = swrd_by_stem.get(stem) if stem else None
            if rec is None:
                missing[source] += 1
                continue
            image, label = args.swrd_raw_dir / rec["image"], args.swrd_raw_dir / rec["json"]
        else:
            stem = f"{source}__{sid}"
            image = args.customer_raw_dir / "crop_weld_images" / source / f"{stem}.tif"
            label = args.customer_raw_dir / "crop_weld_jsons" / source / f"{stem}.json"
            if not image.is_file() or not label.is_file():
                missing[source] += 1
                continue
        films.append(
            {"key": key, "source": source, "split": part, "image": str(image), "json": str(label)}
        )
    return films, missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--swrd-raw-dir", type=Path, required=True)
    ap.add_argument("--swrd-work-dir", type=Path, required=True, help="holds inventory.json")
    ap.add_argument("--customer-raw-dir", type=Path, required=True)
    ap.add_argument("--v213-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--side-fraction", type=float, default=0.5)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--edge", default="drop", choices=["flush", "drop"])
    ap.add_argument("--p-lo", type=float, default=0.5)
    ap.add_argument("--p-hi", type=float, default=99.5)
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-grid", type=int, default=8)
    ap.add_argument(
        "--negative-ratio", type=float, default=1.0, help="clean tiles per positive, per split"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="films per source (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--upload", action="store_true", help="also register the folder as a ClearML dataset"
    )
    ap.add_argument(
        "--upload-only", action="store_true", help="register an existing folder, render nothing"
    )
    ap.add_argument("--dataset-name", default="weld-defect-multiclass-all")
    ap.add_argument("--dataset-version", default="2.14-papertiles-9cls-v213split")
    args = ap.parse_args()

    if args.upload_only:
        upload(args)
        return

    films, missing = collect_films(args)
    if args.limit:
        per: Counter = Counter()
        kept = []
        for f in films:
            if per[f["source"]] < args.limit:
                kept.append(f)
                per[f["source"]] += 1
        films = kept
    print(f"[seg] {len(films)} films from v2.13's split; missing in the exports: {dict(missing)}")
    print("[seg] per source x split:", dict(Counter((f["source"], f["split"]) for f in films)))

    # ---- pass 1: index every window ------------------------------------------------------------------
    jobs = [
        (f["key"], f["image"], f["json"], args.side_fraction, args.overlap, args.edge)
        for f in films
    ]
    index: dict[str, dict] = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for r in tqdm(
            ex.map(index_film, jobs, chunksize=4), total=len(jobs), unit="film", desc="index"
        ):
            index[r["key"]] = r

    # ---- select: all positives + equal random clean tiles, per split ---------------------------------
    rng = random.Random(args.seed)
    by_film_split = {f["key"]: f["split"] for f in films}
    selected: dict[str, list[dict]] = defaultdict(list)
    stats = {"positives": Counter(), "clean_pool": Counter(), "negatives": Counter()}
    class_presence: Counter = Counter()
    for part in ("train", "val"):
        pos, clean = [], []
        for key, r in index.items():
            if by_film_split[key] != part:
                continue
            for t in r["tiles"]:
                (pos if t["classes"] else clean).append((key, t))
        n_neg = min(len(clean), round(len(pos) * args.negative_ratio))
        neg = rng.sample(clean, n_neg)
        for key, t in pos + neg:
            selected[key].append(t)
        for _, t in pos:
            for c in t["classes"]:
                class_presence[(part, CLASS_NAMES[c])] += 1
        stats["positives"][part] = len(pos)
        stats["clean_pool"][part] = len(clean)
        stats["negatives"][part] = n_neg
        print(
            f"[seg] {part}: positives {len(pos):,} | clean pool {len(clean):,} | negatives drawn {n_neg:,}"
        )

    # ---- pass 2: render --------------------------------------------------------------------------------
    for part in ("train", "val"):
        for sub in ("images", "masks"):
            (args.out / part / sub).mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            f["key"],
            f["image"],
            f["json"],
            selected[f["key"]],
            str(args.out),
            f["split"],
            args.p_lo,
            args.p_hi,
            args.clahe_clip,
            args.clahe_grid,
        )
        for f in films
        if selected[f["key"]]
    ]
    s = s2 = n = 0.0
    written = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for r in tqdm(
            ex.map(render_film, jobs, chunksize=2), total=len(jobs), unit="film", desc="render"
        ):
            s, s2, n, written = s + r["sum"], s2 + r["sumsq"], n + r["n"], written + r["written"]
    mean = s / n
    std = float(np.sqrt(max(s2 / n - mean * mean, 0.0)))

    # ---- manifests ---------------------------------------------------------------------------------
    patch_index = {}
    for f in films:
        stem = Path(f["image"]).stem
        for t in selected[f["key"]]:
            tid = f"{stem}__x{t['x']}__y{t['y']}__s{t['side']}.png"
            patch_index[tid] = {
                "film": f["key"],
                "source": f["source"],
                "split": f["split"],
                "x": t["x"],
                "y": t["y"],
                "side": t["side"],
                "classes": t["classes"],
            }
    write_json(args.out / "split.json", {f["key"]: f["split"] for f in films})
    write_json(args.out / "patch_index.json", patch_index)
    per_source_split = Counter((f["source"], f["split"]) for f in films)
    build = {
        "films_per_source": dict(Counter(f["source"] for f in films)),
        "films_per_source_split": {f"{k[0]}/{k[1]}": v for k, v in per_source_split.items()},
        "films_missing_in_exports": dict(missing),
        "tiles_written": written,
        "tiles_per_split": {
            p: sum(1 for v in patch_index.values() if v["split"] == p) for p in ("train", "val")
        },
        "positives_per_split": dict(stats["positives"]),
        "clean_pool_per_split": dict(stats["clean_pool"]),
        "negatives_per_split": dict(stats["negatives"]),
        "tiles_with_class_per_split": {
            f"{k[0]}/{k[1]}": v for k, v in sorted(class_presence.items())
        },
        "class_names": CLASS_NAMES,
        "normalization_train": {"mean": round(mean, 4), "std": round(std, 4)},
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "source_split": "v2.13 split.json (weld_defect_all_v2.13_pullv5, task c3df0377)",
        "preprocessing": "paper tiles: side = short/2, 50 % overlap, edge drop, per-tile P0.5-99.5 stretch, CLAHE 2.0/8",
    }
    write_json(args.out / "build_stats.json", build)
    print(
        json.dumps({k: v for k, v in build.items() if k not in ("params", "class_names")}, indent=1)
    )
    print(
        f"[seg] normalization mean {mean:.4f} std {std:.4f} -> "
        f"data.normalization.mean=[{mean:.4f}] data.normalization.std=[{std:.4f}]"
    )
    if args.upload:
        upload(args)


def upload(args) -> None:
    from clearml import Dataset

    build = json.loads((args.out / "build_stats.json").read_text())
    ds = Dataset.create(
        dataset_name=args.dataset_name,
        dataset_project=PROJECT,
        dataset_version=args.dataset_version,
        output_uri=OUTPUT_URI,
        description="Paper-preprocessed tiles (Zhao 2025 recipe) with 9-class masks on the v2.13 film split; "
        "SegFormer preprocessing comparison against weld_defect_all_v2.13 (thesis side run for Felix)",
    )
    for part in ("train", "val"):
        for sub in ("images", "masks"):
            ds.add_files(
                path=str(args.out / part / sub), dataset_path=f"{part}/{sub}", verbose=False
            )
    for extra in ("split.json", "patch_index.json", "build_stats.json"):
        ds.add_files(path=str(args.out / extra))
    ds.set_metadata(build, metadata_name="pipeline")
    ds.upload(show_progress=True)
    ds.finalize()
    print(f"[seg] dataset id {ds.id} ({PROJECT} / {args.dataset_name} {args.dataset_version})")


if __name__ == "__main__":
    main()
