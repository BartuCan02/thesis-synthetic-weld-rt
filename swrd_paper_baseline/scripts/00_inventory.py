"""Step 1 — inventory the raw SWRD release and test it against the paper's own numbers.

Reads the local mirror of ``s3://swdr/cropped/`` (``<raw-dir>/crop_weld_images/{L,T}/*.tif`` and
``<raw-dir>/crop_weld_jsons/{L,T}/*.json``), pairs every image with its label file, hashes the images to
find byte-identical duplicates, reads image sizes from the TIFF headers, and tabulates every label string.

Then it answers the questions that gate step 2:
  * do the JSONs reproduce Fig. 4 (instances and mean area per class) with our label mapping?
  * why does the mirror hold a different number of TIFFs than the paper's 4,930 weld images?
  * how many tiles does the paper's window rule produce on these sizes — close to "over 380,000"?

Nothing is modified. Outputs:
  <work-dir>/inventory.json       one record per image (path, size, hash, exposure id, polygons summary)
  <work-dir>/inventory_report.md  the comparison tables, to paste into results/

Run on the box after ``aws s3 sync s3://swdr/cropped/ ~/swrd_paper_baseline/data/raw/`` (AWS_PROFILE=data-rw):
  uv run python scripts/00_inventory.py --raw-dir ~/swrd_paper_baseline/data/raw --work-dir ~/swrd_paper_baseline/data/work
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from _common import (
    CLASS_NAMES,
    FIG4_INSTANCES_ORIGINAL,
    FIG4_MEAN_AREA_PX2,
    IMAGES_SUBDIR,
    JSONS_SUBDIR,
    LABEL_TO_CLASS,
    PAPER_N_TILES_MIN,
    PAPER_N_WELD_IMAGES,
    TIF_SUFFIXES,
    count_tiles,
    exposure_id,
    load_polygons,
    sha1_of_file,
    tif_shape,
    write_json,
)
from tqdm import tqdm


def find_images(raw_dir: Path) -> list[Path]:
    img_root = raw_dir / IMAGES_SUBDIR
    if not img_root.is_dir():
        raise SystemExit(f"missing {img_root}; sync s3://swdr/cropped/ first")
    return sorted(p for p in img_root.rglob("*") if p.suffix.lower() in TIF_SUFFIXES)


def json_for(raw_dir: Path, img: Path) -> Path:
    rel = img.relative_to(raw_dir / IMAGES_SUBDIR)
    return (raw_dir / JSONS_SUBDIR / rel).with_suffix(".json")


def scan_one(raw_dir: Path, img: Path) -> dict:
    h, w, dtype = tif_shape(img)
    rec = {
        "stem": img.stem,
        "subset": img.relative_to(raw_dir / IMAGES_SUBDIR).parts[0],  # "L" or "T"
        "exposure": exposure_id(img.stem),
        "image": str(img.relative_to(raw_dir)),
        "height": h,
        "width": w,
        "dtype": dtype,
        "sha1": sha1_of_file(img),
        "json": None,
        "json_header": None,
        "labels": {},  # raw label string -> count
        "shape_types": {},
        "class_instances": [0] * len(CLASS_NAMES),
        "class_area_sum": [0.0] * len(CLASS_NAMES),
    }
    jp = json_for(raw_dir, img)
    if jp.is_file():
        rec["json"] = str(jp.relative_to(raw_dir))
        polys, header = load_polygons(jp)
        rec["json_header"] = header
        lab, st = Counter(), Counter()
        for p in polys:
            lab[p.label] += 1
            st[p.shape_type] += 1
            if p.class_id is not None:
                rec["class_instances"][p.class_id] += 1
                rec["class_area_sum"][p.class_id] += p.area
        rec["labels"], rec["shape_types"] = dict(lab), dict(st)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    images = find_images(args.raw_dir)
    jsons = sorted((args.raw_dir / JSONS_SUBDIR).rglob("*.json"))
    print(f"[inventory] {len(images)} TIFFs, {len(jsons)} JSONs under {args.raw_dir}")

    with ThreadPoolExecutor(args.workers) as ex:
        records = list(
            tqdm(ex.map(lambda p: scan_one(args.raw_dir, p), images), total=len(images), unit="img")
        )

    # --- pairing -----------------------------------------------------------------------------------
    image_stems = {r["stem"] for r in records}
    orphan_jsons = sorted(
        str(j.relative_to(args.raw_dir)) for j in jsons if j.stem not in image_stems
    )
    images_without_json = sorted(r["image"] for r in records if r["json"] is None)

    # --- duplicates ----------------------------------------------------------------------------------
    by_hash: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_hash[r["sha1"]].append(r["stem"])
    dup_groups = sorted((v for v in by_hash.values() if len(v) > 1), key=len, reverse=True)
    n_dup_extra = sum(len(g) - 1 for g in dup_groups)

    # --- header vs TIFF size -----------------------------------------------------------------------
    size_mismatch = [
        r["stem"]
        for r in records
        if r["json_header"]
        and (r["json_header"]["imageHeight"], r["json_header"]["imageWidth"])
        != (r["height"], r["width"])
    ]

    # --- labels --------------------------------------------------------------------------------------
    all_labels, shape_types = Counter(), Counter()
    for r in records:
        all_labels.update(r["labels"])
        shape_types.update(r["shape_types"])
    unmapped = {k: v for k, v in all_labels.items() if k not in LABEL_TO_CLASS}

    inst = np.sum([r["class_instances"] for r in records], axis=0)
    area = np.sum([r["class_area_sum"] for r in records], axis=0)
    mean_area = np.divide(area, np.maximum(inst, 1))
    films_with_defect = sum(1 for r in records if sum(r["class_instances"]) > 0)

    # --- tile count under the paper's rule and its two edge readings ---------------------------------
    tile_counts = {}
    for edge in ("flush", "drop"):
        tile_counts[edge] = int(
            sum(count_tiles(r["height"], r["width"], 0.5, 0.5, edge) for r in records)
        )
    # the alternative reading "window = the whole shorter side" is reported too, in case 380k matches it
    tile_counts["alt_full_short_side_flush"] = int(
        sum(count_tiles(r["height"], r["width"], 1.0, 0.5, "flush") for r in records)
    )

    H = np.array([r["height"] for r in records])
    W = np.array([r["width"] for r in records])
    portrait = int((H > W).sum())

    # --- report --------------------------------------------------------------------------------------
    lines = []
    lines.append("# SWRD raw-release inventory\n")
    lines.append(
        f"TIFFs: **{len(images)}** (paper Sect. 3.1: {PAPER_N_WELD_IMAGES}); JSONs: **{len(jsons)}**"
    )
    lines.append(f"- subsets: {dict(Counter(r['subset'] for r in records))}")
    lines.append(f"- exposures (A_/B_ halves merged): {len({r['exposure'] for r in records})}")
    lines.append(
        f"- images without JSON: {len(images_without_json)}; JSONs without image: {len(orphan_jsons)}"
    )
    lines.append(
        f"- byte-identical duplicate groups: {len(dup_groups)} ({n_dup_extra} redundant files)"
    )
    lines.append(f"- JSON imageHeight/imageWidth != TIFF size: {len(size_mismatch)}")
    lines.append(
        f"- dtypes: {dict(Counter(r['dtype'] for r in records))}; portrait (H > W): {portrait}"
    )
    lines.append(
        f"- width  median/min/max: {int(np.median(W))} / {W.min()} / {W.max()}; "
        f"height median/min/max: {int(np.median(H))} / {H.min()} / {H.max()}"
    )
    lines.append(
        f"- films with >= 1 mapped defect: {films_with_defect}; defect-free: {len(records) - films_with_defect}\n"
    )

    lines.append("## Label strings in the JSONs\n")
    lines.append("| label | count | mapped to |\n|---|---:|---|")
    for lab, n in all_labels.most_common():
        cls = LABEL_TO_CLASS.get(lab)
        lines.append(f"| {lab} | {n} | {CLASS_NAMES[cls] if cls is not None else '— (ignored)'} |")
    lines.append(f"\nshape types: {dict(shape_types)}\n")

    lines.append("## Fig. 4 check (instances and mean polygon area on the weld images)\n")
    lines.append(
        "| class | ours | paper | diff | mean area ours | paper |\n|---|---:|---:|---:|---:|---:|"
    )
    for i, name in enumerate(CLASS_NAMES):
        p = FIG4_INSTANCES_ORIGINAL[name]
        lines.append(
            f"| {name} | {int(inst[i])} | {p} | {int(inst[i]) - p:+d} | {mean_area[i]:.1f} | {FIG4_MEAN_AREA_PX2[name]:.1f} |"
        )
    lines.append(
        "\nNote: Fig. 4 was counted on the 3,675 original films; a T-joint defect cut by the A_/B_ split "
        "or a polygon straddling a crop border would count twice here.\n"
    )

    lines.append("## Tile count under Sect. 3.2 (side = short/2, 50 % overlap)\n")
    lines.append(f"- edge=flush: **{tile_counts['flush']:,}**")
    lines.append(f"- edge=drop:  **{tile_counts['drop']:,}**")
    lines.append(f"- paper: over {PAPER_N_TILES_MIN:,}")
    lines.append(
        f"- (alternative reading, window = whole short side, flush: {tile_counts['alt_full_short_side_flush']:,})\n"
    )

    if dup_groups:
        lines.append("## Duplicate groups (first 20)\n")
        for g in dup_groups[:20]:
            lines.append(f"- {', '.join(g)}")
        lines.append("")
    if orphan_jsons:
        lines.append(f"## JSONs without image (first 20 of {len(orphan_jsons)})\n")
        lines.extend(f"- {j}" for j in orphan_jsons[:20])
        lines.append("")
    if images_without_json:
        lines.append(f"## Images without JSON (first 20 of {len(images_without_json)})\n")
        lines.extend(f"- {j}" for j in images_without_json[:20])
        lines.append("")
    if unmapped:
        lines.append(
            "## Unmapped label strings (ignored at training time; decide in step 1 gate)\n"
        )
        lines.extend(f"- {k}: {v}" for k, v in sorted(unmapped.items(), key=lambda kv: -kv[1]))
        lines.append("")

    report = "\n".join(lines)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "inventory_report.md").write_text(report, encoding="utf-8")
    write_json(
        args.work_dir / "inventory.json",
        {
            "raw_dir": str(args.raw_dir),
            "n_images": len(images),
            "n_jsons": len(jsons),
            "duplicate_groups": dup_groups,
            "orphan_jsons": orphan_jsons,
            "images_without_json": images_without_json,
            "size_mismatch": size_mismatch,
            "tile_counts": tile_counts,
            "records": records,
        },
    )
    print(report)
    print(f"[inventory] wrote {args.work_dir / 'inventory.json'} and inventory_report.md")


if __name__ == "__main__":
    main()
