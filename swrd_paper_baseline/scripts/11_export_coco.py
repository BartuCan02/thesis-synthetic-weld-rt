"""Step 11 — write the tile set as COCO annotation files, one per split part, for the non-Ultralytics detectors.

DEIM / D-FINE and RF-DETR read COCO json, not YOLO txt. This script turns the YOLO labels of the v1.0 tile folder plus
a film split (02_select_split.py / 09_test_split.py) into ``instances_train.json``, ``instances_val.json`` and
``instances_test.json``. The images are not copied: every ``file_name`` is the tile's path relative to the dataset
root (``images/train/<stem>.png`` or ``images/val/<stem>.png``, whichever folder of the v1.0 release holds it), so the
json files can be added to ClearML as a child dataset of the tile dataset and both codebases read the parent's PNGs.

Conventions (shared with 12_predict_coco.py, 13_score_coco.py, 14_train_deim.py, 15_train_rfdetr.py):
  category id   = the YOLO class index 0..5 (the order of swrd6.yaml). DEIM uses category ids directly as labels
                  when remap_mscoco_category is False, so they must be 0-based and contiguous; RF-DETR maps the
                  ids it finds in train to contiguous labels, which is the identity here.
  image id      = 1-based position of the stem in the sorted list of all tiles (train, val and test together), so an
                  id is the same in every json. Each image entry also carries "stem" and "exposure".
  bbox          = [x, y, w, h] in pixels of the tile, converted from the normalised YOLO centre format; area = w * h.
  A defect-free tile is an image entry without annotations (both codebases keep such images).

Run on the box (the v1.0 folder and the split file live there):
  uv run python scripts/11_export_coco.py --yolo-dir ~/swrd_paper_baseline/data/yolo_v1.0_papergrid \
      --split-file ~/swrd_paper_baseline/data/work/split_films_test.json --out-dir ~/swrd_paper_baseline/data/coco_v1.0
  AWS_PROFILE=data-rw uv run python scripts/11_export_coco.py ... --upload --parent-id de772ad9363c4067bed5835e13a9be81 \
      --name swrd-paper-tiles-coco --version 1.0.0
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path

import yaml

PROJECT = "thesis_wp1_benchmark"
DEFAULT_OUTPUT_URI = "s3://clearml-data-deeplify/clearml-datasets"
PARENT_TILES = "de772ad9363c4067bed5835e13a9be81"
PARTS = ("train", "val", "test")


def exposure_of(stem: str) -> str:
    """Original exposure of a tile (same rule as 05_train.exposure_of)."""
    head = stem.split("__")[0]
    return head[2:] if head[:2] in ("A_", "B_") else head


def png_size(path: Path) -> tuple[int, int]:
    """(width, height) from the PNG header without decoding the image."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError(f"{path}: not a PNG")
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


def yolo_line_to_coco(line: str, width: int, height: int) -> tuple[int, list[float]]:
    """One YOLO label line 'cls cx cy w h' (normalised) -> (class id, [x, y, w, h] in pixels)."""
    cls, cx, cy, w, h = line.split()[:5]
    bw, bh = float(w) * width, float(h) * height
    x = float(cx) * width - bw / 2
    y = float(cy) * height - bh / 2
    return int(cls), [round(x, 3), round(y, 3), round(bw, 3), round(bh, 3)]


def build_coco(records: list[dict], names: dict[int, str]) -> dict:
    """COCO dict for one part.

    ``records``: [{"id", "stem", "file_name", "width", "height", "labels": [yolo lines]}].
    """
    categories = [
        {"id": int(i), "name": name, "supercategory": "weld_defect"}
        for i, name in sorted(names.items())
    ]
    images, annotations = [], []
    ann_id = 1
    for r in records:
        images.append(
            {
                "id": r["id"],
                "file_name": r["file_name"],
                "width": r["width"],
                "height": r["height"],
                "stem": r["stem"],
                "exposure": exposure_of(r["stem"]),
            }
        )
        for line in r["labels"]:
            if not line.strip():
                continue
            cls, bbox = yolo_line_to_coco(line, r["width"], r["height"])
            if cls not in names:
                raise ValueError(f"{r['stem']}: class {cls} is not in the data yaml")
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": r["id"],
                    "category_id": cls,
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 3),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {
        "info": {
            "description": "SWRD paper tiles (v1.0), film split, COCO boxes from the YOLO labels"
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def collect_records(yolo_dir: Path, split: dict) -> dict[str, list[dict]]:
    """Every tile of the v1.0 folder assigned to train / val / test by the split file's tile-id lists."""
    where: dict[str, str] = {}
    for part in PARTS:
        for stem in split.get(part, []):
            if stem in where:
                raise ValueError(f"{stem} is in both {where[stem]} and {part} of the split")
            where[stem] = part
    pngs = {
        p.stem: p
        for folder in ("train", "val")
        for p in (yolo_dir / "images" / folder).glob("*.png")
    }
    missing = sorted(set(where) - set(pngs))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} tiles of the split are not in {yolo_dir}: {missing[:5]} ..."
        )
    unassigned = sorted(set(pngs) - set(where))
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} tiles of {yolo_dir} are in no list of the split: {unassigned[:5]} ..."
        )
    ids = {stem: i + 1 for i, stem in enumerate(sorted(pngs))}
    records: dict[str, list[dict]] = {part: [] for part in PARTS}
    for stem in sorted(pngs):
        png = pngs[stem]
        label = yolo_dir / "labels" / png.parent.name / f"{stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(
                label
            )  # a missing label would silently turn a defect tile into a negative
        width, height = png_size(png)
        records[where[stem]].append(
            {
                "id": ids[stem],
                "stem": stem,
                "file_name": f"images/{png.parent.name}/{png.name}",
                "width": width,
                "height": height,
                "labels": label.read_text().splitlines(),
            }
        )
    return records


def summarize(coco: dict) -> dict:
    per_class = Counter(a["category_id"] for a in coco["annotations"])
    names = {c["id"]: c["name"] for c in coco["categories"]}
    with_box = len({a["image_id"] for a in coco["annotations"]})
    return {
        "images": len(coco["images"]),
        "images_with_boxes": with_box,
        "images_clean": len(coco["images"]) - with_box,
        "boxes": len(coco["annotations"]),
        "boxes_per_class": {names[c]: n for c, n in sorted(per_class.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--yolo-dir", type=Path, required=True, help="local v1.0 tile folder with swrd6.yaml"
    )
    ap.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="split json with train / val / test tile lists",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="writes <out-dir>/coco/instances_*.json"
    )
    ap.add_argument(
        "--upload", action="store_true", help="register <out-dir>/coco as a ClearML child dataset"
    )
    ap.add_argument(
        "--parent-id", default=PARENT_TILES, help="the tile dataset whose PNGs the json files name"
    )
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--name", default="swrd-paper-tiles-coco")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--output-uri", default=DEFAULT_OUTPUT_URI)
    args = ap.parse_args()

    data = yaml.safe_load((args.yolo_dir / "swrd6.yaml").read_text())
    names = {int(k): v for k, v in data["names"].items()}
    split = json.loads(args.split_file.read_text())
    if "test" not in split:
        raise SystemExit(
            f"{args.split_file.name} has no test list: use the split from 09_test_split.py"
        )
    records = collect_records(args.yolo_dir, split)

    out = args.out_dir / "coco"
    out.mkdir(parents=True, exist_ok=True)
    report = {"yolo_dir": str(args.yolo_dir), "split_file": str(args.split_file), "parts": {}}
    for part in PARTS:
        coco = build_coco(records[part], names)
        (out / f"instances_{part}.json").write_text(json.dumps(coco))
        report["parts"][part] = summarize(coco)
        print(f"[coco] {part}: {report['parts'][part]}")
    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    (out / "split_films_test.json").write_text(args.split_file.read_text())
    (out / "README.md").write_text(
        "COCO annotation files for the SWRD paper tiles (11_export_coco.py).\n"
        "file_name is relative to the dataset root (images/train/... or images/val/... of the v1.0 tiles).\n"
        "category id = YOLO class index (0 porosity ... 5 lack_of_penetration); image id = sorted position + 1.\n"
    )

    if not args.upload:
        print(f"[coco] written to {out}; not uploaded (pass --upload)")
        return
    from clearml import Dataset

    ds = Dataset.create(
        dataset_name=args.name,
        dataset_project=args.project,
        dataset_version=args.version,
        parent_datasets=[args.parent_id],
        output_uri=args.output_uri,
        description="COCO json for the SWRD paper tiles (film split with held-out test); images from the parent",
    )
    ds.add_files(path=str(out), dataset_path="coco", verbose=False)
    ds.set_metadata(report)
    ds.upload(show_progress=True)
    ds.finalize()
    print(f"[coco] dataset {ds.id} ({args.name} {args.version}), parent {args.parent_id}")


if __name__ == "__main__":
    main()
