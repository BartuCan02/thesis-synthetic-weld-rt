"""Step 6: register "SWRD v1.0 tiles + synthetic train tiles" as a ClearML child dataset of v1.0.

Same construction as swrd_paper_baseline/scripts/07_merge_upload.py (the customer side run): every file of
the parent is kept, the synthetic tiles are added to images/train and labels/train only, and the val folder
stays byte-identical. Under the film split (05_train.py --split film) every synthetic tile lands in train,
because no synthetic exposure (SYN-...) is in split_films.json's val list.

Refuses to run if the synthetic folder has val tiles or a tile id that already exists in the parent.

ClearML 2.1.12 de-duplicates by content: a new file whose SHA-256 equals a file of the parent is stored as a
reference to the parent's copy but under the new name, and get_local_copy() then fails ("Extracted file missing",
run C 67ab6191, 2026-10-11). Two synthetic crack labels were byte-identical to real ones by chance (full-width box,
same rows). So every synthetic file is hashed first: an image that matches a parent file stops the upload; a label
that matches gets a trailing zero on each number (same values for YOLO, different bytes). --dry-run only reports.

    AWS_PROFILE=data-rw python 04_merge_upload.py --parent-id de772ad9363c4067bed5835e13a9be81 \
        --yolo-dir ~/swrd_synthetic_physical/data/yolo_physical_v1 \
        --work-dir ~/swrd_synthetic_physical/data/work_physical_v1 \
        --raw-dir ~/swrd_synthetic_physical/data/raw_physical_v1 --arm physical
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from clearml import Dataset

HERE = Path(__file__).resolve().parent
PROJECT = "thesis_wp1_benchmark"
DEFAULT_OUTPUT_URI = "s3://clearml-data-deeplify/clearml-datasets"


def sha256(path: Path) -> str:
    """The hash ClearML compares files by (clearml.storage.hashing.sha256sum, whole file)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_label_unique(path: Path, taken: set[str]) -> int:
    """Append a zero to every number of a YOLO label until its hash is not taken. Returns zeros added.
    An empty label (null tile) becomes a single newline, which Ultralytics reads as no boxes."""
    added = 0
    if path.stat().st_size == 0:
        path.write_text("\n")
        assert sha256(path) not in taken, (
            f"{path.name}: even the newline-only label exists in the parent"
        )
        return 1
    while sha256(path) in taken:
        lines = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
        path.write_text(
            "\n".join(" ".join([f[0]] + [v + "0" for v in f[1:]]) for f in lines) + "\n"
        )
        added += 1
        assert added < 5, f"could not make {path.name} unique"
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--parent-id", required=True, help="the SWRD v1.0 tile dataset")
    ap.add_argument("--yolo-dir", type=Path, required=True, help="output of 03_render.py")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True, help="output of 02_make_films.py")
    ap.add_argument("--budget", type=Path, default=HERE.parent / "results" / "budget.json")
    ap.add_argument("--arm", choices=["physical", "naive"], required=True)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--name", default=None, help="default: swrd-paper-tiles-plus-synthetic-<arm>")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--output-uri", default=DEFAULT_OUTPUT_URI)
    ap.add_argument("--dry-run", action="store_true", help="run every check, create nothing")
    a = ap.parse_args()
    name = a.name or f"swrd-paper-tiles-plus-synthetic-{a.arm}"

    img_dir, lab_dir = a.yolo_dir / "images" / "train", a.yolo_dir / "labels" / "train"
    imgs = sorted(p.name for p in img_dir.glob("*.png"))
    labs = sorted(p.name for p in lab_dir.glob("*.txt"))
    assert imgs, f"no synthetic train tiles under {img_dir}"
    assert [Path(i).stem for i in imgs] == [Path(x).stem for x in labs], "image/label mismatch"
    assert all(i.startswith("SYN-") for i in imgs), (
        "a non-synthetic tile is in the synthetic folder"
    )
    val_dir = a.yolo_dir / "images" / "val"
    assert not val_dir.exists() or not any(val_dir.iterdir()), "synthetic folder has val tiles"

    parent = Dataset.get(dataset_id=a.parent_id)
    parent_files = set(parent.list_files())
    clashes = [
        i for i in imgs if f"images/train/{i}" in parent_files or f"images/val/{i}" in parent_files
    ]
    assert not clashes, f"{len(clashes)} tile ids already exist in the parent, e.g. {clashes[:3]}"
    n_parent = Counter(p.split("/")[1] for p in parent_files if p.startswith("images/"))
    selection = json.loads((a.work_dir / "selection_report.json").read_text())
    # 08_match_arms.py may have moved surplus tiles out to match another arm exactly
    trim_path = a.yolo_dir / "trim_report.json"
    trim = json.loads(trim_path.read_text()) if trim_path.is_file() else None
    trimmed_null = (trim or {}).get("dropped_per_bucket", {}).get("null", 0)
    n_null = sum(1 for x in labs if not (lab_dir / x).read_text().strip())
    expected_null = selection.get("null_tiles_train", 0) - trimmed_null
    assert n_null == expected_null, (
        f"{n_null} empty labels, but the selection report minus the trim gives {expected_null} null tiles"
    )
    if trim:
        after = {b: v["after"] for b, v in trim["buckets"].items()}
        assert sum(after.values()) == len(labs), "the trim report does not match the tiles on disk"
        print(f"[merge] matched to {trim['matched_with']} (seed {trim['seed']}): {after}")

    # ClearML content de-duplication guard (see the docstring)
    parent_hashes = {
        fe.hash for fe in parent.file_entries_dict.values() if getattr(fe, "hash", None)
    }
    img_dups = [i for i in imgs if sha256(img_dir / i) in parent_hashes]
    assert not img_dups, (
        f"{len(img_dups)} synthetic images identical to a parent file: {img_dups[:3]}"
    )
    lab_dups = [x for x in labs if sha256(lab_dir / x) in parent_hashes]
    print(f"[merge] content check: 0 images and {len(lab_dups)} labels identical to a parent file")
    if lab_dups and not a.dry_run:
        for x in lab_dups:
            make_label_unique(lab_dir / x, parent_hashes)
        assert not [x for x in labs if sha256(lab_dir / x) in parent_hashes]
        print(f"[merge] rewrote {len(lab_dups)} labels with a trailing zero per number: {lab_dups}")

    budget = json.loads(a.budget.read_text())
    make = json.loads((a.raw_dir / "make_films_report.json").read_text())
    version = make.get("version", 2)
    tiling = json.loads((a.work_dir / "tiles_summary.json").read_text())
    render = json.loads((a.yolo_dir / "render_params.json").read_text())
    print(
        f"[merge] parent {parent.name} {parent.version}: {dict(n_parent)}; "
        f"adding {len(imgs):,} synthetic train tiles ({a.arm})"
    )
    if a.dry_run:
        print("[merge] dry run OK: no clash with the parent; nothing created or rewritten")
        return

    ds = Dataset.create(
        dataset_name=name,
        dataset_project=a.project,
        dataset_version=a.version,
        parent_datasets=[a.parent_id],
        output_uri=a.output_uri,
        description=(
            f"SWRD paper-baseline tiles v1.0 (parent, unchanged incl. val) + synthetic train tiles: real "
            f"SWRD defects of the rare classes moved into other SWRD training films ({a.arm} insertion, "
            f"pipeline version {version}), budget = {budget.get('scale', 1)} x repeat factor sampling "
            f"t={budget['rfs_threshold']}; {n_null} null-implant negative tiles. Thesis WP2."
        ),
    )
    ds.add_files(path=str(img_dir), dataset_path="images/train", verbose=False)
    ds.add_files(path=str(lab_dir), dataset_path="labels/train", verbose=False)
    for p in (
        a.budget,
        a.work_dir / "selection_report.json",
        a.raw_dir / "make_films_report.json",
        a.raw_dir / "inserted.jsonl",
        a.yolo_dir / "render_params.json",
        a.work_dir / "tiles_summary.json",
        *([trim_path] if trim else []),
    ):
        ds.add_files(path=str(p), dataset_path="synthetic")
    ds.set_metadata(
        {
            "parent": a.parent_id,
            "arm": a.arm,
            "synthetic_train_tiles": len(imgs),
            "synthetic_positive_tiles": len(imgs) - n_null,
            "null_tiles_train": n_null,
            "null_tiles_holdout": selection.get("null_tiles_holdout", 0),
            "pipeline_version": version,
            "v3": make.get("v3"),
            "synthetic_films": make["films"],
            "per_class_insertions": make["per_class"],
            "per_class_tiles": selection["per_class"],
            "budget": {r["class"]: r["synthetic_instances"] for r in budget["classes"]},
            "tiling": tiling.get("params"),
            "render": {
                k: render[k] for k in ("p_lo", "p_hi", "clahe_clip", "clahe_grid", "three_channel")
            },
            "val": "identical to parent; film split puts every synthetic tile in train",
            "labels_made_unique_vs_parent": lab_dups,
            "matched_counts": (
                {
                    "matched_with": trim["matched_with"],
                    "seed": trim["seed"],
                    "tiles_per_bucket_after": {b: v["after"] for b, v in trim["buckets"].items()},
                    "dropped_per_bucket": trim["dropped_per_bucket"],
                }
                if trim
                else None
            ),
        },
        metadata_name="pipeline",
    )
    ds.upload(show_progress=True)
    ds.finalize()
    n_child = Counter(p.split("/")[1] for p in ds.list_files() if p.startswith("images/"))
    print(f"[merge] child dataset {ds.id}: {dict(n_child)} (parent {dict(n_parent)})")
    assert n_child["val"] == n_parent["val"], "val changed: must not happen"
    assert n_child["train"] == n_parent["train"] + len(imgs), "train count mismatch"
    print(f"[merge] dataset id {ds.id}  -> use with 05_train.py --dataset-id {ds.id}")


if __name__ == "__main__":
    main()
