"""Export Deeplify's customer films in the SWRD release layout so the paper pipeline can tile them unchanged.

Felix's side question (meeting, relayed 2026-10-05): keep the paper's configuration, train on SWRD plus our
customer films, evaluate on the SWRD val tiles — does our data help or hurt?

Every exported film looks like one of SWRD's ``crop_weld_data`` pairs:

  <out>/crop_weld_images/<source>/<source>__<mongo id>.tif   16-bit, single channel, cropped to the weld seam,
                                                             SWRD polarity (more metal = brighter, pores dark)
  <out>/crop_weld_jsons/<source>/<source>__<mongo id>.json   LabelMe: one polygon per defect instance, labels are
                                                             the six English class names (+ "other_defect")
  <out>/export_report.json                                   per-source counts, skip reasons, polarity and crop checks
  <out>/montage.png                                          --montage N random crops with label and seam overlays

Then run, with --raw-dir pointing at <out>:  00_inventory.py -> 01_tile.py --exclude-from-negatives other_defect
-> 02_select_split.py --val-ratio 0 -> 03_render.py -> 07_merge_upload.py.  (run_customer_pipeline.sh does it.)

Per film, reusing the deeplify dataset_builder readers so the labels mean exactly what they mean there:
  1. newest weld_defect_segmentation annotation (pick_annotation)
  2. image = the raw_extraction_16bit variant (pick_v2_image_variant); films without one are skipped unless
     --allow-fallback-variant (then the annotation's own variant is used, usually an 8-bit CLAHE preview)
  3. polarity canonicalised (canonicalise_polarity): MONOCHROME1 films and aramco's raw extraction are inverted.
     Then MEASURED, not assumed: pores/inclusions must be darker than their surroundings and the seam brighter
     than the plate, as on SWRD (98 % / 95 % / 92 % of instances, 300-film reference). A source that fails the
     pore test aborts the export unless --ignore-polarity-check.
  4. six-class mask (build_mask target=multiclass) at image resolution; the other weld-defect classes
     (burn-through, spatter, excess material, ...) become "other_defect" polygons so their tiles are never negatives
  5. seam = the film's own weld_seam_segmentation label when it has one (--seam-source gt-first, default), else the
     cached seam prediction (--seam-mask-dir, then --fallback-seam-mask-dir); the report says which was used
  6. crop = seam bbox + margin_frac x seam short side (deeplify pipeline-v2 rule), grown to contain every label;
     films without any seam are skipped; crops that are nearly the whole film or whose seam is a sliver are flagged
  7. films with no six-class defect pixel are skipped: on a Labelbox source an empty mask may be a lost label
     (same reasoning as the builder's --drop-unannotated)

Runs in the deeplify mlops environment (Mongo catalogue + S3 + dataset_builder), not in this project's venv:

  cd ~/deeplify-wt-weldsuite/ml/scripts/mlops && AWS_PROFILE=data-rw uv run python \\
      ~/thesis/swrd_paper_baseline/scripts/export_customer_films.py --mlops-dir . \\
      --env-file ~/deeplify/ml/data_management/.env --sources oge aramco maroca \\
      --seam-mask-dir ~/seam_masks_defect_v2 --fallback-seam-mask-dir ~/seam_masks_defect \\
      --montage 24 --out ~/swrd_paper_baseline/data/raw_customer

Smoke test first:  ... --limit 5 --out ~/swrd_paper_baseline/data/raw_customer_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _customer import (
    crop_rect,
    instance_contrast,
    labelme_document,
    mask_to_shapes,
    region_contrast,
    resize_nearest,
)

USE_CASE = "weld_defect_segmentation"
SEAM_USE_CASE = "weld_seam_segmentation"
SIX = ["porosity", "inclusion", "crack", "undercut", "lack_of_fusion", "lack_of_penetration"]
OTHER_DEFECT_LABEL = "other_defect"
SWRD_REFERENCE = {  # 300 SWRD films, 2026-10-05, see _customer.instance_contrast / region_contrast
    "pores_darker": 0.984,
    "inclusions_darker": 0.95,
    "seam_brighter": 0.916,
}
POLARITY_FAIL_BELOW = 0.8  # fraction of pores darker than their ring; SWRD is 0.984
COLORS = {  # BGR for the montage
    1: (0, 0, 255),
    2: (0, 165, 255),
    3: (255, 0, 255),
    4: (0, 255, 255),
    5: (255, 255, 0),
    6: (0, 255, 0),
    7: (128, 128, 128),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mlops-dir", type=Path, required=True, help="deeplify/ml/scripts/mlops checkout"
    )
    ap.add_argument(
        "--env-file",
        type=Path,
        default=Path("~/deeplify/ml/data_management/.env").expanduser(),
        help="dotenv with MONGODB_URI",
    )
    ap.add_argument("--sources", nargs="+", default=["oge", "aramco", "maroca"])
    ap.add_argument(
        "--seam-source",
        choices=["gt-first", "predicted-only", "gt-only"],
        default="gt-first",
        help="where the seam comes from: the film's weld_seam_segmentation label and/or the cached prediction",
    )
    ap.add_argument("--seam-mask-dir", type=Path, required=True, help="<dir>/<source>/<id>.png")
    ap.add_argument("--fallback-seam-mask-dir", type=Path, default=None)
    ap.add_argument(
        "--seam-close-kernel",
        type=int,
        default=15,
        help="gap-closing kernel for GT seam masks drawn as brush masks (seam dataset default 15)",
    )
    ap.add_argument(
        "--margin-frac",
        type=float,
        default=0.1,
        help="crop pad = this x seam short side (deeplify pipeline-v2 default 0.1)",
    )
    ap.add_argument(
        "--allow-fallback-variant",
        action="store_true",
        help="use the annotation's own image variant when no raw_extraction_16bit exists",
    )
    ap.add_argument("--ignore-polarity-check", action="store_true")
    ap.add_argument(
        "--seam-model-task",
        default=None,
        help="ClearML task of a seam model (e.g. 5de01eef56674891a520503d4b400fca = v1.3): films with neither a "
        "seam label nor a cached prediction get one predicted here (GPU) and saved into --seam-mask-dir",
    )
    ap.add_argument(
        "--exclude-films",
        type=Path,
        default=None,
        help="text file, one '<source>/<mongo id>' (or '<source>__<id>') per line: films to leave out",
    )
    ap.add_argument(
        "--include-films",
        type=Path,
        default=None,
        help="text file as above: use only these films",
    )
    ap.add_argument(
        "--keep-clean-films",
        type=Path,
        default=None,
        help="text file as above: films with NO six-class defect that are verified clean and must be exported "
        "anyway (empty label file -> clean tiles only, like SWRD's defect-free films)",
    )
    ap.add_argument(
        "--montage", type=int, default=24, help="crops to draw into <out>/montage.png (0 = none)"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N films per query (smoke test)")
    ap.add_argument(
        "--concurrency", type=int, default=4, help="parallel S3 fetches; each holds a full film"
    )
    ap.add_argument("--dry-run", action="store_true", help="fetch and report, write nothing")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.mlops_dir.resolve()))
    from dotenv import load_dotenv

    load_dotenv(args.env_file)
    if "MONGODB_URI" not in os.environ:
        sys.exit(f"MONGODB_URI not set; expected it in {args.env_file}")

    from dataset_builder.build_weld_defect_dataset import (
        LABEL_MASK,
        LABEL_POLYGON,
        build_mask,
        decode_image,
        fetch,
        pick_annotation,
    )
    from dataset_builder.src.sources.labelbox_json import (
        seam_mask_from_indexed,
        seam_mask_from_objects,
    )
    from dataset_builder.src.sources.oge import binary_defect_mask_from_indexed
    from dataset_builder.src.sources.swrd import (
        ALL_DEFECT_CLASSES,
        BINARY_DEFECT_CLASSES,
        MULTICLASS_DEFECT_ORDER,
        binary_defect_mask,
        parse_labelme,
    )
    from dataset_builder.src.steps.pipeline_v2 import canonicalise_polarity, pick_v2_image_variant
    from dataset_builder.src.steps.seam_crop import load_seam_mask
    from pymongo import MongoClient

    assert list(MULTICLASS_DEFECT_ORDER) == SIX, "class order drifted from the paper's"
    extra_classes = frozenset(ALL_DEFECT_CLASSES - BINARY_DEFECT_CLASSES)
    names = {i + 1: n for i, n in enumerate(SIX)}

    col = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=30000)["deeplify_ml"][
        "samples"
    ]

    def film_list(path: Path | None) -> set[str] | None:
        if path is None:
            return None
        keys = set()
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keys.add(line.replace("__", "/", 1))
        return keys

    exclude, include = film_list(args.exclude_films), film_list(args.include_films)
    keep_clean = film_list(args.keep_clean_films) or set()

    annotated = {"$elemMatch": {"use_case": USE_CASE, "annotations.0": {"$exists": True}}}
    query = {"customer_id": {"$in": args.sources}, "use_cases": annotated}
    if keep_clean:
        # verified-clean films carry no annotation at all; admit them by id like the deeplify builder does
        clean_ids = [k.split("/", 1)[1] for k in keep_clean]
        query = {
            "customer_id": {"$in": args.sources},
            "$or": [{"use_cases": annotated}, {"_id": {"$in": clean_ids}}],
        }
    cursor = col.find(query)
    if args.limit:
        cursor = cursor.limit(args.limit)
    samples = list(cursor)
    print(
        f"[export] {len(samples)} films with a {USE_CASE} annotation or on the keep-clean list in {args.sources}"
    )
    if include is not None or exclude is not None:
        before = len(samples)
        samples = [
            s
            for s in samples
            if (include is None or f"{s['customer_id']}/{s['_id']}" in include)
            and (exclude is None or f"{s['customer_id']}/{s['_id']}" not in exclude)
        ]
        print(f"[export] film lists applied: {before} -> {len(samples)} films")

    def gt_seam(sample: dict, h: int, w: int) -> np.ndarray | None:
        """The film's own seam label as a bool mask at (h, w), or None when it has none."""
        for entry in sample.get("use_cases") or []:
            if entry.get("use_case") != SEAM_USE_CASE:
                continue
            for ann in entry.get("annotations") or []:
                if not ann.get("label_uri"):
                    continue
                payload = fetch(ann["label_uri"])
                if ann.get("label_type") == LABEL_POLYGON:
                    objects = json.loads(payload.decode("utf-8"))
                    lw = ann.get("image_width") or w
                    lh = ann.get("image_height") or h
                    m = seam_mask_from_objects(objects, lw, lh, w, h)
                elif ann.get("label_type") == LABEL_MASK:
                    indexed = decode_image(payload)
                    if indexed is None:
                        continue
                    m = seam_mask_from_indexed(indexed, w, h, args.seam_close_kernel)
                else:
                    continue
                if m is not None and m.any():
                    return m > 0
        return None

    def predicted_seam(source: str, sid: str, shape: tuple[int, int]):
        for d, tag in (
            (args.seam_mask_dir, "predicted"),
            (args.fallback_seam_mask_dir, "predicted_fallback"),
        ):
            if d is None:
                continue
            seam = load_seam_mask(Path(d).expanduser() / source / f"{sid}.png", shape)
            if seam is not None and seam.any():
                return seam, tag
        return None, None

    def seam_for(sample: dict, source: str, sid: str, h: int, w: int):
        if args.seam_source in ("gt-first", "gt-only"):
            m = gt_seam(sample, h, w)
            if m is not None:
                return m, "gt"
            if args.seam_source == "gt-only":
                return None, None
        return predicted_seam(source, sid, (h, w))

    def extra_mask(annotation: dict, payload: bytes, h: int, w: int) -> np.ndarray:
        """{0,1} over the weld-defect classes that are NOT one of the six, at (h, w)."""
        if annotation.get("label_type") == LABEL_MASK:
            indexed = decode_image(payload)
            m = binary_defect_mask_from_indexed(indexed, extra_classes)
            return resize_nearest(m, h, w)
        if annotation.get("label_type") == LABEL_POLYGON:
            parsed = parse_labelme(json.loads(payload.decode("utf-8")))
            return binary_defect_mask(parsed, height=h, width=w, classes=extra_classes)
        return np.zeros((h, w), np.uint8)

    montage_items: list[np.ndarray] = []

    seam_model = {}
    seam_lock = threading.Lock()

    def inline_seam(image_original: np.ndarray, source: str, sid: str) -> np.ndarray | None:
        """Predict the seam with --seam-model-task on the film in its ORIGINAL polarity (what the seam dataset
        was built from), cache it as <seam-mask-dir>/<source>/<id>.png, return it as bool at film resolution."""
        if not args.seam_model_task:
            return None
        with seam_lock:
            if not seam_model:
                seam_model.update(load_seam_model(args.seam_model_task))
            model, dev, mean, std = (seam_model[k] for k in ("model", "dev", "mean", "std"))
            pred = predict_seam(model, dev, mean, std, image_original)
        dest = Path(args.seam_mask_dir).expanduser() / source / f"{sid}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), pred * 255)
        return pred > 0 if pred.any() else None

    def process(sample: dict) -> dict:
        source, sid = sample["customer_id"], str(sample["_id"])
        rec = {"key": f"{source}/{sid}", "source": source}
        picked = pick_annotation(sample)
        if picked is None and rec["key"] not in keep_clean:
            return rec | {"status": "skip: no usable annotation"}
        annotation, ann_variant = picked if picked else (None, None)
        variant = pick_v2_image_variant(sample, fallback=None)
        if variant is None:
            if not args.allow_fallback_variant:
                return rec | {"status": "skip: no raw_extraction_16bit variant"}
            variant = ann_variant
        rec["variant_kind"] = variant.get("kind")
        rec["label_type"] = annotation.get("label_type") if annotation else None
        try:
            image = decode_image(fetch(variant["uri"]))
            if image is None:
                return rec | {"status": "error: image decode failed"}
            if image.ndim == 3:
                image = image[:, :, 0]
            rec["dtype_in"] = str(image.dtype)
            rec["polarity_tag"] = sample.get("polarity")
            image_original = image
            image, inverted = canonicalise_polarity(image, sample.get("polarity"), source)
            rec["inverted"] = bool(inverted)
            h, w = image.shape[:2]
            if annotation is None:
                mask6 = np.zeros((h, w), np.uint8)
                other = np.zeros((h, w), np.uint8)
            else:
                payload = fetch(annotation["label_uri"])
                mask6 = build_mask(annotation, payload, h, w, target="multiclass")
                other = extra_mask(annotation, payload, h, w)
            seam, seam_src = seam_for(sample, source, sid, h, w)
            if seam is None:
                seam = inline_seam(image_original, source, sid)
                seam_src = "predicted_inline" if seam is not None else None
        except Exception as exc:  # noqa: BLE001 — one bad film must not kill the export
            return rec | {"status": f"error: {type(exc).__name__}: {exc}"}
        rec["size_film"] = [int(w), int(h)]
        rec["seam_source"] = seam_src
        if not (mask6 > 0).any() and rec["key"] not in keep_clean:
            return rec | {"status": "skip: no six-class defect pixel (possibly lost labels)"}
        rec["verified_clean"] = rec["key"] in keep_clean and not (mask6 > 0).any()

        # polarity evidence, measured on the whole film before cropping
        pores = instance_contrast(image, mask6 == 1)
        incl = instance_contrast(image, mask6 == 2)
        rec["polarity"] = {
            "pores": [sum(v < 0 for v in pores), len(pores)],
            "inclusions": [sum(v < 0 for v in incl), len(incl)],
            "seam_minus_plate": region_contrast(image, seam) if seam is not None else None,
        }

        rect, crop_status = crop_rect(seam, (mask6 > 0) | (other > 0), args.margin_frac, (h, w))
        rec["crop"] = crop_status
        if rect is None:
            return rec | {"status": "skip: no seam (neither label nor prediction)"}
        y1, x1, y2, x2 = rect
        ys, xs = np.nonzero(seam)
        seam_short = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
        rec["crop_rect"] = [int(v) for v in rect]
        rec["crop_frac_of_film"] = round((y2 - y1) * (x2 - x1) / (h * w), 3)
        rec["seam_short_frac"] = round(seam_short / min(h, w), 3)
        rec["crop_flags"] = [
            flag
            for flag, bad in (
                ("crop_is_whole_film", rec["crop_frac_of_film"] > 0.95),
                ("seam_is_sliver", rec["seam_short_frac"] < 0.05),
                ("crop_extended_to_labels", crop_status == "extended"),
            )
            if bad
        ]
        image = image[y1:y2, x1:x2]
        mask6 = mask6[y1:y2, x1:x2]
        other = other[y1:y2, x1:x2]
        seam = seam[y1:y2, x1:x2]
        rec["size_crop"] = [int(image.shape[1]), int(image.shape[0])]

        shapes = mask_to_shapes(mask6, names)
        shapes += mask_to_shapes(other, {1: OTHER_DEFECT_LABEL})
        counts = Counter(s["label"] for s in shapes)
        rec["instances"] = {k: counts.get(k, 0) for k in SIX + [OTHER_DEFECT_LABEL]}
        if not any(counts.get(k) for k in SIX) and not rec["verified_clean"]:
            return rec | {"status": "skip: no six-class polygon survived the crop"}

        stem = f"{source}__{sid}"
        if not args.dry_run:
            img_path = args.out / "crop_weld_images" / source / f"{stem}.tif"
            json_path = args.out / "crop_weld_jsons" / source / f"{stem}.json"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            if image.dtype != np.uint16:
                image = image.astype(np.uint16)  # 8-bit fallback variants keep their values
            if not cv2.imwrite(str(img_path), image):
                return rec | {"status": f"error: could not write {img_path}"}
            doc = labelme_document(stem, image.shape[0], image.shape[1], shapes)
            json_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        if args.montage:
            overlay = np.where(other > 0, 7, mask6).astype(np.uint8)
            # thumbnail only: keeping full crops for 600 films OOM-killed the box (12.6 GB) on 2026-10-05
            montage_items.append(montage_tile(image, overlay, seam.astype(np.uint8), stem))
        return rec | {"status": "ok", "stem": stem}

    reports: list[dict] = []
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.concurrency) as ex:
        futures = [ex.submit(process, s) for s in samples]
        for i, f in enumerate(as_completed(futures), 1):
            reports.append(f.result())
            if i % 25 == 0 or i == len(futures):
                print(f"[export] {i}/{len(futures)}", flush=True)
            if i % 50 == 0:  # a crash still leaves a usable partial report
                (args.out / "export_report.partial.json").write_text(
                    json.dumps({"n_done": i, "n_total": len(futures), "films": reports}, indent=1)
                )

    per_source = summarise(reports)
    summary = {
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "swrd_reference": SWRD_REFERENCE,
        "n_films": len(reports),
        "per_source": per_source,
        "films": sorted(reports, key=lambda r: r["key"]),
    }
    (args.out / "export_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "export_report.partial.json").unlink(missing_ok=True)
    print_tables(per_source)
    if args.montage and montage_items:
        draw_montage(montage_items, args.montage, args.out / "montage.png")
    print(f"[export] wrote {args.out / 'export_report.json'}")

    bad = {
        s: d["polarity"]["pores_darker"]
        for s, d in per_source.items()
        if d["polarity"]["pores_darker"] is not None
        and d["polarity"]["n_pores"] >= 20
        and d["polarity"]["pores_darker"] < POLARITY_FAIL_BELOW
    }
    if bad and not args.ignore_polarity_check:
        sys.exit(
            f"[export] POLARITY CHECK FAILED for {bad}: pores are not darker than their surroundings "
            f"(SWRD: {SWRD_REFERENCE['pores_darker']}). Do not tile this export."
        )


def load_seam_model(task_id: str, pattern: str = "best_val_loss") -> dict:
    """SegFormer mit_b4 seam model from a ClearML training task (seam_cache_v2 / seam_review recipe)."""
    import re

    import segmentation_models_pytorch as smp
    import torch
    from clearml import Task

    t = Task.get_task(task_id=task_id)
    hp = t.get_parameters() or {}
    mean = float(
        str(hp.get("General/data/normalization/mean", "[0.4832]")).strip("[]").split(",")[0]
    )
    std = float(str(hp.get("General/data/normalization/std", "[0.2406]")).strip("[]").split(",")[0])
    cands = []
    for m in t.get_models().get("output") or []:
        if pattern in (m.name or ""):
            ep = re.search(r"epoch=(\d+)", m.name or "")
            cands.append((int(ep.group(1)) if ep else -1, m))
    assert cands, f"no output model of task {task_id} matches {pattern!r}"
    _, ck = max(cands, key=lambda x: x[0])
    sd = torch.load(ck.get_local_copy(), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    state = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
    model = smp.Segformer(encoder_name="mit_b4", encoder_weights=None, in_channels=1, classes=2)
    model.load_state_dict(state, strict=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[export] seam model {t.name} / {ck.name} on {dev}", flush=True)
    return {"model": model.eval().to(dev), "dev": dev, "mean": mean, "std": std}


def predict_seam(model, dev, mean, std, image: np.ndarray) -> np.ndarray:
    """Short-side-driven inference scale (seam_cache_v2 rule), P1-99 normalised grey input, {0,1} at film size."""
    import torch
    from torchvision.transforms import v2

    a = image.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    h, w = a.shape
    s = min(1.0, max(1024 / max(h, w), 512 / min(h, w)))
    if h * w * s * s > 2_800_000:
        s = (2_800_000 / (h * w)) ** 0.5
    t_ = torch.from_numpy(a)[None]
    t_ = v2.functional.resize(t_, [max(1, int(h * s)), max(1, int(w * s))], antialias=True)
    t_ = v2.Normalize(mean=[mean], std=[std])(t_)
    hh, ww = t_.shape[1], t_.shape[2]
    batch = torch.zeros(1, 1, (hh + 31) // 32 * 32, (ww + 31) // 32 * 32)
    batch[0, :, :hh, :ww] = t_
    with torch.no_grad():
        pr = model(batch.to(dev)).argmax(1).cpu().numpy().astype(np.uint8)[0, :hh, :ww]
    return cv2.resize(pr, (w, h), interpolation=cv2.INTER_NEAREST)


def summarise(reports: list[dict]) -> dict:
    by_source: dict[str, dict] = {}
    for r in reports:
        d = by_source.setdefault(
            r["source"],
            {
                "films": 0,
                "exported": 0,
                "status": Counter(),
                "crop": Counter(),
                "seam_source": Counter(),
                "crop_flags": Counter(),
                "instances": Counter(),
                "pores": [0, 0],
                "inclusions": [0, 0],
                "seam_brighter": [0, 0],
                "inverted": 0,
            },
        )
        d["films"] += 1
        d["status"][r["status"]] += 1
        if "polarity" in r:
            for k in ("pores", "inclusions"):
                d[k][0] += r["polarity"][k][0]
                d[k][1] += r["polarity"][k][1]
            s = r["polarity"]["seam_minus_plate"]
            if s is not None:
                d["seam_brighter"][0] += int(s > 0)
                d["seam_brighter"][1] += 1
        if r["status"] == "ok":
            d["exported"] += 1
            d["crop"][r.get("crop")] += 1
            d["seam_source"][r.get("seam_source")] += 1
            d["crop_flags"].update(r.get("crop_flags", []))
            d["instances"].update(r.get("instances", {}))
            d["inverted"] += int(bool(r.get("inverted")))

    def frac(pair: list[int]) -> float | None:
        return round(pair[0] / pair[1], 3) if pair[1] else None

    return {
        s: {
            "films": d["films"],
            "exported": d["exported"],
            "inverted": d["inverted"],
            "status": dict(d["status"]),
            "seam_source": dict(d["seam_source"]),
            "crop": dict(d["crop"]),
            "crop_flags": dict(d["crop_flags"]),
            "instances": dict(d["instances"]),
            "polarity": {
                "pores_darker": frac(d["pores"]),
                "n_pores": d["pores"][1],
                "inclusions_darker": frac(d["inclusions"]),
                "n_inclusions": d["inclusions"][1],
                "seam_brighter": frac(d["seam_brighter"]),
                "n_seams": d["seam_brighter"][1],
            },
        }
        for s, d in sorted(by_source.items())
    }


def print_tables(per_source: dict) -> None:
    print(
        "\n| source | films | exported | inverted | seam source | crop | flags | skipped / errors |"
    )
    print("|---|---:|---:|---:|---|---|---|---|")
    for s, d in per_source.items():
        skipped = "; ".join(f"{k} ({v})" for k, v in d["status"].items() if k != "ok")
        print(
            f"| {s} | {d['films']} | {d['exported']} | {d['inverted']} | {d['seam_source']} | "
            f"{d['crop']} | {d['crop_flags']} | {skipped} |"
        )
    print("\n| source | pores darker (n) | inclusions darker (n) | seam brighter (n) |")
    print("|---|---:|---:|---:|")
    print(
        f"| SWRD reference | {SWRD_REFERENCE['pores_darker']} (2174) | "
        f"{SWRD_REFERENCE['inclusions_darker']} (119) | {SWRD_REFERENCE['seam_brighter']} (298) |"
    )
    for s, d in per_source.items():
        p = d["polarity"]
        print(
            f"| {s} | {p['pores_darker']} ({p['n_pores']}) | {p['inclusions_darker']} "
            f"({p['n_inclusions']}) | {p['seam_brighter']} ({p['n_seams']}) |"
        )
    cols = SIX + [OTHER_DEFECT_LABEL]
    print("\n| source | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols))
    for s, d in per_source.items():
        print(f"| {s} | " + " | ".join(str(d["instances"].get(k, 0)) for k in cols) + " |")


def montage_tile(
    image: np.ndarray, overlay: np.ndarray, seam: np.ndarray, stem: str, tile_h: int = 256
) -> np.ndarray:
    """One crop as a small BGR tile: P0.5–99.5 stretch, labels filled in class colours, seam outlined in white."""
    lo, hi = np.percentile(image, [0.5, 99.5])
    g = np.clip((image.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1)), 0, 255)
    bgr = cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for v, col in COLORS.items():
        m = overlay == v
        if m.any():
            bgr[m] = (0.5 * bgr[m] + 0.5 * np.array(col)).astype(np.uint8)
    contours, _ = cv2.findContours(seam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, (255, 255, 255), 2)
    scale = tile_h / bgr.shape[0]
    bgr = cv2.resize(bgr, (max(1, int(bgr.shape[1] * scale)), tile_h))
    cv2.putText(bgr, stem[:60], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return bgr


def draw_montage(tiles: list[np.ndarray], n: int, path: Path) -> None:
    """n random prepared tiles stacked vertically."""
    rows = random.Random(0).sample(tiles, min(n, len(tiles)))
    tile_h = rows[0].shape[0]
    width = max(r.shape[1] for r in rows)
    canvas = np.zeros((tile_h * len(rows), width, 3), np.uint8)
    for i, r in enumerate(rows):
        canvas[i * tile_h : (i + 1) * tile_h, : r.shape[1]] = r
    cv2.imwrite(str(path), canvas)
    print(
        f"[export] montage of {len(rows)} crops -> {path}  (red porosity, orange inclusion, magenta crack, "
        f"yellow undercut, cyan LoF, green LoP, grey other_defect, white outline = seam)"
    )


if __name__ == "__main__":
    main()
