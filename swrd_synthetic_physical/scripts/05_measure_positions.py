"""Where across the weld do the defects sit? Measured on real films, and on synthetic films for the check.

Felix (2026-10-16) asked how the pipeline makes sure an inserted defect sits where its class occurs. This
script gives the number behind the answer: for every defect polygon, the across-weld coordinate u of its
centroid from the film's own bead geometry (src/rtsynth/weld_geometry.py: 0 = bead centre line, +-1 = the
toes at half height, |u| > 1 = parent metal). Real films give the per-class distribution; the same
measurement on the synthetic films (polygons with flags.synthetic) is the check.

Writes <out>/<name>.jsonl (one record per polygon) and <out>/<name>.json (per-class summary: counts,
percentiles of u and |u|, and the acceptance band [p2.5, p97.5] of |u| that 02_make_films.py uses).

Real, training films only (film split):
    python 05_measure_positions.py --raw ~/swrd_paper_baseline/data/raw --work-dir ~/swrd_paper_baseline/data/work \
        --grey-steps ../results/grey_steps.json --out ../results --name positions_real
Synthetic films (any version):
    python 05_measure_positions.py --raw ~/swrd_synthetic_physical/data/raw_physical_v3 --synthetic \
        --out ~/swrd_synthetic_physical/data/raw_physical_v3 --name positions_synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "swrd_paper_baseline" / "scripts"))
from _common import CLASS_NAMES, LABEL_TO_CLASS, read_tif

from rtsynth.weld_geometry import polygon_u

TUNGSTEN = "夹钨"
PCT = [2.5, 5, 25, 50, 75, 95, 97.5]


def exposure_of(stem: str) -> str:
    return stem[2:] if stem[:2] in ("A_", "B_") else stem


def measure_film(job: tuple) -> list[dict]:
    raw, image, js, stem, synthetic_only = job
    d = json.loads(Path(js).read_text(encoding="utf-8"))
    shapes = []
    for s in d.get("shapes", []):
        lab = (s.get("label") or "").strip()
        cid = LABEL_TO_CLASS.get(lab)
        if cid is None or lab == TUNGSTEN:
            continue
        syn = bool((s.get("flags") or {}).get("synthetic"))
        if synthetic_only and not syn:
            continue
        pts = np.asarray(s.get("points", []), np.float64)
        if s.get("shape_type") == "rectangle" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
        if len(pts) >= 3:
            shapes.append((CLASS_NAMES[cid], pts, syn, s.get("description") or ""))
    if not shapes:
        return []
    img = read_tif(Path(raw) / image)
    rot = img.shape[0] > img.shape[1]
    if rot:
        w = img.shape[1]
        img = np.rot90(img, 1).copy()
    out = []
    for cls, pts, syn, desc in shapes:
        if rot:
            pts = np.stack([pts[:, 1], w - 1 - pts[:, 0]], axis=1)
        r = polygon_u(img, pts)
        out.append(
            {
                "stem": stem,
                "class": cls,
                "synthetic": syn,
                "u": None if r is None else round(r[0], 4),
                "u_top": None if r is None else round(r[1], 4),
                "u_bottom": None if r is None else round(r[2], 4),
                "height_px": round(float(pts[:, 1].max() - pts[:, 1].min()), 1),
                "description": desc[:80],
            }
        )
    return out


def summarise(rows: list[dict]) -> dict:
    by = defaultdict(list)
    n_all = Counter()
    for r in rows:
        n_all[r["class"]] += 1
        if r["u"] is not None:
            by[r["class"]].append(r["u"])
    out = {}
    for c in CLASS_NAMES:
        u = np.array(by.get(c, []))
        if not len(u):
            out[c] = {"n": n_all[c], "n_defined": 0}
            continue
        au = np.abs(u)
        out[c] = {
            "n": n_all[c],
            "n_defined": len(u),
            "u_percentiles": {
                str(p): round(float(v), 3) for p, v in zip(PCT, np.percentile(u, PCT))
            },
            "abs_u_percentiles": {
                str(p): round(float(v), 3) for p, v in zip(PCT, np.percentile(au, PCT))
            },
            "abs_u_band": [
                round(float(np.percentile(au, 2.5)), 3),
                round(float(np.percentile(au, 97.5)), 3),
            ],
            "share_inside_bead": round(float((au <= 1).mean()), 3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument(
        "--work-dir", type=Path, default=None, help="inventory.json + split_films.json (real)"
    )
    ap.add_argument("--grey-steps", type=Path, default=HERE.parent / "results" / "grey_steps.json")
    ap.add_argument(
        "--synthetic", action="store_true", help="measure only polygons with flags.synthetic"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="positions")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    jobs = []
    if a.synthetic:
        for js in sorted((a.raw / "crop_weld_jsons").rglob("*.json")):
            img = (
                a.raw
                / "crop_weld_images"
                / js.relative_to(a.raw / "crop_weld_jsons").with_suffix(".tif")
            )
            jobs.append((str(a.raw), str(img.relative_to(a.raw)), str(js), js.stem, True))
    else:
        assert a.work_dir, "--work-dir is needed for real films"
        split = json.loads((a.work_dir / "split_films.json").read_text())
        val_exp = {exposure_of(t.split("__")[0]) for t in split["val"]}
        steps = json.loads(a.grey_steps.read_text()) if a.grey_steps.is_file() else {}
        inv = json.loads((a.work_dir / "inventory.json").read_text())
        for r in inv["records"]:
            if (
                r["dtype"] != "uint16"
                or r["exposure"] in val_exp
                or (steps and steps.get(r["stem"], 0) != 1)
                or not r.get("json")
                or not any(r.get("class_instances") or [])
            ):
                continue
            jobs.append((str(a.raw), r["image"], str(a.raw / r["json"]), r["stem"], False))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"[positions] {len(jobs):,} films to measure ({'synthetic' if a.synthetic else 'real'})")
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(measure_film, jobs, chunksize=8)):
            rows += res
            if i % 200 == 0:
                print(f"[positions] {i:,}/{len(jobs):,} films, {len(rows):,} polygons", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / f"{a.name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "films": len(jobs),
        "polygons": len(rows),
        "synthetic": a.synthetic,
        "per_class": summarise(rows),
        "definition": "u = (row - bead centre) / bead half width; bead = longest run above half height "
        "of the smoothed (sigma 6) column-median profile of the polygon's own columns",
    }
    (a.out / f"{a.name}.json").write_text(json.dumps(summary, indent=2))
    for c, s in summary["per_class"].items():
        if s.get("n_defined"):
            print(
                f"{c:20s} n {s['n']:6d} defined {s['n_defined']:6d}  |u| p2.5/50/97.5 "
                f"{s['abs_u_percentiles']['2.5']:.2f}/{s['abs_u_percentiles']['50']:.2f}/"
                f"{s['abs_u_percentiles']['97.5']:.2f}  inside bead {s['share_inside_bead']:.0%}"
            )


if __name__ == "__main__":
    main()
