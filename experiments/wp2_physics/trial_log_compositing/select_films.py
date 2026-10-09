"""Pick trial films: one source defect per class, and defect-free host films.

Rules (all deterministic under --seed):
- Films: uint16, landscape (seam runs left-right).
- Hosts: no polygon of a defect class; at least 2000 px wide. Pseudo-defects (伪缺陷), other weld labels and
  any second seam polygon (the T-joint intersection) are allowed but become keep-out zones for placement.
  Strictly clean landscape uint16 films do not exist in SWRD: 320 of the 357 defect-free crops are portrait
  B_ halves of T-joints, and 341 carry two seam polygons.
- Sources, per class: an instance whose centroid lies in the seam, that is isolated (no other labelled
  polygon within --isolation px of its box), not on a T-joint's crossing weld (second seam polygon),
  running along the main weld if the class is elongated (crack, undercut, lack of fusion or penetration),
  whose longest box side is <= --max-side px, and whose area
  lies between the 50th and 90th percentile of its class (large enough to see, not an outlier), and whose
  contrast-to-noise ratio (CNR = mean dip in the polygon / grain sigma around it) is at least the median
  CNR of up to --cnr-sample such candidates. This is a visibility filter for a qualitative demo: it
  picks clearly visible defects on purpose and must not be used to build a training set.

Copies the chosen TIFF + LabelMe JSON into --out and writes manifest.json.

Runs on the data box:
    python select_films.py --raw ~/swrd_paper_baseline/data/raw \
        --common-dir ~/thesis/swrd_paper_baseline/scripts --out films
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

SEAM = "焊缝"
PSEUDO = "伪缺陷"
CLASS_LABELS = {  # class name -> raw label strings (tungsten inclusion left out: it is denser than steel)
    "porosity": ["气孔"],
    "inclusion": ["夹渣"],
    "crack": ["裂纹"],
    "undercut": ["咬边"],
    "lack_of_fusion": ["未熔合"],
    "lack_of_penetration": ["未焊透"],
}
ELONGATED = {"crack", "undercut", "lack_of_fusion", "lack_of_penetration"}
DEFECT_LABELS = {"气孔", "夹渣", "夹钨", "裂纹", "咬边", "未熔合", "未焊透"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument(
        "--common-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "swrd_paper_baseline" / "scripts",
    )
    ap.add_argument("--out", type=Path, default=Path("films"))
    ap.add_argument("--n-hosts", type=int, default=6)
    ap.add_argument("--isolation", type=int, default=20)
    ap.add_argument("--max-side", type=int, default=700)
    ap.add_argument("--cnr-sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sys.path.insert(0, str(args.common_dir))
    from _common import load_polygons, read_tif

    def cnr(d: dict) -> float:
        img = read_tif(Path(d["tif"])).astype(np.float64)
        pts = np.asarray(d["points"], np.float32)
        x0, y0 = np.floor(pts.min(0)).astype(int) - 60
        x1, y1 = np.ceil(pts.max(0)).astype(int) + 61
        x0, y0 = max(0, x0), max(0, y0)
        crop = img[y0:y1, x0:x1]
        m = np.zeros(crop.shape, np.uint8)
        cv2.fillPoly(m, [np.round(pts - [x0, y0]).astype(np.int32)], 1)
        ring = (cv2.dilate(m, np.ones((19, 19), np.uint8)) > 0) & ~(
            cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0
        )
        hp = crop - cv2.GaussianBlur(crop, (0, 0), 2.0)
        v = hp[cv2.dilate(m, np.ones((7, 7), np.uint8)) == 0]
        sigma = 1.4826 * np.median(np.abs(v - np.median(v)))
        return float((np.median(crop[ring]) - crop[m > 0].mean()) / max(sigma, 1e-9))

    rng = random.Random(args.seed)
    tif_by_stem = {p.stem: p for p in (args.raw / "crop_weld_images").glob("*/*/*.tif")}
    jsons = sorted((args.raw / "crop_weld_jsons").glob("*/*/*.json"))

    hosts, cands = [], {c: [] for c in CLASS_LABELS}
    area_by_class = {c: [] for c in CLASS_LABELS}
    for jp in jsons:
        tif = tif_by_stem.get(jp.stem)
        if tif is None:
            continue
        polys, hdr = load_polygons(jp)
        h, w = hdr["imageHeight"], hdr["imageWidth"]
        for c, labels in CLASS_LABELS.items():
            area_by_class[c] += [p.area for p in polys if p.label in labels]
        seams = [p for p in polys if p.label == SEAM]
        if w <= h or not seams:
            continue
        with tifffile.TiffFile(tif) as tf:
            if tf.pages[0].dtype != np.uint16:
                continue
        non_seam = [p for p in polys if p.label != SEAM]
        if not any(p.label in DEFECT_LABELS for p in polys):
            if w >= 2000:
                hosts.append({"stem": jp.stem, "tif": str(tif), "json": str(jp), "w": w, "h": h})
            continue
        main_seam = max(seams, key=lambda p: p.area)
        seam_pts = main_seam.points.astype(np.float32)
        side_seams = [p.points.astype(np.float32) for p in seams if p is not main_seam]
        for c, labels in CLASS_LABELS.items():
            for i, p in enumerate(non_seam):
                if p.label not in labels:
                    continue
                x0, y0, x1, y1 = p.bbox
                if max(x1 - x0, y1 - y0) > args.max_side:
                    continue
                cx, cy = p.points.mean(axis=0)
                if cv2.pointPolygonTest(seam_pts, (float(cx), float(cy)), False) < 0:
                    continue
                if any(
                    cv2.pointPolygonTest(q, (float(cx), float(cy)), False) >= 0 for q in side_seams
                ):
                    continue  # on the T-joint's crossing weld: no counterpart on a host
                if c in ELONGATED and (x1 - x0) < (y1 - y0):
                    continue  # must run along the main weld
                m = args.isolation
                clash = False
                for j, q in enumerate(non_seam):
                    if j == i:
                        continue
                    qx0, qy0, qx1, qy1 = q.bbox
                    if qx0 < x1 + m and qx1 > x0 - m and qy0 < y1 + m and qy1 > y0 - m:
                        clash = True
                        break
                if clash:
                    continue
                cands[c].append(
                    {
                        "stem": jp.stem,
                        "tif": str(tif),
                        "json": str(jp),
                        "w": w,
                        "h": h,
                        "label": p.label,
                        "points": p.points.tolist(),
                        "area": p.area,
                    }
                )

    chosen = {"seed": args.seed, "hosts": [], "sources": {}}
    rng.shuffle(hosts)
    chosen["hosts"] = hosts[: args.n_hosts]
    for c, lst in cands.items():
        lo, hi = np.percentile(area_by_class[c], [50, 90])
        ok = [d for d in lst if lo <= d["area"] <= hi]
        print(f"{c}: {len(lst)} isolated in-seam candidates, {len(ok)} in the 50-90th area band")
        if not ok:
            ok = lst
        if ok:
            sample = rng.sample(ok, min(len(ok), args.cnr_sample))
            scored = [(cnr(d), d) for d in sample]
            med = float(np.median([v for v, _ in scored]))
            vis = [d | {"cnr": round(v, 2)} for v, d in scored if v >= med]
            print(f"   CNR median {med:.2f} over {len(scored)}; {len(vis)} at or above it")
            chosen["sources"][c] = rng.choice(vis)
    print(f"hosts available: {len(hosts)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for d in chosen["hosts"] + list(chosen["sources"].values()):
        for key in ("tif", "json"):
            shutil.copy2(d[key], args.out / Path(d[key]).name)
            d[key] = Path(d[key]).name
    (args.out / "manifest.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "hosts": [h["stem"] for h in chosen["hosts"]],
                "sources": {c: (d["stem"], round(d["area"])) for c, d in chosen["sources"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
