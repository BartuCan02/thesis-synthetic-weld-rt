"""Step 2: make the synthetic films (physical insertion of real SWRD defects into SWRD training films).

For every class in budget.json, real instances on *training* films (film split) are cut out by removal
(inpainting) and inserted into other training films, with the method of the WP2 trial
(experiments/wp2_physics/trial_log_compositing): grain-ratio contrast scaling, placement by the brightness
profile across the weld, darkening only. Each use of a source is mirrored left-right with probability 0.5.

Sources: a polygon of the class (tungsten inclusions excluded: denser than steel, the sign is flipped),
on a true 16-bit training film (not 8-bit data stored as 16-bit, see 00_grey_steps.py) that is not a byte
copy of a val film, inside the main seam polygon and not on a
T-joint's crossing weld, not cut by the film border, running along the weld if the class is elongated, and
not touched by any other labelled polygon within the removal area. Portrait films are turned 90 degrees so
the weld runs left-right (the profile placement assumes it); hosts are landscape films only.

Hosts: true 16-bit landscape training films (not copies of val films), at least 2000 px wide. A host whose
grain differs from the source's by more than --scale-range is rejected for that defect, which then tries
another host: the contrast scale (host grain / source grain) assumes similar film, and is not forced. Everything already
labelled on a host (defects, pseudo-defects, a second seam polygon) is a keep-out zone grown by one tile side,
so tiles that hold a synthetic defect rarely hold a real one too. Inserted defects keep one tile side apart.
Version 2 (default): no inserted pixel may lie inside any tile the baseline dataset already uses (the
positives and the randomly sampled clean tiles of split_films.json), so the model never sees an inserted
defect's spot without the defect. --allow-baseline-tiles gives version 1.

Writes, under --out:
  crop_weld_images/S/1/<stem>.tif   16-bit film (the release layout, so 00_inventory/01_tile read it as is)
  crop_weld_jsons/S/1/<stem>.json   LabelMe: the host's shapes + the inserted ones (flags.synthetic = true)
  inserted.jsonl                    one record per inserted defect
  make_films_report.json            counts per class, failures, scale and placement statistics, params
  qc_sheet.png                      before/after crops of some insertions

--arm naive writes the A/B control with the identical sources, flips, hosts and positions.

Run on the box (any python with numpy, opencv, tifffile; e.g. the swrd_paper_baseline venv):
    python 02_make_films.py --raw ~/swrd_paper_baseline/data/raw --work-dir ~/swrd_paper_baseline/data/work \
        --budget ../results/budget.json --out ~/swrd_synthetic_physical/data/raw_physical_v1 --arm physical
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "swrd_paper_baseline" / "scripts"))
from _common import CLASS_NAMES, LABEL_TO_CLASS, read_tif, tile_side

from rtsynth.compositing import (
    AdditiveSpace,
    composite_naive,
    composite_physical,
    contrast_scale,
    extract_defect,
    flip_patch,
    place_with_profile,
    source_profile,
)

SEAM = "焊缝"
TUNGSTEN = "夹钨"
ELONGATED = {"crack", "undercut", "lack_of_fusion", "lack_of_penetration"}
SPACE = AdditiveSpace(None)  # linear: the pore-dip test found SWRD raw16 near-additive


def exposure_of(stem: str) -> str:
    return stem[2:] if stem[:2] in ("A_", "B_") else stem


def load_shapes(path: Path) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    shapes = []
    for s in d.get("shapes", []):
        pts = np.asarray(s.get("points", []), np.float64)
        if s.get("shape_type") == "rectangle" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
        if len(pts):
            shapes.append(((s.get("label") or "").strip(), pts))
    return d, shapes


def rotate_points(pts: np.ndarray, width: int) -> np.ndarray:
    """Points of np.rot90(img, 1) (counter-clockwise): (x, y) -> (y, W - 1 - x)."""
    return np.stack([pts[:, 1], width - 1 - pts[:, 0]], axis=1)


def area(pts: np.ndarray) -> float:
    return float(cv2.contourArea(pts.astype(np.float32))) if len(pts) >= 3 else 0.0


def fill(shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.round(pts).astype(np.int32)
    if len(p) >= 3:
        cv2.fillPoly(m, [p], 1)
    else:
        cv2.polylines(m, [p], False, 1, 3)
    return m


# ---------------------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------------------


def find_sources(rec: dict, raw: Path, targets: set[str], a: argparse.Namespace) -> list[dict]:
    h, w = rec["height"], rec["width"]
    rot = h > w
    _, shapes = load_shapes(raw / rec["json"])
    if rot:
        shapes = [(lab, rotate_points(p, w)) for lab, p in shapes]
        h, w = w, h
    seams = [p for lab, p in shapes if lab == SEAM and len(p) >= 3]
    if not seams:
        return []
    main = max(seams, key=area)
    side_seams = [p for p in seams if p is not main]
    others = [(i, lab, p) for i, (lab, p) in enumerate(shapes) if lab != SEAM]
    out = []
    for i, lab, p in others:
        cid = LABEL_TO_CLASS.get(lab)
        if cid is None or lab == TUNGSTEN or CLASS_NAMES[cid] not in targets:
            continue
        if len(p) < 3 or area(p) < a.min_area:
            continue
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        if x0 < 3 or y0 < 3 or x1 > w - 4 or y1 > h - 4:
            continue  # cut by the film border
        if max(x1 - x0, y1 - y0) > a.max_side:
            continue
        name = CLASS_NAMES[cid]
        if name in ELONGATED and (x1 - x0) < (y1 - y0):
            continue
        cx, cy = (float(v) for v in p.mean(0))
        if cv2.pointPolygonTest(main.astype(np.float32), (cx, cy), False) < 0:
            continue
        if any(
            cv2.pointPolygonTest(q.astype(np.float32), (cx, cy), False) >= 0 for q in side_seams
        ):
            continue
        # no other labelled polygon may touch the removal area
        g = a.dilate + 2
        clash = False
        for j, _, q in others:
            if j == i:
                continue
            qx0, qy0 = q.min(0)
            qx1, qy1 = q.max(0)
            if qx0 > x1 + g or qx1 < x0 - g or qy0 > y1 + g or qy1 < y0 - g:
                continue
            ox, oy = int(min(x0, qx0)) - g - 2, int(min(y0, qy0)) - g - 2
            sh = (int(max(y1, qy1)) - oy + g + 3, int(max(x1, qx1)) - ox + g + 3)
            mp = cv2.dilate(fill(sh, p - [ox, oy]), np.ones((2 * g + 1, 2 * g + 1), np.uint8))
            if (mp & fill(sh, q - [ox, oy])).any():
                clash = True
                break
        if clash:
            continue
        out.append(
            {
                "key": f"{rec['stem']}#{i}",
                "stem": rec["stem"],
                "exposure": rec["exposure"],
                "image": rec["image"],
                "rot": rot,
                "index": i,
                "label": lab,
                "class": name,
                "points": p.tolist(),
                "area": round(area(p), 1),
            }
        )
    return out


def extract_film(job: tuple) -> list[tuple]:
    """Extract every chosen source on one film. Returns (key, patch, profile) tuples."""
    raw, image, rot, items, dilate, denoise = job
    film = read_tif(Path(raw) / image)
    if rot:
        film = np.rot90(film, 1).copy()
    out = []
    for key, pts in items:
        patch = extract_defect(
            film, np.asarray(pts), SPACE, dilate_px=dilate, denoise_sigma=denoise
        )
        ph, pw = patch.residual.shape
        sx, sy = patch.origin
        band = film[:, sx : sx + pw].copy()
        band[sy : sy + ph, :] = patch.cleaned_crop
        cleaned = np.zeros_like(film[:, : sx + pw])
        cleaned[:, sx : sx + pw] = band
        out.append((key, patch, source_profile(cleaned, patch)))
    return out


# ---------------------------------------------------------------------------------------------------
# synthetic films
# ---------------------------------------------------------------------------------------------------


def make_film(job: tuple) -> dict:
    (film_idx, rnd, seed, host, raw, out_dir, arm, uses, scale_range, qc, windows) = job
    rng = np.random.default_rng([seed, rnd, film_idx])
    img = read_tif(Path(raw) / host["image"])
    d, shapes = load_shapes(Path(raw) / host["json"])
    hh, hw = img.shape
    side = tile_side(hh, hw)
    seams = [p for lab, p in shapes if lab == SEAM and len(p) >= 3]
    main = max(seams, key=area) if seams else None
    keep = np.zeros(img.shape, np.uint8)
    for lab, p in shapes:
        if lab == SEAM and p is main:
            continue
        keep |= fill(img.shape, p)
    grow = np.ones((2 * side + 1, 2 * side + 1), np.uint8)
    keep = cv2.dilate(keep, grow) > 0
    # version 2: no inserted pixel may fall inside a tile the baseline already trains on, so the model
    # never sees the defect's spot clean
    for wx, wy, ws in windows:
        keep[wy : wy + ws, wx : wx + ws] = True

    cur = img.copy()
    placed, failed, qc_crops = [], [], []
    for use in uses:
        patch, prof = use["patch"], use["prof"]
        if use["flip"]:
            patch = flip_patch(patch)
        try:
            (x, y), score = place_with_profile(cur, keep, prof, patch, rng)
        except RuntimeError:
            failed.append((use["uid"], "no_spot"))
            continue
        ph, pw = patch.residual.shape
        scale = float(contrast_scale(cur, patch, (x, y)))
        if not scale_range[0] <= scale <= scale_range[1]:
            failed.append((use["uid"], "grain_mismatch"))
            continue
        before = cur
        if arm == "physical":
            cur = composite_physical(cur, patch, (x, y), SPACE, scale)
        else:
            cur = composite_naive(cur, patch, (x, y))
        sup = np.zeros(img.shape, np.uint8)
        sup[y : y + ph, x : x + pw] = patch.weight > 0
        keep |= cv2.dilate(sup, grow) > 0
        pts = patch.points + [x, y]
        placed.append(
            {
                "uid": use["uid"],
                "class": use["class"],
                "label": use["label"],
                "source": use["key"],
                "flip": bool(use["flip"]),
                "top_left": [int(x), int(y)],
                "patch_hw": [int(ph), int(pw)],
                "profile_match": round(score, 4),
                "contrast_scale": round(scale, 4),
                "points": np.round(pts, 2).tolist(),
            }
        )
        if qc and len(qc_crops) < 2:
            cx0, cy0 = max(0, x - 60), max(0, y - 60)
            cx1, cy1 = min(hw, x + pw + 60), min(hh, y + ph + 60)
            qc_crops.append((use["class"], before[cy0:cy1, cx0:cx1], cur[cy0:cy1, cx0:cx1]))

    if not placed:
        return {"film_idx": film_idx, "stem": None, "placed": [], "failed": failed, "qc": []}
    code = "P" if arm == "physical" else "N"
    stem = f"SYN-{code}{rnd}{film_idx:05d}-{host['stem']}"
    img_path = Path(out_dir) / "crop_weld_images" / "S" / "1" / f"{stem}.tif"
    js_path = Path(out_dir) / "crop_weld_jsons" / "S" / "1" / f"{stem}.json"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    js_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(img_path, cur)
    d = dict(d)
    d["imagePath"] = f"{stem}.tif"
    d["imageData"] = None
    d["imageHeight"], d["imageWidth"] = hh, hw
    d["shapes"] = list(d.get("shapes", [])) + [
        {
            "label": r["label"],
            "points": r["points"],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {"synthetic": True},
            "description": f"{arm}; source {r['source']}; flip {r['flip']}",
        }
        for r in placed
    ]
    js_path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    for r in placed:
        r["film"] = stem
        r["host"] = host["stem"]
    return {"film_idx": film_idx, "stem": stem, "placed": placed, "failed": failed, "qc": qc_crops}


def qc_sheet(crops: list[tuple], path: Path, n: int = 24) -> None:
    rows = []
    for cls, a, b in crops[:n]:
        lo, hi = np.percentile(a, [0.5, 99.5])

        def st(x, lo=lo, hi=hi):
            return np.clip((x.astype(np.float64) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(
                np.uint8
            )

        pair = np.hstack([st(a), np.full((a.shape[0], 6), 255, np.uint8), st(b)])
        scale = 180 / pair.shape[0]
        pair = cv2.resize(pair, (int(pair.shape[1] * scale), 180), interpolation=cv2.INTER_AREA)
        pair = cv2.copyMakeBorder(pair, 22, 6, 6, 6, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(pair, f"{cls}: before | after", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        rows.append(pair)
    if not rows:
        return
    width = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=255)
        for r in rows
    ]
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="SWRD raw release root")
    ap.add_argument("--work-dir", type=Path, required=True, help="inventory.json, split_films.json")
    ap.add_argument("--budget", type=Path, default=HERE.parent / "results" / "budget.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", choices=["physical", "naive"], default="physical")
    ap.add_argument("--per-film", type=int, default=4, help="defects inserted per synthetic film")
    ap.add_argument("--dilate", type=int, default=6)
    ap.add_argument("--denoise-sigma", type=float, default=1.0)
    ap.add_argument("--min-area", type=float, default=20.0)
    ap.add_argument("--max-side", type=int, default=1500)
    ap.add_argument("--min-host-width", type=int, default=2000)
    ap.add_argument(
        "--scale-range",
        type=float,
        nargs=2,
        default=[0.5, 2.0],
        help="accepted host/source grain ratio",
    )
    ap.add_argument("--grey-steps", type=Path, default=HERE.parent / "results" / "grey_steps.json")
    ap.add_argument("--rounds", type=int, default=10, help="re-try failed placements on new hosts")
    ap.add_argument(
        "--allow-baseline-tiles",
        action="store_true",
        help="version 1: allow defects on spots whose clean tile is in the baseline training set",
    )
    ap.add_argument("--limit", type=int, default=0, help="only this many uses (smoke test)")
    ap.add_argument(
        "--topup-from",
        type=Path,
        default=None,
        help="selection_report.json of an earlier pass on --out: add films until each class reaches its\n"
        "tile target, on hosts not used yet, preferring sources used least so far",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    budget = json.loads(a.budget.read_text())
    target = {r["class"]: r["synthetic_instances"] for r in budget["classes"]}
    prev, prev_uses = [], Counter()
    if a.topup_from:
        # measured on the earlier pass: tiles per inserted defect; missing tiles -> extra defects
        sel = json.loads(a.topup_from.read_text())["per_class"]
        prev = [json.loads(x) for x in (a.out / "inserted.jsonl").read_text("utf-8").splitlines()]
        placed_prev = Counter(r["class"] for r in prev)
        prev_uses = Counter(r["source"] for r in prev)
        target = {}
        for r in budget["classes"]:
            c = r["class"]
            got = sel.get(c, {}).get("tiles_with_inserted", 0)
            tpd = got / max(placed_prev[c], 1)
            missing = r["extra_tiles_to_match_rfs"] - got
            target[c] = math.ceil(1.1 * missing / tpd) if missing > 0 and tpd > 0 else 0
            print(
                f"[topup] {c}: {got:,} tiles from {placed_prev[c]:,} defects ({tpd:.2f}/defect), "
                f"missing {max(missing, 0):,} -> {target[c]:,} more defects"
            )
    split = json.loads((a.work_dir / "split_films.json").read_text())
    val_exp = {exposure_of(t.split("__")[0]) for t in split["val"]}
    # every tile the baseline uses (train and val, positives and the sampled clean ones), per film
    windows = defaultdict(list)
    if not a.allow_baseline_tiles:
        for tid in split["train"] + split["val"]:
            stem, xs, ys, ss = tid.split("__")
            windows[stem].append((int(xs[1:]), int(ys[1:]), int(ss[1:])))
    tainted = set(budget.get("train_exposures_identical_to_val", []))
    steps = json.loads(a.grey_steps.read_text())
    inv = json.loads((a.work_dir / "inventory.json").read_text())
    films = [
        r
        for r in inv["records"]
        if r["dtype"] == "uint16"
        and r["exposure"] not in val_exp
        and r["exposure"] not in tainted
        and steps.get(r["stem"], 0) == 1
        and r.get("json")
        and (a.raw / r["image"]).is_file()
    ]
    print(f"[films] eligible training films: {len(films):,}")

    # ---- sources --------------------------------------------------------------------------------
    rng = random.Random(a.seed)
    sources = defaultdict(list)
    for r in films:
        ci = r.get("class_instances") or []
        if not any(ci[CLASS_NAMES.index(c)] for c in target if CLASS_NAMES.index(c) < len(ci)):
            continue
        for s in find_sources(r, a.raw, set(target), a):
            sources[s["class"]].append(s)
    uses = []
    for c, n in target.items():
        pool = sorted(sources[c], key=lambda s: s["key"])
        rng.shuffle(pool)
        pool.sort(key=lambda s: prev_uses[s["key"]])  # stable: least-used sources first
        print(f"[sources] {c}: {len(pool):,} usable instances for {n:,} insertions")
        if not pool:
            continue
        for k in range(n):
            s = pool[k % len(pool)]
            uses.append(
                {
                    "uid": f"{c}-{'t' if a.topup_from else ''}{k:05d}",
                    "class": c,
                    "label": s["label"],
                    "key": s["key"],
                    "src": s,
                    "flip": rng.random() < 0.5,
                }
            )
    rng.shuffle(uses)
    if a.limit:
        uses = uses[: a.limit]

    # ---- library: extract each needed source once ---------------------------------------------------
    need = {}
    for u in uses:
        need[u["key"]] = u["src"]
    by_film = defaultdict(list)
    for s in need.values():
        by_film[(s["image"], s["rot"])].append((s["key"], s["points"]))
    jobs = [
        (str(a.raw), img, rot, items, a.dilate, a.denoise_sigma)
        for (img, rot), items in by_film.items()
    ]
    lib = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(extract_film, jobs, chunksize=4):
            for key, patch, prof in res:
                lib[key] = (patch, prof)
    print(f"[library] {len(lib):,} defects extracted from {len(jobs):,} films")
    for u in uses:
        u["patch"], u["prof"] = lib[u["key"]]
        del u["src"]

    # ---- hosts and films ----------------------------------------------------------------------------
    used_hosts = {r["host"] for r in prev}
    hosts = [
        r
        for r in films
        if r["width"] > r["height"]
        and r["width"] >= a.min_host_width
        and r["stem"] not in used_hosts
    ]
    hosts.sort(key=lambda r: r["stem"])
    rng.shuffle(hosts)
    print(f"[hosts] {len(hosts):,} landscape training films")
    a.out.mkdir(parents=True, exist_ok=True)
    manifest, failures, qc_crops, film_stems = [], Counter(), [], []
    reasons = Counter()
    pending, host_i, film_idx = uses, 0, (50000 if a.topup_from else 0)
    for rnd in range(a.rounds):
        if not pending:
            break
        jobs = []
        for start in range(0, len(pending), a.per_film):
            group = pending[start : start + a.per_film]
            jobs.append(
                (
                    film_idx,
                    rnd,
                    a.seed,
                    hosts[host_i % len(hosts)],
                    str(a.raw),
                    str(a.out),
                    a.arm,
                    group,
                    tuple(a.scale_range),
                    film_idx % 6 == 0,
                    windows.get(hosts[host_i % len(hosts)]["stem"], []),
                )
            )
            host_i += 1
            film_idx += 1
        by_uid = {u["uid"]: u for u in pending}
        retry = []
        with ProcessPoolExecutor(a.workers) as ex:
            for res in ex.map(make_film, jobs, chunksize=2):
                manifest += res["placed"]
                qc_crops += res["qc"]
                if res["stem"]:
                    film_stems.append(res["stem"])
                for uid, why in res["failed"]:
                    reasons[why] += 1
                    retry.append(by_uid[uid])
        print(
            f"[round {rnd}] films {len(jobs):,}, placed {len(pending) - len(retry):,}, to retry {len(retry):,}"
        )
        pending = retry
    for u in pending:
        failures[u["class"]] += 1

    with open(a.out / "inserted.jsonl", "a" if a.topup_from else "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    placed = Counter(r["class"] for r in manifest)
    sc = np.array([r["contrast_scale"] for r in manifest]) if manifest else np.zeros(1)
    pm = np.array([r["profile_match"] for r in manifest]) if manifest else np.zeros(1)
    report = {
        "arm": a.arm,
        "films": len(film_stems),
        "hosts_used": len({r["host"] for r in manifest}),
        "per_class": {
            c: {
                "target": target[c],
                "placed": placed[c],
                "failed": failures[c],
                "distinct_sources": len({r["source"] for r in manifest if r["class"] == c}),
            }
            for c in target
        },
        "contrast_scale_p5_p50_p95": [round(float(v), 3) for v in np.percentile(sc, [5, 50, 95])],
        "attempts_rejected": dict(reasons),
        "profile_match_p5_p50_p95": [round(float(v), 3) for v in np.percentile(pm, [5, 50, 95])],
        "params": {k: str(v) for k, v in vars(a).items()},
    }
    if a.topup_from:
        old = json.loads((a.out / "make_films_report.json").read_text())
        (a.out / "make_films_report_pass1.json").write_text(json.dumps(old, indent=2))
        all_rows = prev + manifest
        merged = dict(old)
        merged["films"] = old["films"] + report["films"]
        merged["hosts_used"] = len({r["host"] for r in all_rows})
        for c in old["per_class"]:
            merged["per_class"][c] = {
                "target": old["per_class"][c]["target"]
                + report["per_class"].get(c, {}).get("target", 0),
                "placed": sum(1 for r in all_rows if r["class"] == c),
                "failed": old["per_class"][c]["failed"]
                + report["per_class"].get(c, {}).get("failed", 0),
                "distinct_sources": len({r["source"] for r in all_rows if r["class"] == c}),
            }
        merged["topup"] = report
        report = merged
    (a.out / "make_films_report.json").write_text(json.dumps(report, indent=2))
    qc_sheet(qc_crops, a.out / ("qc_sheet_topup.png" if a.topup_from else "qc_sheet.png"))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "films",
                    "hosts_used",
                    "per_class",
                    "contrast_scale_p5_p50_p95",
                    "profile_match_p5_p50_p95",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
