"""Follow one real source defect through the synthetic builds: REAL | run C | run D | run E, as 8-bit training tiles.

For each class, picks source defects that were inserted in all three builds (run C = physical_v2, run D = physical_v3,
run E = naive_v3; D and E paired by uid and identical position). For every insertion, the kept training tile that
holds it best (polygon centroid nearest the tile centre) is drawn with its label boxes; the inserted defect's box is
thick, any other box thin. The REAL cell is the source film's own tile from the v1.0 dataset.

    python follow_source.py --n 3 --out ~/debug_pairs/follow
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

H = Path.home()
DATA = H / "swrd_synthetic_physical/data"
RAW = H / "swrd_paper_baseline/data/raw"
REAL_WORK = H / "swrd_paper_baseline/data/work"
REAL_YOLO = H / "swrd_paper_baseline/data/yolo_v1.0_papergrid"
BUILDS = {"C": "physical_v2", "D": "physical_v3", "E": "naive_v3"}
HEAD = {
    "R": "REAL source tile",
    "C": "run C: physical v2 (copied)",
    "D": "run D: physical v3 (varied)",
    "E": "run E: naive v3 (same as D, pasted)",
}
CLASSES = ["inclusion", "crack", "undercut", "lack_of_fusion"]
NAMES = ["porosity", "inclusion", "crack", "undercut", "lack_of_fusion", "lack_of_penetration"]
COLORS = [
    (255, 140, 0),
    (0, 200, 255),
    (60, 60, 255),
    (0, 220, 0),
    (255, 0, 255),
    (0, 255, 255),
]  # BGR
CELL = 340


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def tiles_by_stem(work: Path, keep: set[str] | None) -> dict[str, list[dict]]:
    out = defaultdict(list)
    with open(work / "tiles.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if keep is None or t["tile_id"] in keep:
                out[t["stem"]].append(t)
    return out


def best_tile(tiles: list[dict], pts: np.ndarray) -> dict | None:
    """Tile that contains the whole polygon bbox if possible, else the centroid; nearest-centred wins."""
    cx, cy = pts.mean(0)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    best, score = None, None
    for t in tiles:
        tx, ty, s = t["x"], t["y"], t["side"]
        if not (tx <= cx < tx + s and ty <= cy < ty + s):
            continue
        inside = tx <= x0 and ty <= y0 and x1 < tx + s and y1 < ty + s
        d = np.hypot(cx - (tx + s / 2), cy - (ty + s / 2)) / s
        sc = (0 if inside else 1, d)
        if score is None or sc < score:
            best, score = t, sc
    return best


def source_points(key: str, inv: dict) -> np.ndarray:
    stem, idx = key.rsplit("#", 1)
    d = json.loads((RAW / inv[stem]["json"]).read_text(encoding="utf-8"))
    shapes = []
    for s in d.get("shapes", []):
        pts = np.asarray(s.get("points", []), np.float64)
        if s.get("shape_type") == "rectangle" and len(pts) == 2:
            (a, b), (c, e) = pts
            pts = np.array([[a, b], [c, b], [c, e], [a, e]], np.float64)
        if len(pts):
            shapes.append(pts)
    return shapes[int(idx)]


def draw(
    png: Path, label: Path, tile: dict, pts: np.ndarray, cls: int, head: str, sub: str
) -> np.ndarray:
    g = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    s = g.shape[0]
    b = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    # the box of the followed defect: the label box whose centre is closest to the polygon centroid
    cx, cy = (pts.mean(0) - [tile["x"], tile["y"]]) / tile["side"]
    rows = (
        [ln.split() for ln in label.read_text().splitlines() if ln.strip()]
        if label.is_file()
        else []
    )
    target = None
    if rows:
        dists = [
            (abs(float(r[1]) - cx) + abs(float(r[2]) - cy)) if int(r[0]) == cls else 9 for r in rows
        ]
        target = int(np.argmin(dists)) if min(dists) < 9 else None
    for i, r in enumerate(rows):
        k, bx, by, bw, bh = int(r[0]), *map(float, r[1:5])
        p0 = (int((bx - bw / 2) * s), int((by - bh / 2) * s))
        p1 = (int((bx + bw / 2) * s), int((by + bh / 2) * s))
        cv2.rectangle(b, p0, p1, COLORS[k], max(2, s // 120) if i == target else 1)
    b = cv2.resize(b, (CELL, CELL), interpolation=cv2.INTER_AREA)
    bar = np.full((44, CELL, 3), 255, np.uint8)
    cv2.putText(bar, head, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 90, 90), 1, cv2.LINE_AA)
    return np.vstack([bar, b])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", type=Path, default=H / "debug_pairs/follow")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    rows = {
        v: [r for r in load_jsonl(DATA / f"raw_{b}/inserted.jsonl") if r["class"] != "null"]
        for v, b in BUILDS.items()
    }
    keep = {
        v: set(json.loads((DATA / f"work_{b}/split_tiles.json").read_text())["train"])
        for v, b in BUILDS.items()
    }
    tiles = {v: tiles_by_stem(DATA / f"work_{b}", keep[v]) for v, b in BUILDS.items()}
    inv = {r["stem"]: r for r in json.loads((REAL_WORK / "inventory.json").read_text())["records"]}
    real_tiles = tiles_by_stem(REAL_WORK, None)
    real_png = {
        p.stem: p for part in ("train", "val") for p in (REAL_YOLO / "images" / part).glob("*.png")
    }

    by_src_c = defaultdict(list)
    for r in rows["C"]:
        by_src_c[r["source"]].append(r)
    e_by_uid = {r["uid"]: r for r in rows["E"]}

    for cname in CLASSES:
        cls = NAMES.index(cname)
        cand = []
        for d in rows["D"]:
            if (
                d["class"] != cname
                or d.get("placement") != "geometry"
                or d["source"] not in by_src_c
            ):
                continue
            e = e_by_uid.get(d["uid"])
            if not e or e["source"] != d["source"] or e["top_left"] != d["top_left"]:
                continue
            cand.append((d, e))
        rng.shuffle(cand)
        picked, used = [], set()
        for d, e in cand:
            if d["source"] in used:
                continue
            c = by_src_c[d["source"]][0]
            spts = source_points(d["source"], inv)
            src_stem = d["source"].rsplit("#", 1)[0]
            rt = best_tile(real_tiles[src_stem], spts)
            got = {
                "C": best_tile(tiles["C"][c["film"]], np.asarray(c["points"])),
                "D": best_tile(tiles["D"][d["film"]], np.asarray(d["points"])),
                "E": best_tile(tiles["E"][e["film"]], np.asarray(e["points"])),
            }
            if rt is None or rt["tile_id"] not in real_png or any(v is None for v in got.values()):
                continue
            picked.append((spts, rt, c, d, e, got))
            used.add(d["source"])
            if len(picked) == a.n:
                break
        print(f"{cname}: {len(cand)} candidate pairs, picked {len(picked)}", flush=True)

        lines = []
        for spts, rt, c, d, e, got in picked:
            png = real_png[rt["tile_id"]]
            cells = [
                draw(
                    png,
                    REAL_YOLO / "labels" / png.parent.name / f"{png.stem}.txt",
                    rt,
                    spts,
                    cls,
                    HEAD["R"],
                    d["source"][:44],
                )
            ]
            for v, r in (("C", c), ("D", d), ("E", e)):
                t = got[v]
                y = DATA / f"yolo_{BUILDS[v]}"
                aug = r.get("aug") or {}
                if v == "C":
                    sub = f"flip {r['flip']}  grain scale {r['contrast_scale']:.2f}"
                else:
                    sub = (
                        f"size {aug.get('scale', 1):.2f} depth {aug.get('amplitude', 1):.2f} "
                        f"rot {aug.get('angle_deg', 0):.0f}  u {r.get('u_realised') or 0:+.2f}"
                    )
                cells.append(
                    draw(
                        y / "images/train" / f"{t['tile_id']}.png",
                        y / "labels/train" / f"{t['tile_id']}.txt",
                        t,
                        np.asarray(r["points"]),
                        cls,
                        HEAD[v],
                        sub,
                    )
                )
            gap = np.full((cells[0].shape[0], 8, 3), 255, np.uint8)
            lines.append(np.hstack([x for cell in cells for x in (cell, gap)][:-1]))
            lines.append(np.full((10, lines[-1].shape[1], 3), 255, np.uint8))
        if lines:
            out = np.vstack(lines[:-1])
            path = a.out / f"follow_{cname}.jpg"
            cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print("  ->", path, out.shape, flush=True)


if __name__ == "__main__":
    main()
