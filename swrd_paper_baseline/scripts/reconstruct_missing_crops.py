"""Rebuild weld-crop images that the release lacks, from the uncropped originals.

The release (and the official archive) ship label files for 190 T-joint crops in ``crop_weld_jsons/T/2/``
but no image for them. Their originals exist (``Raw_data/images/DJ-RT-20240105-*.tif`` with
``Raw_data/json/*.json``). Sect. 3.1 of the paper says the crops are cut-outs with polygon coordinates
"recalculated" — i.e. translated. So for a cropped label file we can

  1. match each cropped polygon to the original polygon with the same shape (identical point-to-point
     differences) and read the translation (dx, dy) = original − cropped; all matches must agree,
  2. cut ``original[dy:dy+H, dx:dx+W]`` with H, W = the cropped file's imageHeight/imageWidth,
  3. (validation mode) compare that cut-out pixel-for-pixel with a released crop.

Four orientations are tried when matching (identity, rot90, rot180, rot270) in case a crop was rotated.

Run on the box:
  # validate on crops that exist (pixel-exact match proves the method):
  uv run python scripts/reconstruct_missing_crops.py --raw-dir ~/swrd_paper_baseline/data/raw \
      --orig-images ~/swrd_paper_baseline/data/official/Raw_data/images --orig-jsons ~/swrd_paper_baseline/data/official/Raw_data/json \
      --validate crop_weld_jsons/T/1 --limit 40
  # rebuild the missing ones:
  uv run python scripts/reconstruct_missing_crops.py ... --rebuild crop_weld_jsons/T/2 --out-dir ~/swrd_paper_baseline/data/raw/crop_weld_images/T/2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
from _common import exposure_id, load_polygons, read_tif


def _rotate_points(points: np.ndarray, k: int, h: int, w: int) -> np.ndarray:
    """Point coordinates of an (h, w) image after np.rot90(img, k)."""
    x, y = points[:, 0], points[:, 1]
    if k % 4 == 0:
        return points
    if k % 4 == 1:  # rot90 CCW: new (x', y') = (y, w - 1 - x)
        return np.stack([y, (w - 1) - x], axis=1)
    if k % 4 == 2:
        return np.stack([(w - 1) - x, (h - 1) - y], axis=1)
    return np.stack([(h - 1) - y, x], axis=1)  # rot270


def _shape_key(points: np.ndarray) -> tuple:
    """Translation-invariant signature: point count + rounded differences from the first point."""
    d = np.round(points - points[0], 1)
    return (len(points), tuple(map(tuple, d.tolist())))


def find_offset(
    crop_polys, orig_polys, orig_h: int, orig_w: int
) -> tuple[int, int, int, int] | None:
    """Return (k_rot, dx, dy, n_matched) such that crop point = rotated-original point − (dx, dy)."""
    best = None
    for k in range(4):
        rotated = [
            (
                _shape_key(_rotate_points(p.points, k, orig_h, orig_w)),
                _rotate_points(p.points, k, orig_h, orig_w),
            )
            for p in orig_polys
        ]
        by_key: dict[tuple, list[np.ndarray]] = {}
        for key, pts in rotated:
            by_key.setdefault(key, []).append(pts)
        offsets = Counter()
        for cp in crop_polys:
            for pts in by_key.get(_shape_key(cp.points), []):
                d = np.round(pts[0] - cp.points[0]).astype(int)
                offsets[(int(d[0]), int(d[1]))] += 1
        if offsets:
            (dx, dy), n = offsets.most_common(1)[0]
            if best is None or n > best[3]:
                best = (k, dx, dy, n)
    return best


def cut(orig: np.ndarray, k: int, dx: int, dy: int, h: int, w: int) -> np.ndarray:
    img = np.rot90(orig, k) if k else orig
    if dy < 0 or dx < 0 or dy + h > img.shape[0] or dx + w > img.shape[1]:
        raise ValueError(f"crop ({dx},{dy},{w}x{h}) leaves the {img.shape} original")
    return np.ascontiguousarray(img[dy : dy + h, dx : dx + w])


def process(
    crop_json: Path,
    raw_dir: Path,
    orig_images: Path,
    orig_jsons: Path,
    mode: str,
    out_dir: Path | None,
) -> dict:
    crop_polys, header = load_polygons(crop_json)
    h, w = int(header["imageHeight"]), int(header["imageWidth"])
    exp = exposure_id(crop_json.stem)
    orig_json = orig_jsons / f"{exp}.json"
    orig_tif = next(
        (p for p in (orig_images / f"{exp}.tif", orig_images / f"{exp}.tiff") if p.is_file()), None
    )
    res = {"crop": crop_json.stem, "exposure": exp, "status": None}
    if not orig_json.is_file() or orig_tif is None:
        res["status"] = "missing original " + ("json" if not orig_json.is_file() else "tif")
        return res
    orig_polys, _ = load_polygons(orig_json)
    orig = read_tif(orig_tif)
    off = find_offset(crop_polys, orig_polys, orig.shape[0], orig.shape[1])
    if off is None or off[3] < 1:
        res["status"] = "no polygon match"
        return res
    k, dx, dy, n = off
    res |= {"rot90": k, "dx": dx, "dy": dy, "n_matched": n, "n_crop_polys": len(crop_polys)}
    try:
        tile = cut(orig, k, dx, dy, h, w)
    except ValueError as e:
        res["status"] = f"bad offset: {e}"
        return res

    if mode == "validate":
        released = read_tif(
            raw_dir
            / "crop_weld_images"
            / crop_json.parent.relative_to(raw_dir / "crop_weld_jsons")
            / f"{crop_json.stem}.tif"
        )
        if released.shape != tile.shape:
            res["status"] = f"shape differs {released.shape} vs {tile.shape}"
        elif released.dtype != tile.dtype:
            res["status"] = f"dtype differs {released.dtype} vs {tile.dtype}"
        else:
            diff = int(np.count_nonzero(released != tile))
            res["status"] = "exact" if diff == 0 else f"{diff} px differ ({diff / tile.size:.2%})"
    else:
        out = out_dir / f"{crop_json.stem}.tif"
        out.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out, tile)
        res["status"] = f"written {out.name} {tile.shape} {tile.dtype}"
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--orig-images", type=Path, required=True)
    ap.add_argument("--orig-jsons", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--validate",
        help="crop json folder (relative to raw-dir) whose images exist; compare pixel-exact",
    )
    g.add_argument(
        "--rebuild",
        help="crop json folder (relative to raw-dir) whose images are missing; write them",
    )
    ap.add_argument("--out-dir", type=Path, help="destination for --rebuild")
    ap.add_argument(
        "--glob", default="*.json", help="which label files in the folder, e.g. 'B_*.json'"
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.rebuild and not args.out_dir:
        ap.error("--rebuild needs --out-dir")

    folder = args.raw_dir / (args.validate or args.rebuild)
    jsons = sorted(folder.glob(args.glob))[: args.limit]
    mode = "validate" if args.validate else "rebuild"
    results = [
        process(j, args.raw_dir, args.orig_images, args.orig_jsons, mode, args.out_dir)
        for j in jsons
    ]
    for r in results:
        print(
            f"{r['crop']:<28} rot90={r.get('rot90', '-')} dx={r.get('dx', '-')} dy={r.get('dy', '-')} matched={r.get('n_matched', '-')}/{r.get('n_crop_polys', '-')}  {r['status']}"
        )
    print("\nsummary:", dict(Counter(r["status"].split(" ")[0] for r in results)))
    (args.raw_dir.parent / "work").mkdir(exist_ok=True)
    (args.raw_dir.parent / "work" / f"reconstruct_{mode}.json").write_text(
        json.dumps(results, indent=1)
    )


if __name__ == "__main__":
    main()
