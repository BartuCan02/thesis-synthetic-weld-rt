"""Step 0: find 16-bit films that are really 8-bit images scaled up.

Measured 2026-10-09: 158 of the 4,650 uint16 SWRD crops have grey values on a grid of 257 (= 65535 / 255),
i.e. 8-bit data stored as 16-bit, with only tens to a few hundred distinct levels. Adding a smooth defect
to such a film would put finer grey steps inside the defect than around it, a tell a model could learn.
These films are excluded as sources and hosts.

For every uint16 film: the greatest common divisor of the steps between its distinct grey values
(every 3rd pixel in both directions). 1 = true 16-bit data. Writes <out>/grey_steps.json {stem: gcd}.

    python 00_grey_steps.py --raw ~/swrd_paper_baseline/data/raw --work-dir ~/swrd_paper_baseline/data/work \
        --out ../results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "swrd_paper_baseline" / "scripts"))
from _common import read_tif


def step(job: tuple[str, str]) -> tuple[str, int]:
    raw, image = job
    u = np.unique(read_tif(Path(raw) / image)[::3, ::3])
    return Path(image).stem, int(np.gcd.reduce(np.diff(u).astype(np.int64))) if len(u) > 1 else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE.parent / "results")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = a.out / "grey_steps.json"
    if out.is_file():
        print(f"[grey steps] {out} exists, kept")
        return
    recs = json.loads((a.work_dir / "inventory.json").read_text())["records"]
    jobs = [
        (str(a.raw), r["image"])
        for r in recs
        if r["dtype"] == "uint16" and (a.raw / r["image"]).is_file()
    ]
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(step, jobs, chunksize=16))
    a.out.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=0, sort_keys=True))
    print(
        f"[grey steps] {len(res):,} uint16 films; step gcd: {Counter(res.values()).most_common()}"
    )


if __name__ == "__main__":
    main()
