"""The placement check Felix asked for: across-weld position of real vs synthetic defects, per class.

Input: one or more 05_measure_positions.py .jsonl files (real first, then any synthetic versions). Output:
one PNG with a histogram of |u| per class and source (0 = bead centre line, 1 = toe, > 1 parent metal),
plus a small JSON with the per-class medians and a two-sample Kolmogorov-Smirnov distance D between the
real and each synthetic distribution (no scipy: D is computed directly; the value matters, not a p-value).

    python 07_position_plot.py --real positions/positions_real.jsonl \
        --synthetic "run C (v2)=raw_physical_v2/positions_synthetic.jsonl" \
        --synthetic "v3=raw_physical_v3/positions_synthetic.jsonl" --out position_check.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["inclusion", "crack", "undercut", "lack_of_fusion"]


def load(path: Path) -> dict[str, np.ndarray]:
    by = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("u") is not None:
            by[r["class"]].append(abs(r["u"]))
    return {c: np.array(v) for c, v in by.items()}


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--synthetic", action="append", default=[], help="name=path.jsonl, repeatable")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    real = load(a.real)
    syn = []
    for spec in a.synthetic:
        name, path = spec.split("=", 1)
        syn.append((name, load(Path(path).expanduser())))

    bins = np.linspace(0, 2.5, 26)
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(4.2 * len(CLASSES), 3.6), sharey=False)
    summary = {}
    for ax, c in zip(axes, CLASSES):
        r = real.get(c, np.array([]))
        ax.hist(r, bins=bins, density=True, alpha=0.45, color="0.3", label=f"real (n={len(r)})")
        summary[c] = {"real_median": round(float(np.median(r)), 3) if len(r) else None}
        for k, (name, d) in enumerate(syn):
            s = d.get(c, np.array([]))
            ax.hist(
                s,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=f"C{k}",
                label=f"{name} (n={len(s)})",
            )
            summary[c][name] = {
                "median": round(float(np.median(s)), 3) if len(s) else None,
                "ks_distance_to_real": round(ks_distance(r, s), 3),
            }
        ax.axvline(1.0, color="k", linestyle=":", linewidth=1)
        ax.set_title(c.replace("_", " "))
        ax.set_xlabel("|u|  (0 = bead centre, 1 = toe)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("density")
    fig.suptitle("Across-weld position of defects: real SWRD vs synthetic insertions", y=1.02)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
