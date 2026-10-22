"""Unit tests for swrd_synthetic_physical/scripts/08_match_arms.py. Run: uv run --with pytest pytest tests/"""

import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "match08",
    Path(__file__).resolve().parents[1]
    / "swrd_synthetic_physical"
    / "scripts"
    / "08_match_arms.py",
)
match08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(match08)


def _arm(root: Path, tiles: dict[str, str]) -> Path:
    for sub in ("images/train", "labels/train"):
        (root / sub).mkdir(parents=True)
    for tid, label in tiles.items():
        (root / "images/train" / f"{tid}.png").write_bytes(b"png")
        (root / "labels/train" / f"{tid}.txt").write_text(label)
    return root


def _tiles(n_inclusion: int, n_undercut: int, n_null: int, tag: str) -> dict[str, str]:
    out = {f"{tag}inc{i:03d}": "1 0.5 0.5 0.1 0.1\n" for i in range(n_inclusion)}
    out |= {f"{tag}und{i:03d}": "3 0.5 0.5 0.4 0.1\n" for i in range(n_undercut)}
    out |= {f"{tag}nul{i:03d}": "" for i in range(n_null)}
    return out


def test_bucket_of():
    assert match08.bucket_of("") == "null"
    assert (
        match08.bucket_of("\n") == "null"
    )  # 04_merge_upload's de-duplication guard writes a bare newline
    assert match08.bucket_of("3 0.1 0.1 0.1 0.1\n3 0.5 0.5 0.1 0.1\n") == "3"
    assert match08.bucket_of("1 0.1 0.1 0.1 0.1\n3 0.5 0.5 0.1 0.1\n") == "1+3"


def test_plan_trim_takes_the_smaller_count_per_bucket_and_is_seeded():
    a = {"1": [f"a{i}" for i in range(10)], "null": ["n1", "n2"]}
    b = {"1": [f"b{i}" for i in range(7)], "null": ["m1", "m2", "m3"], "3": ["u1"]}
    drops = match08.plan_trim([a, b], seed=0)
    assert len(drops[0]["1"]) == 3 and "1" not in drops[1]
    assert len(drops[1]["null"]) == 1 and drops[1]["3"] == [
        "u1"
    ]  # a bucket missing in one arm drops to zero
    assert drops == match08.plan_trim([a, b], seed=0)
    assert drops != match08.plan_trim([a, b], seed=1)


def test_apply_moves_surplus_and_writes_reports(tmp_path, monkeypatch):
    pa = _arm(tmp_path / "phys", _tiles(10, 4, 3, "p"))
    pb = _arm(tmp_path / "naive", _tiles(8, 5, 3, "n"))
    monkeypatch.setattr(sys, "argv", ["08", "--arm", str(pa), "--arm", str(pb), "--apply"])
    match08.main()
    for d in (pa, pb):
        counts = {k: len(v) for k, v in match08.read_arm(d).items()}
        assert counts == {"1": 8, "3": 4, "null": 3}
    assert (
        len(list((pa / "trimmed/images").glob("*.png"))) == 2
    )  # two inclusion tiles moved, not deleted
    assert len(list((pb / "trimmed/labels").glob("*.txt"))) == 1  # one undercut tile
    rep = json.loads((pa / "trim_report.json").read_text())
    assert rep["buckets"]["1"] == {"before": 10, "after": 8} and rep["dropped_per_bucket"] == {
        "1": 2
    }
    assert rep["matched_with"] == str(pb.resolve())

    # a second run finds the arms equal: nothing moves, the original "before" counts and drops are kept
    match08.main()
    rep2 = json.loads((pa / "trim_report.json").read_text())
    assert rep2["buckets"]["1"] == {"before": 10, "after": 8} and rep2["dropped_per_bucket"] == {
        "1": 2
    }
    assert len(list((pa / "trimmed/images").glob("*.png"))) == 2


def test_dry_run_moves_nothing(tmp_path, monkeypatch):
    pa = _arm(tmp_path / "phys", _tiles(5, 0, 1, "p"))
    pb = _arm(tmp_path / "naive", _tiles(3, 0, 1, "n"))
    monkeypatch.setattr(sys, "argv", ["08", "--arm", str(pa), "--arm", str(pb)])
    match08.main()
    assert len(match08.read_arm(pa)["1"]) == 5 and not (pa / "trim_report.json").exists()
