"""Unit tests for 09_test_split.py. Run: uv run pytest tests/test_test_split.py"""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "split09", Path(__file__).resolve().parents[1] / "scripts" / "09_test_split.py"
)
split09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split09)


def _toy(n: int = 120):
    """n exposures: class 0 common (every second one), class 3 rare (every 12th), some with nothing."""
    labels = {}
    for i in range(n):
        labs = set()
        if i % 2 == 0:
            labs.add(0)
        if i % 12 == 5:
            labs.add(3)
        labels[f"E{i:03d}"] = labs
    groups = [{"exposures": [e], "forced": None} for e in sorted(labels)]
    return groups, labels


def test_exposure_of_tile_merges_tjoint_halves():
    assert split09.exposure_of_tile("A_DJ-RT-1__x0__y0__s10") == "DJ-RT-1"
    assert split09.exposure_of_tile("B_DJ-RT-1__x5__y0__s10") == "DJ-RT-1"
    assert split09.exposure_of_tile("DJ-RT-2__x0__y0__s10") == "DJ-RT-2"


def test_split_hits_the_size_and_balances_every_class():
    groups, labels = _toy()
    side = split09.stratified_split(groups, labels, n_test=30, seed=0)
    assert sum(1 for s in side.values() if s == "test") == 30
    for c, total in ((0, 60), (3, 10)):
        in_test = sum(1 for e, s in side.items() if s == "test" and c in labels[e])
        assert abs(in_test - total * 30 / 120) <= 1  # 15 of 60, 2.5 of 10
    rare_val = sum(1 for e, s in side.items() if s == "val" and 3 in labels[e])
    assert rare_val >= 7  # the rare class keeps most of its exposures in val too


def test_split_is_seeded():
    groups, labels = _toy()
    a = split09.stratified_split(groups, labels, n_test=30, seed=3)
    b = split09.stratified_split(groups, labels, n_test=30, seed=3)
    c = split09.stratified_split(groups, labels, n_test=30, seed=4)
    assert a == b and a != c


def test_groups_stay_together_and_forced_groups_stay_in_val():
    groups, labels = _toy()
    groups = groups[2:]
    groups.append({"exposures": ["E000", "E001"], "forced": None})  # byte-identical pair, one group
    groups.append({"exposures": ["X"], "forced": "val"})  # has a copy in training
    labels["X"] = {3}
    for seed in range(20):
        side = split09.stratified_split(groups, labels, n_test=30, seed=seed)
        assert side["E000"] == side["E001"]
        assert side["X"] == "val"


def test_build_groups_links_identical_films_and_flags_train_copies():
    val = {"V1", "V2", "V3"}
    train = {"T1"}
    dups = [["A_V1", "B_V2"], ["V3", "T1"]]  # V1 and V2 identical; V3 identical to a training film
    groups = split09.build_groups(val, train, dups)
    by_first = {g["exposures"][0]: g for g in groups}
    assert by_first["V1"]["exposures"] == ["V1", "V2"] and by_first["V1"]["forced"] is None
    assert by_first["V3"]["exposures"] == ["V3"] and by_first["V3"]["forced"] == "val"
    assert len(groups) == 2


def test_summarise_counts_per_side():
    tiles = [
        {"exposure": "E1", "stem": "A_E1", "boxes": ["3 0.5 0.5 0.1 0.1", "3 0.2 0.2 0.1 0.1"]},
        {"exposure": "E1", "stem": "B_E1", "boxes": []},
        {"exposure": "E2", "stem": "E2", "boxes": ["0 0.5 0.5 0.1 0.1"]},
    ]
    s = split09.summarise(tiles, {"E1": "test", "E2": "val"})
    assert s["test"]["exposures"] == 1 and s["test"]["crops"] == 2 and s["test"]["tiles"] == 2
    assert s["test"]["per_class"]["undercut"] == {"exposures": 1, "tiles": 1, "boxes": 2}
    assert s["val"]["positive_tiles"] == 1


def test_split_refuses_a_size_outside_the_set():
    groups, labels = _toy(10)
    with pytest.raises(ValueError):
        split09.stratified_split(groups, labels, n_test=10, seed=0)
    assert Counter(split09.stratified_split(groups, labels, n_test=1, seed=0).values())["test"] == 1
