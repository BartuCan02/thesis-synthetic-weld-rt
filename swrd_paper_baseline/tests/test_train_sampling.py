"""Unit tests for the split and oversampling options of 05_train.py. Run: uv run pytest tests/test_train_sampling.py"""

import importlib.util
import random
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "train05", Path(__file__).resolve().parents[1] / "scripts" / "05_train.py"
)
train05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train05)


def test_split_defaults_to_film_and_tile_refuses_a_split_file(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["05_train.py", "--data-dir", "d", "--name", "t"])
    assert train05.parse_args().split == "film"
    monkeypatch.setattr(sys, "argv", sys.argv + ["--split", "tile", "--split-file", "s.json"])
    with pytest.raises(SystemExit):
        train05.parse_args()


def test_exposure_of_merges_tjoint_halves_and_drops_the_window():
    assert train05.exposure_of("A_DJ-RT-20220621-10__x0__y0__s343") == "DJ-RT-20220621-10"
    assert train05.exposure_of("B_DJ-RT-20220621-10__x0__y0__s343") == "DJ-RT-20220621-10"
    assert train05.exposure_of("DJ-RT-20220621-10__x172__y0__s343") == "DJ-RT-20220621-10"


def test_repeat_factors_follow_the_lvis_formula():
    # 100 tiles: class 0 in 51 of them (share 0.51), class 1 in 2 of them (share 0.02), 48 empty
    classes = [{0}] * 50 + [{1}] + [{0, 1}] + [set()] * 48
    r_tile, r_class = train05.repeat_factors(classes, threshold=0.08)
    assert r_class[0] == 1.0  # above the threshold: never repeated
    assert r_class[1] == pytest.approx(2.0)  # sqrt(0.08 / 0.02)
    assert r_tile[50] == pytest.approx(2.0)  # the class-1 tile
    assert r_tile[51] == pytest.approx(2.0)  # a tile takes the largest factor of its classes
    assert r_tile[0] == 1.0 and r_tile[-1] == 1.0  # common class, empty tile


def test_round_repeats_keeps_the_mean_and_is_seeded():
    copies = train05.round_repeats([2.25] * 20_000, seed=0)
    assert set(copies) == {2, 3}
    assert sum(copies) / len(copies) == pytest.approx(2.25, abs=0.01)
    assert copies == train05.round_repeats([2.25] * 20_000, seed=0)
    assert train05.round_repeats([1.0, 3.0], seed=5) == [1, 3]  # whole factors are exact


def _fake_dataset(root: Path) -> None:
    """Six tiles of three exposures, spread over the paper split's train and val folders."""
    tiles = {
        "train": {
            "A_E1__x0__y0__s10": "0 0.5 0.5 0.1 0.1",
            "B_E1__x0__y0__s10": "3 0.5 0.5 0.1 0.1",
            "E2__x0__y0__s10": "",
            "E3__x0__y0__s10": "0 0.5 0.5 0.1 0.1",
        },
        "val": {
            "A_E1__x5__y0__s10": "0 0.5 0.5 0.1 0.1",
            "E3__x5__y0__s10": "1 0.5 0.5 0.1 0.1",
        },
    }
    for part, by_id in tiles.items():
        (root / "images" / part).mkdir(parents=True)
        (root / "labels" / part).mkdir(parents=True)
        for tile_id, label in by_id.items():
            (root / "images" / part / f"{tile_id}.png").write_bytes(b"png")
            (root / "labels" / part / f"{tile_id}.txt").write_text(label + ("\n" if label else ""))
    names = {0: "porosity", 1: "inclusion", 2: "crack", 3: "undercut"}
    (root / "swrd6.yaml").write_text(
        yaml.safe_dump({"path": "x", "train": "images/train", "val": "images/val", "names": names})
    )


def _args(root: Path, runs: Path, **kw) -> Namespace:
    base = {"data_dir": root, "dataset_id": None, "split": "tile", "rfs_threshold": 0.0, "seed": 0}
    return Namespace(**(base | kw), runs_dir=runs, name="t")


def test_tile_split_without_rfs_trains_from_the_dataset_folder_untouched(tmp_path):
    root = tmp_path / "ds"
    _fake_dataset(root)
    out, info = train05.resolve_data_yaml(_args(root, tmp_path / "runs"))
    data = yaml.safe_load(out.read_text())
    assert data["path"] == str(root.resolve()) and data["train"] == "images/train"
    assert info == {"split": "tile", "train_tiles": 4, "train_entries": 4}
    assert not (tmp_path / "runs" / "t_view").exists()


def test_film_split_with_rfs_builds_a_symlink_view(tmp_path):
    root = tmp_path / "ds"
    _fake_dataset(root)
    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    film = {"val_exposures": ["E3"], "expected_val_tiles": 2, "source": "test"}
    out, info = train05.resolve_data_yaml(
        _args(root, tmp_path / "runs", split="film", rfs_threshold=0.5), film
    )
    view = Path(yaml.safe_load(out.read_text())["path"])

    # film split: both E3 tiles are val, wherever they sat; E1's val-folder tile moves to train
    assert sorted(p.stem for p in (view / "images" / "val").iterdir()) == [
        "E3__x0__y0__s10",
        "E3__x5__y0__s10",
    ]
    train_ids = sorted(p.stem for p in (view / "images" / "train").iterdir())
    assert train_ids == [
        "A_E1__x0__y0__s10",
        "A_E1__x5__y0__s10",
        "B_E1__x0__y0__s10",
        "E2__x0__y0__s10",
    ]
    # every view file is a symlink into the dataset, and each label sits where Ultralytics looks for it
    for img in (view / "images").rglob("*.png"):
        assert img.is_symlink() and img.resolve().parents[2] == root.resolve()
        label = view / "labels" / img.parent.name / f"{img.stem}.txt"
        assert label.is_symlink() and label.resolve().stem == img.stem

    # RFS on the 4 train tiles: porosity share 2/4 -> r 1; undercut share 1/4 -> r sqrt(0.5/0.25) = 1.41
    lines = (view / "train_rfs.txt").read_text().splitlines()
    assert len(lines) == info["train_entries"]
    assert sum(1 for x in lines if Path(x).stem == "E2__x0__y0__s10") == 1  # empty tile: once
    assert sum(1 for x in lines if Path(x).stem == "B_E1__x0__y0__s10") in (1, 2)
    assert all(Path(x).parent == view / "images" / "train" for x in lines)
    undercut = next(r for r in info["per_class"] if r["class"] == "undercut")
    assert undercut["r_class"] == pytest.approx(1.414, abs=1e-3)
    assert info["train_tiles"] == 4 and info["val_tiles"] == 2

    # the dataset itself is never written to
    assert sorted(p.relative_to(root) for p in root.rglob("*")) == before


def test_film_split_refuses_a_val_count_that_does_not_match_the_split_file(tmp_path):
    root = tmp_path / "ds"
    _fake_dataset(root)
    film = {"val_exposures": ["E3"], "expected_val_tiles": 3, "source": "test"}
    with pytest.raises(RuntimeError, match="film split"):
        train05.resolve_data_yaml(_args(root, tmp_path / "runs", split="film"), film)


def test_film_split_holds_test_exposures_out_of_train_and_val(tmp_path):
    root = tmp_path / "ds"
    _fake_dataset(root)
    film = {
        "val_exposures": ["E3"],
        "expected_val_tiles": 2,
        "test_exposures": ["E1"],
        "expected_test_tiles": 3,  # A_E1 x0, A_E1 x5, B_E1: both T-joint halves of exposure E1
        "source": "test",
    }
    out, info = train05.resolve_data_yaml(_args(root, tmp_path / "runs", split="film"), film)
    view = Path(yaml.safe_load(out.read_text())["path"])
    seen = {p.stem for p in (view / "images").rglob("*.png")}
    assert not any(train05.exposure_of(s) == "E1" for s in seen)  # never in train, never in val
    assert sorted(p.stem for p in (view / "images" / "train").iterdir()) == ["E2__x0__y0__s10"]
    assert not (view / "images" / "test").exists()
    assert info["train_tiles"] == 1 and info["val_tiles"] == 2 and info["test_tiles_held_out"] == 3


def test_film_split_refuses_a_test_count_that_does_not_match_the_split_file(tmp_path):
    root = tmp_path / "ds"
    _fake_dataset(root)
    film = {
        "val_exposures": ["E3"],
        "expected_val_tiles": 2,
        "test_exposures": ["E1"],
        "expected_test_tiles": 2,
        "source": "test",
    }
    with pytest.raises(RuntimeError, match="test tiles"):
        train05.resolve_data_yaml(_args(root, tmp_path / "runs", split="film"), film)


def test_film_split_without_a_test_list_behaves_as_before(tmp_path):
    root = tmp_path / "ds"
    _fake_dataset(root)
    film = {"val_exposures": ["E3"], "expected_val_tiles": 2, "source": "old split file"}
    _, info = train05.resolve_data_yaml(_args(root, tmp_path / "runs", split="film"), film)
    assert info["train_tiles"] == 4 and info["val_tiles"] == 2 and info["test_tiles_held_out"] == 0


class _FakeTask:
    def connect_configuration(self, cfg, name):
        return cfg


def test_connect_film_split_reads_the_test_list(tmp_path):
    import json

    f = tmp_path / "split.json"
    f.write_text(
        json.dumps(
            {
                "train": ["A_E1__x0__y0__s10"],
                "val": ["E3__x0__y0__s10"],
                "test": ["E2__x0__y0__s10", "E2__x5__y0__s10"],
            }
        )
    )
    cfg = train05.connect_film_split(_FakeTask(), f)
    assert cfg["val_exposures"] == ["E3"] and cfg["test_exposures"] == ["E2"]
    assert cfg["expected_test_tiles"] == 2
    f.write_text(json.dumps({"train": [], "val": ["E3__x0__y0__s10"], "test": ["E3__x5__y0__s10"]}))
    with pytest.raises(SystemExit):
        train05.connect_film_split(_FakeTask(), f)


def test_equal_factors_make_every_class_equal_to_the_most_common_one():
    # class 0 in 6 tiles, class 1 in 3, class 3 in 2 (one tile shares 0 and 3), 2 empty tiles
    classes = [{0}] * 5 + [{0, 3}] + [{1}] * 3 + [{3}] + [set()] * 2
    r_tile, r_class = train05.equal_factors(classes)
    # target 6: class 1 x2, class 3 x3; the shared tile (x3) already gives class 0 three views, so the five
    # tiles holding only class 0 keep a fraction (6 - 3) / 5 = 0.6
    assert r_class == pytest.approx({0: 0.6, 1: 2.0, 3: 3.0})
    assert r_tile[:5] == pytest.approx([0.6] * 5)
    assert r_tile[5] == 3.0  # a shared tile takes the largest factor of its other classes
    assert r_tile[6:9] == [2.0] * 3 and r_tile[9] == 3.0
    assert r_tile[10:] == [1.0, 1.0]  # defect-free tiles: once
    views = {c: sum(r for cs, r in zip(classes, r_tile) if c in cs) for c in (0, 1, 3)}
    assert views == pytest.approx({0: 6.0, 1: 6.0, 3: 6.0})


def _mixed_tiles(n: int, pores_with_rare: float, seed: int = 1) -> list[set[int]]:
    """Random tiles: rarer classes 1-5 at SWRD-like rates, sometimes two together; pores in about half of
    the tiles without a rarer class and in a fraction ``pores_with_rare`` of the tiles with one."""
    rng = random.Random(seed)
    classes = []
    for _ in range(n):
        cs = set()
        for c, p in ((1, 0.08), (2, 0.12), (3, 0.01), (4, 0.06), (5, 0.1)):
            if rng.random() < p:
                cs.add(c)
        if rng.random() < (pores_with_rare if cs else 0.5):
            cs.add(0)
        classes.append(cs)
    return classes


def test_equal_factors_are_exact_when_the_pores_leave_room():
    # in SWRD + run E, tiles shared with rarer classes give porosity 29,678 of its 44,259 target: room left
    classes = _mixed_tiles(3000, pores_with_rare=0.2)
    r_tile, _ = train05.equal_factors(classes)
    target = max(sum(1 for cs in classes if c in cs) for c in range(6))
    views = {c: sum(r for cs, r in zip(classes, r_tile) if c in cs) for c in range(6)}
    assert views == pytest.approx({c: target for c in range(6)})
    assert all(r >= 0 for r in r_tile)
    # whole copies keep it within a few percent (each pores-only tile gets 0 or 1 copy)
    copies = train05.round_repeats(r_tile, seed=0)
    for c in range(6):
        seen = sum(k for cs, k in zip(classes, copies) if c in cs)
        assert abs(seen - target) / target < 0.05


def test_equal_factors_cannot_lower_the_most_common_class_below_its_shared_views():
    # pores in most rare-class tiles: repeating those alone already shows pores far above the target, so every
    # pores-only tile is left out (factor 0) and porosity still ends above; the training script warns about this
    classes = _mixed_tiles(3000, pores_with_rare=0.9)
    r_tile, r_class = train05.equal_factors(classes)
    target = max(sum(1 for cs in classes if c in cs) for c in range(6))
    assert r_class[0] == 0.0
    assert sum(r for cs, r in zip(classes, r_tile) if 0 in cs) > 1.02 * target
    assert train05.equal_spread(classes, train05.round_repeats(r_tile, seed=0)) > 0.02


def test_equal_spread_is_the_gap_between_the_most_and_least_seen_class():
    classes = [{0}, {0}, {1}, {1, 0}, set()]
    assert train05.equal_spread(classes, [1, 1, 1, 1, 1]) == pytest.approx((3 - 2) / 3)
    assert train05.equal_spread(classes, [1, 0, 1, 1, 1]) == pytest.approx(
        0.0
    )  # both classes seen twice
    assert train05.equal_spread([set()], [1]) == 0.0


def test_equal_factors_without_defects_is_a_no_op():
    assert train05.equal_factors([set(), set()]) == ([1.0, 1.0], {})


def test_equal_sampling_builds_a_repeat_list_on_the_film_split(tmp_path):
    root = tmp_path / "ds"
    _fake_dataset(root)
    film = {"val_exposures": ["E3"], "expected_val_tiles": 2, "source": "test"}
    out, info = train05.resolve_data_yaml(
        _args(root, tmp_path / "runs", split="film", equal_sampling=True), film
    )
    data = yaml.safe_load(out.read_text())
    view = Path(data["path"])
    assert data["train"] == "train_equal.txt"
    lines = (view / "train_equal.txt").read_text().splitlines()
    # 4 train tiles: porosity in 2 (A_E1 x0, A_E1 x5), undercut in 1 (B_E1), 1 empty (E2): undercut factor 2
    assert sum(1 for x in lines if Path(x).stem == "B_E1__x0__y0__s10") == 2
    assert sum(1 for x in lines if Path(x).stem == "E2__x0__y0__s10") == 1
    assert info["equal_sampling"] is True and info["rfs_threshold"] == 0.0
    undercut = next(r for r in info["per_class"] if r["class"] == "undercut")
    assert undercut["r_class"] == 2.0 and undercut["tiles_per_epoch"] == 2


def test_equal_sampling_with_the_tile_split_does_not_take_the_folder_shortcut(tmp_path):
    root = tmp_path / "ds"
    _fake_dataset(root)
    out, _ = train05.resolve_data_yaml(_args(root, tmp_path / "runs", equal_sampling=True))
    assert yaml.safe_load(out.read_text())["train"] == "train_equal.txt"


def test_equal_sampling_and_rfs_cannot_be_combined(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "05_train.py",
            "--data-dir",
            "d",
            "--name",
            "t",
            "--equal-sampling",
            "--rfs-threshold",
            "0.1",
        ],
    )
    with pytest.raises(SystemExit):
        train05.parse_args()
