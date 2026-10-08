"""Unit tests for the split and oversampling options of 05_train.py. Run: uv run pytest tests/test_train_sampling.py"""

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "train05", Path(__file__).resolve().parents[1] / "scripts" / "05_train.py"
)
train05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train05)


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


def test_default_trains_from_the_dataset_folder_untouched(tmp_path):
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
