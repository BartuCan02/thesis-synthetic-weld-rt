"""Unit tests for the data side of 15_train_rfdetr.py (no rfdetr, no GPU). Run: uv run pytest tests/test_train_rfdetr.py"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "rfdetr15", Path(__file__).resolve().parents[1] / "scripts" / "15_train_rfdetr.py"
)
rfdetr15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rfdetr15)

CATEGORIES = [
    {"id": 0, "name": "porosity", "supercategory": "weld_defect"},
    {"id": 1, "name": "inclusion", "supercategory": "weld_defect"},
]


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "dataset"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "coco").mkdir()
    images = {
        "train": [(1, "images/train/a__x0__y0__s100.png")],
        "val": [(2, "images/val/b__x0__y0__s100.png"), (3, "images/train/c__x0__y0__s100.png")],
        "test": [(4, "images/val/d__x0__y0__s100.png")],
    }
    for part, items in images.items():
        for _, rel in items:
            (root / rel).write_bytes(b"png")
        coco = {
            "images": [
                {"id": i, "file_name": rel, "width": 100, "height": 100} for i, rel in items
            ],
            "annotations": [],
            "categories": CATEGORIES,
        }
        (root / "coco" / f"instances_{part}.json").write_text(json.dumps(coco))
    return root


def test_rf_layout_has_train_and_valid_with_symlinks_and_basenames(data_root, tmp_path):
    out = rfdetr15.build_rf_layout(data_root, tmp_path / "rf")
    assert sorted(p.name for p in out.iterdir()) == ["train", "valid"]  # no test folder: held out
    valid = json.loads((out / "valid" / "_annotations.coco.json").read_text())
    assert [im["file_name"] for im in valid["images"]] == [
        "b__x0__y0__s100.png",
        "c__x0__y0__s100.png",
    ]
    link = out / "valid" / "c__x0__y0__s100.png"
    assert (
        link.is_symlink()
        and link.resolve() == (data_root / "images" / "train" / "c__x0__y0__s100.png").resolve()
    )
    assert (out / "train" / "a__x0__y0__s100.png").is_symlink()
    assert valid["categories"] == CATEGORIES
    # rebuilding replaces the folder instead of failing on existing links
    rfdetr15.build_rf_layout(data_root, tmp_path / "rf")


def test_label_to_category_matches_by_name_and_refuses_unknown_names():
    assert rfdetr15.label_to_category(["porosity", "inclusion"], CATEGORIES) == {0: 0, 1: 1}
    assert rfdetr15.label_to_category(["inclusion", "porosity"], CATEGORIES) == {0: 1, 1: 0}
    with pytest.raises(ValueError):
        rfdetr15.label_to_category(["porosity", "weld"], CATEGORIES)


def test_parse_args_enforces_the_block_size(monkeypatch):
    base = ["15_train_rfdetr.py", "--dataset-id", "d", "--name", "t"]
    monkeypatch.setattr(sys, "argv", base + ["--imgsz", "350"])
    with pytest.raises(SystemExit):
        rfdetr15.parse_args()
    monkeypatch.setattr(sys, "argv", base)
    args = rfdetr15.parse_args()
    assert (
        args.imgsz == 352 and args.variant == "large" and args.batch == 4 and args.grad_accum == 1
    )
    assert rfdetr15.VARIANTS[args.variant] == "RFDETRLarge"


def test_data_dir_is_local_only(monkeypatch):
    base = ["15_train_rfdetr.py", "--name", "t"]
    monkeypatch.setattr(sys, "argv", base + ["--data-dir", "/x", "--queue", "multi-gpu"])
    with pytest.raises(SystemExit):
        rfdetr15.parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--data-dir", "/x"])
    assert rfdetr15.parse_args().data_dir == Path("/x")
