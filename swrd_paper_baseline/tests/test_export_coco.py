"""Unit tests for 11_export_coco.py. Run: uv run pytest tests/test_export_coco.py"""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "export11", Path(__file__).resolve().parents[1] / "scripts" / "11_export_coco.py"
)
export11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export11)

NAMES = {
    0: "porosity",
    1: "inclusion",
    2: "crack",
    3: "undercut",
    4: "lack_of_fusion",
    5: "lack_of_penetration",
}
TILES = [
    # folder, stem, (width, height), YOLO label text
    ("train", "film1__x0__y0__s100", (100, 80), "0 0.5 0.5 0.2 0.5\n"),
    ("val", "B_film2__x0__y0__s100", (100, 100), ""),
    ("train", "film3__x0__y0__s64", (64, 64), "3 0.25 0.25 0.5 0.5\n1 0.5 0.5 1 1\n"),
]


@pytest.fixture
def yolo_dir(tmp_path):
    root = tmp_path / "yolo"
    for folder, stem, (w, h), label in TILES:
        (root / "images" / folder).mkdir(parents=True, exist_ok=True)
        (root / "labels" / folder).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / "images" / folder / f"{stem}.png"), np.zeros((h, w), np.uint8))
        (root / "labels" / folder / f"{stem}.txt").write_text(label)
    (root / "swrd6.yaml").write_text(yaml.safe_dump({"names": NAMES}))
    return root


SPLIT = {
    "train": ["film1__x0__y0__s100"],
    "val": ["film3__x0__y0__s64"],
    "test": ["B_film2__x0__y0__s100"],
}


def test_png_size_reads_the_header_without_decoding(yolo_dir):
    assert export11.png_size(yolo_dir / "images" / "train" / "film1__x0__y0__s100.png") == (100, 80)


def test_yolo_line_to_coco_converts_centre_format_to_top_left_pixels():
    cls, bbox = export11.yolo_line_to_coco("0 0.5 0.5 0.2 0.5", 100, 80)
    assert cls == 0
    assert bbox == [40.0, 20.0, 20.0, 40.0]


def test_collect_records_assigns_by_split_and_numbers_tiles_in_sorted_order(yolo_dir):
    records = export11.collect_records(yolo_dir, SPLIT)
    assert [r["stem"] for r in records["train"]] == ["film1__x0__y0__s100"]
    assert [r["stem"] for r in records["val"]] == ["film3__x0__y0__s64"]
    assert [r["stem"] for r in records["test"]] == ["B_film2__x0__y0__s100"]
    # ids are global: B_film2 < film1 < film3 in sorted order
    assert (
        records["test"][0]["id"] == 1
        and records["train"][0]["id"] == 2
        and records["val"][0]["id"] == 3
    )
    # file_name keeps the folder of the v1.0 release, not the split part
    assert records["test"][0]["file_name"] == "images/val/B_film2__x0__y0__s100.png"
    assert records["val"][0]["file_name"] == "images/train/film3__x0__y0__s64.png"


def test_collect_records_refuses_unlisted_or_double_listed_tiles(yolo_dir):
    with pytest.raises(ValueError):
        export11.collect_records(
            yolo_dir, {"train": SPLIT["train"], "val": SPLIT["val"], "test": []}
        )
    with pytest.raises(ValueError):
        export11.collect_records(yolo_dir, SPLIT | {"test": SPLIT["test"] + SPLIT["train"]})
    with pytest.raises(FileNotFoundError):
        export11.collect_records(yolo_dir, SPLIT | {"test": SPLIT["test"] + ["ghost__x0__y0__s1"]})


def test_build_coco_writes_zero_based_categories_and_pixel_boxes(yolo_dir):
    records = export11.collect_records(yolo_dir, SPLIT)
    coco = export11.build_coco(records["val"], NAMES)
    assert [c["id"] for c in coco["categories"]] == [0, 1, 2, 3, 4, 5]
    assert coco["categories"][3]["name"] == "undercut"
    assert len(coco["images"]) == 1 and coco["images"][0]["width"] == 64
    assert coco["images"][0]["exposure"] == "film3"
    assert [a["category_id"] for a in coco["annotations"]] == [3, 1]
    assert coco["annotations"][0]["bbox"] == [0.0, 0.0, 32.0, 32.0]
    assert coco["annotations"][0]["area"] == 1024.0
    assert coco["annotations"][1]["bbox"] == [0.0, 0.0, 64.0, 64.0]
    assert [a["id"] for a in coco["annotations"]] == [1, 2]


def test_build_coco_keeps_clean_tiles_as_images_without_annotations(yolo_dir):
    records = export11.collect_records(yolo_dir, SPLIT)
    coco = export11.build_coco(records["test"], NAMES)
    assert len(coco["images"]) == 1 and coco["annotations"] == []
    assert coco["images"][0]["exposure"] == "film2"  # the B_ half of a T-joint film
    assert export11.summarize(coco) == {
        "images": 1,
        "images_with_boxes": 0,
        "images_clean": 1,
        "boxes": 0,
        "boxes_per_class": {},
    }
    json.dumps(coco)  # serialisable
