"""Unit tests for 13_score_coco.py. Run: uv run pytest tests/test_score_coco.py"""

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "score13", Path(__file__).resolve().parents[1] / "scripts" / "13_score_coco.py"
)
score13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score13)

GT = {
    "images": [
        {
            "id": 1,
            "file_name": "images/train/a__x0__y0__s100.png",
            "width": 100,
            "height": 100,
            "stem": "a__x0__y0__s100",
        },
        {
            "id": 2,
            "file_name": "images/val/b__x0__y0__s100.png",
            "width": 100,
            "height": 100,
            "stem": "b__x0__y0__s100",
        },
        {
            "id": 3,
            "file_name": "images/val/c__x0__y0__s100.png",
            "width": 100,
            "height": 100,
            "stem": "c__x0__y0__s100",
        },
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [10, 10, 20, 20],
            "area": 400,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [50, 50, 30, 10],
            "area": 300,
            "iscrowd": 0,
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 0,
            "bbox": [0, 0, 40, 40],
            "area": 1600,
            "iscrowd": 0,
        },
    ],
    "categories": [
        {"id": 0, "name": "porosity", "supercategory": "weld_defect"},
        {"id": 1, "name": "inclusion", "supercategory": "weld_defect"},
    ],
}


@pytest.fixture
def gt_path(tmp_path):
    p = tmp_path / "instances_val.json"
    p.write_text(json.dumps(GT))
    return p


def perfect_predictions(image_id_as="int", cat_base=0):
    rows = []
    for a in GT["annotations"]:
        img = next(i for i in GT["images"] if i["id"] == a["image_id"])
        rows.append(
            {
                "image_id": img["id"] if image_id_as == "int" else img["stem"],
                "category_id": a["category_id"] + cat_base,
                "bbox": a["bbox"],
                "score": 0.9,
            }
        )
    return rows


def test_perfect_predictions_score_one_everywhere(gt_path, tmp_path):
    pred = tmp_path / "preds.json"
    pred.write_text(json.dumps(perfect_predictions()))
    r = score13.score(gt_path, pred)
    assert r["mAP50"] == pytest.approx(1.0) and r["mAP50-95"] == pytest.approx(1.0)
    assert r["per_class_AP50"] == {"porosity": pytest.approx(1.0), "inclusion": pytest.approx(1.0)}
    assert r["per_class_AR50@100"]["porosity"] == pytest.approx(1.0)
    assert r["images"] == 3 and r["gt_boxes"] == 3 and r["predictions"] == 3


def test_stem_image_ids_are_mapped_and_unknown_images_dropped(gt_path, tmp_path):
    rows = perfect_predictions(image_id_as="stem")
    rows.append(
        {"image_id": "zzz__x0__y0__s1", "category_id": 0, "bbox": [0, 0, 1, 1], "score": 0.5}
    )
    rows.append({"image_id": 99, "category_id": 0, "bbox": [0, 0, 1, 1], "score": 0.5})
    pred = tmp_path / "preds.json"
    pred.write_text(json.dumps(rows))
    r = score13.score(gt_path, pred)
    assert r["predictions"] == 3 and r["predictions_dropped_unknown_image"] == 2
    assert r["mAP50"] == pytest.approx(1.0)


def test_one_based_categories_need_the_cat_base_flag(gt_path, tmp_path):
    pred = tmp_path / "preds.json"
    pred.write_text(json.dumps(perfect_predictions(cat_base=1)))
    assert score13.score(gt_path, pred, cat_base=1)["mAP50"] == pytest.approx(1.0)
    shifted = score13.score(gt_path, pred, cat_base=0)  # inclusion predicted where porosity is
    assert shifted["per_class_AP50"]["porosity"] == pytest.approx(0.0)


def test_missing_class_scores_zero_and_empty_file_does_not_crash(gt_path, tmp_path):
    pred = tmp_path / "preds.json"
    pred.write_text(json.dumps([r for r in perfect_predictions() if r["category_id"] == 0]))
    r = score13.score(gt_path, pred)
    assert r["per_class_AP50"]["porosity"] == pytest.approx(1.0)
    assert r["per_class_AP50"]["inclusion"] == pytest.approx(0.0)
    assert r["mAP50"] == pytest.approx(0.5)
    pred.write_text("[]")
    r = score13.score(gt_path, pred)
    assert r["mAP50"] == 0.0 and r["per_class_AP50"] == {"porosity": 0.0, "inclusion": 0.0}
