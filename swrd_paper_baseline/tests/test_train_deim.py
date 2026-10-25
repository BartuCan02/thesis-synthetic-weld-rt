"""Unit tests for the config side of 14_train_deim.py (no DEIM, no GPU). Run: uv run pytest tests/test_train_deim.py"""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "deim14", Path(__file__).resolve().parents[1] / "scripts" / "14_train_deim.py"
)
deim14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deim14)


def test_schedule_keeps_deim_shape_for_50_epochs():
    # repo: 58 epochs = 50 augmented + 8 plain, flat lr until 4 + 50 // 2 = 29, policy [4, 29, 50]
    assert deim14.deim_schedule(58, 8) == {
        "epoches": 58,
        "flat_epoch": 29,
        "no_aug_epoch": 8,
        "policy_epochs": [4, 29, 50],
        "mixup_epochs": [4, 29],
        "stop_epoch": 50,
    }
    # ours: 50 epochs = 42 augmented + 8 plain
    assert deim14.deim_schedule(50, 8) == {
        "epoches": 50,
        "flat_epoch": 25,
        "no_aug_epoch": 8,
        "policy_epochs": [4, 25, 42],
        "mixup_epochs": [4, 25],
        "stop_epoch": 42,
    }


def _args(**over):
    base = {
        "epochs": 50,
        "no_aug_epochs": 8,
        "imgsz": 352,
        "total_batch": 32,
        "val_batch": 64,
        "workers": 4,
        "checkpoint_freq": 10,
        "lr": None,
    }
    return Namespace(**(base | over))


def test_derived_config_overrides_only_data_size_and_schedule(tmp_path):
    deim_dir, data_root, out_dir = tmp_path / "DEIM", tmp_path / "data", tmp_path / "run"
    cfg = deim14.derived_config(deim_dir, data_root, out_dir, _args())
    assert cfg["__include__"] == [str((deim_dir / deim14.DEIM_CONFIG).resolve())]
    assert cfg["num_classes"] == 6 and cfg["remap_mscoco_category"] is False
    assert cfg["epoches"] == 50 and cfg["flat_epoch"] == 25 and cfg["no_aug_epoch"] == 8
    assert cfg["eval_spatial_size"] == [352, 352]
    train = cfg["train_dataloader"]
    assert train["dataset"]["ann_file"] == str(data_root / "coco" / "instances_train.json")
    assert train["dataset"]["img_folder"] == str(data_root)
    ops = {op["type"]: op for op in train["dataset"]["transforms"]["ops"]}
    assert ops["Mosaic"]["output_size"] == 176  # four quarter tiles make one 352 canvas
    assert ops["Resize"]["size"] == [352, 352]
    assert train["dataset"]["transforms"]["policy"]["epoch"] == [4, 25, 42]
    assert train["collate_fn"] == {"base_size": 352, "mixup_epochs": [4, 25], "stop_epoch": 42}
    assert train["total_batch_size"] == 32
    val = cfg["val_dataloader"]
    assert val["dataset"]["ann_file"] == str(data_root / "coco" / "instances_val.json")
    assert [op["type"] for op in val["dataset"]["transforms"]["ops"]] == [
        "Resize",
        "ConvertPILImage",
    ]
    assert "optimizer" not in cfg  # DEIM's own learning rates unless --lr is given
    assert cfg["HGNetv2"]["local_model_dir"].endswith("weight/hgnetv2/")


def test_lr_override_scales_backbone_lr_with_it(tmp_path):
    cfg = deim14.derived_config(tmp_path, tmp_path, tmp_path, _args(lr=0.00025))
    opt = cfg["optimizer"]
    assert opt["lr"] == 0.00025
    assert opt["params"][0]["lr"] == pytest.approx(0.0000125)  # keeps the 1:20 ratio
    assert opt["params"][1]["weight_decay"] == 0.0


def test_parse_args_refuses_bad_sizes_and_schedules(monkeypatch):
    base = ["14_train_deim.py", "--dataset-id", "d", "--name", "t"]
    monkeypatch.setattr(sys, "argv", base + ["--imgsz", "350"])
    with pytest.raises(SystemExit):
        deim14.parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--epochs", "10"])
    with pytest.raises(SystemExit):
        deim14.parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--init-checkpoint", "/nowhere/x.pth"])
    with pytest.raises(SystemExit):
        deim14.parse_args()
    monkeypatch.setattr(sys, "argv", base)
    args = deim14.parse_args()
    assert args.epochs == 50 and args.imgsz == 352 and args.init_checkpoint is None


def test_parse_log_reads_json_lines(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(
        '{"epoch": 0, "test_coco_eval_bbox": [0.1, 0.2]}\n\n{"epoch": 1, "test_coco_eval_bbox": [0.3, 0.4]}\n'
    )
    rows = deim14.parse_log(log)
    assert [r["epoch"] for r in rows] == [0, 1]
    assert deim14.parse_log(tmp_path / "missing.txt") == []


def test_data_dir_is_local_only(monkeypatch):
    base = ["14_train_deim.py", "--name", "t"]
    monkeypatch.setattr(sys, "argv", base + ["--data-dir", "/x", "--queue", "multi-gpu"])
    with pytest.raises(SystemExit):
        deim14.parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--data-dir", "/x"])
    assert deim14.parse_args().data_dir == Path("/x")
    monkeypatch.setattr(sys, "argv", base)  # neither source
    with pytest.raises(SystemExit):
        deim14.parse_args()
