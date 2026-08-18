from pathlib import Path
import zipfile

import pandas as pd
import pytest

import validator


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def test_duplicate_train_is_rejected(tmp_path, monkeypatch):
    _write_zip(
        tmp_path / "dataset.zip",
        {
            "a/train.csv": "x,target\n1,a\n2,b\n",
            "b/train.csv": "x,target\n3,a\n4,b\n",
        },
    )
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        with pytest.raises(ValueError, match="multiple train.csv"):
            source.unique_csv("train", required=True)
    finally:
        source.close()


def test_minimum_rows_counts_usable_targets(tmp_path, monkeypatch):
    rows = ["x,target"] + [f"{i},a" for i in range(25)] + [f"{i+25}," for i in range(25)]
    _write_zip(tmp_path / "dataset.zip", {"train.csv": "\n".join(rows) + "\n"})
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, meta = validator.build_checks(source, {})
    finally:
        source.close()
    minimum = next(c for c in checks if c["name"] == "minimum_usable_rows")
    assert minimum["successful"] is False
    assert meta["usableTrainRows"] == 25


def test_valid_binary_dataset_passes_core_checks(tmp_path, monkeypatch):
    frame = pd.DataFrame({"x": range(60), "target": ["a", "b"] * 30})
    frame.to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, meta = validator.build_checks(source, {})
    finally:
        source.close()
    assert all(c["successful"] for c in checks)
    assert meta["classCount"] == 2
