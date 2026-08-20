import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

import validator


@pytest.fixture(autouse=True)
def _reset_limits():
    # Re-load limits from a clean environment before each test.
    validator._load_limits()
    yield


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


def test_malformed_numeric_env_yields_structured_failure(tmp_path, monkeypatch):
    # A malformed platform limit must produce a failure result.json via main(),
    # not crash before result/callback handling.
    monkeypatch.setenv("DIMER_MAX_SINGLE_CSV_BYTES", "not-an-int")
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(validator, "RESULT_PATH", tmp_path / "result.json")
    assert validator.main() == 1
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["successful"] is False


def test_compression_ratio_rejected(tmp_path, monkeypatch):
    zpath = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("train.csv", "x,target\n" + ("0,0\n" * 200_000))  # highly compressible
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    with pytest.raises(ValueError, match="compression ratio"):
        validator.DatasetSource()


def test_row_count_limit_enforced_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("DIMER_MAX_TOTAL_ROWS", "10")
    validator._load_limits()
    pd.DataFrame({"x": range(60), "target": ["a", "b"] * 30}).to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, meta = validator.build_checks(source, {})
    finally:
        source.close()
    names = {c["name"]: c["successful"] for c in checks}
    assert names["row_count_within_limit"] is False
    # early return: downstream checks did not run on a truncated frame
    assert "target_has_multiple_classes" not in names
    assert meta["rowCount"] == ">10"


def test_non_finite_limit_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DIMER_MAX_COMPRESSION_RATIO", "inf")
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(validator, "RESULT_PATH", tmp_path / "result.json")
    assert validator.main() == 1
    assert json.loads((tmp_path / "result.json").read_text())["successful"] is False


def test_normalize_member_rejects_traversal_and_absolute():
    assert validator._normalize_member("train.csv") == "train.csv"
    assert validator._normalize_member("./train.csv") == "train.csv"
    assert validator._normalize_member("dataset/train.csv") == "train.csv"
    for hostile in ("../train.csv", "../../etc/passwd", "/train.csv", "dataset/../secret.csv"):
        with pytest.raises(ValueError, match="unsafe archive member"):
            validator._normalize_member(hostile)


def test_target_in_drop_columns_is_rejected(tmp_path, monkeypatch):
    frame = pd.DataFrame({"x": range(60), "target": ["a", "b"] * 30})
    frame.to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, _ = validator.build_checks(source, {"target_column": "target", "drop_columns": "target,x"})
    finally:
        source.close()
    check = next(c for c in checks if c["name"] == "target_not_dropped")
    assert check["successful"] is False


def test_val_split_target_semantics_are_validated(tmp_path, monkeypatch):
    # train labels a,b; val has too few usable rows and an unseen label c
    pd.DataFrame({"x": range(60), "target": ["a", "b"] * 30}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"x": range(3), "target": ["a", "b", "c"]}).to_csv(tmp_path / "val.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, _ = validator.build_checks(source, {})
    finally:
        source.close()
    by = {c["name"]: c["successful"] for c in checks}
    assert by["val_has_usable_rows"] is False  # 3 rows < MIN_EVAL_ROWS
    assert by["val_labels_subset_of_train"] is False  # 'c' unseen in train


def test_good_val_split_passes_depth_checks(tmp_path, monkeypatch):
    pd.DataFrame({"x": range(60), "target": ["a", "b"] * 30}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"x": range(20), "target": ["a", "b"] * 10}).to_csv(tmp_path / "val.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, _ = validator.build_checks(source, {})
    finally:
        source.close()
    assert all(c["successful"] for c in checks)


def test_high_class_count_flagged_infeasible(tmp_path, monkeypatch):
    # 400 classes, 2 rows each; passes stratifiable_classes but the finetuner's
    # stratified cap of 300 cannot represent every class -> must be flagged.
    labels = [f"c{i}" for i in range(400)] * 2
    frame = pd.DataFrame({"x": range(len(labels)), "target": labels})
    frame.to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, _ = validator.build_checks(source, {"max_train_rows": 300})
    finally:
        source.close()
    stratifiable = next(c for c in checks if c["name"] == "stratifiable_classes")
    feasible = next(c for c in checks if c["name"] == "stratified_split_feasible")
    assert stratifiable["successful"] is True  # old shallow check still passes
    assert feasible["successful"] is False  # new feasibility check catches it
