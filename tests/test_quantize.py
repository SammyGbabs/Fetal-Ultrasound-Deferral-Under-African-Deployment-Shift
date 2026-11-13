"""Checks for src/quantize.py and the calibration list (scripts/quantize.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxconverter_common")
torch = pytest.importorskip("torch")


def test_calibration_list():
    from scripts.quantize import CALIBRATION_FILE, calibration_images
    calib = calibration_images()
    assert CALIBRATION_FILE.exists() and len(calib) == 500
    assert calib["label"].value_counts().tolist() == [125] * 4          # stratified by view
    train = pd.read_csv(ROOT / "splits" / "spain_train.csv", keep_default_na=False)
    assert set(calib["image_id"]) <= set(train["image_id"])             # training images only


def test_export_fp16_int8(tmp_path):
    from src import quantize as q
    from src.models import build_model
    torch.manual_seed(0)
    model = build_model("effb0", 4, 0.2, pretrained=False).eval()
    fp32 = tmp_path / "fp32.onnx"
    q.export_onnx(model, fp32)

    rng = np.random.default_rng(0)
    images = rng.standard_normal((3, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        ref = model.predict(torch.from_numpy(images), mc_passes=2)
    logits, features = q.run_session(q.session(fp32), images)
    assert np.abs(logits - ref["logits"].numpy()).max() < 1e-4             # same as predict()
    assert np.abs(features - ref["features"].numpy()).max() < 1e-4

    fp16 = tmp_path / "fp16.onnx"
    q.convert_fp16(fp32, fp16)
    logits16, features16 = q.run_session(q.session(fp16), images)
    assert logits16.dtype == np.float32 and logits16.shape == (3, 4)
    assert features16.shape == features.shape

    prepared, int8 = tmp_path / "prepared.onnx", tmp_path / "int8.onnx"
    q.prepare_for_quantization(fp32, prepared)
    q.quantize_int8(prepared, int8, [images[:2], images[2:]], "minmax")
    logits8, features8 = q.run_session(q.session(int8, threads=1), images)
    assert logits8.shape == (3, 4) and features8.shape == features.shape
    assert int8.stat().st_size < fp32.stat().st_size / 2                    # weights stored as 8 bits
