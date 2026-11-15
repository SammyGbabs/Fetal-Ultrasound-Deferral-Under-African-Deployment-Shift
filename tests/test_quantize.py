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


def test_mixed_precision(tmp_path):
    """Activation ranges, and nodes kept in float by FP16 and INT8 mixed precision."""
    import onnx
    from src import quantize as q
    from src.models import build_model
    torch.manual_seed(0)
    model = build_model("effb0", 4, 0.2, pretrained=False).eval()
    fp32 = tmp_path / "fp32.onnx"
    q.export_onnx(model, fp32)
    rng = np.random.default_rng(0)
    images = rng.standard_normal((4, 3, 224, 224)).astype(np.float32)

    ranges = q.tensor_ranges(fp32, [images[:2], images[2:]])
    logits, _ = q.run_session(q.session(fp32), images)
    assert ranges["logits"]["max_abs"] == pytest.approx(float(np.abs(logits).max()), rel=1e-5)
    biggest = sorted(ranges, key=lambda t: ranges[t]["max_abs"])[-3:]
    kept = q.nodes_touching(fp32, set(biggest))
    assert kept and all(kept)

    fp16 = tmp_path / "fp16.onnx"
    q.convert_fp16(fp32, fp16, keep_fp32=kept)
    graph = onnx.load(str(fp16)).graph
    assert {n.name for n in graph.node} >= set(kept)
    assert np.isfinite(q.run_session(q.session(fp16), images)[0]).all()

    prepared, int8 = tmp_path / "prepared.onnx", tmp_path / "int8.onnx"
    q.prepare_for_quantization(fp32, prepared)
    kept8 = q.nodes_touching(prepared, set(biggest) & set(q.tensor_ranges(prepared, [images])))
    q.quantize_int8(prepared, int8, [images[:2], images[2:]], "minmax", keep_float=kept8)
    quantized = {n.input[0] for n in onnx.load(str(int8)).graph.node if n.op_type == "QuantizeLinear"}
    assert kept8 and not any(t in quantized for t in biggest if t not in ("logits", "features"))


def test_fp16_tracing(tmp_path):
    """first_nonfinite finds where FP16 overflows; kept FP32 nodes then stay FP32 end to end."""
    import onnx
    from src import quantize as q
    from src.models import build_model
    torch.manual_seed(0)
    model = build_model("effb0", 4, 0.2, pretrained=False).eval()
    fp32 = tmp_path / "fp32.onnx"
    q.export_onnx(model, fp32)
    images = np.random.default_rng(0).standard_normal((2, 3, 224, 224)).astype(np.float32) * 1e5

    fp16 = tmp_path / "fp16.onnx"
    q.convert_fp16(fp32, fp16)
    first = q.first_nonfinite(fp16, images)
    assert first is not None
    graph = onnx.load(str(fp16)).graph
    producer = next(n for n in graph.node if first in n.output)
    assert q.first_nonfinite(fp32, images) is None

    # keep the overflowing node and the nodes reading it: no FP16 round trip between them
    names = {n.name for n in onnx.load(str(fp32)).graph.node}
    kept = [producer.name] + [n.name for n in graph.node if first in n.input and n.name in names]
    mixed = tmp_path / "mixed.onnx"
    q.convert_fp16(fp32, mixed, keep_fp32=kept)
    g = onnx.load(str(mixed)).graph
    casts = {n.output[0]: n for n in g.node if n.op_type == "Cast"}
    to = lambda n: next(a.i for a in n.attribute if a.name == "to")  # noqa: E731
    assert not any(to(n) == onnx.TensorProto.FLOAT and n.input[0] in casts
                   and to(casts[n.input[0]]) == onnx.TensorProto.FLOAT16 for n in casts.values())
    assert q.first_nonfinite(mixed, images) != first
