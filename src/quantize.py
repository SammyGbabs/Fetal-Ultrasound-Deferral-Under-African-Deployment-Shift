"""ONNX export and ONNX Runtime quantization of trained plane classifiers (MIDL).

Shared by scripts/quantize.py (export, quantize, run the test conditions) and
scripts/time_models.py (deployment numbers). Everything runs on the CPU.

The ONNX graph starts after preprocessing: its input "image" is the N x 3 x 224 x 224
tensor that src/data.py builds (resize to 224 x 224, grey copied to 3 channels, the
model's own mean and std), so every precision sees exactly the tensors the PyTorch
models saw. Outputs are "logits" and "features", defined as in PlaneClassifier.predict:
features are the pooled backbone output, logits the linear head on them, dropout off.
"""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

OPSET = 17               # LayerNormalization is a single op from opset 17
INPUT, OUTPUTS = "image", ["logits", "features"]


class ExportWrapper(nn.Module):
    """PlaneClassifier without dropout, returning (logits, features) as in predict()."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # predict() adds .float() for mixed precision; every exported model is already FP32,
        # and a no-op Cast in the graph breaks the FP16 conversion, so it is left out here
        features = self.model.features(x)
        return self.model.head(features), features


def export_onnx(model: nn.Module, path: Path, image_size: int = 224) -> str:
    """Export to ONNX with a dynamic batch axis, as one self-contained file.

    Uses the TorchScript exporter, falling back to the dynamo exporter if it fails.
    Returns which exporter was used.
    """
    wrapper = ExportWrapper(model).eval()
    example = torch.zeros(2, 3, image_size, image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    # timm's fused attention exports its 1/sqrt(head_dim) scale as a run-time Shape -> Cast ->
    # Sqrt chain, which the FP16 converter mistypes. The unfused path multiplies by the same
    # constant and gives the same result in eval mode.
    fused = {m: m.fused_attn for m in model.modules() if hasattr(m, "fused_attn")}
    for m in fused:
        m.fused_attn = False
    try:
        torch.onnx.export(wrapper, (example,), str(path), input_names=[INPUT], output_names=OUTPUTS,
                          dynamic_axes={name: {0: "batch"} for name in [INPUT, *OUTPUTS]},
                          opset_version=OPSET, dynamo=False)
        exporter = "torchscript"
    except Exception as error:                                   # noqa: BLE001
        print(f"  TorchScript export failed ({type(error).__name__}: {error}); trying dynamo")
        torch.onnx.export(wrapper, (example,), str(path), input_names=[INPUT], output_names=OUTPUTS,
                          dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=4096)}},
                          opset_version=OPSET, dynamo=True)
        exporter = "dynamo"
    finally:
        for m, value in fused.items():
            m.fused_attn = value
    # Store weights inside the .onnx (the dynamo exporter may write a separate .data file)
    graph = identity_outputs(onnx.load(str(path)))
    onnx.save_model(graph, str(path), save_as_external_data=False)
    for extra in path.parent.glob(path.name + ".data"):
        extra.unlink()
    onnx.checker.check_model(str(path))
    return exporter


def identity_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """End each graph output with its own Identity node.

    "features" is both a graph output and the head's input; the FP16 converter mistypes
    a tensor that is both, so the output becomes a separate copy.
    """
    graph = model.graph
    for output in graph.output:
        inner = f"{output.name}_inner"
        for node in graph.node:
            node.output[:] = [inner if o == output.name else o for o in node.output]
            node.input[:] = [inner if i == output.name else i for i in node.input]
        graph.node.append(onnx.helper.make_node("Identity", [inner], [output.name],
                                                name=f"output_{output.name}"))
    return model


def session(path: Path, threads: int | None = None) -> ort.InferenceSession:
    """CPU session. threads=None lets ONNX Runtime use every core."""
    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_session(sess: ort.InferenceSession, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits, features = sess.run(OUTPUTS, {INPUT: images.astype(np.float32)})
    return logits.astype(np.float32), features.astype(np.float32)


def convert_fp16(src: Path, dst: Path) -> None:
    """FP16 copy of an FP32 model; input and outputs stay float32."""
    from onnxconverter_common import float16
    model = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(dst))
    onnx.checker.check_model(str(dst))


class CalibrationReader:
    """Feeds preprocessed calibration batches to ONNX Runtime's static quantizer.

    __len__ and set_range let ONNX Runtime calibrate one stride of batches at a time
    ("CalibStridedMinMax") and merge the results, so only one batch of intermediate
    activations is in memory at once. Without it, Percentile calibration keeps every
    activation of all 500 images (tens of GB).
    """

    def __init__(self, batches: list[np.ndarray]):
        self.batches = batches
        self.set_range(0, len(batches))

    def __len__(self) -> int:
        return len(self.batches)

    def set_range(self, start_index: int, end_index: int) -> None:
        self.iterator = iter([{INPUT: b.astype(np.float32)} for b in self.batches[start_index:end_index]])

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.set_range(0, len(self.batches))


CALIB_BATCH = 10         # calibration images per batch (and per stride)

# Settings shared by every INT8 model; written to each meta.json
INT8_SETTINGS = {
    "format": "QDQ",
    "per_channel": True,
    "activation_type": "QUInt8",
    "weight_type": "QInt8",
    "reduce_range": False,
    "percentile": 99.999,             # ONNX Runtime's default for the Percentile method
    "preprocessing": "onnxruntime.quantization.quant_pre_process",
}


def prepare_for_quantization(src: Path, dst: Path) -> None:
    """Shape inference and graph optimisation, as ONNX Runtime recommends before quantizing."""
    from onnxruntime.quantization.shape_inference import quant_pre_process
    try:
        quant_pre_process(str(src), str(dst))
    except Exception as error:                                   # noqa: BLE001
        print(f"  pre-processing with symbolic shapes failed ({type(error).__name__}); retrying without")
        quant_pre_process(str(src), str(dst), skip_symbolic_shape=True)


def quantize_int8(src: Path, dst: Path, batches: list[np.ndarray], method: str) -> None:
    """Static post-training INT8 quantization, QDQ format, per-channel weights."""
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    methods = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}
    # Calibrate one batch at a time and merge (see CalibrationReader). Not
    # CalibMaxIntermediateOutputs: in onnxruntime 1.31 MinMax discards each chunk unused.
    extra = {"CalibStridedMinMax": 1}
    if method == "percentile":
        extra["CalibPercentile"] = INT8_SETTINGS["percentile"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(str(src), str(dst), CalibrationReader(batches),
                    quant_format=QuantFormat.QDQ, per_channel=INT8_SETTINGS["per_channel"],
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    reduce_range=INT8_SETTINGS["reduce_range"], calibrate_method=methods[method],
                    extra_options=extra)
    onnx.checker.check_model(str(dst))


def versions() -> dict:
    import onnxconverter_common
    return {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
            "onnxconverter_common": onnxconverter_common.__version__}


def cpu_model() -> str:
    """CPU name as the OS reports it (Linux /proc/cpuinfo, else platform)."""
    import platform
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
