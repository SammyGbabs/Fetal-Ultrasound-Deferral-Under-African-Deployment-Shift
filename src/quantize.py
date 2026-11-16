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


def session(path: Path, threads: int | None = None, fuse: bool = True) -> ort.InferenceSession:
    """CPU session. threads=None lets ONNX Runtime use every core.

    fuse=False turns off graph optimisation, so an INT8 QDQ model runs as
    DequantizeLinear + float ops (quantization error only, no integer kernels).
    """
    options = ort.SessionOptions()
    if not fuse:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_session(sess: ort.InferenceSession, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits, features = sess.run(OUTPUTS, {INPUT: images.astype(np.float32)})
    return logits.astype(np.float32), features.astype(np.float32)


def convert_fp16(src: Path, dst: Path, keep_fp32: list[str] | None = None) -> None:
    """FP16 copy of an FP32 model; input and outputs stay float32.

    keep_fp32: names of nodes left in FP32 (the converter adds casts around them).
    """
    from onnxconverter_common import float16
    model = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True,
                                             node_block_list=keep_fp32 or None)
    if keep_fp32:
        float32_outputs(model)
        remove_cast_round_trips(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(dst))
    onnx.checker.check_model(str(dst))


def float32_outputs(model: onnx.ModelProto) -> None:
    """Make every graph output float32 again.

    When the node writing a graph output is kept in FP32, the converter casts that output
    to FP16 and declares it FP16, despite keep_io_types, so a logit above 65504 becomes
    infinite on the way out. The final cast is pointed at float32 instead.
    """
    producers = {o: n for n in model.graph.node for o in n.output}
    for out in model.graph.output:
        if out.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16:
            continue
        node = producers[out.name]
        if node.op_type != "Cast":
            raise ValueError(f"FP16 graph output {out.name} is not written by a Cast")
        for a in node.attribute:
            if a.name == "to":
                a.i = onnx.TensorProto.FLOAT
        out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT


def remove_cast_round_trips(model: onnx.ModelProto) -> int:
    """Drop FP32 -> FP16 -> FP32 cast pairs between two nodes kept in FP32.

    The converter casts a kept node's output to FP16 and back before the next kept node
    reads it, so a value above 65504 still overflows on the way. The FP32 readers are
    pointed at the original FP32 tensor instead. Returns the number of casts removed.
    """
    graph = model.graph
    to = {n.output[0]: n for n in graph.node if n.op_type == "Cast"}
    outputs = {o.name for o in graph.output}

    def cast_to(node, dtype):
        return node.op_type == "Cast" and next(a.i for a in node.attribute if a.name == "to") == dtype

    rename, drop = {}, []
    for node in graph.node:
        first = to.get(node.input[0]) if node.input else None
        if (cast_to(node, onnx.TensorProto.FLOAT) and first is not None
                and cast_to(first, onnx.TensorProto.FLOAT16) and node.output[0] not in outputs):
            rename[node.output[0]] = first.input[0]
            drop.append(node)
    for node in drop:
        graph.node.remove(node)
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in rename:
                node.input[i] = rename[name]
    used = {i for n in graph.node for i in n.input} | outputs
    unused = [n for n in graph.node if n.op_type == "Cast" and n.output[0] not in used]
    for node in unused:
        graph.node.remove(node)
    return len(drop) + len(unused)


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

# Settings shared by every INT8 model; written to each meta.json. Symmetry is passed to
# ONNX Runtime explicitly (these are also its defaults for QUInt8 activations and QInt8
# weights), so what is recorded is what was used.
INT8_SETTINGS = {
    "format": "QDQ",
    "weights": "signed 8-bit (QInt8), symmetric (zero point 0), per-channel along the output "
               "channel (Conv axis 0, MatMul axis 1, Gemm axis 0 with transB); 1-D weights such as "
               "LayerNorm scales fall back to per-tensor",
    "activations": "unsigned 8-bit (QUInt8), asymmetric (scale and zero point from the calibrated "
                   "range), per-tensor",
    "bias": "32-bit integer, scale = input scale x weight scale",
    "per_channel": True,
    "activation_type": "QUInt8",
    "weight_type": "QInt8",
    "activation_symmetric": False,
    "weight_symmetric": True,
    "reduce_range": False,
    "percentile": 99.999,             # ONNX Runtime's default for the Percentile method
    "op_types": "every op type ONNX Runtime quantizes in QDQ format (default list)",
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
    # Every node needs a name, so mixed precision can list the nodes it keeps in float
    model = onnx.load(str(dst))
    if any(not n.name for n in model.graph.node):
        for i, n in enumerate(model.graph.node):
            n.name = n.name or f"{n.op_type}_{i}"
        onnx.save_model(model, str(dst))


def quantize_int8(src: Path, dst: Path, batches: list[np.ndarray], method: str,
                  keep_float: list[str] | None = None, settings: dict | None = None) -> None:
    """Static post-training INT8 quantization, QDQ format, per-channel weights.

    keep_float: names of nodes left unquantized (mixed precision).
    settings: INT8_SETTINGS with some values changed (weight_type, weight_symmetric,
    reduce_range); INT8_SETTINGS itself when None.
    """
    settings = {**INT8_SETTINGS, **(settings or {})}
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    methods = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}
    # Calibrate one batch at a time and merge (see CalibrationReader). Not
    # CalibMaxIntermediateOutputs: in onnxruntime 1.31 MinMax discards each chunk unused.
    extra = {"CalibStridedMinMax": 1, "ActivationSymmetric": settings["activation_symmetric"],
             "WeightSymmetric": settings["weight_symmetric"]}
    if method == "percentile":
        extra["CalibPercentile"] = settings["percentile"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(str(src), str(dst), CalibrationReader(batches),
                    quant_format=QuantFormat.QDQ, per_channel=settings["per_channel"],
                    activation_type=QuantType[settings["activation_type"]],
                    weight_type=QuantType[settings["weight_type"]],
                    reduce_range=settings["reduce_range"], calibrate_method=methods[method],
                    nodes_to_exclude=keep_float or None, extra_options=extra)
    onnx.checker.check_model(str(dst))


def tensor_ranges(path: Path, batches) -> dict[str, dict]:
    """Largest |value| of every float tensor computed inside the model, over the given batches.

    Every float node output becomes an extra graph output of a temporary copy, so this
    is the FP32 graph's own activations. Returns {tensor: {"max_abs", "nonfinite"}}.
    """
    import tempfile
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    floats = {v.name for v in [*model.graph.value_info, *model.graph.output]
              if v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT}
    existing = {o.name for o in model.graph.output}
    names = [o for node in model.graph.node for o in node.output if o in floats]
    for name in names:
        if name not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.onnx"
        onnx.save_model(model, str(probe))
        sess = ort.InferenceSession(str(probe), providers=["CPUExecutionProvider"])
        ranges = {n: {"max_abs": 0.0, "nonfinite": False} for n in names}
        for images in batches:
            values = sess.run(names, {INPUT: images.astype(np.float32)})
            for name, v in zip(names, values):
                finite = np.isfinite(v)
                r = ranges[name]
                r["nonfinite"] |= bool(not finite.all())
                if finite.any():
                    r["max_abs"] = max(r["max_abs"], float(np.abs(v[finite]).max()))
    return ranges


def nodes_touching(path: Path, tensors: set[str]) -> list[str]:
    """Names of the nodes that read or write any of the given tensors, in graph order."""
    graph = onnx.load(str(path)).graph
    return [n.name for n in graph.node if tensors & {*n.input, *n.output}]


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


def cpu_int8_flags() -> dict:
    """Instruction sets that matter for INT8 kernels, from /proc/cpuinfo (Linux only).

    Without VNNI, ONNX Runtime's x86 U8S8 kernels add pairs of uint8 x int8 products
    in int16 (vpmaddubsw), which can saturate; reduce_range or U8U8 avoids it.
    """
    cpuinfo = Path("/proc/cpuinfo")
    flags = set()
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
                break
    names = ["avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8"]
    found = {f: f in flags for f in names}
    found["vnni"] = found["avx512_vnni"] or found["avx_vnni"]
    found["flags_read"] = bool(flags)
    return found


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
