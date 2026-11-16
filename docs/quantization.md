# Quantization (FP16 and INT8 on the CPU)

For MIDL we test whether the models survive the compression a low-resource deployment
would use. Each FP32 checkpoint is exported to ONNX and converted to FP16 and to INT8
with ONNX Runtime, and the quantized versions are run on the same conditions as the
originals. No retraining.

    !python scripts/quantize.py --models effb0 mnv4 convnext vits16
    !python scripts/quantize.py --variants fp16mixed int8mixed
    !python scripts/time_models.py

## Method

**Export.** `src/quantize.py` exports each checkpoint with the TorchScript exporter,
opset 17, dynamic batch. The ONNX input is the preprocessed tensor that `src/data.py`
builds (224 x 224, grey copied to 3 channels, each model's own mean and std, stored in
meta.json), so preprocessing is the same code for every precision. The two outputs are
the same as in `infer.py`: `features` is the pooled backbone output and `logits` the
linear head on it, dropout off.

Two export details keep the graph convertible to FP16, and both give the same numbers:

- timm's fused attention is switched off during export. It writes the 1/sqrt(head_dim)
  scale as a run-time Shape, Cast, Sqrt chain that the FP16 converter mistypes; the
  unfused path multiplies by the same constant.
- Each output ends in its own Identity node, because `features` is also the head's
  input and the converter mistypes a tensor that is both.

**Export check.** Before anything is quantized, the FP32 ONNX model is compared with
PyTorch on the CPU over all 555 spain_val images (`runs/<run>/onnx_check.json`). It
passes if the largest logit difference divided by the largest |logit| is at most 1e-4
and all 555 predictions are the same; otherwise the run is not quantized. The check is
relative because some checkpoints have very large logits: mnv4_s1 reaches 1145, so its
absolute difference of 0.07 is only 6e-5 of the logit, with every prediction the same.

**FP16.** `onnxconverter_common.float16.convert_float_to_float16` with float32 input
and outputs.

**INT8.** Static post-training quantization with ONNX Runtime, QDQ format, after ONNX
Runtime's `quant_pre_process`, with these settings (also in each meta.json under
`quantization`):

- Weights: signed 8-bit (QInt8), symmetric (zero point 0), per-channel along the output
  channel (Conv axis 0, MatMul axis 1, Gemm axis 0). 1-D weights such as LayerNorm
  scales can't be split by channel and are quantized per-tensor.
- Activations: unsigned 8-bit (QUInt8), asymmetric (scale and zero point from the
  calibrated range), per-tensor.
- Bias: 32-bit integer with scale = input scale x weight scale.
- No reduced range; every op type ONNX Runtime quantizes in QDQ format.

Unsigned activations with signed weights (U8S8) is the usual choice for x86 CPUs.
Symmetry is passed to ONNX Runtime explicitly, with the same values as its defaults.

- Calibration uses 500 spain_train images, 125 per view, drawn with seed 0
  (`splits/quant_calibration.csv`, 306 patients). The same images are used for every model.
- MinMax and Percentile (99.999) calibration are both run, and the one with the higher
  spain_val accuracy is kept per run, MinMax on a tie. Accuracy is computed on spain_val
  without the 14 images leaked from training (`splits/duplicates.csv`), as in the
  evaluation.
- Calibration runs one batch of 10 images at a time and merges the ranges (ONNX Runtime's
  `CalibStridedMinMax`); otherwise Percentile keeps every activation of all 500 images in
  memory (over 30 GB for EfficientNet-B0). On a small check this gives identical MinMax
  ranges and Percentile scales within 0.07% of one-shot calibration. ONNX Runtime's other
  chunking option, `CalibMaxIntermediateOutputs`, isn't used: in version 1.31 MinMax
  discards each chunk without using it.

**Activation ranges.** For every FP32 run, the largest |value| of every tensor inside
the model on the 500 calibration images is saved in `runs/<run>/activation_ranges.json`,
with one row per run in `outputs/quantization/activations.csv`: the median over tensors,
the largest, where it is, and how many tensors are outliers or above the FP16 maximum.

**Mixed precision.** Plain FP16 and plain static INT8 are the main result, and their
failures are reported as they are. Two extra variants keep the problem layers in float,
each by a rule fixed on Spanish data and written to meta.json (`mixed_precision`):

- `<model>-fp16mixed_s<seed>`: if plain FP16 gives non-finite outputs on spain_train or
  spain_val, or any tensor of the FP32 model goes above the FP16 maximum on the
  calibration images (an overflow that the later layers may hide), every node that reads or writes a tensor whose FP32 value on spain_train or
  spain_val goes above a limit stays in FP32. The limit starts at the FP16 maximum
  (65504) and is halved (32752, 16376, ... down to 65504/32) until no spain_train or
  spain_val output is non-finite, since FP16 rounding can push a value that is just
  under the maximum over it. Each limit tried is listed in meta.json. The converter
  sends values between two FP32 nodes through FP16 and back, which would bring the
  overflow back, so those cast pairs are removed after converting. When the node writing
  a model output is kept in FP32, the converter also casts that output to FP16 and
  declares it FP16, so its last cast is set back to float32. If the rule doesn't trigger,
  the variant is the plain FP16 model.
- `<model>-int8mixed_s<seed>`: every node that reads or writes an outlier tensor stays
  in float, an outlier being a tensor whose largest |value| on the calibration images
  is more than 100 times the median over all the model's tensors. Calibration is chosen
  as for plain INT8. With no outlier tensors the variant is the plain INT8 model.

Each variant exists for all five seeds, so it can be evaluated as an ensemble like the
others. Where a rule keeps nothing in float, meta.json says `same_as` the plain run.

**African data is never used for a choice.** The export threshold, the calibration
images, the calibration method and both mixed-precision rules all come from Spanish
training or validation data.
The script doesn't print African accuracies.

**Sanity check.** Each quantized model's spain_val accuracy is compared with its FP32
original, and a drop of more than 1 point is flagged (`flagged` in
`outputs/quantization/checks.csv` and in meta.json).

## Outputs

- `outputs/<model>-<precision>_s<seed>/` for fp16, int8, fp16mixed and int8mixed:
  contract D for spain_train, spain_val, spain_test, africa_test and africa_test_q10, in
  the manifest. **Some FP16 outputs have rows with NaN logits** (see Results); the
  evaluation has to count them, not drop them silently.
  **`mc_logits` holds the plain logits repeated 20 times: the ONNX models have no
  dropout, so MC-dropout results are not valid for quantized runs.**
- `runs/<run>/model_fp32.onnx` and `onnx_check.json` next to each original checkpoint.
- `runs/<model>-<precision>_s<seed>/model.onnx` and `meta.json` (settings, calibration
  method and both candidates' spain_val accuracy, FP32 accuracy, drop, versions, CPU).
- `outputs/quantization/checks.csv`: one row per quantized run.
- `outputs/quantization/failed/`: models that couldn't be converted, with the error.
- `outputs/quantization/activations.csv`: activation ranges per FP32 run (above).
- `outputs/quantization/int8_saturation.csv`: the effb0 saturation check (above), with
  the CPU model and its VNNI flags.
- `outputs/quantization/deployment.csv`: file size and CPU time per image for FP32 and
  every quantized variant. Batch size 1, one thread, 20 warm-up runs, then the median
  over 200 spain_val images. The time covers the model only, not image loading and
  resizing. It also counts the saved output rows with non-finite logits per condition,
  which is where plain FP16 overflow shows up.

Versions are pinned in requirements.txt (onnx 1.23.2, onnxruntime 1.31.1,
onnxconverter-common 1.16.0).

## Limits

- **FP16 is a size saving, not a speed-up, on most CPUs.** ONNX Runtime has few FP16
  kernels for x86, so FP16 halves the file but may run no faster than FP32 (on a test
  Apple M1, EfficientNet-B0 took 34 ms per image in both, and 14 ms in INT8).
- **DINOv3 doesn't convert to FP16 yet.** Its rotary position embedding has casts the
  FP16 converter mistypes, the same converter problem as above. INT8 works. It's low
  priority and recorded in `failed/` if attempted.
- **INT8 times depend on the CPU's integer instructions** (VNNI on recent x86, dot-product
  instructions on ARM), so deployment.csv records the CPU model.

## Results

Colab CPU run, all four models x 5 seeds. Accuracies are on spain_val without the 14
leaked images; a drop of more than 1 point against FP32 is flagged.

**Export.** All 20 FP32 ONNX models pass the export check, the largest relative logit
difference being 6.3e-5 (mnv4_s1) and every prediction the same.

**Plain FP16** matches FP32 on spain_val for every run (the largest drop is 0.2 points,
mnv4_s0), so nothing is flagged. It halves the file but is no faster on this CPU.
Two runs give NaN logits, both MobileNetV4:

- mnv4-fp16_s4: 51 rows (1 spain_train, 25 africa_test, 25 africa_test_q10). On that
  spain_train image the FP32 logits reach 76,425, above the FP16 maximum, so FP16 can't
  represent the output at all.
- mnv4-fp16_s1: 2 rows (1 spain_test, 1 africa_test_q10).

**Plain INT8** passes on 7 of 20 runs.

| Model | spain_val, INT8 / FP32 per seed (s0 to s4) | Flagged |
|---|---|---|
| effb0 | 0.861/0.998, 0.878/1.000, 0.893/0.996, 0.867/0.998, 0.791/1.000 | 5 of 5 |
| mnv4 | 0.471/0.996, 0.111/0.994, 0.470/0.994, 0.989/0.998, 0.471/0.996 | 4 of 5 |
| convnext | 0.991/0.996, 0.948/0.996, 0.989/0.996, 0.996/0.998, 0.667/0.998 | 2 of 5 (s1, s4) |
| vits16 | 0.974/0.987, 0.998/0.998, 0.994/0.996, 0.994/0.998, 0.980/0.994 | 2 of 5 (s0, s4) |

**Activation outliers explain some of this, not all.** The largest activation over the
median, on the calibration images:

| Model | Largest / median | Outlier tensors (over 100x) |
|---|---|---|
| mnv4 s0, s1, s2, s4 | 499x, 1943x, 258x, 384x | 9, 19, 7, 8 |
| mnv4 s3 | 10x | 0 |
| convnext, all seeds | 275x to 343x | 5 each |
| effb0, all seeds | 43x to 48x | 0 |
| vits16, all seeds | about 40x | 0 |

The 100x threshold falls in the gap between 48x and 258x. MobileNetV4's broken seeds
are exactly its outlier seeds, and s3, the one without outliers, is the one that works.
effb0 breaks on every seed with small activations, so its failure has another cause
(below); vits16's two flagged seeds lose 1.3 and 1.5 points, also without outliers.

**effb0 is not saturation.** Without VNNI, ONNX Runtime's x86 kernels for uint8
activations and int8 weights add pairs of products in 16 bits, which can saturate.
`scripts/check_int8_saturation.py` quantizes each effb0 seed three ways (plain,
reduce_range, uint8 weights), each with MinMax and Percentile, and scores it on
spain_val twice: as normal, and with graph optimisation off, so the QDQ nodes run as
DequantizeLinear and float ops with no integer kernels. The gap between the two is
what the kernels cost. On a Colab Intel Xeon (2.20 GHz, AVX2, no VNNI), mean spain_val
accuracy over the five seeds (FP32 0.996 to 1.000):

| Setting | MinMax | Percentile | Normal minus no integer kernels (Percentile) |
|---|---|---|---|
| plain (the settings used) | 0.734 | 0.858 | -1.5 to -2.4 points |
| reduce_range (7-bit weights) | 0.780 | 0.884 | -0.6 to +1.3 |
| uint8 weights | 0.736 | 0.891 | -0.7 to +1.1 |

Saturation is there, about 2 points with the plain settings, and both other settings
remove it. Most of the drop is quantization error: without the integer kernels the
plain Percentile models still only reach 0.82 to 0.91. The best result on any seed is 0.941
(s0, reduce_range with Percentile); every seed stays 5.7 to 15.3 points below FP32
whatever the setting, and no setting is best on every seed. Percentile beats MinMax
by 4 to 22 points throughout. Plain INT8 stays the main result. The plain Percentile
accuracies here are the same as in the table above, so the check reproduces the
original models. Times in this check come from a different, shared machine and aren't
comparable with deployment.csv.

**Mixed INT8** (outlier layers in float) passes on 10 of 20 runs:

- convnext: fixed on all seeds, equal to FP32 on s0, s2 and s4 and within 0.6 points on
  s1 and s3 (s1 0.948 to 0.991, s4 0.667 to 0.998). Same size and speed as plain INT8.
- mnv4: fixed on s1 only (0.111 to 0.987). s0, s2 and s4 stay broken (0.471, 0.555,
  0.471) with 10 to 13 nodes in float, and the files grow from 8.7 to 12 to 17 MB.
- effb0, vits16 and mnv4_s3: no outliers, so the same model as plain INT8.

**Mixed FP16** triggers on mnv4_s4 (NaN on spain_train) and mnv4_s1 (7 tensors above
65504 on the calibration images), keeping 18 and 11 nodes in FP32; every other run is
the plain FP16 model. Both keep FP32 accuracy on spain_val. mnv4-fp16mixed_s4 has no NaN
rows left on any condition. mnv4-fp16mixed_s1 still has the 2 NaN rows of the plain
model (spain_test and africa_test_q10): those images overflow somewhere the Spanish
training images don't reach, and the rule isn't tuned on test or African data.

**Size and CPU time** (AMD EPYC 7B12, one thread, median over seeds):

| Model | FP32 | FP16 | INT8 |
|---|---|---|---|
| effb0 | 15.3 MB, 16.7 ms | 7.7 MB, 16.4 ms | 4.6 MB, 17.7 ms |
| mnv4 | 32.1 MB, 20.9 ms | 16.1 MB, 21.1 ms | 8.7 MB, 13.6 ms |
| convnext | 106.2 MB, 126.4 ms | 53.2 MB, 136.9 ms | 27.3 MB, 87.2 ms |
| vits16 | 82.8 MB, 120.9 ms | 41.5 MB, 123.2 ms | 21.4 MB, 81.7 ms |

INT8 is about 4x smaller and 31 to 35% faster for mnv4, convnext and vits16; effb0
gets no speed-up. One timing, effb0_s0 FP32 at 25.7 ms, is a warm-up outlier against
16.4 to 16.8 ms for the other seeds; the table uses medians over seeds.
