# Quantization (FP16 and INT8 on the CPU)

For MIDL we test whether the models survive the compression a low-resource deployment
would use. Each FP32 checkpoint is exported to ONNX and converted to FP16 and to INT8
with ONNX Runtime, and the quantized versions are run on the same conditions as the
originals. No retraining.

    !python scripts/quantize.py --models effb0 mnv4 convnext vits16
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
PyTorch on the CPU over all 555 spain_val images (`runs/<run>/onnx_check.json`). A run
whose logits differ by more than 1e-3 is not quantized. On test models the largest
differences were around 1e-6 for logits and 1e-5 for features.

**FP16.** `onnxconverter_common.float16.convert_float_to_float16` with float32 input
and outputs.

**INT8.** Static post-training quantization with ONNX Runtime: QDQ format, per-channel
weights, unsigned 8-bit activations and signed 8-bit weights (U8S8, the usual choice for
x86 CPUs), after ONNX Runtime's `quant_pre_process`.

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

**African data is never used for a choice.** The export threshold, the calibration
images and the calibration method all come from Spanish training or validation data.
The script doesn't print African accuracies.

**Sanity check.** Each quantized model's spain_val accuracy is compared with its FP32
original, and a drop of more than 1 point is flagged (`flagged` in
`outputs/quantization/checks.csv` and in meta.json).

## Outputs

- `outputs/<model>-fp16_s<seed>/` and `outputs/<model>-int8_s<seed>/`: contract D for
  spain_train, spain_val, spain_test, africa_test and africa_test_q10, in the manifest.
  **`mc_logits` holds the plain logits repeated 20 times: the ONNX models have no
  dropout, so MC-dropout results are not valid for quantized runs.**
- `runs/<run>/model_fp32.onnx` and `onnx_check.json` next to each original checkpoint.
- `runs/<model>-<precision>_s<seed>/model.onnx` and `meta.json` (settings, calibration
  method and both candidates' spain_val accuracy, FP32 accuracy, drop, versions, CPU).
- `outputs/quantization/checks.csv`: one row per quantized run.
- `outputs/quantization/failed/`: models that couldn't be converted, with the error.
- `outputs/quantization/deployment.csv`: file size and CPU time per image for FP32,
  FP16 and INT8. Batch size 1, one thread, 20 warm-up runs, then the median over 200
  spain_val images. The time covers the model only, not image loading and resizing.

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

To fill in after the Colab run: export check, spain_val accuracy against FP32 per model
and precision, calibration methods chosen, flagged models, file sizes and CPU times.
