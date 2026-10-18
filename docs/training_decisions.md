# Training decisions and pretrained weights

Record of where each model's recipe departs from the base recipe in the T4 ticket
(AdamW, lr 1e-4, weight decay 1e-4, batch 64, up to 30 epochs, early stopping on
Spanish validation balanced accuracy with patience 5), when each choice was made, and
what we had looked at when we made it. Times are UTC, from git commit times, Drive file
times and each run's `meta.json`.

## Short version

- No hyperparameter was chosen or changed after looking at African results for the
  model it applies to. Every departure was set either before that model's first run
  or, in one case, from Spanish validation curves.
- No learning rate or epoch setting was searched. Each configuration ran once per seed.
  The only reruns are frozen DINOv3 (below) and `fetalclip_s3`, which a second Colab
  session retrained with the same config by accident.
- Checkpoints are always selected on Spanish validation. The African set is test-only.

## Frozen models: lr 1e-3

Set in `f7f0f8f` (12 Oct 14:10), together with the frozen-backbone option, before any
DINOv3 or FetalCLIP run. Reason: with the backbone frozen only a 4-class linear head is
trained, and a single linear layer learns very slowly at 1e-4. A higher rate for linear
probing is standard practice. No results of any foundation model existed at that point.

## Frozen models: up to 100 epochs, patience 10

Set in `2b3f4d8` (15 Oct 09:34), merged as `fcda2a3` (09:59). Applies to frozen DINOv3
and, from `b5fd958`, frozen FetalCLIP.

What we looked at: the training logs of the first five frozen DINOv3 runs (15 Oct
08:45-09:03), that is Spanish training and validation only. Four of five reached their
best validation score at epoch 28-30 of 30 with validation loss still falling, and
`dinov3s_frozen_s4` was stopped at epoch 16 by a noisy plateau (0.901 against 0.931-0.939
for the others). The heads had not finished training.

What we had not looked at: any African result for a frozen model. Inference was never
run on the 30-epoch frozen runs; at 09:28 `outputs/` held only the fine-tuned DINOv3
runs. The 30-epoch frozen runs were then retrained with `--force` (17:14-17:58) and
exported for the first time at 18:33-18:39.

One thing in the open: fine-tuned DINOv3 African results (exported 09:12-09:28) were
visible before this change. The change was based on the frozen validation curves, not
on those results, and it does not touch any fine-tuned model.

Actual training length: frozen DINOv3 44-62 epochs, frozen FetalCLIP 14-17 epochs.

## Fine-tuned FetalCLIP: lr 1e-5, gradient checkpointing

Written when FetalCLIP was first added (morning of 15 Oct, before that day's DINOv3
runs), committed in `bf9d9da` (19:23) and merged as `b5fd958` (19:33). The first
FetalCLIP run started at 19:56; no FetalCLIP config has changed since.

Reason: FetalCLIP's image encoder is a ViT-L/14 with 304M parameters, about 14 times
ResNet-50. Fully fine-tuning a large pretrained ViT or CLIP encoder at 1e-4 tends to
overwrite the pretrained features, so 1e-5 is the usual choice. It was not tuned: no
other value was tried.

At that point African results existed for EfficientNet-B0, ResNet-50, MobileNetV4,
ConvNeXt-Tiny, ViT-S/16 and fine-tuned DINOv3, all trained at 1e-4. None of them
suggested a different rate for FetalCLIP, and the value follows from model size alone.

Gradient checkpointing only recomputes activations to save GPU memory, so the ViT-L fits
a T4 at batch 64. It does not change what is computed.

We learned only afterwards (16 Oct) that the 2026 foundation-model benchmark
(arXiv 2605.27796), which fine-tuned FetalCLIP on the same Spain-to-Africa setup, got
0.61 on Africa. Ours gets 0.967. A too-high learning rate is a likely reason for the gap.

## Input normalization

Each model uses the mean and std its pretrained weights expect, read from the weights'
own config (`normalization()` in `src/models.py`): ImageNet values for most models,
0.5/0.5 for ViT-S/16, and the OpenAI CLIP values for FetalCLIP, which match the
preprocessing open_clip builds for the FetalCLIP release. All models get the same resize
to 224x224 and the same augmentation.

## Hardware

| Runs | GPU |
| --- | --- |
| effb0, rn50, mnv4, convnext, vits16 | Tesla T4 |
| dinov3s, dinov3s_frozen | NVIDIA RTX PRO 6000 Blackwell |
| fetalclip, fetalclip_frozen | RTX PRO 6000 Blackwell, except `fetalclip_s3` on a Tesla T4 |

Each run's `meta.json` records its GPU.

## Pretrained weights

All weights are pinned to a Hugging Face commit in `WEIGHT_REVISIONS` in `src/models.py`,
so reruns load exactly these files. Each repo was last updated on the Hub before training
started (11 Oct), so these are the revisions every run downloaded. The SHA-256 is the
file's hash as listed by the Hub.

| Model | Weights | Revision | File | Hub last updated | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| effb0 | timm/efficientnet_b0.ra_in1k | 1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a | model.safetensors | 2025-01-21 | d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd |
| rn50 | timm/resnet50.tv_in1k | 78f3ecfdb38e06d9b8397f662e7ab8fee96026fa | model.safetensors | 2025-01-21 | 5d061a3c593d795bfe682d9b152bafbcf550579873492def3515b46db1189888 |
| mnv4 | timm/mobilenetv4_conv_medium.e500_r224_in1k | 02a09fbfb82b289e871ba8255f9da58c056fb13b | model.safetensors | 2025-01-21 | 10f9f8057d574f71d22c52d31cadab8af0bc325841fd93939c86d45d20998cad |
| convnext | timm/convnext_tiny.fb_in1k | b43a6303c9fcf176d2d707478a128c2c91e93528 | model.safetensors | 2025-01-21 | 08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276 |
| vits16 | timm/vit_small_patch16_224.augreg_in1k | d01bbafd7491e7caeb940a9ceda55eb8f0f99f53 | model.safetensors | 2025-01-21 | c20a91d93f4757b0f581519f21ad91a875cf041764873a7a5bccc8e36f360dbf |
| dinov3s | timm/vit_small_patch16_dinov3.lvd1689m | 3bf4720a82ec2066db88137180ff1f83a675cef0 | model.safetensors | 2025-09-24 | 2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040 |
| fetalclip | numansaeed/fetalclip-model | 2c02ad7b3eea314a328c5ebe5b81be4ef3a4d5db | FetalCLIP_weights.pt | 2025-05-01 | 90916caa408baf9374ee3f8c1533463331a4ace37a13e61385e782905b35c310 |

FetalCLIP's weights are CC BY-NC 4.0.

## Reproducibility check

On 18 Oct we reran inference for seed 0 of each of the 9 model variants on `spain_val`
with `scripts/check_reproducibility.py`, using the current code (pinned weights included)
and the library versions now pinned in `requirements.txt`: torch 2.11.0+cu130 (Colab's),
timm 1.0.30, open_clip_torch 3.3.0. Each run was rechecked on the GPU type it was trained
on: the five ImageNet models on a Tesla T4, DINOv3 and FetalCLIP on an RTX PRO 6000
Blackwell.

All 9 matched the saved `.npz` files exactly: largest difference 0 for logits, features
and MC-dropout logits, and the same prediction on all 555 images. That includes runs
saved at `cd4ed34`, `4f2e970`, `fcda2a3` and `b5fd958`, and both frozen runs, whose
backbone is rebuilt from the pinned pretrained weights at inference.

Reruns on a different GPU type than the original may not be bit-identical: newer GPUs use
lower-precision TF32 maths for some operations by default and the T4 does not.

