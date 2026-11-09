# Duplicate images across patients

`scripts/find_duplicates.py` compared every pair of images in spain_train, spain_val,
spain_test and africa_test (7,011 images), within and across splits and Spain against
Africa. It looks for identical pixels (exact) and for the same picture resized, re-saved
or slightly shifted (near: perceptual hash, then thumbnail correlation of 0.95 or more).
Frames from the same patient are expected to look alike, so only matches between
different patient IDs count. Checked in Nov 2026.

The result is `splits/duplicates.csv`: one row per duplicate image, with the image it
copies (`duplicate_of`, from the earliest split), `exact` or `near`, and the patient it
should be merged into. Every pair, including borderline ones that aren't counted, is in
`outputs/duplicate_pairs.csv` on the Drive, and `outputs/near_duplicates.png` shows the
near and borderline pairs side by side.

## What was found

84 duplicate images (66 exact, 18 near), in 20 patient groups.

| Where | Images | Patient groups |
| --- | --- | --- |
| africa_test, within | 55 (47 exact, 8 near) | Algeria 6, Malawi 6, Egypt 3 |
| spain_train, within | 15 (6 exact, 9 near) | 3 |
| spain_train to spain_val | 14 (13 exact, 1 near) | 2 |
| Anything to spain_test | 0 | - |
| Spain to Africa | 0 | - |

- **Africa.** Algeria: six patients appear twice with all four views, under IDs 27 apart
  (004 = 031, 007 = 034, ...). Malawi: 20 copies across six groups; patient009 shares
  views with both 020 and 030. Egypt: 11 copies, 8 of them near rather than exact
  (001 = 003 = 006, 002 = 023, 005 = 008). Ghana and Uganda: none. 395 of the 450 images
  are distinct.
- **Spain.** All five groups are Voluson E6 scans whose IDs are 10-13 apart (779 = 791,
  781 = 793, 786 = 799, 778 = 790, 820 = 830), often with the frame numbers shuffled.
  Two of them cross into validation: ES_00793 (3 of its 4 images) and ES_00778 (11 of its
  14) are copies of training patients ES_00781 and ES_00790. That's 14 of the 555
  validation images.
- **Labels.** Every duplicate has the same label as its original.

## What it means

- spain_test and africa_test share no images with training, so the test results don't
  include training images.
- Validation is slightly optimistic: 2.5% of it was seen in training. That affects early
  stopping, and any threshold or calibration fitted on validation.
- africa_test counts 55 copies, mostly Algeria (24) and Malawi (20), so those countries
  have fewer independent images than the counts suggest. Per-country confidence
  intervals should be computed on the deduplicated set.

## Borderline pairs

Between 0.85 and 0.95 correlation there are 9 more pairs. Eight are other frames of
scans already merged (Spain 820/830 and 778/790, Malawi 012/023, Egypt 001/003), so they
change no patient merges; the ninth (Uganda 037/063) is two different scans.
