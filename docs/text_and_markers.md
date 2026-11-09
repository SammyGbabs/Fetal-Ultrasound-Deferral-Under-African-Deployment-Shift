# Burned-in text and measurement markers

Some test images carry text or markers that could tell a model the view without it
looking at the anatomy. This note records what is there, how the masked test sets
remove it, and what masking can't remove. Checked in Nov 2026 on spain_val,
spain_test and africa_test.

## What the images show

| Source | Text or markers | Differs between views? |
| --- | --- | --- |
| Voluson E6, S10, S8 (Spain) | Machine name top-left; grey bar and depth scale down the left edge; settings block top-right with the exam preset, frequency, gain, harmonics | **Yes: the preset** (below) |
| Aloka (Spain) | Grey bar top-left; frame count and frequency top-right (2 lines on 640-wide screens, 8 on 740-wide) | Not seen |
| Algeria | "Voluson S8", faint, top-left | No |
| Malawi | "M" logo, top-left | No |
| Egypt | Nothing | - |
| Uganda | **Calipers inside the fan on all 75 images**: "OFD"/"BPD" on brain, "ATD"/"ASD" on abdomen (one also "RIB", "UV", "SP", "ST"), numbered calipers with dotted lines on femur | **Yes** |
| Ghana | **Caliper crosses with dotted ellipses or lines inside the fan on 73 of 75 images**, no text: ellipses on brain and abdomen, a straight line on femur | **Yes**, by marker shape |

The African counts come from looking at every one of the 450 images
(`splits/markers_reviewed_africa.csv`). No calipers were seen in the 137 Spanish test
images looked at; `scripts/find_markers.py` checked all of them and found a few (below).

### The exam preset gives the thorax view away

OCR of the settings block on all 3,569 Spanish validation and test images:

| Scanner | Thorax images showing "Cardiac" | Other views showing "Cardiac" |
| --- | --- | --- |
| Voluson E6 | 510 of 614 (83%) | abdomen 7%, brain 2%, femur 0% |
| Voluson S10 | 66 of 71 (93%) | abdomen 17%, brain and femur 0% |

The other views mostly show a trimester preset ("2. Trim.", "3. Trim./OB"). On a
Voluson E6, an image labelled "Cardiac" is a thorax view 94% of the time. Training
images come from the same scanners, so a model could learn the word instead of the
anatomy.

## The masked test sets

`spain_val_masked`, `spain_test_masked` and `africa_test_masked` (in
`splits/conditions.csv`) are the same images with everything outside the ultrasound
fan blacked out, plus the scanner's screen overlays (`src/masking.py`,
`scripts/make_masked.py`):

- **Fan.** Not just the largest bright region: on low-gain images and where black fluid
  splits the fan, that loses real anatomy (half a skull on one Ghana image). Instead,
  every large region where above-background pixels are dense is kept, small ones (text,
  scales, logos) are dropped, and holes are filled so pure-black fluid stays.
- **Overlays.** On images cropped so the fan fills the frame (490 Spanish images, mostly
  Voluson E6, S10 and Aloka), the settings text sits on top of the image and a fan mask
  keeps it. The overlays sit at fixed screen positions per scanner, measured on the
  validation and test images, so those boxes are blacked out too.

Checks: original and masked side by side for one image per scanner and country
(`data/masked/preview_grid.png` on the Drive), fan coverage for every image
(`data/masked/<condition>_fan.csv`) and tests in `tests/test_masking.py`. OCR of the top
corners of all 3,569 masked Spanish images still reads text on 26 (0.7%), none of it a
preset: machine-name fragments ("E6", "S10") placed elsewhere on some screens, and
operator annotations in Spanish ("DCHO ANT", "IZQ") that turn up across all views.
`scripts/find_markers.py` repeats this on the whole image for every test image.

## Markers flagged by find_markers.py

`outputs/markers_ocr.csv` on the Drive covers every image in spain_test and africa_test.

- **Africa.** 112 of 450 flagged, with no false flags: Ghana 66 of 73 (caliper
  crosses) and Uganda 46 of 75 (OCR of the labels). It misses all 25 Uganda femur
  images, whose calipers carry only a small digit. For Africa, use
  `splits/markers_reviewed_africa.csv` (all 148 marked images, checked by eye) and
  treat the OCR file as a cross-check.
- **Spain.** 64 of 3,014 flagged, 58 of them on Voluson E6 and 40 of them brain views.
  Most are operator notes inside the fan: twin labels ("feto a", "feto b", 34 images)
  and sides ("DCHO", "IZQ", "DER"). About 10 carry measurement text ("D 20.57mm",
  "AC", "EFW"). 5 have caliper crosses.
- **Left after masking.** 75 Spanish and 45 Ugandan images still have readable text,
  nearly all of it the same notes and labels inside the fan. Only about 5 Spanish images
  keep a fragment of the machine name ("E6", "Voluson"). No preset word is left.

## Effect on the results

Masking barely changes accuracy. Averaged over the 45 runs, accuracy moves by -0.1
points on spain_test and -0.2 on africa_test (worst run -1.3 and -1.1). The exception is
frozen DINOv3, which drops about 1 point on spain_test (95.6% to 94.6%). The fine-tuned
models and FetalCLIP move by 0.1 points or less, so they don't appear to rely on the
burned-in text.

## What masking does not remove

- **Calipers inside the fan.** Uganda's and Ghana's markers can't be cut out without
  cutting the anatomy they sit on. `scripts/find_markers.py` flags them instead
  (`outputs/markers_ocr.csv`), so results can be compared on marked and unmarked images.
- **The fan's shape.** Sector angle and depth stay visible, and could differ by view.
- **Very dark edges.** Tissue at the fan's edge that is nearly black (pixel values 0-3)
  can be cut off with the background, mainly on a few low-gain images. It carries next to
  no visible content.
