"""Detect measurement calipers drawn inside the ultrasound fan.

Some scanners draw calipers as small '+' crosses: saturated 1-pixel lines, about 4
pixels per arm, with a hollow centre (Ghana's images; also with dotted ellipses and
lines between them). OCR does not see these, so they are counted here.

Checked against a visual review of all 450 African test images
(splits/markers_reviewed_africa.csv): counting inside the fan, as find_markers.py does,
it finds 66 of the 73 Ghana images that carry crosses, with no false hits in Algeria,
Egypt or Malawi or in 137 Spanish images. It does not find Uganda's markers, which are
drawn in grey rather than saturated white; those come with text labels that OCR picks up.
"""
import numpy as np
from scipy import ndimage

SATURATED = 250     # crosses are drawn at 255
ARM = 4             # arm length checked, in pixels from the centre
MIN_ARM = 3         # saturated pixels needed on each arm


def caliper_crosses(gray: np.ndarray) -> int:
    """Number of '+' caliper marks: every arm has at least MIN_ARM saturated pixels
    and the diagonal neighbours of the centre are not saturated."""
    s = (gray >= SATURATED).astype(np.int8)
    h, w = s.shape
    p = ARM + 1
    padded = np.pad(s, p)

    def shifted(dy: int, dx: int) -> np.ndarray:
        return padded[p + dy:p + dy + h, p + dx:p + dx + w]

    arms = [sum(shifted(sy * d, sx * d) for d in range(1, ARM + 1))
            for sy, sx in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    diagonal = sum(shifted(dy, dx) for dy in (-2, -1, 1, 2) for dx in (-2, -1, 1, 2) if abs(dy) == abs(dx))
    hits = np.logical_and.reduce([a >= MIN_ARM for a in arms]) & (diagonal == 0)
    return int(ndimage.label(hits)[1])
