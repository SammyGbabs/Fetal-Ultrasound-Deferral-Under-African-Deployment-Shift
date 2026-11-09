"""Checks for src/masking.py and src/markers.py on synthetic images."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.markers import caliper_crosses  # noqa: E402
from src.masking import fan_mask, image_mask  # noqa: E402

rng = np.random.default_rng(0)


def synthetic_image() -> np.ndarray:
    """A 400x600 sector of speckle on a black background, with burned-in text outside it."""
    h, w = 400, 600
    y, x = np.mgrid[:h, :w]
    angle = np.degrees(np.arctan2(x - w / 2, y + 60))
    radius = np.hypot(x - w / 2, y + 60)
    fan = (np.abs(angle) < 35) & (radius > 90) & (radius < 450)
    image = np.zeros((h, w), dtype=np.uint8)
    image[fan] = rng.integers(20, 200, fan.sum())
    image[(np.hypot(x - 300, y - 250) < 40)] = 0          # fluid inside the fan: pure black
    dark_edge = fan & (x > 420) & (y > 300)
    image[dark_edge] = rng.integers(4, 9, dark_edge.sum())   # very dark tissue at the fan's edge
    image[20:28, 480:580] = 230                              # a line of settings text
    image[100:300:20, 5:12] = 255                            # depth scale ticks
    return image, fan


def test_fan_mask_keeps_fan_and_drops_text():
    image, fan = synthetic_image()
    mask = fan_mask(image)
    assert mask[fan].mean() > 0.97                           # fan kept, fluid and dark edge included
    assert not mask[20:28, 480:580].any()                    # text removed
    assert not mask[100:300, 5:12].any()                     # scale removed


def test_fan_mask_handles_noisy_background():
    image, fan = synthetic_image()
    noisy = image.copy()
    background = ~fan & (image == 0)
    noisy[background] = rng.integers(0, 4, background.sum())   # Malawi-style background noise
    mask = fan_mask(noisy)
    assert mask[fan].mean() > 0.97
    assert mask[~fan].mean() < 0.05


def test_caliper_crosses():
    image = np.full((60, 60), 40, dtype=np.uint8)
    assert caliper_crosses(image) == 0
    for cy, cx in ((20, 20), (40, 45)):                         # two hollow '+' crosses
        image[cy - 4:cy, cx] = image[cy + 1:cy + 5, cx] = 255
        image[cy, cx - 4:cx] = image[cy, cx + 1:cx + 5] = 255
    assert caliper_crosses(image) == 2
    image[5:15, 50:58] = 255                                    # a saturated blob is not a cross
    assert caliper_crosses(image) == 2


def test_overlay_boxes_removed_on_full_frame_images():
    """When the fan fills the frame, the screen overlays sit on the image and must still go."""
    image, _ = synthetic_image()
    h, w = image.shape
    black = image == 0
    image[black] = rng.integers(20, 200, black.sum())        # speckle right up to the edges
    image[10:30, w - 110:w - 10] = 255                       # settings text on top of tissue
    fan = fan_mask(image)
    assert fan[10:30, w - 110:w - 10].all()                  # the fan mask alone keeps it
    mask = image_mask(image, "Voluson E6")
    assert not mask[10:30, w - 110:w - 10].any()             # the overlay box removes it
    assert not mask[:, :40].any()                            # depth scale strip
    assert (mask == fan)[200:, 100:w - 130].all()            # nothing else changes
    assert (image_mask(image, "Egypt") == fan).all()         # no overlays listed for Egypt
