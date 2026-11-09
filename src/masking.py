"""Find the ultrasound fan in an image and black out everything outside it.

Outside the fan the image is background (exactly 0 for every source except Malawi,
where it is 1-3), plus burned-in text, depth scales, grey bars and logos. Those are
all small. Inside the fan even dark tissue carries some speckle above background,
though on low-gain images it is sparse, and black fluid can split the fan into
separate pieces. So rather than keep one connected region:

1. estimate the background level from the image border,
2. mark where above-background pixels are dense within a small window, so sparse
   speckle in dark tissue still counts as fan,
3. keep every region of at least 2% of the image (all pieces of the fan) and drop
   the small ones (text, scales, bars, logos),
4. close with a larger disk and fill holes, so fluid that shows as pure black
   inside the fan is kept.

On images cropped so tightly that the fan fills the frame, the scanner's screen
overlays (settings block, machine name, grey bar, depth scale) sit on top of the image
area and the fan mask keeps them. Those overlays sit at fixed screen positions for each
scanner, so overlay_boxes() lists them and mask_outside_fan() blacks them out as well.

fan_mask() returns a boolean array, True inside the fan; mask_outside_fan() applies it
together with the overlay boxes.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

WINDOW_FRACTION = 0.015  # density window, as a fraction of the longer image side
MIN_DENSITY = 0.3        # share of above-background pixels in the window
MIN_REGION = 0.02        # smallest region kept, as a fraction of the image area
CLOSE_FRACTION = 0.03    # closing radius, as a fraction of the longer image side


def background_level(gray: np.ndarray) -> int:
    """Highest value of the background noise, from the outermost 2% of the image."""
    h, w = gray.shape
    b = max(2, int(0.02 * min(h, w)))
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    mode = int(np.bincount(border).argmax())
    near = border[border <= mode + 5]
    return int(np.percentile(near, 99))


def fan_mask(gray: np.ndarray) -> np.ndarray:
    side = max(gray.shape)
    above = (gray > background_level(gray)).astype(np.float32)
    k = max(3, int(WINDOW_FRACTION * side))
    region = ndimage.uniform_filter(above, size=k) > MIN_DENSITY
    labels, n = ndimage.label(region)
    if n == 0:
        return np.ones_like(gray, dtype=bool)          # nothing found: leave the image as is
    sizes = ndimage.sum(region, labels, index=np.arange(1, n + 1))
    keep = np.flatnonzero(sizes >= MIN_REGION * gray.size) + 1
    if len(keep) == 0:
        keep = [1 + int(np.argmax(sizes))]
    fan = np.isin(labels, keep)
    # Closing with a disk of radius r, done with distance transforms so the cost doesn't grow
    # with r. Padded first so the fan isn't eaten away where it touches the image edge.
    r = max(1, int(CLOSE_FRACTION * side))
    padded = np.pad(fan, r + 1, mode="edge")
    dilated = ndimage.distance_transform_edt(~padded) <= r
    closed = ndimage.distance_transform_edt(dilated) > r
    return ndimage.binary_fill_holes(closed[r + 1:-r - 1, r + 1:-r - 1])


def overlay_boxes(source: str, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Screen overlays as (x0, y0, x1, y1) boxes for a source's layout, measured on the
    validation and test images: source is the Spanish machine or the African country."""
    if source == "Aloka":
        # grey bar top-left; readout top-right (2 lines on 640-wide screens, 8 on 740-wide)
        return [(0, 0, 40, 100), (width - 85, 0, width, 55 if width < 700 else 135)]
    if source.startswith("Voluson") or source == "Other":        # "Other" is the Voluson S8
        # grey bar and depth ticks down the left edge, machine name top-left, settings top-right
        right, bottom = (125, 125) if width < 1000 else (165, 180)
        return [(0, 0, 40, height), (0, 0, 90, 75), (width - right, 0, width, bottom)]
    if source == "Algeria":
        return [(25, 0, 95, 50)]                                  # "Voluson S8", top-left
    return []


def image_mask(gray: np.ndarray, source: str) -> np.ndarray:
    """True where the image is kept: inside the fan and outside the screen overlays."""
    mask = fan_mask(gray)
    for x0, y0, x1, y1 in overlay_boxes(source, gray.shape[1], gray.shape[0]):
        mask[max(y0, 0):y1, max(x0, 0):x1] = False
    return mask


def mask_outside_fan(image: Image.Image, source: str) -> tuple[Image.Image, np.ndarray]:
    """Return the image with everything outside the fan and every screen overlay set to
    black, and the mask."""
    gray = np.asarray(image.convert("L"))
    mask = image_mask(gray, source)
    pixels = np.asarray(image).copy()
    pixels[~mask] = 0
    return Image.fromarray(pixels), mask
