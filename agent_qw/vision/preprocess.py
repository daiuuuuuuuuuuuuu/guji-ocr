"""Image preprocessing: binarization and optional deskew."""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL RGB to OpenCV BGR."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR to PIL RGB."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def adaptive_binarize(
    image: np.ndarray,
    block_size: int = 31,
    c: int = 10,
) -> np.ndarray:
    """Apply adaptive threshold binarization.

    Classical Chinese texts often have uneven ink density, so adaptive
    thresholding works better than global Otsu.

    Args:
        image: BGR or grayscale numpy array.
        block_size: Size of the pixel neighborhood (must be odd).
        c: Constant subtracted from the mean.

    Returns:
        Binary image (single channel, 0=black text, 255=white background).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold: keeps text clear even with uneven backgrounds
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )

    # Invert so text is white (255) on black (0) — easier for projection
    binary = cv2.bitwise_not(binary)
    return binary


def sharpen(
    image: np.ndarray,
    amount: float = 1.5,
    radius: int = 3,
) -> np.ndarray:
    """Sharpen a binary or grayscale image using unsharp masking.

    Unsharp masking works by:
      1. Blur the image (remove detail)
      2. Extract detail = original - blurred
      3. Add amplified detail back: result = original + amount * detail

    For古籍 OCR, this makes faded/thin strokes more distinct,
    which helps both projection-based layout and character recognition.

    Args:
        image: Single-channel numpy array (binary or grayscale).
        amount: How much to amplify edges (1.0 = none, 1.5 = moderate, 3.0 = strong).
        radius: Gaussian blur kernel radius.

    Returns:
        Sharpened image (same dtype as input).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Gaussian blur to get the "unsharp" mask
    blurred = cv2.GaussianBlur(gray, (0, 0), radius)

    # detail = original - blurred
    # Clamp result to valid range
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)

    logger.debug("Sharpened image (amount=%.1f, radius=%d)", amount, radius)
    return sharpened


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct small rotations in the input image.

    Uses the dominant line angle from Hough line detection.
    Only handles rotations up to ~5 degrees.

    Args:
        image: Grayscale or BGR numpy array.

    Returns:
        Deskewed image in the same color space.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough lines to find dominant angle
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        logger.debug("No lines found for deskew; returning original")
        return image

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.rad2deg(theta) - 90  # convert to degrees from horizontal
        if -5 < angle < 5:
            angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.3:
        logger.debug("Skew angle %.2f too small; skipping deskew", median_angle)
        return image

    logger.info("Deskewing by %.2f degrees", median_angle)

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, rot_mat, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess(
    pil_image: Image.Image,
    binarize: bool = True,
    do_sharpen: bool = True,
    sharpen_amount: float = 1.5,
    do_deskew: bool = False,
) -> tuple:
    """Full preprocessing pipeline: PIL → OpenCV → binarize → sharpen → deskew.

    Returns (processed_numpy_array, original_bgr_array).
    The processed array is used for layout analysis; original for OCR.
    """
    original_bgr = pil_to_cv2(pil_image)

    if binarize:
        processed = adaptive_binarize(original_bgr)
        # Sharpen after binarize: makes thin strokes more distinct for projection
        if do_sharpen:
            processed = sharpen(processed, amount=sharpen_amount)
    else:
        processed = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    if do_deskew:
        processed = deskew(processed)

    return processed, original_bgr
