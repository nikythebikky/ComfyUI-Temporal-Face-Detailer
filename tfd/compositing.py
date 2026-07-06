"""Face masks and feathered paste-back compositing."""

import cv2
import numpy as np

from .utils import make_odd


def face_mask_for_crop(entry, crop_w, crop_h, dilation=8, feather=15):
    """Soft face mask in stabilized-crop coordinates, float32 (h, w) 0..1.

    An ellipse fitted to the detection bbox (slightly widened, extended
    toward the chin), refined by landmarks when available, then dilated
    and Gaussian-feathered so paste-back boundaries never crawl.
    """
    x1, y1, x2, y2 = entry["crop"]
    bx1, by1, bx2, by2 = entry["bbox"]
    # bbox in crop-local coordinates
    bx1, bx2 = bx1 - x1, bx2 - x1
    by1, by2 = by1 - y1, by2 - y1

    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
    ax = (bx2 - bx1) * 0.5 * 1.1
    ay = (by2 - by1) * 0.5 * 1.2
    cy += (by2 - by1) * 0.05  # bias toward the chin

    kps = entry.get("kps")
    if kps is not None:
        pts = np.asarray(kps, dtype=np.float32) - np.array([x1, y1], np.float32)
        # widen the ellipse if landmarks fall near its horizontal edge
        spread_x = (pts[:, 0].max() - pts[:, 0].min()) * 0.5
        ax = max(ax, spread_x * 1.6)

    cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                (max(2, int(round(ax))), max(2, int(round(ay)))),
                0, 0, 360, 255, -1)

    if dilation > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (make_odd(dilation * 2 + 1),) * 2)
        mask = cv2.dilate(mask, k)

    mask = mask.astype(np.float32) / 255.0
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (make_odd(feather * 2 + 1),) * 2,
                                feather * 0.5)
    return np.clip(mask, 0.0, 1.0)


def paste_crop(frame_f32, crop_f32, mask_f32, crop_box):
    """Composite a detailed crop back into the full frame, mask-limited.

    Only pixels under the mask change; everything else is left untouched
    so backgrounds stay bit-exact.
    """
    x1, y1, x2, y2 = crop_box
    region = frame_f32[y1:y2, x1:x2]
    m = mask_f32[..., None]
    frame_f32[y1:y2, x1:x2] = region * (1.0 - m) + crop_f32 * m
    return frame_f32


def accumulate_mask(full_mask, mask_f32, crop_box):
    x1, y1, x2, y2 = crop_box
    full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask_f32)
    return full_mask
