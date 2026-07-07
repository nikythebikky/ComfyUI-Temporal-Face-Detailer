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


_ANGLE_EPS = 0.05  # degrees below which the fast axis-aligned path is used


def _crop_affine(crop_box, angle):
    """Affine mapping frame coords -> rotation-registered crop coords."""
    x1, y1, x2, y2 = crop_box
    s = x2 - x1
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    m[0, 2] += s * 0.5 - cx
    m[1, 2] += s * 0.5 - cy
    return m


def extract_crop(frame_f32, crop_box, angle=0.0):
    """Cut the stabilized square crop, rotation-registered when angle != 0."""
    x1, y1, x2, y2 = crop_box
    if abs(angle) < _ANGLE_EPS:
        return frame_f32[y1:y2, x1:x2].copy()
    s = x2 - x1
    m = _crop_affine(crop_box, angle)
    return cv2.warpAffine(frame_f32, m, (s, s), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _warp_back(img_f32, crop_box, angle, frame_w, frame_h, replicate):
    """Warp crop-space content back to frame space.

    Returns ((rx1, ry1, rx2, ry2), warped) where the rect is the clipped
    frame-space footprint of the rotated crop square.
    """
    s = crop_box[2] - crop_box[0]
    m_inv = cv2.invertAffineTransform(_crop_affine(crop_box, angle))
    corners = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float64)
    fc = corners @ m_inv[:, :2].T + m_inv[:, 2]
    rx1 = int(np.clip(np.floor(fc[:, 0].min()), 0, frame_w))
    ry1 = int(np.clip(np.floor(fc[:, 1].min()), 0, frame_h))
    rx2 = int(np.clip(np.ceil(fc[:, 0].max()) + 1, 0, frame_w))
    ry2 = int(np.clip(np.ceil(fc[:, 1].max()) + 1, 0, frame_h))
    m_local = m_inv.copy()
    m_local[0, 2] -= rx1
    m_local[1, 2] -= ry1
    border = cv2.BORDER_REPLICATE if replicate else cv2.BORDER_CONSTANT
    warped = cv2.warpAffine(img_f32, m_local, (rx2 - rx1, ry2 - ry1),
                            flags=cv2.INTER_LINEAR, borderMode=border,
                            borderValue=0)
    return (rx1, ry1, rx2, ry2), warped


def paste_crop(frame_f32, crop_f32, mask_f32, crop_box, angle=0.0):
    """Composite a detailed crop back into the full frame, mask-limited.

    Only pixels under the mask change; everything else is left untouched
    so backgrounds stay bit-exact. When the crop was rotation-registered,
    both crop and mask are warped back through the inverse transform (the
    mask's zero border guarantees nothing outside the face changes).
    """
    if abs(angle) < _ANGLE_EPS:
        x1, y1, x2, y2 = crop_box
        region = frame_f32[y1:y2, x1:x2]
        m = mask_f32[..., None]
        frame_f32[y1:y2, x1:x2] = region * (1.0 - m) + crop_f32 * m
        return frame_f32
    h, w = frame_f32.shape[:2]
    rect, wcrop = _warp_back(crop_f32, crop_box, angle, w, h, replicate=True)
    _, wmask = _warp_back(mask_f32, crop_box, angle, w, h, replicate=False)
    rx1, ry1, rx2, ry2 = rect
    region = frame_f32[ry1:ry2, rx1:rx2]
    m = np.clip(wmask, 0.0, 1.0)[..., None]
    frame_f32[ry1:ry2, rx1:rx2] = region * (1.0 - m) + wcrop * m
    return frame_f32


def accumulate_mask(full_mask, mask_f32, crop_box, angle=0.0):
    if abs(angle) < _ANGLE_EPS:
        x1, y1, x2, y2 = crop_box
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2],
                                             mask_f32)
        return full_mask
    h, w = full_mask.shape[:2]
    rect, wmask = _warp_back(mask_f32, crop_box, angle, w, h, replicate=False)
    rx1, ry1, rx2, ry2 = rect
    full_mask[ry1:ry2, rx1:rx2] = np.maximum(
        full_mask[ry1:ry2, rx1:rx2], np.clip(wmask, 0.0, 1.0))
    return full_mask
