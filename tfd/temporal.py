"""Anti-flicker temporal operators (numpy, float32 RGB in 0..1).

* Farneback optical flow between adjacent *original* frames guides an
  occlusion-aware EMA blend of the *detailed* frames, damping residual
  sampling shimmer without smearing real motion.
* Color matching pins each detailed crop's mean/std (per RGB channel,
  face region only if a mask is given) to its own source crop, so
  exposure follows the original footage instead of pulsing.
"""

import cv2
import numpy as np


def _to_gray_u8(img_f32):
    return cv2.cvtColor(np.clip(img_f32 * 255.0, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY)


def farneback_flow(prev_f32, next_f32, downscale_to=384):
    """Dense flow prev -> next. Computed at reduced resolution for speed."""
    h, w = prev_f32.shape[:2]
    scale = 1.0
    if max(h, w) > downscale_to:
        scale = downscale_to / max(h, w)
        sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
        a = cv2.resize(prev_f32, (sw, sh), interpolation=cv2.INTER_AREA)
        b = cv2.resize(next_f32, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        a, b = prev_f32, next_f32
    flow = cv2.calcOpticalFlowFarneback(
        _to_gray_u8(a), _to_gray_u8(b), None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)
    if scale != 1.0:
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow *= 1.0 / scale
    return flow


def warp_by_flow(img_f32, flow):
    """Warp img (content of the *previous* frame) onto the next frame's grid."""
    h, w = img_f32.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x = gx - flow[..., 0]
    map_y = gy - flow[..., 1]
    return cv2.remap(img_f32, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _flow_confidence(orig_prev, orig_next, flow, err_scale=0.12):
    """Per-pixel confidence that the flow is valid (occlusion awareness)."""
    warped = warp_by_flow(orig_prev, flow)
    err = np.abs(warped - orig_next).mean(axis=2)
    conf = np.clip(1.0 - err / err_scale, 0.0, 1.0)
    return cv2.GaussianBlur(conf, (0, 0), 3.0)[..., None]


def flow_blend_sequence(detailed, originals, strength, bidirectional=True):
    """Occlusion-aware temporal EMA over a sequence of frames.

    detailed / originals: lists of float32 (H, W, 3), same length & size.
    Flow is estimated on the originals (true motion); the blend is applied
    to the detailed frames. Returns a new list.
    """
    n = len(detailed)
    strength = float(np.clip(strength, 0.0, 1.0))
    if n < 2 or strength <= 0.0:
        return list(detailed)
    alpha = 0.85 * strength  # cap so current frame always contributes

    def one_pass(order):
        out = [None] * n
        acc = detailed[order[0]].copy()
        out[order[0]] = acc
        for k in range(1, n):
            i_prev, i_cur = order[k - 1], order[k]
            flow = farneback_flow(originals[i_prev], originals[i_cur])
            conf = _flow_confidence(originals[i_prev], originals[i_cur], flow)
            warped = warp_by_flow(acc, flow)
            a = alpha * conf
            cur = detailed[i_cur]
            acc = cur * (1.0 - a) + warped * a
            out[i_cur] = acc
        return out

    fwd = one_pass(list(range(n)))
    if not bidirectional:
        return fwd
    bwd = one_pass(list(range(n - 1, -1, -1)))
    return [(f + b) * 0.5 for f, b in zip(fwd, bwd)]


def match_color(detailed, original, strength, mask=None):
    """Match per-channel mean/std of `detailed` to `original` (0..1 RGB).

    mask: optional float (H, W) weighting; stats are taken over the face
    region so background pixels don't skew the transfer.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return detailed
    if mask is not None and mask.sum() > 16:
        w = mask.astype(np.float64)[..., None]
    else:
        w = np.ones(detailed.shape[:2], dtype=np.float64)[..., None]
    wsum = w.sum()

    def stats(img):
        mean = (img * w).sum(axis=(0, 1)) / wsum
        var = ((img - mean) ** 2 * w).sum(axis=(0, 1)) / wsum
        return mean, np.sqrt(np.maximum(var, 1e-8))

    d_mean, d_std = stats(detailed.astype(np.float64))
    o_mean, o_std = stats(original.astype(np.float64))
    matched = (detailed - d_mean) * (o_std / d_std) + o_mean
    out = detailed * (1.0 - strength) + matched * strength
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def propagate_keyframe(detailed_key, orig_key, orig_target, err_scale=0.12):
    """Warp a detailed keyframe crop onto a non-key frame (skip-N mode).

    Returns (warped_detail, confidence) — low-confidence pixels should fall
    back to the target's original content.
    """
    flow = farneback_flow(orig_key, orig_target)
    conf = _flow_confidence(orig_key, orig_target, flow, err_scale)
    return warp_by_flow(detailed_key, flow), conf
