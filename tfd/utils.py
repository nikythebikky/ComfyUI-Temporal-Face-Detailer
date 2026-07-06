"""Small shared helpers: tensor <-> numpy conversion, geometry, drawing."""

import numpy as np
import torch


def tensor_frame_to_np(frame: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE frame (H, W, C) float 0-1 -> uint8 RGB (H, W, 3)."""
    a = frame.detach().cpu().numpy()
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


def tensor_frame_to_np_f32(frame: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE frame (H, W, C) float 0-1 -> float32 RGB (H, W, 3)."""
    return frame.detach().cpu().numpy().astype(np.float32)


def np_to_tensor(a: np.ndarray) -> torch.Tensor:
    """float32/uint8 RGB (H, W, 3) or (B, H, W, 3) -> float 0-1 tensor."""
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(a.astype(np.float32)))


def round_to_multiple(x, m: int = 8) -> int:
    return max(m, int(round(x / m)) * m)


def make_odd(x: int) -> int:
    x = int(x)
    return x if x % 2 == 1 else x + 1


def bbox_iou(a, b) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1e-6))


def bbox_center(b):
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along axis 0 with reflect padding."""
    window = make_odd(max(1, window))
    if window <= 1 or len(values) <= 2:
        return values
    window = min(window, make_odd(len(values)))
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.empty_like(values, dtype=np.float64)
    v = values.astype(np.float64)
    if v.ndim == 1:
        v = v[:, None]
        out = out.reshape(-1, 1)
    for c in range(v.shape[1]):
        padded = np.pad(v[:, c], pad, mode="reflect")
        out[:, c] = np.convolve(padded, kernel, mode="valid")
    return out.reshape(values.shape).astype(values.dtype)


TRACK_COLORS = [
    (66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36),
]


def track_color(track_id: int):
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]
