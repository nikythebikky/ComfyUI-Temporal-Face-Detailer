"""FACE_TRACKS construction: stabilized crops + serializable track dicts.

FACE_TRACKS is a plain dict so it can flow between the split nodes:

    {
      "width": W, "height": H, "num_frames": N,
      "tracks": [
        {
          "track_id": int,
          "seed_offset": int,           # per-track seed = base_seed + this
          "frames": {
            frame_idx: {
              "bbox": [x1, y1, x2, y2],   # raw detection (float)
              "kps": [[x, y] * 5] | None, # landmarks (float)
              "score": float,             # 0.0 for interpolated frames
              "crop": [x1, y1, x2, y2],   # stabilized square crop (int)
              "interpolated": bool,
            }, ...
          },
        }, ...
      ],
    }
"""

import cv2
import numpy as np

from .utils import moving_average, np_to_tensor, tensor_frame_to_np, track_color


def stabilized_crop_boxes(track, width, height, crop_factor, crop_smoothing):
    """Per-frame square crop boxes: smoothed center, constant per-track size.

    The moving-average filter on (cx, cy) removes frame-to-frame jitter of
    the crop window itself — one of the main sources of flicker in naive
    per-frame detailing. The size is fixed per track (75th percentile of
    the smoothed detection sizes) so every crop in a track has identical
    pixel dimensions: latent shapes stay constant (a requirement for
    reusing the same sampling noise on every frame) and temporal ops run
    at a single resolution.
    """
    idxs = track.sorted_idxs()
    centers = np.array(
        [[(track.entries[i].bbox[0] + track.entries[i].bbox[2]) * 0.5,
          (track.entries[i].bbox[1] + track.entries[i].bbox[3]) * 0.5]
         for i in idxs], dtype=np.float64)
    sizes = np.array(
        [max(track.entries[i].bbox[2] - track.entries[i].bbox[0],
             track.entries[i].bbox[3] - track.entries[i].bbox[1])
         for i in idxs], dtype=np.float64)
    sizes = sizes * float(crop_factor)

    # window scales with smoothing strength; ~1s of video at the top end
    window = 1 + 2 * int(round(float(crop_smoothing) * 14))
    centers = moving_average(centers, window)
    sizes = moving_average(sizes, window)

    size = int(round(float(np.quantile(sizes, 0.75))))
    size = max(32, min(size, min(width, height)))

    boxes = {}
    for k, i in enumerate(idxs):
        cx, cy = centers[k]
        x1 = int(round(cx - size / 2))
        y1 = int(round(cy - size / 2))
        # keep the crop square by shifting it back inside the frame
        x1 = max(0, min(x1, width - size))
        y1 = max(0, min(y1, height - size))
        boxes[i] = [x1, y1, x1 + size, y1 + size]
    return boxes


def tracks_to_face_tracks(tracks, width, height, num_frames,
                          crop_factor, crop_smoothing):
    """Convert tracker output into the FACE_TRACKS dict."""
    out = {"width": int(width), "height": int(height),
           "num_frames": int(num_frames), "tracks": []}
    for t in tracks:
        crops = stabilized_crop_boxes(t, width, height,
                                      crop_factor, crop_smoothing)
        frames = {}
        for i in t.sorted_idxs():
            d = t.entries[i]
            frames[int(i)] = {
                "bbox": [float(v) for v in d.bbox],
                "kps": None if d.kps is None else d.kps.tolist(),
                "score": float(d.score),
                "crop": [int(v) for v in crops[i]],
                "interpolated": i in t.interpolated,
            }
        out["tracks"].append({
            "track_id": int(t.track_id),
            # decorrelate sampling noise between different people
            "seed_offset": int(t.track_id) * 1000003,
            "frames": frames,
        })
    return out


def draw_debug_overlay(images, face_tracks):
    """Draw bboxes, stabilized crops, landmarks and track IDs -> IMAGE."""
    frames_out = []
    n = images.shape[0]
    per_frame = [[] for _ in range(n)]
    for tr in face_tracks["tracks"]:
        for idx, e in tr["frames"].items():
            if 0 <= int(idx) < n:
                per_frame[int(idx)].append((tr["track_id"], e))

    for i in range(n):
        img = tensor_frame_to_np(images[i]).copy()
        for tid, e in per_frame[i]:
            color = track_color(tid)
            x1, y1, x2, y2 = [int(v) for v in e["bbox"]]
            thick = 1 if e["interpolated"] else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            cx1, cy1, cx2, cy2 = e["crop"]
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), color, 1,
                          lineType=cv2.LINE_AA)
            if e["kps"] is not None:
                for (px, py) in e["kps"]:
                    cv2.circle(img, (int(px), int(py)), 2, color, -1)
            label = f"id {tid}" + (" (interp)" if e["interpolated"] else "")
            cv2.putText(img, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        frames_out.append(img)

    return np_to_tensor(np.stack(frames_out, axis=0))
