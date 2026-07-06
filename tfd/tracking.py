"""Link per-frame detections into temporally consistent identity tracks.

Greedy association on a combined IoU + normalized-centroid-distance cost
against a constant-velocity prediction of each active track. Tracks
tolerate up to ``max_gap`` missed frames (brief occlusion); the gap is
filled afterwards by linear interpolation of bbox and landmarks so the
detailer sees a continuous crop sequence.
"""

import numpy as np

from .detection import Detection
from .utils import bbox_center, bbox_iou


class Track:
    def __init__(self, track_id, frame_idx, det):
        self.track_id = track_id
        self.entries = {frame_idx: det}     # frame_idx -> Detection
        self.interpolated = set()           # frame idxs that were gap-filled
        self.last_idx = frame_idx
        self.velocity = np.zeros(4, dtype=np.float32)
        self.missed = 0

    @property
    def last_det(self):
        return self.entries[self.last_idx]

    def predicted_bbox(self):
        return self.last_det.bbox + self.velocity * (self.missed + 1)

    def add(self, frame_idx, det):
        prev = self.last_det.bbox
        dt = max(1, frame_idx - self.last_idx)
        new_v = (det.bbox - prev) / dt
        # damped velocity update keeps the prediction from overshooting
        self.velocity = 0.5 * self.velocity + 0.5 * new_v
        self.entries[frame_idx] = det
        self.last_idx = frame_idx
        self.missed = 0

    def sorted_idxs(self):
        return sorted(self.entries.keys())


def _association_cost(track, det):
    pred = track.predicted_bbox()
    iou = bbox_iou(pred, det.bbox)
    pc, dc = bbox_center(pred), bbox_center(det.bbox)
    size = max(pred[2] - pred[0], pred[3] - pred[1],
               det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1], 1.0)
    dist = np.hypot(pc[0] - dc[0], pc[1] - dc[1]) / size
    return iou, dist


def build_tracks(detections_per_frame, iou_threshold=0.25, max_gap=10,
                 min_track_length=2):
    """detections_per_frame: list (len == num frames) of list[Detection].

    Returns list[Track] with gaps interpolated.
    """
    active = []
    finished = []
    next_id = 0

    for frame_idx, dets in enumerate(detections_per_frame):
        # score all (track, det) pairs
        pairs = []
        for ti, tr in enumerate(active):
            for di, det in enumerate(dets):
                iou, dist = _association_cost(tr, det)
                if iou >= iou_threshold or dist < 0.5:
                    pairs.append((iou - 0.3 * dist, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            active[ti].add(frame_idx, dets[di])
            used_t.add(ti)
            used_d.add(di)

        # unmatched detections start new tracks
        for di, det in enumerate(dets):
            if di not in used_d:
                active.append(Track(next_id, frame_idx, det))
                next_id += 1

        # age out tracks that exceeded the gap tolerance
        still_active = []
        for ti, tr in enumerate(active):
            if ti in used_t or tr.last_idx == frame_idx:
                still_active.append(tr)
            else:
                tr.missed += 1
                if tr.missed > max_gap:
                    finished.append(tr)
                else:
                    still_active.append(tr)
        active = still_active

    finished.extend(active)
    tracks = [t for t in finished if len(t.entries) >= max(1, min_track_length)]

    for t in tracks:
        _interpolate_gaps(t)

    tracks.sort(key=lambda t: t.track_id)
    for new_id, t in enumerate(tracks):
        t.track_id = new_id
    return tracks


def _interpolate_gaps(track):
    """Fill missing frame indices inside a track by linear interpolation."""
    idxs = track.sorted_idxs()
    for a, b in zip(idxs[:-1], idxs[1:]):
        if b - a <= 1:
            continue
        da, db = track.entries[a], track.entries[b]
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            bbox = da.bbox * (1 - t) + db.bbox * t
            kps = None
            if da.kps is not None and db.kps is not None:
                kps = da.kps * (1 - t) + db.kps * t
            track.entries[i] = Detection(bbox, kps, 0.0)
            track.interpolated.add(i)
