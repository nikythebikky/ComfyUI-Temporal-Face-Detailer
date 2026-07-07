"""Per-frame face detection backends.

Backends (chosen via the node's ``detector`` widget):

* ``insightface`` — RetinaFace via the ``insightface`` package (bbox +
  5-point landmarks + score). Best quality; needs ``insightface`` +
  ``onnxruntime`` installed. Models are stored/shared under
  ``models/insightface`` like other face packs (ReActor, IP-Adapter, ...).
* ``yunet`` — OpenCV's YuNet ONNX detector (bbox + 5-point landmarks +
  score). Small and fast; the model file is auto-downloaded once.
* ``haar`` — OpenCV Haar cascade. No extra downloads or deps, but no
  landmarks and weak on non-frontal faces. Last-resort fallback.
* ``yolo:<model>`` — any ultralytics YOLO detection model found under
  ``models/ultralytics`` (Impact-Pack convention, e.g.
  ``bbox/face_yolov8m_anime.pt``). This is the right choice for anime /
  stylized faces, which the realistic-face detectors miss or jitter on.
  Needs ``pip install ultralytics``. Bbox models yield no landmarks;
  pose-style face models with >=5 keypoints are used when present.
"""

import os
import urllib.request

import cv2
import numpy as np


YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"

DETECTOR_CHOICES = ["insightface", "yunet", "haar"]


def _ultralytics_model_paths():
    """Map of selectable YOLO model names -> paths under models/ultralytics."""
    found = {}
    try:
        import folder_paths
    except Exception:
        return found
    try:
        for n in folder_paths.get_filename_list("ultralytics"):
            if n.lower().endswith(".pt"):
                p = folder_paths.get_full_path("ultralytics", n)
                if p:
                    found[n] = p
    except Exception:
        pass
    if not found:  # Impact Pack not installed; scan the directory directly
        base = os.path.join(folder_paths.models_dir, "ultralytics")
        for sub in ("bbox", "segm", ""):
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(".pt"):
                    name = f"{sub}/{fn}" if sub else fn
                    found[name] = os.path.join(d, fn)
    return found


def detector_choices():
    """Static backends plus any YOLO models present on disk."""
    return DETECTOR_CHOICES + [f"yolo:{n}" for n in _ultralytics_model_paths()]


class Detection:
    __slots__ = ("bbox", "kps", "score")

    def __init__(self, bbox, kps, score):
        self.bbox = np.asarray(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.kps = None if kps is None else np.asarray(kps, dtype=np.float32)  # (5, 2)
        self.score = float(score)


def _models_root():
    """Directory for downloaded detector weights."""
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, "tfd")
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _insightface_root():
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, "insightface")
    except Exception:
        return os.path.join(_models_root(), "insightface")


class FaceDetector:
    def __init__(self, backend="insightface", device="cuda",
                 det_threshold=0.5, min_face_size=24, max_faces=4):
        self.backend = backend
        self.device = device
        self.det_threshold = float(det_threshold)
        self.min_face_size = int(min_face_size)
        self.max_faces = int(max_faces)
        self._impl = None
        self._init_backend()

    # ------------------------------------------------------------------ init
    def _init_backend(self):
        if self.backend.startswith("yolo:"):
            try:
                self._init_yolo()
                return
            except Exception as e:
                print(f"[TemporalFaceDetailer] YOLO detector unavailable "
                      f"({e}); falling back to insightface.")
                self.backend = "insightface"
        if self.backend == "insightface":
            try:
                self._init_insightface()
                return
            except Exception as e:
                print(f"[TemporalFaceDetailer] insightface unavailable ({e}); "
                      f"falling back to YuNet.")
                self.backend = "yunet"
        if self.backend == "yunet":
            try:
                self._init_yunet()
                return
            except Exception as e:
                print(f"[TemporalFaceDetailer] YuNet unavailable ({e}); "
                      f"falling back to Haar cascade.")
                self.backend = "haar"
        self._init_haar()

    def _init_insightface(self):
        from insightface.app import FaceAnalysis

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(
            name="buffalo_l",
            root=_insightface_root(),
            allowed_modules=["detection"],
            providers=providers,
        )
        app.prepare(
            ctx_id=0 if self.device == "cuda" else -1,
            det_thresh=self.det_threshold,
            det_size=(640, 640),
        )
        self._impl = app

    def _init_yunet(self):
        root = _models_root()
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, YUNET_FILENAME)
        if not os.path.exists(path):
            print(f"[TemporalFaceDetailer] downloading YuNet model to {path}")
            tmp = path + ".part"
            urllib.request.urlretrieve(YUNET_URL, tmp)
            os.replace(tmp, path)
        self._impl = cv2.FaceDetectorYN.create(
            path, "", (320, 320),
            score_threshold=self.det_threshold,
            nms_threshold=0.35,
            top_k=200,
        )

    def _init_yolo(self):
        from ultralytics import YOLO

        name = self.backend.split(":", 1)[1]
        path = _ultralytics_model_paths().get(name)
        if path is None:
            raise FileNotFoundError(
                f"YOLO model '{name}' not found under models/ultralytics")
        self._impl = YOLO(path)
        self._yolo_device = "cuda:0" if self.device == "cuda" else "cpu"

    def _init_haar(self):
        path = os.path.join(cv2.data.haarcascades,
                            "haarcascade_frontalface_default.xml")
        self._impl = cv2.CascadeClassifier(path)
        if self._impl.empty():
            raise RuntimeError("failed to load Haar cascade for face detection")
        self.backend = "haar"

    # ---------------------------------------------------------------- detect
    def detect(self, frame_rgb_u8: np.ndarray):
        """Detect faces in one uint8 RGB frame -> list[Detection]."""
        if self.backend.startswith("yolo:"):
            dets = self._detect_yolo(frame_rgb_u8)
        elif self.backend == "insightface":
            dets = self._detect_insightface(frame_rgb_u8)
        elif self.backend == "yunet":
            dets = self._detect_yunet(frame_rgb_u8)
        else:
            dets = self._detect_haar(frame_rgb_u8)

        h, w = frame_rgb_u8.shape[:2]
        out = []
        for d in dets:
            b = d.bbox
            b[0] = np.clip(b[0], 0, w - 1)
            b[1] = np.clip(b[1], 0, h - 1)
            b[2] = np.clip(b[2], 0, w - 1)
            b[3] = np.clip(b[3], 0, h - 1)
            if min(b[2] - b[0], b[3] - b[1]) < self.min_face_size:
                continue
            if d.score < self.det_threshold:
                continue
            out.append(d)
        # keep the largest N faces (closest to camera first)
        out.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                 reverse=True)
        return out[: self.max_faces]

    def _detect_yolo(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # ultralytics expects BGR
        res = self._impl.predict(bgr, conf=self.det_threshold,
                                 device=self._yolo_device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        kps_all = None
        if getattr(res, "keypoints", None) is not None:
            try:
                k = res.keypoints.xy.cpu().numpy()  # (n, K, 2)
                if k.ndim == 3 and k.shape[1] >= 5:
                    kps_all = k[:, :5]
            except Exception:
                pass
        return [Detection(xyxy[i],
                          None if kps_all is None else kps_all[i],
                          conf[i])
                for i in range(len(xyxy))]

    def _detect_insightface(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self._impl.get(bgr)
        return [Detection(f.bbox, getattr(f, "kps", None), f.det_score)
                for f in faces]

    def _detect_yunet(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        self._impl.setInputSize((w, h))
        _, faces = self._impl.detect(bgr)
        dets = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = f[0], f[1], f[2], f[3]
                kps = f[4:14].reshape(5, 2)
                dets.append(Detection([x, y, x + bw, y + bh], kps, f[14]))
        return dets

    def _detect_haar(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rects = self._impl.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size))
        return [Detection([x, y, x + w, y + h], None, 1.0)
                for (x, y, w, h) in rects]
