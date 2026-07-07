"""ComfyUI node definitions.

All-in-one ``TemporalFaceDetailer`` plus split nodes
(``FaceDetectTrack`` -> ``TrackedFaceDetail`` -> ``TemporalSmooth``) that
share the same internals, a ``FaceTrackPreview`` visualizer and a
``TFDLoRAStack`` helper for stacking face LoRAs.
"""

import numpy as np
import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
import folder_paths

from .detailing import detail_tracks, encode_text, parse_track_prompts
from .detection import FaceDetector, detector_choices
from .face_tracks import draw_debug_overlay, tracks_to_face_tracks
from .temporal import FLOW_BACKENDS, flow_blend_sequence, get_flow_fn
from .tracking import build_tracks
from .utils import np_to_tensor, tensor_frame_to_np, tensor_frame_to_np_f32

CATEGORY = "TemporalFaceDetailer"

_SEED_MAX = 0xFFFFFFFFFFFFFFFF


# ------------------------------------------------------------ shared inputs

def _detect_inputs():
    return {
        "detector": (detector_choices(),
                     {"default": "insightface",
                      "tooltip": "yolo:* entries are ultralytics models "
                                 "found in models/ultralytics — use an "
                                 "anime face model (e.g. "
                                 "bbox/face_yolov8m_anime.pt) for "
                                 "stylized characters"}),
        "det_threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0,
                                    "step": 0.01}),
        "min_face_size": ("INT", {"default": 24, "min": 8, "max": 1024}),
        "max_faces": ("INT", {"default": 4, "min": 1, "max": 16}),
        "detector_device": (["cuda", "cpu"], {"default": "cuda"}),
        "iou_threshold": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.95,
                                    "step": 0.05,
                                    "tooltip": "min overlap to link a "
                                               "detection to a track"}),
        "max_track_gap": ("INT", {"default": 10, "min": 0, "max": 120,
                                  "tooltip": "frames a face may vanish "
                                             "(occlusion) before its track "
                                             "ends; gaps are interpolated"}),
        "min_track_length": ("INT", {"default": 2, "min": 1, "max": 120}),
        "crop_factor": ("FLOAT", {"default": 1.7, "min": 1.0, "max": 4.0,
                                  "step": 0.1}),
        "crop_smoothing": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0,
                                     "step": 0.05,
                                     "tooltip": "temporal smoothing of the "
                                                "crop window (anti-jitter)"}),
        "crop_anchor": (["landmarks", "bbox"],
                        {"default": "landmarks",
                         "tooltip": "landmarks centers the crop on the "
                                    "facial-landmark centroid (far more "
                                    "stable than the detection box); falls "
                                    "back to bbox when the detector yields "
                                    "no landmarks"}),
        "align_rotation": ("BOOLEAN",
                           {"default": False,
                            "tooltip": "rotation-register crops so the eye "
                                       "line is horizontal in every sampled "
                                       "crop (similarity transform, "
                                       "inverse-warped on paste-back); "
                                       "needs a landmark-capable detector"}),
    }


def _detail_inputs():
    return {
        "guide_size": ("INT", {"default": 768, "min": 256, "max": 2048,
                               "step": 8,
                               "tooltip": "resolution faces are resampled "
                                          "at"}),
        "max_size": ("INT", {"default": 1024, "min": 256, "max": 2048,
                             "step": 8}),
        "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
        "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0,
                          "step": 0.1}),
        "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
        "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
        "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0,
                              "step": 0.01,
                              "tooltip": "main quality/consistency lever: "
                                         "higher = more detail but more "
                                         "flicker (0.3-0.45 recommended)"}),
        "denoise_max": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                                  "step": 0.01,
                                  "tooltip": "adaptive denoise: when above "
                                             "'denoise', steady frames keep "
                                             "the low base value while "
                                             "high-motion frames ramp "
                                             "toward this. 0 = off (single "
                                             "global denoise)"}),
        "noise_mode": (["fixed_per_track", "per_frame"],
                       {"default": "fixed_per_track",
                        "tooltip": "fixed_per_track reuses the same seed "
                                   "and noise on every frame of a track — "
                                   "the biggest identity-stability lever"}),
        "detail_mode": (["img2img", "inpaint"],
                        {"default": "img2img",
                         "tooltip": "inpaint restricts sampling to the face "
                                    "mask in latent space; img2img resamples "
                                    "the whole crop (mask still limits the "
                                    "paste-back)"}),
        "mask_dilation": ("INT", {"default": 8, "min": 0, "max": 128}),
        "feather": ("INT", {"default": 15, "min": 0, "max": 128}),
        "temporal_strength": ("FLOAT", {"default": 1.0, "min": 0.0,
                                        "max": 1.0, "step": 0.05,
                                        "tooltip": "master anti-flicker "
                                                   "strength; scales flow "
                                                   "blend + color match"}),
        "flow_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                    "step": 0.05,
                                    "tooltip": "optical-flow-guided temporal "
                                               "blend of detailed crops "
                                               "(pixel space)"}),
        "latent_blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                                   "step": 0.05,
                                   "tooltip": "flow-guided temporal blend in "
                                              "LATENT space before decode — "
                                              "smooths in the VAE's semantic "
                                              "space, letting you raise "
                                              "denoise with less flicker. "
                                              "Try 0.3-0.5 with denoise "
                                              "0.25+; 0 = off"}),
        "flow_bidirectional": ("BOOLEAN", {"default": True}),
        "flow_backend": (FLOW_BACKENDS,
                         {"default": "farneback",
                          "tooltip": "raft_small/raft_large (torchvision) "
                                     "give much cleaner flow on fast motion "
                                     "at some VRAM/time cost; auto-falls "
                                     "back to farneback on any failure"}),
        "color_match": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                  "step": 0.05,
                                  "tooltip": "match each detailed crop's "
                                             "color to its source (stops "
                                             "brightness pulsing)"}),
        "chunk_size": ("INT", {"default": 4, "min": 1, "max": 64,
                               "tooltip": "crops sampled per batch; lower "
                                          "if you hit OOM"}),
        "detail_every": ("INT", {"default": 1, "min": 1, "max": 30,
                                 "tooltip": "keyframe mode: sample every "
                                            "Nth frame, flow-propagate the "
                                            "rest (speed on long clips)"}),
    }


def _optional_cond_inputs():
    return {
        "positive": ("CONDITIONING", {"tooltip": "overrides positive_text"}),
        "negative": ("CONDITIONING", {"tooltip": "overrides negative_text"}),
        "lora_stack": ("LORA_STACK",),
        "track_prompts": ("STRING", {
            "multiline": True, "default": "",
            "tooltip": "per-track positive prompt overrides, one per line: "
                       "'track_id: prompt' (see the debug overlay for IDs)"}),
        "reference_image": ("IMAGE", {
            "tooltip": "identity anchor: a reference face image the "
                       "detailed faces are biased toward (init-latent nudge "
                       "+ color anchoring). For strong identity conditioning "
                       "also patch the MODEL with IPAdapter FaceID upstream "
                       "— crops are sampled with whatever model you feed "
                       "in, so it composes"}),
        "reference_strength": ("FLOAT", {
            "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": "how hard to pull toward reference_image (no effect "
                       "unless it is connected)"}),
    }


_OPT_DEFAULTS = {"denoise_max": 0.0, "latent_blend": 0.0,
                 "flow_backend": "farneback", "reference_strength": 0.35}


def _collect_opts(kw):
    keys = ("guide_size", "max_size", "seed", "steps", "cfg", "sampler_name",
            "scheduler", "denoise", "denoise_max", "noise_mode",
            "detail_mode", "mask_dilation", "feather", "temporal_strength",
            "flow_strength", "latent_blend", "flow_bidirectional",
            "flow_backend", "color_match", "chunk_size", "detail_every",
            "reference_strength")
    return {k: kw.get(k, _OPT_DEFAULTS.get(k)) if k in _OPT_DEFAULTS
            else kw[k] for k in keys}


def _resolve_conditioning(clip, positive, negative,
                          positive_text, negative_text):
    if positive is None:
        positive = encode_text(clip, positive_text)
    if negative is None:
        negative = encode_text(clip, negative_text)
    return positive, negative


# --------------------------------------------------------- detection driver

def run_detect_track(image, detector, det_threshold, min_face_size, max_faces,
                     detector_device, iou_threshold, max_track_gap,
                     min_track_length, crop_factor, crop_smoothing,
                     crop_anchor="landmarks", align_rotation=False):
    n, h, w = image.shape[0], image.shape[1], image.shape[2]
    det = FaceDetector(backend=detector, device=detector_device,
                       det_threshold=det_threshold,
                       min_face_size=min_face_size, max_faces=max_faces)
    pbar = comfy.utils.ProgressBar(n)
    dets_per_frame = []
    for i in range(n):
        comfy.model_management.throw_exception_if_processing_interrupted()
        dets_per_frame.append(det.detect(tensor_frame_to_np(image[i])))
        pbar.update(1)

    tracks = build_tracks(dets_per_frame, iou_threshold=iou_threshold,
                          max_gap=max_track_gap,
                          min_track_length=min_track_length)
    face_tracks = tracks_to_face_tracks(tracks, w, h, n,
                                        crop_factor, crop_smoothing,
                                        crop_anchor=crop_anchor,
                                        align_rotation=align_rotation)
    print(f"[TemporalFaceDetailer] {len(face_tracks['tracks'])} track(s) "
          f"across {n} frames (detector: {det.backend})")
    return face_tracks


# ------------------------------------------------------------------- nodes

class TemporalFaceDetailer:
    """All-in-one: detect + track + detail + temporal smooth + paste back."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE", {"tooltip": "video frame batch"}),
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "positive_text": ("STRING", {
                "multiline": True,
                "default": "detailed face, sharp eyes, high quality skin "
                           "texture"}),
            "negative_text": ("STRING", {
                "multiline": True,
                "default": "blurry, deformed, low quality"}),
        }
        required.update(_detect_inputs())
        required.update(_detail_inputs())
        return {"required": required, "optional": _optional_cond_inputs()}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "FACE_TRACKS")
    RETURN_NAMES = ("image", "face_masks", "debug_overlay", "face_tracks")
    FUNCTION = "detail"
    CATEGORY = CATEGORY

    def detail(self, image, model, clip, vae, positive_text, negative_text,
               positive=None, negative=None, lora_stack=None,
               track_prompts="", reference_image=None, **kw):
        face_tracks = run_detect_track(
            image, kw["detector"], kw["det_threshold"], kw["min_face_size"],
            kw["max_faces"], kw["detector_device"], kw["iou_threshold"],
            kw["max_track_gap"], kw["min_track_length"], kw["crop_factor"],
            kw["crop_smoothing"], kw["crop_anchor"], kw["align_rotation"])

        positive, negative = _resolve_conditioning(
            clip, positive, negative, positive_text, negative_text)

        out, masks = detail_tracks(
            image, face_tracks, model, clip, vae, positive, negative,
            _collect_opts(kw), lora_stack=lora_stack,
            track_prompt_overrides=parse_track_prompts(track_prompts),
            reference_image=reference_image)

        debug = draw_debug_overlay(image, face_tracks)
        return (out, masks, debug, face_tracks)


class FaceDetectTrack:
    """Detect faces per frame and link them into stable identity tracks."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {"image": ("IMAGE",)}
        required.update(_detect_inputs())
        return {"required": required}

    RETURN_TYPES = ("FACE_TRACKS", "IMAGE")
    RETURN_NAMES = ("face_tracks", "debug_overlay")
    FUNCTION = "detect"
    CATEGORY = CATEGORY

    def detect(self, image, **kw):
        face_tracks = run_detect_track(
            image, kw["detector"], kw["det_threshold"], kw["min_face_size"],
            kw["max_faces"], kw["detector_device"], kw["iou_threshold"],
            kw["max_track_gap"], kw["min_track_length"], kw["crop_factor"],
            kw["crop_smoothing"], kw["crop_anchor"], kw["align_rotation"])
        return (face_tracks, draw_debug_overlay(image, face_tracks))


class TrackedFaceDetail:
    """SDXL-detail every tracked face with temporal stabilization."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "face_tracks": ("FACE_TRACKS",),
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "positive_text": ("STRING", {
                "multiline": True,
                "default": "detailed face, sharp eyes, high quality skin "
                           "texture"}),
            "negative_text": ("STRING", {
                "multiline": True,
                "default": "blurry, deformed, low quality"}),
        }
        required.update(_detail_inputs())
        return {"required": required, "optional": _optional_cond_inputs()}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "face_masks")
    FUNCTION = "detail"
    CATEGORY = CATEGORY

    def detail(self, image, face_tracks, model, clip, vae, positive_text,
               negative_text, positive=None, negative=None, lora_stack=None,
               track_prompts="", reference_image=None, **kw):
        positive, negative = _resolve_conditioning(
            clip, positive, negative, positive_text, negative_text)
        out, masks = detail_tracks(
            image, face_tracks, model, clip, vae, positive, negative,
            _collect_opts(kw), lora_stack=lora_stack,
            track_prompt_overrides=parse_track_prompts(track_prompts),
            reference_image=reference_image)
        return (out, masks)


class TemporalSmooth:
    """Standalone flow-guided anti-flicker blend for any frame sequence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                   "step": 0.05}),
            "bidirectional": ("BOOLEAN", {"default": True}),
            "flow_backend": (FLOW_BACKENDS, {"default": "farneback"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "smooth"
    CATEGORY = CATEGORY

    def smooth(self, image, strength, bidirectional, flow_backend="farneback"):
        if image.shape[0] < 2 or strength <= 0.0:
            return (image,)
        frames = [tensor_frame_to_np_f32(image[i])
                  for i in range(image.shape[0])]
        out = flow_blend_sequence(frames, frames, strength,
                                  bidirectional=bidirectional,
                                  flow_fn=get_flow_fn(flow_backend))
        return (np_to_tensor(np.stack(out, axis=0)),)


class FaceTrackPreview:
    """Visualize tracks: bboxes, stabilized crops, landmarks, IDs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "face_tracks": ("FACE_TRACKS",),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    CATEGORY = CATEGORY

    def preview(self, image, face_tracks):
        return (draw_debug_overlay(image, face_tracks),)


class TFDLoRAStack:
    """Stack face LoRAs to apply inside the detailer (chainable)."""

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (loras,),
                "model_weight": ("FLOAT", {"default": 1.0, "min": -10.0,
                                           "max": 10.0, "step": 0.01}),
                "clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0,
                                          "max": 10.0, "step": 0.01}),
            },
            "optional": {"lora_stack": ("LORA_STACK",)},
        }

    RETURN_TYPES = ("LORA_STACK",)
    FUNCTION = "stack"
    CATEGORY = CATEGORY

    def stack(self, lora_name, model_weight, clip_weight, lora_stack=None):
        stack = list(lora_stack) if lora_stack else []
        if lora_name != "None":
            stack.append((lora_name, model_weight, clip_weight))
        return (stack,)


NODE_CLASS_MAPPINGS = {
    "TemporalFaceDetailer": TemporalFaceDetailer,
    "FaceDetectTrack": FaceDetectTrack,
    "TrackedFaceDetail": TrackedFaceDetail,
    "TemporalSmooth": TemporalSmooth,
    "FaceTrackPreview": FaceTrackPreview,
    "TFDLoRAStack": TFDLoRAStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TemporalFaceDetailer": "Temporal Face Detailer (SDXL video)",
    "FaceDetectTrack": "Face Detect + Track",
    "TrackedFaceDetail": "Tracked Face Detail (SDXL)",
    "TemporalSmooth": "Temporal Smooth (flow blend)",
    "FaceTrackPreview": "Face Track Preview",
    "TFDLoRAStack": "TFD LoRA Stack",
}
