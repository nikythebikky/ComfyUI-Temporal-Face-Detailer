"""Core per-track SDXL detailing engine shared by all nodes.

For each tracked face: stabilized crop -> resize to guide size ->
VAE encode -> KSampler img2img/inpaint with a *fixed per-track seed and
noise* -> decode -> color match -> flow-guided temporal blend ->
feathered, mask-limited paste-back.

VRAM discipline (fits a 22 GB 2080 Ti): only small face crops ever touch
the sampler/VAE, work is chunked ``chunk_size`` frames at a time, and the
cache is flushed between tracks. Attention is whatever ComfyUI is
configured with — plain sdpa works; nothing here needs SageAttention,
fp8, or any SM75-incompatible path.
"""

import cv2
import numpy as np
import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths

from .compositing import accumulate_mask, face_mask_for_crop, paste_crop
from .temporal import flow_blend_sequence, match_color, propagate_keyframe
from .utils import round_to_multiple, tensor_frame_to_np_f32


# --------------------------------------------------------------------- setup

def apply_lora_stack(model, clip, lora_stack):
    """Apply a LORA_STACK (list of (name, model_weight, clip_weight))."""
    if not lora_stack:
        return model, clip
    for lora_name, model_weight, clip_weight in lora_stack:
        if not lora_name or lora_name == "None":
            continue
        path = folder_paths.get_full_path("loras", lora_name)
        if path is None:
            raise FileNotFoundError(f"LoRA not found: {lora_name}")
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(
            model, clip, lora, float(model_weight), float(clip_weight))
    return model, clip


def encode_text(clip, text):
    tokens = clip.tokenize(text)
    if hasattr(clip, "encode_from_tokens_scheduled"):
        return clip.encode_from_tokens_scheduled(tokens)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def parse_track_prompts(text):
    """Parse per-track prompt overrides: one ``track_id: prompt`` per line."""
    overrides = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        tid, _, prompt = line.partition(":")
        try:
            overrides[int(tid.strip())] = prompt.strip()
        except ValueError:
            continue
    return overrides


# ------------------------------------------------------------------ sampling

def _resize(img_f32, w, h, up):
    interp = cv2.INTER_LANCZOS4 if up else cv2.INTER_AREA
    return cv2.resize(img_f32, (w, h), interpolation=interp)


def _sample_chunk(model, vae, positive, negative, pixels_f32, latent_masks,
                  seed, opts):
    """Sample one chunk of crops. pixels_f32: (b, S, S, 3) numpy in 0..1."""
    pixels = torch.from_numpy(np.ascontiguousarray(pixels_f32))
    latent = vae.encode(pixels)
    b = latent.shape[0]

    if opts["noise_mode"] == "fixed_per_track":
        # identical noise on every frame of the track — the main
        # identity-stability lever: sampling can't diverge frame-to-frame
        noise = comfy.sample.prepare_noise(latent[:1], seed).repeat(b, 1, 1, 1)
    else:
        noise = torch.cat([
            comfy.sample.prepare_noise(latent[:1], seed + int(fi))
            for fi in opts["_chunk_frame_idxs"]], dim=0)

    noise_mask = None
    if opts["detail_mode"] == "inpaint" and latent_masks is not None:
        lh, lw = latent.shape[-2], latent.shape[-1]
        ms = [cv2.resize(m, (lw, lh), interpolation=cv2.INTER_AREA)
              for m in latent_masks]
        noise_mask = torch.from_numpy(
            np.stack(ms, axis=0)[:, None, :, :].astype(np.float32))

    samples = comfy.sample.sample(
        model, noise, opts["steps"], opts["cfg"], opts["sampler_name"],
        opts["scheduler"], positive, negative, latent,
        denoise=opts["denoise"], noise_mask=noise_mask, seed=seed)

    decoded = vae.decode(samples)
    if decoded.ndim == 5:  # some VAEs return (b, t, h, w, c)
        decoded = decoded.reshape(-1, *decoded.shape[-3:])
    return decoded.cpu().float().numpy()


def _detail_track(images, out_images, out_masks, track, model, vae,
                  positive, negative, opts, pbar):
    idxs = sorted(int(k) for k in track["frames"].keys())
    entries = {int(k): v for k, v in track["frames"].items()}
    if not idxs:
        return

    x1, y1, x2, y2 = entries[idxs[0]]["crop"]
    cs = x2 - x1  # constant per track
    S = round_to_multiple(
        min(max(int(opts["guide_size"]), 256), int(opts["max_size"])), 8)
    seed = int(opts["seed"]) + int(track.get("seed_offset", 0))

    every = max(1, int(opts["detail_every"]))
    key_idxs = idxs[::every]
    if idxs[-1] not in key_idxs:
        key_idxs.append(idxs[-1])
    key_set = set(key_idxs)

    # gather originals and paste masks (all at the track's crop size)
    orig = {}
    masks = {}
    for i in idxs:
        e = entries[i]
        cx1, cy1, cx2, cy2 = e["crop"]
        orig[i] = tensor_frame_to_np_f32(images[i][cy1:cy2, cx1:cx2])
        masks[i] = face_mask_for_crop(e, cs, cs,
                                      dilation=int(opts["mask_dilation"]),
                                      feather=int(opts["feather"]))

    # ---- sample keyframes in VRAM-bounded chunks
    detailed = {}
    chunk = max(1, int(opts["chunk_size"]))
    if opts["denoise"] <= 0.0:
        detailed = {i: orig[i].copy() for i in key_idxs}
    else:
        for c0 in range(0, len(key_idxs), chunk):
            comfy.model_management.throw_exception_if_processing_interrupted()
            batch_idxs = key_idxs[c0:c0 + chunk]
            pixels = np.stack([_resize(orig[i], S, S, S > cs)
                               for i in batch_idxs], axis=0)
            lat_masks = [masks[i] for i in batch_idxs]
            opts["_chunk_frame_idxs"] = batch_idxs
            dec = _sample_chunk(model, vae, positive, negative, pixels,
                                lat_masks, seed, opts)
            for j, i in enumerate(batch_idxs):
                detailed[i] = _resize(
                    np.clip(dec[j], 0.0, 1.0).astype(np.float32),
                    cs, cs, cs > S)
            pbar.update(len(batch_idxs))

    # ---- skip-N keyframe mode: flow-propagate detail to in-between frames
    for i in idxs:
        if i in key_set:
            continue
        prev_k = max((k for k in key_idxs if k < i), default=None)
        next_k = min((k for k in key_idxs if k > i), default=None)
        cands = []
        for k in (prev_k, next_k):
            if k is not None:
                warped, conf = propagate_keyframe(detailed[k], orig[k], orig[i])
                dist = abs(i - k)
                cands.append((dist, warped, conf))
        if len(cands) == 2:
            (d0, w0, c0_), (d1, w1, c1_) = cands
            t = d0 / max(d0 + d1, 1)
            warped = w0 * (1 - t) + w1 * t
            conf = c0_ * (1 - t) + c1_ * t
        else:
            _, warped, conf = cands[0]
        # low-confidence (occluded/mismatched) pixels fall back to source
        detailed[i] = (warped * conf + orig[i] * (1.0 - conf)).astype(np.float32)

    # ---- anti-flicker post: color match + flow-guided temporal blend
    color_s = float(opts["color_match"]) * float(opts["temporal_strength"])
    if color_s > 0.0:
        for i in idxs:
            detailed[i] = match_color(detailed[i], orig[i], color_s,
                                      mask=masks[i])

    flow_s = float(opts["flow_strength"]) * float(opts["temporal_strength"])
    if flow_s > 0.0 and len(idxs) > 1:
        seq = flow_blend_sequence([detailed[i] for i in idxs],
                                  [orig[i] for i in idxs],
                                  flow_s,
                                  bidirectional=bool(opts["flow_bidirectional"]))
        detailed = {i: seq[k] for k, i in enumerate(idxs)}

    # ---- feathered paste-back, mask output
    for i in idxs:
        e = entries[i]
        box = e["crop"]
        m = masks[i]
        frame = out_images[i].numpy()  # shares storage; in-place paste
        paste_crop(frame, np.clip(detailed[i], 0.0, 1.0), m, box)
        accumulate_mask(out_masks[i].numpy(), m, box)


def detail_tracks(images, face_tracks, model, clip, vae,
                  positive, negative, opts, lora_stack=None,
                  track_prompt_overrides=None):
    """Run the full detailing pass. Returns (IMAGE batch, MASK batch).

    images: (N, H, W, C) float tensor. Frames without tracked faces pass
    through untouched; frame count and order are preserved exactly.
    """
    model, clip = apply_lora_stack(model, clip, lora_stack)

    out_images = images.clone().cpu()
    n, h, w = images.shape[0], images.shape[1], images.shape[2]
    out_masks = torch.zeros((n, h, w), dtype=torch.float32)

    # larger (closer) faces composite last so they win overlaps
    def mean_area(tr):
        boxes = [e["bbox"] for e in tr["frames"].values()]
        return float(np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])) \
            if boxes else 0.0
    tracks = sorted(face_tracks.get("tracks", []), key=mean_area)

    overrides = track_prompt_overrides or {}
    every = max(1, int(opts["detail_every"]))
    total = sum(len(tr["frames"]) // every + 1 for tr in tracks)
    pbar = comfy.utils.ProgressBar(max(1, total))

    for tr in tracks:
        comfy.model_management.throw_exception_if_processing_interrupted()
        pos = positive
        override = overrides.get(int(tr["track_id"]))
        if override:
            if clip is None:
                raise ValueError(
                    "track prompt overrides require a CLIP input")
            pos = encode_text(clip, override)
        _detail_track(images, out_images, out_masks, tr, model, vae,
                      pos, negative, opts, pbar)
        comfy.model_management.soft_empty_cache()

    return out_images, out_masks
