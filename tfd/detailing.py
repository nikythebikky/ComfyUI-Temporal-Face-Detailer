"""Core per-track SDXL detailing engine shared by all nodes.

For each tracked face: stabilized (optionally rotation-registered) crop
-> resize to guide size -> VAE encode -> KSampler img2img/inpaint with a
*fixed per-track seed and noise* -> optional latent-space temporal blend
-> decode -> color match (per-frame source + optional reference anchor)
-> flow-guided pixel temporal blend -> feathered, mask-limited paste-back.

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

from .compositing import (accumulate_mask, extract_crop, face_mask_for_crop,
                          paste_crop)
from .temporal import (flow_blend_sequence, free_flow_models, get_flow_fn,
                       latent_blend_sequence, match_color, propagate_keyframe)
from .utils import moving_average, round_to_multiple, tensor_frame_to_np_f32


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


# ---------------------------------------------------------------- reference

def prepare_reference(reference_image, vae, sample_size):
    """Build the identity anchor from a reference image.

    Tries to find the face in the reference (any available detector, CPU);
    falls back to a centered square if none is found. Returns
    (ref_crop_f32 at sample_size, ref_latent cpu tensor).
    """
    img = tensor_frame_to_np_f32(reference_image[0])
    h, w = img.shape[:2]
    box = None
    try:
        from .detection import FaceDetector
        det = FaceDetector(backend="insightface", device="cpu",
                           det_threshold=0.3, min_face_size=16, max_faces=1)
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        found = det.detect(u8)
        if found:
            b = found[0].bbox
            size = int(round(max(b[2] - b[0], b[3] - b[1]) * 1.6))
            size = max(32, min(size, min(w, h)))
            cx = int(round((b[0] + b[2]) * 0.5))
            cy = int(round((b[1] + b[3]) * 0.5))
            x1 = max(0, min(cx - size // 2, w - size))
            y1 = max(0, min(cy - size // 2, h - size))
            box = (x1, y1, x1 + size, y1 + size)
    except Exception as e:
        print(f"[TemporalFaceDetailer] reference face detection failed "
              f"({e}); using center crop.")
    if box is None:
        size = min(w, h)
        x1, y1 = (w - size) // 2, (h - size) // 2
        box = (x1, y1, x1 + size, y1 + size)

    crop = img[box[1]:box[3], box[0]:box[2]]
    crop = cv2.resize(crop, (sample_size, sample_size),
                      interpolation=cv2.INTER_AREA)
    latent = vae.encode(torch.from_numpy(
        np.ascontiguousarray(crop))[None]).cpu()
    return crop, latent


# ------------------------------------------------------------------ sampling

def _resize(img_f32, w, h, up):
    interp = cv2.INTER_LANCZOS4 if up else cv2.INTER_AREA
    return cv2.resize(img_f32, (w, h), interpolation=interp)


def _sample_chunk(model, vae, positive, negative, pixels_f32, latent_masks,
                  seed, denoise, opts, reference):
    """Sample one chunk of crops. pixels_f32: (b, S, S, 3) numpy in 0..1.

    Returns the output *latents* (cpu float32 tensor, (b, C, h, w)).
    """
    pixels = torch.from_numpy(np.ascontiguousarray(pixels_f32))
    latent = vae.encode(pixels)
    b = latent.shape[0]

    ref_strength = float(opts.get("reference_strength", 0.0))
    if reference is not None and ref_strength > 0.0:
        # nudge the init latent toward the reference identity; scaled by
        # denoise so at low denoise (where the init dominates the output
        # and a differently-posed reference would ghost) the nudge stays
        # negligible, and grows with the sampler's freedom to integrate it
        a = min(0.5, ref_strength * float(denoise))
        latent = latent * (1.0 - a) + reference[1].to(latent) * a

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
        denoise=denoise, noise_mask=noise_mask, seed=seed)
    return samples.cpu().float()


def _decode_latents(vae, latents):
    """VAE-decode a (b, C, h, w) tensor -> (b, S, S, 3) float32 numpy."""
    decoded = vae.decode(latents)
    if decoded.ndim == 5:  # some VAEs return (b, t, h, w, c)
        decoded = decoded.reshape(-1, *decoded.shape[-3:])
    return decoded.cpu().float().numpy()


def _denoise_groups(key_idxs, orig, denoise, denoise_max):
    """Group keyframes by adaptive per-frame denoise level.

    Steady frames (low inter-frame motion) keep the base denoise so they
    stay locked; frames with more motion — where WAN/animation output
    typically needs the most correction and flicker is masked by the
    motion itself — get up to denoise_max. Returns [(denoise, [idxs])].
    """
    denoise = float(denoise)
    denoise_max = float(denoise_max)
    if denoise_max <= denoise or len(key_idxs) < 2:
        return [(denoise, list(key_idxs))]

    grays = [cv2.cvtColor(cv2.resize(orig[i], (64, 64),
                                     interpolation=cv2.INTER_AREA),
                          cv2.COLOR_RGB2GRAY) for i in key_idxs]
    diffs = np.zeros(len(key_idxs), dtype=np.float64)
    for k in range(1, len(grays)):
        diffs[k] = float(np.mean(np.abs(grays[k] - grays[k - 1])))
    diffs[0] = diffs[1]
    diffs = moving_average(diffs, 5)
    scores = np.clip(diffs / 0.08, 0.0, 1.0)

    groups = {}
    for i, s in zip(key_idxs, scores):
        eff = denoise + (denoise_max - denoise) * float(s)
        eff = round(eff / 0.05) * 0.05  # quantize so frames share batches
        groups.setdefault(eff, []).append(i)
    return sorted(groups.items())


def _detail_track(images, out_images, out_masks, track, model, vae,
                  positive, negative, opts, sample_size, reference,
                  flow_fn, pbar):
    idxs = sorted(int(k) for k in track["frames"].keys())
    entries = {int(k): v for k, v in track["frames"].items()}
    if not idxs:
        return

    x1, y1, x2, y2 = entries[idxs[0]]["crop"]
    cs = x2 - x1  # constant per track
    S = sample_size
    seed = int(opts["seed"]) + int(track.get("seed_offset", 0))

    every = max(1, int(opts["detail_every"]))
    key_idxs = idxs[::every]
    if idxs[-1] not in key_idxs:
        key_idxs.append(idxs[-1])
    key_set = set(key_idxs)

    # gather originals and paste masks (all at the track's crop size)
    orig = {}
    masks = {}
    angles = {}
    for i in idxs:
        e = entries[i]
        angles[i] = float(e.get("angle", 0.0))
        if abs(angles[i]) < 0.05:
            cx1, cy1, cx2, cy2 = e["crop"]
            orig[i] = tensor_frame_to_np_f32(images[i][cy1:cy2, cx1:cx2])
        else:
            orig[i] = extract_crop(tensor_frame_to_np_f32(images[i]),
                                   e["crop"], angles[i])
        masks[i] = face_mask_for_crop(e, cs, cs,
                                      dilation=int(opts["mask_dilation"]),
                                      feather=int(opts["feather"]))

    # ---- sample keyframes in VRAM-bounded chunks
    detailed = {}
    latent_blend = float(opts.get("latent_blend", 0.0)) \
        * float(opts["temporal_strength"])
    chunk = max(1, int(opts["chunk_size"]))
    if opts["denoise"] <= 0.0:
        detailed = {i: orig[i].copy() for i in key_idxs}
    else:
        latents = {}  # kept only when latent-space blending is on
        groups = _denoise_groups(key_idxs, orig, opts["denoise"],
                                 opts.get("denoise_max", 0.0))
        for denoise, group_idxs in groups:
            for c0 in range(0, len(group_idxs), chunk):
                comfy.model_management.throw_exception_if_processing_interrupted()
                batch_idxs = group_idxs[c0:c0 + chunk]
                pixels = np.stack([_resize(orig[i], S, S, S > cs)
                                   for i in batch_idxs], axis=0)
                lat_masks = [masks[i] for i in batch_idxs]
                opts["_chunk_frame_idxs"] = batch_idxs
                out_lat = _sample_chunk(model, vae, positive, negative,
                                        pixels, lat_masks, seed, denoise,
                                        opts, reference)
                if latent_blend > 0.0:
                    for j, i in enumerate(batch_idxs):
                        latents[i] = out_lat[j].numpy()
                else:
                    dec = _decode_latents(vae, out_lat)
                    for j, i in enumerate(batch_idxs):
                        detailed[i] = _resize(
                            np.clip(dec[j], 0.0, 1.0).astype(np.float32),
                            cs, cs, cs > S)
                pbar.update(len(batch_idxs))

        if latent_blend > 0.0:
            # temporal EMA in latent space, then decode
            seq = latent_blend_sequence(
                [latents[i] for i in key_idxs],
                [orig[i] for i in key_idxs],
                latent_blend,
                bidirectional=bool(opts["flow_bidirectional"]),
                flow_fn=flow_fn)
            blended = dict(zip(key_idxs, seq))
            for c0 in range(0, len(key_idxs), chunk):
                comfy.model_management.throw_exception_if_processing_interrupted()
                batch_idxs = key_idxs[c0:c0 + chunk]
                lat = torch.from_numpy(np.stack(
                    [blended[i] for i in batch_idxs], axis=0))
                dec = _decode_latents(vae, lat)
                for j, i in enumerate(batch_idxs):
                    detailed[i] = _resize(
                        np.clip(dec[j], 0.0, 1.0).astype(np.float32),
                        cs, cs, cs > S)

    # ---- skip-N keyframe mode: flow-propagate detail to in-between frames
    for i in idxs:
        if i in key_set:
            continue
        prev_k = max((k for k in key_idxs if k < i), default=None)
        next_k = min((k for k in key_idxs if k > i), default=None)
        cands = []
        for k in (prev_k, next_k):
            if k is not None:
                warped, conf = propagate_keyframe(detailed[k], orig[k],
                                                  orig[i], flow_fn=flow_fn)
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

    # reference color anchor: pin exposure/tone to the identity reference
    ref_strength = float(opts.get("reference_strength", 0.0))
    if reference is not None and ref_strength > 0.0:
        ref_cs = _resize(reference[0], cs, cs, cs > reference[0].shape[0])
        for i in idxs:
            detailed[i] = match_color(detailed[i], ref_cs,
                                      0.5 * ref_strength, mask=masks[i])

    flow_s = float(opts["flow_strength"]) * float(opts["temporal_strength"])
    if flow_s > 0.0 and len(idxs) > 1:
        seq = flow_blend_sequence([detailed[i] for i in idxs],
                                  [orig[i] for i in idxs],
                                  flow_s,
                                  bidirectional=bool(opts["flow_bidirectional"]),
                                  flow_fn=flow_fn)
        detailed = {i: seq[k] for k, i in enumerate(idxs)}

    # ---- feathered paste-back, mask output
    for i in idxs:
        e = entries[i]
        box = e["crop"]
        m = masks[i]
        frame = out_images[i].numpy()  # shares storage; in-place paste
        paste_crop(frame, np.clip(detailed[i], 0.0, 1.0), m, box, angles[i])
        accumulate_mask(out_masks[i].numpy(), m, box, angles[i])


def detail_tracks(images, face_tracks, model, clip, vae,
                  positive, negative, opts, lora_stack=None,
                  track_prompt_overrides=None, reference_image=None):
    """Run the full detailing pass. Returns (IMAGE batch, MASK batch).

    images: (N, H, W, C) float tensor. Frames without tracked faces pass
    through untouched; frame count and order are preserved exactly.
    """
    model, clip = apply_lora_stack(model, clip, lora_stack)

    out_images = images.clone().cpu()
    n, h, w = images.shape[0], images.shape[1], images.shape[2]
    out_masks = torch.zeros((n, h, w), dtype=torch.float32)

    sample_size = round_to_multiple(
        min(max(int(opts["guide_size"]), 256), int(opts["max_size"])), 8)

    reference = None
    if reference_image is not None \
            and float(opts.get("reference_strength", 0.0)) > 0.0:
        reference = prepare_reference(reference_image, vae, sample_size)

    flow_fn = get_flow_fn(opts.get("flow_backend", "farneback"))

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

    try:
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
                          pos, negative, opts, sample_size, reference,
                          flow_fn, pbar)
            comfy.model_management.soft_empty_cache()
    finally:
        free_flow_models()

    return out_images, out_masks
