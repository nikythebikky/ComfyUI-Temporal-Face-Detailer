# ComfyUI-Temporal-Face-Detailer

**Video FaceDetailer for SDXL.** Fixes/enhances faces in a video frame batch —
the video equivalent of Impact Pack's `FaceDetailer` — while suppressing the
temporal **flicker** that naive per-frame detailing produces.

It detects and **tracks faces across frames** (stable identities, stabilized
crops), re-details each tracked face with SDXL img2img/inpaint using a
**fixed per-track seed and noise**, blends results with **optical-flow-guided
temporal smoothing**, and feather-pastes only the face region back — so
backgrounds stay pixel-exact and the output recombines into a video with the
same frame count and order.

```
Load Video → TemporalFaceDetailer → Video Combine
                (frames)               (frames)
```

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nikythebikky/ComfyUI-Temporal-Face-Detailer
pip install -r ComfyUI-Temporal-Face-Detailer/requirements.txt
```

Notes:
- `insightface` (RetinaFace) is the recommended detector; its models download
  automatically into `models/insightface` (shared with ReActor & friends).
  For a CUDA detector install `onnxruntime-gpu` instead of `onnxruntime`.
- If `insightface` isn't installed, the nodes fall back automatically to
  OpenCV **YuNet** (small ONNX model, auto-downloaded to `models/tfd`) and
  finally to a Haar cascade (no downloads, no landmarks).

## Nodes

| Node | Role |
|------|------|
| **Temporal Face Detailer** (all-in-one) | Full pipeline in one node: detect → track → detail → smooth → paste back. Outputs frames, face masks, a debug overlay and the `FACE_TRACKS`. |
| **Face Detect + Track** | Detection + tracking + crop stabilization only. Outputs `FACE_TRACKS` + debug overlay. |
| **Tracked Face Detail (SDXL)** | Details all tracks from a `FACE_TRACKS` input. Outputs frames + masks. |
| **Temporal Smooth (flow blend)** | Standalone flow-guided anti-flicker blend for any `IMAGE` sequence. |
| **Face Track Preview** | Visualize tracks (bboxes, stabilized crop windows, landmarks, IDs). |
| **TFD LoRA Stack** | Chainable `LORA_STACK` builder for face LoRAs applied inside the detailer. |

The all-in-one node and the split path share the same internals — use the
split nodes when you want to inspect/tune tracking separately from detailing
(tracking runs once, so you can iterate on sampler settings cheaply).

### Quick start (all-in-one)

`Load Video → Temporal Face Detailer → Video Combine`. Connect your SDXL
`MODEL`/`CLIP`/`VAE`, type prompts (or feed `CONDITIONING` into the optional
`positive`/`negative` inputs — they take precedence). Defaults are the
**least-flicker preset**.

### Split path

`Face Detect + Track → Tracked Face Detail → (optional Temporal Smooth) → Video Combine`

### LoRAs

Two ways, both supported:
1. Apply LoRAs to `MODEL`/`CLIP` *before* the detailer with regular
   `LoraLoader` nodes.
2. Feed a **TFD LoRA Stack** into the `lora_stack` input; the stack is
   applied internally (convenient for face LoRAs you only want on the crops).

### Per-track prompts

The optional `track_prompts` text box overrides the positive prompt for
specific identities, one per line (get IDs from the debug overlay):

```
0: photo of johndoe person, detailed face
1: photo of janedoe person, detailed face
```

## How flicker is suppressed (layered)

1. **Stabilized tracked crops** — per-track constant crop size + temporally
   smoothed crop center (`crop_smoothing`), so the sampled window doesn't
   jitter. Removes region jitter, and keeps latent shapes constant.
2. **Fixed per-track seed + noise** (`noise_mode: fixed_per_track`) — every
   frame of a track is sampled from the *same* noise, so sampling can't
   diverge frame-to-frame. The biggest identity-stability lever.
3. **Moderate `denoise`** — default 0.35. Higher = more detail but more
   flicker; 0.3–0.45 is the sweet spot.
4. **Flow-guided temporal blend** (`flow_strength`) — Farneback optical flow
   between adjacent *original* crops warps the running result onto each new
   frame and blends it in, occlusion-aware so motion doesn't smear.
5. **Color match** (`color_match`) — pins each detailed crop's mean/std to
   its own source crop, killing brightness pulsing.
6. **Feathered, mask-limited paste-back** (`mask_dilation`, `feather`) — only
   the face changes; boundaries don't crawl.

`temporal_strength` is the master knob scaling levers 4–5; each lever also
has its own control and can be disabled individually (set `flow_strength` /
`color_match` to 0, switch `noise_mode` to `per_frame`, set `crop_smoothing`
to 0) for A/B comparisons against naive per-frame detailing.

## Key parameters

| Param | Default | Meaning |
|-------|---------|---------|
| `denoise` | 0.35 | img2img strength — main quality vs. consistency lever |
| `guide_size` / `max_size` | 768 / 1024 | resolution crops are resampled at |
| `crop_factor` | 1.7 | context around the face bbox |
| `crop_smoothing` | 0.8 | temporal smoothing of the crop window |
| `noise_mode` | fixed_per_track | reuse seed+noise across a track's frames |
| `detail_mode` | img2img | `inpaint` = latent-masked sampling (face only) |
| `max_track_gap` | 10 | frames a face may vanish before its track ends; gaps are interpolated |
| `chunk_size` | 4 | crops sampled per batch — lower it if you OOM |
| `detail_every` | 1 | keyframe mode: sample every Nth frame, flow-propagate the rest (speed) |

## Hardware notes (tested target: 22 GB 2080 Ti, Turing/SM75)

- Uses ComfyUI's standard sampling path with **sdpa / PyTorch
  cross-attention** (`--use-pytorch-cross-attention`). **Do not** enable
  SageAttention on Turing — it silently produces NaN/black frames.
- fp16/bf16 only; **no fp8 requirement** (unsupported on Turing).
- VRAM stays bounded on long clips: only small face crops are encoded,
  sampled and decoded, `chunk_size` frames at a time, per track, with cache
  flushes between tracks. Full frames are never re-encoded.
- `detector_device: cpu` offloads detection if you need every MB of VRAM
  (flow runs on CPU/OpenCV already).
- For long clips, `detail_every: 2-4` samples keyframes only and propagates
  detail via optical flow — a large speedup at a small quality cost.

## FACE_TRACKS type

A plain dict: `{width, height, num_frames, tracks: [{track_id, seed_offset,
frames: {frame_idx: {bbox, kps, score, crop, interpolated}}}]}` — `crop` is
the stabilized square crop, `interpolated` marks occlusion-gap frames filled
by interpolation.

## Acceptance behavior

- Frame count/order always preserved; frames with no tracked face pass
  through byte-identical.
- Multiple simultaneous faces are detailed independently; overlaps composite
  larger-face-last (closer face wins).
- Track IDs persist through brief occlusion (`max_track_gap`), with linear
  bbox/landmark interpolation across the gap.
