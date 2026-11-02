---
name: render-turntable
description: Render a ray traced turntable film (MP4) of a model with KoltinRayTracer's CUDA renderer - the model turning in place while the camera, lamps and room stand still - e.g. Yennefer in StlViewer's gold studio (scene studio-model). Use when the user asks for a turntable / "film obrotowy" / "obrót" / "wyrenderuj obrót X" in this repo, with a length, number of turns, fps, samples or quality, or wants it compared with StlViewer's film of the same model.
---

# Rendering a turntable film

The film comes from the renderer itself: `-Drt.turntable=<file.mp4>` on any scene turns the
scene's objects (its parts that are neither lights nor walls) about the middle of their
footprint, one frame per step, denoises each frame (OIDN) and pipes it to ffmpeg. What the
code does and why: `AGENTS.md`, section *Turntable Films*.

## Before starting

- **CUDA renders it.** It needs an NVIDIA card and driver; the kernel compiler (NVRTC) ships in
  `native/cuda/windows-x64`, so run from the repository root (or point `-Drt.nvrtc=<dir>` at it).
  "GPU rendering unavailable" or "NVRTC not found" means one of those is missing - there is no
  CPU or OpenCL fallback any more.
- **The GPU is exclusive.** A film keeps the GTX 1070 busy for ~24 min. If the StlViewer session
  (`stl-viewer` in ListAgents) may be rendering, tell it before you start and send "GPU wolne"
  when done; two renders at once spoil both timings and risk a driver reset (TDR).
- **ffmpeg** must run: `ffmpeg -version` (scoop: `~/scoop/shims/ffmpeg.exe`). Elsewhere point
  `-Drt.ffmpeg=<path>` or `FFMPEG` at it. Without it the film stops with that message.
- **A new look gets a still first.** For a scene or setting the user has not seen yet, render one
  frame (the same command without `-Drt.turntable`, it writes PPM + PNG + `-denoised`) and send it
  (SendUserFile) before spending 24 min on the film. Skip this when the user asks for the film
  directly or the look is already approved.

## Where the models come from

The STL models repository `domer-sz/models-to-render` is checked out (14 GB) inside StlViewer:
`C:/Users/szymondomeracki/Documents/projects/StlViewer/models-to-render` - a working directory of
this session (`.claude/settings.local.json`), so its files can be read directly. Use it where it
is; don't clone it again and don't commit to it (it belongs to the StlViewer work).

- `models/<nr>_<character>/` holds each character's recommended model (95 of 99) at full
  resolution, z-up (studio-model and cornell-model stand STL/3MF up from z), plus `INFO.md`:
  author, licence, "Licencja – co wolno", the ready attribution line, triangle counts.
- `MODELE.md` is the table of all characters, licences (🟢 commercial OK, 🟡 no modifications,
  🟠/🔴 non-commercial only, ⚪ personal use) and alternatives; `data/models.json` the same for
  scripts.
- Files over 99 MiB are zipped (`*.stl.zip`, sometimes split `.001`, `.002`): unpack them with
  `python3 tools/unpack_models.py` run in that repository (the unpacked files are git-ignored
  there) - or ask before touching it.
- Check the licence before a model goes into anything published: NC / NC-ND / personal-use
  models are not for a Reel; CC-BY and CC-BY-ND need the attribution line from `INFO.md`.

## The command

Run it in the background (`run_in_background`) - it prints a `\r` progress line, so log through
`tr`, line-buffered with `stdbuf -oL` (Git Bash has it): a plain `tr` into a file buffers 4 KB
blocks, and on 2026-11-02 that left the log empty for the whole 24-minute film:

```
mvn -q -o compile
mvn -q -o exec:java -Drt.scene=studio-model \
  -Drt.model="C:/Users/szymondomeracki/Documents/projects/StlViewer/models-to-render/models/003_yennefer/single_color_19.stl" \
  -Drt.samples=64 \
  -Drt.turntable=renders/003-yennefer-gold-koltin-max.mp4 \
  -Drt.turntable.seconds=15 -Drt.turntable.fps=30 -Drt.turntable.turns=1 -Drt.turntable.quality=max \
  2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v '^\s*$' > renders/003-yennefer-gold-koltin-max.log
```

If the log stays empty anyway, the film is still running when `nvidia-smi` shows the GPU busy
and `renders/<name>.part.mp4` grows; a finished film is ~17 MB, so its size gives a rough
percentage.

**A silent "tiktok" copy** (the user's standing request for every film): after the film,
`ffmpeg -v error -y -i renders/<name>.mp4 -map 0:v -c copy -an -movflags +faststart renders/<name>-tiktok.mp4`
- the picture copied as it is, no audio track (TikTok's own music is added in the app).

Films and their logs go to `renders/` (untracked; never commit them). Name them after the model
folder, like StlViewer does: `renders/003-yennefer-...mp4`.

While it runs, report progress every 5 minutes without being asked: arm a Monitor (load it with
ToolSearch `select:Monitor`, `timeout_ms` 1800000, re-arm on expiry) that every 300 s prints the
last `[film] frame N of M, X s a frame, about T s left` line of the log, and also the end and
failures (`Film saved|Exception|Error|BUILD FAILURE`) - silence must not look like progress. The
log fills line by line with `stdbuf -oL`; without it, it fills in blocks. If no event has arrived after
about 6 minutes, read the last `[film]` line of the log yourself and report it - a quiet
monitor has happened before.

## Settings

| Property | Default | Meaning |
|---|---|---|
| `-Drt.scene` | – | `studio-model` = StlViewer's LOOK=gold-reel studio; `model`, `cornell-model` or any book scene also work |
| `-Drt.model` | `model.stl` | model file (STL/3MF taken as z-up in studio-model and cornell-model) |
| `-Drt.turntable` | – | output MP4; switches the film on |
| `-Drt.turntable.seconds` | 15 | length of one turn (1-60) |
| `-Drt.turntable.turns` | 1 | turns in the film (1-10); one is rendered, the rest are copies (free) |
| `-Drt.turntable.fps` | 30 | frames a second; frames of a turn = round(seconds * fps) |
| `-Drt.turntable.reverse` | off | clockwise seen from above instead of counter-clockwise |
| `-Drt.turntable.quality` | small | `max`: 0.15 bit per pixel and frame (~9.3 Mbit/s at 1080x1920@30); small: 0.085 (~5.3) |
| `-Drt.turntable.parts` | all objects | `a;b`: turn only the scene parts with these names (studio-model's figure is `Figura`) |
| `-Drt.samples` | scene's (64 in studio-model) | samples per pixel per frame; time scales with it |
| `-Drt.width` / `-Drt.aspect` | scene's (1080, 9:16) | frame size; must come out even |
| `-Drt.denoise=off` | on | raw frames |
| `-Drt.studio.turn` | 0 | studio-model: the figure's starting angle (degrees) |

studio-model is StlViewer's gold-reel scene at angle 0 with its own dumped numbers (gold metal
fuzz 0.05, black glossy floor fuzz 0.004, navy wall, key/fill/rim softboxes 8/4/12, flat
#0a1020 sky, -0.4 EV, depth 8, DoF 0.4 degrees, 1080x1920, 45 degree FOV). StlViewer turns its
camera with the lamps and wall; turning the figure looks the same.

In the viewer (`-Drt.gui`), the panel's *Film obrotowy…* button does the same with a dialog;
`-Drt.gui.film=<file.mp4>` (with the `-Drt.turntable.*` settings) starts it without clicks.

## Check and hand over

1. `ffprobe -v error -show_entries stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_frames:format=duration,size,bit_rate -of default=nw=1 <film>`
   - expect h264 High, yuv420p, 30/1, frames = turns x round(seconds x fps), duration as asked.
2. Look at it: a strip of every 30th frame,
   `ffmpeg -v error -y -i <film> -vf "select='not(mod(n\,30))',scale=120:-1,tile=8x2" -frames:v 1 <strip.png>`,
   then Read the PNG - the figure must turn in place, lamps and backdrop still.
3. Tell the user where it is and send it (SendUserFile) when they may be on another device;
   tell the StlViewer session "GPU wolne" if you reserved the card.

To check the renderer itself, compare a raw still (the command without `-Drt.turntable`, with
`-Drt.denoise=off -Drt.seed=<n>`) against the OpenCL stills in `reference/turntable-opencl`
(`still-64spp-seed1.png`, `-seed2.png`): `ffmpeg -i <still.png> -i <reference.png> -lavfi
"[0][1]psnr" -f null -`. Two OpenCL seeds sit 27.93 dB / SSIM 0.847 apart, and CUDA lands there
too; clearly lower means the image changed, not just its noise.

To compare with StlViewer's film, export frames 0 and 225 (`-vf "select=eq(n\,0)"`) and ask
the StlViewer session to compute PSNR/SSIM against frames decoded from its own film,
`renders/v4-film-frame0-decoded.png` / `v4-film-frame225-decoded.png` in its repository (that
folder is outside this one; reading it needs the user's permission). Not against
`003-yennefer-gold-v4-max-frame.png`: StlViewer confirmed that one is another angle, not frame 0.
Expect a systematic gap, not noise: frame 0 came out at 25.8 dB / SSIM 0.887 and frame 225 at
24.5 / 0.879, against ~32.7 / 0.90 for StlViewer's film vs its own 2048-spp reference.
StlViewer smooths vertex normals and clamps fireflies at 16; this renderer uses flat facet
normals and no clamp.

## Measured (GTX 1070, Yennefer 717 592 triangles, 1080x1920, 64 spp)

- CUDA (the renderer now), 2026-11-02: a 30-frame film with the command's settings took 8.8 s
  for the first frame (model tree and kernels built) and then 3.0-3.2 s a frame (frames 10-30).
  Full film, same day: Master Chief CC0 (1 184 002 triangles), 450 frames at 3.18 s = 23.9 min,
  17.1 MB, 9.14 Mbit/s,
  so 450 frames take ~24 min. A still on its own traces in 3.27 s; at 256 spp 11.7 s of GPU time.
- Film frame: tracing + 32-spp guides + scene rebuild (~0.25 s flatten, ~0.17 s upload) on the
  render thread; denoise (1.5 s, OIDN on the CPU) and encoding overlap the next frame.
- The OpenCL backend this replaced (last in commit `1b65ee9`), for comparison: still tracing 2.66 s;
  2026-11-02, the command above (Yennefer, studio-model, 15 s, 30 fps, 1 turn, max): 450 frames
  at 3.48 s = 26 min; 17.06 MB, 9.10 Mbit/s, H.264 High, BT.709 tags. That film and its frames are
  the reference in `reference/turntable-opencl`.
- StlViewer's own film of the same scene: 13.6 min for 450 frames, 18.0 MB (9.6 Mbit/s).
