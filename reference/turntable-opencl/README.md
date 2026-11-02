# Turntable reference - OpenCL renderer

The 15 s Yennefer turntable rendered by the OpenCL backend on 2026-11-02, kept to compare any
later backend (CUDA) against: same scene, same settings, same frames.

| File | What |
|---|---|
| `003-yennefer-gold-koltin-max.mp4` | the film: 1080x1920, 30 fps, 450 frames, 15.000 s, H.264 High, yuv420p, BT.709 tags, 17.06 MB, 9.10 Mbit/s |
| `003-yennefer-gold-koltin-max-frame0.png` | frame 0, decoded from the film |
| `003-yennefer-gold-koltin-max-frame225.png` | frame 225 (half a turn), decoded from the film |
| `003-yennefer-gold-koltin-max-strip.png` | every 30th frame |
| `003-yennefer-gold-koltin-max.log` | the render's log |

## How it was made

Branch `gpu-max-perf`, renderer as of the turntable work on top of `1089a3e` (not yet committed
when rendered), GTX 1070 (driver 577.00), OIDN 2.5.1 on the CPU:

```
mvn -q -o exec:java -Drt.scene=studio-model \
  -Drt.model="C:/Users/szymondomeracki/Documents/projects/StlViewer/models-to-render/models/003_yennefer/single_color_19.stl" \
  -Drt.renderer=gpu -Drt.samples=64 \
  -Drt.turntable=renders/003-yennefer-gold-koltin-max.mp4 \
  -Drt.turntable.seconds=15 -Drt.turntable.fps=30 -Drt.turntable.turns=1 -Drt.turntable.quality=max
```

studio-model is StlViewer's LOOK=gold-reel studio at angle 0 (its numbers dumped by StlViewer):
gold metal fuzz 0.05, black glossy floor fuzz 0.004, navy wall, key/fill/rim softboxes 8/4/12,
flat #0a1020 sky, -0.4 EV, depth 8, DoF 0.4 degrees, 45 degree vertical FOV. The figure turns
counter-clockwise seen from above; each frame has its own seed (session seed + frame), 32-spp
denoiser guides, denoised by OIDN's RT filter.

Time: 3.48 s a frame, 26 min in all (tracing ~2.7 s, guides ~0.07 s, scene rebuild ~0.4 s;
denoising overlapped with the next frame).

## Comparing

Frames carry the per-frame noise, so compare statistically (PSNR/SSIM, mean colour), never byte
for byte. A new backend rendering the same command should land at the noise level of two OpenCL
runs; anything well below that is a real difference.

## Licence of what is shown

The model is "The Witcher - Yennefer" by LM3Dtisk Lukáš Mlýnek, CC-BY-ND
(https://www.printables.com/model/1673033-the-witcher-yennefer). These renders are kept for
local comparison; do not publish them without that attribution, and never the model itself.
