# Changelog

## v1.0.0 - 2026-11-04

First release.

- CUDA is now the only renderer. The kernel is CUDA C++ (`raytracer.cu`), compiled at run time by
  NVRTC for the card found (sm_61 on a GTX 1070); no CUDA Toolkit is needed. The OpenCL backend and
  the CPU path tracer are gone, so without a CUDA device a render fails with the reason. On a GTX 1070
  CUDA traces about 15% slower than OpenCL did (Yennefer at 256 spp: 11.7 s vs 10.1 s). The CPU renders
  the GPU tests compare against are kept as frozen references.
- Turntable films: `-Drt.turntable=film.mp4` renders a model turning in place under still camera and
  lamps, and the viewer's "Film obrotowy" button does the same from a dialog. Frames are traced in
  ~100 ms slices, denoised and encoded to H.264 through ffmpeg while the GPU traces the next one.
- `studio-model` scene, the gold-reel studio of StlViewer, for comparing the two renderers; moving a
  placed mesh is folded into its placement, so turning one costs a tree copy, not a rebuild.
- The viewer denoises and draws beside the sampling, keeps the OIDN device between calls, and swaps the
  materials of placed models and of the Cornell box walls.
- Meshes in STL, OBJ, PLY, OFF, glTF/GLB and 3MF, with per-face colours from OBJ material libraries.
- Skills for Claude Code: `render-turntable`, `publish-reel` (Instagram Reels through the Graph API,
  with music), `crp` (this release flow). AGENTS.md carries the orchestration rules.
