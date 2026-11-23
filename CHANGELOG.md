# Changelog

## v2.3.0 - 2026-11-23

- Wave films (`-Drt.turntable.wave=amplitude,wavelength[,lift]`): instead of turning, the placed
  meshes ripple like cloth in a breeze - two waves running across the floor, each going round a
  whole number of times so the film loops. The mesh is only moved, its normals following the
  wave's gradient rather than welded again each frame. `-Drt.turntable.wave.native=on` ripples the
  model a sponge (`-Drt.infill=sponge`) was made from and builds the sponge again every frame, so
  its holes are laid out anew instead of riding on the surface.

## v2.2.0 - 2026-11-22

- Laser scene (`-Drt.scene=laser`): a diode module, a source 0.04 mm across at the focus of the
  Thorlabs 354710-C collimator, sends a beam 6 mm across and 5.5 mrad wide through a smoky room.
  `-Drt.laser.target=diamond|ball|prism|none` stands a round brilliant, a glass ball or a prism
  in the beam, breaking it into a star of beams, focusing it into a cone or bending it onto the
  wall; `-Drt.laser.nm`, `.fog`, `.clouds`, `.room`, `.power`, `.source` tune it (1600 px at
  1024 spp with photons: 35 s).
- Photons stop in fog: with `-Drt.caustics=on` a beam a lens sends through smoke is
  photon-mapped (a ball gather in the volume) where camera paths alone would almost never find
  it. Smoke under a glass slab comes out at 0.884 against 0.886 by brute force, in both engines.
- Real lenses, traced exactly in both engines: spherical, conic and even-asphere surfaces with
  the glasses between them, so cemented doublets bend light glass to glass. A catalog of 258
  Thorlabs and Edmund lenses and 35 glasses, converted from DCC-Lab/RayTracing (MIT), every lens
  focusing within 1% of its quoted back focal length; `.zmx` files from lens makers are read too.
- Dispersion: catalog glass bends each wavelength by its own index, so prisms and singlets split
  white light into colour (`-Drt.dispersion=off` turns it off). Prisms of any cross-section.
- New scenes: `optical-bench` (sunlight through any catalog lens onto a screen, `-Drt.lens`),
  `prism` (Newton's spectrum), `lens-gallery` (every kind side by side, with their caustics).
- Fixed: light sampling towards a target narrower than a float's precision (a small lens
  metres from the sun) collapsed onto one point; the cone is now drawn without cancellation.

## v2.1.0 - 2026-11-21

- Viewer objects take a surface form: a loaded model, the ball and the block can each turn into a
  sponge, pyramids or diamonds from the panel's *Forma powierzchni* box. The converted mesh is
  built once on its own thread and then placed, moved and resized like any model; the form and
  the pores' spacing are saved in the .rtscene.
- Sponge conversion (`-Drt.infill=sponge`): the model's surface becomes a soft, thick foam of
  round uneven holes beside each other, built by the Blender recipe (Voronoi cells, inset holes,
  solidify, smoothing) as a distance field, matched side by side against a reference photo
  (a 0.5 x 0.5 patch: 89 s, 8.5 M triangles).
- Model infills (`-Drt.infill=threads|pyramids|diamonds`): the model is left out and filled with
  swirling bead chains of many thicknesses, or packed pyramids or round brilliants in 7 sizes;
  `-Drt.infill.material=glass` and `.ior` make them glass or diamond.
- Dark-studio options for caustic shots in `studio-model`: a matte floor, a lamp with a cone,
  dimmable or no softboxes, no wall, a camera override and a glass or diamond figure
  (`-Drt.studio.floor|lamp|lamp.cone|softboxes|wall|material`, `-Drt.camera.from|at|vfov`).
- Turntables can turn by less than a full turn (`-Drt.turntable.degrees`) and swing out to that
  angle with easing (`-Drt.turntable.swing=on`), so a there-and-back loop costs half the frames.
- Caustics with tens of thousands of casting spheres (glass beads) start at once: more than 64
  share one aim sphere instead of being merged pair by pair.

## v2.0.0 - 2026-11-18

- Caustics by photon mapping (`-Drt.caustics=on`, off by default): a photon pass in both engines
  shoots photons from the lights through glass and polished metal, the card sorts them into a
  hashed grid, and camera paths gather them at the first matte surface they meet while the path
  tracer gives up the light they carry. At equal time the caustics come out clean where brute
  force is still specks (the fluted tumbler: 1024 spp in 12.5 s against 2048 spp of noise in
  8.4 s), the firefly clamp can stay on, and the floor under a glass slab matches brute force
  within 6 % on both engines.
- Models from files keep their look: texture coordinates and `map_Kd` images, physically based
  metal (GGX, Schlick Fresnel, Smith shadowing, `map_Pr` roughness), `map_Ke` glow, normal maps,
  and glass coloured through its body (`Tf`, Beer-Lambert). Amazon Lumberyard Bistro (2.83 M
  triangles, 132 textures) renders at 1920x1080 and 256 spp in 5.3 s; Sun Temple with its own
  skybox.
- `-Drt.scene=file` renders a whole scene from one file under a sun, with lamps from a `.lights`
  file beside it (their radius settable), an HDR sky (`-Drt.env`, Radiance .hdr) and camera flags;
  every output goes through a tone map (`-Drt.tonemap=aces`, `-Drt.exposure`).
- `-Drt.flythrough` films the camera along key positions (Catmull-Rom, constant speed, eased);
  Sun Temple's 15 s at 60 fps took 0.81 s a frame.
- Films can be denoised with OptiX's temporal model (`-Drt.temporal=on`) along a flow guide for
  turning meshes, and the viewer renders in motion as a film does: 16 spp, temporal upscaling,
  10-12 frames a second on Yennefer at 1080x1920. Stills and films trace at half size by default.
- The viewer picks a model by clicking it, and moving one no longer freezes it (a CPU tree of
  every mesh was rebuilt on each mouse move); `gui.cmd` opens it in a few keystrokes.
- OptiX traces spheres (custom primitives) and fog too, so it takes every scene the CUDA kernel
  does.
- Removed: Open Image Denoise (OptiX denoises, or nothing does; `-Drt.denoiser`, `-Drt.oidn`),
  sharing one image between cards (`-Drt.gpus` becomes `-Drt.gpu=<card>`; one card traces and
  denoises, the GTX 1070 is no longer used), the compiler experiment switches, `-Drt.stl` and the
  numbered scenes.
- Publishing skills from StlViewer: Instagram, YouTube and a TikTok draft in one go
  (`publish-film`), TikTok alone (`publish-tiktok`).

## v1.1.0 - 2026-11-16

- `-Drt.upscale=2`: stills and films trace at half the width and height and OptiX's UPSCALE2X
  model denoises and enlarges in one call. The Yennefer film takes 0.16-0.19 s a frame instead of
  0.50 (450 frames in 1.2 min), with softer fine detail (29.1 dB from the full-size frame). The
  model needs a Volta or newer card: on the GTX 1070 it returns a smeared image without an error,
  so it runs on the RTX 2060, and an upscaled film traces on that card alone.
- Both denoiser guides come from one pass over the first hit, and reads come back through
  page-locked memory: guides 87 -> 70 ms a film frame. Gathering the guides in the image's own
  pass was measured and dropped (it slowed the image by 81 ms a frame).
- Models from files are shaded smooth (area-weighted vertex normals, `-Drt.smooth=off` keeps the
  facets) and every sample is capped at 16 (`-Drt.firefly`), as StlViewer does; a Yennefer frame
  now sits 33 dB from StlViewer's instead of 25.5.
- A turntable turns its figure about the middle of its foot (the lowest 5% of its height), as
  StlViewer does, so a raised arm or sword no longer sets the base wandering;
  `-Drt.studio.height` sizes studio-model's figure.
- OptiX engine (`-Drt.backend=auto|optix|cuda`): on an RTX card the RT cores find the hits,
  through `nvoptix.dll` from the driver called via libffi, with NVIDIA's headers fetched rather
  than committed (`native/optix/fetch.ps1`). On the RTX 2060 it traces Yennefer 3.2 times as fast
  as the CUDA kernel; placed meshes are OptiX instances, so a film frame costs a matrix, not a
  tree copy. Scenes with spheres or fog stay on the CUDA kernel.
- Several cards render one image together (`-Drt.gpus=all|one|<list>`, default all): they share
  its samples by measured speed, and the viewer's panel switches cards while it runs.
- NVIDIA's OptiX denoiser (`-Drt.denoiser=auto|optix|oidn`, `-Drt.denoiser.gpu`): by default on
  the second card while the first traces - a 1080x1920 frame in 0.31 s on the GTX 1070 against
  1.34 s for OIDN on the CPU, and a touch sharper (34.0 dB vs 33.3 against 4096 spp). A denoiser
  that cannot start falls back to OIDN.
- The full Yennefer film at 64 spp: 26 min on the GTX 1070 with OpenCL, 3.5 min on the RTX 2060
  with the GTX 1070 denoising, 1.2 min upscaled.

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
