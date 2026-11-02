# NVIDIA NVRTC

`windows-x64/` holds NVRTC, NVIDIA's runtime compiler for CUDA C++, version 12.6.85 (CUDA
12.6.3), taken unchanged from NVIDIA's redistributable archive
`cuda_nvrtc-windows-x86_64-12.6.85-archive.zip`
(https://developer.download.nvidia.com/compute/cuda/redist/, listed in `redistrib_12.6.3.json`,
sha256 `b1221a95a3758561bc73c56905743730f7459cbaff54a4870aa899fa3489219e`): the compiler and its
builtins library, and the CUDA licence. NVRTC is one of the files the licence's Attachment A
names as redistributable.

| File | sha256 |
|---|---|
| `nvrtc64_120_0.dll` | `01700cdf0f7f566439260201a8b7ee57f914b56f639770f83feb0d1f2a078567` |
| `nvrtc-builtins64_126.dll` | `49b2d9298a97425f32bd74767ed784d702c2831a132459e254b49144ecf23c02` |

The renderer compiles `src/main/resources/kernels/raytracer.cu` with it at run time, for the
card it finds and with the scene's features switched in, and loads the result through the CUDA
driver (`nvcuda.dll`, installed with the NVIDIA driver). No CUDA Toolkit is needed;
`-Drt.nvrtc=<dir>` or the `NVRTC_DIR` environment variable point at another copy instead.

Why 12.6: the driver must understand what NVRTC produces (driver 577 supports CUDA 12.9), and
CUDA 13 no longer compiles for Pascal cards such as the GTX 1070 (sm_61). To update, take the
same two files from a newer 12.x `cuda_nvrtc` archive's `bin/` folder, and the new builtins'
name (`nvrtc-builtins64_12x.dll`) in `gpu/CudaContext`.
