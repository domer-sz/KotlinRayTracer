# NVIDIA OptiX headers

OptiX is how the renderer reaches the RT cores of an RTX card. Its library, `nvoptix.dll`, comes
with the NVIDIA driver; what a program needs besides is the SDK's headers, which the kernel
includes when NVRTC compiles it.

**The headers are not in this repository.** They are NVIDIA's, under the NVIDIA SDK licence
(`LICENSE.txt` once fetched), which allows using them to build an application and forbids
passing them on. Everything in this folder except this file and `fetch.ps1` is ignored by git.

To get them, once per clone:

```
powershell -ExecutionPolicy Bypass -File native/optix/fetch.ps1
```

It downloads the tag `v9.0.0` of https://github.com/NVIDIA/optix-dev and keeps its `include/`
folder and the licence. Running it is accepting that licence. `-Drt.optix.include=<dir>` or the
`OPTIX_INCLUDE` environment variable point at another copy instead.

Why 9.0.0: the driver must know the version the headers describe. OptiX 9.0 needs a driver of
the 570 series or newer; 9.1 needs the 590 series, which no longer runs a GTX 1070 - and the last
driver that runs a GTX 1070 and an RTX card together is 581.

Without the headers the renderer still works: it traces with its own CUDA kernel and says why
OptiX was left out.
