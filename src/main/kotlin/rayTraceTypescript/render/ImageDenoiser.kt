package rayTraceTypescript.render

/**
 * Something that takes a noisy linear image and the albedo and normal of the first hit of each
 * pixel (see the kernel's AOV_MODE), and returns a clean image of the same size: Intel Open
 * Image Denoise on the CPU ([Denoiser]), or NVIDIA's OptiX denoiser on a card
 * ([rayTraceTypescript.gpu.optix.OptixDenoiser]).
 *
 * Images are 3 floats a pixel, rows top to bottom. One call at a time; a denoiser keeps what it
 * set up for one size and reuses it for the next call of that size.
 */
interface ImageDenoiser : AutoCloseable {
    /** What denoises and where, for a log or a status line: "OIDN (CPU)", "OptiX (GTX 1070)". */
    val name: String

    fun denoise(width: Int, height: Int, color: FloatArray, albedo: FloatArray, normal: FloatArray): FloatArray
}
