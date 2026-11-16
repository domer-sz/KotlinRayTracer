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

    /**
     * How many times wider and taller the clean image comes back than the noisy one: 1, or 2 for
     * OptiX's model that denoises and enlarges in one go (`-Drt.upscale=2`). The guides are at
     * the noisy image's size.
     */
    val upscale: Int get() = 1

    fun denoise(width: Int, height: Int, color: FloatArray, albedo: FloatArray, normal: FloatArray): FloatArray
}
