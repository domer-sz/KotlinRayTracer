package rayTraceTypescript.render

/**
 * Something that takes a noisy linear image and the albedo and normal of the first hit of each
 * pixel (see the kernel's AOV_MODE), and returns a clean one: NVIDIA's OptiX denoiser on a card
 * ([rayTraceTypescript.gpu.optix.OptixDenoiser]).
 *
 * Images are 3 floats a pixel, rows top to bottom. One call at a time; a denoiser keeps what it
 * set up for one size and reuses it for the next call of that size.
 */
interface ImageDenoiser : AutoCloseable {
    /** What denoises and where, for a log or a status line: "OptiX (RTX 2060)", "OptiX (RTX 2060), 2x". */
    val name: String

    /**
     * How many times wider and taller the clean image comes back than the noisy one: 1, or 2 for
     * OptiX's model that denoises and enlarges in one go (`-Drt.upscale=2`). The guides are at
     * the noisy image's size.
     */
    val upscale: Int get() = 1

    fun denoise(width: Int, height: Int, color: FloatArray, albedo: FloatArray, normal: FloatArray): FloatArray

    /**
     * Whether it carries what it learnt from one frame of a film into the next ([denoiseNext]),
     * which keeps a few samples' noise from flickering: OptiX's temporal models.
     */
    val temporal: Boolean get() = false

    /**
     * The next frame of a film: [flow] is how far each pixel's first hit moved since the frame
     * before (3 floats a pixel, the third unused, as the guides pass leaves it), [continuing]
     * whether that frame was the last one denoised here. A denoiser without [temporal] memory
     * denoises the frame alone.
     */
    fun denoiseNext(
        width: Int, height: Int, color: FloatArray, albedo: FloatArray, normal: FloatArray, flow: FloatArray, continuing: Boolean
    ): FloatArray = denoise(width, height, color, albedo, normal)
}
