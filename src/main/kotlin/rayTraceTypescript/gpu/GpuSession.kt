package rayTraceTypescript.gpu

import rayTraceTypescript.render.CameraSetup
import rayTraceTypescript.utils.RandomSource
import kotlin.random.Random

/**
 * A scene living on the GPU: its buffers uploaded once, one compiled kernel per mode (the image,
 * the two denoiser guides and the quick shaded look), and two output buffers - the image, and a
 * scratch one the guides render into so they never disturb samples gathered so far.
 *
 * Everything that changes between launches - the camera, the resolution, which samples - is a
 * launch argument, so the same session serves a file render, a film and an interactive viewer.
 * An edited scene replaces the old one in place ([replaceScene]): same device, same outputs, and
 * the same kernels as long as the scene needs the same features. Launches block until the GPU
 * is done; a session belongs to one thread.
 *
 * [open] makes one; the implementation is [CudaSession].
 */
interface GpuSession : AutoCloseable {
    /**
     * Keys every random stream of the next launches, with the pixel and the sample. A film sets
     * it per frame, so its frames do not all share one noise pattern.
     */
    var seed: Long

    /**
     * Swaps the scene for [next]: its buffers replace the old ones, and the kernels are kept -
     * only pointed at the new buffers - unless [next] needs other features compiled in.
     */
    fun replaceScene(next: SceneBuffers)

    /** Points the camera at [setup]; the next launch shoots rays from there. */
    fun setCamera(setup: CameraSetup)

    /**
     * Traces samples [sampleStart] until [sampleStart] + [count] of every pixel of [setup]'s
     * image, in [mode], into [target], scaled by [scale]. A launch whose [sampleStart] is above
     * zero adds to what the target holds. Returns when the GPU is done.
     */
    fun launch(mode: Int, setup: CameraSetup, sampleStart: Int, count: Int, scale: Float, target: Int = IMAGE)

    /** Copies the first [into].size floats of [target] back to the host. */
    fun read(target: Int, into: FloatArray)

    override fun close()

    /**
     * All [samples] of every pixel of [setup]'s image in [mode], into [target] averaged (each
     * launch scales by 1 / [samples]), as a series of launches each taking a slice of the
     * samples sized to about [sliceMs]: plenty of threads to keep every SM full, and short enough
     * for the display driver watchdog (a single launch of a heavy scene at a large size can pass
     * its two seconds). The random streams are keyed by pixel and sample, so the slicing does not
     * change the image. [onProgress] gets the samples done after each launch; returns the launches.
     */
    fun traceSliced(
        mode: Int, setup: CameraSetup, samples: Int, target: Int = IMAGE,
        sliceMs: Double = SLICE_MS, onProgress: ((Int) -> Unit)? = null
    ): Int {
        var launches = 0
        var done = 0
        var slice = 1
        while (done < samples) {
            val count = minOf(slice, samples - done)
            val launchStart = System.nanoTime()
            launch(mode, setup, done, count, 1.0f / samples, target)
            val launchMs = maxOf((System.nanoTime() - launchStart) / 1_000_000.0, 1.0)
            done += count
            launches++
            slice = (count * sliceMs / launchMs).toInt().coerceIn(1, count * 4)
            onProgress?.invoke(done)
        }
        return launches
    }

    companion object {
        /** Kernel modes (AOV_MODE): the image, the denoiser's albedo and normal guides, and the quick shaded look. */
        const val MODE_IMAGE = 0
        const val MODE_ALBEDO = 1
        const val MODE_NORMAL = 2
        const val MODE_SHADED = 3
        internal val MODE_NAMES = listOf("image", "albedo guide", "normal guide", "shaded preview")

        /** Output targets: the image, and the scratch buffer the guides use. */
        const val IMAGE = 0
        const val GUIDE = 1

        /** Length [traceSliced] aims each launch at. */
        const val SLICE_MS = 200.0
        internal const val CAMERA_FLOATS = 23

        /** A session for [scene], with outputs for up to [maxPixels] pixels. */
        fun open(scene: SceneBuffers, maxPixels: Int): GpuSession = CudaSession(scene, maxPixels)

        /** Null when a CUDA device and the kernel compiler are usable here, otherwise a short reason. */
        fun unavailabilityReason(): String? = CudaContext.unavailabilityReason()

        internal fun initialSeed(): Long = RandomSource.seedValue() ?: Random.Default.nextLong()

        /** The camera as the kernel reads it: 23 floats. */
        internal fun cameraParameters(setup: CameraSetup): FloatArray = floatArrayOf(
            setup.cameraCenter.x, setup.cameraCenter.y, setup.cameraCenter.z,
            setup.pixel00.x, setup.pixel00.y, setup.pixel00.z,
            setup.pixelDeltaU.x, setup.pixelDeltaU.y, setup.pixelDeltaU.z,
            setup.pixelDeltaV.x, setup.pixelDeltaV.y, setup.pixelDeltaV.z,
            setup.defocusDiscU.x, setup.defocusDiscU.y, setup.defocusDiscU.z,
            setup.defocusDiscV.x, setup.defocusDiscV.y, setup.defocusDiscV.z,
            setup.defocusAngle,
            // A scene without a background colour keeps the sky gradient.
            if (setup.background == null) 0.0f else 1.0f,
            setup.background?.r ?: 0.0f,
            setup.background?.g ?: 0.0f,
            setup.background?.b ?: 0.0f
        )
    }
}
