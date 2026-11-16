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
    fun replaceScene(next: GpuScene)

    /** Points the camera at [setup]; the next launch shoots rays from there. */
    fun setCamera(setup: CameraSetup)

    /**
     * As [setCamera], and [previous] is where the camera stood for the frame before, which the flow
     * guide measures each first hit's move against (null: it stood here). A session that keeps no
     * flow takes only [setup].
     */
    fun setCamera(setup: CameraSetup, previous: CameraSetup?) = setCamera(setup)

    /**
     * Traces samples [sampleStart] until [sampleStart] + [count] of every pixel of [setup]'s
     * image, in [mode], into [target], scaled by [scale]. A launch whose [sampleStart] is above
     * zero adds to what the target holds. Returns when the GPU is done.
     */
    fun launch(mode: Int, setup: CameraSetup, sampleStart: Int, count: Int, scale: Float, target: Int = IMAGE)

    /** Copies the first [into].size floats of [target] back to the host. */
    fun read(target: Int, into: FloatArray)

    /**
     * Empties [target], so that a launch may add to it whatever sample it starts at. Cards that
     * share an image need it: each traces some of the samples, and only one of them the first.
     */
    fun clear(target: Int)

    /**
     * Builds what tracing in [mode] needs - its kernel - unless it is built. A launch does so
     * anyway; calling this first keeps the compiler's time out of a launch that is being timed.
     */
    fun prepare(mode: Int)

    /** The cards tracing and their engines, for a status line: "RTX 2060 (OptiX)". */
    val devices: List<String>

    /**
     * Whether the guides pass fills [FLOW] with how far each pixel's first hit moved since the
     * frame before, given [setTurn]. Only OptiX knows which hits belong to a turning placed mesh;
     * elsewhere the flow is all zeros and a temporal denoiser must not be fed it.
     */
    val tracesFlow: Boolean get() = false

    /** The turn of the film frame being traced, for the flow guide; null when nothing turns. */
    fun setTurn(turn: FrameTurn?) {}

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
        /**
         * Both of the denoiser's guides from one pass over the first hit: the albedo into the
         * launch's target, the normal into [NORMAL] and the flow into [FLOW]. Half the rays of the
         * two separate passes.
         * (Gathering them in the image's own pass instead was tried, and on the RTX 2060 through
         * OptiX it slowed the image by more than both passes take.)
         */
        const val MODE_GUIDES = 4
        internal val MODE_NAMES = listOf("image", "albedo guide", "normal guide", "shaded preview", "guides")

        /** The kernel's AOV_MODE for [mode], and whether it writes the normal guide beside it. */
        internal fun aovMode(mode: Int): Int = if (mode == MODE_GUIDES) MODE_ALBEDO else mode
        internal fun gathersGuides(mode: Int): Boolean = mode == MODE_GUIDES

        /**
         * Output targets: the image, the scratch buffer guides go to, and the two outputs
         * [MODE_GUIDES] fills - the albedo, when it is the launch's target, the normal and the flow.
         */
        const val IMAGE = 0
        const val GUIDE = 1
        const val ALBEDO = 2
        const val NORMAL = 3
        const val FLOW = 4
        internal const val TARGETS = 5

        /** The outputs a launch in [mode] into [target] writes. */
        internal fun targetsOf(mode: Int, target: Int): List<Int> =
            if (gathersGuides(mode)) listOf(target, NORMAL, FLOW) else listOf(target)

        /** Length [traceSliced] aims each launch at. */
        const val SLICE_MS = 200.0
        internal const val CAMERA_FLOATS = 35

        /**
         * A session for [scene], with outputs for up to [maxPixels] pixels, on the cards and with
         * the engines [choice] names: by default every card, through OptiX where the card has RT
         * cores and through the renderer's own CUDA kernel elsewhere. One card is driven directly;
         * several share each image's samples ([MultiGpuSession]).
         *
         * In `all`, a card that cannot be started is left out with a message, as long as one is
         * left; a card named outright that fails is an error.
         */
        fun open(scene: GpuScene, maxPixels: Int, choice: GpuChoice = GpuChoice.fromProperties()): GpuSession {
            val picks = choice.resolve(CudaContext.devices(), rayTraceTypescript.gpu.optix.OptixApi.unavailabilityReason())
            if (picks.size == 1) return openOn(picks.single(), scene, maxPixels)
            return MultiGpuSession.open(picks, scene, maxPixels, lenient = choice.gpus.trim().lowercase() == GpuChoice.ALL)
        }

        /** A session on one card, for the calling thread. */
        internal fun openOn(pick: GpuPick, scene: GpuScene, maxPixels: Int): GpuSession = when {
            pick.engine == Engine.CUDA -> CudaSession(scene, maxPixels, pick.device)
            pick.fallback -> SwitchingSession(pick.device, scene, maxPixels)
            else -> rayTraceTypescript.gpu.optix.OptixSession(scene, maxPixels, pick.device)
        }

        /** Null when a CUDA device and the kernel compiler are usable here, otherwise a short reason. */
        fun unavailabilityReason(): String? = CudaContext.unavailabilityReason()

        internal fun initialSeed(): Long = RandomSource.seedValue() ?: Random.Default.nextLong()

        /**
         * The camera as the kernel reads it: 35 floats - the camera, then where it stood for the
         * frame before ([previous], or the same), which only the flow guide reads.
         */
        internal fun cameraParameters(setup: CameraSetup, previous: CameraSetup? = null): FloatArray = floatArrayOf(
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
        ) + (previous ?: setup).let { p ->
            floatArrayOf(
                p.cameraCenter.x, p.cameraCenter.y, p.cameraCenter.z,
                p.pixel00.x, p.pixel00.y, p.pixel00.z,
                p.pixelDeltaU.x, p.pixelDeltaU.y, p.pixelDeltaU.z,
                p.pixelDeltaV.x, p.pixelDeltaV.y, p.pixelDeltaV.z
            )
        }
    }
}

/**
 * How a turntable film turns its placed meshes from one frame to the next: [degrees] about the
 * vertical line through ([axisX], [axisZ]), as [Placement.turned] turns them. The flow guide
 * takes each hit on a turning mesh back by that much to see where it was a frame ago.
 */
data class FrameTurn(val axisX: Float, val axisZ: Float, val degrees: Float)
