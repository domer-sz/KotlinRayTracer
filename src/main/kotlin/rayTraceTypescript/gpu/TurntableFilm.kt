package rayTraceTypescript.gpu

import rayTraceTypescript.render.CameraSetup
import rayTraceTypescript.render.Denoiser
import rayTraceTypescript.render.VideoEncoder
import java.nio.file.Path
import java.util.concurrent.Executors
import java.util.concurrent.Future

/**
 * What a turntable film is made of, with StlViewer's defaults: one turn of [seconds] at [fps]
 * frames a second, played [turns] times, [samples] per pixel each frame.
 */
data class TurntableSettings(
    val seconds: Double = 15.0,
    val turns: Int = 1,
    val fps: Int = 30,
    val samples: Int = 64,
    /** Clockwise seen from above instead of counter-clockwise. */
    val reverse: Boolean = false,
    /** The smaller file (about 0.085 bits per pixel and frame) rather than the best (0.15). */
    val compact: Boolean = true
) {
    /** Frames of one turn; the turns after the first are copies of it. */
    val framesPerTurn: Int get() = maxOf(2, Math.round(seconds * fps).toInt())

    /**
     * How far the figure has turned at [frame], in degrees: a full turn over the frames, the last
     * frame one step short of the first, so the turn loops without a doubled frame.
     */
    fun degrees(frame: Int): Float = ((if (reverse) -360.0 else 360.0) * frame / framesPerTurn).toFloat()

    companion object {
        /** Settings from -Drt.turntable.seconds / .turns / .fps / .reverse / .quality and -Drt.samples. */
        fun fromProperties(defaultSamples: Int): TurntableSettings {
            fun prop(name: String) = System.getProperty("rt.turntable.$name")
            return TurntableSettings(
                seconds = prop("seconds")?.toDoubleOrNull()?.coerceIn(1.0, 60.0) ?: 15.0,
                turns = prop("turns")?.toIntOrNull()?.coerceIn(1, 10) ?: 1,
                fps = prop("fps")?.toIntOrNull()?.coerceIn(1, 120) ?: 30,
                samples = System.getProperty("rt.samples")?.toIntOrNull() ?: defaultSamples,
                reverse = prop("reverse")?.let { it != "false" && it != "off" } ?: false,
                compact = prop("quality") != "max"
            )
        }
    }
}

/** How a film is getting on: [done] of [total] frames written after [elapsedNanos]. */
fun interface FilmProgress {
    fun report(done: Int, total: Int, elapsedNanos: Long)
}

/**
 * Where a film's frames spend their time, on average: the render thread readies the scene, traces
 * the image and the guides, then waits until the worker has the frame before in the film; the
 * worker denoises and encodes. A wait above zero means the worker sets the pace.
 */
internal class FrameTiming {
    private var frames = 0
    private var scene = 0L
    private var trace = 0L
    private var guides = 0L
    private var wait = 0L
    private val workerFrames = java.util.concurrent.atomic.AtomicInteger()
    private val denoise = java.util.concurrent.atomic.AtomicLong()
    private val encode = java.util.concurrent.atomic.AtomicLong()

    fun renderThread(sceneNs: Long, traceNs: Long, guidesNs: Long, waitNs: Long) {
        frames++
        scene += sceneNs
        trace += traceNs
        guides += guidesNs
        wait += waitNs
    }

    fun worker(denoiseNs: Long, encodeNs: Long) {
        workerFrames.incrementAndGet()
        denoise.addAndGet(denoiseNs)
        encode.addAndGet(encodeNs)
    }

    fun summary(): String {
        fun ms(total: Long, count: Int) = if (count == 0) 0 else total / count / 1_000_000
        val w = workerFrames.get()
        return "a frame on average: scene ${ms(scene, frames)} ms, trace ${ms(trace, frames)} ms, guides ${ms(guides, frames)} ms, " +
            "waiting for the frame before ${ms(wait, frames)} ms; beside it, denoising ${ms(denoise.get(), w)} ms and encoding ${ms(encode.get(), w)} ms"
    }
}

/**
 * Renders a turntable film frame by frame on the GPU and hands the frames to ffmpeg.
 *
 * Each frame is traced in slices of about [SLICE_MS] (well inside the display driver's two
 * second watchdog even when a hot card drops its clock), followed by the denoiser's guides; then
 * the CPU denoises it and writes it to the film while the GPU already traces the next frame -
 * OIDN runs on the CPU, so the two overlap. Only one frame waits at a time, which keeps the
 * order and the memory bounded. Every frame gets its own seed.
 */
object TurntableFilm {
    /** Samples for each frame's denoiser guides: they only see the first hit. */
    const val GUIDE_SAMPLES = 32
    /** Length each launch is aimed at; StlViewer settled on 100 ms after two driver resets. */
    const val SLICE_MS = 100.0

    /**
     * Renders the film into [output]. [sessionFor] readies the GPU for a frame - the figure turned
     * to that frame's angle - and returns the session to trace it in. Returns false, writing
     * nothing, when [cancelled] says so between frames; a failure throws and writes nothing either.
     */
    fun render(
        setup: CameraSetup,
        settings: TurntableSettings,
        denoiser: Denoiser?,
        output: Path,
        sessionFor: (frame: Int) -> GpuSession,
        cancelled: () -> Boolean = { false },
        progress: FilmProgress = FilmProgress { _, _, _ -> }
    ): Boolean {
        val width = setup.imageWidth
        val height = setup.imageHeight
        val frames = settings.framesPerTurn
        val encoder = VideoEncoder.start(
            output, width, height, settings.fps,
            VideoEncoder.bitrate(width, height, settings.fps, settings.compact)
        )
        val worker = Executors.newSingleThreadExecutor { Thread(it, "film-denoise").apply { isDaemon = true } }
        var pending: Future<*>? = null
        var finished = false
        val start = System.nanoTime()
        // Where a frame's time goes: on this thread, and on the worker beside it.
        val timing = FrameTiming()
        try {
            var baseSeed: Long? = null
            for (frame in 0 until frames) {
                if (cancelled()) break
                val sceneStart = System.nanoTime()
                val session = sessionFor(frame)
                val traceStart = System.nanoTime()
                val seed = baseSeed ?: session.seed.also { baseSeed = it }
                session.seed = seed + frame
                session.setCamera(setup)
                session.traceSliced(GpuSession.MODE_IMAGE, setup, settings.samples, GpuSession.IMAGE, SLICE_MS)
                val image = FloatArray(setup.totalPixels * 3).also { session.read(GpuSession.IMAGE, it) }
                val guideStart = System.nanoTime()
                val guides = denoiser?.let {
                    val guideSamples = minOf(settings.samples, GUIDE_SAMPLES)
                    fun guide(mode: Int): FloatArray {
                        session.traceSliced(mode, setup, guideSamples, GpuSession.GUIDE, SLICE_MS)
                        return FloatArray(setup.totalPixels * 3).also { session.read(GpuSession.GUIDE, it) }
                    }
                    guide(GpuSession.MODE_ALBEDO) to guide(GpuSession.MODE_NORMAL)
                }
                val waitStart = System.nanoTime()
                // The frame before must be in the film before this one goes after it.
                pending?.get()
                timing.renderThread(traceStart - sceneStart, guideStart - traceStart, waitStart - guideStart, System.nanoTime() - waitStart)
                if (frame > 0) progress.report(frame, frames, System.nanoTime() - start)
                pending = worker.submit {
                    val denoiseStart = System.nanoTime()
                    val clean = if (denoiser != null && guides != null) {
                        denoiser.denoise(width, height, image, guides.first, guides.second)
                    } else image
                    val encodeStart = System.nanoTime()
                    encoder.write(clean)
                    timing.worker(encodeStart - denoiseStart, System.nanoTime() - encodeStart)
                }
            }
            pending?.get()
            if (cancelled()) return false
            progress.report(frames, frames, System.nanoTime() - start)
            encoder.finish(settings.turns)
            System.err.println("\n[film] ${timing.summary()}")
            finished = true
            return true
        } finally {
            worker.shutdownNow()
            if (!finished) encoder.abort()
        }
    }
}
