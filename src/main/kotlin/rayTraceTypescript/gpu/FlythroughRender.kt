package rayTraceTypescript.gpu

import rayTraceTypescript.Camera
import rayTraceTypescript.Point
import rayTraceTypescript.render.CameraSetup
import rayTraceTypescript.render.VideoEncoder
import rayTraceTypescript.scenes.SceneDefinition
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.sqrt

/**
 * A film of the camera moving through a still scene: -Drt.flythrough=<file.mp4>, along the key
 * positions in -Drt.flythrough.path=<file> - one a line, `fromX fromY fromZ atX atY atZ [vfov]`,
 * `#` for comments - in -Drt.flythrough.seconds (15) at -Drt.flythrough.fps (60), each frame at
 * -Drt.samples, -Drt.flythrough.quality=max for the larger bitrate.
 *
 * The camera runs through the keys on a Catmull-Rom curve, the point it looks at on another, at
 * one speed along the way - the time between keys goes by how far apart they stand - easing in
 * at the start and out at the end. The scene is uploaded once; each frame is traced, its guides
 * gathered and handed to a worker that denoises and encodes it while the next frame traces, as
 * a turntable film does. -Drt.temporal=on denoises with OptiX's temporal model, along the flow
 * the camera's move makes.
 */
object FlythroughRender {

    /** One key of the path: where the camera stands, where it looks, its vertical field of view. */
    data class Key(val from: DoubleArray, val at: DoubleArray, val vfov: Double)

    fun render(scene: SceneDefinition, output: Path) {
        GpuRenderer.unavailabilityReason()?.let { throw IllegalStateException("a flythrough needs the GPU: $it") }
        val pathFile = System.getProperty("rt.flythrough.path") ?: throw IllegalArgumentException("-Drt.flythrough.path=<file> names the camera's keys")
        val keys = readKeys(java.nio.file.Paths.get(pathFile), scene.camera.vfov.toDouble())
        require(keys.size >= 2) { "a flythrough needs at least two keys, $pathFile has ${keys.size}" }
        val seconds = System.getProperty("rt.flythrough.seconds")?.toDoubleOrNull() ?: 15.0
        val fps = System.getProperty("rt.flythrough.fps")?.toIntOrNull() ?: 60
        val frames = (seconds * fps).toInt().coerceAtLeast(2)
        val samples = scene.camera.samplesPerPixel
        val path = CameraPath(keys)

        val base = scene.camera
        fun setupAt(frame: Int): CameraSetup {
            val key = path.at(frame.toDouble() / (frames - 1))
            return Camera().apply {
                aspectRatio = base.aspectRatio
                imageWidth = base.imageWidth
                samplesPerPixel = samples
                maxReflectionDepth = base.maxReflectionDepth
                vfov = key.vfov.toFloat()
                lookFrom = Point(key.from[0], key.from[1], key.from[2])
                lookAt = Point(key.at[0], key.at[1], key.at[2])
                vUp = base.vUp
                defocusAngle = 0f
                background = base.background
                environment = base.environment
                lights = base.lights
                fireflyClamp = base.fireflyClamp
            }.initialize()
        }
        val first = setupAt(0)
        println("Flythrough: ${keys.size} keys, ${first.imageWidth}x${first.imageHeight}, $frames frames ($seconds s at $fps fps), $samples spp -> $output")

        val denoiser = GpuRenderer.requestedDenoiser(temporal = true)
        val compact = System.getProperty("rt.flythrough.quality") != "max"
        val encoder = VideoEncoder.start(output, first.imageWidth, first.imageHeight, fps, VideoEncoder.bitrate(first.imageWidth, first.imageHeight, fps, compact))
        val worker = Executors.newSingleThreadExecutor { Thread(it, "flythrough-denoise").apply { isDaemon = true } }
        var finished = false
        var pending: Future<*>? = null
        val start = System.nanoTime()
        val timing = FrameTiming()
        try {
            GpuSession.open(SceneFlattener.scene(scene.world, base.lights), first.totalPixels).use { session ->
                val temporal = denoiser?.temporal == true && session.tracesFlow
                val baseSeed = session.seed
                var previous: CameraSetup? = null
                for (frame in 0 until frames) {
                    val setup = setupAt(frame)
                    val traced = if (denoiser?.upscale == 2) setup.halved() else setup
                    val before = previous?.let { if (denoiser?.upscale == 2) it.halved() else it }
                    val traceStart = System.nanoTime()
                    session.seed = baseSeed + frame
                    session.setCamera(traced, if (temporal) before ?: traced else null)
                    session.traceSliced(GpuSession.MODE_IMAGE, traced, samples, GpuSession.IMAGE, TurntableFilm.SLICE_MS)
                    val image = FloatArray(traced.totalPixels * 3).also { session.read(GpuSession.IMAGE, it) }
                    val guideStart = System.nanoTime()
                    val guides = denoiser?.let { GpuRenderer.renderGuides(session, traced, minOf(samples, TurntableFilm.GUIDE_SAMPLES), TurntableFilm.SLICE_MS) }
                    val flow = if (temporal) FloatArray(traced.totalPixels * 3).also { session.read(GpuSession.FLOW, it) } else null
                    val waitStart = System.nanoTime()
                    pending?.get()
                    timing.renderThread(0L, guideStart - traceStart, waitStart - guideStart, System.nanoTime() - waitStart)
                    if (frame > 0) {
                        val perFrame = (System.nanoTime() - start) / 1e9 / frame
                        System.err.print("\r[flythrough] frame $frame of $frames, %.2f s a frame, about %.0f s left   ".format(perFrame, perFrame * (frames - frame)))
                    }
                    pending = worker.submit {
                        val denoiseStart = System.nanoTime()
                        val clean = when {
                            denoiser == null || guides == null -> image
                            flow != null -> denoiser.denoiseNext(traced.imageWidth, traced.imageHeight, image, guides.first, guides.second, flow, continuing = frame > 0)
                            else -> denoiser.denoise(traced.imageWidth, traced.imageHeight, image, guides.first, guides.second)
                        }
                        val encodeStart = System.nanoTime()
                        encoder.write(clean)
                        timing.worker(encodeStart - denoiseStart, System.nanoTime() - encodeStart)
                    }
                    previous = setup
                }
                pending?.get()
            }
            encoder.finish()
            finished = true
            System.err.println("\n[flythrough] ${timing.summary()}")
            println("Film saved: ${output.toAbsolutePath()}")
        } finally {
            worker.shutdownNow()
            if (!finished) encoder.abort()
            denoiser?.close()
        }
    }

    fun readKeys(file: Path, defaultVfov: Double): List<Key> = Files.readAllLines(file)
        .map { it.substringBefore('#').trim() }
        .filter { it.isNotEmpty() }
        .map { line ->
            val n = line.split(Regex("[ \t,]+")).map { it.toDouble() }
            require(n.size >= 6) { "a key is fromX fromY fromZ atX atY atZ [vfov]: '$line'" }
            Key(doubleArrayOf(n[0], n[1], n[2]), doubleArrayOf(n[3], n[4], n[5]), n.getOrElse(6) { defaultVfov })
        }

    /**
     * The keys joined into one smooth move: Catmull-Rom through the camera's positions and through
     * the points it looks at, the share of time between two keys going by the distance between them.
     */
    class CameraPath(private val keys: List<Key>) {
        private val ends: DoubleArray

        init {
            val lengths = (0 until keys.size - 1).map { distance(keys[it].from, keys[it + 1].from).coerceAtLeast(1e-6) }
            val total = lengths.sum()
            ends = DoubleArray(keys.size)
            for (i in 1 until keys.size) ends[i] = ends[i - 1] + lengths[i - 1] / total
        }

        /** The key at [t] in [0, 1] of the film, eased in and out. */
        fun at(t: Double): Key {
            val eased = t * t * (3 - 2 * t)
            var segment = 0
            while (segment < keys.size - 2 && eased > ends[segment + 1]) segment++
            val local = ((eased - ends[segment]) / (ends[segment + 1] - ends[segment])).coerceIn(0.0, 1.0)
            fun k(i: Int) = keys[i.coerceIn(0, keys.size - 1)]
            val p0 = k(segment - 1); val p1 = k(segment); val p2 = k(segment + 1); val p3 = k(segment + 2)
            return Key(
                spline(p0.from, p1.from, p2.from, p3.from, local),
                spline(p0.at, p1.at, p2.at, p3.at, local),
                p1.vfov + (p2.vfov - p1.vfov) * local
            )
        }

        private fun spline(a: DoubleArray, b: DoubleArray, c: DoubleArray, d: DoubleArray, t: Double) = DoubleArray(3) { i ->
            val t2 = t * t
            val t3 = t2 * t
            0.5 * (2 * b[i] + (c[i] - a[i]) * t + (2 * a[i] - 5 * b[i] + 4 * c[i] - d[i]) * t2 + (3 * b[i] - a[i] - 3 * c[i] + d[i]) * t3)
        }

        private fun distance(a: DoubleArray, b: DoubleArray) = sqrt((0 until 3).sumOf { (a[it] - b[it]) * (a[it] - b[it]) })
    }
}
