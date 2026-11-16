package rayTraceTypescript.gpu

import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.render.CameraSetup
import rayTraceTypescript.gpu.optix.OptixDenoiser
import rayTraceTypescript.render.Denoiser
import rayTraceTypescript.render.ImageDenoiser
import rayTraceTypescript.render.PpmWriter
import rayTraceTypescript.render.ProgressBar
import rayTraceTypescript.render.Renderer
import java.nio.file.Path

/**
 * Runs the whole path tracer on a CUDA device: one thread per pixel, all samples,
 * bounces, BVH traversal and shading happen in the kernel (see [GpuSession]). The CPU only
 * flattens the scene, hands over the camera basis and writes the files.
 */
class GpuRenderer : Renderer {

    override fun render(world: Hittable, setup: CameraSetup, outputPath: Path): Long {
        val layoutStart = System.nanoTime()
        val scene = SceneFlattener.scene(world, setup.lights)
        val layoutMs = (System.nanoTime() - layoutStart) / 1_000_000
        // The CUDA kernel's own tree is not in this: it is built when a card on that engine asks.
        val laid = scene.instanced
        System.err.println(
            "[gpu] scene laid out in $layoutMs ms (${laid.instances.size} placed meshes, ${laid.sceneTriangles} triangles, " +
                "${laid.sceneQuads.size} quads, ${laid.sceneSpheres.size} spheres, ${laid.mediumCount} volumes)"
        )
        val denoiser = requestedDenoiser()

        GpuSession.open(scene, setup.totalPixels).use { session ->
            session.setCamera(setup)
            // -Drt.aov=albedo|normal renders a denoiser guide instead of the image.
            val mainMode = when (System.getProperty("rt.aov")) {
                "albedo" -> GpuSession.MODE_ALBEDO
                "normal" -> GpuSession.MODE_NORMAL
                else -> GpuSession.MODE_IMAGE
            }
            val pixels = renderAll(session, setup, mainMode, setup.samplesPerPixel, GpuSession.IMAGE)
            PpmWriter.write(outputPath, setup.imageWidth, setup.imageHeight, pixels)

            if (denoiser != null && mainMode == GpuSession.MODE_IMAGE) {
                denoiser.use {
                    // The guides only see the first hit, so a few hundred samples leave them
                    // clean; the denoiser needs them antialiased like the image, not more.
                    val guideSamples = minOf(setup.samplesPerPixel, GUIDE_SAMPLES)
                    val start = System.nanoTime()
                    val (albedo, normal) = renderGuides(session, setup, guideSamples)
                    System.err.println("[gpu] guides: ${(System.nanoTime() - start) / 1_000_000} ms")
                    // -Drt.denoise.repeat=N denoises the image N times more, to time a denoiser once it is set up.
                    val runs = 1 + (System.getProperty("rt.denoise.repeat")?.toIntOrNull() ?: 0).coerceIn(0, 100)
                    var clean = pixels
                    repeat(runs) { run ->
                        val start = System.nanoTime()
                        clean = it.denoise(setup.imageWidth, setup.imageHeight, pixels, albedo, normal)
                        val parts = (it as? OptixDenoiser)?.lastTimes?.let { t ->
                            ": up %.1f ms, denoiser %.1f ms, back %.1f ms, setting up %.1f ms".format(t.uploadMs, t.denoiseMs, t.downloadMs, t.setupMs)
                        } ?: ""
                        System.err.println("[gpu] denoised${if (runs > 1) " (${run + 1} of $runs)" else ""} by ${it.name} in ${(System.nanoTime() - start) / 1_000_000} ms$parts")
                    }
                    PpmWriter.write(denoisedPath(outputPath), setup.imageWidth, setup.imageHeight, clean)
                }
            }
        }

        return setup.totalPixels.toLong() * setup.samplesPerPixel
    }

    /**
     * All [samples] of every pixel in [mode], as a series of launches over the whole image,
     * each taking a slice of the samples sized to about TARGET_LAUNCH_MS: plenty of threads to
     * keep every SM full, and short enough for the display driver watchdog. The random streams
     * are keyed by pixel and sample, so the slicing does not change the image.
     */
    private fun renderAll(session: GpuSession, setup: CameraSetup, mode: Int, samples: Int, target: Int): FloatArray {
        val progress = ProgressBar(PROGRESS_STEPS)
        val start = System.nanoTime()
        val launches = session.traceSliced(mode, setup, samples, target, TARGET_LAUNCH_MS) { done ->
            progress.report((done.toLong() * PROGRESS_STEPS / samples).toInt())
        }
        progress.finish()
        // Kept apart so the kernel's own speed is not buried under scene preparation.
        System.err.println("\n[gpu] ${MODE_NAMES[mode]}: kernel ${(System.nanoTime() - start) / 1_000_000} ms in $launches launches")
        return FloatArray(setup.totalPixels * 3).also { session.read(target, it) }
    }

    /** image.ppm -> image-denoised.ppm, beside it. */
    private fun denoisedPath(outputPath: Path): Path {
        val name = outputPath.fileName.toString()
        val stem = if (name.lowercase().endsWith(".ppm")) name.dropLast(4) else name
        return outputPath.resolveSibling("$stem-denoised.ppm")
    }

    companion object {
        private const val TARGET_LAUNCH_MS = 200.0
        private const val PROGRESS_STEPS = 1000
        private val MODE_NAMES = listOf("image", "albedo guide", "normal guide")
        /** Samples for the denoiser guides; they only see the first hit, so few are enough. */
        private const val GUIDE_SAMPLES = 256
        fun unavailabilityReason(): String? = GpuSession.unavailabilityReason()

        /**
         * The denoiser's albedo and normal guides for [setup], averaged over [samples], from one
         * pass over the first hit ([GpuSession.MODE_GUIDES]), in slices of about [sliceMs] so a
         * heavy scene at a large size never holds the GPU past the watchdog.
         */
        fun renderGuides(
            session: GpuSession, setup: CameraSetup, samples: Int, sliceMs: Double = GpuSession.SLICE_MS
        ): Pair<FloatArray, FloatArray> {
            session.traceSliced(GpuSession.MODE_GUIDES, setup, samples, GpuSession.ALBEDO, sliceMs)
            fun read(target: Int) = FloatArray(setup.totalPixels * 3).also { session.read(target, it) }
            return read(GpuSession.ALBEDO) to read(GpuSession.NORMAL)
        }

        /** Set once the OptiX denoiser has turned out not to start here: the rest of the run denoises with OIDN. */
        @Volatile private var optixDenoiserFailed = false


        fun cudaKernelSource(): String = resource("/kernels/raytracer.cu")

        private fun resource(path: String): String = GpuRenderer::class.java.getResourceAsStream(path)
            ?.bufferedReader()?.use { it.readText() }
            ?: throw IllegalStateException("kernel resource $path is missing from the jar")

        /**
         * The denoiser, when -Drt.denoise asks for one (any value but "off"): NVIDIA's OptiX
         * denoiser on the card [optixDenoiserCard] picks, or Intel Open Image Denoise on the CPU
         * when OptiX is not to be had or -Drt.denoiser=oidn says so. An OptiX denoiser that
         * cannot start gives way to OIDN; without OIDN's DLLs either (see [Denoiser.folder]) the
         * render still runs, just without denoising.
         */
        fun requestedDenoiser(): ImageDenoiser? {
            val wanted = System.getProperty("rt.denoise")?.let { it != "off" } ?: false
            if (!wanted) return null
            optixDenoiserCard()?.let { card ->
                try {
                    return OptixDenoiser(card)
                } catch (t: Throwable) {
                    // From now on no card is kept for a denoiser that is not there.
                    optixDenoiserFailed = true
                    System.err.println("[gpu] the OptiX denoiser could not start on ${card.shortName} (${t.message}); denoising with OIDN on the CPU")
                }
            }
            val folder = Denoiser.folder()
            if (folder == null) {
                System.err.println("[gpu] -Drt.denoise: no OpenImageDenoise.dll in -Drt.oidn / OIDN_DIR, rendering without denoising")
                return null
            }
            return try {
                Denoiser.open(folder)
            } catch (t: Throwable) {
                System.err.println("[gpu] could not load Open Image Denoise from $folder: ${t.message}")
                null
            }
        }

        /**
         * The card the OptiX denoiser is to run on, or null when renders denoise with OIDN on the
         * CPU, or not at all.
         *
         * -Drt.denoiser is `auto` (the default), `optix` or `oidn`. `auto` is OptiX wherever it
         * can run - on a GTX 1070 it denoises a 1080x1920 frame in 0.3 s, OIDN on a four-core CPU
         * in 1.3 s - and `optix` the same, insisting. The card is the one -Drt.denoiser.gpu names
         * (its number in the driver's list or a piece of its name), else a card that does not
         * trace best: beside an RTX card that traces, a second card would sit idle. On a machine
         * of one card, that card does both.
         */
        fun optixDenoiserCard(): GpuDevice? {
            if (System.getProperty("rt.denoise")?.let { it != "off" } != true || optixDenoiserFailed) return null
            val mode = System.getProperty("rt.denoiser")?.lowercase() ?: GpuChoice.AUTO
            if (mode == "oidn") return null
            val devices = CudaContext.devices()
            val optixReason = rayTraceTypescript.gpu.optix.OptixApi.unavailabilityReason()
            if (devices.isEmpty() || optixReason != null) {
                if (mode == "optix") System.err.println("[gpu] -Drt.denoiser=optix: ${optixReason ?: "no CUDA device"}; denoising with OIDN on the CPU")
                optixDenoiserFailed = true
                return null
            }
            System.getProperty("rt.denoiser.gpu")?.takeIf { it.isNotBlank() }?.let { named ->
                return GpuChoice(named, "cuda").resolve(devices, null).first().device
            }
            val tracer = GpuChoice(GpuChoice.ONE).resolve(devices, optixReason).single().device
            return devices.firstOrNull { it.ordinal != tracer.ordinal } ?: tracer
        }
    }
}
