package rayTraceTypescript.gpu

import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.render.CameraSetup
import rayTraceTypescript.render.Denoiser
import rayTraceTypescript.render.PpmWriter
import rayTraceTypescript.render.ProgressBar
import rayTraceTypescript.render.Renderer
import java.nio.file.Path

/**
 * Runs the whole path tracer on an OpenCL device: one work item per pixel, all samples,
 * bounces, BVH traversal and shading happen in the kernel (see [GpuSession]). The CPU only
 * flattens the scene, hands over the camera basis and writes the files.
 */
class GpuRenderer : Renderer {

    override fun render(world: Hittable, setup: CameraSetup, outputPath: Path): Long {
        val flattenStart = System.nanoTime()
        val scene = SceneFlattener.flatten(world, setup.lights)
        val flattenMs = (System.nanoTime() - flattenStart) / 1_000_000
        System.err.println("[gpu] flatten $flattenMs ms (${scene.nodeCount} nodes, depth ${scene.maxDepth})")
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
                    val albedo = renderAll(session, setup, GpuSession.MODE_ALBEDO, guideSamples, GpuSession.GUIDE)
                    val normal = renderAll(session, setup, GpuSession.MODE_NORMAL, guideSamples, GpuSession.GUIDE)
                    val start = System.nanoTime()
                    val clean = it.denoise(setup.imageWidth, setup.imageHeight, pixels, albedo, normal)
                    System.err.println("[gpu] denoised in ${(System.nanoTime() - start) / 1_000_000} ms")
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

        fun openClKernelSource(): String = resource("/kernels/raytracer.cl")

        fun cudaKernelSource(): String = resource("/kernels/raytracer.cu")

        private fun resource(path: String): String = GpuRenderer::class.java.getResourceAsStream(path)
            ?.bufferedReader()?.use { it.readText() }
            ?: throw IllegalStateException("kernel resource $path is missing from the jar")

        /**
         * The denoiser, when -Drt.denoise asks for one (any value but "off") and its DLLs are
         * found (see [Denoiser.folder]). Without them the render still runs, just without it.
         */
        fun requestedDenoiser(): Denoiser? {
            val wanted = System.getProperty("rt.denoise")?.let { it != "off" } ?: false
            if (!wanted) return null
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
    }
}
