package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.nio.file.Files
import java.util.zip.GZIPInputStream
import rayTraceTypescript.gpu.GpuRenderer
import rayTraceTypescript.render.PpmWriter
import rayTraceTypescript.render.Renderer
import rayTraceTypescript.utils.RandomSource

/**
 * The One Weekend finale, rendered by the GPU, must look like the golden image - a CPU render
 * with a pinned seed, stored in src/test/resources (the CPU path tracer that made it is gone,
 * so it is frozen). The GPU draws other random numbers, so the
 * yardstick is the noise: a second CPU render with another seed, stored beside it.
 */
class ImageRegressionTest {
    @Test
    fun `rendered ppm stays close to golden image`() {
        val reason = GpuRenderer.unavailabilityReason()
        assumeTrue(reason == null, "No GPU available: $reason")
        val golden = PpmSupport.parse(loadReference(GOLDEN))
        val otherSeed = PpmSupport.parse(loadReference(OTHER_SEED_REFERENCE))
        val actual = render(GpuRenderer(), SEED)
        assertEquals(golden.width, actual.width, "reference width")
        assertEquals(golden.height, actual.height, "reference height")

        val noise = PpmSupport.diff(golden, otherSeed, HIGH_DIFF_THRESHOLD)
        val diff = PpmSupport.diff(golden, actual, HIGH_DIFF_THRESHOLD)
        val allowedMean = noise.meanAbsDiff * NOISE_TOLERANCE
        assertTrue(
            diff.meanAbsDiff <= allowedMean,
            "GPU differs from the golden image by ${diff.meanAbsDiff} on average (max ${diff.maxDiff}); " +
                "two CPU seeds differ by ${noise.meanAbsDiff}, so anything above $allowedMean is a real difference"
        )
        assertTrue(
            diff.highDiffCount <= maxOf(noise.highDiffCount * 2, MIN_HIGH_DIFF_CHANNELS),
            "${diff.highDiffCount} channels differ by >= $HIGH_DIFF_THRESHOLD (CPU-to-CPU noise produces ${noise.highDiffCount})"
        )
    }

    private fun camera() = Camera().apply {
        aspectRatio = 16.0f / 9.0f
        imageWidth = 300
        samplesPerPixel = 50
        maxReflectionDepth = 45
        vfov = 20.0f
        lookFrom = Point(13.0f, 2.0f, 3.0f)
        lookAt = Point(0.0f, 0.0f, 0.0f)
        vUp = Vector(0.0f, 1.0f, 0.0f)
        defocusAngle = 0.6f
        focusDistance = 10.0f
    }

    private fun render(renderer: Renderer, seed: Long): PpmSupport.PpmImage {
        val world = prepareWorld()
        val tempFile = Files.createTempFile("raytracer-image-test", ".ppm")
        RandomSource.withSeed(seed)
        return try {
            renderer.render(world, camera().initialize(), tempFile)
            PpmSupport.parse(Files.readAllBytes(tempFile))
        } finally {
            Files.deleteIfExists(tempFile)
            Files.deleteIfExists(PpmWriter.pngPath(tempFile))
            Files.deleteIfExists(tempFile.resolveSibling(tempFile.fileName.toString().removeSuffix(".ppm") + "-denoised.ppm"))
            RandomSource.reset()
        }
    }

    private fun loadReference(name: String): ByteArray =
        GZIPInputStream(requireNotNull(javaClass.getResourceAsStream("/$name")) { "Missing reference image $name" })
            .use { it.readBytes() }

    companion object {
        private const val GOLDEN = "reference-image.ppm.gz"
        private const val OTHER_SEED_REFERENCE = "reference-image-seed1337.ppm.gz"
        private const val SEED = 4242L
        private const val OTHER_SEED = 1337L
        private const val HIGH_DIFF_THRESHOLD = 55
        /** Headroom over the CPU-to-CPU noise floor, for float differences between devices. */
        private const val NOISE_TOLERANCE = 1.5f
        private const val MIN_HIGH_DIFF_CHANNELS = 8
    }
}
