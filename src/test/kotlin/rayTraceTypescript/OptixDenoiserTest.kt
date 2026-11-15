package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.CudaContext
import rayTraceTypescript.gpu.GpuRenderer
import rayTraceTypescript.gpu.optix.OptixApi
import rayTraceTypescript.gpu.optix.OptixDenoiser
import java.util.Random

/**
 * NVIDIA's OptiX denoiser on every card of the machine: it must smooth a noisy image as Open
 * Image Denoise does, keep what it set up between calls of a size, and set up again for another.
 * Skipped where OptiX cannot run (no card, no SDK headers - native/optix/README.md).
 */
class OptixDenoiserTest {

    /** A grey ramp with noise on it, and the matching albedo and normal guides. */
    private class Frame(val width: Int, val height: Int, seed: Long) {
        val clean = FloatArray(width * height * 3)
        val noisy = FloatArray(width * height * 3)
        val albedo = FloatArray(width * height * 3) { 0.5f }
        val normal = FloatArray(width * height * 3) { if (it % 3 == 2) 1f else 0f }

        init {
            val random = Random(seed)
            for (pixel in 0 until width * height) {
                val level = 0.2f + 0.6f * (pixel % width) / width
                for (c in 0 until 3) {
                    clean[pixel * 3 + c] = level
                    noisy[pixel * 3 + c] = level + (random.nextFloat() - 0.5f) * 0.4f
                }
            }
        }

        fun error(image: FloatArray): Double = image.indices.sumOf { ((image[it] - clean[it]).toDouble()).let { d -> d * d } } / image.size
    }

    @Test
    fun smoothsTheNoiseOnEveryCard() {
        assumeOptix()
        for (device in CudaContext.devices()) {
            OptixDenoiser(device).use { denoiser ->
                val frame = Frame(128, 96, 4)
                val clean = denoiser.denoise(frame.width, frame.height, frame.noisy, frame.albedo, frame.normal)
                assertTrue(
                    frame.error(clean) < frame.error(frame.noisy) / 4,
                    "${denoiser.name}: denoised ${frame.error(clean)} vs noisy ${frame.error(frame.noisy)}"
                )
            }
        }
    }

    @Test
    fun givesTheSameImageAgainAndSetsUpAgainForAnotherSize() {
        assumeOptix()
        OptixDenoiser(CudaContext.devices().first()).use { denoiser ->
            val small = Frame(96, 64, 2)
            val large = Frame(160, 120, 3)
            val first = denoiser.denoise(small.width, small.height, small.noisy, small.albedo, small.normal)
            val again = denoiser.denoise(small.width, small.height, small.noisy, small.albedo, small.normal)
            assertArrayEquals(first, again)
            val largeImage = denoiser.denoise(large.width, large.height, large.noisy, large.albedo, large.normal)
            assertEquals(large.width * large.height * 3, largeImage.size)
            assertArrayEquals(first, denoiser.denoise(small.width, small.height, small.noisy, small.albedo, small.normal))
        }
    }

    private fun assumeOptix() {
        val gpu = GpuRenderer.unavailabilityReason()
        assumeTrue(gpu == null, "No CUDA device available: $gpu")
        val optix = OptixApi.unavailabilityReason()
        assumeTrue(optix == null, "OptiX is not available: $optix")
    }
}
