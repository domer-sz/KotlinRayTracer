package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.render.Denoiser
import java.util.Random

/**
 * The denoiser keeps its OIDN device, filter and buffers between calls. A second call on the
 * same size must give the same image as the first, a call on another size must rebuild them,
 * and the result must still be the noisy image made smoother. Skipped without the OIDN DLLs.
 */
class DenoiserTest {

    private fun open(): Denoiser {
        val folder = Denoiser.folder()
        assumeTrue(folder != null, "no OpenImageDenoise.dll in -Drt.oidn / OIDN_DIR / native/oidn")
        return try {
            Denoiser.open(folder!!)
        } catch (t: Throwable) {
            assumeTrue(false, "OIDN could not be loaded: ${t.message}")
            throw t
        }
    }

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
    fun givesTheSameImageWhenCalledAgainOnTheSameSize() {
        open().use { denoiser ->
            val frame = Frame(96, 64, 1)
            val first = denoiser.denoise(frame.width, frame.height, frame.noisy, frame.albedo, frame.normal)
            val second = denoiser.denoise(frame.width, frame.height, frame.noisy, frame.albedo, frame.normal)
            assertArrayEquals(first, second)
        }
    }

    @Test
    fun rebuildsForAnotherSizeAndComesBack() {
        open().use { denoiser ->
            val small = Frame(96, 64, 2)
            val large = Frame(160, 120, 3)
            val smallFirst = denoiser.denoise(small.width, small.height, small.noisy, small.albedo, small.normal)
            val largeImage = denoiser.denoise(large.width, large.height, large.noisy, large.albedo, large.normal)
            val smallAgain = denoiser.denoise(small.width, small.height, small.noisy, small.albedo, small.normal)
            assertEquals(large.width * large.height * 3, largeImage.size)
            assertArrayEquals(smallFirst, smallAgain)
        }
    }

    @Test
    fun stillSmoothsTheNoise() {
        open().use { denoiser ->
            val frame = Frame(128, 96, 4)
            val clean = denoiser.denoise(frame.width, frame.height, frame.noisy, frame.albedo, frame.normal)
            assertTrue(frame.error(clean) < frame.error(frame.noisy) / 4, "denoised ${frame.error(clean)} vs noisy ${frame.error(frame.noisy)}")
        }
    }

    @Test
    fun reusingTheFilterIsFasterThanTheFirstCall() {
        open().use { denoiser ->
            val frame = Frame(480, 270, 5)
            fun run(): Long {
                val start = System.nanoTime()
                denoiser.denoise(frame.width, frame.height, frame.noisy, frame.albedo, frame.normal)
                return System.nanoTime() - start
            }
            val first = run()
            val later = (1..3).minOf { run() }
            System.err.println("[denoiser] first call ${first / 1_000_000} ms, later calls ${later / 1_000_000} ms")
            assertTrue(later <= first, "a later call ($later ns) took longer than the first ($first ns)")
        }
    }
}
