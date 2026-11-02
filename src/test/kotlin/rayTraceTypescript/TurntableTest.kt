package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import rayTraceTypescript.gpu.MeshBlock
import rayTraceTypescript.gpu.PlacedMesh
import rayTraceTypescript.gpu.Placement
import rayTraceTypescript.gpu.TurntableSettings
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.objects.RotateY
import rayTraceTypescript.objects.Translate
import rayTraceTypescript.render.VideoEncoder
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs

class TurntableTest {

    @Test
    fun oneTurnEndsAStepShortOfWhereItStarted() {
        val settings = TurntableSettings(seconds = 2.0, fps = 30)
        assertEquals(60, settings.framesPerTurn)
        assertEquals(0f, settings.degrees(0))
        assertEquals(6f, settings.degrees(1), 1e-4f)
        assertEquals(354f, settings.degrees(59), 1e-3f)
        assertEquals(-6f, settings.copy(reverse = true).degrees(1), 1e-4f)
        // A very short film still turns.
        assertEquals(2, TurntableSettings(seconds = 0.01, fps = 24).framesPerTurn)
    }

    /** A small tent of two triangles, off-centre so a turn moves it. */
    private fun tent(): Mesh = Mesh(
        floatArrayOf(
            0.2f, 0f, -0.5f, 1.4f, 0f, -0.5f, 0.8f, 1f, 0f,
            1.4f, 0f, 0.5f, 0.2f, 0f, 0.5f, 0.8f, 1f, 0f
        )
    )

    @Test
    fun movingAPlacedMeshMatchesMovingItWithInstances() {
        val material = Lambertian(Color(0.5f, 0.5f, 0.5f))
        val placed = PlacedMesh(MeshBlock(tent()), 1.5f, 20f, Vector(0.3f, 0.1f, -0.2f), material)
        val from = Vector(0.5f, 0f, 0.1f)
        val to = Vector(-1.0f, 0.2f, 2.0f)
        val moved = Placement.moved(placed, from, 35f, to)
        assertTrue(moved is PlacedMesh, "a placed mesh stays a placed mesh")
        val wrapped = Translate(RotateY(Translate(placed, Vector(-from.x, -from.y, -from.z)), 35f), to)

        // Rays from all round; both must hit at the same distance, or both miss.
        var hits = 0
        for (i in 0 until 24) {
            val angle = Math.toRadians(i * 15.0)
            val origin = Point(to.x + 6f * kotlin.math.sin(angle).toFloat(), 0.9f, to.z + 6f * kotlin.math.cos(angle).toFloat())
            val ray = Ray(origin, Point(to.x, 0.6f, to.z) - origin, 0f)
            val a = moved.hit(ray, Interval(0.001f, Float.POSITIVE_INFINITY))
            val b = wrapped.hit(ray, Interval(0.001f, Float.POSITIVE_INFINITY))
            assertEquals(a == null, b == null, "ray $i: one hit, the other missed")
            if (a != null && b != null) {
                hits++
                assertTrue(abs(a.t - b.t) < 1e-3f, "ray $i: ${a.t} vs ${b.t}")
            }
        }
        assertTrue(hits > 4, "the rays should mostly find the tent ($hits hits)")
    }

    private fun ffmpegRuns(): Boolean = try {
        ProcessBuilder(VideoEncoder.ffmpeg(), "-version").redirectErrorStream(true).start().let {
            it.inputStream.readAllBytes()
            it.waitFor() == 0
        }
    } catch (e: Exception) {
        false
    }

    @Test
    fun writesAFilmAndRepeatsItsTurn(@TempDir dir: Path) {
        assumeTrue(ffmpegRuns(), "ffmpeg is not available")
        val output = dir.resolve("film.mp4")
        val encoder = VideoEncoder.start(output, 32, 16, 10, 2_000_000)
        repeat(5) { frame -> encoder.write(FloatArray(32 * 16 * 3) { (frame + 1) / 6f }) }
        encoder.finish(turns = 3)
        assertTrue(Files.size(output) > 0)
        assertEquals(listOf("film.mp4"), Files.list(dir).use { s -> s.map { it.fileName.toString() }.toList() })
        val probe = ProcessBuilder(
            VideoEncoder.ffmpeg().replace(Regex("ffmpeg(\\.exe)?$"), "ffprobe$1"),
            "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", output.toString()
        ).redirectErrorStream(true).start()
        val frames = probe.inputStream.bufferedReader().readText().trim()
        if (probe.waitFor() == 0) assertEquals("15", frames, "three turns of five frames")
    }

    @Test
    fun anAbortedFilmLeavesNothing(@TempDir dir: Path) {
        assumeTrue(ffmpegRuns(), "ffmpeg is not available")
        val output = dir.resolve("film.mp4")
        val encoder = VideoEncoder.start(output, 32, 16, 10, 2_000_000)
        encoder.write(FloatArray(32 * 16 * 3) { 0.5f })
        encoder.abort()
        assertFalse(Files.exists(output))
        assertEquals(0L, Files.list(dir).use { it.count() })
    }

    @Test
    fun anOddSizeIsRefused(@TempDir dir: Path) {
        val failure = runCatching { VideoEncoder.start(dir.resolve("odd.mp4"), 33, 16, 10, 2_000_000) }.exceptionOrNull()
        assertNotNull(failure)
    }
}
