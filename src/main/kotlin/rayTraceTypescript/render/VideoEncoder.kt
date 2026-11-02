package rayTraceTypescript.render

import java.io.BufferedOutputStream
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import kotlin.math.sqrt

/**
 * An H.264 MP4 written by ffmpeg, fed raw frames through its standard input - no video library
 * in the build. ffmpeg comes from -Drt.ffmpeg, the FFMPEG environment variable, or the PATH.
 *
 * Frames go to `<name>.part.mp4` beside the target, which takes the real name only once the
 * film is whole: a cancelled or failed film leaves nothing behind that looks finished.
 */
class VideoEncoder private constructor(
    private val ffmpeg: String,
    private val process: Process,
    private val partial: Path,
    private val output: Path,
    val width: Int,
    val height: Int
) {
    private val input = BufferedOutputStream(process.outputStream, 1 shl 20)
    private val errors = StringBuilder()
    // ffmpeg blocks once its error pipe fills; whatever it says is kept for the failure message.
    private val errorReader = Thread({
        process.errorStream.bufferedReader().forEachLine { synchronized(errors) { errors.appendLine(it) } }
    }, "ffmpeg-errors").apply { isDaemon = true; start() }
    private val frame = ByteArray(width * height * 3)

    /**
     * Appends one frame of linear colour (3 floats per pixel, rows top to bottom), with the same
     * sqrt gamma and clamp as the PPM and PNG files.
     */
    fun write(linear: FloatArray) {
        require(linear.size == frame.size) { "frame of ${linear.size / 3} pixels, film is ${width}x$height" }
        for (i in frame.indices) {
            val value = linear[i]
            val gamma = if (value > 0f) sqrt(value).coerceAtMost(0.999f) else 0f
            frame[i] = (256f * gamma).toInt().toByte()
        }
        try {
            input.write(frame)
        } catch (e: IOException) {
            throw IllegalStateException("ffmpeg stopped taking frames: ${failure()}", e)
        }
    }

    /**
     * Ends the film: the frames written so far make one turn, which is played [turns] times
     * (copied, not encoded again - every turn starts on a key frame), and the file takes its name.
     */
    fun finish(turns: Int = 1) {
        input.close()
        val code = process.waitFor()
        errorReader.join(2000)
        check(code == 0) { "ffmpeg failed ($code): ${failure()}" }
        if (turns > 1) {
            val looped = partial.resolveSibling(partial.fileName.toString().removeSuffix(".part.mp4") + ".loop.part.mp4")
            run(
                listOf(
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-stream_loop", (turns - 1).toString(), "-i", partial.toString(),
                    "-c", "copy", "-movflags", "+faststart", "-f", "mp4", looped.toString()
                )
            )
            Files.move(looped, partial, StandardCopyOption.REPLACE_EXISTING)
        }
        try {
            Files.move(partial, output, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
        } catch (e: AtomicMoveNotSupportedException) {
            Files.move(partial, output, StandardCopyOption.REPLACE_EXISTING)
        }
    }

    /** Stops ffmpeg and removes the unfinished file. */
    fun abort() {
        try {
            input.close()
        } catch (_: IOException) {
        }
        process.destroyForcibly()
        process.waitFor()
        Files.deleteIfExists(partial)
    }

    private fun failure(): String = synchronized(errors) { errors.toString().trim().ifEmpty { "no message" } }

    companion object {
        /** The ffmpeg to run: -Drt.ffmpeg, FFMPEG, or whatever the PATH finds. */
        fun ffmpeg(): String = System.getProperty("rt.ffmpeg") ?: System.getenv("FFMPEG") ?: "ffmpeg"

        /**
         * Bits per second for a film of this size: about 0.085 bits per pixel and frame for the
         * smaller file, 0.15 for the best one (StlViewer's two settings), within 2 to 40 Mbit/s.
         */
        fun bitrate(width: Int, height: Int, fps: Int, compact: Boolean): Long {
            val bitsPerPixel = if (compact) 0.085 else 0.15
            return (width.toLong() * height * fps * bitsPerPixel).toLong().coerceIn(2_000_000L, 40_000_000L)
        }

        /** Starts ffmpeg for a film of [width] x [height] at [fps], ending up in [output]. */
        fun start(output: Path, width: Int, height: Int, fps: Int, bitsPerSecond: Long): VideoEncoder {
            // 4:2:0 colour halves both sides: H.264 players expect them even.
            require(width % 2 == 0 && height % 2 == 0) { "a film needs an even width and height, not ${width}x$height" }
            val target = output.toAbsolutePath()
            target.parent?.let { Files.createDirectories(it) }
            val stem = target.fileName.toString().replace(Regex("\\.mp4$", RegexOption.IGNORE_CASE), "")
            val partial = target.resolveSibling("$stem.part.mp4")
            val tool = ffmpeg()
            val command = listOf(
                tool, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "${width}x$height", "-framerate", fps.toString(), "-i", "-",
                "-an",
                // Pixels are sRGB-like: convert with the HD matrix and say so in the file.
                "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                "-c:v", "libx264", "-preset", "medium", "-profile:v", "high", "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                "-b:v", bitsPerSecond.toString(),
                "-maxrate", (bitsPerSecond * 3 / 2).toString(), "-bufsize", (bitsPerSecond * 2).toString(),
                // A key frame every two seconds, the first frame among them.
                "-g", (2 * fps).toString(),
                "-movflags", "+faststart", "-f", "mp4", partial.toString()
            )
            val process = try {
                ProcessBuilder(command).start()
            } catch (e: IOException) {
                throw IllegalStateException("could not start ffmpeg ('$tool'): install it, or point -Drt.ffmpeg / FFMPEG at it", e)
            }
            return VideoEncoder(tool, process, partial, target, width, height)
        }

        private fun run(command: List<String>) {
            val process = ProcessBuilder(command).redirectErrorStream(true).start()
            val said = process.inputStream.bufferedReader().readText()
            val code = process.waitFor()
            check(code == 0) { "ffmpeg failed ($code): ${said.trim()}" }
        }
    }
}
