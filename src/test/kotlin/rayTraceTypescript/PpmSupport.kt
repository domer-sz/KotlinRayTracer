package rayTraceTypescript

import java.nio.charset.StandardCharsets
import kotlin.math.abs

/** Shared PPM parsing and comparison helpers for the image tests. */
object PpmSupport {

    data class PpmImage(val width: Int, val height: Int, val pixels: IntArray)

    data class DiffStats(val meanAbsDiff: Float, val maxDiff: Int, val highDiffCount: Int)

    fun parse(bytes: ByteArray): PpmImage {
        val content = bytes.toString(StandardCharsets.UTF_8).trim()
        val tokens = content.split(Regex("\\s+"))
        require(tokens.size >= 4) { "Malformed PPM header" }
        require(tokens[0] == "P3") { "Only ASCII PPM supported" }
        val width = tokens[1].toInt()
        val height = tokens[2].toInt()
        val maxVal = tokens[3].toInt()
        require(maxVal == 255) { "Unexpected max color value $maxVal" }
        val expectedValues = width * height * 3
        val remaining = tokens.drop(4)
        require(remaining.size >= expectedValues) { "Not enough pixel data" }
        val pixels = IntArray(expectedValues) { idx -> remaining[idx].toInt() }
        return PpmImage(width, height, pixels)
    }

    /** [image] as an ASCII PPM again, for storing a reference. */
    fun encode(image: PpmImage): ByteArray {
        val text = StringBuilder("P3\n${image.width} ${image.height}\n255\n")
        for (i in image.pixels.indices step 3) {
            text.append(image.pixels[i]).append(' ').append(image.pixels[i + 1]).append(' ').append(image.pixels[i + 2]).append('\n')
        }
        return text.toString().toByteArray(StandardCharsets.UTF_8)
    }

    fun diff(expected: PpmImage, actual: PpmImage, highDiffThreshold: Int): DiffStats {
        require(expected.pixels.size == actual.pixels.size) { "Pixel counts differ" }
        var sum = 0.0f
        var maxDiff = 0
        var highDiffCount = 0
        expected.pixels.indices.forEach { idx ->
            val delta = abs(expected.pixels[idx] - actual.pixels[idx])
            sum += delta.toFloat()
            if (delta > maxDiff) maxDiff = delta
            if (delta >= highDiffThreshold) highDiffCount += 1
        }
        return DiffStats(sum / expected.pixels.size.toFloat(), maxDiff, highDiffCount)
    }
}
