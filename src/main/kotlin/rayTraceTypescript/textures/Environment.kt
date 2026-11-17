package rayTraceTypescript.textures

import java.nio.file.Files
import java.nio.file.Path

/**
 * The light from everywhere far away: an equirectangular (latitude-longitude) HDR image, which a
 * ray that hits nothing takes its colour from instead of the sky gradient. Rows run from straight
 * up to straight down, columns once round the horizon, starting behind the camera's -z.
 *
 * [rgb] holds linear floats, 3 a pixel; [intensity] scales them, [turnDegrees] turns the map about
 * the vertical, to bring its sun round to where the scene's stands.
 */
class Environment(
    val width: Int,
    val height: Int,
    val rgb: FloatArray,
    val intensity: Float = 1.0f,
    val turnDegrees: Float = 0.0f
) {
    init {
        require(width > 0 && height > 0 && rgb.size == width * height * 3) { "an environment of ${width}x$height needs ${width * height * 3} floats" }
    }

    fun with(intensity: Float = this.intensity, turnDegrees: Float = this.turnDegrees) =
        Environment(width, height, rgb, intensity, turnDegrees)

    companion object {
        /**
         * Reads a Radiance .hdr (RGBE, flat or run-length encoded scanlines, -Y +X orientation), and
         * halves it until it is at most [maxWidth] wide: the map lights the scene and shows behind
         * it, and a 4k one would take 100 MB on the card for no visible gain.
         */
        fun load(path: Path, maxWidth: Int = 2048): Environment {
            val bytes = Files.readAllBytes(path)
            var at = 0
            fun line(): String {
                val start = at
                while (at < bytes.size && bytes[at] != '\n'.code.toByte()) at++
                return String(bytes, start, at++ - start, Charsets.US_ASCII).trim()
            }
            require(line().startsWith("#?")) { "$path is not a Radiance HDR file" }
            while (true) {
                val l = line()
                if (l.isEmpty()) break
                require(!l.startsWith("FORMAT=") || l == "FORMAT=32-bit_rle_rgbe") { "$path: unsupported $l" }
            }
            val size = line().split(Regex("[ \\t]+"))
            require(size.size == 4 && size[0] == "-Y" && size[2] == "+X") { "$path: unsupported orientation ${size.joinToString(" ")}" }
            val height = size[1].toInt()
            val width = size[3].toInt()

            val rgbe = ByteArray(width * 4)
            var full = FloatArray(width * height * 3)
            for (y in 0 until height) {
                val newRle = width in 8..32767 && at + 4 <= bytes.size && bytes[at].toInt() == 2 && bytes[at + 1].toInt() == 2 &&
                    ((bytes[at + 2].toInt() and 0xFF) shl 8 or (bytes[at + 3].toInt() and 0xFF)) == width
                if (newRle) {
                    at += 4
                    // Each of the four channels on its own, in runs or literal stretches.
                    for (channel in 0 until 4) {
                        var x = 0
                        while (x < width) {
                            var count = bytes[at++].toInt() and 0xFF
                            if (count > 128) {
                                count -= 128
                                val value = bytes[at++]
                                repeat(count) { rgbe[(x++) * 4 + channel] = value }
                            } else {
                                repeat(count) { rgbe[(x++) * 4 + channel] = bytes[at++] }
                            }
                        }
                    }
                } else {
                    System.arraycopy(bytes, at, rgbe, 0, width * 4)
                    at += width * 4
                }
                for (x in 0 until width) {
                    val e = rgbe[x * 4 + 3].toInt() and 0xFF
                    val o = (y * width + x) * 3
                    if (e == 0) continue
                    val f = Math.scalb(1.0f, e - 136)
                    full[o] = (rgbe[x * 4].toInt() and 0xFF) * f
                    full[o + 1] = (rgbe[x * 4 + 1].toInt() and 0xFF) * f
                    full[o + 2] = (rgbe[x * 4 + 2].toInt() and 0xFF) * f
                }
            }

            var w = width
            var h = height
            while (w > maxWidth && w % 2 == 0 && h % 2 == 0) {
                val half = FloatArray((w / 2) * (h / 2) * 3)
                for (y in 0 until h / 2) for (x in 0 until w / 2) for (c in 0 until 3) {
                    half[(y * (w / 2) + x) * 3 + c] = 0.25f * (
                        full[((2 * y) * w + 2 * x) * 3 + c] + full[((2 * y) * w + 2 * x + 1) * 3 + c] +
                            full[((2 * y + 1) * w + 2 * x) * 3 + c] + full[((2 * y + 1) * w + 2 * x + 1) * 3 + c])
                }
                full = half
                w /= 2
                h /= 2
            }
            return Environment(w, h, full)
        }
    }
}
