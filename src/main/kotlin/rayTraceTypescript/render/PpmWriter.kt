package rayTraceTypescript.render

import rayTraceTypescript.Color
import java.awt.image.BufferedImage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import javax.imageio.ImageIO

/**
 * Linear RGB samples -> ASCII PPM, plus the same pixels as a PNG beside it (see [pngPath]):
 * the PPM is the project's reference format, the PNG a sixth of the size and viewable anywhere.
 * Quantization goes through [ToneMap] once for both, as the viewer and films do, so the two files
 * always hold identical pixels.
 */
object PpmWriter {
    fun write(outputPath: Path, width: Int, height: Int, linearRgb: FloatArray) {
        require(linearRgb.size == width * height * 3) { "Expected ${width * height * 3} samples, got ${linearRgb.size}" }
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        val sb = StringBuilder("P3\n${width} ${height}\n255\n")
        val tone = ToneMap.current()
        for (pixel in 0 until width * height) {
            val r = tone.byte(linearRgb[pixel * 3])
            val g = tone.byte(linearRgb[pixel * 3 + 1])
            val b = tone.byte(linearRgb[pixel * 3 + 2])
            sb.append("$r $g $b\n")
            image.setRGB(pixel % width, pixel / width, (r shl 16) or (g shl 8) or b)
        }
        Files.write(outputPath, sb.toString().toByteArray(StandardCharsets.UTF_8))
        ImageIO.write(image, "png", pngPath(outputPath).toFile())
        // -Drt.pfm also keeps the unclipped linear pixels, for a denoiser or later exposure changes.
        if (System.getProperty("rt.pfm") != null) writePfm(siblingPath(outputPath, "pfm"), width, height, linearRgb)
    }

    /** Where the PNG copy of [outputPath] goes: the same name with a .png extension. */
    fun pngPath(outputPath: Path): Path = siblingPath(outputPath, "png")

    /** Little-endian colour PFM, rows stored bottom to top as the format wants. */
    private fun writePfm(path: Path, width: Int, height: Int, linearRgb: FloatArray) {
        val header = "PF\n$width $height\n-1.0\n".toByteArray(StandardCharsets.US_ASCII)
        val body = ByteBuffer.allocate(width * height * 12).order(ByteOrder.LITTLE_ENDIAN)
        for (row in height - 1 downTo 0) {
            for (i in row * width * 3 until (row + 1) * width * 3) body.putFloat(linearRgb[i])
        }
        Files.write(path, header + body.array())
    }

    private fun siblingPath(outputPath: Path, extension: String): Path {
        val name = outputPath.fileName.toString()
        val stem = if (name.lowercase().endsWith(".ppm")) name.dropLast(4) else name
        return outputPath.resolveSibling("$stem.$extension")
    }
}
