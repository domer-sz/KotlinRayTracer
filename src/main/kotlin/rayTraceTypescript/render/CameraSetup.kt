package rayTraceTypescript.render

import rayTraceTypescript.Color
import rayTraceTypescript.Point
import rayTraceTypescript.Vector

/**
 * Result of Camera.initialize(): everything a renderer needs to shoot rays.
 * Computed once on the CPU so that every backend starts from identical numbers.
 */
data class CameraSetup(
    val imageWidth: Int,
    val imageHeight: Int,
    val samplesPerPixel: Int,
    val maxReflectionDepth: Int,
    val pixelSamplesScale: Float,
    val cameraCenter: Point,
    val pixel00: Point,
    val pixelDeltaU: Vector,
    val pixelDeltaV: Vector,
    val defocusAngle: Float,
    val background: Color? = null,
    val lights: rayTraceTypescript.objects.Hittable? = null,
    val defocusDiscU: Vector,
    val defocusDiscV: Vector,
    /** The brightest a sample may be, 0 for no limit (see Camera.fireflyClamp). */
    val fireflyClamp: Float = 0.0f,
    /** Where escaping rays take their colour from, instead of [background] or the gradient (see Camera.environment). */
    val environment: rayTraceTypescript.textures.Environment? = null
) {
    val totalPixels: Int get() = imageWidth * imageHeight

    /**
     * The same view at half the width and height, each pixel covering a 2x2 block of this one's:
     * its centre is the middle of the block. What a denoiser that enlarges twice is handed.
     */
    fun halved(): CameraSetup {
        require(imageWidth % 2 == 0 && imageHeight % 2 == 0) { "a ${imageWidth}x$imageHeight image cannot be halved" }
        return copy(
            imageWidth = imageWidth / 2,
            imageHeight = imageHeight / 2,
            pixel00 = (pixel00 + pixelDeltaU * 0.5f + pixelDeltaV * 0.5f).let { Point(it.x, it.y, it.z) },
            pixelDeltaU = pixelDeltaU * 2.0f,
            pixelDeltaV = pixelDeltaV * 2.0f
        )
    }
}
