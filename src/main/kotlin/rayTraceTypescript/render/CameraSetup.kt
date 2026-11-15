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
    val fireflyClamp: Float = 0.0f
) {
    val totalPixels: Int get() = imageWidth * imageHeight
}
