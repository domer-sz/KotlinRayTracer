package rayTraceTypescript

import kotlin.math.*
import rayTraceTypescript.utils.degreesToRadians
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.render.CameraSetup
import java.nio.file.Path
import java.nio.file.Paths

class Camera {
    var aspectRatio: Float = 16.0f / 9.0f
    var imageWidth: Int = 400
    var samplesPerPixel: Int = 100
    var maxReflectionDepth: Int = 10

    var vfov: Float = 90.0f
    var lookFrom: Point = Point(0.0f, 0.0f, 0.0f)
    var lookAt: Point = Point(0.0f, 0.0f, -1.0f)
    var vUp: Vector = Vector(0.0f, 1.0f, 0.0f)

    var defocusAngle: Float = 0.0f
    var focusDistance: Float = 10.0f

    /** Colour of rays that escape the scene; null keeps the sky gradient of the first book. */
    var background: Color? = null

    /**
     * Shapes worth aiming samples at - the lights. Given them, every diffuse bounce sends half
     * its samples straight at a light, which is what clears the noise out of a dim room.
     */
    var lights: Hittable? = null

    /**
     * Renders [world] into [outputPath] and returns the number of primary rays traced.
     * It renders on the GPU through CUDA ([rayTraceTypescript.gpu.GpuRenderer]); without a CUDA
     * device it fails with the reason rather than falling back to anything slower.
     */
    fun render(world: Hittable, outputPath: Path = Paths.get("image.ppm")): Long {
        rayTraceTypescript.gpu.GpuRenderer.unavailabilityReason()?.let { throw IllegalStateException("GPU rendering unavailable: $it") }
        return rayTraceTypescript.gpu.GpuRenderer().render(world, initialize(), outputPath)
    }

    fun initialize(): CameraSetup {
        var imageHeight = floor(imageWidth.toFloat() / aspectRatio).toInt()
        if (imageHeight < 1) imageHeight = 1

        val pixelSamplesScale = 1.0f / samplesPerPixel.toFloat()
        val cameraCenter = lookFrom

        val theta = degreesToRadians(vfov)
        val h = tan((theta / 2.0f).toDouble()).toFloat()
        val viewportHeight = 2.0f * h * focusDistance
        val viewportWidth = viewportHeight * (imageWidth.toFloat() / imageHeight.toFloat())

        val w = Vector.unit(lookFrom - lookAt)
        val u = Vector.unit(Vector.cross(vUp, w))
        val v = Vector.cross(w, u)

        val viewportU = u * viewportWidth
        val viewportV = (-v) * viewportHeight

        val pixelDeltaU = viewportU / imageWidth.toFloat()
        val pixelDeltaV = viewportV / imageHeight.toFloat()

        val viewportCorner = cameraCenter - w * focusDistance
        val viewportUpperLeftShiftedU = viewportCorner - (viewportU / 2.0f)
        val viewportUpperLeft = viewportUpperLeftShiftedU - (viewportV / 2.0f)

        val pixel00 = Point(
            viewportUpperLeft.x + pixelDeltaU.x * 0.5f + pixelDeltaV.x * 0.5f,
            viewportUpperLeft.y + pixelDeltaU.y * 0.5f + pixelDeltaV.y * 0.5f,
            viewportUpperLeft.z + pixelDeltaU.z * 0.5f + pixelDeltaV.z * 0.5f
        )

        val defocusRadius =
            focusDistance * tan((degreesToRadians(defocusAngle) / 2.0f).toDouble()).toFloat()

        return CameraSetup(
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            samplesPerPixel = samplesPerPixel,
            maxReflectionDepth = maxReflectionDepth,
            pixelSamplesScale = pixelSamplesScale,
            cameraCenter = cameraCenter,
            pixel00 = pixel00,
            pixelDeltaU = pixelDeltaU,
            pixelDeltaV = pixelDeltaV,
            defocusAngle = defocusAngle,
            background = background,
            lights = lights,
            defocusDiscU = u * defocusRadius,
            defocusDiscV = v * defocusRadius
        )
    }
}
