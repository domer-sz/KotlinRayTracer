package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.FrameTurn
import rayTraceTypescript.gpu.GpuChoice
import rayTraceTypescript.gpu.GpuRenderer
import rayTraceTypescript.gpu.GpuSession
import rayTraceTypescript.gpu.MeshBlock
import rayTraceTypescript.gpu.PlacedMesh
import rayTraceTypescript.gpu.SceneFlattener
import rayTraceTypescript.gpu.optix.OptixApi
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.render.CameraSetup
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * The flow guide a temporal denoiser is fed: for each pixel whose first hit is a mesh a
 * turntable turns, how far that point moved across the screen since the frame before - here
 * worked out again on the CPU for the ray through the pixel's centre, and held to what OptiX
 * gathered. Points of the floor, which stands still, must not move at all.
 */
class FlowGuideTest {

    @Test
    fun `optix flow follows a turning mesh and leaves the floor still`() {
        val gpu = GpuRenderer.unavailabilityReason()
        assumeTrue(gpu == null, "No CUDA device available: $gpu")
        assumeTrue(OptixApi.unavailabilityReason() == null, "OptiX is not available")

        val setup = camera().initialize()
        val figureMaterial = Lambertian(Color(0.7f, 0.5f, 0.3f))
        val turn = FrameTurn(axisX = 0.2f, axisZ = -0.1f, degrees = 6.0f)
        val figure = PlacedMesh(MeshBlock(octahedron()), 1.4f, 20.0f, Vector(0.2f, 0.0f, -0.1f), figureMaterial, turning = true)
        val world = HittableList(mutableListOf(floor(), figure))

        val flow = GpuSession.open(SceneFlattener.scene(world), setup.totalPixels, GpuChoice(gpus = "0:optix")).use { session ->
            assertTrue(session.tracesFlow)
            session.setCamera(setup)
            session.setTurn(turn)
            GpuRenderer.renderGuides(session, setup, 64)
            FloatArray(setup.totalPixels * 3).also { session.read(GpuSession.FLOW, it) }
        }

        var meshPixels = 0
        var floorPixels = 0
        var largest = 0.0f
        var errorSum = 0.0f
        var worst = 0.0f
        for (y in 2 until setup.imageHeight - 2) for (x in 2 until setup.imageWidth - 2) {
            // Only pixels well inside one surface: an edge mixes the flows of two.
            val hits = (-1..1).flatMap { dy -> (-1..1).map { dx -> firstHit(world, setup, x + dx, y + dy) } }
            if (hits.any { it == null }) continue
            val onMesh = hits.all { it!!.material === figureMaterial }
            val onFloor = hits.none { it!!.material === figureMaterial }
            val gx = flow[(y * setup.imageWidth + x) * 3]
            val gy = flow[(y * setup.imageWidth + x) * 3 + 1]
            if (onMesh) {
                val (ex, ey) = expectedFlow(hits[4]!!.point, turn, setup)
                // The guide averages 64 jittered rays over the pixel, the centre ray stands for them:
                // where the flow bends within a pixel (a ridge, a steep slope) the two part a little.
                val error = maxOf(abs(ex - gx), abs(ey - gy))
                errorSum += error
                worst = maxOf(worst, error)
                largest = maxOf(largest, abs(ex), abs(ey))
                meshPixels++
            } else if (onFloor) {
                // Not exactly zero: the two projections of the same point round apart by a millionth.
                assertEquals(0.0f, gx, 1e-4f, "the floor at ($x, $y) does not move")
                assertEquals(0.0f, gy, 1e-4f, "the floor at ($x, $y) does not move")
                floorPixels++
            }
        }
        println("flow: $meshPixels mesh pixels, mean error ${errorSum / meshPixels} px, worst $worst px, largest flow $largest px")
        assertTrue(errorSum / meshPixels < 0.02f, "the flow is off by ${errorSum / meshPixels} pixels on average")
        assertTrue(worst < 0.15f, "the flow is off by $worst pixels at worst")
        assertTrue(meshPixels > 200, "only $meshPixels pixels inside the mesh")
        assertTrue(floorPixels > 200, "only $floorPixels pixels inside the floor")
        // A turn this large must move the mesh by whole pixels, or the test proves little.
        assertTrue(largest > 1.0f, "the mesh moved at most $largest pixels")
    }

    @Test
    fun `cuda flow follows a moving camera`() = cameraFlow("0:cuda")

    @Test
    fun `optix flow follows a moving camera`() {
        // The card first: OptiX's library is found beside the CUDA one, once that is loaded.
        assumeTrue(GpuRenderer.unavailabilityReason() == null, "No CUDA device available")
        assumeTrue(OptixApi.unavailabilityReason() == null, "OptiX is not available")
        cameraFlow("0:optix")
    }

    /**
     * A still scene and a camera that orbited a little since the frame before, as the viewer's
     * does while dragged: each hit moves across the screen by where the camera sees it now less
     * where it saw it then.
     */
    private fun cameraFlow(gpus: String) {
        val gpu = GpuRenderer.unavailabilityReason()
        assumeTrue(gpu == null, "No CUDA device available: $gpu")
        val setup = camera().initialize()
        val previous = camera().apply { lookFrom = Point(4.3f, 2.4f, 4.7f) }.initialize()
        val figure = PlacedMesh(MeshBlock(octahedron()), 1.4f, 20.0f, Vector(0.2f, 0.0f, -0.1f), Lambertian(Color(0.7f, 0.5f, 0.3f)))
        val world = HittableList(mutableListOf(floor(), figure))

        val flow = GpuSession.open(SceneFlattener.scene(world), setup.totalPixels, GpuChoice(gpus = gpus)).use { session ->
            session.setCamera(setup, previous)
            GpuRenderer.renderGuides(session, setup, 64)
            FloatArray(setup.totalPixels * 3).also { session.read(GpuSession.FLOW, it) }
        }

        var counted = 0
        var errorSum = 0.0f
        var largest = 0.0f
        for (y in 2 until setup.imageHeight - 2) for (x in 2 until setup.imageWidth - 2) {
            val hits = (-1..1).flatMap { dy -> (-1..1).map { dx -> firstHit(world, setup, x + dx, y + dy) } }
            if (hits.any { it == null } || hits.map { it!!.material }.toSet().size != 1) continue
            val p = hits[4]!!.point
            val (nx, ny) = screen(p, setup)
            val (px, py) = screen(p, previous)
            val ex = nx - px
            val ey = ny - py
            errorSum += maxOf(abs(ex - flow[(y * setup.imageWidth + x) * 3]), abs(ey - flow[(y * setup.imageWidth + x) * 3 + 1]))
            largest = maxOf(largest, abs(ex), abs(ey))
            counted++
        }
        println("camera flow ($gpus): $counted pixels, mean error ${errorSum / counted} px, largest flow $largest px")
        assertTrue(counted > 1000, "only $counted pixels inside one surface")
        assertTrue(largest > 2.0f, "the camera moved the image by at most $largest pixels")
        assertTrue(errorSum / counted < 0.03f, "the flow is off by ${errorSum / counted} pixels on average")
    }

    private fun firstHit(world: HittableList, setup: CameraSetup, x: Int, y: Int) =
        world.hit(Ray(setup.cameraCenter, setup.pixel00 + setup.pixelDeltaU * x.toFloat() + setup.pixelDeltaV * y.toFloat() - setup.cameraCenter), Interval(0.001f, Float.POSITIVE_INFINITY))

    /** Where [p] moved on the screen in the frame's turn: its position now, less where it stood a frame ago. */
    private fun expectedFlow(p: Vector, turn: FrameTurn, setup: CameraSetup): Pair<Float, Float> {
        // A frame ago the mesh stood one turn back, turning as Placement.turned does.
        val back = Math.toRadians(-turn.degrees.toDouble())
        val c = cos(back).toFloat()
        val s = sin(back).toFloat()
        val dx = p.x - turn.axisX
        val dz = p.z - turn.axisZ
        val before = Vector(turn.axisX + c * dx + s * dz, p.y, turn.axisZ - s * dx + c * dz)
        val (nx, ny) = screen(p, setup)
        val (bx, by) = screen(before, setup)
        return (nx - bx) to (ny - by)
    }

    private fun screen(p: Vector, setup: CameraSetup): Pair<Float, Float> {
        fun dot(a: Vector, b: Vector) = Vector.dotProduct(a, b)
        val du = setup.pixelDeltaU
        val dv = setup.pixelDeltaV
        val n = Vector.cross(du, dv)
        val d = p - setup.cameraCenter
        val q = d * (dot(setup.pixel00 - setup.cameraCenter, n) / dot(d, n)) + setup.cameraCenter - setup.pixel00
        return dot(q, du) / dot(du, du) to dot(q, dv) / dot(dv, dv)
    }

    private fun camera() = Camera().apply {
        aspectRatio = 4.0f / 3.0f
        imageWidth = 160
        samplesPerPixel = 64
        vfov = 30.0f
        lookFrom = Point(4.0f, 2.5f, 5.0f)
        lookAt = Point(0.2f, 0.6f, -0.1f)
        vUp = Vector(0.0f, 1.0f, 0.0f)
        defocusAngle = 0.0f
    }

    private fun floor() = Quad(
        Point(-6.0f, 0.0f, -6.0f), Vector(12.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 12.0f),
        Lambertian(Color(0.6f, 0.6f, 0.62f))
    )

    /** A unit octahedron standing on y = 0. */
    private fun octahedron(): Mesh = Mesh(
        floatArrayOf(
            1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f,
            0f, 1f, 0f, -1f, 0f, 0f, 0f, 0f, 1f,
            -1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, 1f,
            0f, -1f, 0f, 1f, 0f, 0f, 0f, 0f, 1f,
            0f, 1f, 0f, 1f, 0f, 0f, 0f, 0f, -1f,
            -1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, -1f,
            0f, -1f, 0f, -1f, 0f, 0f, 0f, 0f, -1f,
            1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, -1f
        )
    ).standingOnFloor()
}
