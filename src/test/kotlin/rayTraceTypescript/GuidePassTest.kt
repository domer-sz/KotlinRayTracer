package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.GpuChoice
import rayTraceTypescript.gpu.GpuRenderer
import rayTraceTypescript.gpu.GpuSession
import rayTraceTypescript.gpu.MeshBlock
import rayTraceTypescript.gpu.PlacedMesh
import rayTraceTypescript.gpu.SceneFlattener
import rayTraceTypescript.gpu.optix.OptixApi
import rayTraceTypescript.materials.DiffuseLight
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Metal
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.render.CameraSetup
import kotlin.math.abs

/**
 * The one guide pass ([GpuSession.MODE_GUIDES]) must give the albedo and the normal the two
 * separate guide passes give: the camera rays are the same - keyed by pixel and sample - so only
 * the order of float sums may differ. On each engine.
 */
class GuidePassTest {

    @Test
    fun `the cuda kernel traces both guides in one pass`() = check("cuda")

    @Test
    fun `optix traces both guides in one pass`() {
        assumeTrue(OptixApi.unavailabilityReason() == null, "OptiX is not available")
        check("optix")
    }

    private fun check(backend: String) {
        val gpu = GpuRenderer.unavailabilityReason()
        assumeTrue(gpu == null, "No CUDA device available: $gpu")
        val setup = camera().initialize()
        val light = Quad(
            Point(-1.5f, 3.0f, -1.5f), Vector(3.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 3.0f),
            DiffuseLight(Color(6.0f, 6.0f, 6.0f))
        )
        val block = MeshBlock(octahedron().smoothed())
        val world = HittableList(
            mutableListOf(
                Quad(Point(-6.0f, 0.0f, -6.0f), Vector(12.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 12.0f), Lambertian(Color(0.6f, 0.6f, 0.62f))),
                PlacedMesh(block, 1.0f, 30.0f, Vector(-1.1f, 0.0f, 0.3f), Metal(Color(0.9f, 0.7f, 0.3f), 0.05f)),
                PlacedMesh(block, 0.8f, -75.0f, Vector(0.9f, 0.0f, -0.4f), Lambertian(Color(0.2f, 0.4f, 0.7f))),
                light
            )
        )
        GpuSession.open(SceneFlattener.scene(world, light), setup.totalPixels, GpuChoice(gpu = "0", backend = backend)).use { session ->
            session.seed = 4242L
            session.setCamera(setup)
            fun trace(mode: Int, target: Int, vararg reads: Int): List<FloatArray> {
                session.traceSliced(mode, setup, SAMPLES, target, 20.0)
                return reads.map { t -> FloatArray(setup.totalPixels * 3).also { session.read(t, it) } }
            }
            val (albedo, normal) = trace(GpuSession.MODE_GUIDES, GpuSession.ALBEDO, GpuSession.ALBEDO, GpuSession.NORMAL)
            val (albedoPass) = trace(GpuSession.MODE_ALBEDO, GpuSession.GUIDE, GpuSession.GUIDE)
            val (normalPass) = trace(GpuSession.MODE_NORMAL, GpuSession.GUIDE, GpuSession.GUIDE)

            assertClose("albedo", albedoPass, albedo, 0.002f, setup)
            assertClose("normal", normalPass, normal, 0.002f, setup)
            assertTrue(albedo.average() > 0.05f, "the albedo guide is empty")
        }
    }

    private fun assertClose(what: String, expected: FloatArray, actual: FloatArray, tolerance: Float, setup: CameraSetup) {
        var sum = 0.0
        for (i in expected.indices) sum += abs(expected[i] - actual[i])
        val mean = sum / expected.size
        assertTrue(mean <= tolerance, "the $what from the one guide pass differs by $mean on average at ${setup.imageWidth}x${setup.imageHeight}")
    }

    private fun camera() = Camera().apply {
        aspectRatio = 16.0f / 9.0f
        imageWidth = 96
        samplesPerPixel = SAMPLES
        maxReflectionDepth = 12
        vfov = 25.0f
        lookFrom = Point(6.0f, 2.0f, 6.0f)
        lookAt = Point(0.0f, 0.5f, 0.0f)
        vUp = Vector(0.0f, 1.0f, 0.0f)
        background = Color(0.05f, 0.05f, 0.06f)
    }

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

    private companion object {
        const val SAMPLES = 32
    }
}
