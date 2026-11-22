package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.GpuChoice
import rayTraceTypescript.gpu.GpuRenderer
import rayTraceTypescript.gpu.GpuSession
import rayTraceTypescript.gpu.SceneFlattener
import rayTraceTypescript.gpu.optix.OptixApi
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.DiffuseLight
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.objects.ConstantMedium
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.objects.box
import kotlin.math.abs

/**
 * Photon mapping (-Drt.caustics) must bring the light the path tracer gives up, no more and no
 * less: under a thick glass slab lit by a small lamp, the floor seen through the slab comes out
 * as bright with the photons as by brute force (which finds the lamp through the glass by chance,
 * so it takes many samples), and the open floor, which no photon reaches, the same either way.
 * The same with smoke under the slab and a black floor: photons stopped in the smoke past the
 * glass light it as brute force does (caustic_gather_volume), and the smoke beside the slab, lit
 * straight from the lamp, is the path tracer's either way.
 */
class CausticsTest {

    @Test
    fun `the cuda kernel's photons light the floor under glass as brute force does`() = check("cuda")

    @Test
    fun `optix's photons light the floor under glass as brute force does`() {
        assumeTrue(OptixApi.unavailabilityReason() == null, "OptiX is not available")
        check("optix")
    }

    @Test
    fun `the cuda kernel's photons light smoke under glass as brute force does`() = check("cuda", smoke = true)

    @Test
    fun `optix's photons light smoke under glass as brute force does`() {
        assumeTrue(OptixApi.unavailabilityReason() == null, "OptiX is not available")
        check("optix", smoke = true)
    }

    private fun check(backend: String, smoke: Boolean = false) {
        val gpu = GpuRenderer.unavailabilityReason()
        assumeTrue(gpu == null, "No CUDA device available: $gpu")
        val setup = Camera().apply {
            aspectRatio = 1.0f
            imageWidth = 64
            maxReflectionDepth = 12
            vfov = 40.0f
            lookFrom = Point(0.0f, 5.0f, 0.5f)
            lookAt = Point(0.0f, 0.0f, 0.0f)
            vUp = Vector(0.0f, 1.0f, 0.0f)
            background = Color(0.0f, 0.0f, 0.0f)
            fireflyClamp = 0f
        }.initialize()
        val lamp = Sphere(Point(0.3f, 6.0f, -0.2f), 0.6f, DiffuseLight(Color(30.0f, 30.0f, 30.0f)))
        val floor = Lambertian(if (smoke) Color(0.0f, 0.0f, 0.0f) else Color(0.6f, 0.6f, 0.6f))
        val world = HittableList(
            mutableListOf(
                Quad(Point(-8.0f, 0.0f, -8.0f), Vector(16.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 16.0f), floor),
                box(Point(-1.0f, 0.5f, -1.0f), Point(1.0f, 0.8f, 1.0f), Dielectric(1.5f)),
                lamp
            )
        )
        if (smoke) world.add(ConstantMedium(box(Point(-3.0f, 0.05f, -3.0f), Point(3.0f, 0.45f, 3.0f), floor), 1.5f, Color(0.8f, 0.8f, 0.8f)))
        fun render(caustics: Boolean, samples: Int): FloatArray {
            if (caustics) System.setProperty("rt.caustics", "on") else System.clearProperty("rt.caustics")
            try {
                GpuSession.open(SceneFlattener.scene(world, lamp), setup.totalPixels, GpuChoice(gpu = "0", backend = backend)).use { session ->
                    session.seed = 77L
                    session.setCamera(setup)
                    session.traceSliced(GpuSession.MODE_IMAGE, setup, samples, GpuSession.IMAGE, 50.0)
                    return FloatArray(setup.totalPixels * 3).also { session.read(GpuSession.IMAGE, it) }
                }
            } finally {
                System.clearProperty("rt.caustics")
            }
        }
        val brute = render(false, if (smoke) 32768 else 8192)
        val photons = render(true, 512)

        // The floor seen through the slab - the middle of the picture - and seen open, at the corners.
        val w = setup.imageWidth
        fun mean(image: FloatArray, x0: Int, y0: Int, size: Int): Double {
            var sum = 0.0
            for (y in y0 until y0 + size) for (x in x0 until x0 + size) for (c in 0..2) sum += image[(y * w + x) * 3 + c]
            return sum / (size * size * 3)
        }
        val underBrute = mean(brute, 24, 24, 16)
        val underPhotons = mean(photons, 24, 24, 16)
        val openBrute = mean(brute, 1, 1, 6)
        val openPhotons = mean(photons, 1, 1, 6)
        val what = if (smoke) "smoke" else "floor"
        println("[caustics test] $backend $what: under the slab photons $underPhotons, brute force $underBrute; open $openPhotons, $openBrute")
        assertTrue(underBrute > 0.05, "the $what under the slab is dark even by brute force: $underBrute")
        assertTrue(
            abs(underPhotons / underBrute - 1.0) < 0.06,
            "under the slab, photons give $underPhotons where brute force gives $underBrute ($backend, $what)"
        )
        assertTrue(
            abs(openPhotons / openBrute - 1.0) < 0.05,
            "in the open, photons give $openPhotons where brute force gives $openBrute ($backend, $what)"
        )
    }
}
