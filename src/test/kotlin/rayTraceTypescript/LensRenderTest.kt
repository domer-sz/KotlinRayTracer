package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
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
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.optics.GlassCatalog
import rayTraceTypescript.render.CameraSetup
import rayTraceTypescript.optics.LensCatalog
import rayTraceTypescript.scenes.OpticsScenes
import kotlin.math.abs

/**
 * Lenses and dispersion on the card. Sunlight through AC254-100-A comes to a point on a screen at
 * the catalog's back focal length and spreads into a disc nearer the lens - in both engines, which
 * agree on where and how bright. A prism of flint sends blue further aside than red, and does not
 * with -Drt.dispersion=off.
 */
class LensRenderTest {

    @Test
    fun `the cuda kernel focuses sunlight where the catalog says`() = focus("cuda")

    @Test
    fun `optix focuses sunlight where the catalog says`() {
        assumeGpu()
        assumeTrue(OptixApi.unavailabilityReason() == null, "OptiX is not available")
        focus("optix")
    }

    @Test
    fun `the two engines see the same spot`() {
        assumeGpu()
        assumeTrue(OptixApi.unavailabilityReason() == null, "OptiX is not available")
        assumeGpu()
        val spec = LensCatalog.lens("AC254-100-A")
        val cuda = bench(spec.backFocalLength, "cuda")
        val optix = bench(spec.backFocalLength, "optix")
        assertEquals(cuda.total, optix.total, cuda.total * 0.05, "light on the screen")
        assertEquals(cuda.x, optix.x, 0.5, "spot across")
        assertEquals(cuda.y, optix.y, 0.5, "spot up")
    }

    @Test
    fun `a flint prism spreads blue further than red`() {
        assumeGpu()
        // Light through a 60-degree prism of N-SF11 seen on a screen: blue bends more.
        val split = prism(dispersion = true)
        assertTrue(split.blueY - split.redY > 3.0, "blue at ${split.blueY}, red at ${split.redY} (pixels down the screen)")
        val white = prism(dispersion = false)
        assertTrue(abs(white.blueY - white.redY) < 0.5, "without dispersion blue at ${white.blueY}, red at ${white.redY}")
    }

    @Test
    fun `the cuda kernel's asphere brings sunlight to a point`() = asphere("cuda")

    @Test
    fun `optix's asphere brings sunlight to a point`() {
        assumeGpu()
        assumeTrue(OptixApi.unavailabilityReason() == null, "OptiX is not available")
        asphere("optix")
    }

    /**
     * Thorlabs' 354710-C collimator (f = 1.49 mm, NA 0.53) read backwards: sunlight through it (without
     * the diode window it is designed with) onto a screen at its focus for the d line, worked out on the CPU, seen at
     * 2.7 microns a pixel. The sun's image there is 14 microns across (23 long on the turned screen);
     * the same lens made spherical blurs it over a tenth of a millimetre.
     */
    private fun asphere(backend: String) {
        assumeGpu()
        val file = rayTraceTypescript.optics.ZemaxLens.read(java.nio.file.Paths.get(LensRenderTest::class.java.getResource("/optics/354710-C.zmx")!!.toURI()))
        // The asphere alone: seen past the diode window the spot would be seen through its glass.
        val lenses = file.build(Point(0f, 0f, 0f), Vector(1f, 0f, 0f)).take(1)
        val focusX = cpuFocus(lenses)
        fun spot(made: List<rayTraceTypescript.objects.Lens>): Spot {
            val sun = Sphere(Point(-4000f, 0f, 0f), 4000f * 0.00465f, DiffuseLight(Color(4000f, 3900f, 3700f)))
            val world = HittableList(
                // The screen through the focus turned 37 degrees towards the camera (its normal
                // (-0.6, 0, 0.8)), so that the camera, on that normal, looks past the lens and the stop.
                (listOf(sun, OpticsScenes.annulus(Point(-0.3f, 0f, 0f), Vector(1f, 0f, 0f), 0.72f, 1.5f, Lambertian(Color(0f, 0f, 0f))),
                    Quad(Point(focusX - 4f, -5f, -3f), Vector(8f, 0f, 6f), Vector(0f, 10f, 0f), Lambertian(Color(0.8f, 0.8f, 0.8f)))) + made).toMutableList()
            )
            val setup = Camera().apply {
                aspectRatio = 1f
                imageWidth = PIXELS
                maxReflectionDepth = 12
                vfov = 0.25f
                lookFrom = Point(focusX - 35f, 0f, 46.67f)
                lookAt = Point(focusX, 0f, 0f)
                vUp = Vector(0f, 1f, 0f)
                background = Color(0f, 0f, 0f)
                fireflyClamp = 0f
            }.initialize()
            System.setProperty("rt.dispersion", "off")
            try {
                return spotOf(render(world, sun, setup, backend, caustics = true, samples = 256, radius = "0.001"), setup.imageWidth, setup.imageHeight)
            } finally {
                System.clearProperty("rt.dispersion")
            }
        }
        val sharp = spot(lenses)
        val blurred = spot(lenses.map { rayTraceTypescript.objects.Lens(it.vertex, it.axis, it.semiDiameter, it.vertices, it.curvatures, it.media) })
        assertTrue(sharp.core > 0.8, "asphere: ${sharp.core} of the light in view within 21 microns")
        // The spherical lens spreads most of its light past the view (0.24 mm across): compare the
        // light each puts within those 21 microns, not the share of what is in view. Traced on the
        // CPU, the spherical lens brings only rays within 0.37 mm of the axis that close - 26% of the
        // aperture, so 3.8 times less; measured 4.1 in both engines.
        val sharpCore = sharp.core * sharp.total
        val blurredCore = blurred.core * blurred.total
        assertTrue(sharpCore > 3 * blurredCore, "within 21 microns: asphere $sharpCore, spherical $blurredCore")
    }

    /** Where a paraxial ray (5% of the aperture) crosses the axis behind [lenses], refracted on the CPU at the d line. */
    private fun cpuFocus(lenses: List<rayTraceTypescript.objects.Lens>): Float {
        var ray = Ray(Point(-2f, 0.04f, 0f), Vector(1f, 0f, 0f))
        while (true) {
            val hits = lenses.mapNotNull { lens -> lens.hit(ray, Interval(0.0001f, Float.POSITIVE_INFINITY))?.let { lens to it } }
            val (lens, hit) = hits.minByOrNull { it.second.t } ?: break
            val (front, back) = lens.sides(hit.u.toInt())
            val ratio = if (hit.frontFace) (front?.ri ?: 1f) / (back?.ri ?: 1f) else (back?.ri ?: 1f) / (front?.ri ?: 1f)
            ray = Ray(hit.point, Vector.refract(ray.direction.unit(), hit.normal, ratio))
        }
        return ray.origin.x - ray.origin.y / ray.direction.y * ray.direction.x
    }

    private fun focus(backend: String) {
        assumeGpu()
        val spec = LensCatalog.lens("AC254-100-A")
        val sharp = bench(spec.backFocalLength, backend)
        val blurred = bench(spec.backFocalLength * 0.8f, backend)
        // The spot sits on the axis: the middle of the picture.
        assertEquals(PIXELS / 2.0, sharp.x, 1.0, "spot across")
        assertEquals(PIXELS / 2.0, sharp.y, 1.0, "spot up")
        // At focus the sun's image is 0.9 mm across (its 0.27 degrees times f): about 9 pixels, nearly
        // all of the light within 8 of the middle; 20 mm nearer the lens the beam is still 4.9 mm wide,
        // a tenth of it there. (The rest of the screen catches a faint glow, photons the lens's faces
        // and rim reflected - the ghosts of an uncoated lens.)
        assertTrue(sharp.core > 0.8, "focused spot: ${sharp.core} of the light within 8 px")
        assertTrue(blurred.core < 0.3, "defocused: ${blurred.core} of the light within 8 px")
        // Both catch the same light: what the stop lets through, less what the glass reflects.
        assertEquals(sharp.total, blurred.total, sharp.total * 0.1)
    }

    /** The light in the picture: its sum, its centroid in pixels, and the share of it within 8 pixels of the middle. */
    private class Spot(val total: Double, val x: Double, val y: Double, val core: Double)

    /** The screen [behind] mm after the lens, seen from the side and close up (0.1 mm a pixel), lit through it by photons. */
    private fun bench(behind: Float, backend: String): Spot {
        val spec = LensCatalog.lens("AC254-100-A")
        val axis = Vector(1f, 0f, 0f)
        val screenX = spec.length + behind
        val sun = Sphere(Point(-4000f, 0f, 0f), 4000f * 0.00465f, DiffuseLight(Color(4000f, 3900f, 3700f)))
        val world = HittableList(
            mutableListOf(
                sun,
                spec.build(Point(0f, 0f, 0f), axis),
                OpticsScenes.annulus(Point(-2f, 0f, 0f), axis, spec.diameter / 2f - 0.5f, 80f, Lambertian(Color(0f, 0f, 0f))),
                Quad(Point(screenX, -40f, 40f), Vector(0f, 0f, -80f), Vector(0f, 80f, 0f), Lambertian(Color(0.8f, 0.8f, 0.8f)))
            )
        )
        val setup = Camera().apply {
            aspectRatio = 1f
            imageWidth = PIXELS
            maxReflectionDepth = 12
            vfov = 9f
            lookFrom = Point(screenX - 50f, 0f, 30f)
            lookAt = Point(screenX, 0f, 0f)
            vUp = Vector(0f, 1f, 0f)
            background = Color(0f, 0f, 0f)
            fireflyClamp = 0f
        }.initialize()
        val image = render(world, sun, setup, backend, caustics = true, samples = 256)
        return spotOf(image, setup.imageWidth, setup.imageHeight)
    }

    private class Split(val blueY: Double, val redY: Double)

    /** A narrow beam through a flint prism onto a screen, seen square on: where blue and red land. */
    private fun prism(dispersion: Boolean): Split {
        val glass = Dielectric(GlassCatalog.glass("N-SF11"))
        // A prism with its apex up and a beam rising at 33.2 degrees onto its left face - minimum
        // deviation for n = 1.785 - through a slit; it leaves falling at about as much, blue the most.
        val rise = Math.toRadians(33.2)
        val toSun = Vector(-kotlin.math.cos(rise).toFloat(), -kotlin.math.sin(rise).toFloat(), 0f)
        val sun = Sphere((Point(-5f, 0f, 0f) + toSun * 4000f).toPoint(), 4000f * 0.00465f, DiffuseLight(Color(4000f, 4000f, 4000f)))
        val slitY = (-25.0 * kotlin.math.tan(rise)).toFloat()
        val black = Lambertian(Color(0f, 0f, 0f))
        val half = Math.toRadians(30.0)
        val s = 20f
        val a = Point(0f, (s * kotlin.math.cos(half)).toFloat() / 2f, 0f)
        val b = Point((-s * kotlin.math.sin(half)).toFloat(), -(s * kotlin.math.cos(half)).toFloat() / 2f, 0f)
        val c = Point((s * kotlin.math.sin(half)).toFloat(), -(s * kotlin.math.cos(half)).toFloat() / 2f, 0f)
        val faces = ArrayList<rayTraceTypescript.objects.Hittable>()
        fun p(x: Point, z: Float) = Point(x.x, x.y, z)
        faces += rayTraceTypescript.objects.Triangle(p(a, 20f), p(b, 20f), p(c, 20f), glass)
        faces += rayTraceTypescript.objects.Triangle(p(a, -20f), p(c, -20f), p(b, -20f), glass)
        for ((x, y) in listOf(a to b, b to c, c to a)) {
            faces += rayTraceTypescript.objects.Triangle(p(x, -20f), p(y, -20f), p(y, 20f), glass)
            faces += rayTraceTypescript.objects.Triangle(p(x, -20f), p(y, 20f), p(x, 20f), glass)
        }
        val world = HittableList(
            mutableListOf(
                sun,
                HittableList(faces),
                // A slit 1 mm high in a tall black wall before the prism, aimed at its left face.
                Quad(Point(-30f, slitY + 0.5f, -50f), Vector(0f, 0f, 100f), Vector(0f, 300f, 0f), black),
                Quad(Point(-30f, slitY - 300.5f, -50f), Vector(0f, 0f, 100f), Vector(0f, 300f, 0f), black),
                Quad(Point(80f, -80f, 40f), Vector(0f, 0f, -80f), Vector(0f, 100f, 0f), Lambertian(Color(0.8f, 0.8f, 0.8f)))
            )
        )
        val setup = Camera().apply {
            aspectRatio = 1f
            imageWidth = PIXELS
            maxReflectionDepth = 12
            vfov = 70f
            lookFrom = Point(20f, -45f, 0f)
            lookAt = Point(80f, -45f, 0f)
            vUp = Vector(0f, 1f, 0f)
            background = Color(0f, 0f, 0f)
            fireflyClamp = 0f
        }.initialize()
        if (!dispersion) System.setProperty("rt.dispersion", "off")
        val image = try {
            render(world, sun, setup, "cuda", caustics = true, samples = 256)
        } finally {
            System.clearProperty("rt.dispersion")
        }
        // Where the blue and the red channels' light sits down the screen.
        fun meanY(channel: Int): Double {
            var sum = 0.0; var weight = 0.0
            for (y in 0 until setup.imageHeight) for (x in 0 until setup.imageWidth) {
                val v = image[(y * setup.imageWidth + x) * 3 + channel].toDouble()
                sum += v * y; weight += v
            }
            return sum / weight
        }
        return Split(meanY(2), meanY(0))
    }

    private fun render(world: HittableList, sun: Sphere, setup: CameraSetup, backend: String, caustics: Boolean, samples: Int, radius: String = "0.05"): FloatArray {
        if (caustics) System.setProperty("rt.caustics", "on")
        System.setProperty("rt.caustics.radius", radius)
        try {
            GpuSession.open(SceneFlattener.scene(world, sun), setup.totalPixels, GpuChoice(gpu = "0", backend = backend)).use { session ->
                session.seed = 11L
                session.setCamera(setup)
                session.traceSliced(GpuSession.MODE_IMAGE, setup, samples, GpuSession.IMAGE, 50.0)
                return FloatArray(setup.totalPixels * 3).also { session.read(GpuSession.IMAGE, it) }
            }
        } finally {
            System.clearProperty("rt.caustics")
            System.clearProperty("rt.caustics.radius")
        }
    }

    private fun spotOf(image: FloatArray, w: Int, h: Int): Spot {
        var total = 0.0; var sx = 0.0; var sy = 0.0
        for (y in 0 until h) for (x in 0 until w) {
            val at = (y * w + x) * 3
            val v = (image[at] + image[at + 1] + image[at + 2]).toDouble()
            total += v; sx += v * (x + 0.5); sy += v * (y + 0.5)
        }
        var core = 0.0
        for (y in 0 until h) for (x in 0 until w) {
            val at = (y * w + x) * 3
            val dx = x + 0.5 - w / 2.0; val dy = y + 0.5 - h / 2.0
            if (dx * dx + dy * dy <= 64.0) core += (image[at] + image[at + 1] + image[at + 2]).toDouble()
        }
        return Spot(total, sx / total, sy / total, core / total)
    }

    private fun assumeGpu() {
        val gpu = GpuRenderer.unavailabilityReason()
        assumeTrue(gpu == null, "No CUDA device available: $gpu")
    }

    private companion object {
        const val PIXELS = 96
    }
}
