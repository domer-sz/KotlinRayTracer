package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.objects.Lens
import rayTraceTypescript.optics.GlassCatalog
import rayTraceTypescript.optics.LensCatalog
import rayTraceTypescript.optics.Spectrum
import kotlin.math.abs

class LensTest {

    private val axis = Vector(0f, 0f, 1f)
    private val origin = Point(0f, 0f, 0f)

    @Test
    fun glassesMatchTheirDataSheets() {
        // Schott's data sheets: n_d and Abbe's number.
        val bk7 = GlassCatalog.glass("N-BK7")
        assertEquals(1.5168f, bk7.n(GlassCatalog.D_LINE), 1e-4f)
        assertEquals(64.17f, bk7.abbe(), 0.2f)
        assertEquals(1.6727f, GlassCatalog.glass("SF5").n(GlassCatalog.D_LINE), 2e-4f)
        assertEquals(1.4585f, GlassCatalog.glass("FusedSilica").n(GlassCatalog.D_LINE), 2e-4f)
        // Schott's polynomial rather than Sellmeier: CDGM's E-BAF11, n_d 1.6667.
        assertEquals(1.6667f, GlassCatalog.glass("E-BAF11").n(GlassCatalog.D_LINE), 5e-4f)
        // Every glass: n falls with wavelength, and the catalog's Abbe number is the computed one.
        for (g in GlassCatalog.all.values) {
            assertTrue(g.n(0.45f) > g.n(0.55f) && g.n(0.55f) > g.n(0.7f), "${g.name} disperses the wrong way")
            g.catalogAbbe?.let { assertEquals(it, g.abbe(), it * 0.02f, g.name) }
        }
    }

    @Test
    fun whiteLightSplitIntoWavelengthsAveragesToWhite() {
        // A path draws its wavelength evenly over the band; the weights it then carries average to
        // (1, 1, 1), and each colour sits where it should: blue at 450 nm, green at 530, red at 620.
        val n = 20000
        var r = 0.0; var g = 0.0; var b = 0.0
        for (i in 0 until n) {
            val w = Spectrum.weight(Spectrum.FROM + (Spectrum.TO - Spectrum.FROM) * (i + 0.5f) / n)
            r += w.r; g += w.g; b += w.b
        }
        assertEquals(1.0, r / n, 1e-3); assertEquals(1.0, g / n, 1e-3); assertEquals(1.0, b / n, 1e-3)
        Spectrum.weight(0.45f).let { assertTrue(it.b > it.g && it.b > it.r) }
        Spectrum.weight(0.53f).let { assertTrue(it.g > it.r && it.g > it.b) }
        Spectrum.weight(0.62f).let { assertTrue(it.r > it.g && it.r > it.b) }
    }

    @Test
    fun aRayAlongTheAxisCrossesEverySurfaceAtItsVertex() {
        val lens = LensCatalog.lens("AC254-100-A").build(origin, axis)
        val first = lens.hit(Ray(Point(0f, 0f, -10f), axis), Interval(0.001f, Float.POSITIVE_INFINITY))
        assertNotNull(first)
        assertEquals(10f, first!!.t, 1e-4f)
        assertEquals(0f, first.u)
        assertTrue(first.frontFace)
        assertEquals(-1f, first.normal.z, 1e-6f)
        val second = lens.hit(Ray(first.point, axis), Interval(0.001f, Float.POSITIVE_INFINITY))!!
        assertEquals(4.0f, second.point.z, 1e-4f)
        assertEquals(1f, second.u)
        val third = lens.hit(Ray(second.point, axis), Interval(0.001f, Float.POSITIVE_INFINITY))!!
        assertEquals(6.5f, third.point.z, 1e-4f)
        assertEquals(2f, third.u)
        assertNull(lens.hit(Ray(third.point, axis), Interval(0.001f, Float.POSITIVE_INFINITY)))
    }

    @Test
    fun theRimAndTheCurvedFacesMeetWhereTheCatalogSays() {
        val spec = LensCatalog.lens("LA1608-A") // plano-convex, R1 = 38.6, flat back, 4.1 thick, 2.0 at the edge
        val lens = spec.build(origin, axis)
        assertEquals(spec.edgeThickness, lens.rims[1] - lens.rims[0], 0.05f)
        // Sideways through the rim, between the two rims: the cylinder, its normal pointing out.
        val z = (lens.rims[0] + lens.rims[1]) / 2f
        val side = lens.hit(Ray(Point(-50f, 0f, z), Vector(1f, 0f, 0f)), Interval(0.001f, Float.POSITIVE_INFINITY))!!
        assertEquals(2f, side.u)
        assertEquals(50f - 12.7f, side.t, 1e-3f)
        assertTrue(side.frontFace)
        // Just outside the rim, parallel to the axis: a miss.
        assertNull(lens.hit(Ray(Point(12.8f, 0f, -10f), axis), Interval(0.001f, Float.POSITIVE_INFINITY)))
        // Just inside: the curved face, then the flat one.
        val front = lens.hit(Ray(Point(12.6f, 0f, -10f), axis), Interval(0.001f, Float.POSITIVE_INFINITY))!!
        assertEquals(0f, front.u)
        assertEquals(Lens.sag(1f / 38.6f, 12.6f), front.point.z, 1e-3f)
    }

    @Test
    fun aLensTooThinForTheRayEpsilonIsRefused() {
        assertThrows<IllegalArgumentException> { LensCatalog.lens("AC254-100-A").build(origin, axis, unitsPerMm = 0.001f) }
    }

    @Test
    fun aLensMayPointAnyWay() {
        val tilted = Vector(1f, 1f, 0f).unit()
        val lens = LensCatalog.lens("AC254-100-A").build(Point(5f, 5f, 5f), tilted)
        val hit = lens.hit(Ray((Point(5f, 5f, 5f) - tilted * 10f).toPoint(), tilted), Interval(0.001f, Float.POSITIVE_INFINITY))!!
        assertEquals(10f, hit.t, 1e-3f)
        val box = lens.aabbBoundingBox()
        assertTrue(box.z.min <= 5f - 12.7f + 0.01f && box.z.max >= 5f + 12.7f - 0.01f)
    }

    @Test
    fun aRayFromMetresAwayCrossesEachSurfaceOnce() {
        // Sunlight 4 m off: solved from there, the quadratic once left the hit microns short of the
        // first surface, which the ray then crossed again - a lens of 62 mm instead of 100.
        val lens = LensCatalog.lens("AC254-100-A").build(origin, axis)
        for (h in listOf(1f, 6f, 12f)) {
            var ray = Ray(Point(h, 0f, -4000f), axis)
            val codes = mutableListOf<Int>()
            while (true) {
                val hit = lens.hit(ray, Interval(0.001f, Float.POSITIVE_INFINITY)) ?: break
                codes += hit.u.toInt()
                val (front, back) = lens.sides(hit.u.toInt())
                val ratio = if (hit.frontFace) (front?.ri ?: 1f) / (back?.ri ?: 1f) else (back?.ri ?: 1f) / (front?.ri ?: 1f)
                ray = Ray(hit.point, Vector.refract(ray.direction.unit(), hit.normal, ratio))
            }
            assertEquals(listOf(0, 1, 2), codes, "surfaces crossed $h mm off the axis")
            val near = backFocalLength(lens, h, GlassCatalog.D_LINE)
            assertEquals(near, ray.origin.z - ray.origin.x / ray.direction.x * ray.direction.z - lens.vertices.last(), 0.01f)
        }
    }

    @Test
    fun everyCatalogLensFocusesWhereItsMakerSays() {
        val failures = mutableListOf<String>()
        for (spec in LensCatalog.all.values) {
            val lens = spec.build(origin, axis)
            val bfl = backFocalLength(lens, spec.diameter * 0.01f, spec.reference)
            val error = abs(bfl - spec.backFocalLength) / abs(spec.backFocalLength)
            if (!(error < 0.01f)) failures += "${spec.name}: %.3f mm, catalog %.3f".format(bfl, spec.backFocalLength)
        }
        assertTrue(failures.isEmpty(), "${failures.size} of ${LensCatalog.all.size} lenses focus elsewhere:\n" + failures.joinToString("\n"))
    }

    @Test
    fun anAchromatBringsRedAndBlueTogetherAndASingletDoesNot() {
        val singlet = LensCatalog.lens("LA1608-A")
        val doublet = LensCatalog.lens("AC254-075-A")
        fun shift(spec: LensCatalog.Spec): Float {
            val lens = spec.build(origin, axis)
            val h = spec.diameter * 0.01f
            return backFocalLength(lens, h, GlassCatalog.F_LINE) - backFocalLength(lens, h, GlassCatalog.C_LINE)
        }
        val singletShift = shift(singlet)
        // A thin singlet's focus moves by f / V between F and C.
        val expected = singlet.focalLength / GlassCatalog.glass("N-BK7").abbe()
        assertEquals(-expected, singletShift, expected * 0.1f)
        // AC254-075-A brings F and C 9.5 times closer than the plano-convex lens of the same focal length
        // (0.12 mm against 1.14): what is left is its paraxial residue - an achromat matches the two
        // colours over its aperture, not exactly on the axis.
        assertTrue(abs(shift(doublet)) < abs(singletShift) / 5f, "achromat ${shift(doublet)} vs singlet $singletShift")
    }

    /**
     * Where a ray [h] off the axis, parallel to it, crosses the axis behind [lens], measured from its
     * last vertex - refracted surface by surface at [microns], never reflected.
     */
    private fun backFocalLength(lens: Lens, h: Float, microns: Float): Float {
        fun n(medium: Dielectric?) = medium?.glass?.n(microns) ?: medium?.ri ?: 1f
        var ray = Ray(Point(h, 0f, -10f), axis)
        while (true) {
            val hit = lens.hit(ray, Interval(0.0001f, Float.POSITIVE_INFINITY)) ?: break
            val (front, back) = lens.sides(hit.u.toInt())
            val ratio = if (hit.frontFace) n(front) / n(back) else n(back) / n(front)
            ray = Ray(hit.point, Vector.refract(ray.direction.unit(), hit.normal, ratio))
        }
        val t = -ray.origin.x / ray.direction.x
        return ray.origin.z + t * ray.direction.z - lens.vertices.last()
    }
}
