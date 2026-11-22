package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.objects.Lens
import rayTraceTypescript.optics.ZemaxLens
import java.nio.file.Paths


/**
 * Aspheric surfaces and Zemax files: Thorlabs' 354710-C, a molded asphere that collimates a laser
 * diode (f = 1.49 mm, NA 0.53, with the diode's window), read from its own .zmx.
 */
class AsphereTest {

    private val axis = Vector(0f, 0f, 1f)
    private val file = ZemaxLens.read(Paths.get(AsphereTest::class.java.getResource("/optics/354710-C.zmx")!!.toURI()))

    @Test
    fun theFileReadsAsAnAsphereAndAWindow() {
        assertEquals(2, file.elementGroups.size)
        val (asphere, window) = file.build(Point(0f, 0f, 0f), axis)
        assertEquals(-0.4776343f, asphere.conics[0], 1e-6f)
        assertEquals(-6.313587842e-3f, asphere.aspherics[1], 1e-9f)
        assertEquals(0.8625269f, asphere.vertices[1], 1e-6f)
        assertEquals(0.25f, window.vertices[1] - window.vertices[0], 1e-6f)
    }

    @Test
    fun aSurfaceIsHitWhereItsSagSays() {
        val (asphere, _) = file.build(Point(0f, 0f, 0f), axis)
        for (h in listOf(0f, 0.2f, 0.5f, 0.7f)) {
            val hit = asphere.hit(Ray(Point(h, 0f, -5f), axis), Interval(0.001f, Float.POSITIVE_INFINITY))!!
            assertEquals(0f, hit.u)
            assertEquals(asphere.surfaceSag(0, h * h), hit.point.z, 2e-6f, "sag at $h mm")
            // Slanted rays too (inside the clear aperture): the hit lies on the surface.
            if (h > 0.5f) continue
            val slanted = asphere.hit(Ray(Point(h - 1f, 0.3f, -2f), Vector(0.5f, -0.15f, 1f)), Interval(0.001f, Float.POSITIVE_INFINITY))
                ?: error("slanted ray at $h missed; rim ${asphere.semiDiameter}, rims ${asphere.rims.toList()}")
            val r2 = slanted.point.x * slanted.point.x + slanted.point.y * slanted.point.y
            assertEquals(asphere.surfaceSag(0, r2), slanted.point.z, 2e-6f, "slanted at $h mm")
        }
    }

    @Test
    fun theAsphereBringsEveryZoneToOneFocusWhereASphereCannot() {
        val lenses = file.build(Point(0f, 0f, 0f), axis)
        val crossings = focusOf(lenses)
        // The same glass and radius with the conic and the polynomial taken off.
        val spherical = lenses.map { it.withoutAsphere() }
        val sphereCrossings = focusOf(spherical)
        val spread = crossings.max() - crossings.min()
        val sphereSpread = sphereCrossings.max() - sphereCrossings.min()
        println("asphere: focus ${crossings.joinToString { "%.4f".format(it) }} (spread %.5f mm); sphere: spread %.4f mm".format(spread, sphereSpread))
        assertTrue(spread < sphereSpread / 20f, "asphere spread $spread mm against the sphere's $sphereSpread")
        // Its focus is 41 microns short of the image surface the file puts at 1.886 mm (where Zemax,
        // with CDGM's own D-ZK3M, brings the light to a spot 9 nm across): D-ZK3M is not in the
        // catalog, so it is the model glass of the n_d and V_d the file gives, at 1550 nm.
        assertEquals(file.imageAt, crossings.first(), 0.05f)
    }

    /** Where rays parallel to the axis at 5% to 95% of the clear aperture cross it behind [lenses], at the file's wavelength. */
    private fun focusOf(lenses: List<Lens>): List<Float> = listOf(0.05f, 0.25f, 0.45f, 0.65f, 0.85f, 0.95f).map { fraction ->
        val h = 0.75f * fraction
        var ray = Ray(Point(h, 0f, -2f), axis)
        while (true) {
            val hits = lenses.mapNotNull { it.hit(ray, Interval(0.0001f, Float.POSITIVE_INFINITY))?.let { hit -> it to hit } }
            val (lens, hit) = hits.minByOrNull { it.second.t } ?: break
            val (front, back) = lens.sides(hit.u.toInt())
            fun n(m: Dielectric?) = m?.glass?.n(file.wavelength) ?: m?.ri ?: 1f
            val ratio = if (hit.frontFace) n(front) / n(back) else n(back) / n(front)
            ray = Ray(hit.point, Vector.refract(ray.direction.unit(), hit.normal, ratio))
        }
        ray.origin.z - ray.origin.x / ray.direction.x * ray.direction.z
    }

    private fun Lens.withoutAsphere() = Lens(vertex, axis, semiDiameter, vertices, curvatures, media)

}
