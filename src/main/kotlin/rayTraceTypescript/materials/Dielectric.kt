package rayTraceTypescript.materials

import rayTraceTypescript.Color
import rayTraceTypescript.Ray
import rayTraceTypescript.Vector
import rayTraceTypescript.objects.Hit
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import rayTraceTypescript.utils.RandomSource

/**
 * Glass of refractive index [ri]. Clear by default; with a [tint] it is coloured through its
 * body, not on its surface: light that has gone [tintDistance] through it keeps [tint] of itself
 * (Beer-Lambert), so a thick wall is darker than a thin one, as amber glass is. The ray learns
 * how far it went inside when it leaves - a hit on the inside of the surface.
 */
class Dielectric(val ri: Float, val tint: Color? = null, val tintDistance: Float = 1.0f) : Material {
    constructor(ri: Double) : this(ri.toFloat())

    override fun scatter(rayIn: Ray, hit: Hit): ScatterRecord {
        val attenuation = if (tint != null && !hit.frontFace) transmitted(hit.t * rayIn.direction.length()) else Color(1.0f, 1.0f, 1.0f)
        val refractionRatio = if (hit.frontFace) 1.0f / ri else ri
        val unitDirection = rayIn.direction.unit()
        val cosTheta = min(Vector.dotProduct(-unitDirection, hit.normal), 1.0f)
        val sinTheta = sqrt(1.0f - cosTheta * cosTheta)

        val cannotRefract = refractionRatio * sinTheta > 1.0f
        val useReflect = cannotRefract || reflectance(cosTheta, refractionRatio) > RandomSource.nextFloat()
        val direction = if (useReflect)
            Vector.reflect(unitDirection, hit.normal)
        else
            Vector.refract(unitDirection, hit.normal, refractionRatio)

        return ScatterRecord(attenuation, skipPdfRay = Ray(hit.point, direction, rayIn.time))
    }

    /** What is left of the light after [distance] through the glass. */
    fun transmitted(distance: Float): Color {
        val t = tint ?: return Color(1.0f, 1.0f, 1.0f)
        val k = (distance / tintDistance).toDouble()
        return Color(t.r.toDouble().pow(k).toFloat(), t.g.toDouble().pow(k).toFloat(), t.b.toDouble().pow(k).toFloat())
    }

    private fun reflectance(cosine: Float, ri: Float): Float {
        var r0 = (1f - ri) / (1f + ri)
        r0 *= r0
        return r0 + (1f - r0) * (1f - cosine).toDouble().pow(5.0).toFloat()
    }
}
