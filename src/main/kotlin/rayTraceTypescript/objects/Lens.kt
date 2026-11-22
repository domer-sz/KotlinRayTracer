package rayTraceTypescript.objects

import rayTraceTypescript.Aabb
import rayTraceTypescript.Interval
import rayTraceTypescript.Point
import rayTraceTypescript.Ray
import rayTraceTypescript.Vector
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.Material
import rayTraceTypescript.materials.ScatterRecord
import rayTraceTypescript.optics.GlassCatalog
import rayTraceTypescript.utils.RandomSource
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * A real lens: a solid of revolution about [axis], bounded by spherical (or flat) surfaces one
 * after another along it and a cylinder of radius [semiDiameter] round their rims - a singlet has
 * two surfaces, a cemented doublet three, its middle one the boundary between two glasses. Traced
 * exactly, not as triangles: the kernel tests the same surfaces ([lens_t] in raytracer.cu).
 *
 * Surface k has its vertex [vertices]`[k]` along the axis from [vertex] and the curvature
 * [curvatures]`[k]` = 1/R, R positive when the centre of curvature lies further along the axis than
 * the vertex (the sign convention of Thorlabs, Zemax and DCC-Lab/RayTracing), 0 for a flat face.
 * In those terms the surface is c(rho^2 + s^2) - 2s = 0 with s the distance along the axis from its
 * vertex: no centre far away to subtract from, so a nearly flat face stays exact in floats.
 *
 * A surface may be aspheric, as Zemax's EVENASPH writes one: a conic of constant [conics]`[k]` (0 a
 * sphere, -1 a paraboloid) plus a polynomial in rho^2, [aspherics]`[8k + i - 1]` times rho^(2i) for i
 * 1 to 8 - sag(rho) = c rho^2 / (1 + sqrt(1 - (1 + k) c^2 rho^2)) + sum of a_i rho^(2i). The conic
 * alone is a quadric, c(rho^2 + (1 + k) s^2) - 2s = 0, solved like the sphere; the polynomial is then
 * found by Newton's method from there (s - sag(rho) along the ray), a few steps to a float's precision.
 *
 * [media] are what lies between the surfaces, one more than there are surfaces: air (null) before
 * the first, then each element's glass, and air after the last. A hit's u is which surface it is
 * ([code]: the surfaces 0 until n, then n + j for element j's rim), and its normal points to the
 * medium before it along the axis (out of the glass for the rim), so the books' frontFace says
 * the ray goes from medium k to medium k + 1.
 */
class Lens(
    val vertex: Point,
    axis: Vector,
    val semiDiameter: Float,
    val vertices: FloatArray,
    val curvatures: FloatArray,
    val media: List<Dielectric?>,
    val conics: FloatArray = FloatArray(vertices.size),
    val aspherics: FloatArray = FloatArray(ASPHERIC_TERMS * vertices.size)
) : Hittable {
    val axis: Vector = axis.unit()
    val surfaceCount: Int get() = vertices.size

    /** Where each surface's rim lies along the axis. */
    val rims: FloatArray = FloatArray(vertices.size) { vertices[it] + surfaceSag(it, semiDiameter * semiDiameter) }

    /** Whether surface [k] has a polynomial on top of its conic. */
    fun polynomial(k: Int): Boolean = (0 until ASPHERIC_TERMS).any { aspherics[ASPHERIC_TERMS * k + it] != 0f }

    /** Surface [k]'s sag at rho^2 = [u], from its vertex. */
    fun surfaceSag(k: Int, u: Float): Float {
        val c = curvatures[k]
        var z = c * u / (1f + sqrt(maxOf(0f, 1f - (1f + conics[k]) * c * c * u)))
        var power = u
        for (i in 0 until ASPHERIC_TERMS) {
            z += aspherics[ASPHERIC_TERMS * k + i] * power
            power *= u
        }
        return z
    }

    /** d sag / d(rho^2) of surface [k] at rho^2 = [u]. */
    private fun sagSlope(k: Int, u: Float): Float {
        val c = curvatures[k]
        var slope = c / (2f * sqrt(maxOf(1e-12f, 1f - (1f + conics[k]) * c * c * u)))
        var power = 1f
        for (i in 0 until ASPHERIC_TERMS) {
            slope += (i + 1) * aspherics[ASPHERIC_TERMS * k + i] * power
            power *= u
        }
        return slope
    }

    init {
        require(vertices.size in 2..MAX_SURFACES) { "a lens has 2 to $MAX_SURFACES surfaces, not ${vertices.size}" }
        require(curvatures.size == vertices.size && media.size == vertices.size + 1) { "one curvature per surface and one medium more" }
        require(conics.size == vertices.size && aspherics.size == ASPHERIC_TERMS * vertices.size) { "one conic and $ASPHERIC_TERMS aspheric terms per surface" }
        require(media.first() == null && media.last() == null) { "a lens stands in air" }
        require((1 until media.size - 1).all { media[it] != null }) { "every element of a lens needs its glass" }
        require(semiDiameter > 0f) { "a lens needs a diameter" }
        for (k in vertices.indices) {
            val c = curvatures[k]
            require((1f + conics[k]) * c * c * semiDiameter * semiDiameter < 1f) {
                "surface $k: a radius of ${1f / c} (conic ${conics[k]}) cannot reach a rim ${semiDiameter} from the axis"
            }
        }
        for (k in 0 until vertices.size - 1) {
            val centre = vertices[k + 1] - vertices[k]
            val edge = rims[k + 1] - rims[k]
            require(min(centre, edge) >= MIN_THICKNESS) {
                "element $k is %.4g thick in the middle and %.4g at the rim; rays start %.0e away from a surface, ".format(centre, edge, RAY_EPSILON) +
                    "so a lens must be at least $MIN_THICKNESS units thick everywhere - lay the scene out in millimetres"
            }
        }
    }

    /** The same lens with its first vertex at [vertex], pointing along [axis]. */
    fun placed(vertex: Point, axis: Vector) = Lens(vertex, axis, semiDiameter, vertices, curvatures, media, conics, aspherics)

    /** A code's two sides: the medium the normal points into, then the one behind it. */
    fun sides(code: Int): Pair<Dielectric?, Dielectric?> =
        if (code < surfaceCount) media[code] to media[code + 1] else null to media[code - surfaceCount + 1]

    private val box: Aabb = run {
        // The box of the cylinder through both outermost rims and vertices: its two end discs.
        val start = minOf(vertices.first(), rims.first())
        val end = maxOf(vertices.last(), rims.last())
        val a = this.axis
        val ex = semiDiameter * sqrt(maxOf(0f, 1f - a.x * a.x))
        val ey = semiDiameter * sqrt(maxOf(0f, 1f - a.y * a.y))
        val ez = semiDiameter * sqrt(maxOf(0f, 1f - a.z * a.z))
        val p = vertex + a * start
        val q = vertex + a * end
        Aabb.fromPoints(
            Point(minOf(p.x, q.x) - ex, minOf(p.y, q.y) - ey, minOf(p.z, q.z) - ez),
            Point(maxOf(p.x, q.x) + ex, maxOf(p.y, q.y) + ey, maxOf(p.z, q.z) + ez)
        ).padded()
    }

    override fun aabbBoundingBox(): Aabb = box

    override fun hit(ray: Ray, rayT: Interval): Hit? {
        val found = nearest(ray.origin, ray.direction, rayT.min, rayT.max) ?: return null
        val (t, code) = found
        val point = ray.at(t)
        val outward = normal(point, code)
        return Hit.facing(ray, point, outward, surfaceMaterials[code], t, code.toFloat(), 0f)
    }

    /** The middle of the lens and the radius of a ball round all of it. */
    private val centre: Float = (minOf(vertices.first(), rims.first()) + maxOf(vertices.last(), rims.last())) / 2f
    private val reach: Float = run {
        val half = (maxOf(vertices.last(), rims.last()) - minOf(vertices.first(), rims.first())) / 2f
        sqrt(half * half + semiDiameter * semiDiameter) * 1.01f + MIN_THICKNESS
    }

    /** The nearest crossing of [origin] + t [direction] in (tMin, tMax) and which surface it is. */
    fun nearest(origin: Vector, direction: Vector, tMin: Float, tMax: Float): Pair<Float, Int>? {
        // A ray from far off - the sun, metres away - is first brought up to the ball round the
        // lens: solved from there, the quadratics lose nothing to |q|^2 of millions (from 4 m off
        // a hit landed microns short of the first surface, which the ray then crossed twice).
        val dd0 = Vector.dotProduct(direction, direction)
        val shift = maxOf(0f, Vector.dotProduct(vertex + axis * centre - origin, direction) / dd0 - reach / sqrt(dd0))
        if (shift > 0f) {
            val found = nearestFrom(origin + direction * shift, direction, tMin - shift, tMax - shift) ?: return null
            return found.first + shift to found.second
        }
        return nearestFrom(origin, direction, tMin, tMax)
    }

    private fun nearestFrom(origin: Vector, direction: Vector, tMin: Float, tMax: Float): Pair<Float, Int>? {
        val a = axis
        val o = origin - vertex
        var best = tMax
        var bestCode = -1
        val dd = Vector.dotProduct(direction, direction)
        val ds = Vector.dotProduct(direction, a)
        val h2 = semiDiameter * semiDiameter
        for (k in 0 until surfaceCount) {
            val c = curvatures[k]
            val conic = conics[k]
            val q = o - a * vertices[k]
            val qs = Vector.dotProduct(q, a)
            // The conic, c(|q + t d|^2 + k (qs + t ds)^2) - 2(qs + t ds) = 0, as A t^2 + 2B t + C = 0.
            val qa = c * (dd + conic * ds * ds)
            val qb = c * (Vector.dotProduct(q, direction) + conic * qs * ds) - ds
            val qc = c * (Vector.dotProduct(q, q) + conic * qs * qs) - 2f * qs
            val conicRoots = roots(qa, qb, qc)
            if (!polynomial(k)) {
                for (t in conicRoots) {
                    if (!(t > tMin && t < best)) continue
                    val s = qs + t * ds
                    val rho2 = (q + direction * t).lengthSquared() - s * s
                    // The sheet through the vertex (on it 1 - c(1 + k)s is the root in the sag), inside the rim.
                    if (1f - c * (1f + conic) * s > 0f && rho2 <= h2) {
                        best = t; bestCode = k
                    }
                }
                continue
            }
            // With a polynomial on top: Newton on f(t) = s(t) - sag(rho(t)^2), from where the ray
            // meets the conic (each sheet) and the vertex's plane.
            val qr = q - a * qs
            val dr = direction - a * ds
            val starts = conicRoots.toMutableList()
            if (ds != 0f) starts += -qs / ds
            for (start in starts) {
                var t = start
                var converged = false
                for (step in 0 until NEWTON_STEPS) {
                    val pr = qr + dr * t
                    val u = pr.lengthSquared()
                    val f = qs + t * ds - surfaceSag(k, u)
                    val df = ds - sagSlope(k, u) * 2f * Vector.dotProduct(pr, dr)
                    if (df == 0f) break
                    val next = t - f / df
                    if (abs(next - t) <= 1e-6f * (1f + abs(t))) { t = next; converged = true; break }
                    t = next
                }
                if (!converged || !(t > tMin && t < best)) continue
                val u = (qr + dr * t).lengthSquared()
                if (u <= h2 && (1f + conic) * c * c * u < 1f && abs(qs + t * ds - surfaceSag(k, u)) < 1e-4f * (1f + semiDiameter)) {
                    best = t; bestCode = k
                }
            }
        }
        // The rim: rho = semiDiameter, between the rims of the surfaces either side of an element.
        val os = Vector.dotProduct(o, a)
        val orad = o - a * os
        val drad = direction - a * ds
        val ca = drad.lengthSquared()
        if (ca > 0f) {
            val cb = Vector.dotProduct(orad, drad)
            val cc = orad.lengthSquared() - semiDiameter * semiDiameter
            for (t in roots(ca, cb, cc)) {
                if (!(t > tMin && t < best)) continue
                val s = os + t * ds
                for (j in 0 until surfaceCount - 1) {
                    if (s >= rims[j] && s <= rims[j + 1]) {
                        best = t; bestCode = surfaceCount + j
                        break
                    }
                }
            }
        }
        return if (bestCode < 0) null else best to bestCode
    }

    /** The unit normal at [point] on surface [code], pointing to the medium before it (out of the glass on a rim). */
    fun normal(point: Vector, code: Int): Vector {
        val o = point - vertex
        if (code >= surfaceCount) {
            val radial = o - axis * Vector.dotProduct(o, axis)
            return radial.unit()
        }
        // The gradient of sag(rho^2) - s: 2 sag'(rho^2) r - a, r the radial part.
        val q = o - axis * vertices[code]
        val radial = q - axis * Vector.dotProduct(q, axis)
        return (radial * (2f * sagSlope(code, radial.lengthSquared())) - axis).unit()
    }

    /**
     * What each surface does to a ray on the CPU (picking, tests): bends it from one medium into the
     * other at the d line, reflecting by Fresnel as glass does. The GPU works from [media] itself.
     */
    private val surfaceMaterials: List<Material> = List(surfaceCount * 2 - 1) { code ->
        val (front, back) = sides(code)
        Interface(front?.ri ?: 1f, back?.ri ?: 1f)
    }

    /** A boundary between two indices, [front] on the side the normal points to. */
    class Interface(val front: Float, val back: Float) : Material {
        override fun scatter(rayIn: Ray, hit: Hit): ScatterRecord {
            val ratio = if (hit.frontFace) front / back else back / front
            val unit = rayIn.direction.unit()
            val cosTheta = min(Vector.dotProduct(-unit, hit.normal), 1f)
            val sinTheta = sqrt(1f - cosTheta * cosTheta)
            var r0 = (1f - ratio) / (1f + ratio)
            r0 *= r0
            val reflect = ratio * sinTheta > 1f || r0 + (1f - r0) * (1f - cosTheta).pow(5) > RandomSource.nextFloat()
            val direction = if (reflect) Vector.reflect(unit, hit.normal) else Vector.refract(unit, hit.normal, ratio)
            return ScatterRecord(rayTraceTypescript.Color(1f, 1f, 1f), skipPdfRay = Ray(hit.point, direction, rayIn.time))
        }
    }

    companion object {
        const val MAX_SURFACES = 3
        /** Zemax's EVENASPH terms: rho^2 to rho^16. */
        const val ASPHERIC_TERMS = 8
        /** Newton's steps for an aspheric surface; it converges in three or four. */
        const val NEWTON_STEPS = 12
        /** How far from a surface the renderer starts its rays (world_hit's tMin). */
        const val RAY_EPSILON = 0.001f
        /** The thinnest a lens may be anywhere: 20 times that distance. */
        const val MIN_THICKNESS = 20f * RAY_EPSILON

        /** How far along the axis a surface of curvature [c] has come at [h] from it. */
        fun sag(c: Float, h: Float): Float = c * h * h / (1f + sqrt(1f - c * c * h * h))

        /** The real roots of a t^2 + 2b t + c = 0, worked out without cancellation; a = 0 leaves the linear one. */
        internal fun roots(a: Float, b: Float, c: Float): FloatArray {
            if (abs(a) < 1e-12f) return if (b == 0f) FloatArray(0) else floatArrayOf(-c / (2f * b))
            val disc = b * b - a * c
            if (disc < 0f) return FloatArray(0)
            val q = -(b + if (b >= 0f) sqrt(disc) else -sqrt(disc))
            return if (q == 0f) floatArrayOf(0f) else floatArrayOf(q / a, c / q)
        }

        /** A singlet of radii [r1], [r2] (infinite for flat), [thickness] in the middle, [diameter] across. */
        fun singlet(vertex: Point, axis: Vector, r1: Float, r2: Float, thickness: Float, diameter: Float, glass: Dielectric) =
            Lens(vertex, axis, diameter / 2f, floatArrayOf(0f, thickness), floatArrayOf(curvature(r1), curvature(r2)), listOf(null, glass, null))

        /** A cemented doublet: [r1], [r2], [r3], the two elements [t1] and [t2] thick, [glass1] in front. */
        fun doublet(
            vertex: Point, axis: Vector, r1: Float, r2: Float, r3: Float, t1: Float, t2: Float, diameter: Float,
            glass1: Dielectric, glass2: Dielectric
        ) = Lens(
            vertex, axis, diameter / 2f, floatArrayOf(0f, t1, t1 + t2),
            floatArrayOf(curvature(r1), curvature(r2), curvature(r3)), listOf(null, glass1, glass2, null)
        )

        fun curvature(radius: Float): Float = if (radius.isInfinite()) 0f else 1f / radius

        /** Catalog glass by name, dispersive. */
        fun glass(name: String) = Dielectric(GlassCatalog.glass(name))
    }
}
