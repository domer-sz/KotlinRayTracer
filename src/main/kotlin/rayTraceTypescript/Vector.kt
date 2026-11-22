package rayTraceTypescript

import rayTraceTypescript.utils.randomFloat
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.sqrt

open class Vector(val x: Float, val y: Float, val z: Float) {
    constructor(x: Double, y: Double, z: Double) : this(x.toFloat(), y.toFloat(), z.toFloat())
    fun negate(): Vector = Vector(-x, -y, -z)
    operator fun unaryMinus(): Vector = negate()
    operator fun plus(other: Vector): Vector = Vector(x + other.x, y + other.y, z + other.z)
    operator fun minus(other: Vector): Vector = Vector(x - other.x, y - other.y, z - other.z)
    fun scale(s: Float): Vector = Vector(x * s, y * s, z * s)
    operator fun times(s: Float): Vector = scale(s)
    fun divide(s: Float): Vector = Vector(x / s, y / s, z / s)
    operator fun div(s: Float): Vector = divide(s)
    fun multiply(other: Vector): Vector = Vector(x * other.x, y * other.y, z * other.z)
    operator fun times(other: Vector): Vector = multiply(other)

    fun length(): Float = sqrt(lengthSquared())
    fun lengthSquared(): Float = x * x + y * y + z * z
    fun unit(): Vector = divide(length())

    fun nearZero(): Boolean {
        val s = 1e-8f
        return abs(x) < s && abs(y) < s && abs(z) < s
    }

    override fun toString(): String = "$" + "x $" + "y $" + "z"

    operator fun get(i: Int): Float = when (i) {
        0 -> x
        1 -> y
        2 -> z
        else -> throw IndexOutOfBoundsException("Vector index $i")
    }

    fun toPoint(): Point {
        return Point(x, y, z)
    }

    companion object {
        @JvmStatic
        fun dotProduct(u: Vector, v: Vector): Float = u.x * v.x + u.y * v.y + u.z * v.z
        @JvmStatic
        fun cross(u: Vector, v: Vector): Vector =
            Vector(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x)

        @JvmStatic
        fun unit(v: Vector): Vector = v.unit()

        @JvmStatic
        fun random(min: Float = 0.0f, max: Float = 1.0f): Vector =
            Vector(randomFloat(min, max), randomFloat(min, max), randomFloat(min, max))

        @JvmStatic
        fun randomMinMax(min: Float, max: Float): Vector = random(min, max)

        @JvmStatic
        fun randomVectorInUnitDisc(): Vector {
            while (true) {
                val p = Vector(randomFloat(-1.0f, 1.0f), randomFloat(-1.0f, 1.0f), 0.0f)
                if (p.lengthSquared() < 1.0f) return p
            }
        }

        @JvmStatic
        fun randomInUnitSphere(): Vector {
            while (true) {
                val p = random(-1.0f, 1.0f)
                if (p.lengthSquared() < 1.0f) return p
            }
        }

        /**
         * A direction around +z, weighted by cosine: the distribution a diffuse surface
         * actually scatters with, so the renderer spends samples where they matter.
         */
        @JvmStatic
        fun randomCosineDirection(): Vector {
            val r1 = randomFloat(0.0f, 1.0f)
            val r2 = randomFloat(0.0f, 1.0f)
            val phi = 2.0f * rayTraceTypescript.utils.pi * r1
            val root = kotlin.math.sqrt(r2)
            return Vector(
                kotlin.math.cos(phi) * root,
                kotlin.math.sin(phi) * root,
                kotlin.math.sqrt(1.0f - r2)
            )
        }

        /** A direction towards a sphere of [radius] whose centre is [distanceSquared] away. */
        @JvmStatic
        fun randomToSphere(radius: Float, distanceSquared: Float): Vector {
            val r1 = randomFloat(0.0f, 1.0f)
            val r2 = randomFloat(0.0f, 1.0f)
            // 1 - z without cancellation, as the kernel works it out: a cone narrower than a float's
            // precision (a small target metres off) would otherwise collapse to its axis.
            val x = radius * radius / distanceSquared
            val oneMinusZ = r2 * x / (1.0f + kotlin.math.sqrt(maxOf(0.0f, 1.0f - x)))
            val z = 1.0f - oneMinusZ

            val phi = 2.0f * rayTraceTypescript.utils.pi * r1
            val root = kotlin.math.sqrt(maxOf(0.0f, oneMinusZ * (2.0f - oneMinusZ)))
            return Vector(kotlin.math.cos(phi) * root, kotlin.math.sin(phi) * root, z)
        }

        /** A direction drawn evenly from the sphere of directions. */
        @JvmStatic
        fun randomUnitVector(): Vector = randomInUnitSphere().unit()

        @JvmStatic
        fun randomOnHemisphere(normal: Vector): Vector {
            val onUnitSphere = randomInUnitSphere().unit()
            return if (dotProduct(onUnitSphere, normal) > 0.0f) onUnitSphere else -onUnitSphere
        }

        @JvmStatic
        fun reflect(v: Vector, n: Vector): Vector =
            v - n * (2.0f * dotProduct(v, n))

        @JvmStatic
        fun refract(uv: Vector, n: Vector, etai_over_etat: Float): Vector {
            val cosTheta = min(dotProduct(-uv, n), 1.0f)
            val rOutPerp = (n * cosTheta + uv) * etai_over_etat
            val rOutParallel = -(n * sqrt(abs(1.0f - rOutPerp.lengthSquared())))
            return rOutPerp + rOutParallel
        }
    }
}
