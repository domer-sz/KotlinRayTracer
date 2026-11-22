package rayTraceTypescript.objects

import rayTraceTypescript.Point
import rayTraceTypescript.Vector
import rayTraceTypescript.materials.Material
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.sin

/**
 * Prisms: a triangle drawn out along the ridge, closed at both ends. Their faces are flat, so the
 * triangles they are made of are the surfaces themselves, not an approximation of them - with a
 * catalog glass ([rayTraceTypescript.materials.Dielectric] of a [rayTraceTypescript.optics.GlassCatalog]
 * glass) the GPU splits light into its colours through them.
 */
object Prism {

    /**
     * A prism whose cross-section, in the plane square to the ridge, is the triangle [a], [b], [c]
     * (given counter-clockwise looking along [ridge]), drawn [depth] along [ridge], centred on the plane
     * of the three points. The faces are wound outwards.
     */
    fun of(a: Point, b: Point, c: Point, ridge: Vector, depth: Float, material: Material): Hittable {
        val half = ridge.unit() * (depth / 2f)
        fun front(p: Point) = (p + half).toPoint()
        fun back(p: Point) = (p - half).toPoint()
        val faces = ArrayList<Hittable>()
        faces += Triangle(front(a), front(b), front(c), material)
        faces += Triangle(back(a), back(c), back(b), material)
        for ((x, y) in listOf(a to b, b to c, c to a)) {
            faces += Triangle(back(x), back(y), front(y), material)
            faces += Triangle(back(x), front(y), front(x), material)
        }
        return HittableList(faces)
    }

    /**
     * An isosceles prism standing on its base in the x-y plane, ridge along z: apex angle [apex]
     * degrees (60 for the equilateral dispersing prism), faces [side] long from the apex, base
     * centred at [base].
     */
    fun standing(base: Point, apex: Float, side: Float, depth: Float, material: Material): Hittable {
        val half = Math.toRadians(apex / 2.0)
        val height = (side * cos(half)).toFloat()
        val reach = (side * sin(half)).toFloat()
        return of(
            Point(base.x, base.y + height, base.z),
            Point(base.x - reach, base.y, base.z),
            Point(base.x + reach, base.y, base.z),
            Vector(0f, 0f, 1f), depth, material
        )
    }

    /** The 60-degree dispersing prism: equilateral, [side] each way. */
    fun equilateral(base: Point, side: Float, depth: Float, material: Material) = standing(base, 60f, side, depth, material)

    /** The 45-90-45 prism that turns light by 90 degrees (or back on itself), legs [leg] long, standing on its hypotenuse. */
    fun rightAngle(base: Point, leg: Float, depth: Float, material: Material) = standing(base, 90f, leg, depth, material)

    /**
     * The angle of incidence, in radians, at which a prism of apex angle [apex] (radians) and index
     * [n] deviates light least - the ray inside then crosses square to the bisector, and the spectrum
     * is sharpest: sin(i) = n sin(apex / 2).
     */
    fun minimumDeviationIncidence(apex: Double, n: Double): Double = asin(n * sin(apex / 2))

    /** The deviation at that incidence: 2 i - apex. */
    fun minimumDeviation(apex: Double, n: Double): Double = 2 * minimumDeviationIncidence(apex, n) - apex

}
