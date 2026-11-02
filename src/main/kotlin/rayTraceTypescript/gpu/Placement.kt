package rayTraceTypescript.gpu

import rayTraceTypescript.Vector
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.RotateY
import rayTraceTypescript.objects.Translate

/** Moving and turning finished objects about the vertical, the one rigid move scenes make. */
object Placement {

    /**
     * [h] turned [degrees] about the vertical line through [from], then moved so that [from]
     * lands on [to]: each point p goes to turn(p - from) + to, turning as [RotateY] does.
     *
     * A [PlacedMesh] comes back as a placed mesh again, with the move folded into its own
     * placement - the GPU then copies its prebuilt tree in rather than building one over every
     * triangle, which for a model of hundreds of thousands of triangles is the difference between
     * a fraction of a second and seconds. Anything else gets instance wrappers.
     */
    fun moved(h: Hittable, from: Vector, degrees: Float, to: Vector): Hittable {
        if (h is PlacedMesh) {
            val radians = Math.toRadians(degrees.toDouble())
            val c = kotlin.math.cos(radians).toFloat()
            val s = kotlin.math.sin(radians).toFloat()
            val dx = h.offset.x - from.x
            val dy = h.offset.y - from.y
            val dz = h.offset.z - from.z
            val offset = Vector(c * dx + s * dz + to.x, dy + to.y, -s * dx + c * dz + to.z)
            return PlacedMesh(h.block, h.scale, h.degrees + degrees, offset, h.material, h.palette)
        }
        val centred = Translate(h, Vector(-from.x, -from.y, -from.z))
        return Translate(RotateY(centred, degrees), to)
    }

    /** [h] turned [degrees] in place about the vertical line through ([x], [z]). */
    fun turned(h: Hittable, x: Float, z: Float, degrees: Float): Hittable {
        val axis = Vector(x, 0f, z)
        return moved(h, axis, degrees, axis)
    }
}
