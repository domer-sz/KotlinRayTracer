package rayTraceTypescript.gpu

import rayTraceTypescript.Vector
import rayTraceTypescript.objects.BvhNode
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.RotateY
import rayTraceTypescript.objects.Translate
import rayTraceTypescript.objects.Triangle

/**
 * The vertical axis a figure stands on, which a turntable turns it about: the middle of the
 * footprint of its lowest [FOOT_FRACTION] of height - its base, its feet. The middle of its whole
 * box would be pulled aside by a raised sword, an outstretched arm or a tail, and the base would
 * then wander round in a circle as the figure turns. StlViewer finds the axis the same way, so the
 * two renderers turn a figure alike.
 *
 * Meshes and triangles count with their vertices; anything else - a sphere, a quad - with the
 * corners of its box.
 */
object Footprint {
    const val FOOT_FRACTION = 0.05f

    /** The axis of [parts] taken together, as (x, z). */
    fun axis(parts: List<Hittable>): Pair<Float, Float> {
        var minY = Float.POSITIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        parts.forEach { points(it, Transform.IDENTITY) { _, y, _ -> if (y < minY) minY = y; if (y > maxY) maxY = y } }
        if (minY > maxY) return 0f to 0f
        val limit = minY + (maxY - minY) * FOOT_FRACTION
        var x0 = Float.POSITIVE_INFINITY; var x1 = Float.NEGATIVE_INFINITY
        var z0 = Float.POSITIVE_INFINITY; var z1 = Float.NEGATIVE_INFINITY
        parts.forEach {
            points(it, Transform.IDENTITY) { x, y, z ->
                if (y <= limit) {
                    if (x < x0) x0 = x; if (x > x1) x1 = x
                    if (z < z0) z0 = z; if (z > z1) z1 = z
                }
            }
        }
        return (x0 + x1) / 2f to (z0 + z1) / 2f
    }

    /** Hands every point of [h], as [transform] places it, to [visit]. */
    private fun points(h: Hittable, transform: Transform, visit: (Float, Float, Float) -> Unit) {
        fun at(x: Float, y: Float, z: Float) {
            if (transform === Transform.IDENTITY) visit(x, y, z)
            else transform.point(Vector(x, y, z)).let { visit(it.x, it.y, it.z) }
        }
        when (h) {
            is PlacedMesh -> {
                // Where the placement puts each vertex: scale * turn(p) + offset.
                val v = h.block.mesh.vertices
                val c = h.cos; val s = h.sin; val k = h.scale; val o = h.offset
                var i = 0
                while (i < v.size) {
                    at((c * v[i] + s * v[i + 2]) * k + o.x, v[i + 1] * k + o.y, (-s * v[i] + c * v[i + 2]) * k + o.z)
                    i += 3
                }
            }
            is Triangle -> listOf(h.v0, h.v1, h.v2).forEach { at(it.x, it.y, it.z) }
            is BvhNode -> {
                points(h.left, transform, visit)
                // A one-object node holds the same child twice.
                if (h.left !== h.right) points(h.right, transform, visit)
            }
            is HittableList -> h.objects.forEach { points(it, transform, visit) }
            is Translate -> points(h.obj, transform.after(Transform.translation(h.offset)), visit)
            is RotateY -> points(h.obj, transform.after(Transform.rotation(h.cosTheta, h.sinTheta)), visit)
            else -> {
                val box = h.aabbBoundingBox()
                for (x in floatArrayOf(box.x.min, box.x.max)) for (y in floatArrayOf(box.y.min, box.y.max)) for (z in floatArrayOf(box.z.min, box.z.max)) at(x, y, z)
            }
        }
    }
}
