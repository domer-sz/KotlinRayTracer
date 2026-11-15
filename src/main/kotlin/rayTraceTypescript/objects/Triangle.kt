package rayTraceTypescript.objects

import rayTraceTypescript.Aabb
import rayTraceTypescript.Interval
import rayTraceTypescript.Point
import rayTraceTypescript.Ray
import rayTraceTypescript.Vector
import rayTraceTypescript.materials.Material
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * A single mesh facet, intersected with the Moller-Trumbore algorithm. The hit's u/v are the
 * barycentric coordinates, so textures map across the facet, and the normal comes from the
 * winding rather than from the file: STL normals are often wrong, and [Hit] flips it against
 * the ray anyway, which keeps meshes visible from both sides.
 */
class Triangle(
    val v0: Point, val v1: Point, val v2: Point, val material: Material,
    /**
     * Normals at the three corners (9 floats) for smooth shading on the GPU ([Mesh.smoothNormals]),
     * or null for a flat facet. Hits here keep the facet's own normal: they serve picking.
     */
    val cornerNormals: FloatArray? = null
) : Hittable {

    val edge1: Vector = v1 - v0
    val edge2: Vector = v2 - v0

    private val bbox: Aabb = Aabb.fromPoints(
        Point(min(v0.x, min(v1.x, v2.x)), min(v0.y, min(v1.y, v2.y)), min(v0.z, min(v1.z, v2.z))),
        Point(max(v0.x, max(v1.x, v2.x)), max(v0.y, max(v1.y, v2.y)), max(v0.z, max(v1.z, v2.z)))
    ).padded()

    override fun hit(ray: Ray, rayT: Interval): Hit? {
        val pvec = Vector.cross(ray.direction, edge2)
        val determinant = Vector.dotProduct(edge1, pvec)
        if (abs(determinant) < PARALLEL_EPSILON) return null   // ray runs along the facet plane

        val inverseDeterminant = 1.0f / determinant
        val tvec = ray.origin - v0
        val u = Vector.dotProduct(tvec, pvec) * inverseDeterminant
        if (u < 0.0f || u > 1.0f) return null

        val qvec = Vector.cross(tvec, edge1)
        val v = Vector.dotProduct(ray.direction, qvec) * inverseDeterminant
        if (v < 0.0f || u + v > 1.0f) return null

        val t = Vector.dotProduct(edge2, qvec) * inverseDeterminant
        if (!rayT.surrounds(t)) return null

        val outwardNormal = Vector.unit(Vector.cross(edge1, edge2))
        return Hit.facing(ray, ray.at(t), outwardNormal, material, t, u, v)
    }

    override fun aabbBoundingBox(): Aabb = bbox

    companion object {
        const val PARALLEL_EPSILON = 1e-8f
    }
}
