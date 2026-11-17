package rayTraceTypescript.objects

import rayTraceTypescript.Aabb
import rayTraceTypescript.Point
import rayTraceTypescript.Vector
import rayTraceTypescript.materials.Material

/**
 * A triangle soup, stored as flat vertex data (9 floats per facet: three vertices, each x/y/z).
 * Every supported 3D file format is parsed into this one shape by [MeshLoader].
 *
 * Mesh files carry no shared convention for units, origin or orientation, so a freshly loaded
 * model can sit anywhere and be any size. [standingOnFloor] is the default placement: it scales
 * the model to a workable height and puts it centred on the floor, where the scenes in this
 * project put their ground.
 *
 * A [smooth] mesh is shaded as the surface its facets approximate: the GPU turns the normal
 * across each facet between [smoothNormals] at its corners. A model from a file is smooth
 * ([MeshLoader]); a mesh made in code - a test's octahedron - keeps its flat facets.
 */
class Mesh(
    val vertices: FloatArray,
    /** Each triangle's entry in [palette], when the file gave its faces materials. */
    val faceMaterials: IntArray? = null,
    /** The materials the file names (an OBJ's .mtl), in the order [faceMaterials] counts them. */
    val palette: List<MeshMaterial> = emptyList(),
    /** Whether to shade it smooth, from [smoothNormals]. */
    val smooth: Boolean = false,
    /**
     * Texture coordinates at each facet's three corners (6 floats a triangle: u, v of each), when
     * the file gave them (an OBJ's `vt`); NaN in a triangle's first value marks one without.
     */
    val uvs: FloatArray? = null
) {

    init {
        require(vertices.isNotEmpty()) { "Mesh has no triangles" }
        require(vertices.size % VERTEX_FLOATS == 0) { "Expected $VERTEX_FLOATS floats per triangle" }
        require(faceMaterials == null || faceMaterials.size == vertices.size / VERTEX_FLOATS) { "One material per triangle" }
        require(uvs == null || uvs.size == vertices.size / VERTEX_FLOATS * UV_FLOATS) { "Six texture coordinates per triangle" }
    }

    val triangleCount: Int get() = vertices.size / VERTEX_FLOATS

    /** Whether the file coloured its faces itself. */
    val hasColours: Boolean get() = faceMaterials != null && palette.isNotEmpty()

    /** The same mesh, shaded smooth or flat. */
    fun smoothed(on: Boolean = true): Mesh = if (on == smooth) this else Mesh(vertices, faceMaterials, palette, on, uvs)

    /**
     * A normal at each corner of each facet, 9 floats a triangle like [vertices], for smooth
     * shading - worked out once, from the vertices as they are.
     *
     * A mesh file stores every facet on its own, so the corners have to be welded first. Corners
     * one exporter wrote for one point are bit for bit the same, so they are matched on their
     * float bits - no epsilon to tune - with -0 taken as +0. Each welded point's normal is the
     * sum of its facets' unnormalised normals, so a bigger facet weighs more, made unit length.
     * The same as StlViewer does it, so the two renderers shade a model alike.
     */
    val smoothNormals: FloatArray by lazy { weldedNormals(vertices) }

    fun bounds(): Aabb {
        var minX = Float.POSITIVE_INFINITY; var minY = Float.POSITIVE_INFINITY; var minZ = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY; var maxY = Float.NEGATIVE_INFINITY; var maxZ = Float.NEGATIVE_INFINITY
        var i = 0
        while (i < vertices.size) {
            val x = vertices[i]; val y = vertices[i + 1]; val z = vertices[i + 2]
            if (x < minX) minX = x; if (x > maxX) maxX = x
            if (y < minY) minY = y; if (y > maxY) maxY = y
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
            i += 3
        }
        return Aabb.fromPoints(Point(minX, minY, minZ), Point(maxX, maxY, maxZ))
    }

    /**
     * Centres the model over the origin in x/z, rests its lowest point on [floorY] and - unless
     * [targetHeight] is null - scales it uniformly to that height, so models authored in
     * millimetres and models authored in metres both arrive at a sensible size.
     */
    fun standingOnFloor(targetHeight: Float? = DEFAULT_TARGET_HEIGHT, floorY: Float = 0.0f): Mesh {
        val box = bounds()
        val height = box.y.size()
        val scale = if (targetHeight == null || height <= 0.0f) 1.0f else targetHeight / height
        val pivot = Vector((box.x.min + box.x.max) / 2.0f, box.y.min, (box.z.min + box.z.max) / 2.0f)
        return transformed(scale, pivot, Vector(0.0f, floorY, 0.0f))
    }

    /** Moves every vertex to `(vertex - pivot) * scale + target`. */
    fun transformed(scale: Float, pivot: Vector, target: Vector): Mesh {
        val moved = FloatArray(vertices.size)
        var i = 0
        while (i < vertices.size) {
            moved[i] = (vertices[i] - pivot.x) * scale + target.x
            moved[i + 1] = (vertices[i + 1] - pivot.y) * scale + target.y
            moved[i + 2] = (vertices[i + 2] - pivot.z) * scale + target.z
            i += 3
        }
        return Mesh(moved, faceMaterials, palette, smooth, uvs)
    }

    /**
     * Stands up a model authored with z as up - the convention of 3D-printing tools - in this
     * project's y-up world. A proper rotation about x, so the facet winding (and with it every
     * normal) survives.
     */
    fun zUpToYUp(): Mesh = remapped { x, y, z, out -> out[0] = x; out[1] = z; out[2] = -y }

    /** Turns the model about the vertical axis by [degrees], to face it where a scene wants. */
    fun rotatedY(degrees: Float): Mesh {
        val radians = Math.toRadians(degrees.toDouble())
        val cos = kotlin.math.cos(radians).toFloat()
        val sin = kotlin.math.sin(radians).toFloat()
        return remapped { x, y, z, out -> out[0] = cos * x + sin * z; out[1] = y; out[2] = -sin * x + cos * z }
    }

    private inline fun remapped(map: (Float, Float, Float, FloatArray) -> Unit): Mesh {
        val moved = FloatArray(vertices.size)
        val out = FloatArray(3)
        var i = 0
        while (i < vertices.size) {
            map(vertices[i], vertices[i + 1], vertices[i + 2], out)
            moved[i] = out[0]; moved[i + 1] = out[1]; moved[i + 2] = out[2]
            i += 3
        }
        return Mesh(moved, faceMaterials, palette, smooth, uvs)
    }

    /** Every triangle made of [material]. */
    fun toHittables(material: Material): List<Hittable> = toHittables { material }

    /**
     * Every triangle made of what the file gave it, when it gave colours, and of [fallback]
     * otherwise.
     */
    fun toColouredHittables(fallback: Material): List<Hittable> {
        val faces = faceMaterials
        if (faces == null || palette.isEmpty()) return toHittables(fallback)
        val materials = palette.map { it.material() }
        return toHittables { index -> materials.getOrElse(faces[index]) { fallback } }
    }

    private inline fun toHittables(materialOf: (Int) -> Material): List<Hittable> {
        val normals = if (smooth) smoothNormals else null
        return (0 until triangleCount).map { index ->
            val at = index * VERTEX_FLOATS
            Triangle(
                Point(vertices[at], vertices[at + 1], vertices[at + 2]),
                Point(vertices[at + 3], vertices[at + 4], vertices[at + 5]),
                Point(vertices[at + 6], vertices[at + 7], vertices[at + 8]),
                materialOf(index),
                normals?.copyOfRange(at, at + VERTEX_FLOATS)
            )
        }
    }

    companion object {
        /** See [smoothNormals]. */
        internal fun weldedNormals(vertices: FloatArray): FloatArray {
            val corners = vertices.size / 3
            fun key(value: Float): Int = if (value == 0.0f) 0 else java.lang.Float.floatToRawIntBits(value)
            // Open addressing over the corners' bits; each corner learns which corner stands for its point.
            var capacity = 1
            while (capacity < corners * 2) capacity = capacity shl 1
            val mask = capacity - 1
            val table = IntArray(capacity) { -1 }
            val representative = IntArray(corners)
            for (i in 0 until corners) {
                val kx = key(vertices[i * 3]); val ky = key(vertices[i * 3 + 1]); val kz = key(vertices[i * 3 + 2])
                var h = ((kx * -0x61c88647) xor (ky * -0x7a143595) xor (kz * -0x3d4d51cb)) and mask
                while (true) {
                    val slot = table[h]
                    if (slot == -1) {
                        table[h] = i
                        representative[i] = i
                        break
                    }
                    if (key(vertices[slot * 3]) == kx && key(vertices[slot * 3 + 1]) == ky && key(vertices[slot * 3 + 2]) == kz) {
                        representative[i] = slot
                        break
                    }
                    h = (h + 1) and mask
                }
            }
            // Each facet's unnormalised normal, added to its three points: bigger facets weigh more.
            val sums = FloatArray(corners * 3)
            for (corner in 0 until corners step 3) {
                val a = corner * 3
                val e1x = vertices[a + 3] - vertices[a]; val e1y = vertices[a + 4] - vertices[a + 1]; val e1z = vertices[a + 5] - vertices[a + 2]
                val e2x = vertices[a + 6] - vertices[a]; val e2y = vertices[a + 7] - vertices[a + 1]; val e2z = vertices[a + 8] - vertices[a + 2]
                val nx = e1y * e2z - e1z * e2y
                val ny = e1z * e2x - e1x * e2z
                val nz = e1x * e2y - e1y * e2x
                for (k in 0 until 3) {
                    val r = representative[corner + k] * 3
                    sums[r] += nx; sums[r + 1] += ny; sums[r + 2] += nz
                }
            }
            val normals = FloatArray(corners * 3)
            for (i in 0 until corners) {
                val r = representative[i] * 3
                val x = sums[r]; val y = sums[r + 1]; val z = sums[r + 2]
                val length = kotlin.math.sqrt(x * x + y * y + z * z)
                if (length > 0.0f) {
                    normals[i * 3] = x / length; normals[i * 3 + 1] = y / length; normals[i * 3 + 2] = z / length
                } else {
                    normals[i * 3 + 2] = 1.0f
                }
            }
            return normals
        }

        /** Height a loaded model is scaled to, in scene units - roughly the big spheres' size. */
        const val DEFAULT_TARGET_HEIGHT = 2.0f

        /** Three vertices of x/y/z per triangle. */
        const val VERTEX_FLOATS = 9

        /** Three corners of u/v per triangle, see [uvs]. */
        const val UV_FLOATS = 6
    }
}
