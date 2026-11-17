package rayTraceTypescript.objects.formats

import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.objects.MeshMaterial

/** Growable float storage; mesh files rarely say up front how much data they hold. */
internal class FloatList(initialCapacity: Int = 1024) {
    private var data = FloatArray(initialCapacity.coerceAtLeast(16))
    var size: Int = 0
        private set

    val tripleCount: Int get() = size / 3

    fun add(value: Float) {
        if (size == data.size) data = data.copyOf(data.size * 2)
        data[size++] = value
    }

    fun add(x: Float, y: Float, z: Float) {
        add(x); add(y); add(z)
    }

    operator fun get(index: Int): Float = data[index]

    fun toFloatArray(): FloatArray = data.copyOf(size)
}

/** Collects triangles for the format parsers and turns them into a [Mesh]. */
internal class MeshBuilder(expectedTriangles: Int = 128) {
    private val data = FloatList(expectedTriangles * Mesh.VERTEX_FLOATS)
    /** Each triangle's material so far, when the file names any. */
    private var faceMaterials = IntArray(0)

    val triangleCount: Int get() = data.size / Mesh.VERTEX_FLOATS

    /** The palette entry the next triangles are made of; -1 while the file has named none. */
    var material: Int = -1

    /** Each triangle's corner texture coordinates, once the file gave any (see [Mesh.uvs]). */
    private var uvData: FloatList? = null

    /** The corners' texture coordinates of the triangle just added, or none for it. */
    private fun cornerUvs(uvs: FloatArray?) {
        if (uvs == null && uvData == null) return
        val list = uvData ?: FloatList((triangleCount + 1024) * Mesh.UV_FLOATS).also { fresh ->
            // The triangles before the first that had coordinates: none.
            repeat((triangleCount - 1) * Mesh.UV_FLOATS) { fresh.add(Float.NaN) }
            uvData = fresh
        }
        if (uvs == null) repeat(Mesh.UV_FLOATS) { list.add(Float.NaN) } else uvs.forEach { list.add(it) }
    }

    fun triangle(
        ax: Float, ay: Float, az: Float,
        bx: Float, by: Float, bz: Float,
        cx: Float, cy: Float, cz: Float
    ) {
        val index = triangleCount
        data.add(ax, ay, az)
        data.add(bx, by, bz)
        data.add(cx, cy, cz)
        if (material >= 0 || faceMaterials.isNotEmpty()) {
            if (faceMaterials.size <= index) faceMaterials = faceMaterials.copyOf(maxOf(index + 1, faceMaterials.size * 2, 1024))
            faceMaterials[index] = material
        }
    }

    /**
     * Adds one face given as indices into a flat x/y/z vertex list. Faces with more than three
     * corners are triangulated as a fan, which is what every format here assumes for the convex
     * polygons it allows.
     */
    fun polygon(positions: FloatList, indices: IntArray, texCoords: FloatList? = null, uvIndices: IntArray? = null) {
        if (indices.size < 3) return
        // Texture coordinates only when every corner names a valid one.
        val uvs = uvIndices?.takeIf { texCoords != null && it.size == indices.size && it.all { i -> i >= 0 && i * 2 + 1 < texCoords.size } }
        for (corner in 1 until indices.size - 1) {
            val a = indices[0] * 3
            val b = indices[corner] * 3
            val c = indices[corner + 1] * 3
            require(a >= 0 && c + 2 < positions.size) { "Face references a vertex outside the file" }
            triangle(
                positions[a], positions[a + 1], positions[a + 2],
                positions[b], positions[b + 1], positions[b + 2],
                positions[c], positions[c + 1], positions[c + 2]
            )
            cornerUvs(uvs?.let { u ->
                val t = texCoords!!
                floatArrayOf(
                    t[u[0] * 2], t[u[0] * 2 + 1],
                    t[u[corner] * 2], t[u[corner] * 2 + 1],
                    t[u[corner + 1] * 2], t[u[corner + 1] * 2 + 1]
                )
            })
        }
    }

    /**
     * The mesh, with its faces' materials when [palette] names some: faces before the first
     * material, and faces naming one the palette lacks, get entry 0.
     */
    fun build(palette: List<MeshMaterial> = emptyList()): Mesh {
        val uvs = uvData?.toFloatArray()
        if (palette.isEmpty() || faceMaterials.isEmpty()) return Mesh(data.toFloatArray(), uvs = uvs)
        val faces = IntArray(triangleCount) { i ->
            val m = if (i < faceMaterials.size) faceMaterials[i] else 0
            if (m in palette.indices) m else 0
        }
        return Mesh(data.toFloatArray(), faces, palette, uvs = uvs)
    }
}
