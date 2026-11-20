package rayTraceTypescript.objects

import kotlin.math.sqrt

/**
 * The viewer's added shapes as triangle meshes, for the forms that convert a mesh (the lattice,
 * pyramids, diamonds): a ball one unit across and a box one unit tall, both standing on y = 0
 * with their footprint centred on the origin.
 */
object MeshShapes {

    /** An icosphere of diameter 1 centred at (0, 0.5, 0), [subdivisions] times refined (4: 5 120 triangles). */
    fun ball(subdivisions: Int = 4): Mesh {
        val t = ((1.0 + sqrt(5.0)) / 2.0).toFloat()
        var corners = arrayOf(
            floatArrayOf(-1f, t, 0f), floatArrayOf(1f, t, 0f), floatArrayOf(-1f, -t, 0f), floatArrayOf(1f, -t, 0f),
            floatArrayOf(0f, -1f, t), floatArrayOf(0f, 1f, t), floatArrayOf(0f, -1f, -t), floatArrayOf(0f, 1f, -t),
            floatArrayOf(t, 0f, -1f), floatArrayOf(t, 0f, 1f), floatArrayOf(-t, 0f, -1f), floatArrayOf(-t, 0f, 1f)
        ).map { unit(it) }.toMutableList()
        var faces = listOf(
            intArrayOf(0, 11, 5), intArrayOf(0, 5, 1), intArrayOf(0, 1, 7), intArrayOf(0, 7, 10), intArrayOf(0, 10, 11),
            intArrayOf(1, 5, 9), intArrayOf(5, 11, 4), intArrayOf(11, 10, 2), intArrayOf(10, 7, 6), intArrayOf(7, 1, 8),
            intArrayOf(3, 9, 4), intArrayOf(3, 4, 2), intArrayOf(3, 2, 6), intArrayOf(3, 6, 8), intArrayOf(3, 8, 9),
            intArrayOf(4, 9, 5), intArrayOf(2, 4, 11), intArrayOf(6, 2, 10), intArrayOf(8, 6, 7), intArrayOf(9, 8, 1)
        )
        repeat(subdivisions) {
            val cache = HashMap<Long, Int>()
            fun middle(a: Int, b: Int): Int {
                val key = minOf(a, b).toLong() shl 32 or maxOf(a, b).toLong()
                return cache.getOrPut(key) {
                    val p = corners[a]; val q = corners[b]
                    corners += unit(floatArrayOf((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2))
                    corners.size - 1
                }
            }
            val next = ArrayList<IntArray>()
            for (f in faces) {
                val ab = middle(f[0], f[1]); val bc = middle(f[1], f[2]); val ca = middle(f[2], f[0])
                next += intArrayOf(f[0], ab, ca); next += intArrayOf(f[1], bc, ab)
                next += intArrayOf(f[2], ca, bc); next += intArrayOf(ab, bc, ca)
            }
            faces = next
        }
        val out = FloatArray(faces.size * 9)
        faces.forEachIndexed { i, f ->
            for (c in 0 until 3) {
                val p = corners[f[c]]
                out[9 * i + 3 * c] = p[0] * 0.5f
                out[9 * i + 3 * c + 1] = p[1] * 0.5f + 0.5f
                out[9 * i + 3 * c + 2] = p[2] * 0.5f
            }
        }
        return Mesh(out, smooth = true)
    }

    /** A box [width] wide and [depth] deep, one unit tall, on y = 0, wound outwards. */
    fun box(width: Float, depth: Float): Mesh {
        val hx = width / 2f; val hz = depth / 2f
        fun corner(i: Int) = floatArrayOf(if (i and 1 == 0) -hx else hx, if (i shr 1 and 1 == 0) 0f else 1f, if (i shr 2 and 1 == 0) -hz else hz)
        val quads = listOf(
            intArrayOf(0, 2, 3, 1), intArrayOf(4, 5, 7, 6), intArrayOf(0, 1, 5, 4),
            intArrayOf(2, 6, 7, 3), intArrayOf(0, 4, 6, 2), intArrayOf(1, 3, 7, 5)
        )
        val out = ArrayList<Float>()
        for (q in quads) for (t in listOf(intArrayOf(q[0], q[1], q[2]), intArrayOf(q[0], q[2], q[3]))) for (i in t) out.addAll(corner(i).toList())
        return Mesh(out.toFloatArray())
    }

    private fun unit(p: FloatArray): FloatArray {
        val l = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
        return floatArrayOf(p[0] / l, p[1] / l, p[2] / l)
    }
}
