package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.objects.LatticeFill
import rayTraceTypescript.objects.Mesh

/**
 * A unit cube turned into a sponge: a closed surface, wound outwards, near the cube's own
 * surface, full of holes, far less solid than the cube, and the same each time for the same seed.
 */
class LatticeFillTest {

    /** Twelve triangles of the unit cube, wound outwards. */
    private val cube: Mesh = run {
        val c = { i: Int -> floatArrayOf((i and 1).toFloat(), (i shr 1 and 1).toFloat(), (i shr 2 and 1).toFloat()) }
        val quads = listOf(
            intArrayOf(0, 2, 3, 1), intArrayOf(4, 5, 7, 6), intArrayOf(0, 1, 5, 4),
            intArrayOf(2, 6, 7, 3), intArrayOf(0, 4, 6, 2), intArrayOf(1, 3, 7, 5)
        )
        val v = ArrayList<Float>()
        for (q in quads) for (t in listOf(intArrayOf(q[0], q[1], q[2]), intArrayOf(q[0], q[2], q[3]))) for (i in t) v.addAll(c(i).toList())
        Mesh(v.toFloatArray())
    }

    private val settings = LatticeFill.Settings(pitch = 0.3f, detail = 16, seed = 5)
    private val sponge: Mesh by lazy { LatticeFill.build(cube, settings) }

    /** Vertex indices by exact position, and the edges and faces they make. */
    private class Topology(mesh: Mesh) {
        val vertices = HashMap<Triple<Float, Float, Float>, Int>()
        val edges = HashMap<Long, Int>()
        val faces = mesh.triangleCount

        init {
            val v = mesh.vertices
            fun id(o: Int) = vertices.getOrPut(Triple(v[o], v[o + 1], v[o + 2])) { vertices.size }
            for (t in 0 until faces) {
                val a = id(9 * t); val b = id(9 * t + 3); val c = id(9 * t + 6)
                for ((p, q) in listOf(a to b, b to c, c to a)) {
                    val key = minOf(p, q).toLong() shl 32 or maxOf(p, q).toLong()
                    edges[key] = (edges[key] ?: 0) + 1
                }
            }
        }
    }

    @Test
    fun surfaceIsClosedAndWoundOutwards() {
        val v = sponge.vertices
        var volume = 0.0
        for (t in 0 until sponge.triangleCount) {
            val o = 9 * t
            val ax = v[o].toDouble(); val ay = v[o + 1].toDouble(); val az = v[o + 2].toDouble()
            val bx = v[o + 3].toDouble(); val by = v[o + 4].toDouble(); val bz = v[o + 5].toDouble()
            val cx = v[o + 6].toDouble(); val cy = v[o + 7].toDouble(); val cz = v[o + 8].toDouble()
            volume += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
        }
        assertTrue(volume / 6.0 > 0.0, "the sponge has no outward winding: signed volume ${volume / 6.0}")
        // A perforated shell: an unbroken one 0.252 thick round the cube would hold about 6 x 0.252 = 1.5;
        // the holes take well over a third of it.
        assertTrue(volume / 6.0 in 0.1..0.9, "sponge volume ${volume / 6.0}")
        val topology = Topology(sponge)
        val shared = topology.edges.values.count { it == 2 }
        assertTrue(shared > topology.edges.size * 0.99, "only $shared of ${topology.edges.size} edges are shared by two faces")
    }

    @Test
    fun staysNearTheCubeAndHasHoles() {
        val v = sponge.vertices
        // The shell reaches half its thickness (1.5 wall widths of 0.084) past the surface, and a little for the grid.
        val limit = 0.3f * 0.28f * 3f / 2f + 0.01f
        for (i in v.indices step 3) {
            // Distance to the cube's surface: how far inside or outside its faces the point is.
            val inside = minOf(minOf(v[i], 1f - v[i]), minOf(minOf(v[i + 1], 1f - v[i + 1]), minOf(v[i + 2], 1f - v[i + 2])))
            assertTrue(kotlin.math.abs(inside) <= limit, "a vertex ${kotlin.math.abs(inside)} from the surface, past $limit")
        }
        val topology = Topology(sponge)
        // Euler's characteristic V - E + F = 2 - 2g: far below 2 means many holes.
        val euler = topology.vertices.size - topology.edges.size + topology.faces
        assertTrue(euler < -10, "Euler characteristic $euler: too few holes")
        assertTrue(sponge.triangleCount > 2000, "only ${sponge.triangleCount} triangles")
    }

    @Test
    fun sameSeedSameLattice() {
        val again = LatticeFill.build(cube, settings)
        assertEquals(sponge.triangleCount, again.triangleCount)
        assertTrue(sponge.vertices.contentEquals(again.vertices))
    }

    @Test
    fun anotherSeedMovesTheHoles() {
        val other = LatticeFill.build(cube, settings.copy(seed = 6))
        assertTrue(!sponge.vertices.contentEquals(other.vertices))
    }
}
