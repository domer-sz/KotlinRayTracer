package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.objects.MeshFill
import rayTraceTypescript.objects.SolidGrid
import rayTraceTypescript.objects.Sphere
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * A cube from (0, 0, 0) to (1, 1, 1), filled each way: whatever is packed in lies inside it,
 * no two pyramids' balls overlap, and the same seed packs the same.
 */
class MeshFillTest {

    private val material = Lambertian(Color(0.7f, 0.7f, 0.7f))

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

    /** How far a ball reaches past the cube's walls (negative: it stays clear of them). */
    private fun overhang(s: Sphere): Float {
        val p = s.center.origin
        return max(max(max(-p.x, p.x - 1f), max(-p.y, p.y - 1f)), max(-p.z, p.z - 1f)) + s.radius
    }

    private fun settings(style: MeshFill.Style, count: Int) =
        MeshFill.Settings(style, size = 0.05f, count = count, shell = 0.5f, resolution = 48, seed = 7)

    /** The balls pyramids are cut from: inside the cube, none overlapping another. */
    @Test
    fun packsBallsInsideWithoutOverlap() {
        val grid = SolidGrid(cube, 48)
        val balls = MeshFill.pack(grid, grid.cellsWithin(0.5f), 0.015f, 0.05f, 3000, Random(7))
        assertTrue(balls.size > 300, "only ${balls.size} balls")
        val spheres = (0 until balls.size).map { balls.sphere(it, material) }
        spheres.forEach { assertTrue(overhang(it) <= 1f / 48, "ball sticks out by ${overhang(it)}") }
        for (i in spheres.indices) for (j in i + 1 until spheres.size) {
            val a = spheres[i]; val b = spheres[j]
            val d = (a.center.origin - b.center.origin).length()
            assertTrue(d >= (a.radius + b.radius) * 0.999f, "balls $i and $j overlap")
        }
    }

    @Test
    fun threadsStayInsideAsChains() {
        val filling = MeshFill.fill(cube, material, settings(MeshFill.Style.THREADS, 5000))
        val beads = filling.spheres!!
        assertTrue(beads.size > 200, "only ${beads.size} beads")
        beads.forEach { assertTrue(overhang(it) <= 1f / 48 + it.radius * 0.5f, "bead outside by ${overhang(it)}") }
        // Consecutive beads of a chain sit 1.5 of its radii apart (at most 1.5 x 0.05, the largest),
        // even where its ends taper: nearly every bead has its neighbour in the list that close.
        val near = beads.indices.count { i ->
            (maxOf(0, i - 1)..minOf(beads.size - 1, i + 1)).any { j ->
                j != i && (beads[i].center.origin - beads[j].center.origin).length() < 1.5f * 0.05f * 1.01f
            }
        }
        assertTrue(near > beads.size * 0.9, "$near of ${beads.size} beads are in chains")
    }

    @Test
    fun pyramidsLieInsideTheCube() {
        val mesh = MeshFill.fill(cube, material, settings(MeshFill.Style.PYRAMIDS, 2000)).mesh!!
        assertTrue(mesh.triangleCount >= 6 * 300, "only ${mesh.triangleCount / 6} pyramids")
        val v = mesh.vertices
        val slack = 1f / 48 + 0.05f * 0.1f
        v.forEach { assertTrue(it >= -slack && it <= 1f + slack, "a corner at $it") }
    }

    @Test
    fun sameSeedSameFilling() {
        val a = MeshFill.fill(cube, material, settings(MeshFill.Style.THREADS, 2000)).spheres!!
        val b = MeshFill.fill(cube, material, settings(MeshFill.Style.THREADS, 2000)).spheres!!
        assertEquals(a.size, b.size)
        for (i in a.indices) {
            assertEquals(0f, (a[i].center.origin - b[i].center.origin).length(), 0f)
            assertTrue(abs(a[i].radius - b[i].radius) == 0f)
        }
    }

    private fun Vector.length() = sqrt(x * x + y * y + z * z)
}
