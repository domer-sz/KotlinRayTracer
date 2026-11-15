package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.objects.Triangle

class MeshTest {

    /** A single facet, two units tall, sitting well away from the origin. */
    private val mesh = Mesh(floatArrayOf(10f, 4f, 0f, 11f, 4f, 0f, 11f, 6f, 0f))

    @Test
    fun `standing on the floor centers the model and rests it on y zero`() {
        val box = mesh.standingOnFloor().bounds()
        assertEquals(0f, box.y.min, 1e-5f, "lowest point sits on the floor")
        assertEquals(Mesh.DEFAULT_TARGET_HEIGHT, box.y.size(), 1e-5f, "scaled to the default height")
        assertEquals(0f, (box.x.min + box.x.max) / 2f, 1e-5f, "centered in x")
        assertEquals(0f, (box.z.min + box.z.max) / 2f, 1e-5f, "centered in z")
    }

    @Test
    fun `keeps the authored size when no target height is given`() {
        val box = mesh.standingOnFloor(targetHeight = null).bounds()
        assertEquals(2f, box.y.size(), 1e-5f)
        assertEquals(0f, box.y.min, 1e-5f, "still dropped onto the floor")
    }

    @Test
    fun `builds one triangle per facet`() {
        val hittables = mesh.toHittables(Lambertian(Color(1f, 1f, 1f)))
        assertEquals(1, hittables.size)
        assertTrue(hittables.single() is Triangle)
    }

    @Test
    fun `a smooth octahedron's corners point straight out along the axes`() {
        // Each of the six points is shared by four facets; welded, their normals add up along the axis.
        val normals = octahedron().smoothNormals
        val vertices = octahedron().vertices
        for (corner in 0 until vertices.size / 3) {
            for (axis in 0 until 3) {
                assertEquals(vertices[corner * 3 + axis], normals[corner * 3 + axis], 1e-6f, "corner $corner, axis $axis")
            }
        }
    }

    @Test
    fun `two facets in one plane share its normal at every corner`() {
        // A square of two triangles in z = 0, wound to face +z; the big one weighs more, but both point the same way.
        val square = Mesh(floatArrayOf(0f, 0f, 0f, 3f, 0f, 0f, 3f, 1f, 0f, 0f, 0f, 0f, 3f, 1f, 0f, 0f, 1f, 0f))
        val normals = square.smoothNormals
        for (corner in 0 until 6) {
            assertEquals(0f, normals[corner * 3], 1e-6f)
            assertEquals(0f, normals[corner * 3 + 1], 1e-6f)
            assertEquals(1f, normals[corner * 3 + 2], 1e-6f)
        }
    }

    @Test
    fun `minus zero welds to plus zero`() {
        // Two facets meeting at the origin, one writing it as -0: a fold along x, so the shared
        // corner's normal is between the two facets' normals, the same for both.
        val fold = Mesh(floatArrayOf(
            0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f,
            -0f, -0f, -0f, 0f, 0f, 1f, 1f, 0f, 0f
        ))
        val normals = fold.smoothNormals
        for (axis in 0 until 3) assertEquals(normals[axis], normals[9 + axis], 1e-6f, "the origin, axis $axis")
        // The facets face +z and +y; welded, the origin's normal lies halfway. Not welded, the
        // first facet's corner would point along +z alone.
        assertEquals(0.70710677f, normals[1], 1e-6f, "the origin's normal, y")
        assertEquals(0.70710677f, normals[2], 1e-6f, "the origin's normal, z")
    }

    @Test
    fun `transforms keep a mesh smooth and a file's mesh is smooth unless asked`() {
        val smooth = octahedron().smoothed()
        assertTrue(smooth.rotatedY(30f).standingOnFloor().zUpToYUp().smooth)
        assertTrue(!octahedron().smooth, "a mesh made in code keeps its flat facets")
        val triangle = smooth.toHittables(Lambertian(Color(1f, 1f, 1f))).first() as Triangle
        assertTrue(triangle.cornerNormals != null, "a smooth mesh's triangles carry their corners' normals")
    }

    /** A unit octahedron, its vertices on the axes. */
    private fun octahedron(): Mesh = Mesh(
        floatArrayOf(
            1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f,
            0f, 1f, 0f, -1f, 0f, 0f, 0f, 0f, 1f,
            -1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, 1f,
            0f, -1f, 0f, 1f, 0f, 0f, 0f, 0f, 1f,
            0f, 1f, 0f, 1f, 0f, 0f, 0f, 0f, -1f,
            -1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, -1f,
            0f, -1f, 0f, -1f, 0f, 0f, 0f, 0f, -1f,
            1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, -1f
        )
    )
}
