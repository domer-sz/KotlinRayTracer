package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.MeshBlock
import rayTraceTypescript.gpu.PlacedMesh
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.objects.Mesh

/**
 * What the viewer asks when it is clicked: how far along a ray a placed mesh is. It has to agree
 * with the ordinary intersection over the moved triangles, which it avoids building.
 */
class PlacedMeshPickTest {

    private fun placed(scale: Float, degrees: Float, offset: Vector) =
        PlacedMesh(MeshBlock(octahedron()), scale, degrees, offset, Lambertian(Color(0.5f, 0.5f, 0.5f)))

    @Test
    fun `pick finds the same distance as the intersection over the moved mesh`() {
        val mesh = placed(1.7f, 37.0f, Vector(0.4f, 0.0f, -0.6f))
        var hits = 0
        var misses = 0
        for (i in -12..12) for (j in -12..12) {
            // Rays from a camera in front, fanned over the mesh and past it.
            val origin = Point(0.5f, 1.4f, 6.0f)
            val aim = Point(0.4f + i * 0.17f, 1.0f + j * 0.17f, -0.6f)
            val ray = Ray(origin, aim - origin)
            val expected = mesh.hit(ray, Interval(0.001f, Float.POSITIVE_INFINITY))?.t
            val picked = mesh.pick(ray)
            if (expected == null) {
                assertNull(picked, "ray ($i, $j) misses but pick found $picked")
                misses++
            } else {
                assertNotNull(picked, "ray ($i, $j) hits at $expected but pick missed")
                assertEquals(expected, picked!!, 1e-3f * maxOf(1f, expected), "ray ($i, $j)")
                hits++
            }
        }
        assertTrue(hits > 50, "only $hits rays hit the mesh")
        assertTrue(misses > 50, "only $misses rays missed the mesh")
    }

    @Test
    fun `pick sees the inside of a face from behind`() {
        val mesh = placed(2.0f, 0.0f, Vector(0.0f, 0.0f, 0.0f))
        // From inside the octahedron outwards: the back of a face still counts.
        val ray = Ray(Point(0.1f, 0.1f, 0.1f), Vector(1.0f, 1.0f, 1.0f))
        assertNotNull(mesh.pick(ray))
    }

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
