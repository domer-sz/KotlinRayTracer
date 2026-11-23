package rayTraceTypescript.gpu

import rayTraceTypescript.objects.Mesh
import java.util.stream.IntStream
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * A ripple for a film instead of a turn (-Drt.turntable.wave=amplitude,wavelength[,lift], in scene
 * units): every point of a placed mesh is lifted by [lift] and by two waves running across the
 * floor - one along x, a smaller and shorter one at an angle - so a sheet waves like cloth in a
 * breeze. Over the film each wave goes round a whole number of times, so the last frame leads
 * into the first and the film loops.
 *
 * The mesh itself is only moved, never built again: a sponge keeps its holes, which ride along
 * with the surface rather than jumping from frame to frame. The corner normals follow from the
 * wave's gradient (for y' = y + w(x, z) a normal n goes to (n.x - w_x n.y, n.y, n.z - w_z n.y)),
 * which costs nothing next to welding ten million corners again.
 */
class Wave(val amplitude: Float, val wavelength: Float, val lift: Float, val native: Boolean = false) {

    private val k = 2f * PI.toFloat() / wavelength

    /** The second wave: 0.35 of the height, 1.4 times shorter, running at 53 degrees to the first. */
    private val k2 = 1.4f * k
    private val dx2 = 0.6f
    private val dz2 = 0.8f

    /** [mesh] rippled for [frame] of [frames]; its placement is the identity, so its space is the scene's. */
    fun deform(mesh: Mesh, frame: Int, frames: Int): Mesh {
        val phase = 2f * PI.toFloat() * frame / frames
        val v = mesh.vertices
        val n = mesh.smoothNormals
        val out = FloatArray(v.size)
        val normals = FloatArray(v.size)
        val corners = v.size / 3
        IntStream.range(0, (corners + 65535) / 65536).parallel().forEach { chunk ->
            for (c in chunk * 65536 until minOf(corners, (chunk + 1) * 65536)) {
                val o = 3 * c
                val x = v[o]; val y = v[o + 1]; val z = v[o + 2]
                val a1 = k * x - phase
                val a2 = k2 * (dx2 * x + dz2 * z) - 2f * phase
                val w = amplitude * (sin(a1) + 0.35f * sin(a2))
                val wx = amplitude * (k * cos(a1) + 0.35f * k2 * dx2 * cos(a2))
                val wz = amplitude * (0.35f * k2 * dz2 * cos(a2))
                out[o] = x; out[o + 1] = y + lift + w; out[o + 2] = z
                val nx = n[o] - wx * n[o + 1]; val ny = n[o + 1]; val nz = n[o + 2] - wz * n[o + 1]
                val l = sqrt(nx * nx + ny * ny + nz * nz).takeIf { it > 1e-12f } ?: 1f
                normals[o] = nx / l; normals[o + 1] = ny / l; normals[o + 2] = nz / l
            }
        }
        return Mesh(out, mesh.faceMaterials, mesh.palette, mesh.smooth, mesh.uvs, normals)
    }

    companion object {
        /**
         * From -Drt.turntable.wave=amplitude,wavelength[,lift]; null when unset. -Drt.turntable.wave.native=on
         * ripples the model a sponge was made from and builds the sponge again every frame, instead of
         * moving the finished one: what a changing model really gives, its holes laid out anew each frame.
         */
        fun fromProperties(): Wave? {
            val parts = System.getProperty("rt.turntable.wave")?.split(',')?.map { it.trim().toFloat() } ?: return null
            require(parts.size in 2..3 && parts[1] > 0f) { "-Drt.turntable.wave=amplitude,wavelength[,lift]" }
            val native = System.getProperty("rt.turntable.wave.native")?.lowercase() in setOf("on", "true", "1", "")
            return Wave(parts[0], parts[1], parts.getOrElse(2) { 1.4f * parts[0] }, native)
        }
    }
}
