package rayTraceTypescript.objects

import java.util.stream.IntStream
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Turns a model into a sponge: a soft, thick foam that follows its surface, round uneven holes
 * beside each other joined by rounded walls and tunnels - the user's Blender recipe (Cell Fracture
 * or Voronoi, Inset Faces, Solidify, Subdivision Surface; or Houdini's VDB smoothing) as a field.
 *
 * - **The shell** is every point within half [Settings.thickness] wall widths of the model's surface,
 *   by the distance to dense samples laid on every triangle (their tangent plane near the surface),
 *   not to a voxel grid, so its sides are smooth and an open or badly wound file does as well.
 * - **The cells** are a Voronoi diagram on the surface: seeds thinned to 0.9 pitches, then 3 passes of
 *   Lloyd's algorithm - few, so the cells and the holes stay uneven.
 * - **The hole** in each cell leaves a wall within half a wall width of the cell's boundaries; the two
 *   nearest boundaries are joined by a smooth minimum ([Settings.poreWall]), which rounds the holes'
 *   corners as a subdivision surface would.
 * - **The solid** is where both hold, joined with a smooth maximum, which softens every rim. Its
 *   isosurface is extracted with surface nets and handed back as a smooth-shaded [Mesh] wound
 *   outwards (glass takes front and back from the winding).
 *
 * Sizes are fractions of the model's height, so a sponge of a model looks the same at any scale
 * and can be built once and placed at any size. (A honeycomb of hexagonal holes was here too, and
 * removed on 2026-11-20.)
 */
object LatticeFill {

    data class Settings(
        /** Distance between the middles of neighbouring holes, as a fraction of the model's height. */
        val pitch: Float = 0.035f,
        /** Wall width, as a fraction of the pitch. */
        val wall: Float = 0.28f,
        /** Shell thickness, in wall widths. */
        val thickness: Float = 3f,
        /** Rounding of the rims, in wall widths. */
        val round: Float = 0.6f,
        /** Grid cells along one pitch: the mesh's fineness. */
        val detail: Int = 22,
        val seed: Long = 1L,
        /** Rounding of the holes' corners, as a fraction of the pitch: the larger, the rounder the holes. */
        val poreWall: Float = 0.35f
    ) {
        companion object {
            /** From -Drt.infill.lattice.pitch|wall|thickness|round|detail|pore and -Drt.infill.seed. */
            fun fromProperties(): Settings {
                fun float(key: String) = System.getProperty("rt.infill.lattice.$key")?.toFloatOrNull()?.takeIf { it > 0f }
                val d = Settings()
                return Settings(
                    poreWall = float("pore") ?: d.poreWall,
                    pitch = float("pitch") ?: d.pitch,
                    wall = (float("wall") ?: d.wall).coerceAtMost(0.8f),
                    thickness = float("thickness") ?: d.thickness,
                    round = System.getProperty("rt.infill.lattice.round")?.toFloatOrNull()?.takeIf { it >= 0f } ?: d.round,
                    detail = System.getProperty("rt.infill.lattice.detail")?.toIntOrNull()?.coerceIn(12, 96) ?: d.detail,
                    seed = System.getProperty("rt.infill.seed")?.toLongOrNull() ?: d.seed
                )
            }
        }
    }

    /** The sponge of [mesh]: its triangles (smooth-shaded) and how long it took to build. */
    fun build(mesh: Mesh, settings: Settings = Settings()): Mesh {
        val box = mesh.bounds()
        val height = (box.y.max - box.y.min).takeIf { it > 0f }
            ?: max(box.x.max - box.x.min, box.z.max - box.z.min)
        val pitch = settings.pitch * height
        val wallWidth = pitch * settings.wall
        val shellHalf = wallWidth * settings.thickness / 2f
        val wallHalf = wallWidth / 2f
        val k = max(wallWidth * settings.round, 1e-6f)
        val cell = pitch / settings.detail
        val reach = shellHalf + k + 2f * cell

        val t0 = System.nanoTime()
        val log = System.getProperty("rt.infill.lattice.log") != null
        fun lap(what: String) { if (log) System.err.println("[sponge] $what: %.1f s".format((System.nanoTime() - t0) / 1e9)) }
        val samples = surfaceSamples(mesh, cell * 3f)
        lap("${samples.n} surface samples")
        val near = PointGrid(samples.xyz, samples.n, max(reach, 2f * cell), box.x.min, box.y.min, box.z.min,
            box.x.max, box.y.max, box.z.max, reach + 2f * cell)
        lap("sample grid")
        val seeds = relax(samples, pitch, settings, box, 3)
        lap("${seeds.size / 3} cells relaxed")
        val seedGrid = SeedGrid(seeds, pitch * 1.6f, box.x.min, box.y.min, box.z.min, box.x.max, box.y.max, box.z.max)

        val big = 1e6f
        val cornerRound = max(pitch * settings.poreWall, 1e-6f)
        // The shell's distance and the wall's, joined with a smooth maximum: negative is solid.
        val field = fun(x: Float, y: Float, z: Float): Float {
            if (!near.mayBeNear(x, y, z)) return big
            val at = near.nearestIndex(x, y, z)
            if (at < 0) return big
            val ex = x - samples.xyz[3 * at]; val ey = y - samples.xyz[3 * at + 1]; val ez = z - samples.xyz[3 * at + 2]
            val d = sqrt(ex * ex + ey * ey + ez * ez)
            // The tangent plane only where the point lies nearly along the sample's normal (over a face, not off an edge or a corner).
            val plane = abs(ex * samples.normals[3 * at] + ey * samples.normals[3 * at + 1] + ez * samples.normals[3 * at + 2])
            val surface = if (d < shellHalf + k && plane > 0.8f * d) plane else d
            val a = surface - shellHalf
            if (a > k) return a // too far outside for the wall to matter, and nowhere near the surface of the solid
            // The wall: within half its width of a cell boundary, measured to a smooth minimum of the two
            // nearest boundaries, which rounds each hole's corners as a subdivision surface would.
            val b = seedGrid.wallDistance(x, y, z, cornerRound) - wallHalf
            val h = max(k - abs(a - b), 0f) / k
            return max(a, b) + h * h * k * 0.25f
        }

        val result = SurfaceNets.extract(
            field, box.x.min - reach - 2f * cell, box.y.min - reach - 2f * cell, box.z.min - reach - 2f * cell,
            box.x.max + reach + 2f * cell, box.y.max + reach + 2f * cell, box.z.max + reach + 2f * cell, cell
        )
        lap("surface extracted")
        return result
    }

    private class Samples(val xyz: FloatArray, val normals: FloatArray, val n: Int)

    /** Points laid over every triangle at most [spacing] apart (one at the middle of a small one). */
    private fun surfaceSamples(mesh: Mesh, spacing: Float): Samples {
        val v = mesh.vertices
        val corners = if (mesh.smooth) mesh.smoothNormals else null
        var out = FloatArray(3 * 4096)
        var normals = FloatArray(3 * 4096)
        var n = 0
        var fx = 0f; var fy = 1f; var fz = 0f
        fun add(x: Float, y: Float, z: Float) {
            if (3 * n + 3 > out.size) { out = out.copyOf(out.size * 2); normals = normals.copyOf(normals.size * 2) }
            out[3 * n] = x; out[3 * n + 1] = y; out[3 * n + 2] = z
            normals[3 * n] = fx; normals[3 * n + 1] = fy; normals[3 * n + 2] = fz
            n++
        }
        for (t in 0 until mesh.triangleCount) {
            val o = 9 * t
            val ax = v[o]; val ay = v[o + 1]; val az = v[o + 2]
            val bx = v[o + 3]; val by = v[o + 4]; val bz = v[o + 5]
            val cx = v[o + 6]; val cy = v[o + 7]; val cz = v[o + 8]
            fun len(px: Float, py: Float, pz: Float, qx: Float, qy: Float, qz: Float) =
                sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy) + (pz - qz) * (pz - qz))
            val longest = max(len(ax, ay, az, bx, by, bz), max(len(bx, by, bz, cx, cy, cz), len(cx, cy, cz, ax, ay, az)))
            // The facet's normal, from its winding, for telling the outside of the surface from the inside.
            val ux = bx - ax; val uy = by - ay; val uz = bz - az; val wx = cx - ax; val wy = cy - ay; val wz = cz - az
            val nx0 = uy * wz - uz * wy; val ny0 = uz * wx - ux * wz; val nz0 = ux * wy - uy * wx
            val nl = sqrt(nx0 * nx0 + ny0 * ny0 + nz0 * nz0)
            if (nl > 0f) { fx = nx0 / nl; fy = ny0 / nl; fz = nz0 / nl }
            val flatX = fx; val flatY = fy; val flatZ = fz
            // On a smooth mesh each sample takes the corner normals blended by where it lies.
            fun blend(u: Float, w: Float) {
                if (corners == null) return
                val g = 1f - u - w
                var bx2 = g * corners[o] + u * corners[o + 3] + w * corners[o + 6]
                var by2 = g * corners[o + 1] + u * corners[o + 4] + w * corners[o + 7]
                var bz2 = g * corners[o + 2] + u * corners[o + 5] + w * corners[o + 8]
                val l = sqrt(bx2 * bx2 + by2 * by2 + bz2 * bz2)
                if (l > 1e-6f && bx2 * flatX + by2 * flatY + bz2 * flatZ > 0f) { bx2 /= l; by2 /= l; bz2 /= l; fx = bx2; fy = by2; fz = bz2 }
                else { fx = flatX; fy = flatY; fz = flatZ }
            }
            val div = ceil(longest / spacing).toInt().coerceIn(1, 2048)
            if (div == 1) {
                blend(1f / 3f, 1f / 3f)
                add((ax + bx + cx) / 3f, (ay + by + cy) / 3f, (az + bz + cz) / 3f)
            } else {
                for (i in 0..div) for (j in 0..div - i) {
                    val u = i.toFloat() / div; val w = j.toFloat() / div
                    blend(u, w)
                    add(ax + u * (bx - ax) + w * (cx - ax), ay + u * (by - ay) + w * (cy - ay), az + u * (bz - az) + w * (cz - az))
                }
            }
            require(n < 80_000_000) { "the sponge would need more than 80 million surface samples; raise the pitch or lower the detail" }
        }
        return Samples(out, normals, n)
    }

    /**
     * Seeds spread over the surface at least 0.9 pitches apart, then moved by Lloyd's algorithm: each
     * goes to the middle of the surface it is nearest to, snapped back onto the surface.
     */
    private fun relax(samples: Samples, pitch: Float, settings: Settings, box: rayTraceTypescript.Aabb, passes: Int): FloatArray {
        val random = Random(settings.seed)
        val n = samples.n
        val xyz = samples.xyz
        // Thinned in a random order, so the seeds do not follow the triangles' order.
        val balls = MeshFill.Balls(pitch * 0.9f)
        val order = IntArray(n) { it }
        for (i in n - 1 downTo 1) { val j = random.nextInt(i + 1); val t = order[i]; order[i] = order[j]; order[j] = t }
        for (i in order) {
            val x = xyz[3 * i]; val y = xyz[3 * i + 1]; val z = xyz[3 * i + 2]
            if (balls.clear(x, y, z, pitch * 0.45f)) balls.add(x, y, z, pitch * 0.45f)
        }
        var seeds = FloatArray(3 * balls.size) { 0f }
        for (i in 0 until balls.size) { seeds[3 * i] = balls.x(i); seeds[3 * i + 1] = balls.y(i); seeds[3 * i + 2] = balls.z(i) }
        val m = balls.size
        require(m >= 4) { "the model is too small for a sponge of this pitch: ${m} cells" }

        // Lloyd: a stride through the samples keeps each pass cheap.
        val stride = max(1, n / 400_000)
        repeat(passes) {
            val grid = SeedGrid(seeds, pitch * 1.6f, box.x.min, box.y.min, box.z.min, box.x.max, box.y.max, box.z.max)
            val sum = DoubleArray(3 * m); val count = IntArray(m)
            var i = 0
            while (i < n) {
                val s = grid.nearestSeed(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2])
                sum[3 * s] += xyz[3 * i]; sum[3 * s + 1] += xyz[3 * i + 1]; sum[3 * s + 2] += xyz[3 * i + 2]
                count[s]++
                i += stride
            }
            // Snap each mean to the nearest sample that went to the same seed.
            val best = FloatArray(m) { Float.POSITIVE_INFINITY }
            val next = seeds.copyOf()
            i = 0
            while (i < n) {
                val s = grid.nearestSeed(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2])
                if (count[s] > 0) {
                    val dx = xyz[3 * i] - (sum[3 * s] / count[s]).toFloat()
                    val dy = xyz[3 * i + 1] - (sum[3 * s + 1] / count[s]).toFloat()
                    val dz = xyz[3 * i + 2] - (sum[3 * s + 2] / count[s]).toFloat()
                    val d = dx * dx + dy * dy + dz * dz
                    if (d < best[s]) { best[s] = d; next[3 * s] = xyz[3 * i]; next[3 * s + 1] = xyz[3 * i + 1]; next[3 * s + 2] = xyz[3 * i + 2] }
                }
                i += stride
            }
            seeds = next
        }
        return seeds
    }

    /** Surface samples in a grid of cells at least a reach wide: the distance to the surface of a point near it. */
    private class PointGrid(
        val xyz: FloatArray, n: Int, val cell: Float,
        minX: Float, minY: Float, minZ: Float, maxX: Float, maxY: Float, maxZ: Float, pad: Float
    ) {
        private val ox = minX - pad - cell
        private val oy = minY - pad - cell
        private val oz = minZ - pad - cell
        private val dx = ceil((maxX + pad + cell - ox) / cell).toInt() + 2
        private val dy = ceil((maxY + pad + cell - oy) / cell).toInt() + 2
        private val dz = ceil((maxZ + pad + cell - oz) / cell).toInt() + 2
        private val start = IntArray(dx * dy * dz + 1)
        private val items = IntArray(n)
        private val nearCell: BooleanArray

        init {
            fun slot(i: Int) = index(
                floor((xyz[3 * i] - ox) / cell).toInt(), floor((xyz[3 * i + 1] - oy) / cell).toInt(), floor((xyz[3 * i + 2] - oz) / cell).toInt()
            )
            for (i in 0 until n) start[slot(i) + 1]++
            for (c in 0 until dx * dy * dz) start[c + 1] += start[c]
            val fill = IntArray(dx * dy * dz)
            for (i in 0 until n) { val s = slot(i); items[start[s] + fill[s]++] = i }
            // A cell is near when it or a neighbour holds a sample.
            nearCell = BooleanArray(dx * dy * dz)
            for (z in 1 until dz - 1) for (y in 1 until dy - 1) for (x in 1 until dx - 1) {
                if (start[index(x, y, z) + 1] > start[index(x, y, z)]) {
                    for (c in -1..1) for (b in -1..1) for (a in -1..1) nearCell[index(x + a, y + b, z + c)] = true
                }
            }
        }

        private fun index(x: Int, y: Int, z: Int) = (z * dy + y) * dx + x

        fun mayBeNear(x: Float, y: Float, z: Float): Boolean {
            val ix = floor((x - ox) / cell).toInt(); val iy = floor((y - oy) / cell).toInt(); val iz = floor((z - oz) / cell).toInt()
            return ix in 0 until dx && iy in 0 until dy && iz in 0 until dz && nearCell[index(ix, iy, iz)]
        }

        /** The nearest sample within a cell's width, or -1. */
        fun nearestIndex(x: Float, y: Float, z: Float): Int {
            val ix = floor((x - ox) / cell).toInt(); val iy = floor((y - oy) / cell).toInt(); val iz = floor((z - oz) / cell).toInt()
            var best = Float.POSITIVE_INFINITY; var at = -1
            for (c in max(iz - 1, 0)..min(iz + 1, dz - 1)) for (b in max(iy - 1, 0)..min(iy + 1, dy - 1)) for (a in max(ix - 1, 0)..min(ix + 1, dx - 1)) {
                val s = index(a, b, c)
                for (q in start[s] until start[s + 1]) {
                    val p = 3 * items[q]
                    val ex = xyz[p] - x; val ey = xyz[p + 1] - y; val ez = xyz[p + 2] - z
                    val d = ex * ex + ey * ey + ez * ez
                    if (d < best) { best = d; at = items[q] }
                }
            }
            return at
        }

    }

    /** Seeds in a grid: the nearest seed to a point, and the distance to the nearest wall between seeds. */
    private class SeedGrid(
        val seeds: FloatArray, val cell: Float,
        minX: Float, minY: Float, minZ: Float, maxX: Float, maxY: Float, maxZ: Float
    ) {
        private val m = seeds.size / 3
        private val ox = minX - 2 * cell
        private val oy = minY - 2 * cell
        private val oz = minZ - 2 * cell
        private val dx = ceil((maxX + 2 * cell - ox) / cell).toInt() + 1
        private val dy = ceil((maxY + 2 * cell - oy) / cell).toInt() + 1
        private val dz = ceil((maxZ + 2 * cell - oz) / cell).toInt() + 1
        private val start = IntArray(dx * dy * dz + 1)
        private val items = IntArray(m)

        init {
            fun slot(i: Int) = ((floor((seeds[3 * i + 2] - oz) / cell).toInt() * dy + floor((seeds[3 * i + 1] - oy) / cell).toInt()) * dx +
                floor((seeds[3 * i] - ox) / cell).toInt())
            for (i in 0 until m) start[slot(i) + 1]++
            for (c in 0 until dx * dy * dz) start[c + 1] += start[c]
            val fill = IntArray(dx * dy * dz)
            for (i in 0 until m) { val s = slot(i); items[start[s] + fill[s]++] = i }
        }

        private inline fun forNear(x: Float, y: Float, z: Float, visit: (Int) -> Unit) {
            val ix = floor((x - ox) / cell).toInt(); val iy = floor((y - oy) / cell).toInt(); val iz = floor((z - oz) / cell).toInt()
            for (c in max(iz - 1, 0)..min(iz + 1, dz - 1)) for (b in max(iy - 1, 0)..min(iy + 1, dy - 1)) for (a in max(ix - 1, 0)..min(ix + 1, dx - 1)) {
                val s = (c * dy + b) * dx + a
                for (q in start[s] until start[s + 1]) visit(items[q])
            }
        }

        fun nearestSeed(x: Float, y: Float, z: Float): Int {
            var best = -1; var bestD = Float.POSITIVE_INFINITY
            forNear(x, y, z) { i ->
                val ex = seeds[3 * i] - x; val ey = seeds[3 * i + 1] - y; val ez = seeds[3 * i + 2] - z
                val d = ex * ex + ey * ey + ez * ez
                if (d < bestD) { bestD = d; best = i }
            }
            if (best >= 0) return best
            for (i in 0 until m) { // far from every seed, which a sample of the surface never is: brute force
                val ex = seeds[3 * i] - x; val ey = seeds[3 * i + 1] - y; val ez = seeds[3 * i + 2] - z
                val d = ex * ex + ey * ey + ez * ez
                if (d < bestD) { bestD = d; best = i }
            }
            return best
        }

        /**
         * The distance to the cell's boundaries joined by a smooth minimum of width [round], so near a corner, where two boundaries are about as close, it is smaller: the hole inside the
         * walls comes out with round corners.
         */
        fun wallDistance(x: Float, y: Float, z: Float, round: Float): Float {
            var first = -1; var d1 = Float.POSITIVE_INFINITY
            forNear(x, y, z) { i ->
                val ex = seeds[3 * i] - x; val ey = seeds[3 * i + 1] - y; val ez = seeds[3 * i + 2] - z
                val d = ex * ex + ey * ey + ez * ez
                if (d < d1) { d1 = d; first = i }
            }
            if (first < 0) return Float.POSITIVE_INFINITY
            val sx = seeds[3 * first]; val sy = seeds[3 * first + 1]; val sz = seeds[3 * first + 2]
            // The two nearest boundaries, joined by a polynomial smooth minimum: it only bites at a corner,
            // where the two are about as close, and rounds it off.
            var nearest = Float.POSITIVE_INFINITY; var second = Float.POSITIVE_INFINITY
            forNear(x, y, z) { j ->
                if (j != first) {
                    val ex = seeds[3 * j] - x; val ey = seeds[3 * j + 1] - y; val ez = seeds[3 * j + 2] - z
                    val gx = seeds[3 * j] - sx; val gy = seeds[3 * j + 1] - sy; val gz = seeds[3 * j + 2] - sz
                    val d = (ex * ex + ey * ey + ez * ez - d1) / (2f * sqrt(gx * gx + gy * gy + gz * gz))
                    if (d < nearest) { second = nearest; nearest = d } else if (d < second) second = d
                }
            }
            if (nearest.isInfinite()) return nearest
            val h = max(round - (second - nearest), 0f) / round
            return nearest - h * h * round * 0.25f
        }

    }

    /**
     * Surface nets: one vertex in every grid cell the surface crosses (the mean of its crossings of the
     * cell's edges), one quad across every grid edge it crosses. Streams the field a layer of nodes at a
     * time, so a grid of hundreds of cells a side never sits in memory at once.
     */
    internal object SurfaceNets {
        fun extract(
            field: (Float, Float, Float) -> Float,
            minX: Float, minY: Float, minZ: Float, maxX: Float, maxY: Float, maxZ: Float, cell: Float
        ): Mesh {
            val nx = ceil((maxX - minX) / cell).toInt() + 1
            val ny = ceil((maxY - minY) / cell).toInt() + 1
            val nz = ceil((maxZ - minZ) / cell).toInt() + 1
            var below = FloatArray(nx * ny)
            var above = FloatArray(nx * ny)
            var prevCells = IntArray(nx * ny) { -1 }
            var curCells = IntArray(nx * ny) { -1 }

            var vx = FloatArray(3 * 65536); var vn = 0
            var tri = IntArray(3 * 65536); var tn = 0
            fun addVertex(x: Float, y: Float, z: Float): Int {
                if (3 * vn + 3 > vx.size) vx = vx.copyOf(vx.size * 2)
                vx[3 * vn] = x; vx[3 * vn + 1] = y; vx[3 * vn + 2] = z
                return vn++
            }
            fun addTriangle(a: Int, b: Int, c: Int) {
                if (3 * tn + 3 > tri.size) tri = tri.copyOf(tri.size * 2)
                tri[3 * tn] = a; tri[3 * tn + 1] = b; tri[3 * tn + 2] = c
                tn++
            }

            fun layer(k: Int, into: FloatArray) {
                val z = minZ + k * cell
                IntStream.range(0, ny).parallel().forEach { j ->
                    val y = minY + j * cell
                    for (i in 0 until nx) into[j * nx + i] = field(minX + i * cell, y, z)
                }
            }
            layer(0, below)

            // A quad's four cells (in order round the axis), wound so its normal points out of the solid.
            fun quad(c0: Int, c1: Int, c2: Int, c3: Int, axis: Int, startInside: Boolean) {
                if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) return
                var a = c0; var b = c1; var c = c2; var d = c3
                // Normal of (a, b, c) against the axis, and the way the solid-to-empty direction runs.
                val ux = vx[3 * b] - vx[3 * a]; val uy = vx[3 * b + 1] - vx[3 * a + 1]; val uz = vx[3 * b + 2] - vx[3 * a + 2]
                val wx = vx[3 * c] - vx[3 * a]; val wy = vx[3 * c + 1] - vx[3 * a + 1]; val wz = vx[3 * c + 2] - vx[3 * a + 2]
                val nrm = when (axis) { 0 -> uy * wz - uz * wy; 1 -> uz * wx - ux * wz; else -> ux * wy - uy * wx }
                val outward = if (startInside) nrm > 0f else nrm < 0f
                if (!outward) { b = c3; d = c1 }
                addTriangle(a, b, c)
                addTriangle(a, c, d)
            }

            val corner = FloatArray(8)
            for (k in 0 until nz - 1) {
                layer(k + 1, above)
                java.util.Arrays.fill(curCells, -1)
                // The vertices of this layer's cells.
                for (j in 0 until ny - 1) for (i in 0 until nx - 1) {
                    corner[0] = below[j * nx + i]; corner[1] = below[j * nx + i + 1]
                    corner[2] = below[(j + 1) * nx + i]; corner[3] = below[(j + 1) * nx + i + 1]
                    corner[4] = above[j * nx + i]; corner[5] = above[j * nx + i + 1]
                    corner[6] = above[(j + 1) * nx + i]; corner[7] = above[(j + 1) * nx + i + 1]
                    var inside = 0
                    for (c in 0 until 8) if (corner[c] < 0f) inside++
                    if (inside == 0 || inside == 8) continue
                    var sx = 0f; var sy = 0f; var sz = 0f; var crossings = 0
                    for (e in EDGES) {
                        val a = corner[e[0]]; val b = corner[e[1]]
                        if ((a < 0f) == (b < 0f)) continue
                        val t = a / (a - b)
                        sx += (e[0] and 1) + t * ((e[1] and 1) - (e[0] and 1))
                        sy += ((e[0] shr 1) and 1) + t * (((e[1] shr 1) and 1) - ((e[0] shr 1) and 1))
                        sz += ((e[0] shr 2) and 1) + t * (((e[1] shr 2) and 1) - ((e[0] shr 2) and 1))
                        crossings++
                    }
                    curCells[j * nx + i] = addVertex(
                        minX + (i + sx / crossings) * cell, minY + (j + sy / crossings) * cell, minZ + (k + sz / crossings) * cell
                    )
                }
                // The quads across the edges that start at this layer's nodes.
                for (j in 1 until ny - 1) for (i in 1 until nx - 1) {
                    val here = below[j * nx + i]
                    // Along z: from this node up to the next layer.
                    val up = above[j * nx + i]
                    if ((here < 0f) != (up < 0f)) {
                        quad(curCells[(j - 1) * nx + i - 1], curCells[(j - 1) * nx + i], curCells[j * nx + i], curCells[j * nx + i - 1], 2, here < 0f)
                    }
                    if (k >= 1) {
                        // Along x: to the next node in this layer; the four cells lie in layers k - 1 and k.
                        val right = below[j * nx + i + 1]
                        if ((here < 0f) != (right < 0f)) {
                            quad(prevCells[(j - 1) * nx + i], prevCells[j * nx + i], curCells[j * nx + i], curCells[(j - 1) * nx + i], 0, here < 0f)
                        }
                        // Along y.
                        val forward = below[(j + 1) * nx + i]
                        if ((here < 0f) != (forward < 0f)) {
                            quad(prevCells[j * nx + i - 1], prevCells[j * nx + i], curCells[j * nx + i], curCells[j * nx + i - 1], 1, here < 0f)
                        }
                    }
                }
                val t = below; below = above; above = t
                val c = prevCells; prevCells = curCells; curCells = c
            }

            val out = FloatArray(9 * tn)
            for (t in 0 until tn) for (c in 0 until 3) {
                out[9 * t + 3 * c] = vx[3 * tri[3 * t + c]]
                out[9 * t + 3 * c + 1] = vx[3 * tri[3 * t + c] + 1]
                out[9 * t + 3 * c + 2] = vx[3 * tri[3 * t + c] + 2]
            }
            require(tn > 0) { "the sponge came out empty: the shell missed the model" }
            return Mesh(out, smooth = true)
        }

        /** A cube's twelve edges as pairs of corner numbers (bit 0 = x, bit 1 = y, bit 2 = z). */
        private val EDGES = arrayOf(
            intArrayOf(0, 1), intArrayOf(2, 3), intArrayOf(4, 5), intArrayOf(6, 7),
            intArrayOf(0, 2), intArrayOf(1, 3), intArrayOf(4, 6), intArrayOf(5, 7),
            intArrayOf(0, 4), intArrayOf(1, 5), intArrayOf(2, 6), intArrayOf(3, 7)
        )
    }
}
