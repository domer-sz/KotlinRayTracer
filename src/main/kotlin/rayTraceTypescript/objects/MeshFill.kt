package rayTraceTypescript.objects

import rayTraceTypescript.Color
import rayTraceTypescript.Point
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Material
import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Fills a model with small things and leaves the model itself out: only the filling is
 * rendered, so the shape is drawn by what is packed into it.
 *
 * - [Style.THREADS]: chains of beads (spheres) along curves that swirl over the shape, a
 *   curl-noise flow turned to run along the surface, each chain one thickness, its ends tapering;
 * - [Style.PYRAMIDS]: small square pyramids, sizes and turns at random, packed largest first,
 *   smaller ones into the gaps;
 * - [Style.DIAMONDS]: the same with diamonds of the classic round brilliant cut instead;
 * - [Style.SPONGE]: not a filling but a conversion - the model's surface turned into a soft, thick
 *   foam of round uneven holes ([LatticeFill]).
 *
 * The model only gives the volume: [SolidGrid] voxelizes it, works out what is inside (flooding
 * the outside from the grid's border, which forgives the small holes model files often have)
 * and how deep each cell lies under the surface. Everything is placed where it fits whole, no
 * deeper than [Settings.shell] (what lies deeper cannot be seen), and none overlaps another.
 * The same [Settings.seed] gives the same filling.
 */
object MeshFill {

    enum class Style { THREADS, PYRAMIDS, DIAMONDS, SPONGE }

    data class Settings(
        val style: Style,
        /** The largest radius, as a fraction of the model's height. */
        val size: Float = defaultSize(style),
        /** The smallest radius, as a fraction of the largest. */
        val smallest: Float = defaultSmallest(style),
        /** How many beads or pyramids at most. */
        val count: Int = defaultCount(style),
        /** How deep under the surface the filling goes, as a fraction of the height; null: 4 times [size]. */
        val shell: Float? = null,
        /** Grid cells along the model's longest side. */
        val resolution: Int = 256,
        val seed: Long = 1L,
        /** Threads: how many swirls fit across the model's height. */
        val swirl: Float = 3f
    ) {
        val shellFraction: Float get() = shell ?: (4f * size)

        companion object {
            /**
             * From -Drt.infill=threads|pyramids|diamonds|sponge and -Drt.infill.size|smallest|count|shell|res|seed|swirl;
             * null when -Drt.infill is unset or off.
             */
            fun fromProperties(): Settings? {
                val name = System.getProperty("rt.infill")?.trim()?.lowercase() ?: return null
                val style = when (name) {
                    "", "off", "none" -> return null
                    "threads", "thread", "nitki", "strands" -> Style.THREADS
                    "pyramids", "pyramid", "ostroslupy" -> Style.PYRAMIDS
                    "diamonds", "diamond", "diamenty", "brilliants" -> Style.DIAMONDS
                    "sponge", "gabka", "pores" -> Style.SPONGE
                    else -> throw IllegalArgumentException("-Drt.infill=$name: expected threads, pyramids, diamonds or sponge")
                }
                fun float(key: String) = System.getProperty("rt.infill.$key")?.toFloatOrNull()?.takeIf { it > 0f }
                return Settings(
                    style = style,
                    size = float("size") ?: defaultSize(style),
                    smallest = float("smallest")?.coerceAtMost(1f) ?: defaultSmallest(style),
                    count = System.getProperty("rt.infill.count")?.toIntOrNull()?.takeIf { it > 0 } ?: defaultCount(style),
                    shell = float("shell"),
                    resolution = System.getProperty("rt.infill.res")?.toIntOrNull()?.coerceIn(32, 1024) ?: 256,
                    seed = System.getProperty("rt.infill.seed")?.toLongOrNull() ?: 1L,
                    swirl = float("swirl") ?: 3f
                )
            }

            /**
             * The filling's own material instead of the model's: -Drt.infill.material=glass makes it
             * glass (index -Drt.infill.ior, 1.5, spreading colours by Abbe number -Drt.infill.abbe,
             * 64.2) and =diamond diamond (2.417, V_d 55.3), each tinted through its body by
             * -Drt.infill.color if given and its spread of colours scaled by -Drt.infill.dispersion
             * (1; 0 none); otherwise -Drt.infill.color=r,g,b makes it matte in that colour.
             */
            fun materialOverride(): Material? {
                val colour = System.getProperty("rt.infill.color")
                    ?.split(',')?.mapNotNull { it.trim().toFloatOrNull() }?.takeIf { it.size == 3 }
                    ?.let { Color(it[0], it[1], it[2]) }
                val dispersion = System.getProperty("rt.infill.dispersion")?.toFloatOrNull()?.takeIf { it >= 0f } ?: 1f
                return when (val kind = System.getProperty("rt.infill.material")?.trim()?.lowercase()) {
                    null, "", "matte" -> colour?.let { Lambertian(it) }
                    "glass", "szklo" -> Dielectric(
                        rayTraceTypescript.optics.GlassCatalog.model(
                            "glass", System.getProperty("rt.infill.ior")?.toFloatOrNull() ?: 1.5f,
                            System.getProperty("rt.infill.abbe")?.toFloatOrNull() ?: 64.17f, dispersion
                        ), colour
                    )
                    "diamond", "diament" -> Dielectric(rayTraceTypescript.optics.GlassCatalog.diamond(dispersion), colour)
                    else -> throw IllegalArgumentException("-Drt.infill.material=$kind: expected glass, diamond or matte")
                }
            }

            fun defaultSize(style: Style) = when (style) {
                Style.THREADS -> 0.012f
                Style.PYRAMIDS, Style.DIAMONDS -> 0.025f
                Style.SPONGE -> 0.035f
            }

            fun defaultSmallest(style: Style) = when (style) {
                Style.THREADS -> 0.25f
                Style.PYRAMIDS, Style.DIAMONDS -> 0.3f
                Style.SPONGE -> 1f
            }

            fun defaultCount(style: Style) = when (style) {
                Style.THREADS -> 400_000
                Style.PYRAMIDS, Style.DIAMONDS -> 150_000
                Style.SPONGE -> 1
            }
        }
    }

    /** The filling: [spheres] (the beads) for threads, a [mesh] for pyramids. */
    class Filling(val spheres: List<Sphere>?, val mesh: Mesh?, val count: Int, val summary: String)

    fun fill(mesh: Mesh, material: Material, settings: Settings): Filling {
        val started = System.nanoTime()
        if (settings.style == Style.SPONGE) {
            // A conversion of the surface, with no inside to fill: the sponge reads its own settings.
            val sponge = LatticeFill.build(mesh, LatticeFill.Settings.fromProperties())
            val seconds = (System.nanoTime() - started) / 1e9
            return Filling(null, sponge, sponge.triangleCount, "Turned the model into a sponge of %,d triangles in %.1f s".format(sponge.triangleCount, seconds))
        }
        val grid = SolidGrid(mesh, settings.resolution)
        val height = grid.height
        val rMax = settings.size * height
        val rMin = rMax * settings.smallest
        val shell = settings.shellFraction * height
        val random = Random(settings.seed)
        val cells = grid.cellsWithin(shell)
        require(cells.isNotEmpty()) {
            "the model has no inside at -Drt.infill.res=${settings.resolution} (thinner than a cell, or not closed); raise the resolution"
        }
        val filling = when (settings.style) {
            Style.SPONGE -> error("handled above")
            Style.THREADS -> threads(grid, cells, rMin, rMax, shell, settings, random, material)
            Style.PYRAMIDS -> {
                val packed = pack(grid, cells, rMin, rMax, settings.count, random)
                Filling(null, gems(packed, random, PYRAMID), packed.size, "${packed.size} pyramids")
            }
            Style.DIAMONDS -> {
                val packed = pack(grid, cells, rMin, rMax, settings.count, random)
                Filling(null, gems(packed, random, BRILLIANT), packed.size, "${packed.size} diamonds")
            }
        }
        val seconds = (System.nanoTime() - started) / 1e9
        val summary = "Filled the model with ${filling.summary} (radius %.4f-%.4f, shell %.3f, grid %dx%dx%d, %d cells inside) in %.1f s"
            .format(rMin, rMax, shell, grid.nx, grid.ny, grid.nz, grid.insideCount, seconds)
        return Filling(filling.spheres, filling.mesh, filling.count, summary)
    }

    /** Balls with no overlap, kept as flat arrays: x, y, z, r each. */
    internal class Balls(cellSize: Float) {
        private var data = FloatArray(4 * 1024)
        var size = 0
            private set
        private val hash = SpatialHash(cellSize)

        fun x(i: Int) = data[4 * i]
        fun y(i: Int) = data[4 * i + 1]
        fun z(i: Int) = data[4 * i + 2]
        fun r(i: Int) = data[4 * i + 3]

        fun add(x: Float, y: Float, z: Float, r: Float) {
            if (4 * size + 4 > data.size) data = data.copyOf(data.size * 2)
            data[4 * size] = x; data[4 * size + 1] = y; data[4 * size + 2] = z; data[4 * size + 3] = r
            hash.add(x, y, z, size)
            size++
        }

        /** Whether a ball at (x, y, z) of radius [r] keeps [gap] times the radii clear of every one placed. */
        fun clear(x: Float, y: Float, z: Float, r: Float, gap: Float = 1f): Boolean = hash.none(x, y, z) { i ->
            val dx = data[4 * i] - x; val dy = data[4 * i + 1] - y; val dz = data[4 * i + 2] - z
            val reach = (data[4 * i + 3] + r) * gap
            dx * dx + dy * dy + dz * dz < reach * reach
        }

        fun sphere(i: Int, material: Material) = Sphere(Point(x(i), y(i), z(i)), r(i), material)
    }

    /**
     * Throws balls at random points of [cells], largest first: 7 sizes from [rMax] down to
     * [rMin], each spread 15 % either way, so each smaller size goes into the gaps the larger ones
     * left. A ball is kept where it lies wholly inside and touches no other. Through the whole
     * shell, so layers of them behind each other leave no gaps - tried near the surface only, in a
     * wider spread of sizes (larger ones standing out of it), pyramids showed holes, and the user
     * preferred this (2026-11-19).
     */
    internal fun pack(grid: SolidGrid, cells: IntArray, rMin: Float, rMax: Float, count: Int, random: Random): Balls {
        val spread = 0.15f
        val balls = Balls(2f * rMax * (1f + spread))
        val levels = if (rMin >= rMax * 0.99f) 1 else 7
        val volume = cells.size.toDouble() * grid.cell * grid.cell * grid.cell
        for (level in 0 until levels) {
            val t = if (levels == 1) 0f else level / (levels - 1f)
            val r = rMax * exp(t * ln(rMin / rMax))
            // At least 50 000 throws a size: a few large ones need many to find where they fit.
            val attempts = min(4_000_000.0, max(50_000.0, 3.0 * volume / (4.0 / 3.0 * PI * r * r * r))).toInt()
            var misses = 0
            for (attempt in 0 until attempts) {
                if (balls.size >= count) return balls
                val p = grid.randomPoint(cells, random)
                val radius = r * (1f - spread + 2f * spread * random.nextFloat())
                if (grid.clearance(p[0], p[1], p[2]) >= radius && balls.clear(p[0], p[1], p[2], radius)) {
                    balls.add(p[0], p[1], p[2], radius)
                    misses = 0
                } else if (++misses > 200_000) break
            }
        }
        return balls
    }

    /** A solid of unit circumradius about the origin: its corners, and its faces as corner triples wound outwards. */
    internal class Gem(val corners: Array<FloatArray>, val faces: IntArray)

    /** A square pyramid: four base corners and the apex. */
    internal val PYRAMID = run {
        val a = 0.62f
        val b = 0.45f
        Gem(
            arrayOf(
                floatArrayOf(-a, -b, -a), floatArrayOf(a, -b, -a), floatArrayOf(a, -b, a), floatArrayOf(-a, -b, a),
                floatArrayOf(0f, 1f, 0f)
            ),
            intArrayOf(0, 4, 1, 1, 4, 2, 2, 4, 3, 3, 4, 0, 0, 1, 2, 0, 2, 3)
        )
    }

    /**
     * A round brilliant in Tolkowsky's proportions: the girdle of unit radius, the table 53 % of its
     * width, the crown 16.2 % of it high (34.5 degrees), the pavilion 43.1 % deep (40.75 degrees)
     * down to a point. Its facets in eight-fold symmetry: the octagonal table, crown facets from
     * each table corner and side down to the girdle, and the pavilion's sixteen lower girdle facets
     * and eight mains meeting at the culet - 64 triangles, a simplification of the real 57 facets
     * that catches light the same way.
     */
    internal val BRILLIANT = run {
        val girdle = 16
        val tableRadius = 0.53f
        val crown = 0.324f
        val pavilion = 0.862f
        val corners = ArrayList<FloatArray>()
        fun at(radius: Float, angle: Double, y: Float) =
            floatArrayOf((radius * kotlin.math.cos(angle)).toFloat(), y, (radius * kotlin.math.sin(angle)).toFloat())
        // 0..15 the girdle, 16..23 the table's corners, 24..31 the pavilion mains' upper ends, 32 the culet.
        for (k in 0 until girdle) corners += at(1f, 2 * PI * k / girdle, 0f)
        for (k in 0 until 8) corners += at(tableRadius, 2 * PI * (2 * k + 1) / girdle, crown)
        // Down each pavilion main, 77 % of the way to the culet, where the lower girdle facets end.
        for (k in 0 until 8) corners += at(0.23f, 2 * PI * (2 * k) / girdle, -0.77f * pavilion)
        corners += floatArrayOf(0f, -pavilion, 0f)
        val g = { k: Int -> (k + girdle) % girdle }
        val t = { k: Int -> 16 + (k + 8) % 8 }
        val m = { k: Int -> 24 + (k + 8) % 8 }
        val faces = ArrayList<Int>()
        fun face(a: Int, b: Int, c: Int) { faces += a; faces += b; faces += c }
        for (k in 0 until 8) {
            // Crown: under table corner k (over girdle point 2k+1), and between corners k and k+1.
            face(t(k), g(2 * k), g(2 * k + 1)); face(t(k), g(2 * k + 1), g(2 * k + 2))
            face(t(k), g(2 * k + 2), t(k + 1))
            // The table, a fan from corner 0.
            if (k in 1..6) face(t(0), t(k), t(k + 1))
            // Pavilion: lower girdle facets, then the mains to the culet.
            face(g(2 * k), g(2 * k + 1), m(k)); face(g(2 * k + 1), g(2 * k + 2), m(k + 1))
            face(m(k), g(2 * k + 1), m(k + 1)); face(m(k), m(k + 1), 32)
        }
        // Wind every face outwards: the solid is convex and the origin lies inside it.
        val c = corners.toTypedArray()
        for (f in faces.indices step 3) {
            val a = c[faces[f]]; val b = c[faces[f + 1]]; val d = c[faces[f + 2]]
            val ux = b[0] - a[0]; val uy = b[1] - a[1]; val uz = b[2] - a[2]
            val vx = d[0] - a[0]; val vy = d[1] - a[1]; val vz = d[2] - a[2]
            val nx = uy * vz - uz * vy; val ny = uz * vx - ux * vz; val nz = ux * vy - uy * vx
            if (nx * (a[0] + b[0] + d[0]) + ny * (a[1] + b[1] + d[1]) + nz * (a[2] + b[2] + d[2]) < 0f) {
                val swap = faces[f + 1]; faces[f + 1] = faces[f + 2]; faces[f + 2] = swap
            }
        }
        Gem(c, faces.toIntArray())
    }

    /** [gem] in each ball, as big as the ball, turned at random. */
    private fun gems(balls: Balls, random: Random, gem: Gem): Mesh {
        val corners = gem.corners
        val faces = gem.faces
        val vertices = FloatArray(balls.size * faces.size * 3)
        val turned = Array(corners.size) { FloatArray(3) }
        var o = 0
        for (i in 0 until balls.size) {
            val m = randomRotation(random)
            val r = balls.r(i)
            for (c in corners.indices) {
                val p = corners[c]
                for (k in 0 until 3) turned[c][k] = (m[3 * k] * p[0] + m[3 * k + 1] * p[1] + m[3 * k + 2] * p[2]) * r
            }
            val cx = balls.x(i); val cy = balls.y(i); val cz = balls.z(i)
            for (f in faces) {
                vertices[o++] = cx + turned[f][0]; vertices[o++] = cy + turned[f][1]; vertices[o++] = cz + turned[f][2]
            }
        }
        return Mesh(vertices)
    }

    /** A uniformly random rotation, as a row-major 3x3 matrix (Shoemake's quaternion). */
    private fun randomRotation(random: Random): FloatArray {
        val u1 = random.nextFloat(); val u2 = random.nextFloat() * 2f * PI.toFloat(); val u3 = random.nextFloat() * 2f * PI.toFloat()
        val s1 = sqrt(1f - u1); val s2 = sqrt(u1)
        val x = s1 * sin(u2); val y = s1 * cos(u2); val z = s2 * sin(u3); val w = s2 * cos(u3)
        return floatArrayOf(
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
        )
    }

    /**
     * Chains of beads along the streamlines of a curl-noise flow (divergence free, so it swirls
     * and never piles up), with the part across the surface taken out and a pull back to the
     * depth the chain started at - so a chain runs along the shape at one depth, in layers, as
     * threads wound round it. A chain stops where it would touch another or leave the shape;
     * its beads are one radius, 1.5 radii apart, smaller towards its ends. Thick chains are laid
     * first, then ever thinner ones into the room left (sizes 1.35 times apart, from rMax to rMin),
     * so the thick ones run long instead of being cut short by thin ones.
     */
    private fun threads(
        grid: SolidGrid, cells: IntArray, rMin: Float, rMax: Float, shell: Float,
        settings: Settings, random: Random, material: Material
    ): Filling {
        val beads = Balls(2f * rMax * 1.15f)
        val flow = CurlFlow(settings.seed, settings.swirl / grid.height)
        val maxBeads = 600
        val minBeads = 5
        val path = FloatArray(4 * (2 * maxBeads + 1))
        var chains = 0
        var misses = 0
        val direction = FloatArray(3)
        val mid = FloatArray(3)

        // The chain is laid out in path as: slot 0 the seed, slots 1..forward the beads one way,
        // the rest the beads the other way. A bead's place along the chain: 0, +1, +2... one
        // way, -1, -2... the other.
        fun place(slot: Int, forward: Int): Int = when {
            slot == 0 -> 0
            slot <= forward -> slot
            else -> -(slot - forward)
        }

        // Follows the flow from (x, y, z) one way ([sign]), appending beads to path; returns how
        // many it laid. The beads of this chain are checked by hand (they are not in the hash
        // yet), except those within three places of the new one, which it touches by design.
        fun trace(x0: Float, y0: Float, z0: Float, r: Float, depth: Float, sign: Float, forwardDone: Int): Int {
            val step = 1.5f * r
            var x = x0; var y = y0; var z = z0
            var laid = 0
            val start = 1 + forwardDone
            while (laid < maxBeads) {
                if (!flow.along(grid, x, y, z, depth, step, direction)) break
                mid[0] = x + sign * direction[0] * step * 0.5f
                mid[1] = y + sign * direction[1] * step * 0.5f
                mid[2] = z + sign * direction[2] * step * 0.5f
                if (!flow.along(grid, mid[0], mid[1], mid[2], depth, step, direction)) break
                val nx = x + sign * direction[0] * step
                val ny = y + sign * direction[1] * step
                val nz = z + sign * direction[2] * step
                if (grid.clearance(nx, ny, nz) < r * 0.6f) break
                if (!beads.clear(nx, ny, nz, r, 0.95f)) break
                // This chain's own earlier beads, both ways, but not its near neighbours.
                var crosses = false
                val forward = if (sign > 0f) laid else forwardDone
                val here = if (sign > 0f) laid + 1 else -(laid + 1)
                for (q in 0 until start + laid) {
                    if (kotlin.math.abs(place(q, forward) - here) <= 3) continue
                    val dx = path[4 * q] - nx; val dy = path[4 * q + 1] - ny; val dz = path[4 * q + 2] - nz
                    if (dx * dx + dy * dy + dz * dz < 4f * r * r * 0.9f) { crosses = true; break }
                }
                if (crosses) break
                val o = 4 * (start + laid)
                path[o] = nx; path[o + 1] = ny; path[o + 2] = nz; path[o + 3] = r
                x = nx; y = ny; z = nz
                laid++
            }
            return laid
        }

        val chainOrder = IntArray(2 * maxBeads + 1)
        val levels = if (rMin >= rMax * 0.99f) 1 else ceil(ln(rMax / rMin) / ln(1.35f)).toInt() + 1
        var level = 0
        var band = cells
        var bandLevel = -1
        while (beads.size < settings.count && level < levels) {
            // A size is done when seeds keep failing; then the next, thinner one.
            if (misses > 4_000) { level++; misses = 0; continue }
            val t = if (levels == 1) 0f else level / (levels - 1f)
            val base = rMax * exp(t * ln(rMin / rMax))
            // Seeded touching the surface (1 to 1.3 of its radii under it): so a thick
            // chain runs on top and the thin ones laid later fill the gaps between,
            // instead of covering it.
            if (bandLevel != level) {
                band = cells.filter { grid.depthOf(it) <= 1.6f * base + grid.cell }.toIntArray()
                bandLevel = level
                if (band.isEmpty()) { level++; continue }
            }
            val r = base * (0.85f + 0.3f * random.nextFloat())
            val p = grid.randomPoint(band, random)
            val room = grid.clearance(p[0], p[1], p[2])
            if (room < r || room > 1.3f * r || !beads.clear(p[0], p[1], p[2], r, 1.2f)) { misses++; continue }
            val depth = grid.depthAt(p[0], p[1], p[2])
            // The seed goes in slot 0, one way from slot 1, the other way after it.
            path[0] = p[0]; path[1] = p[1]; path[2] = p[2]; path[3] = r
            val forward = trace(p[0], p[1], p[2], r, depth, 1f, 0)
            val backward = trace(p[0], p[1], p[2], r, depth, -1f, forward)
            val n = 1 + forward + backward
            if (n < minBeads) { misses++; continue }
            misses = 0
            // In order along the chain: the backward beads reversed, the seed, the forward beads.
            var k = 0
            for (j in backward - 1 downTo 0) chainOrder[k++] = 1 + forward + j
            chainOrder[k++] = 0
            for (j in 0 until forward) chainOrder[k++] = 1 + j
            val taper = min(6, n / 3).coerceAtLeast(1)
            for (i in 0 until n) {
                val q = chainOrder[i]
                val fromEnd = min(i, n - 1 - i)
                val scale = if (fromEnd >= taper) 1f else 0.35f + 0.65f * fromEnd / taper
                beads.add(path[4 * q], path[4 * q + 1], path[4 * q + 2], path[4 * q + 3] * scale)
            }
            chains++
        }
        val spheres = (0 until beads.size).map { beads.sphere(it, material) }
        return Filling(spheres, null, spheres.size, "${spheres.size} beads in $chains threads")
    }

    /**
     * The curl of a three-component gradient noise, the swirling field threads follow, made to
     * run along the surface and to hold a depth.
     */
    internal class CurlFlow(seed: Long, private val frequency: Float) {
        private val a = GradientNoise(seed * 3 + 1)
        private val b = GradientNoise(seed * 3 + 2)
        private val c = GradientNoise(seed * 3 + 3)

        /** Writes into [out] the unit direction to step along at (x, y, z); false where there is none. */
        fun along(grid: SolidGrid, x: Float, y: Float, z: Float, depth: Float, step: Float, out: FloatArray): Boolean {
            val px = x * frequency; val py = y * frequency; val pz = z * frequency
            val e = 0.01f
            fun d(n: GradientNoise, ax: Int): Float = when (ax) {
                0 -> n.at(px + e, py, pz) - n.at(px - e, py, pz)
                1 -> n.at(px, py + e, pz) - n.at(px, py - e, pz)
                else -> n.at(px, py, pz + e) - n.at(px, py, pz - e)
            }
            var vx = d(c, 1) - d(b, 2)
            var vy = d(a, 2) - d(c, 0)
            var vz = d(b, 0) - d(a, 1)
            // The surface's normal at this depth: the way the depth grows.
            val g = grid.depthGradient(x, y, z)
            val gl = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
            if (gl > 1e-6f) {
                val nx = g[0] / gl; val ny = g[1] / gl; val nz = g[2] / gl
                val across = vx * nx + vy * ny + vz * nz
                vx -= across * nx; vy -= across * ny; vz -= across * nz
                val l = sqrt(vx * vx + vy * vy + vz * vz)
                if (l < 1e-9f) return false
                vx /= l; vy /= l; vz /= l
                // Back towards the starting depth, at most half a step's worth.
                val pull = ((depth - grid.depthAt(x, y, z)) / step).coerceIn(-0.5f, 0.5f)
                vx += pull * nx; vy += pull * ny; vz += pull * nz
            }
            val l = sqrt(vx * vx + vy * vy + vz * vz)
            if (l < 1e-9f) return false
            out[0] = vx / l; out[1] = vy / l; out[2] = vz / l
            return true
        }
    }

    /** Ken Perlin's improved noise, with its own seeded permutation. */
    internal class GradientNoise(seed: Long) {
        private val p = IntArray(512)

        init {
            val perm = IntArray(256) { it }
            val random = Random(seed)
            for (i in 255 downTo 1) {
                val j = random.nextInt(i + 1)
                val t = perm[i]; perm[i] = perm[j]; perm[j] = t
            }
            for (i in 0 until 512) p[i] = perm[i and 255]
        }

        fun at(x: Float, y: Float, z: Float): Float {
            val fx = floor(x); val fy = floor(y); val fz = floor(z)
            val xi = fx.toInt() and 255; val yi = fy.toInt() and 255; val zi = fz.toInt() and 255
            val xf = x - fx; val yf = y - fy; val zf = z - fz
            val u = fade(xf); val v = fade(yf); val w = fade(zf)
            val aa = p[p[xi] + yi]; val ab = p[p[xi] + yi + 1]
            val ba = p[p[xi + 1] + yi]; val bb = p[p[xi + 1] + yi + 1]
            return lerp(w,
                lerp(v,
                    lerp(u, grad(p[aa + zi], xf, yf, zf), grad(p[ba + zi], xf - 1, yf, zf)),
                    lerp(u, grad(p[ab + zi], xf, yf - 1, zf), grad(p[bb + zi], xf - 1, yf - 1, zf))),
                lerp(v,
                    lerp(u, grad(p[aa + zi + 1], xf, yf, zf - 1), grad(p[ba + zi + 1], xf - 1, yf, zf - 1)),
                    lerp(u, grad(p[ab + zi + 1], xf, yf - 1, zf - 1), grad(p[bb + zi + 1], xf - 1, yf - 1, zf - 1))))
        }

        private fun fade(t: Float) = t * t * t * (t * (t * 6 - 15) + 10)
        private fun lerp(t: Float, a: Float, b: Float) = a + t * (b - a)
        private fun grad(hash: Int, x: Float, y: Float, z: Float): Float {
            val h = hash and 15
            val u = if (h < 8) x else y
            val v = if (h < 4) y else if (h == 12 || h == 14) x else z
            return (if (h and 1 == 0) u else -u) + (if (h and 2 == 0) v else -v)
        }
    }

    /** Ball indices bucketed by a cubic grid of [cellSize], for overlap tests. */
    internal class SpatialHash(private val cellSize: Float) {
        private val buckets = HashMap<Long, IntArrayList>()

        private fun key(ix: Int, iy: Int, iz: Int): Long =
            ((ix.toLong() and 0x1FFFFF) shl 42) or ((iy.toLong() and 0x1FFFFF) shl 21) or (iz.toLong() and 0x1FFFFF)

        fun add(x: Float, y: Float, z: Float, index: Int) {
            buckets.getOrPut(key(cell(x), cell(y), cell(z))) { IntArrayList() }.add(index)
        }

        /** True when [hits] is false for every index in the cells round (x, y, z). */
        inline fun none(x: Float, y: Float, z: Float, hits: (Int) -> Boolean): Boolean {
            val cx = cell(x); val cy = cell(y); val cz = cell(z)
            for (ix in cx - 1..cx + 1) for (iy in cy - 1..cy + 1) for (iz in cz - 1..cz + 1) {
                val bucket = bucket(ix, iy, iz) ?: continue
                for (k in 0 until bucket.size) if (hits(bucket[k])) return false
            }
            return true
        }

        fun bucket(ix: Int, iy: Int, iz: Int): IntArrayList? = buckets[key(ix, iy, iz)]
        fun cell(v: Float): Int = floor(v / cellSize).toInt()
    }

    internal class IntArrayList {
        private var data = IntArray(4)
        var size = 0
            private set

        fun add(v: Int) {
            if (size == data.size) data = data.copyOf(size * 2)
            data[size++] = v
        }

        operator fun get(i: Int) = data[i]
    }
}

/**
 * A model voxelized: which cells of a grid round it lie inside, and how deep under the surface.
 *
 * Every triangle marks the cells it passes through (sampled a third of a cell apart). The outside
 * is then flooded from the grid's border through unmarked cells; inside is what the flood did not
 * reach. A model with a hole bigger than a cell floods through it and comes out empty, so the
 * marks are then thickened by a cell (up to twice) to close the hole, and the outside grown back
 * by as much, so the inside does not grow a cell into the air.
 */
internal class SolidGrid(mesh: Mesh, resolution: Int) {
    val cell: Float
    val nx: Int
    val ny: Int
    val nz: Int
    private val ox: Float
    private val oy: Float
    private val oz: Float
    val height: Float
    private val inside: BooleanArray
    /** The cells the flood from the border reached. */
    private val outsideCells: BooleanArray
    /** Distance from each cell's centre to the nearest cell centre that is not inside, in model units. */
    private val depth: FloatArray
    val insideCount: Int

    init {
        val v = mesh.vertices
        var x0 = Float.POSITIVE_INFINITY; var y0 = Float.POSITIVE_INFINITY; var z0 = Float.POSITIVE_INFINITY
        var x1 = Float.NEGATIVE_INFINITY; var y1 = Float.NEGATIVE_INFINITY; var z1 = Float.NEGATIVE_INFINITY
        for (i in v.indices step 3) {
            x0 = min(x0, v[i]); x1 = max(x1, v[i])
            y0 = min(y0, v[i + 1]); y1 = max(y1, v[i + 1])
            z0 = min(z0, v[i + 2]); z1 = max(z1, v[i + 2])
        }
        height = y1 - y0
        cell = max(x1 - x0, max(y1 - y0, z1 - z0)) / resolution
        val pad = 3
        ox = x0 - pad * cell; oy = y0 - pad * cell; oz = z0 - pad * cell
        nx = ceil((x1 - x0) / cell).toInt() + 2 * pad
        ny = ceil((y1 - y0) / cell).toInt() + 2 * pad
        nz = ceil((z1 - z0) / cell).toInt() + 2 * pad

        val surface = BooleanArray(nx * ny * nz)
        for (t in 0 until mesh.triangleCount) {
            val o = 9 * t
            val ex = longest(v, o)
            val n = ceil(ex / (cell / 3f)).toInt().coerceAtLeast(1)
            for (i in 0..n) for (j in 0..n - i) {
                val s = i.toFloat() / n; val w = j.toFloat() / n
                val px = v[o] + (v[o + 3] - v[o]) * s + (v[o + 6] - v[o]) * w
                val py = v[o + 1] + (v[o + 4] - v[o + 1]) * s + (v[o + 7] - v[o + 1]) * w
                val pz = v[o + 2] + (v[o + 5] - v[o + 2]) * s + (v[o + 8] - v[o + 2]) * w
                surface[index(((px - ox) / cell).toInt(), ((py - oy) / cell).toInt(), ((pz - oz) / cell).toInt())] = true
            }
        }
        var walls = surface
        var outside = flood(walls)
        var closed = 0
        while (count(walls, outside) == 0 && closed < 2) {
            walls = dilate(walls)
            closed++
            outside = growBack(flood(walls), surface, closed)
        }
        inside = BooleanArray(nx * ny * nz) { !surface[it] && !outside[it] }
        outsideCells = outside
        insideCount = inside.count { it }
        depth = distance(inside)
    }

    private fun longest(v: FloatArray, o: Int): Float {
        fun len(a: Int, b: Int): Float {
            val dx = v[o + b] - v[o + a]; val dy = v[o + b + 1] - v[o + a + 1]; val dz = v[o + b + 2] - v[o + a + 2]
            return sqrt(dx * dx + dy * dy + dz * dz)
        }
        return max(len(0, 3), max(len(0, 6), len(3, 6)))
    }

    private fun index(x: Int, y: Int, z: Int) = (z * ny + y) * nx + x

    private fun count(walls: BooleanArray, outside: BooleanArray): Int {
        var n = 0
        for (i in walls.indices) if (!walls[i] && !outside[i]) n++
        return n
    }

    /** The cells reachable from the grid's corner through cells that are not [walls], six-connected. */
    private fun flood(walls: BooleanArray): BooleanArray {
        val reached = BooleanArray(walls.size)
        val queue = IntArray(walls.size)
        var head = 0; var tail = 0
        queue[tail++] = 0; reached[0] = true
        while (head < tail) {
            val i = queue[head++]
            val x = i % nx; val y = (i / nx) % ny; val z = i / (nx * ny)
            fun visit(j: Int) { if (!reached[j] && !walls[j]) { reached[j] = true; queue[tail++] = j } }
            if (x > 0) visit(i - 1); if (x < nx - 1) visit(i + 1)
            if (y > 0) visit(i - nx); if (y < ny - 1) visit(i + nx)
            if (z > 0) visit(i - nx * ny); if (z < nz - 1) visit(i + nx * ny)
        }
        return reached
    }

    private fun neighbours(i: Int, visit: (Int) -> Unit) {
        val x = i % nx; val y = (i / nx) % ny; val z = i / (nx * ny)
        if (x > 0) visit(i - 1); if (x < nx - 1) visit(i + 1)
        if (y > 0) visit(i - nx); if (y < ny - 1) visit(i + nx)
        if (z > 0) visit(i - nx * ny); if (z < nz - 1) visit(i + nx * ny)
    }

    private fun dilate(walls: BooleanArray): BooleanArray {
        val out = walls.copyOf()
        for (i in walls.indices) if (walls[i]) neighbours(i) { out[it] = true }
        return out
    }

    /** [outside] grown [steps] cells into what is not [surface]: back up to the model's own walls. */
    private fun growBack(outside: BooleanArray, surface: BooleanArray, steps: Int): BooleanArray {
        var current = outside
        repeat(steps) {
            val next = current.copyOf()
            for (i in current.indices) if (current[i]) neighbours(i) { if (!surface[it]) next[it] = true }
            current = next
        }
        return current
    }

    /** Exact Euclidean distance from each inside cell to the nearest other cell (Felzenszwalb's separable transform). */
    private fun distance(inside: BooleanArray): FloatArray {
        val far = 1e20
        val d = DoubleArray(inside.size) { if (inside[it]) far else 0.0 }
        val longest = max(nx, max(ny, nz))
        val f = DoubleArray(longest); val out = DoubleArray(longest)
        val vv = IntArray(longest); val zz = DoubleArray(longest + 1)
        fun pass(count: Int, lines: Int, at: (Int, Int) -> Int) {
            for (line in 0 until lines) {
                for (k in 0 until count) f[k] = d[at(line, k)]
                edt(f, count, out, vv, zz)
                for (k in 0 until count) d[at(line, k)] = out[k]
            }
        }
        pass(nx, ny * nz) { line, k -> line * nx + k }
        pass(ny, nx * nz) { line, k -> val x = line % nx; val z = line / nx; (z * ny + k) * nx + x }
        pass(nz, nx * ny) { line, k -> val x = line % nx; val y = line / nx; (k * ny + y) * nx + x }
        return FloatArray(d.size) { (sqrt(d[it]) * cell).toFloat() }
    }

    /** One line of the squared distance transform: the lower envelope of parabolas rooted at f. */
    private fun edt(f: DoubleArray, n: Int, d: DoubleArray, v: IntArray, z: DoubleArray) {
        var k = 0
        v[0] = 0; z[0] = Double.NEGATIVE_INFINITY; z[1] = Double.POSITIVE_INFINITY
        for (q in 1 until n) {
            var s = ((f[q] + q.toDouble() * q) - (f[v[k]] + v[k].toDouble() * v[k])) / (2.0 * q - 2.0 * v[k])
            while (s <= z[k]) {
                k--
                s = ((f[q] + q.toDouble() * q) - (f[v[k]] + v[k].toDouble() * v[k])) / (2.0 * q - 2.0 * v[k])
            }
            k++
            v[k] = q; z[k] = s; z[k + 1] = Double.POSITIVE_INFINITY
        }
        k = 0
        for (q in 0 until n) {
            while (z[k + 1] < q) k++
            val dq = (q - v[k]).toDouble()
            d[q] = dq * dq + f[v[k]]
        }
    }

    /** The inside cells no deeper than [shell]. */
    fun cellsWithin(shell: Float): IntArray {
        var n = 0
        for (i in inside.indices) if (inside[i] && depth[i] <= shell + cell) n++
        val out = IntArray(n)
        n = 0
        for (i in inside.indices) if (inside[i] && depth[i] <= shell + cell) out[n++] = i
        return out
    }

    /** Where (x, y, z) lies: [INSIDE], [OUTSIDE] or in a cell the surface crosses ([WALL]); outside the grid is outside. */
    fun kindAt(x: Float, y: Float, z: Float): Int {
        val ix = floor((x - ox) / cell).toInt(); val iy = floor((y - oy) / cell).toInt(); val iz = floor((z - oz) / cell).toInt()
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) return OUTSIDE
        val i = index(ix, iy, iz)
        return if (inside[i]) INSIDE else if (outsideCells[i]) OUTSIDE else WALL
    }

    companion object {
        const val OUTSIDE = 0
        const val WALL = 1
        const val INSIDE = 2
    }

    /** How deep under the surface the centre of cell [index] lies. */
    fun depthOf(index: Int): Float = depth[index]

    /** A random point in one of [cells]. */
    fun randomPoint(cells: IntArray, random: Random): FloatArray {
        val i = cells[random.nextInt(cells.size)]
        val x = i % nx; val y = (i / nx) % ny; val z = i / (nx * ny)
        return floatArrayOf(
            ox + (x + random.nextFloat()) * cell,
            oy + (y + random.nextFloat()) * cell,
            oz + (z + random.nextFloat()) * cell
        )
    }

    /** The depth under the surface at a point, interpolated between cell centres; 0 outside. */
    fun depthAt(x: Float, y: Float, z: Float): Float {
        val gx = (x - ox) / cell - 0.5f; val gy = (y - oy) / cell - 0.5f; val gz = (z - oz) / cell - 0.5f
        val ix = floor(gx).toInt(); val iy = floor(gy).toInt(); val iz = floor(gz).toInt()
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx - 1 || iy >= ny - 1 || iz >= nz - 1) return 0f
        val fx = gx - ix; val fy = gy - iy; val fz = gz - iz
        fun at(dx: Int, dy: Int, dz: Int) = depth[index(ix + dx, iy + dy, iz + dz)]
        val c00 = at(0, 0, 0) + (at(1, 0, 0) - at(0, 0, 0)) * fx
        val c10 = at(0, 1, 0) + (at(1, 1, 0) - at(0, 1, 0)) * fx
        val c01 = at(0, 0, 1) + (at(1, 0, 1) - at(0, 0, 1)) * fx
        val c11 = at(0, 1, 1) + (at(1, 1, 1) - at(0, 1, 1)) * fx
        val c0 = c00 + (c10 - c00) * fy
        val c1 = c01 + (c11 - c01) * fy
        return c0 + (c1 - c0) * fz
    }

    /**
     * How far the surface is from a point, as best the grid tells: the depth less half a cell,
     * since the cells the surface passes through count as not inside.
     */
    fun clearance(x: Float, y: Float, z: Float): Float = depthAt(x, y, z) - 0.5f * cell

    /** The way the depth grows fastest at a point (the inward normal, unnormalised). */
    fun depthGradient(x: Float, y: Float, z: Float): FloatArray {
        val h = cell
        return floatArrayOf(
            depthAt(x + h, y, z) - depthAt(x - h, y, z),
            depthAt(x, y + h, z) - depthAt(x, y - h, z),
            depthAt(x, y, z + h) - depthAt(x, y, z - h)
        )
    }
}
