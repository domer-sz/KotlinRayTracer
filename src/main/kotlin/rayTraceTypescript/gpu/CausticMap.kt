package rayTraceTypescript.gpu

import org.lwjgl.cuda.CU.cuCtxSynchronize
import org.lwjgl.cuda.CU.cuLaunchKernel
import org.lwjgl.cuda.CU.cuMemsetD32
import org.lwjgl.cuda.CU.cuModuleGetFunction
import org.lwjgl.cuda.CU.cuModuleUnload
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil
import rayTraceTypescript.render.CameraSetup
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Caustics by photon mapping (-Drt.caustics=on), shared by both engines: before each launch of
 * the image, a photon pass (the kernel's AOV_MODE 5) shoots photons from the lights through the
 * casters - glass, and metal smoother than the kernel's CAUSTIC_MAX_FUZZ - to where they land on
 * Lambertian surfaces; this class sorts them into a hashed grid (kernels/caustics_grid.cu) and
 * hands the image launch the map, which every camera path gathers from at the first surface it
 * reaches that is not a caster. The path tracer gives up the light it would have found there
 * through casters (raytracer.cu, `causticState`), so each path of light is counted once.
 *
 * Every launch has a map of its own; the gather disc starts at [START_PIXELS] pixels wide at the
 * casters' distance (or -Drt.caustics.radius, in scene units) and shrinks as launches add up, as
 * k^(-1/6) - progressive photon mapping's rate - down to a third of that, so the blur goes while
 * the noise averages out. A launch shoots -Drt.caustics.photons photons per sample of it (a
 * quarter of the pixels by default), at most [MAX_PHOTONS]; [samplesPerLaunch] keeps launches
 * small enough for that, so the maps are many.
 *
 * Only the lights of the scene's light list shoot photons; light from the sky and from glowing
 * surfaces that are not on the list keeps reaching through glass by path tracing alone.
 */
internal class CausticMap(private val cuda: CudaContext) : AutoCloseable {
    private val module: Long
    private val countFn: Long
    private val scanBlocksFn: Long
    private val scanSumsFn: Long
    private val addOffsetsFn: Long
    private val scatterFn: Long
    private val memory = mutableListOf<Long>()

    private var capacity = 0
    private var tableSize = 0
    private var photons = 0L
    private var sorted = 0L
    private var keys = 0L
    private var counts = 0L
    private var starts = 0L
    private var cursor = 0L
    private var blockSums = 0L
    private var lightCdf = 0L
    val params: Long = DeviceMemory.allocate(PARAMS_BYTES.toLong()).also { memory += it }
    private val paramsHost: ByteBuffer = MemoryUtil.memCalloc(PARAMS_BYTES).order(ByteOrder.nativeOrder())

    private var aims = FloatArray(0)
    private var launchIndex = 0
    /** Photons shot and landed since the image's first launch, for the log. */
    private var shot = 0L
    private var landed = 0L

    init {
        val built = cuda.buildKernel(gridSource(), listOf("--use_fast_math", "--std=c++17"), "caustic_count")
        module = built.first
        countFn = built.second
        fun function(name: String): Long = MemoryStack.stackPush().use { stack ->
            val f = stack.mallocPointer(1)
            CudaContext.check(cuModuleGetFunction(f, module, name), "cuModuleGetFunction($name)")
            f.get(0)
        }
        scanBlocksFn = function("caustic_scan_blocks")
        scanSumsFn = function("caustic_scan_sums")
        addOffsetsFn = function("caustic_add_offsets")
        scatterFn = function("caustic_scatter")
    }

    /** Whether [scene] has anything to make caustics with: a caster and a light. */
    var active = false
        private set

    /** Where the photons are aimed and which lights shoot them, for [scene]. */
    fun prepare(scene: InstancedScene) {
        aims = aimSpheres(scene)
        val cdf = lightCdf(scene, aims)
        active = aims.isNotEmpty() && cdf.isNotEmpty()
        if (lightCdf != 0L) { DeviceMemory.free(lightCdf); memory.remove(lightCdf) }
        lightCdf = DeviceMemory.upload(if (cdf.isEmpty()) FloatArray(1) else cdf).also { memory += it }
        System.err.println(
            "[caustics] ${aims.size / 4} aim sphere(s) round the casters, ${cdf.size} light(s)" +
                if (active) "" else " - nothing to make caustics with"
        )
        if (LOG) for (i in 0 until aims.size / 4) System.err.println("[caustics] aim %d: centre (%.3f, %.3f, %.3f), radius %.3f".format(i, aims[i * 4], aims[i * 4 + 1], aims[i * 4 + 2], aims[i * 4 + 3]))
    }

    /** At most this many samples a launch, so each launch's map has [MAX_PHOTONS] photons or fewer. */
    fun samplesPerLaunch(setup: CameraSetup): Int = max(1, MAX_PHOTONS / photonsPerSample(setup))

    private fun photonsPerSample(setup: CameraSetup): Int =
        System.getProperty("rt.caustics.photons")?.toIntOrNull()?.coerceAtLeast(1024) ?: max(16384, setup.totalPixels / 4)

    /**
     * Readies the map of a launch of [count] samples from [sampleStart]: the gather radius for
     * this launch and the photon count, written where the kernels read them. Returns the photons
     * the photon pass is to shoot.
     */
    fun begin(setup: CameraSetup, sampleStart: Int, count: Int): Int {
        if (sampleStart == 0) {
            if (launchIndex > 0) report()
            launchIndex = 0; shot = 0; landed = 0
        }
        val n = min(MAX_PHOTONS, photonsPerSample(setup) * count)
        ensureCapacity(n)
        val start = startRadius(setup)
        val radius = max(start * (launchIndex + 1.0).pow(-1.0 / 6.0), start / 3.0).toFloat()
        launchIndex++
        shot += n
        writeParams(n, radius)
        return n
    }

    /** Sorts the photons the pass landed into the grid. */
    fun buildGrid(photonCount: Int) {
        cuda.makeCurrent()
        CudaContext.check(cuMemsetD32(counts, 0, tableSize.toLong()), "cuMemsetD32(counts)")
        val radius = paramsHost.getFloat(PARAMS_RADIUS)
        val blocks = (tableSize + SCAN_BLOCK - 1) / SCAN_BLOCK
        MemoryStack.stackPush().use { stack ->
            fun args(vararg values: Any): org.lwjgl.PointerBuffer {
                val p = stack.mallocPointer(values.size)
                values.forEachIndexed { i, v ->
                    p.put(i, when (v) {
                        is Long -> MemoryUtil.memAddress(stack.longs(v))
                        is Int -> MemoryUtil.memAddress(stack.ints(v))
                        is Float -> MemoryUtil.memAddress(stack.floats(v))
                        else -> error("argument $v")
                    })
                }
                return p
            }
            fun run(fn: Long, threads: Int, block: Int, a: org.lwjgl.PointerBuffer) {
                val grid = max(1, (threads + block - 1) / block)
                CudaContext.check(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0L, a, null), "cuLaunchKernel(caustics grid)")
            }
            run(countFn, photonCount, 256, args(photons, photonCount, radius * 2f, tableSize - 1, counts, keys))
            run(scanBlocksFn, blocks * SCAN_BLOCK, SCAN_BLOCK, args(counts, starts, blockSums, tableSize))
            run(scanSumsFn, 1, 1, args(blockSums, blocks))
            run(addOffsetsFn, tableSize, 256, args(starts, blockSums, tableSize, blocks, cursor))
            run(scatterFn, photonCount, 256, args(photons, keys, photonCount, cursor, sorted))
            CudaContext.check(cuCtxSynchronize(), "caustics grid (cuCtxSynchronize)")
        }
        if (LOG) {
            // How many of the launch's photons landed: the aim's efficiency, for tuning.
            val total = IntArray(1)
            MemoryStack.stackPush().use { stack ->
                val host = stack.mallocInt(1)
                CudaContext.check(org.lwjgl.cuda.CU.cuMemcpyDtoH(host, starts + tableSize * 4L), "cuMemcpyDtoH(landed)")
                total[0] = host.get(0)
            }
            landed += total[0]
        }
    }

    private fun report() {
        if (LOG && shot > 0) System.err.println("[caustics] $launchIndex maps, ${shot} photons shot, ${landed} landed (${"%.1f".format(100.0 * landed / shot)}%)")
    }

    private fun startRadius(setup: CameraSetup): Double {
        System.getProperty("rt.caustics.radius")?.toDoubleOrNull()?.takeIf { it > 0 }?.let { return it }
        // START_PIXELS pixels at the nearest caster's distance: a pixel's width there.
        val c = setup.cameraCenter
        val centre = setup.pixel00 + setup.pixelDeltaU * ((setup.imageWidth - 1) / 2f) + setup.pixelDeltaV * ((setup.imageHeight - 1) / 2f)
        val plane = sqrt(((centre.x - c.x) * (centre.x - c.x) + (centre.y - c.y) * (centre.y - c.y) + (centre.z - c.z) * (centre.z - c.z)).toDouble())
        val pixelPerUnit = setup.pixelDeltaU.length() / plane
        var nearest = Double.MAX_VALUE
        for (i in 0 until aims.size / 4) {
            val dx = aims[i * 4] - c.x; val dy = aims[i * 4 + 1] - c.y; val dz = aims[i * 4 + 2] - c.z
            nearest = min(nearest, max(sqrt((dx * dx + dy * dy + dz * dz).toDouble()) - aims[i * 4 + 3] * 0.5, aims[i * 4 + 3].toDouble() * 0.1))
        }
        if (nearest == Double.MAX_VALUE) nearest = plane
        return START_PIXELS * pixelPerUnit * nearest
    }

    private fun ensureCapacity(n: Int) {
        if (n <= capacity) return
        listOf(photons, sorted, keys, counts, starts, cursor, blockSums).filter { it != 0L }.forEach { DeviceMemory.free(it); memory.remove(it) }
        capacity = n
        tableSize = Integer.highestOneBit(max(1024, n * 2 - 1)) shl 1
        val blocks = (tableSize + SCAN_BLOCK - 1) / SCAN_BLOCK
        photons = DeviceMemory.allocate(n * PHOTON_FLOATS * 4L).also { memory += it }
        sorted = DeviceMemory.allocate(n * PHOTON_FLOATS * 4L).also { memory += it }
        keys = DeviceMemory.allocate(n * 4L).also { memory += it }
        counts = DeviceMemory.allocate(tableSize * 4L).also { memory += it }
        starts = DeviceMemory.allocate((tableSize + 1) * 4L).also { memory += it }
        cursor = DeviceMemory.allocate(tableSize * 4L).also { memory += it }
        blockSums = DeviceMemory.allocate((blocks + 1) * 4L).also { memory += it }
    }

    private fun writeParams(photonCount: Int, radius: Float) {
        paramsHost.clear()
        paramsHost.putLong(photons).putLong(sorted).putLong(starts).putLong(lightCdf)
        paramsHost.putInt(photonCount).putInt(tableSize).putFloat(radius * 2f).putFloat(radius)
        paramsHost.putInt(aims.size / 4).putInt(0)
        for (i in 0 until AIMS * 4) paramsHost.putFloat(aims.getOrElse(i) { 0f })
        check(paramsHost.position() == PARAMS_BYTES) { "CausticParams is ${paramsHost.position()} bytes, the kernel expects $PARAMS_BYTES" }
        paramsHost.flip()
        DeviceMemory.write(params, paramsHost)
    }

    override fun close() {
        report()
        cuda.makeCurrent()
        cuModuleUnload(module)
        memory.forEach { DeviceMemory.free(it) }
        memory.clear()
        MemoryUtil.memFree(paramsHost)
    }

    companion object {
        /** Whether -Drt.caustics asks for them. */
        val enabled: Boolean get() = System.getProperty("rt.caustics").let { it != null && it != "off" && it != "false" }
        private val LOG: Boolean get() = System.getProperty("rt.caustics.log") != null

        const val MAX_PHOTONS = 2_000_000
        const val START_PIXELS = 3.0
        const val PHOTON_FLOATS = 12
        const val AIMS = 8
        const val SCAN_BLOCK = 1024
        /** The kernel's CausticParams: four pointers, six numbers of 32 bits, the aims. */
        const val PARAMS_BYTES = 4 * 8 + 6 * 4 + AIMS * 4 * 4
        private const val PARAMS_RADIUS = 4 * 8 + 3 * 4
        /** Matches the kernel's CAUSTIC_MAX_FUZZ. */
        private const val MAX_FUZZ = 0.04f

        fun gridSource(): String = CausticMap::class.java.getResource("/kernels/caustics_grid.cu")!!.readText()

        /** Whether material [m] may be a caster: glass, or a metal smooth enough (or with a roughness texture, which may make it so). */
        private fun mayCast(scene: InstancedScene, m: Int): Boolean {
            if (m < 0 || m * 4 >= scene.materialInts.size) return false
            return when (scene.materialInts[m * 4]) {
                2 -> true
                1 -> scene.materialFloats[m * 4] <= MAX_FUZZ || scene.materialInts[m * 4 + 3] >= 0
                else -> false
            }
        }

        /**
         * Spheres round the casters, centre and radius each, at most [AIMS]: one round each
         * object's casting faces - a placed mesh's faces that wear a caster, each loose
         * sphere, the loose triangles and quads together - merged, the closest pairs first,
         * until they fit. Built face by face: a model file is one mesh with its floor in it.
         */
        internal fun aimSpheres(scene: InstancedScene): FloatArray {
            val boxes = mutableListOf<FloatArray>()
            fun box() = floatArrayOf(Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE)
            fun FloatArray.add(x: Float, y: Float, z: Float) {
                this[0] = min(this[0], x); this[1] = min(this[1], y); this[2] = min(this[2], z)
                this[3] = max(this[3], x); this[4] = max(this[4], y); this[5] = max(this[5], z)
            }
            fun FloatArray.empty() = this[0] > this[3]

            for (instance in scene.instances) {
                val mesh = instance.block.mesh
                val faces = mesh.faceMaterials
                val b = box()
                val v = mesh.vertices
                for (t in 0 until mesh.triangleCount) {
                    val material = faces?.let { instance.palette?.getOrNull(it[t]) } ?: instance.material
                    if (!mayCast(scene, material)) continue
                    for (k in 0 until 3) {
                        val at = t * 9 + k * 3
                        val x = v[at]; val y = v[at + 1]; val z = v[at + 2]
                        b.add(
                            (instance.cos * x + instance.sin * z) * instance.scale + instance.offset.x,
                            y * instance.scale + instance.offset.y,
                            (-instance.sin * x + instance.cos * z) * instance.scale + instance.offset.z
                        )
                    }
                }
                if (!b.empty()) boxes += b
            }
            for (s in scene.sceneSpheres) {
                if (!mayCast(scene, scene.sphereMaterials[s])) continue
                val d = scene.spheres; val at = s * SceneBuffers.SPHERE_STRIDE; val r = d[at + 6]
                val b = box()
                for (moved in 0..1) {
                    val x = d[at] + moved * d[at + 3]; val y = d[at + 1] + moved * d[at + 4]; val z = d[at + 2] + moved * d[at + 5]
                    b.add(x - r, y - r, z - r); b.add(x + r, y + r, z + r)
                }
                boxes += b
            }
            val loose = box()
            for (t in 0 until scene.sceneTriangles) {
                if (!mayCast(scene, scene.triangleMaterials[t])) continue
                val d = scene.triangles; val at = t * SceneBuffers.TRIANGLE_STRIDE
                loose.add(d[at], d[at + 1], d[at + 2])
                loose.add(d[at] + d[at + 4], d[at + 1] + d[at + 5], d[at + 2] + d[at + 6])
                loose.add(d[at] + d[at + 8], d[at + 1] + d[at + 9], d[at + 2] + d[at + 10])
            }
            for (q in scene.sceneQuads) {
                if (!mayCast(scene, scene.quadMaterials[q])) continue
                val d = scene.quads; val at = q * SceneBuffers.QUAD_STRIDE
                for (cu in 0..1) for (cv in 0..1) {
                    loose.add(d[at] + cu * d[at + 3] + cv * d[at + 6], d[at + 1] + cu * d[at + 4] + cv * d[at + 7], d[at + 2] + cu * d[at + 5] + cv * d[at + 8])
                }
            }
            if (!loose.empty()) boxes += loose

            // Too many: merge the pair whose union grows least, until they fit.
            while (boxes.size > AIMS) {
                var best = Double.MAX_VALUE; var bi = 0; var bj = 1
                for (i in boxes.indices) for (j in i + 1 until boxes.size) {
                    val a = boxes[i]; val c = boxes[j]
                    val u = (0..2).map { (max(a[it + 3], c[it + 3]) - min(a[it], c[it])).toDouble() }
                    val grow = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
                    if (grow < best) { best = grow; bi = i; bj = j }
                }
                val a = boxes[bi]; val c = boxes[bj]
                boxes[bi] = floatArrayOf(min(a[0], c[0]), min(a[1], c[1]), min(a[2], c[2]), max(a[3], c[3]), max(a[4], c[4]), max(a[5], c[5]))
                boxes.removeAt(bj)
            }
            val out = FloatArray(boxes.size * 4)
            boxes.forEachIndexed { i, b ->
                val hx = (b[3] - b[0]) / 2; val hy = (b[4] - b[1]) / 2; val hz = (b[5] - b[2]) / 2
                out[i * 4] = b[0] + hx; out[i * 4 + 1] = b[1] + hy; out[i * 4 + 2] = b[2] + hz
                // A little over the half diagonal, so a face on the box's corner is inside.
                out[i * 4 + 3] = sqrt(hx * hx + hy * hy + hz * hz) * 1.01f + 1e-4f
            }
            return out
        }

        /**
         * The lights' share of the photons, cumulative: each in proportion to the light it sends
         * into the aim spheres - its radiance's luminance, times its area seen from each aim,
         * times the solid angle the aim fills from it. A sun far off and a lamp close by then
         * get their photons by what they put on the casters.
         */
        internal fun lightCdf(scene: InstancedScene, aims: FloatArray): FloatArray {
            val count = scene.lights.size / 2
            if (count == 0 || aims.isEmpty()) return FloatArray(0)
            val weights = DoubleArray(count) { i ->
                val kind = scene.lights[i * 2]; val index = scene.lights[i * 2 + 1]
                val material = if (kind == SceneBuffers.LEAF_QUAD) scene.quadMaterials[index] else scene.sphereMaterials[index]
                val texture = scene.materialInts[material * 4 + 1]
                val luminance = if (scene.textureInts[texture * 4] == 0) {
                    val at = texture * 4
                    0.2126 * scene.textureFloats[at] + 0.7152 * scene.textureFloats[at + 1] + 0.0722 * scene.textureFloats[at + 2]
                } else 1.0
                var sum = 0.0
                for (a in 0 until aims.size / 4) {
                    val ax = aims[a * 4]; val ay = aims[a * 4 + 1]; val az = aims[a * 4 + 2]; val ar = aims[a * 4 + 3].toDouble()
                    val (cx, cy, cz, projected) = if (kind == SceneBuffers.LEAF_QUAD) {
                        val d = scene.quads; val at = index * SceneBuffers.QUAD_STRIDE
                        val ux = d[at + 3]; val uy = d[at + 4]; val uz = d[at + 5]; val vx = d[at + 6]; val vy = d[at + 7]; val vz = d[at + 8]
                        val cx = d[at] + (ux + vx) / 2; val cy = d[at + 1] + (uy + vy) / 2; val cz = d[at + 2] + (uz + vz) / 2
                        val nx = uy * vz - uz * vy; val ny = uz * vx - ux * vz; val nz = ux * vy - uy * vx
                        val area = sqrt((nx * nx + ny * ny + nz * nz).toDouble())
                        val tx = ax - cx; val ty = ay - cy; val tz = az - cz
                        val tl = sqrt((tx * tx + ty * ty + tz * tz).toDouble()).coerceAtLeast(1e-9)
                        val front = (d[at + 12] * tx + d[at + 13] * ty + d[at + 14] * tz) / tl
                        Quad4(cx, cy, cz, area * max(front, 0.05))
                    } else {
                        val d = scene.spheres; val at = index * SceneBuffers.SPHERE_STRIDE; val r = d[at + 6].toDouble()
                        Quad4(d[at], d[at + 1], d[at + 2], Math.PI * r * r)
                    }
                    val dx = ax - cx; val dy = ay - cy; val dz = az - cz
                    val d2 = max((dx * dx + dy * dy + dz * dz).toDouble(), ar * ar)
                    sum += projected * Math.PI * ar * ar / d2
                }
                luminance * sum
            }
            val total = weights.sum()
            if (total <= 0.0) return FloatArray(0)
            var running = 0.0
            return FloatArray(count) { i -> running += weights[i]; (running / total).toFloat() }.also { it[count - 1] = 1f }
        }

        private data class Quad4(val x: Float, val y: Float, val z: Float, val projected: Double)
    }
}
