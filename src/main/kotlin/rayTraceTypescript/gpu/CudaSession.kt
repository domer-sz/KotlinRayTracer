package rayTraceTypescript.gpu

import org.lwjgl.cuda.CU.cuCtxSynchronize
import org.lwjgl.cuda.CU.cuLaunchKernel
import org.lwjgl.cuda.CU.cuMemAlloc
import org.lwjgl.cuda.CU.cuMemFree
import org.lwjgl.cuda.CU.cuMemcpyDtoH
import org.lwjgl.cuda.CU.cuMemcpyHtoD
import org.lwjgl.cuda.CU.cuModuleUnload
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil
import rayTraceTypescript.render.CameraSetup
import java.nio.ByteBuffer
import java.nio.FloatBuffer

/**
 * The CUDA backend of [GpuSession]: the scene's buffers in device memory, one module per kernel
 * mode compiled by NVRTC with the scene's features switched in, and the two outputs.
 */
internal class CudaSession(initialScene: SceneBuffers, maxPixels: Int) : GpuSession {
    private val keepAwake = GpuKeepAwake.startIfUseful()
    private val cuda = CudaContext.create()
    /** Device memory the session keeps: the camera and the outputs. */
    private val buffers = mutableListOf<Long>()
    /** The scene's device memory, freed when it is replaced. */
    private val sceneMemory = mutableListOf<Long>()
    /** Module and function per kernel mode. */
    private val kernels = HashMap<Int, Pair<Long, Long>>()
    override var seed: Long = GpuSession.initialSeed()

    private var scene = initialScene
    /** The feature switches the kernels were built with, and whether volumes need more registers. */
    private var defines = initialScene.kernelDefines()
    private var hasMedia = initialScene.mediumCount > 0

    private val cameraBuffer: Long
    private var sceneArguments: List<Long> = emptyList()
    private val outputs: List<Long>
    /** Host memory reads come back through, sized for the largest output. */
    private val staging: FloatBuffer = MemoryUtil.memAllocFloat(maxOf(maxPixels * 3, GpuSession.CAMERA_FLOATS))

    init {
        try {
            cameraBuffer = allocate(GpuSession.CAMERA_FLOATS * 4L).also { buffers += it }
            sceneArguments = upload(initialScene)
            // Read as well as written: every slice of samples after the first adds to it.
            outputs = List(2) { allocate(maxPixels * 3L * Float.SIZE_BYTES).also { buffers += it } }
        } catch (t: Throwable) {
            close()
            throw t
        }
    }

    override fun replaceScene(next: SceneBuffers) {
        cuda.makeCurrent()
        CudaContext.check(cuCtxSynchronize(), "cuCtxSynchronize")
        sceneMemory.forEach { cuMemFree(it) }
        sceneMemory.clear()
        scene = next
        sceneArguments = upload(next)
        val nextDefines = next.kernelDefines()
        val nextMedia = next.mediumCount > 0
        if (nextDefines != defines || nextMedia != hasMedia) {
            releaseKernels()
            defines = nextDefines
            hasMedia = nextMedia
        }
    }

    override fun setCamera(setup: CameraSetup) {
        cuda.makeCurrent()
        val parameters = GpuSession.cameraParameters(setup)
        staging.clear()
        staging.put(parameters).flip()
        CudaContext.check(cuMemcpyHtoD(cameraBuffer, staging), "cuMemcpyHtoD(camera)")
    }

    override fun launch(mode: Int, setup: CameraSetup, sampleStart: Int, count: Int, scale: Float, target: Int) {
        cuda.makeCurrent()
        val function = kernel(mode)
        MemoryStack.stackPush().use { stack ->
            // cuLaunchKernel takes a pointer to each argument's value, in the kernel's order:
            // the camera and the scene buffers, then the output and the launch's numbers.
            val pointers = sceneArguments + outputs[target]
            val params = stack.mallocPointer(pointers.size + 9)
            pointers.forEachIndexed { i, buffer -> params.put(i, MemoryUtil.memAddress(stack.longs(buffer))) }
            var at = pointers.size
            for (value in intArrayOf(setup.imageWidth, setup.imageHeight, sampleStart, count, setup.maxReflectionDepth)) {
                params.put(at++, MemoryUtil.memAddress(stack.ints(value)))
            }
            params.put(at++, MemoryUtil.memAddress(stack.floats(scale)))
            params.put(at++, MemoryUtil.memAddress(stack.ints(scene.rootNode)))
            params.put(at++, MemoryUtil.memAddress(stack.ints(scene.lightCount)))
            params.put(at, MemoryUtil.memAddress(stack.longs(seed)))

            // One thread per pixel, a warp of 32 covering an 8x4 tile: neighbouring rays stay
            // together on their first bounces, where they diverge least.
            val tiles = ((setup.imageWidth + 7) / 8).toLong() * ((setup.imageHeight + 3) / 4)
            val block = (System.getProperty("rt.cuda.block")?.toIntOrNull() ?: BLOCK_SIZE)
            val grid = ((tiles * 32 + block - 1) / block).toInt()
            CudaContext.check(cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, 0L, params, null), "cuLaunchKernel")
            CudaContext.check(cuCtxSynchronize(), "kernel (cuCtxSynchronize)")
        }
    }

    override fun read(target: Int, into: FloatArray) {
        cuda.makeCurrent()
        staging.clear()
        staging.limit(into.size)
        CudaContext.check(cuMemcpyDtoH(staging, outputs[target]), "cuMemcpyDtoH")
        staging.get(into)
    }

    /** The compiled kernel for [mode], built on first use. */
    private fun kernel(mode: Int): Long = kernels.getOrPut(mode) {
        val start = System.nanoTime()
        // -Drt.cuda.options adds compiler flags for experiments, space separated.
        val extra = System.getProperty("rt.cuda.options")?.split(' ')?.filter { it.isNotBlank() } ?: emptyList()
        val options = BUILD_OPTIONS + defines.split(' ').filter { it.isNotBlank() } +
            listOf("-DAOV_MODE=$mode") + registerCap(extra) + extra
        val built = cuda.buildKernel(GpuRenderer.cudaKernelSource(), options)
        System.err.println(
            "[gpu] built the ${GpuSession.MODE_NAMES[mode]} kernel for ${cuda.deviceName} (sm_${cuda.computeCapability}) " +
                "in ${(System.nanoTime() - start) / 1_000_000} ms"
        )
        built
    }.second

    /**
     * Left alone, NVIDIA's compiler gives the kernel ~100-128 registers, which fits only a
     * quarter of the threads a Pascal SM can hold. Capping registers costs some spilling but
     * hides memory latency far better: measured on a GTX 1070, 64 is best across the
     * scenes, except that volumes (a nested traversal) want 72.
     */
    private fun registerCap(extra: List<String>): List<String> =
        if (extra.any { it.contains("maxrregcount") }) emptyList() else listOf("--maxrregcount=${if (hasMedia) 72 else 64}")

    private fun releaseKernels() {
        kernels.values.forEach { (module, _) -> cuModuleUnload(module) }
        kernels.clear()
    }

    override fun close() {
        try {
            cuda.makeCurrent()
            releaseKernels()
            sceneMemory.forEach { cuMemFree(it) }
            buffers.forEach { cuMemFree(it) }
        } finally {
            MemoryUtil.memFree(staging)
            cuda.close()
            keepAwake?.close()
        }
    }

    /** The camera buffer, then every scene buffer, in the kernel's argument order. */
    private fun upload(s: SceneBuffers): List<Long> = listOf(
        cameraBuffer,
        copy(s.nodes), copy(s.leaves), copy(s.refs),
        copy(s.spheres), copy(s.sphereMaterials),
        copy(s.triangles), copy(s.triangleMaterials),
        copy(s.quads), copy(s.quadMaterials),
        copy(s.mediumInts), copy(s.mediumFloats), copy(s.lights),
        copy(s.materialInts), copy(s.materialFloats),
        copy(s.textureInts), copy(s.textureFloats),
        copy(s.textureImages),
        copy(s.perlinVectors), copy(s.perlinPermutations)
    )

    private fun allocate(bytes: Long): Long = MemoryStack.stackPush().use { stack ->
        val pointer = stack.mallocPointer(1)
        CudaContext.check(cuMemAlloc(pointer, maxOf(bytes, 4L)), "cuMemAlloc($bytes bytes)")
        pointer.get(0)
    }

    // Empty sections still get a buffer (of one element), so every kernel argument is valid.
    private fun copy(data: FloatArray): Long = copyBytes(data.size * 4L) { it.asFloatBuffer().put(data) }

    private fun copy(data: IntArray): Long = copyBytes(data.size * 4L) { it.asIntBuffer().put(data) }

    private fun copy(data: ByteArray): Long = copyBytes(data.size.toLong()) { it.put(data) }

    private fun copyBytes(bytes: Long, fill: (ByteBuffer) -> Unit): Long {
        val device = allocate(bytes).also { sceneMemory += it }
        if (bytes == 0L) return device
        val host = MemoryUtil.memAlloc(bytes.toInt())
        try {
            fill(host)
            // LWJGL copies what lies between position and limit; a put straight into the buffer
            // (the texture bytes) moved the position to the end.
            host.clear()
            CudaContext.check(cuMemcpyHtoD(device, host), "cuMemcpyHtoD($bytes bytes)")
        } finally {
            MemoryUtil.memFree(host)
        }
        return device
    }

    private companion object {
        const val BLOCK_SIZE = 64
        // --use_fast_math is CUDA's -cl-unsafe-math-optimizations and more: fused multiply-adds,
        // approximate division and square roots, flushing denormals. It does not assume finite
        // values, so the traversal's INFINITY and NaN-tolerant fminf/fmaxf keep working.
        val BUILD_OPTIONS = listOf("--use_fast_math", "--std=c++17")
    }
}
