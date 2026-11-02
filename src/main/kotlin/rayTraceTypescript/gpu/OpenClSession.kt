package rayTraceTypescript.gpu

import org.lwjgl.opencl.CL10.CL_MEM_COPY_HOST_PTR
import org.lwjgl.opencl.CL10.CL_MEM_READ_ONLY
import org.lwjgl.opencl.CL10.CL_MEM_READ_WRITE
import org.lwjgl.opencl.CL10.clCreateBuffer
import org.lwjgl.opencl.CL10.clCreateKernel
import org.lwjgl.opencl.CL10.clEnqueueNDRangeKernel
import org.lwjgl.opencl.CL10.clEnqueueReadBuffer
import org.lwjgl.opencl.CL10.clEnqueueWriteBuffer
import org.lwjgl.opencl.CL10.clFinish
import org.lwjgl.opencl.CL10.clReleaseKernel
import org.lwjgl.opencl.CL10.clReleaseMemObject
import org.lwjgl.opencl.CL10.clReleaseProgram
import org.lwjgl.opencl.CL10.clSetKernelArg1f
import org.lwjgl.opencl.CL10.clSetKernelArg1i
import org.lwjgl.opencl.CL10.clSetKernelArg1l
import org.lwjgl.opencl.CL10.clSetKernelArg1p
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil
import rayTraceTypescript.render.CameraSetup

/**
 * The OpenCL backend of [GpuSession]: a scene living on the GPU: its buffers uploaded once, one compiled kernel per mode (the image,
 * the two denoiser guides and the quick shaded look), and two output buffers - the image, and a
 * scratch one the guides render into so they never disturb samples gathered so far.
 *
 * Everything that changes between launches - the camera, the resolution, which samples - is a
 * launch argument, so the same session serves a file render and an interactive viewer. An
 * edited scene replaces the old one in place ([replaceScene]): same context, same outputs, and
 * the same kernels as long as the scene needs the same features. Launches block until the GPU
 * is done; a session belongs to one thread.
 */
internal class OpenClSession(initialScene: SceneBuffers, maxPixels: Int) : GpuSession {
    private val keepAwake = GpuKeepAwake.startIfUseful()
    private val cl = OpenClContext.create()
    /** Buffers the session keeps: the camera and the outputs. */
    private val buffers = mutableListOf<Long>()
    /** The scene's buffers, released when it is replaced. */
    private val sceneMemory = mutableListOf<Long>()
    private val programs = HashMap<Int, Pair<Long, Long>>()
    /**
     * Keys every random stream of the next launches, with the pixel and the sample. A film sets
     * it per frame, so its frames do not all share one noise pattern.
     */
    override var seed: Long = GpuSession.initialSeed()

    private var scene = initialScene
    /** The feature switches the kernels were built with, and whether volumes need more registers. */
    private var defines = initialScene.kernelDefines()
    private var hasMedia = initialScene.mediumCount > 0

    private val cameraBuffer: Long
    private var sceneArguments: List<Long> = emptyList()
    private val outputs: List<Long>

    init {
        try {
            cameraBuffer = create(FloatArray(CAMERA_FLOATS), buffers)
            sceneArguments = upload(initialScene)
            // Read as well as written: every slice of samples after the first adds to it.
            outputs = List(2) { createOutput(maxPixels * 3L * Float.SIZE_BYTES) }
        } catch (t: Throwable) {
            close()
            throw t
        }
    }

    /**
     * Swaps the scene for [next]: its buffers replace the old ones, and the kernels are kept -
     * only pointed at the new buffers - unless [next] needs other features compiled in.
     */
    override fun replaceScene(next: SceneBuffers) {
        clFinish(cl.queue)
        sceneMemory.forEach { clReleaseMemObject(it) }
        sceneMemory.clear()
        scene = next
        sceneArguments = upload(next)
        val nextDefines = next.kernelDefines()
        val nextMedia = next.mediumCount > 0
        if (nextDefines != defines || nextMedia != hasMedia) {
            releasePrograms()
            defines = nextDefines
            hasMedia = nextMedia
        } else {
            programs.values.forEach { (_, kernel) -> bindScene(kernel) }
        }
    }

    /** Points the camera buffer at [setup]; the next launch shoots rays from there. */
    override fun setCamera(setup: CameraSetup) {
        val parameters = GpuSession.cameraParameters(setup)
        OpenClContext.check(
            clEnqueueWriteBuffer(cl.queue, cameraBuffer, true, 0, parameters, null, null),
            "clEnqueueWriteBuffer(camera)"
        )
    }

    /**
     * Traces samples [sampleStart] until [sampleStart] + [count] of every pixel of [setup]'s
     * image, in [mode], into [target], scaled by [scale]. A launch whose [sampleStart] is above
     * zero adds to what the target holds. Returns when the GPU is done.
     */
    override fun launch(mode: Int, setup: CameraSetup, sampleStart: Int, count: Int, scale: Float, target: Int) {
        val kernel = kernel(mode)
        var arg = sceneArguments.size
        clSetKernelArg1p(kernel, arg++, outputs[target])
        clSetKernelArg1i(kernel, arg++, setup.imageWidth)
        clSetKernelArg1i(kernel, arg++, setup.imageHeight)
        clSetKernelArg1i(kernel, arg++, sampleStart)
        clSetKernelArg1i(kernel, arg++, count)
        clSetKernelArg1i(kernel, arg++, setup.maxReflectionDepth)
        clSetKernelArg1f(kernel, arg++, scale)
        clSetKernelArg1i(kernel, arg++, scene.rootNode)
        clSetKernelArg1i(kernel, arg++, scene.lightCount)
        clSetKernelArg1l(kernel, arg, seed)

        MemoryStack.stackPush().use { stack ->
            // One work item per pixel, a warp of 32 covering an 8x4 tile: neighbouring rays
            // stay together on their first bounces, where they diverge least.
            val tiles = ((setup.imageWidth + 7) / 8).toLong() * ((setup.imageHeight + 3) / 4)
            val localSize = System.getProperty("rt.cl.local")?.toLongOrNull() ?: LOCAL_SIZE
            val global = stack.mallocPointer(1).put(0, (tiles * 32 + localSize - 1) / localSize * localSize)
            val local = stack.mallocPointer(1).put(0, localSize)
            OpenClContext.check(
                clEnqueueNDRangeKernel(cl.queue, kernel, 1, null, global, local, null, null),
                "clEnqueueNDRangeKernel"
            )
            OpenClContext.check(clFinish(cl.queue), "clFinish")
        }
    }

    /** Copies the first [into].size floats of [target] back to the host. */
    override fun read(target: Int, into: FloatArray) {
        OpenClContext.check(clEnqueueReadBuffer(cl.queue, outputs[target], true, 0, into, null, null), "clEnqueueReadBuffer")
    }

    /** The compiled kernel for [mode], built on first use (the driver caches it across runs too). */
    private fun kernel(mode: Int): Long = programs.getOrPut(mode) {
        val start = System.nanoTime()
        // -Drt.cl.options adds compiler flags for experiments, e.g. -cl-nv-verbose.
        val extra = System.getProperty("rt.cl.options") ?: ""
        val options = listOf(BUILD_OPTIONS, defines, "-DAOV_MODE=$mode", registerCap(extra), extra)
            .joinToString(" ").trim()
        val program = cl.buildProgram(GpuRenderer.openClKernelSource(), options)
        val kernel = MemoryStack.stackPush().use { stack ->
            val errcode = stack.mallocInt(1)
            val handle = clCreateKernel(program, "render", errcode)
            OpenClContext.check(errcode, "clCreateKernel")
            handle
        }
        bindScene(kernel)
        System.err.println("[gpu] built the ${MODE_NAMES[mode]} kernel in ${(System.nanoTime() - start) / 1_000_000} ms")
        program to kernel
    }.second

    private fun bindScene(kernel: Long) {
        sceneArguments.forEachIndexed { i, buffer ->
            OpenClContext.check(clSetKernelArg1p(kernel, i, buffer), "clSetKernelArg($i)")
        }
    }

    /**
     * Left alone, NVIDIA's compiler gives the kernel ~100-128 registers, which fits only a
     * quarter of the threads a Pascal SM can hold. Capping registers costs some spilling but
     * hides memory latency far better: measured on a GTX 1070, 64 is best across the scenes,
     * except that volumes (a nested traversal) want 72. Only NVIDIA understands the flag.
     */
    private fun registerCap(extra: String): String {
        if (!cl.extensions.contains("cl_nv_compiler_options") || extra.contains("maxrregcount")) return ""
        return "-cl-nv-maxrregcount=${if (hasMedia) 72 else 64}"
    }

    private fun releasePrograms() {
        programs.values.forEach { (program, kernel) ->
            clReleaseKernel(kernel)
            clReleaseProgram(program)
        }
        programs.clear()
    }

    override fun close() {
        releasePrograms()
        sceneMemory.forEach { clReleaseMemObject(it) }
        buffers.forEach { clReleaseMemObject(it) }
        cl.close()
        keepAwake?.close()
    }

    /** The camera buffer, then every scene buffer, in the kernel's argument order. */
    private fun upload(s: SceneBuffers): List<Long> = listOf(
        cameraBuffer,
        create(s.nodes), create(s.leaves), create(s.refs),
        create(s.spheres), create(s.sphereMaterials),
        create(s.triangles), create(s.triangleMaterials),
        create(s.quads), create(s.quadMaterials),
        create(s.mediumInts), create(s.mediumFloats), create(s.lights),
        create(s.materialInts), create(s.materialFloats),
        create(s.textureInts), create(s.textureFloats),
        createBytes(s.textureImages),
        create(s.perlinVectors), create(s.perlinPermutations)
    )

    // OpenCL rejects zero-sized buffers, so empty sections get one padding element.
    private fun create(data: FloatArray, into: MutableList<Long> = sceneMemory): Long {
        val errcode = IntArray(1)
        val buffer = clCreateBuffer(cl.context, READ_FLAGS, if (data.isEmpty()) FloatArray(1) else data, errcode)
        OpenClContext.check(errcode, "clCreateBuffer")
        return buffer.also { into += it }
    }

    private fun create(data: IntArray): Long {
        val errcode = IntArray(1)
        val buffer = clCreateBuffer(cl.context, READ_FLAGS, if (data.isEmpty()) IntArray(1) else data, errcode)
        OpenClContext.check(errcode, "clCreateBuffer")
        return buffer.also { sceneMemory += it }
    }

    private fun createBytes(data: ByteArray): Long {
        val native = MemoryUtil.memAlloc(maxOf(data.size, 1))
        try {
            if (data.isNotEmpty()) native.put(data).flip()
            val errcode = IntArray(1)
            val buffer = clCreateBuffer(cl.context, READ_FLAGS, native, errcode)
            OpenClContext.check(errcode, "clCreateBuffer(images)")
            return buffer.also { sceneMemory += it }
        } finally {
            MemoryUtil.memFree(native)
        }
    }

    private fun createOutput(bytes: Long): Long = MemoryStack.stackPush().use { stack ->
        val errcode = stack.mallocInt(1)
        val handle = clCreateBuffer(cl.context, CL_MEM_READ_WRITE.toLong(), bytes, errcode)
        OpenClContext.check(errcode, "clCreateBuffer(output)")
        handle.also { buffers += it }
    }

    private companion object {
        val MODE_NAMES = GpuSession.MODE_NAMES
        const val IMAGE = GpuSession.IMAGE
        const val CAMERA_FLOATS = GpuSession.CAMERA_FLOATS
        const val LOCAL_SIZE = 64L
        // Unsafe math (fused multiply-adds, faster division and square roots) leaves the image
        // the same to within 0.1 of 255 per pixel at 2000 spp, and speeds the kernel up by a
        // quarter. Never -cl-fast-relaxed-math or -cl-finite-math-only: the traversal relies on
        // INFINITY for open intervals and on NaN-tolerant fmin/fmax in the slab test.
        private const val BUILD_OPTIONS = "-cl-std=CL1.2 -cl-unsafe-math-optimizations"
        private const val READ_FLAGS = (CL_MEM_READ_ONLY or CL_MEM_COPY_HOST_PTR).toLong()
    }
}
