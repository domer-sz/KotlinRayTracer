package rayTraceTypescript.gpu.optix

import org.lwjgl.cuda.CU.cuCtxSynchronize
import org.lwjgl.cuda.CU.cuMemcpyDtoH
import org.lwjgl.system.MemoryUtil
import rayTraceTypescript.gpu.CudaContext
import rayTraceTypescript.gpu.DeviceMemory

/**
 * The trees OptiX searches ("acceleration structures"), built on the card of the current CUDA
 * context: OptiX is handed the vertices and makes of them whatever its traversal wants - on an
 * RTX card, what the RT cores walk.
 */
internal class OptixAccel(private val layout: OptixLayout, private val context: Long) {

    /** A built tree: the handle rays are traced against, and the device memory it lives in. */
    class Tree(val handle: Long, val memory: Long, val bytes: Long)

    /**
     * One tree over several sets of triangles, each 9 floats a triangle. A hit reports which set
     * it is in (the kernel's optixGetSbtGASIndex) and the triangle's number within it.
     */
    fun triangles(vertexSets: List<FloatArray>): Tree {
        require(vertexSets.isNotEmpty() && vertexSets.all { it.isNotEmpty() && it.size % 9 == 0 }) { "Triangle sets of 9 floats a triangle" }
        val count = vertexSets.size
        val device = mutableListOf<Long>()
        // OptiX takes the address of an array of vertex buffers (one per motion key) per input.
        val vertexBuffers = MemoryUtil.memAllocLong(count)
        val geometryFlags = MemoryUtil.memAllocInt(1).put(0, layout.constant("OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT").toInt())
        val inputs = layout.struct("OptixBuildInput", count)
        try {
            vertexSets.forEachIndexed { i, vertices ->
                vertexBuffers.put(i, DeviceMemory.upload(vertices).also { device += it })
                inputs.int("type", layout.constant("OPTIX_BUILD_INPUT_TYPE_TRIANGLES").toInt(), i)
                    .long("triangleArray.vertexBuffers", MemoryUtil.memAddress(vertexBuffers) + i * 8L, i)
                    .int("triangleArray.numVertices", vertices.size / 3, i)
                    .int("triangleArray.vertexFormat", layout.constant("OPTIX_VERTEX_FORMAT_FLOAT3").toInt(), i)
                    .int("triangleArray.vertexStrideInBytes", 12, i)
                    .long("triangleArray.flags", MemoryUtil.memAddress(geometryFlags), i)
                    .int("triangleArray.numSbtRecords", 1, i)
            }
            val flags = layout.constant("OPTIX_BUILD_FLAG_ALLOW_COMPACTION") or layout.constant("OPTIX_BUILD_FLAG_PREFER_FAST_TRACE")
            return build(inputs, count, flags.toInt())
        } finally {
            device.forEach { DeviceMemory.free(it) }
            inputs.close()
            MemoryUtil.memFree(vertexBuffers)
            MemoryUtil.memFree(geometryFlags)
        }
    }

    /** Builds a tree over [inputs] and, where that saves memory, moves it into a tighter block. */
    private fun build(inputs: OptixStruct, count: Int, buildFlags: Int): Tree {
        val options = layout.struct("OptixAccelBuildOptions")
            .int("buildFlags", buildFlags)
            .int("operation", layout.constant("OPTIX_BUILD_OPERATION_BUILD").toInt())
        val sizes = layout.struct("OptixAccelBufferSizes")
        val emitted = layout.struct("OptixAccelEmitDesc")
        val handle = MemoryUtil.memAllocLong(1)
        var temporary = 0L
        var compactedSize = 0L
        var output = 0L
        try {
            OptixApi.call("optixAccelComputeMemoryUsage", context, options.address, inputs.address, count, sizes.address)
            val outputBytes = sizes.readLong("outputSizeInBytes")
            val temporaryBytes = sizes.readLong("tempSizeInBytes")
            temporary = DeviceMemory.allocate(temporaryBytes)
            output = DeviceMemory.allocate(outputBytes)
            compactedSize = DeviceMemory.allocate(8)
            emitted.long("result", compactedSize).int("type", layout.constant("OPTIX_PROPERTY_TYPE_COMPACTED_SIZE").toInt())
            OptixApi.call(
                "optixAccelBuild", context, 0L, options.address, inputs.address, count,
                temporary, temporaryBytes, output, outputBytes, MemoryUtil.memAddress(handle), emitted.address, 1
            )
            CudaContext.check(cuCtxSynchronize(), "optixAccelBuild (cuCtxSynchronize)")

            val compactBytes = MemoryUtil.memAllocLong(1).let { cell ->
                try {
                    CudaContext.check(cuMemcpyDtoH(cell, compactedSize), "cuMemcpyDtoH(compacted size)")
                    cell.get(0)
                } finally {
                    MemoryUtil.memFree(cell)
                }
            }
            if (compactBytes in 1 until outputBytes) {
                val compact = DeviceMemory.allocate(compactBytes)
                try {
                    OptixApi.call("optixAccelCompact", context, 0L, handle.get(0), compact, compactBytes, MemoryUtil.memAddress(handle))
                    CudaContext.check(cuCtxSynchronize(), "optixAccelCompact (cuCtxSynchronize)")
                } catch (t: Throwable) {
                    DeviceMemory.free(compact)
                    throw t
                }
                DeviceMemory.free(output)
                output = 0L
                return Tree(handle.get(0), compact, compactBytes)
            }
            return Tree(handle.get(0), output, outputBytes).also { output = 0L }
        } finally {
            DeviceMemory.free(temporary)
            DeviceMemory.free(compactedSize)
            DeviceMemory.free(output)
            options.close()
            sizes.close()
            emitted.close()
            MemoryUtil.memFree(handle)
        }
    }
}
