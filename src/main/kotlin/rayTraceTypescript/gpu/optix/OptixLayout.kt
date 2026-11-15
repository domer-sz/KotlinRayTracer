package rayTraceTypescript.gpu.optix

import org.lwjgl.cuda.CU.cuCtxSynchronize
import org.lwjgl.cuda.CU.cuLaunchKernel
import org.lwjgl.cuda.CU.cuMemAlloc
import org.lwjgl.cuda.CU.cuMemFree
import org.lwjgl.cuda.CU.cuMemcpyDtoH
import org.lwjgl.cuda.CU.cuModuleUnload
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil
import rayTraceTypescript.gpu.CudaContext
import java.nio.ByteBuffer

/**
 * Sizes, field offsets and constants of OptiX's C structures - asked of the compiler instead of
 * copied by hand.
 *
 * A one-thread kernel, built by NVRTC against the very headers the renderer's kernel includes,
 * writes each `sizeof`, each field's address in a structure at address zero, and each constant
 * into a buffer. Plain C structures are laid out for the card as they are for the host, so those
 * numbers are the host's too. Nothing here can then disagree with the headers in use, and a
 * field the headers no longer have fails the build by name.
 */
internal class OptixLayout private constructor(private val values: Map<String, Long>) {

    fun size(struct: String): Int = value("sizeof($struct)").toInt()

    fun offset(struct: String, field: String): Int = value("AT($struct, $field)").toInt()

    fun constant(name: String): Long = value(name)

    private fun value(expression: String): Long =
        values[expression] ?: throw IllegalArgumentException("$expression is not among the values OptixLayout asks for")

    /** [count] zeroed structures of type [struct] in native memory. */
    fun struct(struct: String, count: Int = 1) = OptixStruct(this, struct, count)

    companion object {
        /** Structures and the fields of theirs the renderer sets or reads. */
        private val STRUCTS: Map<String, List<String>> = linkedMapOf(
            "OptixDeviceContextOptions" to listOf("logCallbackFunction", "logCallbackLevel", "validationMode"),
            "OptixAccelBuildOptions" to listOf("buildFlags", "operation"),
            "OptixBuildInput" to listOf(
                "type",
                "triangleArray.vertexBuffers", "triangleArray.numVertices", "triangleArray.vertexFormat",
                "triangleArray.vertexStrideInBytes", "triangleArray.flags", "triangleArray.numSbtRecords",
                "customPrimitiveArray.aabbBuffers", "customPrimitiveArray.numPrimitives",
                "customPrimitiveArray.strideInBytes", "customPrimitiveArray.flags", "customPrimitiveArray.numSbtRecords",
                "instanceArray.instances", "instanceArray.numInstances", "instanceArray.instanceStride"
            ),
            "OptixAccelBufferSizes" to listOf("outputSizeInBytes", "tempSizeInBytes"),
            "OptixAccelEmitDesc" to listOf("result", "type"),
            "OptixInstance" to listOf("transform", "instanceId", "sbtOffset", "visibilityMask", "flags", "traversableHandle"),
            "OptixModuleCompileOptions" to listOf("maxRegisterCount", "optLevel", "debugLevel"),
            "OptixPipelineCompileOptions" to listOf(
                "usesMotionBlur", "traversableGraphFlags", "numPayloadValues", "numAttributeValues",
                "exceptionFlags", "pipelineLaunchParamsVariableName", "usesPrimitiveTypeFlags"
            ),
            "OptixProgramGroupDesc" to listOf(
                "kind", "raygen.module", "raygen.entryFunctionName", "miss.module", "miss.entryFunctionName",
                "hitgroup.moduleCH", "hitgroup.entryFunctionNameCH", "hitgroup.moduleIS", "hitgroup.entryFunctionNameIS"
            ),
            "OptixProgramGroupOptions" to emptyList(),
            "OptixPipelineLinkOptions" to listOf("maxTraceDepth"),
            "OptixShaderBindingTable" to listOf(
                "raygenRecord", "missRecordBase", "missRecordStrideInBytes", "missRecordCount",
                "hitgroupRecordBase", "hitgroupRecordStrideInBytes", "hitgroupRecordCount"
            ),
            "OptixStackSizes" to listOf("cssRG", "cssMS", "cssCH", "cssAH", "cssIS", "cssCC", "dssDC"),
            "OptixFunctionTable" to emptyList()
        )

        private val CONSTANTS = listOf(
            "OPTIX_VERSION", "OPTIX_ABI_VERSION",
            "OPTIX_SBT_RECORD_HEADER_SIZE", "OPTIX_SBT_RECORD_ALIGNMENT",
            "OPTIX_BUILD_INPUT_TYPE_TRIANGLES", "OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES", "OPTIX_BUILD_INPUT_TYPE_INSTANCES",
            "OPTIX_VERTEX_FORMAT_FLOAT3",
            "OPTIX_BUILD_FLAG_ALLOW_COMPACTION", "OPTIX_BUILD_FLAG_PREFER_FAST_TRACE", "OPTIX_BUILD_FLAG_PREFER_FAST_BUILD",
            "OPTIX_BUILD_OPERATION_BUILD",
            "OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT",
            "OPTIX_PROPERTY_TYPE_COMPACTED_SIZE",
            "OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS", "OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING",
            "OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE", "OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM",
            "OPTIX_PROGRAM_GROUP_KIND_RAYGEN", "OPTIX_PROGRAM_GROUP_KIND_MISS", "OPTIX_PROGRAM_GROUP_KIND_HITGROUP",
            "OPTIX_INSTANCE_FLAG_NONE",
            "OPTIX_DEVICE_PROPERTY_RTCORE_VERSION"
        )

        private val EXPRESSIONS: List<String> =
            STRUCTS.flatMap { (struct, fields) -> listOf("sizeof($struct)") + fields.map { "AT($struct, $it)" } } + CONSTANTS

        @Volatile private var cached: OptixLayout? = null

        /** The layout of the headers in [OptixApi.includeFolder], worked out once per run on [cuda]'s card. */
        @Synchronized
        fun of(cuda: CudaContext): OptixLayout = cached ?: probe(cuda).also { cached = it }

        /** What a kernel must say before it includes an OptiX header: NVRTC does not tell which host it builds for. */
        const val PREAMBLE = "#if !defined(_WIN64) && !defined(__LP64__)\n#define _WIN64 1\n#endif\n"

        private fun probe(cuda: CudaContext): OptixLayout {
            val source = buildString {
                append(PREAMBLE)
                // The function table's header wants cuda.h only for these two handle types.
                append("#define OPTIX_DONT_INCLUDE_CUDA\n")
                append("typedef struct CUctx_st* CUcontext;\ntypedef struct CUstream_st* CUstream;\n")
                append("#include <optix.h>\n#include <optix_function_table.h>\n")
                append("#define AT(T, m) ((unsigned long long)&(((T*)0)->m))\n")
                append("extern \"C\" __global__ void render(unsigned long long* out) {\n")
                EXPRESSIONS.forEachIndexed { i, expression -> append("    out[$i] = (unsigned long long)($expression);\n") }
                append("}\n")
            }
            val (module, function) = cuda.buildKernel(source, listOf("--std=c++17", OptixApi.includeOption()))
            try {
                MemoryStack.stackPush().use { stack ->
                    val pointer = stack.mallocPointer(1)
                    CudaContext.check(cuMemAlloc(pointer, EXPRESSIONS.size * 8L), "cuMemAlloc(layout)")
                    val buffer = pointer.get(0)
                    try {
                        val params = stack.mallocPointer(1).put(0, MemoryUtil.memAddress(stack.longs(buffer)))
                        CudaContext.check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0L, params, null), "cuLaunchKernel(layout)")
                        CudaContext.check(cuCtxSynchronize(), "layout kernel")
                        val host = MemoryUtil.memAllocLong(EXPRESSIONS.size)
                        try {
                            CudaContext.check(cuMemcpyDtoH(host, buffer), "cuMemcpyDtoH(layout)")
                            return OptixLayout(EXPRESSIONS.withIndex().associate { (i, expression) -> expression to host.get(i) })
                        } finally {
                            MemoryUtil.memFree(host)
                        }
                    } finally {
                        cuMemFree(buffer)
                    }
                }
            } finally {
                cuModuleUnload(module)
            }
        }
    }
}

/** [count] zeroed C structures of one OptiX type, their fields set and read by name. */
internal class OptixStruct(private val layout: OptixLayout, private val type: String, count: Int) : AutoCloseable {
    val stride: Int = layout.size(type)
    val memory: ByteBuffer = MemoryUtil.memCalloc(stride * count)
    val address: Long get() = MemoryUtil.memAddress(memory)

    fun int(field: String, value: Int, index: Int = 0) = apply { memory.putInt(at(field, index), value) }
    fun long(field: String, value: Long, index: Int = 0) = apply { memory.putLong(at(field, index), value) }
    fun float(field: String, value: Float, index: Int = 0, element: Int = 0) =
        apply { memory.putFloat(at(field, index) + element * 4, value) }

    // Named apart from the setters: int("x", 3) must never read element 3 instead of writing 3.
    fun readInt(field: String, index: Int = 0): Int = memory.getInt(at(field, index))
    fun readLong(field: String, index: Int = 0): Long = memory.getLong(at(field, index))

    private fun at(field: String, index: Int) = index * stride + layout.offset(type, field)

    override fun close() = MemoryUtil.memFree(memory)
}
