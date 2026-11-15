package rayTraceTypescript.gpu.optix

import org.lwjgl.cuda.CU.cuCtxSynchronize
import org.lwjgl.cuda.CU.cuMemcpyDtoH
import org.lwjgl.cuda.CU.cuMemcpyHtoD
import org.lwjgl.system.MemoryUtil
import rayTraceTypescript.gpu.CudaContext
import rayTraceTypescript.gpu.DeviceMemory
import rayTraceTypescript.gpu.GpuKeepAwake
import rayTraceTypescript.gpu.GpuRenderer
import rayTraceTypescript.gpu.GpuSession
import rayTraceTypescript.gpu.SceneBuffers
import rayTraceTypescript.render.CameraSetup
import java.nio.ByteBuffer
import java.nio.FloatBuffer

/**
 * The OptiX backend of [GpuSession]: the kernel CudaSession runs, built as an OptiX program
 * (-DRT_OPTIX=1), with the scene's surfaces in a tree OptiX builds and searches - with the
 * card's RT cores where it has them. Sampling, materials and textures are the same code.
 *
 * As yet it takes triangles and quads; a scene with spheres or fog is refused with the reason.
 */
internal class OptixSession(initialScene: SceneBuffers, maxPixels: Int) : GpuSession {
    private val keepAwake = GpuKeepAwake.startIfUseful()
    private val cuda = CudaContext.create()
    override var seed: Long = GpuSession.initialSeed()

    /** Device memory kept for the session's life, and the scene's, freed when it is replaced. */
    private val sessionMemory = mutableListOf<Long>()
    private val sceneMemory = mutableListOf<Long>()
    /** C strings OptiX is handed by address: they must outlive the calls that name them. */
    private val names = HashMap<String, ByteBuffer>()
    private val pipelines = HashMap<Int, Pipeline>()
    private val paramsHost: ByteBuffer = MemoryUtil.memCalloc(PARAMS_BYTES)
    /** Host memory reads come back through, sized for the largest output. */
    private val staging: FloatBuffer = MemoryUtil.memAllocFloat(maxOf(maxPixels * 3, GpuSession.CAMERA_FLOATS))

    private lateinit var layout: OptixLayout
    private var context = 0L
    private var cameraBuffer = 0L
    private var paramsBuffer = 0L
    private var outputs: List<Long> = emptyList()

    private var scene = initialScene
    /** The feature switches the pipelines were built with. */
    private var defines = initialScene.kernelDefines()
    private var sceneArguments: List<Long> = emptyList()
    private var tree: OptixAccel.Tree? = null
    private var quadIds = 0L
    private var quadInput = -1

    /** One mode's programs, linked, with the table that tells OptiX which program serves what. */
    private class Pipeline(val module: Long, val groups: LongArray, val pipeline: Long, val records: Long, val sbt: OptixStruct)

    init {
        try {
            OptixApi.unavailabilityReason()?.let { throw OptixException(it) }
            layout = OptixLayout.of(cuda)
            context = createContext()
            cameraBuffer = DeviceMemory.allocate(GpuSession.CAMERA_FLOATS * 4L).also { sessionMemory += it }
            paramsBuffer = DeviceMemory.allocate(PARAMS_BYTES.toLong()).also { sessionMemory += it }
            // Read as well as written: every slice of samples after the first adds to it.
            outputs = List(2) { DeviceMemory.allocate(maxPixels * 3L * Float.SIZE_BYTES).also { sessionMemory += it } }
            load(initialScene)
        } catch (t: Throwable) {
            close()
            throw t
        }
    }

    override fun replaceScene(next: SceneBuffers) {
        cuda.makeCurrent()
        CudaContext.check(cuCtxSynchronize(), "cuCtxSynchronize")
        releaseScene()
        scene = next
        load(next)
        val nextDefines = next.kernelDefines()
        if (nextDefines != defines) {
            releasePipelines()
            defines = nextDefines
        }
    }

    override fun setCamera(setup: CameraSetup) {
        cuda.makeCurrent()
        staging.clear()
        staging.put(GpuSession.cameraParameters(setup)).flip()
        CudaContext.check(cuMemcpyHtoD(cameraBuffer, staging), "cuMemcpyHtoD(camera)")
    }

    override fun launch(mode: Int, setup: CameraSetup, sampleStart: Int, count: Int, scale: Float, target: Int) {
        cuda.makeCurrent()
        val pipeline = pipeline(mode)
        // The kernel's Params: pointers and 64-bit numbers first, then the 32-bit ones.
        paramsHost.clear()
        sceneArguments.forEach { paramsHost.putLong(it) }
        paramsHost.putLong(outputs[target]).putLong(quadIds).putLong(tree!!.handle).putLong(seed)
        paramsHost.putInt(setup.imageWidth).putInt(setup.imageHeight).putInt(sampleStart).putInt(count)
            .putInt(setup.maxReflectionDepth).putFloat(scale).putInt(scene.rootNode).putInt(scene.lightCount).putInt(quadInput)
        check(paramsHost.position() == PARAMS_BYTES - PARAMS_PADDING) { "Params is ${paramsHost.position()} bytes, the kernel expects ${PARAMS_BYTES - PARAMS_PADDING}" }
        DeviceMemory.write(paramsBuffer, paramsHost)

        OptixApi.call(
            "optixLaunch", pipeline.pipeline, 0L, paramsBuffer, PARAMS_BYTES.toLong(), pipeline.sbt.address,
            setup.imageWidth, setup.imageHeight, 1
        )
        CudaContext.check(cuCtxSynchronize(), "OptiX launch (cuCtxSynchronize)")
    }

    override fun read(target: Int, into: FloatArray) {
        cuda.makeCurrent()
        staging.clear()
        staging.limit(into.size)
        CudaContext.check(cuMemcpyDtoH(staging, outputs[target]), "cuMemcpyDtoH")
        staging.get(into)
    }

    override fun close() {
        try {
            cuda.makeCurrent()
            releasePipelines()
            releaseScene()
            sessionMemory.forEach { DeviceMemory.free(it) }
            if (context != 0L) OptixApi.call("optixDeviceContextDestroy", context)
        } finally {
            names.values.forEach { MemoryUtil.memFree(it) }
            MemoryUtil.memFree(paramsHost)
            MemoryUtil.memFree(staging)
            cuda.close()
            keepAwake?.close()
        }
    }

    private fun createContext(): Long = layout.struct("OptixDeviceContextOptions").use { options ->
        options.long("logCallbackFunction", OptixApi.logCallback).int("logCallbackLevel", OptixApi.logLevel())
        // -Drt.optix.validate has OptiX check every call and launch, at a large cost in speed.
        if (System.getProperty("rt.optix.validate") != null) options.int("validationMode", -1)
        out { OptixApi.call("optixDeviceContextCreate", cuda.context, options.address, it) }
    }

    /** Uploads [s] and builds the tree over its triangles and quads. */
    private fun load(s: SceneBuffers) {
        val kinds = (0 until s.refs.size step 2).map { s.refs[it] }.toSet()
        if (SceneBuffers.LEAF_SPHERE in kinds || SceneBuffers.LEAF_MEDIUM in kinds) {
            throw OptixUnsupportedScene("the OptiX engine does not trace spheres or fog yet (-Drt.backend=cuda renders this scene)")
        }
        fun keep(pointer: Long) = pointer.also { sceneMemory += it }
        // The kernel's buffers, in the order of its Params. The CUDA kernel's own tree
        // (nodes, leaves, refs) is not read here: OptiX searches its own.
        sceneArguments = listOf(
            cameraBuffer,
            keep(DeviceMemory.allocate(0)), keep(DeviceMemory.allocate(0)), keep(DeviceMemory.allocate(0)),
            keep(DeviceMemory.upload(s.spheres)), keep(DeviceMemory.upload(s.sphereMaterials)),
            keep(DeviceMemory.upload(s.triangles)), keep(DeviceMemory.upload(s.triangleMaterials)),
            keep(DeviceMemory.upload(s.quads)), keep(DeviceMemory.upload(s.quadMaterials)),
            keep(DeviceMemory.upload(s.mediumInts)), keep(DeviceMemory.upload(s.mediumFloats)), keep(DeviceMemory.upload(s.lights)),
            keep(DeviceMemory.upload(s.materialInts)), keep(DeviceMemory.upload(s.materialFloats)),
            keep(DeviceMemory.upload(s.textureInts)), keep(DeviceMemory.upload(s.textureFloats)),
            keep(DeviceMemory.upload(s.textureImages)),
            keep(DeviceMemory.upload(s.perlinVectors)), keep(DeviceMemory.upload(s.perlinPermutations))
        )

        // Triangles as the kernel keeps them (first vertex, two edges), turned into three vertices.
        val stride = SceneBuffers.TRIANGLE_STRIDE
        val triangleVertices = FloatArray(s.triangleCount * 9)
        for (t in 0 until s.triangleCount) {
            for (axis in 0 until 3) {
                val v0 = s.triangles[t * stride + axis]
                triangleVertices[t * 9 + axis] = v0
                triangleVertices[t * 9 + 3 + axis] = v0 + s.triangles[t * stride + 4 + axis]
                triangleVertices[t * 9 + 6 + axis] = v0 + s.triangles[t * stride + 8 + axis]
            }
        }
        // The quads that are surfaces of the scene, each as two triangles. The quad buffer also
        // holds the lights' stand-ins for sampling, which no ray may hit a second time.
        val sceneQuads = (0 until s.refs.size step 2).filter { s.refs[it] == SceneBuffers.LEAF_QUAD }.map { s.refs[it + 1] }.toIntArray()
        val quadVertices = FloatArray(sceneQuads.size * 18)
        sceneQuads.forEachIndexed { k, quad ->
            val at = quad * SceneBuffers.QUAD_STRIDE
            // Corner Q, then Q + u, Q + u + v and Q + v.
            QUAD_CORNERS.forEachIndexed { corner, (u, v) ->
                for (axis in 0 until 3) {
                    quadVertices[k * 18 + corner * 3 + axis] = s.quads[at + axis] + u * s.quads[at + 3 + axis] + v * s.quads[at + 6 + axis]
                }
            }
        }
        quadIds = keep(DeviceMemory.upload(sceneQuads))
        val sets = listOf(triangleVertices, quadVertices).filter { it.isNotEmpty() }
        if (sets.isEmpty()) throw OptixUnsupportedScene("the scene has nothing the OptiX engine can trace")
        quadInput = if (quadVertices.isEmpty()) -1 else sets.size - 1
        tree = OptixAccel(layout, context).triangles(sets)
    }

    private fun releaseScene() {
        tree?.let { DeviceMemory.free(it.memory) }
        tree = null
        sceneMemory.forEach { DeviceMemory.free(it) }
        sceneMemory.clear()
    }

    /** The linked pipeline for [mode], built on first use. */
    private fun pipeline(mode: Int): Pipeline = pipelines.getOrPut(mode) {
        val start = System.nanoTime()
        // -Drt.cuda.options adds compiler flags for experiments, space separated.
        val extra = System.getProperty("rt.cuda.options")?.split(' ')?.filter { it.isNotBlank() } ?: emptyList()
        val options = BUILD_OPTIONS + defines.split(' ').filter { it.isNotBlank() } +
            listOf("-DAOV_MODE=$mode", "-DRT_OPTIX=1", OptixApi.includeOption()) + extra
        // -Drt.optix.ptx hands OptiX PTX, the older form, instead of OptiX IR.
        val program = cuda.buildOptixProgram(GpuRenderer.cudaKernelSource(), options, ptx = System.getProperty("rt.optix.ptx") != null)
        val compiled = System.nanoTime()
        val moduleOptions = layout.struct("OptixModuleCompileOptions")
        val pipelineOptions = layout.struct("OptixPipelineCompileOptions")
            .int("traversableGraphFlags", layout.constant("OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS").toInt())
            .int("numPayloadValues", PAYLOAD_VALUES)
            .int("numAttributeValues", 2)
            .long("pipelineLaunchParamsVariableName", name("params"))
            .int("usesPrimitiveTypeFlags", layout.constant("OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE").toInt())
        val descriptions = layout.struct("OptixProgramGroupDesc", GROUPS)
        val groupOptions = layout.struct("OptixProgramGroupOptions")
        val link = layout.struct("OptixPipelineLinkOptions").int("maxTraceDepth", 1)
        val stack = layout.struct("OptixStackSizes")
        val groupHandles = MemoryUtil.memAllocPointer(GROUPS)
        var module = 0L
        val groups = LongArray(GROUPS)
        var linked = 0L
        try {
            module = out { address ->
                OptixApi.callWithLog("optixModuleCreate") { log, logSize ->
                    arrayOf(context, moduleOptions.address, pipelineOptions.address, MemoryUtil.memAddress(program), program.remaining().toLong(), log, logSize, address)
                }
            }
            descriptions.int("kind", layout.constant("OPTIX_PROGRAM_GROUP_KIND_RAYGEN").toInt(), RAYGEN)
                .long("raygen.module", module, RAYGEN).long("raygen.entryFunctionName", name("__raygen__render"), RAYGEN)
            descriptions.int("kind", layout.constant("OPTIX_PROGRAM_GROUP_KIND_MISS").toInt(), MISS)
                .long("miss.module", module, MISS).long("miss.entryFunctionName", name("__miss__nothing"), MISS)
            descriptions.int("kind", layout.constant("OPTIX_PROGRAM_GROUP_KIND_HITGROUP").toInt(), HIT)
                .long("hitgroup.moduleCH", module, HIT).long("hitgroup.entryFunctionNameCH", name("__closesthit__surface"), HIT)
            OptixApi.callWithLog("optixProgramGroupCreate") { log, logSize ->
                arrayOf(context, descriptions.address, GROUPS, groupOptions.address, log, logSize, MemoryUtil.memAddress(groupHandles))
            }
            for (i in 0 until GROUPS) groups[i] = groupHandles.get(i)
            linked = out { address ->
                OptixApi.callWithLog("optixPipelineCreate") { log, logSize ->
                    arrayOf(context, pipelineOptions.address, link.address, MemoryUtil.memAddress(groupHandles), GROUPS, log, logSize, address)
                }
            }

            // The stack a launch needs: the ray generation program's, and under it whichever of
            // the programs a trace can call needs most. Rays are only traced from ray generation.
            val needs = HashMap<String, Int>()
            for (group in groups) {
                OptixApi.call("optixProgramGroupGetStackSize", group, stack.address, linked)
                for (field in STACK_FIELDS) needs[field] = maxOf(needs[field] ?: 0, stack.readInt(field))
            }
            fun need(field: String) = needs[field] ?: 0
            val continuation = need("cssRG") + maxOf(maxOf(need("cssCH"), need("cssMS")), need("cssIS") + need("cssAH"))
            // A tree of the scene alone: rays cross one level.
            OptixApi.call("optixPipelineSetStackSize", linked, 0, 0, continuation, 1)

            // The table: one record each for ray generation and miss, and one per input of the
            // tree for hits - all served by the same hit program. The records carry no data.
            val header = layout.constant("OPTIX_SBT_RECORD_HEADER_SIZE").toInt()
            val hits = 2
            val host = MemoryUtil.memCalloc(header * (2 + hits))
            val records = try {
                val base = MemoryUtil.memAddress(host)
                OptixApi.call("optixSbtRecordPackHeader", groups[RAYGEN], base)
                OptixApi.call("optixSbtRecordPackHeader", groups[MISS], base + header)
                for (h in 0 until hits) OptixApi.call("optixSbtRecordPackHeader", groups[HIT], base + header * (2L + h))
                DeviceMemory.upload(host)
            } finally {
                MemoryUtil.memFree(host)
            }
            val sbt = layout.struct("OptixShaderBindingTable")
                .long("raygenRecord", records)
                .long("missRecordBase", records + header).int("missRecordStrideInBytes", header).int("missRecordCount", 1)
                .long("hitgroupRecordBase", records + header * 2L).int("hitgroupRecordStrideInBytes", header).int("hitgroupRecordCount", hits)
            System.err.println(
                "[gpu] built the ${GpuSession.MODE_NAMES[mode]} pipeline for ${cuda.deviceName} (OptiX) in " +
                    "${(System.nanoTime() - start) / 1_000_000} ms, of it NVRTC ${(compiled - start) / 1_000_000} ms"
            )
            Pipeline(module, groups, linked, records, sbt)
        } catch (t: Throwable) {
            if (linked != 0L) OptixApi.call("optixPipelineDestroy", linked)
            groups.filter { it != 0L }.forEach { OptixApi.call("optixProgramGroupDestroy", it) }
            if (module != 0L) OptixApi.call("optixModuleDestroy", module)
            throw t
        } finally {
            MemoryUtil.memFree(program)
            MemoryUtil.memFree(groupHandles)
            listOf(moduleOptions, pipelineOptions, descriptions, groupOptions, link, stack).forEach { it.close() }
        }
    }

    private fun releasePipelines() {
        pipelines.values.forEach { p ->
            OptixApi.call("optixPipelineDestroy", p.pipeline)
            p.groups.forEach { OptixApi.call("optixProgramGroupDestroy", it) }
            OptixApi.call("optixModuleDestroy", p.module)
            DeviceMemory.free(p.records)
            p.sbt.close()
        }
        pipelines.clear()
    }

    /** The address of [text] as a C string that lives as long as the session. */
    private fun name(text: String): Long = MemoryUtil.memAddress(names.getOrPut(text) { MemoryUtil.memUTF8(text) })

    /** Runs [call] with the address of a pointer-sized cell and returns what it left there. */
    private inline fun out(call: (address: Long) -> Unit): Long {
        val cell = MemoryUtil.memAllocLong(1).put(0, 0L)
        try {
            call(MemoryUtil.memAddress(cell))
            return cell.get(0)
        } finally {
            MemoryUtil.memFree(cell)
        }
    }

    private companion object {
        /** The kernel's Params: 24 pointers and 64-bit numbers, 9 numbers of 32 bits, padded to a multiple of 8. */
        const val PARAMS_PADDING = 4
        const val PARAMS_BYTES = 24 * 8 + 9 * 4 + PARAMS_PADDING
        /** What a hit hands back: kind, index, distance and the two surface coordinates. */
        const val PAYLOAD_VALUES = 5
        const val GROUPS = 3
        const val RAYGEN = 0
        const val MISS = 1
        const val HIT = 2
        val STACK_FIELDS = listOf("cssRG", "cssMS", "cssCH", "cssAH", "cssIS", "cssCC", "dssDC")
        /** A quad as two triangles, in units of its two edges: Q, Q+u, Q+u+v and Q, Q+u+v, Q+v. */
        val QUAD_CORNERS = listOf(0f to 0f, 1f to 0f, 1f to 1f, 0f to 0f, 1f to 1f, 0f to 1f)
        // The CUDA kernel's options, without its register cap: OptiX finishes the code itself.
        val BUILD_OPTIONS = listOf("--use_fast_math", "--std=c++17")
    }
}
