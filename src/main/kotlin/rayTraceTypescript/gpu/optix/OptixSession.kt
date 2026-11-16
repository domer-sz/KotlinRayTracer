package rayTraceTypescript.gpu.optix

import org.lwjgl.cuda.CU.cuCtxSynchronize
import org.lwjgl.cuda.CU.cuMemcpyDtoH
import org.lwjgl.cuda.CU.cuMemcpyHtoD
import org.lwjgl.cuda.CU.cuMemsetD32
import org.lwjgl.system.MemoryUtil
import rayTraceTypescript.gpu.CudaContext
import rayTraceTypescript.gpu.DeviceMemory
import rayTraceTypescript.gpu.Engine
import rayTraceTypescript.gpu.GpuDevice
import rayTraceTypescript.gpu.GpuKeepAwake
import rayTraceTypescript.gpu.GpuPick
import rayTraceTypescript.gpu.GpuRenderer
import rayTraceTypescript.gpu.GpuScene
import rayTraceTypescript.gpu.GpuSession
import rayTraceTypescript.gpu.InstancedScene
import rayTraceTypescript.gpu.MeshBlock
import rayTraceTypescript.gpu.SceneBuffers
import rayTraceTypescript.render.CameraSetup
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.Collections
import java.util.IdentityHashMap

/**
 * The OptiX backend of [GpuSession]: the kernel CudaSession runs, built as an OptiX program
 * (-DRT_OPTIX=1), with the scene's surfaces in trees OptiX builds and searches - with the card's
 * RT cores where it has them. Sampling, materials and textures are the same code.
 *
 * The scene comes as [InstancedScene]: each placed mesh is a tree of its own, built on the card
 * the first time a scene holds that mesh and kept while scenes go on holding it, and the scene is
 * a small tree of instances above them - one per placed mesh, where it stands, and one for the
 * scene's loose triangles and quads. Moving, turning or resizing a model, a film's every frame,
 * then rebuilds only that small tree.
 *
 * As yet it takes triangles, quads and placed meshes; a scene with spheres or fog is refused
 * with the reason.
 */
internal class OptixSession(initial: GpuScene, private val maxPixels: Int, device: GpuDevice) : GpuSession {
    private val keepAwake = GpuKeepAwake.startIfUseful()
    private val cuda = CudaContext.create(device)
    override val devices: List<String> = listOf(GpuPick(device, Engine.OPTIX).toString())
    override var seed: Long = GpuSession.initialSeed()

    /** Device memory kept for the session's life, and the scene's, freed when it is replaced. */
    private val sessionMemory = mutableListOf<Long>()
    private val sceneMemory = mutableListOf<Long>()
    /** C strings OptiX is handed by address: they must outlive the calls that name them. */
    private val names = HashMap<String, ByteBuffer>()
    private val pipelines = HashMap<Int, Pipeline>()
    private val paramsHost: ByteBuffer = MemoryUtil.memCalloc(PARAMS_BYTES)
    /** Host memory reads come back through, sized for the largest output. */
    private val stagingMemory = DeviceMemory.allocateHost(maxOf(maxPixels * 3, GpuSession.CAMERA_FLOATS))
    private val staging: FloatBuffer = stagingMemory.floats

    private lateinit var layout: OptixLayout
    private lateinit var accel: OptixAccel
    private var context = 0L
    private var cameraBuffer = 0L
    private var paramsBuffer = 0L
    private var outputs: List<Long> = emptyList()

    /** A placed mesh on this card: its triangles as loaded, its faces' palette entries, and its tree. */
    private class BlockOnCard(val vertices: Long, val faces: Long, val normals: Long, val tree: OptixAccel.Tree)
    private val blocks = IdentityHashMap<MeshBlock, BlockOnCard>()

    private var scene: InstancedScene = initial.instanced
    /** The feature switches the pipelines were built with. */
    private var defines = definesFor(scene)
    private var sceneArguments: List<Long> = emptyList()
    private var loose: OptixAccel.Tree? = null
    private var top: OptixAccel.Tree? = null
    /** The tree over the scene's spheres, by their boxes, and the sphere each of its primitives stands for. */
    private var spheres: OptixAccel.Tree? = null
    private var sphereIds = 0L
    private var quadIds = 0L
    private var instanceTable = 0L
    private var palette = 0L
    private var quadInput = -1

    /** One mode's programs, linked, with the table that tells OptiX which program serves what. */
    private class Pipeline(val module: Long, val groups: LongArray, val pipeline: Long, val records: Long, val sbt: OptixStruct)

    init {
        try {
            OptixApi.unavailabilityReason()?.let { throw OptixException(it) }
            layout = OptixLayout.of(cuda)
            context = createContext()
            accel = OptixAccel(layout, context)
            cameraBuffer = DeviceMemory.allocate(GpuSession.CAMERA_FLOATS * 4L).also { sessionMemory += it }
            paramsBuffer = DeviceMemory.allocate(PARAMS_BYTES.toLong()).also { sessionMemory += it }
            // Read as well as written: every slice of samples after the first adds to it.
            outputs = List(GpuSession.TARGETS) { DeviceMemory.allocate(maxPixels * 3L * Float.SIZE_BYTES).also { sessionMemory += it } }
            load(scene)
        } catch (t: Throwable) {
            close()
            throw t
        }
    }

    override fun replaceScene(next: GpuScene) {
        cuda.makeCurrent()
        CudaContext.check(cuCtxSynchronize(), "cuCtxSynchronize")
        releaseScene()
        scene = next.instanced
        load(scene)
        val nextDefines = definesFor(scene)
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
        paramsHost.putLong(outputs[target]).putLong(outputs[GpuSession.NORMAL]).putLong(quadIds).putLong(sphereIds).putLong(instanceTable).putLong(palette)
            .putLong(top!!.handle).putLong(seed)
        // The root node is the CUDA kernel's: OptiX searches its own trees.
        paramsHost.putInt(setup.imageWidth).putInt(setup.imageHeight).putInt(sampleStart).putInt(count)
            .putInt(setup.maxReflectionDepth).putFloat(scale).putInt(0).putInt(scene.lightCount).putInt(quadInput)
            .putFloat(if (mode == GpuSession.MODE_IMAGE) setup.fireflyClamp else 0f).putInt(scene.mediumCount)
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

    override fun clear(target: Int) {
        cuda.makeCurrent()
        // All bits zero is 0.0f.
        CudaContext.check(cuMemsetD32(outputs[target], 0, maxPixels * 3L), "cuMemsetD32")
    }

    override fun prepare(mode: Int) {
        cuda.makeCurrent()
        pipeline(mode)
    }

    override fun close() {
        try {
            cuda.makeCurrent()
            releasePipelines()
            releaseScene()
            blocks.values.forEach { release(it) }
            blocks.clear()
            sessionMemory.forEach { DeviceMemory.free(it) }
            if (context != 0L) OptixApi.call("optixDeviceContextDestroy", context)
        } finally {
            names.values.forEach { MemoryUtil.memFree(it) }
            MemoryUtil.memFree(paramsHost)
            stagingMemory.free()
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

    /** Uploads [s], builds the trees of meshes new to this card, and the tree of the scene above them. */
    private fun load(s: InstancedScene) {
        unsupported(s)?.let { throw OptixUnsupportedScene(it) }
        fun keep(pointer: Long) = pointer.also { sceneMemory += it }

        // The meshes: a mesh no scene holds any longer goes, one new to this card gets its tree.
        val held: MutableSet<MeshBlock> = Collections.newSetFromMap(IdentityHashMap())
        s.instances.forEach { held += it.block }
        blocks.keys.filter { it !in held }.forEach { gone -> blocks.remove(gone)?.let { release(it) } }
        held.filter { it !in blocks }.forEach { blocks[it] = upload(it) }

        // The kernel's buffers, in the order of its Params. The node buffers carry the trees of
        // the volumes' boundaries, the only ones the kernel searches itself.
        sceneArguments = listOf(
            cameraBuffer,
            keep(DeviceMemory.upload(s.boundaryNodes)), keep(DeviceMemory.upload(s.boundaryLeaves)), keep(DeviceMemory.upload(s.boundaryRefs)),
            keep(DeviceMemory.upload(s.spheres)), keep(DeviceMemory.upload(s.sphereMaterials)),
            keep(DeviceMemory.upload(s.triangles)), keep(DeviceMemory.upload(s.triangleMaterials)),
            keep(DeviceMemory.upload(s.quads)), keep(DeviceMemory.upload(s.quadMaterials)),
            keep(DeviceMemory.upload(s.mediumInts)), keep(DeviceMemory.upload(s.mediumFloats)), keep(DeviceMemory.upload(s.lights)),
            keep(DeviceMemory.upload(s.materialInts)), keep(DeviceMemory.upload(s.materialFloats)),
            keep(DeviceMemory.upload(s.textureInts)), keep(DeviceMemory.upload(s.textureFloats)),
            keep(DeviceMemory.upload(s.textureImages)),
            keep(DeviceMemory.upload(s.perlinVectors)), keep(DeviceMemory.upload(s.perlinPermutations)),
            keep(DeviceMemory.upload(s.triangleNormals))
        )

        // The loose surfaces: the scene's own triangles - as the kernel keeps them, first vertex
        // and two edges, turned into three vertices - and its quads, two triangles each. The
        // tables hold more (the lights' stand-ins, the volumes' boundaries), which no ray may hit.
        val stride = SceneBuffers.TRIANGLE_STRIDE
        val triangleVertices = FloatArray(s.sceneTriangles * 9)
        for (t in 0 until s.sceneTriangles) {
            for (axis in 0 until 3) {
                val v0 = s.triangles[t * stride + axis]
                triangleVertices[t * 9 + axis] = v0
                triangleVertices[t * 9 + 3 + axis] = v0 + s.triangles[t * stride + 4 + axis]
                triangleVertices[t * 9 + 6 + axis] = v0 + s.triangles[t * stride + 8 + axis]
            }
        }
        val quadVertices = FloatArray(s.sceneQuads.size * 18)
        s.sceneQuads.forEachIndexed { k, quad ->
            val at = quad * SceneBuffers.QUAD_STRIDE
            // Corner Q, then Q + u, Q + u + v and Q + v.
            QUAD_CORNERS.forEachIndexed { corner, (u, v) ->
                for (axis in 0 until 3) {
                    quadVertices[k * 18 + corner * 3 + axis] = s.quads[at + axis] + u * s.quads[at + 3 + axis] + v * s.quads[at + 6 + axis]
                }
            }
        }
        quadIds = keep(DeviceMemory.upload(s.sceneQuads))
        val sets = listOf(triangleVertices, quadVertices).filter { it.isNotEmpty() }
        quadInput = if (quadVertices.isEmpty()) -1 else sets.size - 1
        loose = if (sets.isEmpty()) null else accel.triangles(sets)

        // The spheres a ray may hit - not the volumes' boundaries or the lights' stand-ins - by
        // their boxes, which take in a moving sphere's whole path: where it starts and where it ends.
        val boxes = FloatArray(s.sceneSpheres.size * 6)
        s.sceneSpheres.forEachIndexed { k, sphere ->
            val at = sphere * SceneBuffers.SPHERE_STRIDE
            val r = s.spheres[at + 6]
            for (axis in 0 until 3) {
                val start = s.spheres[at + axis]
                val end = start + s.spheres[at + 3 + axis]
                boxes[k * 6 + axis] = minOf(start, end) - r
                boxes[k * 6 + 3 + axis] = maxOf(start, end) + r
            }
        }
        sphereIds = keep(DeviceMemory.upload(s.sceneSpheres))
        spheres = if (boxes.isEmpty()) null else accel.customPrimitives(boxes)

        // What shading needs to know of each placed mesh (the kernel's Instance), and the
        // materials of the meshes' own palettes, one after another.
        val paletteEntries = ArrayList<Int>()
        val table = MemoryUtil.memCalloc(INSTANCE_BYTES * maxOf(s.instances.size, 1))
        try {
            s.instances.forEach { instance ->
                val onCard = blocks.getValue(instance.block)
                val own = instance.palette ?: IntArray(0)
                table.putLong(onCard.vertices).putLong(if (own.isEmpty()) 0L else onCard.faces).putLong(onCard.normals)
                    .putInt(instance.material).putInt(paletteEntries.size).putInt(own.size)
                    .putFloat(instance.cos).putFloat(instance.sin).putInt(0)
                own.forEach { paletteEntries += it }
            }
            instanceTable = keep(DeviceMemory.upload(table))
        } finally {
            MemoryUtil.memFree(table)
        }
        palette = keep(DeviceMemory.upload(paletteEntries.toIntArray()))

        // The tree of the scene: one instance per placed mesh, where it stands, hits going to
        // the mesh program (the first hit record); one for the loose surfaces, unmoved, whose two
        // inputs take the records after it; and one for the spheres, whose record is the last.
        val count = s.instances.size + (if (loose != null) 1 else 0) + (if (spheres != null) 1 else 0)
        layout.struct("OptixInstance", count).use { records ->
            s.instances.forEachIndexed { i, instance ->
                // A point p of the mesh goes to scale * turn(p) + offset, turning as PlacedMesh does:
                // x' = (c x + s z) k + o.x, y' = k y + o.y, z' = (-s x + c z) k + o.z.
                val k = instance.scale
                val c = instance.cos
                val n = instance.sin
                val o = instance.offset
                val matrix = floatArrayOf(c * k, 0f, n * k, o.x, 0f, k, 0f, o.y, -n * k, 0f, c * k, o.z)
                matrix.forEachIndexed { e, value -> records.float("transform", value, i, e) }
                records.int("instanceId", i, i).int("sbtOffset", 0, i).int("visibilityMask", 255, i)
                    .int("flags", layout.constant("OPTIX_INSTANCE_FLAG_NONE").toInt(), i)
                    .long("traversableHandle", blocks.getValue(instance.block).tree.handle, i)
            }
            loose?.let { tree ->
                val i = s.instances.size
                IDENTITY.forEachIndexed { e, value -> records.float("transform", value, i, e) }
                records.int("instanceId", i, i).int("sbtOffset", 1, i).int("visibilityMask", 255, i)
                    .int("flags", layout.constant("OPTIX_INSTANCE_FLAG_NONE").toInt(), i)
                    .long("traversableHandle", tree.handle, i)
            }
            spheres?.let { tree ->
                val i = count - 1
                IDENTITY.forEachIndexed { e, value -> records.float("transform", value, i, e) }
                records.int("instanceId", i, i).int("sbtOffset", HIT_SPHERE_RECORD, i).int("visibilityMask", 255, i)
                    .int("flags", layout.constant("OPTIX_INSTANCE_FLAG_NONE").toInt(), i)
                    .long("traversableHandle", tree.handle, i)
            }
            val onCard = keep(DeviceMemory.upload(records.memory))
            top = accel.instances(onCard, count)
        }
    }

    /** A mesh's triangles, as loaded and in the file's order, and a tree over them, on this card. */
    private fun upload(block: MeshBlock): BlockOnCard {
        val start = System.nanoTime()
        val mesh = block.mesh
        // The mesh's own order, not the CUDA kernel's leaf order: OptiX numbers the triangles as given.
        val vertices = DeviceMemory.upload(mesh.vertices)
        var faces = 0L
        var normals = 0L
        try {
            faces = mesh.faceMaterials?.takeIf { mesh.hasColours }?.let { DeviceMemory.upload(it) } ?: 0L
            // A smooth mesh's corner normals, in the same order and in its own space.
            normals = if (mesh.smooth) DeviceMemory.upload(mesh.smoothNormals) else 0L
            val tree = accel.trianglesOnCard(listOf(vertices to block.triangleCount))
            System.err.println(
                "[gpu] built the tree of a ${block.triangleCount}-triangle mesh on ${cuda.deviceName} (OptiX) in " +
                    "${(System.nanoTime() - start) / 1_000_000} ms, ${tree.bytes / (1 shl 20)} MB" + if (mesh.smooth) ", shaded smooth" else ""
            )
            return BlockOnCard(vertices, faces, normals, tree)
        } catch (t: Throwable) {
            DeviceMemory.free(vertices)
            DeviceMemory.free(faces)
            DeviceMemory.free(normals)
            throw t
        }
    }

    private fun release(block: BlockOnCard) {
        DeviceMemory.free(block.tree.memory)
        DeviceMemory.free(block.vertices)
        DeviceMemory.free(block.faces)
        DeviceMemory.free(block.normals)
    }

    private fun releaseScene() {
        loose?.let { DeviceMemory.free(it.memory) }
        loose = null
        spheres?.let { DeviceMemory.free(it.memory) }
        spheres = null
        top?.let { DeviceMemory.free(it.memory) }
        top = null
        sceneMemory.forEach { DeviceMemory.free(it) }
        sceneMemory.clear()
    }

    /** The linked pipeline for [mode], built on first use. */
    private fun pipeline(mode: Int): Pipeline = pipelines.getOrPut(mode) {
        val start = System.nanoTime()
        // -Drt.cuda.options adds compiler flags for experiments, space separated.
        val extra = System.getProperty("rt.cuda.options")?.split(' ')?.filter { it.isNotBlank() } ?: emptyList()
        val options = BUILD_OPTIONS + defines.split(' ').filter { it.isNotBlank() } +
            listOf("-DAOV_MODE=${GpuSession.aovMode(mode)}", "-DGUIDES=${if (GpuSession.gathersGuides(mode)) 1 else 0}", "-DRT_OPTIX=1", OptixApi.includeOption()) + extra
        // -Drt.optix.ptx hands OptiX PTX, the older form, instead of OptiX IR.
        val program = cuda.buildOptixProgram(GpuRenderer.cudaKernelSource(), options, ptx = System.getProperty("rt.optix.ptx") != null)
        val compiled = System.nanoTime()
        // Spheres take a program group of their own, and OptiX must be told it meets primitives
        // other than triangles; a scene without them keeps the pipeline it always had.
        val traceSpheres = defines.contains(SPHERES_DEFINE)
        val groupCount = if (traceSpheres) GROUPS_WITH_SPHERES else GROUPS
        val primitiveTypes = layout.constant("OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE") or
            (if (traceSpheres) layout.constant("OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM") else 0L)
        val moduleOptions = layout.struct("OptixModuleCompileOptions")
        val pipelineOptions = layout.struct("OptixPipelineCompileOptions")
            .int("traversableGraphFlags", layout.constant("OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING").toInt())
            .int("numPayloadValues", PAYLOAD_VALUES)
            .int("numAttributeValues", 2)
            .long("pipelineLaunchParamsVariableName", name("params"))
            .int("usesPrimitiveTypeFlags", primitiveTypes.toInt())
        val descriptions = layout.struct("OptixProgramGroupDesc", groupCount)
        val groupOptions = layout.struct("OptixProgramGroupOptions")
        val link = layout.struct("OptixPipelineLinkOptions").int("maxTraceDepth", 1)
        val stack = layout.struct("OptixStackSizes")
        val groupHandles = MemoryUtil.memAllocPointer(groupCount)
        var module = 0L
        val groups = LongArray(groupCount)
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
            descriptions.int("kind", layout.constant("OPTIX_PROGRAM_GROUP_KIND_HITGROUP").toInt(), HIT_MESH)
                .long("hitgroup.moduleCH", module, HIT_MESH).long("hitgroup.entryFunctionNameCH", name("__closesthit__mesh"), HIT_MESH)
            descriptions.int("kind", layout.constant("OPTIX_PROGRAM_GROUP_KIND_HITGROUP").toInt(), HIT_SURFACE)
                .long("hitgroup.moduleCH", module, HIT_SURFACE).long("hitgroup.entryFunctionNameCH", name("__closesthit__surface"), HIT_SURFACE)
            if (traceSpheres) {
                descriptions.int("kind", layout.constant("OPTIX_PROGRAM_GROUP_KIND_HITGROUP").toInt(), HIT_SPHERE)
                    .long("hitgroup.moduleIS", module, HIT_SPHERE).long("hitgroup.entryFunctionNameIS", name("__intersection__sphere"), HIT_SPHERE)
                    .long("hitgroup.moduleCH", module, HIT_SPHERE).long("hitgroup.entryFunctionNameCH", name("__closesthit__sphere"), HIT_SPHERE)
            }
            OptixApi.callWithLog("optixProgramGroupCreate") { log, logSize ->
                arrayOf(context, descriptions.address, groupCount, groupOptions.address, log, logSize, MemoryUtil.memAddress(groupHandles))
            }
            for (i in 0 until groupCount) groups[i] = groupHandles.get(i)
            linked = out { address ->
                OptixApi.callWithLog("optixPipelineCreate") { log, logSize ->
                    arrayOf(context, pipelineOptions.address, link.address, MemoryUtil.memAddress(groupHandles), groupCount, log, logSize, address)
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
            // Instances over meshes: rays cross two levels.
            OptixApi.call("optixPipelineSetStackSize", linked, 0, 0, continuation, 2)

            // The table: one record each for ray generation and miss; for hits, the mesh program's
            // record, then the surface program's for each input of the loose surfaces' tree, then
            // the spheres' (HIT_SPHERE_RECORD). The records carry no data.
            val header = layout.constant("OPTIX_SBT_RECORD_HEADER_SIZE").toInt()
            val hits = if (traceSpheres) HIT_SPHERE_RECORD + 1 else HIT_SPHERE_RECORD
            val host = MemoryUtil.memCalloc(header * (2 + hits))
            val records = try {
                val base = MemoryUtil.memAddress(host)
                OptixApi.call("optixSbtRecordPackHeader", groups[RAYGEN], base)
                OptixApi.call("optixSbtRecordPackHeader", groups[MISS], base + header)
                OptixApi.call("optixSbtRecordPackHeader", groups[HIT_MESH], base + header * 2L)
                for (h in 1 until HIT_SPHERE_RECORD) OptixApi.call("optixSbtRecordPackHeader", groups[HIT_SURFACE], base + header * (2L + h))
                if (traceSpheres) OptixApi.call("optixSbtRecordPackHeader", groups[HIT_SPHERE], base + header * (2L + HIT_SPHERE_RECORD))
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

    companion object {
        /** Why the OptiX engine cannot trace [scene] - what it holds that is not built yet - or null when it can. */
        /** The scene's feature switches, and whether OptiX meets spheres in it: that changes the pipeline. */
        private fun definesFor(scene: InstancedScene): String =
            scene.kernelDefines() + if (scene.sceneSpheres.isNotEmpty()) " $SPHERES_DEFINE" else ""

        fun unsupported(scene: InstancedScene): String? = when {
            scene.sceneTriangles == 0 && scene.sceneQuads.isEmpty() && scene.sceneSpheres.isEmpty() && scene.instances.isEmpty() ->
                // Fog alone leaves OptiX no tree to build; the CUDA kernel takes such a scene.
                "the scene has no surface for OptiX's tree"
            else -> null
        }

        /** The kernel's Params: 29 pointers and 64-bit numbers, then 11 numbers of 32 bits, padded to a multiple of 8 as the compiler pads it. */
        private const val PARAMS_PADDING = 4
        private const val PARAMS_BYTES = 29 * 8 + 11 * 4 + PARAMS_PADDING
        /** The kernel's Instance: three pointers, then six numbers of 32 bits. */
        private const val INSTANCE_BYTES = 3 * 8 + 6 * 4
        /** What a hit hands back: kind, index, distance, the two surface coordinates and the instance; and the ray's time, which it is handed. */
        private const val PAYLOAD_VALUES = 7
        private const val GROUPS = 4
        private const val GROUPS_WITH_SPHERES = 5
        private const val RAYGEN = 0
        private const val MISS = 1
        private const val HIT_MESH = 2
        private const val HIT_SURFACE = 3
        private const val HIT_SPHERE = 4
        /** The spheres' hit record, after the mesh's and the two of the loose surfaces' inputs. */
        private const val HIT_SPHERE_RECORD = 3
        /** Added to the scene's feature switches when it has spheres to hit, so their pipeline is built anew. */
        private const val SPHERES_DEFINE = "-DOPTIX_SPHERES=1"
        private val STACK_FIELDS = listOf("cssRG", "cssMS", "cssCH", "cssAH", "cssIS", "cssCC", "dssDC")
        /** A quad as two triangles, in units of its two edges: Q, Q+u, Q+u+v and Q, Q+u+v, Q+v. */
        private val QUAD_CORNERS = listOf(0f to 0f, 1f to 0f, 1f to 1f, 0f to 0f, 1f to 1f, 0f to 1f)
        /** A 3x4 matrix, rows first, that leaves everything where it is. */
        private val IDENTITY = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f)
        // The CUDA kernel's options, without its register cap: OptiX finishes the code itself.
        private val BUILD_OPTIONS = listOf("--use_fast_math", "--std=c++17")
    }
}
