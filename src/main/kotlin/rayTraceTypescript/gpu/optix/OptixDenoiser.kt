package rayTraceTypescript.gpu.optix

import org.lwjgl.cuda.CU.cuCtxSynchronize
import org.lwjgl.cuda.CU.cuMemcpyDtoH
import org.lwjgl.cuda.CU.cuMemcpyHtoD
import org.lwjgl.system.MemoryUtil
import rayTraceTypescript.gpu.CudaContext
import rayTraceTypescript.gpu.DeviceMemory
import rayTraceTypescript.gpu.GpuDevice
import rayTraceTypescript.render.ImageDenoiser
import java.nio.FloatBuffer

/**
 * NVIDIA's OptiX denoiser on one card: the image and its two guides go up to the card, the
 * built-in AOV model - the one for single images - cleans the image, and it comes back.
 *
 * The model's weights come with the driver, like the rest of OptiX. It wants the normal guide
 * in world space on current drivers, which is what the kernel's normal guide holds, and works
 * out the image's average colour itself when it is not told one.
 *
 * It owns a CUDA context on [device] and makes it current on whichever thread calls it, so it
 * can work beside a session that traces on another card - or on the same one.
 */
class OptixDenoiser(val device: GpuDevice) : ImageDenoiser {
    override val name: String = "OptiX (${device.shortName})"

    private val cuda = CudaContext.create(device)
    private lateinit var layout: OptixLayout
    private var context = 0L
    private var denoiser = 0L

    // Set up for one size and kept for the next call of the same size.
    private var width = 0
    private var height = 0
    private var state = 0L
    private var stateBytes = 0L
    private var scratch = 0L
    private var scratchBytes = 0L
    /** The image, its albedo and normal guides and the result, on the card. */
    private var images = LongArray(0)
    private var staging: FloatBuffer? = null

    /** How long the last call took for each of its parts, in milliseconds, for a log. */
    var lastTimes: Times? = null
        private set

    /** Where a call's time went: copying up, the denoiser itself, copying back. */
    data class Times(val uploadMs: Double, val denoiseMs: Double, val downloadMs: Double, val setupMs: Double)

    init {
        try {
            OptixApi.unavailabilityReason()?.let { throw OptixException(it) }
            layout = OptixLayout.of(cuda)
            context = layout.struct("OptixDeviceContextOptions").use { options ->
                options.long("logCallbackFunction", OptixApi.logCallback).int("logCallbackLevel", OptixApi.logLevel())
                out { OptixApi.call("optixDeviceContextCreate", cuda.context, options.address, it) }
            }
            denoiser = layout.struct("OptixDenoiserOptions").use { options ->
                options.int("guideAlbedo", 1).int("guideNormal", 1)
                    .int("denoiseAlpha", layout.constant("OPTIX_DENOISER_ALPHA_MODE_COPY").toInt())
                out {
                    OptixApi.call("optixDenoiserCreate", context, layout.constant("OPTIX_DENOISER_MODEL_KIND_AOV").toInt(), options.address, it)
                }
            }
        } catch (t: Throwable) {
            close()
            throw t
        }
    }

    @Synchronized
    override fun denoise(width: Int, height: Int, color: FloatArray, albedo: FloatArray, normal: FloatArray): FloatArray {
        cuda.makeCurrent()
        val setupStart = System.nanoTime()
        prepare(width, height)
        val uploadStart = System.nanoTime()
        val host = staging!!
        listOf(color, albedo, normal).forEachIndexed { i, data ->
            host.clear()
            host.put(data, 0, width * height * 3).flip()
            CudaContext.check(cuMemcpyHtoD(images[i], host), "cuMemcpyHtoD(denoiser input)")
        }
        val denoiseStart = System.nanoTime()
        val params = layout.struct("OptixDenoiserParams")
        val guides = layout.struct("OptixDenoiserGuideLayer")
        val layers = layout.struct("OptixDenoiserLayer")
        try {
            // Average colour and intensity left at null: the denoiser works them out for the image.
            image(guides, "albedo", images[ALBEDO], width, height)
            image(guides, "normal", images[NORMAL], width, height)
            image(layers, "input", images[COLOR], width, height)
            image(layers, "output", images[OUTPUT], width, height)
            layers.int("type", layout.constant("OPTIX_DENOISER_AOV_TYPE_BEAUTY").toInt())
            OptixApi.call(
                "optixDenoiserInvoke", denoiser, 0L, params.address, state, stateBytes,
                guides.address, layers.address, 1, 0, 0, scratch, scratchBytes
            )
            CudaContext.check(cuCtxSynchronize(), "optixDenoiserInvoke (cuCtxSynchronize)")
        } finally {
            params.close()
            guides.close()
            layers.close()
        }
        val downloadStart = System.nanoTime()
        val result = FloatArray(width * height * 3)
        host.clear()
        host.limit(result.size)
        CudaContext.check(cuMemcpyDtoH(host, images[OUTPUT]), "cuMemcpyDtoH(denoiser output)")
        host.get(result)
        val end = System.nanoTime()
        lastTimes = Times(
            (denoiseStart - uploadStart) / 1e6, (downloadStart - denoiseStart) / 1e6, (end - downloadStart) / 1e6, (uploadStart - setupStart) / 1e6
        )
        return result
    }

    /** Readies the denoiser and the buffers for images of [width] x [height], unless they are. */
    private fun prepare(width: Int, height: Int) {
        if (width == this.width && height == this.height) return
        releaseBuffers()
        layout.struct("OptixDenoiserSizes").use { sizes ->
            OptixApi.call("optixDenoiserComputeMemoryResources", denoiser, width, height, sizes.address)
            stateBytes = sizes.readLong("stateSizeInBytes")
            scratchBytes = sizes.readLong("withoutOverlapScratchSizeInBytes")
        }
        state = DeviceMemory.allocate(stateBytes)
        scratch = DeviceMemory.allocate(scratchBytes)
        OptixApi.call("optixDenoiserSetup", denoiser, 0L, width, height, state, stateBytes, scratch, scratchBytes)
        CudaContext.check(cuCtxSynchronize(), "optixDenoiserSetup (cuCtxSynchronize)")
        images = LongArray(IMAGES) { DeviceMemory.allocate(width.toLong() * height * PIXEL_BYTES) }
        staging = MemoryUtil.memAllocFloat(width * height * 3)
        this.width = width
        this.height = height
    }

    /** Describes an image of three floats a pixel at [data] in the OptixImage2D [field] of [struct]. */
    private fun image(struct: OptixStruct, field: String, data: Long, width: Int, height: Int) {
        struct.long("$field.data", data).int("$field.width", width).int("$field.height", height)
            .int("$field.rowStrideInBytes", width * PIXEL_BYTES).int("$field.pixelStrideInBytes", PIXEL_BYTES)
            .int("$field.format", layout.constant("OPTIX_PIXEL_FORMAT_FLOAT3").toInt())
    }

    private fun releaseBuffers() {
        images.forEach { DeviceMemory.free(it) }
        images = LongArray(0)
        DeviceMemory.free(state)
        DeviceMemory.free(scratch)
        state = 0L
        scratch = 0L
        staging?.let { MemoryUtil.memFree(it) }
        staging = null
        width = 0
        height = 0
    }

    @Synchronized
    override fun close() {
        try {
            cuda.makeCurrent()
            releaseBuffers()
            if (denoiser != 0L) OptixApi.call("optixDenoiserDestroy", denoiser)
            if (context != 0L) OptixApi.call("optixDeviceContextDestroy", context)
        } finally {
            denoiser = 0L
            context = 0L
            cuda.close()
        }
    }

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
        const val COLOR = 0
        const val ALBEDO = 1
        const val NORMAL = 2
        const val OUTPUT = 3
        const val IMAGES = 4
        const val PIXEL_BYTES = 12
    }
}
