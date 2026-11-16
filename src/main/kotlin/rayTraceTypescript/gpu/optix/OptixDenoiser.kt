package rayTraceTypescript.gpu.optix

import org.lwjgl.cuda.CU.cuCtxSynchronize
import org.lwjgl.cuda.CU.cuMemcpyDtoH
import org.lwjgl.cuda.CU.cuMemcpyHtoD
import org.lwjgl.cuda.CU.cuMemsetD32
import org.lwjgl.cuda.CU.cuMemsetD8
import org.lwjgl.system.MemoryUtil
import rayTraceTypescript.gpu.CudaContext
import rayTraceTypescript.gpu.DeviceMemory
import rayTraceTypescript.gpu.GpuDevice
import rayTraceTypescript.render.ImageDenoiser
import java.nio.FloatBuffer

/**
 * NVIDIA's OptiX denoiser on one card: the image and its two guides go up to the card, the
 * built-in AOV model - the one for single images - cleans the image, and it comes back. With
 * [upscaling], the UPSCALE2X model instead cleans it and returns it twice as wide and tall.
 *
 * The model's weights come with the driver, like the rest of OptiX. It wants the normal guide
 * in world space on current drivers, which is what the kernel's normal guide holds, and works
 * out the image's average colour itself when it is not told one.
 *
 * It owns a CUDA context on [device] and makes it current on whichever thread calls it, so it
 * can work beside a session that traces on another card - or on the same one.
 */
class OptixDenoiser(val device: GpuDevice, upscaling: Boolean = false, temporal: Boolean = false) : ImageDenoiser {
    override val name: String = "OptiX (${device.shortName})" + (if (upscaling) ", 2x" else "") + (if (temporal) ", temporal" else "")
    override val upscale: Int = if (upscaling) 2 else 1
    override val temporal: Boolean = temporal

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
    /** The image, its guides, the two outputs and, for a temporal model, the two internal layers, on the card. */
    private var images = LongArray(0)
    /** Bytes a pixel of the temporal model's internal guide layers takes. */
    private var internalBytes = 0L
    /** Which output and internal layer the next call writes; the other ones hold the frame before's. */
    private var current = 0
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
                    val model = when {
                        upscaling && temporal -> "OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X"
                        upscaling -> "OPTIX_DENOISER_MODEL_KIND_UPSCALE2X"
                        temporal -> "OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV"
                        else -> "OPTIX_DENOISER_MODEL_KIND_AOV"
                    }
                    OptixApi.call("optixDenoiserCreate", context, layout.constant(model).toInt(), options.address, it)
                }
            }
        } catch (t: Throwable) {
            close()
            throw t
        }
    }

    @Synchronized
    override fun denoise(width: Int, height: Int, color: FloatArray, albedo: FloatArray, normal: FloatArray): FloatArray =
        // A temporal model on a lone image: no flow, nothing from a frame before.
        run(width, height, color, albedo, normal, null, continuing = false)

    @Synchronized
    override fun denoiseNext(
        width: Int, height: Int, color: FloatArray, albedo: FloatArray, normal: FloatArray, flow: FloatArray, continuing: Boolean
    ): FloatArray = run(width, height, color, albedo, normal, if (temporal) flow else null, continuing)

    private fun run(
        width: Int, height: Int, color: FloatArray, albedo: FloatArray, normal: FloatArray, flow: FloatArray?, continuing: Boolean
    ): FloatArray {
        cuda.makeCurrent()
        val setupStart = System.nanoTime()
        prepare(width, height)
        val uploadStart = System.nanoTime()
        val host = staging!!
        val inputs = listOf(COLOR to color, ALBEDO to albedo, NORMAL to normal) + listOfNotNull(flow?.let { FLOW to it })
        inputs.forEach { (i, data) ->
            host.clear()
            host.put(data, 0, width * height * 3).flip()
            CudaContext.check(cuMemcpyHtoD(images[i], host), "cuMemcpyHtoD(denoiser input)")
        }
        // No flow: nothing moved.
        if (temporal && flow == null) CudaContext.check(cuMemsetD32(images[FLOW], 0, width * height * 3L), "cuMemsetD32(flow)")
        val denoiseStart = System.nanoTime()
        val outWidth = width * upscale
        val outHeight = height * upscale
        // Two outputs and two internal layers, taking turns: this call's, and the frame before's.
        val output = images[OUTPUT_A + current]
        val params = layout.struct("OptixDenoiserParams")
        val guides = layout.struct("OptixDenoiserGuideLayer")
        val layers = layout.struct("OptixDenoiserLayer")
        try {
            // Average colour and intensity left at null: the denoiser works them out for the image.
            image(guides, "albedo", images[ALBEDO], width, height)
            image(guides, "normal", images[NORMAL], width, height)
            image(layers, "input", images[COLOR], width, height)
            image(layers, "output", output, outWidth, outHeight)
            layers.int("type", layout.constant("OPTIX_DENOISER_AOV_TYPE_BEAUTY").toInt())
            if (temporal) {
                // The flow as the guides pass leaves it: two floats used of every three.
                image(guides, "flow", images[FLOW], width, height, "OPTIX_PIXEL_FORMAT_FLOAT2", PIXEL_BYTES)
                val internalFormat = "OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER"
                image(guides, "previousOutputInternalGuideLayer", images[INTERNAL_A + 1 - current], outWidth, outHeight, internalFormat, internalBytes.toInt())
                image(guides, "outputInternalGuideLayer", images[INTERNAL_A + current], outWidth, outHeight, internalFormat, internalBytes.toInt())
                image(layers, "previousOutput", images[OUTPUT_A + 1 - current], outWidth, outHeight)
                params.int("temporalModeUsePreviousLayers", if (continuing) 1 else 0)
            }
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
        val result = FloatArray(outWidth * outHeight * 3)
        host.clear()
        host.limit(result.size)
        CudaContext.check(cuMemcpyDtoH(host, output), "cuMemcpyDtoH(denoiser output)")
        host.get(result)
        // This frame's output and internal layer are the next one's previous ones.
        if (temporal) current = 1 - current
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
            // Sized for the output - for the model that enlarges, twice as wide and tall as the input;
            // the setup below takes the input's size.
            OptixApi.call("optixDenoiserComputeMemoryResources", denoiser, width * upscale, height * upscale, sizes.address)
            stateBytes = sizes.readLong("stateSizeInBytes")
            scratchBytes = sizes.readLong("withoutOverlapScratchSizeInBytes")
            internalBytes = sizes.readLong("internalGuideLayerPixelSizeInBytes")
        }
        state = DeviceMemory.allocate(stateBytes)
        scratch = DeviceMemory.allocate(scratchBytes)
        OptixApi.call("optixDenoiserSetup", denoiser, 0L, width, height, state, stateBytes, scratch, scratchBytes)
        CudaContext.check(cuCtxSynchronize(), "optixDenoiserSetup (cuCtxSynchronize)")
        // The outputs and internal layers are at the output's size (the enlarged one when the model
        // enlarges), the image and its guides at the input's. A model without memory of frames
        // needs one output only.
        val outPixels = width.toLong() * height * upscale * upscale
        images = LongArray(if (temporal) IMAGES else OUTPUT_A + 1) { i ->
            when (i) {
                OUTPUT_A, OUTPUT_B -> DeviceMemory.allocate(outPixels * PIXEL_BYTES)
                INTERNAL_A, INTERNAL_B -> DeviceMemory.allocate(outPixels * internalBytes).also {
                    CudaContext.check(cuMemsetD8(it, 0, outPixels * internalBytes), "cuMemsetD8(internal layer)")
                }
                else -> DeviceMemory.allocate(width.toLong() * height * PIXEL_BYTES)
            }
        }
        current = 0
        staging = MemoryUtil.memAllocFloat(width * height * 3 * upscale * upscale)
        this.width = width
        this.height = height
    }

    /** Describes an image at [data] in the OptixImage2D [field] of [struct]: three floats a pixel unless told otherwise. */
    private fun image(
        struct: OptixStruct, field: String, data: Long, width: Int, height: Int,
        format: String = "OPTIX_PIXEL_FORMAT_FLOAT3", pixelBytes: Int = PIXEL_BYTES
    ) {
        struct.long("$field.data", data).int("$field.width", width).int("$field.height", height)
            .int("$field.rowStrideInBytes", width * pixelBytes).int("$field.pixelStrideInBytes", pixelBytes)
            .int("$field.format", layout.constant(format).toInt())
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
        const val FLOW = 3
        const val OUTPUT_A = 4
        const val OUTPUT_B = 5
        const val INTERNAL_A = 6
        const val INTERNAL_B = 7
        const val IMAGES = 8
        const val PIXEL_BYTES = 12
    }
}
