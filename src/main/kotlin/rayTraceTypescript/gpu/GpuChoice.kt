package rayTraceTypescript.gpu

/** What a card traces with: the renderer's own CUDA kernel, or OptiX and its RT cores. */
enum class Engine { CUDA, OPTIX }

/**
 * The card picked to render, and how. With [fallback], an [Engine.OPTIX] card traces a scene
 * OptiX does not take with the CUDA kernel instead of refusing it.
 */
data class GpuPick(val device: GpuDevice, val engine: Engine, val fallback: Boolean = false) {
    override fun toString(): String = "${device.shortName} (${if (engine == Engine.OPTIX) "OptiX" else "CUDA"})"
}

/**
 * Which card renders, and with which engine: what -Drt.gpu and -Drt.backend say. One card traces;
 * the same card denoises ([GpuRenderer.requestedDenoiser]); any other card in the machine stays unused.
 *
 * [gpu] is empty for the best card - the one OptiX can use, else the newest architecture - or a
 * card's number in the driver's list, or a piece of its name (`2060`, `1`).
 *
 * [backend] is `auto`, `optix` or `cuda`. `auto` is OptiX on a card with RT cores, where OptiX is
 * installed, and the CUDA kernel elsewhere.
 */
data class GpuChoice(val gpu: String = "", val backend: String = AUTO) {

    /**
     * The card among [devices] this choice means. [optixReason] is why OptiX cannot be used on
     * this machine, null when it can: `auto` then takes the CUDA kernel, `optix` is an error.
     */
    fun resolve(devices: List<GpuDevice>, optixReason: String?): GpuPick {
        val selector = gpu.trim()
        val device = if (selector.isEmpty()) best(devices, optixReason) else {
            if (devices.isEmpty()) throw CudaException("no CUDA device found")
            val number = selector.toIntOrNull()
            (if (number != null && number in devices.indices) devices[number] else devices.firstOrNull { it.name.contains(selector, ignoreCase = true) })
                ?: throw CudaException("no CUDA device matches '$selector' (found ${devices.joinToString { "${it.ordinal}: ${it.name}" }})")
        }
        return when (backend.trim().lowercase()) {
            "cuda" -> GpuPick(device, Engine.CUDA)
            "optix" -> {
                if (optixReason != null) throw CudaException("${device.shortName} cannot trace through OptiX: $optixReason")
                GpuPick(device, Engine.OPTIX)
            }
            AUTO, "" -> if (device.hasRtCores && optixReason == null) GpuPick(device, Engine.OPTIX, fallback = true) else GpuPick(device, Engine.CUDA)
            else -> throw CudaException("no engine called '$backend' (-Drt.backend is auto, optix or cuda)")
        }
    }

    companion object {
        const val AUTO = "auto"

        /** -Drt.gpu and -Drt.backend; the best card through `auto` without them. */
        fun fromProperties(): GpuChoice = GpuChoice(System.getProperty("rt.gpu") ?: "", System.getProperty("rt.backend") ?: AUTO)

        /** The card that traces best: the one OptiX can use first, then the newest architecture; the driver's order breaks ties. */
        fun best(devices: List<GpuDevice>, optixReason: String?): GpuDevice {
            if (devices.isEmpty()) throw CudaException("no CUDA device found")
            return devices.maxWith(compareBy<GpuDevice> { it.hasRtCores && optixReason == null }.thenBy { it.computeCapability }.thenBy { -it.ordinal })
        }
    }
}
