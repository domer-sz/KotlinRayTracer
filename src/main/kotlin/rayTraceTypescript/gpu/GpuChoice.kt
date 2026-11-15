package rayTraceTypescript.gpu

/** What a card traces with: the renderer's own CUDA kernel, or OptiX and its RT cores. */
enum class Engine { CUDA, OPTIX }

/**
 * One card picked to render, and how. With [fallback], an [Engine.OPTIX] card traces a scene
 * OptiX does not take with the CUDA kernel instead of refusing it.
 */
data class GpuPick(val device: GpuDevice, val engine: Engine, val fallback: Boolean = false) {
    override fun toString(): String = "${device.shortName} (${if (engine == Engine.OPTIX) "OptiX" else "CUDA"})"
}

/**
 * Which cards render, and with which engine: what -Drt.gpus and -Drt.backend say.
 *
 * [gpus] is `all` for every card, `one` for the best of them, or a list such as `2060`, `0,1` or
 * `1070,2060`: each entry a card's number in the driver's list or a piece of its name, and
 * optionally its own engine after a colon (`2060:optix,1070:cuda`). Naming a card twice runs two
 * sessions on it, which is of use only for testing without a second card.
 *
 * [backend] is `auto`, `optix` or `cuda`, for the entries that name no engine. `auto` is OptiX on
 * the cards with RT cores, where OptiX is installed, and the CUDA kernel on the others.
 */
data class GpuChoice(val gpus: String = ALL, val backend: String = AUTO) {

    /**
     * The cards among [devices] this choice means, never none. [optixReason] is why OptiX cannot
     * be used on this machine, null when it can: `auto` then leaves it out, a card told to use
     * OptiX is an error.
     */
    fun resolve(devices: List<GpuDevice>, optixReason: String?): List<GpuPick> {
        if (devices.isEmpty()) throw CudaException("no CUDA device found")
        fun pick(device: GpuDevice, engine: String): GpuPick = when (engine.lowercase()) {
            "cuda" -> GpuPick(device, Engine.CUDA)
            "optix" -> {
                if (optixReason != null) throw CudaException("${device.shortName} cannot trace through OptiX: $optixReason")
                GpuPick(device, Engine.OPTIX)
            }
            AUTO -> if (device.hasRtCores && optixReason == null) GpuPick(device, Engine.OPTIX, fallback = true) else GpuPick(device, Engine.CUDA)
            else -> throw CudaException("no engine called '$engine' (-Drt.backend is auto, optix or cuda)")
        }

        return when (gpus.trim().lowercase()) {
            ALL, "" -> devices.map { pick(it, backend) }
            ONE -> {
                // The card OptiX can use first, then the newest architecture; the driver's order breaks ties.
                val best = devices.maxWith(compareBy<GpuDevice> { it.hasRtCores && optixReason == null }.thenBy { it.computeCapability }.thenBy { -it.ordinal })
                listOf(pick(best, backend))
            }
            else -> gpus.split(',').map { it.trim() }.filter { it.isNotEmpty() }.flatMap { entry ->
                val selector = entry.substringBefore(':').trim()
                val engine = entry.substringAfter(':', backend).trim()
                val number = selector.toIntOrNull()
                val named = if (number != null && number in devices.indices) listOf(devices[number])
                else devices.filter { it.name.contains(selector, ignoreCase = true) }
                if (named.isEmpty()) {
                    throw CudaException("no CUDA device matches '$selector' (found ${devices.joinToString { "${it.ordinal}: ${it.name}" }})")
                }
                named.map { pick(it, engine) }
            }.also { if (it.isEmpty()) throw CudaException("-Drt.gpus=$gpus names no card") }
        }
    }

    companion object {
        const val ALL = "all"
        const val ONE = "one"
        const val AUTO = "auto"

        /** -Drt.gpus (or the older RT_GPU_DEVICE name filter) and -Drt.backend; every card and `auto` without them. */
        fun fromProperties(): GpuChoice = GpuChoice(
            System.getProperty("rt.gpus") ?: CudaContext.deviceFilter() ?: ALL,
            System.getProperty("rt.backend") ?: AUTO
        )
    }
}
