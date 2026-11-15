package rayTraceTypescript

import org.junit.jupiter.api.Assumptions.assumeTrue
import rayTraceTypescript.gpu.optix.OptixApi

/**
 * Every scene of [GpuCpuComparisonTest], held to the same CPU renders, traced through OptiX.
 * Skipped where OptiX cannot run (no card, no SDK headers - native/optix/README.md), and scene
 * by scene for whatever the OptiX engine says it does not trace yet.
 */
class OptixCpuComparisonTest : GpuCpuComparisonTest() {
    override val backend: String = "optix"

    override fun assumeGpu() {
        // The card first: OptiX's library is found beside the CUDA one, once that is loaded.
        super.assumeGpu()
        val reason = OptixApi.unavailabilityReason()
        assumeTrue(reason == null, "OptiX is not available: $reason")
    }
}
