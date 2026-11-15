package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.CudaContext
import rayTraceTypescript.gpu.GpuRenderer
import rayTraceTypescript.gpu.optix.OptixApi
import rayTraceTypescript.gpu.optix.OptixLayout

/**
 * OptiX end to end, where a card, the driver's nvoptix.dll and the SDK headers are all there
 * (native/optix/README.md); skipped otherwise, like the other GPU tests.
 */
class OptixSmokeTest {

    @Test
    fun theCompilerDescribesOptixStructures() {
        assumeOptix()
        CudaContext.create().use { cuda ->
            val layout = OptixLayout.of(cuda)
            assertEquals(OptixApi.ABI_VERSION.toLong(), layout.constant("OPTIX_ABI_VERSION"))
            // Three sizes OptiX's own headers pin with static asserts or document.
            assertEquals(80, layout.size("OptixInstance"))
            assertEquals(8 + 1024, layout.size("OptixBuildInput"))
            assertEquals(32L, layout.constant("OPTIX_SBT_RECORD_HEADER_SIZE"))
            assertEquals(48, layout.offset("OptixInstance", "instanceId"))
        }
    }

    @Test
    fun theDriverHandsOutItsFunctions() {
        assumeOptix()
        CudaContext.create().use { assertNull(OptixApi.unavailabilityReason()) }
    }

    private fun assumeOptix() {
        val gpu = GpuRenderer.unavailabilityReason()
        assumeTrue(gpu == null, "No CUDA device available: $gpu")
        assumeTrue(java.io.File(OptixApi.includeFolder(), "optix.h").isFile, "No OptiX headers in ${OptixApi.includeFolder()}")
    }
}
