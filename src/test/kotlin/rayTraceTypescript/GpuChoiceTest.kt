package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.CudaException
import rayTraceTypescript.gpu.Engine
import rayTraceTypescript.gpu.GpuChoice
import rayTraceTypescript.gpu.GpuDevice
import rayTraceTypescript.gpu.GpuPick

/** What -Drt.gpu and -Drt.backend come to on a machine with an RTX and a GTX card. */
class GpuChoiceTest {
    private val rtx = GpuDevice(0, "NVIDIA GeForce RTX 2060", 75, 6L shl 30)
    private val gtx = GpuDevice(1, "NVIDIA GeForce GTX 1070", 61, 8L shl 30)
    private val both = listOf(rtx, gtx)

    /** OptiX where the scene allows, the CUDA kernel otherwise: what `auto` gives a card with RT cores. */
    private val optixWherePossible = GpuPick(rtx, Engine.OPTIX, fallback = true)

    @Test
    fun theCardWithRtCoresRendersByDefaultThroughOptix() {
        assertEquals(optixWherePossible, GpuChoice().resolve(both, null))
    }

    @Test
    fun theBestCardIsTheOneOptixCanUseWhereverTheDriverListsIt() {
        assertEquals(optixWherePossible.copy(device = rtx.copy(ordinal = 1)), GpuChoice().resolve(listOf(gtx.copy(ordinal = 0), rtx.copy(ordinal = 1)), null))
    }

    @Test
    fun withoutOptixInstalledTheNewestCardUsesTheCudaKernel() {
        assertEquals(GpuPick(rtx, Engine.CUDA), GpuChoice().resolve(both, "no headers"))
    }

    @Test
    fun aCardIsNamedByItsNumberOrByAPieceOfItsName() {
        assertEquals(GpuPick(gtx, Engine.CUDA), GpuChoice("1").resolve(both, null))
        // 1070 is no card's number, so it is read as a name.
        assertEquals(GpuPick(gtx, Engine.CUDA), GpuChoice("1070").resolve(both, null))
        assertEquals(optixWherePossible, GpuChoice("rtx").resolve(both, null))
    }

    @Test
    fun theBackendSettingOverridesAuto() {
        assertEquals(GpuPick(rtx, Engine.CUDA), GpuChoice(backend = "cuda").resolve(both, null))
        // Told to use OptiX, a card does not fall back to the CUDA kernel - even one without RT cores.
        assertEquals(GpuPick(gtx, Engine.OPTIX), GpuChoice("1070", "optix").resolve(both, null))
    }

    @Test
    fun askingForOptixWhereItIsMissingIsAnError() {
        assertThrows(CudaException::class.java) { GpuChoice(backend = "optix").resolve(both, "no headers") }
    }

    @Test
    fun aNameNoCardHasIsAnError() {
        assertThrows(CudaException::class.java) { GpuChoice("3090").resolve(both, null) }
        assertThrows(CudaException::class.java) { GpuChoice().resolve(emptyList(), null) }
        assertThrows(CudaException::class.java) { GpuChoice(backend = "vulkan").resolve(both, null) }
    }
}
