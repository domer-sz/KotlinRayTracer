package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.CudaException
import rayTraceTypescript.gpu.Engine
import rayTraceTypescript.gpu.GpuChoice
import rayTraceTypescript.gpu.GpuDevice
import rayTraceTypescript.gpu.GpuPick

/** What -Drt.gpus and -Drt.backend come to on a machine with an RTX and a GTX card. */
class GpuChoiceTest {
    private val rtx = GpuDevice(0, "NVIDIA GeForce RTX 2060", 75, 6L shl 30)
    private val gtx = GpuDevice(1, "NVIDIA GeForce GTX 1070", 61, 8L shl 30)
    private val both = listOf(rtx, gtx)

    /** OptiX where the scene allows, the CUDA kernel otherwise: what `auto` gives a card with RT cores. */
    private val optixWherePossible = GpuPick(rtx, Engine.OPTIX, fallback = true)

    @Test
    fun everyCardRendersByDefaultAndOnlyTheOneWithRtCoresGoesThroughOptix() {
        assertEquals(listOf(optixWherePossible, GpuPick(gtx, Engine.CUDA)), GpuChoice().resolve(both, null))
    }

    @Test
    fun withoutOptixInstalledEveryCardUsesTheCudaKernel() {
        assertEquals(listOf(GpuPick(rtx, Engine.CUDA), GpuPick(gtx, Engine.CUDA)), GpuChoice().resolve(both, "no headers"))
    }

    @Test
    fun oneIsTheCardOptixCanUseWhereverTheDriverListsIt() {
        assertEquals(listOf(optixWherePossible), GpuChoice("one").resolve(both, null))
        assertEquals(listOf(optixWherePossible.copy(device = rtx.copy(ordinal = 1))), GpuChoice("one").resolve(listOf(gtx.copy(ordinal = 0), rtx.copy(ordinal = 1)), null))
    }

    @Test
    fun aCardIsNamedByItsNumberOrByAPieceOfItsName() {
        assertEquals(listOf(GpuPick(gtx, Engine.CUDA)), GpuChoice("1").resolve(both, null))
        // 1070 is no card's number, so it is read as a name.
        assertEquals(listOf(GpuPick(gtx, Engine.CUDA)), GpuChoice("1070").resolve(both, null))
        assertEquals(listOf(GpuPick(gtx, Engine.CUDA), optixWherePossible), GpuChoice("gtx, RTX").resolve(both, null))
    }

    @Test
    fun anEntryMayNameItsOwnEngine() {
        assertEquals(
            listOf(GpuPick(rtx, Engine.CUDA), GpuPick(gtx, Engine.OPTIX)),
            GpuChoice("2060:cuda,1070:optix").resolve(both, null)
        )
    }

    @Test
    fun namingACardTwiceRunsItTwice() {
        assertEquals(listOf(GpuPick(rtx, Engine.CUDA), GpuPick(rtx, Engine.OPTIX)), GpuChoice("0:cuda,0:optix").resolve(both, null))
    }

    @Test
    fun theBackendSettingOverridesAuto() {
        assertEquals(listOf(GpuPick(rtx, Engine.CUDA), GpuPick(gtx, Engine.CUDA)), GpuChoice(backend = "cuda").resolve(both, null))
        // Told to use OptiX, a card does not fall back to the CUDA kernel.
        assertEquals(listOf(GpuPick(rtx, Engine.OPTIX), GpuPick(gtx, Engine.OPTIX)), GpuChoice(backend = "optix").resolve(both, null))
    }

    @Test
    fun askingForOptixWhereItIsMissingIsAnError() {
        assertThrows(CudaException::class.java) { GpuChoice(backend = "optix").resolve(both, "no headers") }
    }

    @Test
    fun theCardThatDenoisesSitsOutOfTracingByAllUnlessItTracesBest() {
        // The GTX 1070 denoising beside the RTX: it does not trace as well.
        assertEquals(listOf(optixWherePossible), GpuChoice(denoising = gtx).resolve(both, null))
        // The RTX denoising: it traces best, so it does both, and the GTX traces too.
        assertEquals(listOf(optixWherePossible, GpuPick(gtx, Engine.CUDA)), GpuChoice(denoising = rtx).resolve(both, null))
        // A card that is all there is does both.
        val lone = gtx.copy(ordinal = 0)
        assertEquals(listOf(GpuPick(lone, Engine.CUDA)), GpuChoice(denoising = lone).resolve(listOf(lone), null))
    }

    @Test
    fun oneAndAListAreTakenAsTheyAreWhateverDenoises() {
        assertEquals(listOf(optixWherePossible), GpuChoice(GpuChoice.ONE, denoising = rtx).resolve(both, null))
        assertEquals(listOf(optixWherePossible, GpuPick(gtx, Engine.CUDA)), GpuChoice("0,1", denoising = gtx).resolve(both, null))
    }

    @Test
    fun aNameNoCardHasIsAnError() {
        assertThrows(CudaException::class.java) { GpuChoice("3090").resolve(both, null) }
        assertThrows(CudaException::class.java) { GpuChoice().resolve(emptyList(), null) }
    }
}
