package rayTraceTypescript.gpu

import rayTraceTypescript.gpu.optix.OptixSession
import rayTraceTypescript.render.CameraSetup

/**
 * One card that traces through OptiX whenever the scene is something the OptiX engine takes, and
 * through the CUDA kernel when it is not: what `auto` means on a card with RT cores. The engine
 * is chosen anew each time the scene is replaced; changing it starts the outputs afresh, as a
 * new scene does for its callers anyway.
 */
internal class SwitchingSession(private val device: GpuDevice, scene: SceneBuffers, private val maxPixels: Int) : GpuSession {
    private var current: GpuSession = openFor(scene)
    private var camera: CameraSetup? = null

    override var seed: Long
        get() = current.seed
        set(value) {
            current.seed = value
        }

    override val devices: List<String> get() = current.devices

    override fun replaceScene(next: SceneBuffers) {
        val wantsOptix = OptixSession.unsupported(next) == null
        if (wantsOptix == (current is OptixSession)) {
            current.replaceScene(next)
            return
        }
        val seed = current.seed
        current.close()
        current = openFor(next)
        current.seed = seed
        camera?.let { current.setCamera(it) }
    }

    override fun setCamera(setup: CameraSetup) {
        camera = setup
        current.setCamera(setup)
    }

    override fun launch(mode: Int, setup: CameraSetup, sampleStart: Int, count: Int, scale: Float, target: Int) =
        current.launch(mode, setup, sampleStart, count, scale, target)

    override fun read(target: Int, into: FloatArray) = current.read(target, into)

    override fun clear(target: Int) = current.clear(target)

    override fun prepare(mode: Int) = current.prepare(mode)

    override fun close() = current.close()

    private fun openFor(scene: SceneBuffers): GpuSession {
        val reason = OptixSession.unsupported(scene) ?: return OptixSession(scene, maxPixels, device)
        System.err.println("[gpu] ${device.shortName} traces with the CUDA kernel: $reason")
        return CudaSession(scene, maxPixels, device)
    }
}
