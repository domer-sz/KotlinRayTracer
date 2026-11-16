package rayTraceTypescript.gpu

import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.scenes.SceneDefinition
import java.nio.file.Path

/**
 * A turntable film of a scene from the command line: -Drt.turntable=<file.mp4>.
 *
 * The camera, the lights and the room stand still and the scene's objects turn - its parts that
 * are neither lights nor walls (-Drt.turntable.parts=a;b picks some by name) - together, about
 * the vertical line through the middle of their footprint. That is what a real turntable does,
 * and what StlViewer's film looks like (it turns the camera with its lamps and wall instead).
 * A part that is a placed mesh ([PlacedMesh], as in studio-model) costs a tree copy per frame;
 * any other is flattened again, which for a large model is slow.
 */
object TurntableRender {

    fun render(scene: SceneDefinition, output: Path) {
        GpuRenderer.unavailabilityReason()?.let { throw IllegalStateException("a turntable film needs the GPU: $it") }
        val camera = scene.camera
        val settings = TurntableSettings.fromProperties(camera.samplesPerPixel)
        val setup = camera.initialize()

        val wanted = System.getProperty("rt.turntable.parts")?.split(';')?.map { it.trim() }?.filter { it.isNotEmpty() }
        val turning = scene.parts.filter { !it.isLight && !it.isSurface && it.enabled }
            .filter { wanted == null || it.name in wanted }
        require(turning.isNotEmpty()) {
            "scene ${scene.name} has nothing to turn" + (wanted?.let { " named ${it.joinToString()}" } ?: "")
        }
        // The axis: the middle of the turning parts' foot, as StlViewer turns a figure (Footprint).
        val (axisX, axisZ) = Footprint.axis(turning.map { it.hittable })

        val frames = settings.framesPerTurn
        println(
            "Turntable: ${turning.joinToString { it.name }} about (%.4f, %.4f) - ${setup.imageWidth}x${setup.imageHeight}, ".format(axisX, axisZ) +
                "$frames frames (${settings.seconds} s at ${settings.fps} fps) x ${settings.turns}, ${settings.samples} spp -> $output"
        )

        fun buffersAt(frame: Int): GpuScene {
            val degrees = settings.degrees(frame)
            val swaps = java.util.IdentityHashMap<Hittable, Hittable>()
            for (part in turning) {
                swaps[part.hittable] = Placement.turned(part.hittable, axisX, axisZ, degrees)
                part.lightProxy?.let { swaps[it] = Placement.turned(it, axisX, axisZ, degrees) }
            }
            return SceneFlattener.scene(swapped(scene.world, swaps), camera.lights?.let { swapped(it, swaps) })
        }

        val denoiser = GpuRenderer.requestedDenoiser()
        // Enlarging, the denoiser runs on the card that traces best, and the second card would only
        // slow a film down: copying it each frame's scene for its CUDA kernel costs more than it
        // traces of a quarter-size frame (Yennefer: 0.31 s a frame with both cards, 0.19 s without).
        // -Drt.gpus still decides when it is given.
        val choice = GpuChoice.fromProperties().let {
            if (denoiser?.upscale == 2 && System.getProperty("rt.gpus") == null) it.copy(gpus = GpuChoice.ONE) else it
        }
        try {
            var session: GpuSession? = null
            try {
                val done = TurntableFilm.render(
                    setup, settings, denoiser, output,
                    sessionFor = { frame ->
                        val buffers = buffersAt(frame)
                        session?.also { it.replaceScene(buffers) } ?: GpuSession.open(buffers, setup.totalPixels, choice).also { session = it }
                    },
                    progress = { done, total, elapsed ->
                        val perFrame = elapsed / 1e9 / done
                        System.err.print(
                            "\r[film] frame $done of $total, %.2f s a frame, about %.0f s left   ".format(perFrame, perFrame * (total - done))
                        )
                    }
                )
                System.err.println()
                if (done) println("Film saved: ${output.toAbsolutePath()}")
            } finally {
                session?.close()
            }
        } finally {
            denoiser?.close()
        }
    }

    /**
     * [h] with the objects in [swaps] replaced, going down through plain lists - the parts of a
     * scene sit in its top-level lists; a tree built over them would have to be built again anyway.
     */
    private fun swapped(h: Hittable, swaps: Map<Hittable, Hittable>): Hittable {
        swaps[h]?.let { return it }
        if (h is HittableList) return HittableList(h.objects.map { swapped(it, swaps) }.toMutableList())
        return h
    }
}
