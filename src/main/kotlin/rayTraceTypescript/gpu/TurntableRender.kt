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

        // -Drt.turntable.wave ripples the placed meshes instead of turning them.
        val wave = Wave.fromProperties()
        if (wave != null) {
            for (part in turning) {
                val placed = part.hittable as? PlacedMesh
                    ?: throw IllegalArgumentException("-Drt.turntable.wave needs placed meshes; ${part.name} is not one")
                require(placed.scale == 1f && placed.degrees == 0f && placed.offset.length() == 0f) {
                    "-Drt.turntable.wave needs ${part.name} placed as it is"
                }
            }
            if (wave.native) require(turning.all { (it.hittable as PlacedMesh).source != null }) {
                "-Drt.turntable.wave.native needs a sponge (-Drt.infill=sponge) to build again"
            }
            println("Wave: amplitude ${wave.amplitude}, wavelength ${wave.wavelength}, lifted ${wave.lift}" +
                if (wave.native) ", the sponge built again every frame" else "")
        }
        // The sponge's pitch is a fraction of the model's height, which the wave changes: keep it the same length.
        fun heightOf(mesh: rayTraceTypescript.objects.Mesh): Float {
            val box = mesh.bounds()
            return (box.y.max - box.y.min).takeIf { it > 0f } ?: maxOf(box.x.max - box.x.min, box.z.max - box.z.min)
        }
        val spongeSettings = rayTraceTypescript.objects.LatticeFill.Settings.fromProperties()
        fun rippled(placed: PlacedMesh, frame: Int): rayTraceTypescript.objects.Mesh {
            if (wave?.native != true) return wave!!.deform(placed.block.mesh, frame, frames)
            val source = placed.source!!
            val moved = wave.deform(source, frame, frames)
            val pitch = spongeSettings.pitch * heightOf(source) / heightOf(moved)
            val started = System.nanoTime()
            val sponge = rayTraceTypescript.objects.LatticeFill.build(moved, spongeSettings.copy(pitch = pitch))
            System.err.print("[wave] frame $frame: sponge of %,d triangles built in %.1f s   ".format(sponge.triangleCount, (System.nanoTime() - started) / 1e9))
            return sponge
        }

        fun buffersAt(frame: Int): GpuScene {
            val degrees = settings.degrees(frame)
            val swaps = java.util.IdentityHashMap<Hittable, Hittable>()
            if (wave != null) {
                for (part in turning) {
                    val placed = part.hittable as PlacedMesh
                    swaps[part.hittable] = PlacedMesh(
                        MeshBlock(rippled(placed, frame)), 1f, 0f, placed.offset, placed.material, placed.palette
                    )
                }
                return SceneFlattener.scene(swapped(scene.world, swaps), camera.lights?.let { swapped(it, swaps) })
            }
            for (part in turning) {
                swaps[part.hittable] = Placement.turned(part.hittable, axisX, axisZ, degrees, turning = true)
                part.lightProxy?.let { swaps[it] = Placement.turned(it, axisX, axisZ, degrees) }
            }
            return SceneFlattener.scene(swapped(scene.world, swaps), camera.lights?.let { swapped(it, swaps) })
        }

        // A temporal denoiser needs the flow guide, which follows placed meshes only.
        val flowFollows = wave == null && turning.all { it.hittable is PlacedMesh }
        val denoiser = GpuRenderer.requestedDenoiser(temporal = flowFollows)
        val turn = if (flowFollows) FrameTurn(axisX, axisZ, settings.degrees(1) - settings.degrees(0)) else null
        try {
            var session: GpuSession? = null
            try {
                val done = TurntableFilm.render(
                    setup, settings, denoiser, output,
                    sessionFor = { frame ->
                        val buffers = buffersAt(frame)
                        session?.also { it.replaceScene(buffers) } ?: GpuSession.open(buffers, setup.totalPixels).also { session = it }
                    },
                    progress = { done, total, elapsed ->
                        val perFrame = elapsed / 1e9 / done
                        System.err.print(
                            "\r[film] frame $done of $total, %.2f s a frame, about %.0f s left   ".format(perFrame, perFrame * (total - done))
                        )
                    },
                    turn = turn
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
