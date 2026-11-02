package rayTraceTypescript

import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Metal
import rayTraceTypescript.objects.BvhNode
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.objects.MeshLoader
import rayTraceTypescript.materials.Material
import rayTraceTypescript.textures.CheckerTexture
import rayTraceTypescript.utils.randomFloat
import kotlin.system.measureTimeMillis
import rayTraceTypescript.scenes.BookScenes

fun main() {
    // -Drt.seed pins every random draw - the scene's layout and the render's samples - so two
    // runs, or two versions of the renderer, can be compared image to image.
    System.getProperty("rt.seed")?.toLongOrNull()?.let { rayTraceTypescript.utils.RandomSource.withSeed(it) }
    // Renders from the command line are denoised unless -Drt.denoise=off. The renderer itself
    // only denoises when asked, so tests and other callers keep getting the raw image.
    if (System.getProperty("rt.denoise") == null) System.setProperty("rt.denoise", "on")
    // -Drt.gui.load=scene.rtscene opens a scene saved in the window: its base scene and the
    // switches it was built with come from the file (flags given here still win), then the
    // window restores the rest.
    System.getProperty("rt.gui.load")?.let { path ->
        val saved = rayTraceTypescript.gui.SceneFile.read(java.io.File(path))
        System.setProperty("rt.scene", saved.scene)
        saved.properties.forEach { (key, value) -> if (System.getProperty(key) == null) System.setProperty(key, value) }
        if (System.getProperty("rt.gui") == null) System.setProperty("rt.gui", "")
    }
    val scene =BookScenes.byName(System.getProperty("rt.scene", "earth"))
    // The books' final scenes are quoted at sample counts that take hours; these let a scene
    // be scaled down without editing it.
    System.getProperty("rt.samples")?.toIntOrNull()?.let { scene.camera.samplesPerPixel = it }
    System.getProperty("rt.width")?.toIntOrNull()?.let { scene.camera.imageWidth = it }
    System.getProperty("rt.depth")?.toIntOrNull()?.let { scene.camera.maxReflectionDepth = it }
    // A scene's aspect ratio is part of its framing, but a wider one is sometimes wanted -
    // the books' square rooms do not fit a 16:9 screen.
    System.getProperty("rt.aspect")?.toFloatOrNull()?.let { scene.camera.aspectRatio = it }
    // -Drt.lights=off renders the same scene without aiming any samples at its lights.
    if (System.getProperty("rt.lights") == "off") scene.camera.lights = null
    println("Scene: ${scene.name} (${scene.camera.imageWidth}px, ${scene.camera.samplesPerPixel} spp)")

    // -Drt.out redirects the render, so a quick look does not overwrite the committed image.ppm.
    val output = java.nio.file.Paths.get(System.getProperty("rt.out", "image.ppm"))

    // -Drt.turntable=film.mp4 renders a film of the scene's objects turning (see TurntableRender).
    System.getProperty("rt.turntable")?.let { film ->
        rayTraceTypescript.gpu.TurntableRender.render(scene, java.nio.file.Paths.get(film))
        return
    }

    // -Drt.gui opens the scene in a window to steer the camera with the mouse; S there saves
    // beside where -Drt.out points.
    if (System.getProperty("rt.gui") != null) {
        rayTraceTypescript.gui.Viewer(scene, output.toAbsolutePath().parent).run()
        return
    }

    var numberOfRays: Long = 0
    val time = measureTimeMillis {
        numberOfRays = scene.camera.render(scene.world, output)
    }
    printRenderReport(time, numberOfRays)
}

private fun printRenderReport(time: Long, numberOfRays: Long) {
    println("Execution time: ${formatDuration(time)}")
    println("Number of rays: ${numberOfRays.toHumanReadable()}")
    // GPU renders can finish inside a second, so rate from milliseconds.
    val raysPerSecond = if (time <= 0) numberOfRays else numberOfRays * 1000 / time
    println("Number of rays per second: ${raysPerSecond.toHumanReadable()}")
}

fun prepareWorld(): HittableList = WorldData.hardcodedWorld()

fun randomWorld(): HittableList {
    val worldObjects: MutableList<Hittable> = mutableListOf()

    val checker = CheckerTexture(.32F, Color(.2, .3, .1), Color(.9, .9, .9))
    val groundMaterial = Lambertian(checker)
    val groundSphere = Sphere(Point(0.0f, -1000.0f, 0.0f), 1000.0F, groundMaterial)
    worldObjects.add(groundSphere)

    fun rand() = randomFloat(0.0f, 1.0f)
    fun randRange(min: Float = 0.0f, max: Float = 1.0f) = randomFloat(min, max)
    fun randomColor(min: Float = 0.0f, max: Float = 1.0f) =
        Color(randRange(min, max), randRange(min, max), randRange(min, max))

    for (a in -11..10) {
        for (b in -11..10) {
            val chooseMat = rand()
            val center = Point(a + 0.9f * rand(), 0.2f, b + 0.9f * rand())
            if ((center - Point(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                if (chooseMat < 0.8f) {
                    val albedo = randomColor() * randomColor()
                    val center2: Point = center + Point(0f, randRange(0f, 0.5f), 0f)
                    worldObjects.add(Sphere(center, center2, 0.2f, Lambertian(albedo)))
                } else if (chooseMat < 0.95f) {
                    val albedo = randomColor(0.5f, 1.0f)
                    val fuzz = randRange(0.0f, 0.5f)
                    worldObjects.add(Sphere(center, 0.2f, Metal(albedo, fuzz)))
                } else {
                    worldObjects.add(Sphere(center, 0.2f, Dielectric(1.5f)))
                }
            }
        }
    }

    worldObjects.add(Sphere(Point(0.0f, 1.0f, 0.0f), 1.0f, Dielectric(1.5f)))
    worldObjects.add(Sphere(Point(-4.0f, 1.0f, 0.0f), 1.0f, Lambertian(Color(0.4f, 0.2f, 0.1f))))
    worldObjects.add(Sphere(Point(4.0f, 1.0f, 0.0f), 1.0f, Metal(Color(0.7f, 0.6f, 0.5f), 0.0f)))
    return HittableList(worldObjects)
}

fun formatDuration(ms: Long): String {
    val minutes = ms / 60000
    val seconds = (ms % 60000) / 1000
    val millis = ms % 1000

    return "%02d:%02d.%03d".format(minutes, seconds, millis)
}


fun Long.toHumanReadable(): String {
    val abs = kotlin.math.abs(this)

    return when {
        abs >= 1_000_000_000 -> "%.1fB".format(this / 1_000_000_000.0)
        abs >= 1_000_000     -> "%.1fM".format(this / 1_000_000.0)
        abs >= 1_000         -> "%.1fK".format(this / 1_000.0)
        else -> this.toString()
    }
}

