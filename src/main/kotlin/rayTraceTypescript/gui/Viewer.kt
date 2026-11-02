package rayTraceTypescript.gui

import rayTraceTypescript.Camera
import rayTraceTypescript.Interval
import rayTraceTypescript.Point
import rayTraceTypescript.Ray
import rayTraceTypescript.Vector
import rayTraceTypescript.gpu.GpuRenderer
import rayTraceTypescript.gpu.GpuSession
import rayTraceTypescript.gpu.SceneFlattener
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.MeshLoader
import rayTraceTypescript.render.CameraSetup
import rayTraceTypescript.render.Denoiser
import rayTraceTypescript.render.PpmWriter
import rayTraceTypescript.scenes.SceneDefinition
import java.awt.BasicStroke
import java.awt.BorderLayout
import java.awt.Color
import java.awt.Dimension
import java.awt.Font
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.event.KeyEvent
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.awt.event.MouseWheelEvent
import java.awt.event.WindowAdapter
import java.awt.event.WindowEvent
import java.awt.image.BufferedImage
import java.awt.image.DataBufferInt
import java.io.File
import java.nio.file.Path
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicInteger
import javax.swing.JFileChooser
import javax.swing.JFrame
import javax.swing.JOptionPane
import javax.swing.filechooser.FileNameExtensionFilter
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.SwingUtilities
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * An interactive window onto a scene, steered like a 3D modelling program:
 *  - left drag orbits the camera around the point it looks at,
 *  - Shift + left drag, or middle drag, pans,
 *  - the wheel, or right drag, moves the camera in and out,
 *  - Ctrl + left click stands the selected object on the floor under the cursor,
 *  - R resets the view, D toggles denoising, S saves the image, + / - change the field of view.
 * The panel beside the image ([ControlPanel]) sets the resolution and the sample target, and
 * adds, places and dresses models loaded from any format [MeshLoader] reads.
 *
 * It opens in the quick look for setting the scene up: one ray per pixel, shaded by the
 * surface's colour, at full size and sharp at once. T switches to ray tracing: while the camera
 * moves it renders a quarter-size image at one sample per pixel, so motion stays fluid; once it
 * stops it renders the full image up to the sample target, denoising every time the sample
 * count quadruples and once more at the end.
 */
class Viewer(private val scene: SceneDefinition, private val saveFolder: Path) {
    init {
        // Before any Swing component exists: the look applies to what is made after it.
        Theme.install()
    }

    private val base: Camera = scene.camera

    /** The camera as an orbit around [target]; everything the mouse changes. Compared by identity. */
    private class View(val target: DoubleArray, val yaw: Double, val pitch: Double, val radius: Double, val vfov: Double) {
        val from: DoubleArray
            get() = doubleArrayOf(
                target[0] + radius * cos(pitch) * sin(yaw),
                target[1] + radius * sin(pitch),
                target[2] + radius * cos(pitch) * cos(yaw)
            )

        fun with(
            target: DoubleArray = this.target, yaw: Double = this.yaw, pitch: Double = this.pitch,
            radius: Double = this.radius, vfov: Double = this.vfov
        ) = View(target, yaw, pitch, radius, vfov)
    }

    /** Output size in pixels. */
    data class Resolution(val width: Int, val height: Int) {
        val pixels: Int get() = width * height
    }

    private val initialView: View = run {
        val from = base.lookFrom
        val at = base.lookAt
        val dx = (from.x - at.x).toDouble()
        val dy = (from.y - at.y).toDouble()
        val dz = (from.z - at.z).toDouble()
        val radius = sqrt(dx * dx + dy * dy + dz * dz)
        View(doubleArrayOf(at.x.toDouble(), at.y.toDouble(), at.z.toDouble()), atan2(dx, dz), asin(dy / radius), radius, base.vfov.toDouble())
    }

    val sceneResolution = Resolution(base.imageWidth, base.initialize().imageHeight)

    @Volatile private var view = initialView
    @Volatile var resolution = sceneResolution
        set(value) { field = value; touch() }
    @Volatile var targetSamples = maxOf(base.samplesPerPixel, DEFAULT_TARGET)
    /** How many times a path may bounce before it is cut off (the scene's own value to start). */
    @Volatile var maxDepth = base.maxReflectionDepth.coerceAtLeast(1)
    @Volatile var denoiseOn = true
    /**
     * Ray tracing, or (the default, for setting the scene up) the quick look: one ray per pixel
     * shaded by the surface's colour, sharp at once. -Drt.gui.raytrace opens in ray tracing.
     */
    @Volatile var raytrace = System.getProperty("rt.gui.raytrace") != null
    @Volatile private var lastInput = 0L
    @Volatile private var dragging = false
    @Volatile private var running = true
    @Volatile private var saveRequested = false
    // OIDN runs on the CPU, so it can work while the GPU keeps sampling; and turning the buffer
    // into a picture needs no GPU either. One thread each, so their results come in order.
    private val denoiseExecutor = Executors.newSingleThreadExecutor { Thread(it, "viewer-denoise").apply { isDaemon = true } }
    private val displayExecutor = Executors.newSingleThreadExecutor { Thread(it, "viewer-display").apply { isDaemon = true } }

    /** Models added in this window; editing one bumps [sceneVersion] so the GPU copy is rebuilt. */
    val objects = CopyOnWriteArrayList<PlacedObject>()
    private val sceneVersion = AtomicInteger(0)
    @Volatile private var lastSceneEdit = 0L
    /** The scene's own switchable lights and removable objects. */
    val builtIns: List<BuiltInObject> = scene.parts.filter { !it.isSurface }.map { BuiltInObject(it) }
    /** The room's walls, whose materials can be changed. */
    val walls: List<WallObject> = scene.parts.filter { it.isSurface }.map { WallObject(it) }.filter { it.editable }
    val lights: List<BuiltInObject> get() = builtIns.filter { it.part.isLight }
    @Volatile var selected: SceneEntry? = null
    /** The selection when it is an added object - the only kind with a size and a material. */
    val selectedModel: PlacedObject? get() = selected as? PlacedObject
    /** The selection when it can be moved: an added object, or one of the scene's own. */
    val selectedMovable: Movable? get() = selected as? Movable

    /** What the object list shows: the scene's objects still in it, the added models, then the walls. */
    fun entries(): List<SceneEntry> = builtIns.filter { !it.part.isLight && it.enabled } + objects + walls

    /** The selection when its material can be changed: an added object or a wall. */
    val selectedMaterial: HasMaterial? get() = selected as? HasMaterial

    @Volatile private var shown: BufferedImage? = null
    private val statusLabel = JLabel("Przygotowuję scenę…")
    /** The status bar's text; set from any thread, shown on the UI thread. */
    @Volatile private var status = "Przygotowuję scenę…"
        set(value) {
            if (field == value) return
            field = value
            SwingUtilities.invokeLater { statusLabel.text = value }
        }
    @Volatile private var placeMarker: java.awt.Point? = null
    /** Where a Ctrl-dragged model would stand, while the button is held. */
    @Volatile private var ghost: DoubleArray? = null
    private val panel = ImagePanel()
    private lateinit var controls: ControlPanel

    /** A size fitting what the camera sees at the start: a quarter of the frame's height. */
    val defaultObjectHeight: Double = 0.25 * 2.0 * initialView.radius * tan(Math.toRadians(initialView.vfov / 2))

    /** Where new objects stand: the middle of the scene's footprint, on y = 0. */
    private val defaultObjectPosition: DoubleArray = scene.world.aabbBoundingBox().let { box ->
        doubleArrayOf(((box.x.min + box.x.max) / 2).toDouble(), 0.0, ((box.z.min + box.z.max) / 2).toDouble())
    }

    /** Opens the window and blocks until it is closed. */
    fun run() {
        val closed = CountDownLatch(1)
        SwingUtilities.invokeLater {
            val frame = JFrame("Ray tracer - ${scene.name}")
            frame.defaultCloseOperation = JFrame.DISPOSE_ON_CLOSE
            controls = ControlPanel(this, frame)
            val display = fit(sceneResolution)
            panel.preferredSize = Dimension(display.width, display.height)
            frame.contentPane.add(panel, BorderLayout.CENTER)
            // The panel scrolls when the window is shorter than its fields.
            frame.contentPane.add(javax.swing.JScrollPane(controls).apply {
                horizontalScrollBarPolicy = javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER
                border = javax.swing.BorderFactory.createMatteBorder(0, 1, 0, 0, Theme.separator)
                verticalScrollBar.unitIncrement = 16
            }, BorderLayout.EAST)
            frame.contentPane.add(statusBar(), BorderLayout.SOUTH)
            frame.pack()
            frame.setLocationRelativeTo(null)
            frame.addWindowListener(object : WindowAdapter() {
                override fun windowClosed(e: WindowEvent) {
                    running = false
                    closed.countDown()
                }
            })
            installControls(frame)
            frame.isVisible = true
            // Windows does not let a new window take the foreground on its own; a moment on top does.
            frame.isAlwaysOnTop = true
            frame.toFront()
            frame.isAlwaysOnTop = false
            panel.requestFocusInWindow()
            // -Drt.gui.models=a.stl;b.obj opens the window with those models already placed.
            System.getProperty("rt.gui.models")?.split(';')?.filter { it.isNotBlank() }?.forEach { path ->
                addModel(File(path.trim())) { _, error ->
                    if (error != null) System.err.println("[viewer] could not load $path: $error")
                    controls.refreshObjects()
                }
            }
            // -Drt.gui.load=scene.rtscene opens a saved scene (Main has already picked its base scene).
            System.getProperty("rt.gui.load")?.let { controls.loadScene(File(it)) }
            // -Drt.gui.film=film.mp4 starts a turntable film at once, at the panel's size and the
            // -Drt.turntable.* settings - the dialog's path without the clicks, for scripts.
            System.getProperty("rt.gui.film")?.let { path ->
                val size = resolution
                val settings = rayTraceTypescript.gpu.TurntableSettings.fromProperties(64)
                startFilm(FilmJob(settings, Resolution(size.width - size.width % 2, size.height - size.height % 2), File(path), object : FilmListener {
                    override fun progress(done: Int, total: Int, elapsedNanos: Long) =
                        System.err.println("[viewer] film frame $done of $total, %.2f s a frame".format(elapsedNanos / 1e9 / done))

                    override fun finished(saved: Boolean, error: String?) =
                        System.err.println("[viewer] film ${if (saved) "saved: $path" else "not saved" + (error?.let { ": $it" } ?: "")}")
                }))
            }
        }
        val renderThread = Thread({ renderLoop() }, "viewer-render").apply { start() }
        closed.await()
        renderThread.join()
    }

    /** The strip along the bottom: what the renderer is doing on the left, the main gestures on the right. */
    private fun statusBar(): JPanel = JPanel(BorderLayout(16, 0)).apply {
        border = javax.swing.BorderFactory.createCompoundBorder(
            javax.swing.BorderFactory.createMatteBorder(1, 0, 0, 0, Theme.separator),
            javax.swing.BorderFactory.createEmptyBorder(6, 12, 6, 12)
        )
        statusLabel.font = Theme.font(12.5f)
        add(statusLabel, BorderLayout.CENTER)
        add(JLabel(HELP).apply {
            font = Theme.font(11.5f)
            foreground = Theme.muted
        }, BorderLayout.EAST)
    }

    /** The window's first size: the image at most 900 pixels tall and 1200 wide on screen. */
    private fun fit(r: Resolution): Dimension {
        val scale = minOf(1.0, 900.0 / r.height, 1200.0 / r.width)
        return Dimension((r.width * scale).toInt(), (r.height * scale).toInt())
    }

    // ------------------------------------------------------------------ scene edits (UI thread)

    /** Loads [file] off the UI thread, then adds it standing in the middle of the scene. */
    fun addModel(file: File, onDone: (PlacedObject?, String?) -> Unit) {
        status = "Wczytuję ${file.name}…"
        panel.repaint()
        Thread({
            try {
                val mesh = MeshLoader.load(file.toPath())
                val extension = file.extension.lowercase()
                val placed = PlacedObject(file.nameWithoutExtension, Shape.Model(mesh), file.absoluteFile).apply {
                    // Files made for 3D printing stand on z; the rest mostly use y as up.
                    zUp = extension == "stl" || extension == "3mf"
                    x = defaultObjectPosition[0]
                    y = defaultObjectPosition[1]
                    z = defaultObjectPosition[2]
                    height = defaultObjectHeight
                }
                status = "Przygotowuję ${file.name}…"
                panel.repaint()
                placed.prepare()
                objects += placed
                selected = placed
                sceneChanged()
                SwingUtilities.invokeLater { onDone(placed, null) }
            } catch (t: Throwable) {
                SwingUtilities.invokeLater { onDone(null, t.message ?: t.toString()) }
            }
        }, "viewer-load").apply { isDaemon = true; start() }
    }

    /** Takes [entry] out: an added model for good, one of the scene's own until [restoreBuiltIns]. */
    /**
     * Adds a sphere (glass, like the scene's) or a block (white, shaped like the column: twice
     * as tall as wide) in the middle of the scene, sized like a newly loaded model.
     */
    fun addShape(shape: Shape): PlacedObject {
        val ball = shape == Shape.Ball
        val number = objects.count { it.shape === shape } + 1
        val placed = PlacedObject(if (ball) "Kula $number" else "Graniastosłup $number", shape).apply {
            x = defaultObjectPosition[0]
            y = defaultObjectPosition[1]
            z = defaultObjectPosition[2]
            height = defaultObjectHeight
            width = defaultObjectHeight / 2
            depth = defaultObjectHeight / 2
            material = if (ball) MaterialPreset.GLASS.spec else MaterialPreset.WHITE.spec
        }
        objects += placed
        selected = placed
        sceneChanged()
        return placed
    }

    fun removeObject(entry: SceneEntry) {
        when (entry) {
            is PlacedObject -> objects -= entry
            is BuiltInObject -> entry.enabled = false
            // The room keeps its walls.
            is WallObject -> return
        }
        if (selected === entry) selected = null
        sceneChanged()
    }

    /**
     * Puts every removed, moved or restyled object of the scene back as it was (lights keep
     * their switches and power).
     */
    fun restoreBuiltIns() {
        builtIns.filter { !it.part.isLight }.forEach {
            it.enabled = true
            it.returnHome()
            it.material = it.original
        }
        sceneChanged()
    }

    fun setLight(light: BuiltInObject, on: Boolean) {
        light.enabled = on
        sceneChanged()
    }

    /** Sets how many times its own strength [light] shines with. */
    fun setLightPower(light: BuiltInObject, power: Double) {
        light.power = power
        sceneChanged()
    }

    /** Stands [o] where the middle of the view meets its floor level, or under the view's target. */
    fun centreInView(o: Movable) {
        val v = view
        val hit = floorPoint(v, 0.5, 0.5, o.y) ?: doubleArrayOf(v.target[0], o.y, v.target[2])
        o.x = hit[0]
        o.z = hit[2]
        sceneChanged()
    }

    fun sceneChanged() {
        lastSceneEdit = System.currentTimeMillis()
        sceneVersion.incrementAndGet()
    }

    // ------------------------------------------------------------------ turntable film

    /** What a film dialog hears back: progress while it renders, then how it ended. */
    interface FilmListener {
        fun progress(done: Int, total: Int, elapsedNanos: Long)
        /** [saved] is false when it was cancelled or failed; [error] says why it failed. */
        fun finished(saved: Boolean, error: String?)
    }

    /** One film to render: picked up by the render thread, which pauses the view meanwhile. */
    class FilmJob(
        val settings: rayTraceTypescript.gpu.TurntableSettings,
        val size: Resolution,
        val output: File,
        val listener: FilmListener
    ) {
        @Volatile var cancelled = false
    }

    @Volatile private var filmJob: FilmJob? = null

    /**
     * What a film turns: the selected object, or else every added object, or else the scene's own
     * objects (never its lights or walls) - together, about the middle of their footprints.
     */
    fun filmTurning(): List<Movable> = selectedMovable?.let { listOf(it) }
        ?: objects.toList().ifEmpty { builtIns.filter { !it.part.isLight && it.enabled } }

    /** Whether frames would be denoised: the D switch, and the OIDN library being there. */
    fun filmDenoised(): Boolean = denoiseOn && rayTraceTypescript.render.Denoiser.folder() != null

    /** Hands [job] to the render thread; it reports through its listener. */
    fun startFilm(job: FilmJob) {
        filmJob = job
    }

    /** Where films go unless the user picks another folder: beside saved images. */
    val filmFolder: File get() = lastSaveFolder
    val sceneName: String get() = scene.name

    /**
     * Renders [job] on the render thread: each frame turns [filmTurning]'s objects to that frame's
     * angle and rebuilds the GPU scene through [open]; afterwards they stand as before. Returns the
     * session it last used, which may be a new one when the film is larger than the view.
     */
    private fun runFilm(job: FilmJob, denoiser: Denoiser?, pixels: Int, open: (Int) -> GpuSession): GpuSession? {
        val turning = filmTurning()
        val originals = turning.map { doubleArrayOf(it.x, it.z, it.rotation) }
        val axisX = originals.map { it[0] }.average()
        val axisZ = originals.map { it[1] }.average()
        val filmSetup = setup(view, job.size)
        var last: GpuSession? = null
        status = "Renderuję film obrotowy…"
        try {
            val saved = rayTraceTypescript.gpu.TurntableFilm.render(
                filmSetup, job.settings, if (denoiseOn) denoiser else null, job.output.toPath(),
                sessionFor = { frame ->
                    val radians = Math.toRadians(job.settings.degrees(frame).toDouble())
                    val c = cos(radians)
                    val s = sin(radians)
                    turning.forEachIndexed { i, o ->
                        // Its place turned about the axis, as Placement turns points, and itself as much.
                        val dx = originals[i][0] - axisX
                        val dz = originals[i][1] - axisZ
                        o.x = c * dx + s * dz + axisX
                        o.z = -s * dx + c * dz + axisZ
                        o.rotation = originals[i][2] + job.settings.degrees(frame)
                    }
                    open(pixels).also { last = it }
                },
                cancelled = { job.cancelled || !running },
                progress = { done, total, elapsed -> job.listener.progress(done, total, elapsed) }
            )
            status = if (saved) "Zapisano film ${job.output.name}" else "Film przerwany - nic nie zapisano"
            job.listener.finished(saved, null)
        } catch (t: Throwable) {
            t.printStackTrace()
            val message = generateSequence(t) { it.cause }.last().message ?: t.toString()
            status = "Film nie powstał: $message"
            job.listener.finished(false, message)
        } finally {
            turning.forEachIndexed { i, o ->
                o.x = originals[i][0]
                o.z = originals[i][1]
                o.rotation = originals[i][2]
            }
            sceneChanged()
        }
        return last
    }

    /** Where the next save goes, without an extension; chosen in [chooseSave]. */
    @Volatile private var savePath: Path? = null
    private var lastSaveFolder: File = saveFolder.toFile()

    /**
     * Asks for the file name and folder (UI thread), suggesting viewer-<scene>-<time>, then has
     * the render thread save: PPM and PNG under that name, the denoised copy with -denoised added.
     */
    fun chooseSave(owner: java.awt.Component) {
        val stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"))
        val chooser = JFileChooser(lastSaveFolder).apply {
            dialogTitle = "Zapisz render"
            fileFilter = FileNameExtensionFilter("Obraz (PNG i PPM)", "png", "ppm")
            selectedFile = File(lastSaveFolder, "viewer-${scene.name}-$stamp.png")
        }
        while (true) {
            if (chooser.showSaveDialog(owner) != JFileChooser.APPROVE_OPTION) return
            // The name without .png / .ppm: both files, and the denoised pair, are made from it.
            val chosen = chooser.selectedFile
            val base = File(chosen.parentFile, chosen.name.replace(Regex("\\.(png|ppm)$", RegexOption.IGNORE_CASE), ""))
            val existing = listOf(".png", ".ppm").map { File(base.path + it) }.firstOrNull { it.exists() }
            if (existing != null && JOptionPane.showConfirmDialog(
                    owner, "Plik ${existing.name} już istnieje. Zastąpić go?", "Zapisz render", JOptionPane.YES_NO_OPTION
                ) != JOptionPane.YES_OPTION
            ) continue
            lastSaveFolder = base.parentFile
            savePath = base.toPath()
            saveRequested = true
            return
        }
    }

    // ------------------------------------------------------------------ scene files (UI thread)

    /** Writes the window's scene to [file]: camera, image settings, lights, and every object. */
    fun saveScene(file: File) {
        val v = view
        val saved = SavedScene(
            scene = scene.name,
            properties = SceneFile.SCENE_PROPERTIES.mapNotNull { key -> System.getProperty(key)?.let { key to it } }.toMap(),
            camera = doubleArrayOf(v.target[0], v.target[1], v.target[2], v.yaw, v.pitch, v.radius, v.vfov),
            width = resolution.width,
            height = resolution.height,
            samples = targetSamples,
            raytrace = raytrace,
            denoise = denoiseOn,
            parts = builtIns.map { b ->
                SavedPart(
                    b.part.name, b.enabled, if (b.moved) doubleArrayOf(b.x, b.y, b.z) else null, b.rotation,
                    b.material.takeIf { b.changed }, b.power
                )
            },
            objects = objects.map { o ->
                val shape = when (o.shape) {
                    is Shape.Model -> "model"
                    Shape.Ball -> "ball"
                    Shape.Block -> "block"
                }
                SavedObject(o.name, shape, o.source, o.x, o.y, o.z, o.rotation, o.height, o.width, o.depth, o.zUp, o.material, o.fileColours)
            },
            depth = maxDepth,
            walls = walls.filter { it.changed }.map { SavedWall(it.part.name, it.material) }
        )
        SceneFile.write(file, saved)
        status = "Zapisano scenę ${file.name}"
        panel.repaint()
    }

    /**
     * Reads [file] off the UI thread - loading its models, each file once - and then puts the
     * window in that state all at once. [onDone] gets an error that stopped the load, or null,
     * and warnings about what could not be restored (a missing model file, other scene switches).
     */
    fun loadScene(file: File, onDone: (String?, List<String>) -> Unit) {
        status = "Wczytuję scenę ${file.name}…"
        panel.repaint()
        Thread({
            val warnings = mutableListOf<String>()
            try {
                val s = SceneFile.read(file)
                require(s.scene == scene.name) {
                    "Plik ${file.name} jest dla sceny ${s.scene}, a okno pokazuje ${scene.name}.\n" +
                        "Otwórz go od początku:  mvn exec:java \"-Drt.gui.load=${file.path}\""
                }
                s.properties.forEach { (key, value) ->
                    if (System.getProperty(key) != value) {
                        warnings += "Scena była zapisana z -D$key=$value; otwórz ją przez -Drt.gui.load, żeby to odtworzyć."
                    }
                }
                val meshes = HashMap<File, rayTraceTypescript.objects.Mesh>()
                val loaded = s.objects.mapNotNull { o ->
                    val shape = when (o.shape) {
                        "ball" -> Shape.Ball
                        "block" -> Shape.Block
                        else -> {
                            val source = o.file
                            if (source == null || !source.isFile) {
                                warnings += "Nie ma pliku modelu ${source?.path ?: o.name} - pominięty."
                                return@mapNotNull null
                            }
                            status = "Wczytuję ${source.name}…"
                            panel.repaint()
                            try {
                                Shape.Model(meshes.getOrPut(source) { MeshLoader.load(source.toPath()) })
                            } catch (t: Throwable) {
                                warnings += "Nie udało się wczytać ${source.name}: ${t.message} - pominięty."
                                return@mapNotNull null
                            }
                        }
                    }
                    PlacedObject(o.name, shape, if (shape is Shape.Model) o.file else null).apply {
                        x = o.x
                        y = o.y
                        z = o.z
                        rotation = o.rotation
                        height = o.height
                        width = o.width
                        depth = o.depth
                        zUp = o.zUp
                        material = o.material
                        fileColours = o.fileColours && hasFileColours
                        prepare()
                    }
                }

                objects.clear()
                objects.addAll(loaded)
                builtIns.forEach { b ->
                    val saved = s.parts.firstOrNull { it.name == b.part.name } ?: return@forEach
                    b.enabled = saved.enabled
                    b.returnHome()
                    b.material = saved.material ?: b.original
                    b.power = saved.power
                    saved.position?.let { p ->
                        b.x = p[0]
                        b.y = p[1]
                        b.z = p[2]
                        b.rotation = saved.rotation
                    }
                }
                selected = null
                targetSamples = s.samples.coerceAtLeast(1)
                if (s.depth > 0) maxDepth = s.depth
                walls.forEach { w -> w.material = s.walls.firstOrNull { it.name == w.part.name }?.material ?: w.original }
                raytrace = s.raytrace
                denoiseOn = s.denoise
                view = View(doubleArrayOf(s.camera[0], s.camera[1], s.camera[2]), s.camera[3], s.camera[4], s.camera[5], s.camera[6])
                resolution = Resolution(s.width.coerceIn(64, 8192), s.height.coerceIn(64, 8192))
                sceneChanged()
                status = "Wczytano scenę ${file.name}"
                panel.repaint()
                SwingUtilities.invokeLater { onDone(null, warnings) }
            } catch (t: Throwable) {
                status = "Nie wczytano sceny ${file.name}"
                panel.repaint()
                SwingUtilities.invokeLater { onDone(t.message ?: t.toString(), warnings) }
            }
        }, "viewer-scene-load").apply { isDaemon = true; start() }
    }

    // ------------------------------------------------------------------ rendering (render thread)

    /**
     * The scene as it now stands: the room without any of its parts, the parts that are
     * switched on, and the added models.
     */
    private fun buildWorld(): Hittable {
        val parts = mutableListOf<Hittable>(sceneWithoutModels())
        for (o in objects) parts += o.hittable()
        return HittableList(parts)
    }

    /**
     * The room and the parts of the scene switched on, where they now stand - everything but the
     * added models - leaving out [except] (the part being dragged, so it does not land on itself).
     */
    private fun sceneWithoutModels(except: SceneEntry? = null): Hittable {
        val world = scene.world
        val all = builtIns.map { it.part.hittable } + walls.map { it.part.hittable }
        val room = if (world is HittableList && all.isNotEmpty()) {
            HittableList(world.objects.filter { o -> all.none { it === o } }.toMutableList())
        } else world
        val parts = mutableListOf<Hittable>(room)
        walls.forEach { parts += it.hittable() }
        builtIns.filter { it.enabled && it !== except }.forEach { parts += it.hittable() }
        return HittableList(parts)
    }

    /**
     * The lights to aim samples at: the camera's list without the stand-ins of any part, plus
     * those of the parts switched on. Null when nothing is left to aim at.
     */
    private fun activeLights(): Hittable? {
        val proxies = builtIns.mapNotNull { it.part.lightProxy }
        fun strip(h: Hittable): Hittable? = when {
            proxies.any { it === h } -> null
            h is HittableList -> h.objects.mapNotNull { strip(it) }.takeIf { it.isNotEmpty() }?.let { HittableList(it.toMutableList()) }
            else -> h
        }
        val kept = listOfNotNull(base.lights?.let { strip(it) }) +
            builtIns.filter { it.enabled }.mapNotNull { it.lightProxy() }
        return kept.takeIf { it.isNotEmpty() }?.let { HittableList(it.toMutableList()) }
    }

    private fun renderLoop() {
        var session: GpuSession? = null
        var capacity = 0
        try {
            GpuRenderer.requestedDenoiser().use { denoiser ->
                val open = { pixels: Int ->
                    val start = System.nanoTime()
                    val buffers = SceneFlattener.flatten(buildWorld(), activeLights())
                    val flattened = System.nanoTime()
                    val current = session
                    // An edited scene goes into the session there is; a bigger image needs a new one.
                    val next = if (current != null && pixels <= capacity) {
                        current.also { it.replaceScene(buffers) }
                    } else {
                        current?.close()
                        session = null
                        GpuSession.open(buffers, pixels).also {
                            session = it
                            capacity = pixels
                        }
                    }
                    System.err.println("[viewer] scene rebuilt: flatten ${(flattened - start) / 1_000_000} ms, upload ${(System.nanoTime() - flattened) / 1_000_000} ms")
                    next
                }
                try {
                    loop(denoiser, open)
                } finally {
                    // Before the denoiser's library is freed, whatever it is still running ends.
                    stopBackground()
                }
            }
        } catch (t: Throwable) {
            status = "Błąd: ${t.message}"
            panel.repaint()
            t.printStackTrace()
        } finally {
            session?.close()
        }
    }

    private fun loop(denoiser: Denoiser?, open: (Int) -> GpuSession) {
        var session: GpuSession? = null
        var builtVersion = -1
        var sessionPixels = 0

        var accumulated: View? = null
        var accumulatedSize: Resolution? = null
        var accumulatedRaytrace = false
        var accumulatedDepth = 0
        var setup: CameraSetup? = null
        var pixels = FloatArray(0)
        var samples = 0
        var slice = 1
        var lastShown = 0L
        var nextDenoise = FIRST_DENOISE
        var guides: Pair<FloatArray, FloatArray>? = null
        var denoised: FloatArray? = null
        var finalDenoiseDone = false
        var rateSamples = 0L
        var rateStart = System.nanoTime()
        var kernelNs = 0L
        // Time spent denoising, on the denoise thread and in the hand-over to it.
        val denoiseNs = AtomicLong()
        var finishedSeconds = 0.0
        // The denoiser runs beside the sampling: it takes a snapshot and the GPU carries on.
        // A result belongs to the accumulation it was made from (epoch) and is dropped otherwise.
        var epoch = 0
        var pendingDenoise: Future<FloatArray>? = null
        var pendingEpoch = 0
        // Turning the buffer into a picture also happens off the render thread, one at a time.
        var displayPixels = FloatArray(0)
        val displayBusy = AtomicBoolean(false)
        var displayIntervalNs = DISPLAY_INTERVAL_NS
        var lastShownState: List<Any?>? = null

        while (running) {
            // A film takes the GPU over until it is done; the view then starts afresh.
            val job = filmJob
            if (job != null) {
                filmJob = null
                val filmPixels = maxOf(sessionPixels, job.size.pixels)
                runFilm(job, denoiser, filmPixels, open)?.let {
                    session = it
                    sessionPixels = filmPixels
                }
                accumulated = null
                continue
            }
            val size = resolution
            val version = sceneVersion.get()
            // Rebuild the GPU copy when the objects changed (once edits pause) or the image outgrew it.
            val sceneStale = version != builtVersion && System.currentTimeMillis() - lastSceneEdit > REBUILD_DELAY_MS
            if (session == null || sceneStale || size.pixels > sessionPixels) {
                sessionPixels = maxOf(size.pixels, sceneResolution.pixels)
                session = open(sessionPixels)
                builtVersion = version
                accumulated = null
            }

            val current = view
            if (!raytrace) {
                // The quick look: full size from the first frame, a few jittered rays per pixel
                // added while the view holds still to smooth the edges, then idle.
                if (current !== accumulated || size != accumulatedSize || accumulatedRaytrace) {
                    setup = setup(current, size)
                    session.setCamera(setup)
                    pixels = FloatArray(setup.totalPixels * 3)
                    accumulated = current
                    accumulatedSize = size
                    accumulatedRaytrace = false
                    samples = 0
                }
                val s = setup!!
                if (samples < QUICK_SAMPLES || saveRequested) {
                    val frameStart = System.nanoTime()
                    if (samples < QUICK_SAMPLES) {
                        session.launch(GpuSession.MODE_SHADED, s, samples, 1, 1.0f)
                        samples++
                    }
                    session.read(GpuSession.IMAGE, pixels)
                    show(
                        toImage(pixels, s.imageWidth, s.imageHeight, 1.0f / samples),
                        "szybki podgląd ${s.imageWidth}×${s.imageHeight} · T: ray tracing"
                    )
                    val shown = samples
                    SwingUtilities.invokeLater { if (::controls.isInitialized) controls.showProgress(shown, QUICK_SAMPLES) }
                    if (saveRequested) save(s, pixels, samples, null)
                    val frameMs = (System.nanoTime() - frameStart) / 1_000_000
                    if (frameMs < FRAME_MS) Thread.sleep(FRAME_MS - frameMs)
                } else {
                    Thread.sleep(FRAME_MS)
                }
                continue
            }

            val moving = dragging || System.currentTimeMillis() - lastInput < SETTLE_MS
            if (moving) {
                // Quarter size, one fresh sample: rough, but it follows the mouse.
                val preview = Resolution(maxOf(size.width / PREVIEW_DIVISOR, 32), maxOf(size.height / PREVIEW_DIVISOR, 32))
                val previewSetup = setup(current, preview)
                val frameStart = System.nanoTime()
                session.setCamera(previewSetup)
                // Always from sample 0: a launch starting later adds to the buffer instead of
                // replacing it. The noise then holds still between frames, which reads better.
                session.launch(GpuSession.MODE_IMAGE, previewSetup, 0, PREVIEW_SAMPLES, 1.0f / PREVIEW_SAMPLES)
                val image = FloatArray(previewSetup.totalPixels * 3).also { session.read(GpuSession.IMAGE, it) }
                show(
                    toImage(image, previewSetup.imageWidth, previewSetup.imageHeight, 1.0f),
                    "podgląd ${previewSetup.imageWidth}×${previewSetup.imageHeight}, $PREVIEW_SAMPLES spp"
                )
                accumulated = null
                val frameMs = (System.nanoTime() - frameStart) / 1_000_000
                if (frameMs < FRAME_MS) Thread.sleep(FRAME_MS - frameMs)
                continue
            }

            if (current !== accumulated || size != accumulatedSize || !accumulatedRaytrace || maxDepth != accumulatedDepth) {
                // Noted before the setup reads it: a change in between restarts on the next pass.
                accumulatedDepth = maxDepth
                setup = setup(current, size)
                session.setCamera(setup)
                pixels = FloatArray(setup.totalPixels * 3)
                displayPixels = FloatArray(setup.totalPixels * 3)
                accumulated = current
                accumulatedSize = size
                accumulatedRaytrace = true
                samples = 0
                slice = 1
                nextDenoise = FIRST_DENOISE
                guides = null
                denoised = null
                finalDenoiseDone = false
                epoch++
                lastShownState = null
                rateSamples = 0
                rateStart = System.nanoTime()
                kernelNs = 0L
                denoiseNs.set(0L)
            }
            val s = setup!!
            val target = targetSamples

            if (samples < target) {
                // The sum of all samples so far; the display divides by their count.
                val count = minOf(slice, target - samples)
                val launchStart = System.nanoTime()
                session.launch(GpuSession.MODE_IMAGE, s, samples, count, 1.0f)
                val launchNs = System.nanoTime() - launchStart
                val launchMs = maxOf(launchNs / 1_000_000.0, 1.0)
                samples += count
                rateSamples += count.toLong() * s.totalPixels
                kernelNs += launchNs
                slice = (count * LAUNCH_MS / launchMs).toInt().coerceIn(1, count * 2)
                if (samples >= target) {
                    val wall = (System.nanoTime() - rateStart) / 1e9
                    finishedSeconds = wall
                    System.err.printf(
                        "[viewer] %d spp at %dx%d: %.1f s, of it kernel %.1f s (%.1f M samples/s), denoising %.1f s%n",
                        samples, s.imageWidth, s.imageHeight, wall, kernelNs / 1e9,
                        rateSamples / (kernelNs / 1e9) / 1e6, denoiseNs.get() / 1e9
                    )
                }
            }
            val done = samples >= target

            // A finished denoise replaces the picture, unless the view changed since it began.
            pendingDenoise?.takeIf { it.isDone }?.let { finished ->
                pendingDenoise = null
                if (pendingEpoch == epoch) {
                    try {
                        denoised = finished.get()
                    } catch (t: Exception) {
                        System.err.println("[viewer] denoising failed: ${t.cause?.message ?: t.message}")
                    }
                }
            }

            val now = System.nanoTime()
            val denoiseNow = denoiser != null && denoiseOn && pendingDenoise == null &&
                ((samples >= nextDenoise && !done) || (done && !finalDenoiseDone))
            if (denoiseNow) {
                val handOverStart = System.nanoTime()
                session.read(GpuSession.IMAGE, pixels)
                val guidePair = guides ?: renderGuides(session, s).also { guides = it }
                val average = FloatArray(pixels.size) { pixels[it] / samples }
                val width = s.imageWidth
                val height = s.imageHeight
                pendingEpoch = epoch
                pendingDenoise = denoiseExecutor.submit(Callable {
                    val start = System.nanoTime()
                    try {
                        denoiser!!.denoise(width, height, average, guidePair.first, guidePair.second)
                    } finally {
                        denoiseNs.addAndGet(System.nanoTime() - start)
                    }
                })
                while (nextDenoise <= samples) nextDenoise *= 4
                if (done) finalDenoiseDone = true
                denoiseNs.addAndGet(System.nanoTime() - handOverStart)
            }
            val cleanShown = denoiseOn && denoised != null
            // Once the picture is final there is nothing new to draw until something changes.
            val state = listOf(samples, if (cleanShown) System.identityHashCode(denoised) else 0, denoiseOn, pendingDenoise != null)
            val changed = !(done && state == lastShownState)
            if (saveRequested) {
                session.read(GpuSession.IMAGE, pixels)
                showNow(s, pixels, if (cleanShown) denoised else null, samples, target, done, finishedSeconds,
                    rateSamples, kernelNs, denoiser != null, nextDenoise)
                lastShownState = state
                lastShown = now
                save(s, pixels, samples, if (cleanShown) denoised else null)
            } else if (changed && now - lastShown > displayIntervalNs && displayBusy.compareAndSet(false, true)) {
                val readStart = System.nanoTime()
                // The finished picture needs no read at all; a noisy one a single copy.
                if (!cleanShown) session.read(GpuSession.IMAGE, displayPixels)
                val readNs = System.nanoTime() - readStart
                // Reading stalls the GPU: spread the reads out when they are slow.
                displayIntervalNs = maxOf(DISPLAY_INTERVAL_NS, readNs * DISPLAY_READ_SHARE)
                val noisy = displayPixels
                val clean = if (cleanShown) denoised else null
                val count = samples
                val rate = rateSamples
                val kernel = kernelNs
                val seconds = finishedSeconds
                val denoising = denoiser != null
                val nextAt = nextDenoise
                displayExecutor.execute {
                    try {
                        showNow(s, noisy, clean, count, target, done, seconds, rate, kernel, denoising, nextAt)
                    } finally {
                        displayBusy.set(false)
                    }
                }
                lastShownState = state
                lastShown = now
            }
            if (done) Thread.sleep(FRAME_MS)
        }
    }

    /** Turns the accumulated picture into an image and puts it in the window, with its status line. */
    private fun showNow(
        s: CameraSetup, noisy: FloatArray, clean: FloatArray?, samples: Int, target: Int, done: Boolean,
        finishedSeconds: Double, rateSamples: Long, kernelNs: Long, hasDenoiser: Boolean, nextDenoise: Int
    ) {
        // The GPU's own speed: samples over the time spent in their launches. Wall time
        // would count the denoiser's pauses too, which early on outlast the sampling.
        val rate = if (kernelNs > 0) rateSamples / (kernelNs / 1e9) / 1e6 else 0.0
        val image = if (clean != null) toImage(clean, s.imageWidth, s.imageHeight, 1.0f)
        else toImage(noisy, s.imageWidth, s.imageHeight, 1.0f / samples)
        val denoiseNote = when {
            !hasDenoiser -> "bez odszumiania (brak OIDN)"
            !denoiseOn -> "odszumianie wył."
            clean != null && done -> "odszumione, gotowe"
            clean != null -> "odszumione"
            else -> "odszumię przy ${minOf(nextDenoise, target)} spp"
        }
        val speed = if (done) " · %.1f s".format(finishedSeconds) else " · %.1f M próbek/s".format(rate)
        show(image, "${s.imageWidth}×${s.imageHeight} · $samples / $target spp$speed · $denoiseNote")
        SwingUtilities.invokeLater { if (::controls.isInitialized) controls.showProgress(samples, target) }
    }

    /** Lets the denoise and display threads finish: the denoiser's library must not be freed under them. */
    private fun stopBackground() {
        denoiseExecutor.shutdown()
        denoiseExecutor.awaitTermination(60, TimeUnit.SECONDS)
        displayExecutor.shutdown()
        displayExecutor.awaitTermination(5, TimeUnit.SECONDS)
    }

    /** The denoiser's albedo and normal guides for the current view, averaged over a few samples. */
    private fun renderGuides(session: GpuSession, setup: CameraSetup): Pair<FloatArray, FloatArray> {
        fun guide(mode: Int): FloatArray {
            // In slices, so a heavy scene at a large size never holds the GPU past the watchdog.
            session.traceSliced(mode, setup, GUIDE_SAMPLES, GpuSession.GUIDE)
            return FloatArray(setup.totalPixels * 3).also { session.read(GpuSession.GUIDE, it) }
        }
        return guide(GpuSession.MODE_ALBEDO) to guide(GpuSession.MODE_NORMAL)
    }

    private fun save(setup: CameraSetup, sums: FloatArray, samples: Int, denoised: FloatArray?) {
        saveRequested = false
        val base = savePath ?: saveFolder.resolve(
            "viewer-${scene.name}-${LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"))}"
        )
        val path = Path.of("$base.ppm")
        try {
            path.parent?.let { java.nio.file.Files.createDirectories(it) }
            PpmWriter.write(path, setup.imageWidth, setup.imageHeight, FloatArray(sums.size) { sums[it] / samples })
            if (denoised != null) {
                PpmWriter.write(Path.of("$base-denoised.ppm"), setup.imageWidth, setup.imageHeight, denoised)
            }
            status = "Zapisano ${base.fileName}.png / .ppm${if (denoised != null) " i -denoised" else ""} ($samples spp)"
            System.err.println("[viewer] saved $path ($samples spp)")
        } catch (e: Exception) {
            status = "Nie udało się zapisać ${path.fileName}: ${e.message}"
            System.err.println("[viewer] could not save $path: $e")
        }
    }

    /** The scene's own camera, moved to [view] and sized to [size]. */
    private fun setup(view: View, size: Resolution): CameraSetup {
        val from = view.from
        val camera = Camera().apply {
            // A quarter pixel over the height keeps the camera's floor(width / aspect) at size.height.
            aspectRatio = size.width / (size.height + 0.25f)
            imageWidth = size.width
            samplesPerPixel = base.samplesPerPixel
            maxReflectionDepth = maxDepth
            vfov = view.vfov.toFloat()
            lookFrom = Point(from[0].toFloat(), from[1].toFloat(), from[2].toFloat())
            lookAt = Point(view.target[0].toFloat(), view.target[1].toFloat(), view.target[2].toFloat())
            vUp = base.vUp
            defocusAngle = base.defocusAngle
            // Keep what the camera looks at in focus when there is depth of field.
            focusDistance = if (base.defocusAngle > 0f) view.radius.toFloat() else base.focusDistance
            background = base.background
            lights = base.lights
        }
        return camera.initialize()
    }

    /**
     * Where the ray through the image point ([fx], [fy], as fractions of its width and height)
     * meets the horizontal plane y = [floorY]; null when it points away from it.
     */
    private fun floorPoint(v: View, fx: Double, fy: Double, floorY: Double): DoubleArray? {
        val (origin, direction) = cursorRay(v, fx, fy)
        if (kotlin.math.abs(direction[1]) < 1e-9) return null
        val t = (floorY - origin[1]) / direction[1]
        if (t <= 0) return null
        return DoubleArray(3) { origin[it] + t * direction[it] }
    }

    /** The camera's ray through the image point ([fx], [fy], as fractions of its width and height). */
    private fun cursorRay(v: View, fx: Double, fy: Double): Pair<DoubleArray, DoubleArray> {
        val size = resolution
        val s = setup(v, size)
        val px = fx * size.width
        val py = fy * size.height
        val point = doubleArrayOf(
            (s.pixel00.x + s.pixelDeltaU.x * px + s.pixelDeltaV.x * py),
            (s.pixel00.y + s.pixelDeltaU.y * px + s.pixelDeltaV.y * py),
            (s.pixel00.z + s.pixelDeltaU.z * px + s.pixelDeltaV.z * py)
        )
        val origin = doubleArrayOf(s.cameraCenter.x.toDouble(), s.cameraCenter.y.toDouble(), s.cameraCenter.z.toDouble())
        return origin to DoubleArray(3) { point[it] - origin[it] }
    }

    /**
     * Where [model] would stand if dropped under the panel point ([mouseX], [mouseY]): on the
     * first surface the ray meets - the scene's own geometry, or the box around another added
     * model - when that surface faces up (a floor, a block's top, a sphere's crown); on the floor
     * (y = 0) under the cursor when it is a wall or a side. Null outside the image.
     */
    private fun dropPoint(model: Movable, mouseX: Int, mouseY: Int): DoubleArray? {
        val r = panel.imageBounds()
        if (r.width <= 0 || r.height <= 0) return null
        val fx = (mouseX - r.x).toDouble() / r.width
        val fy = (mouseY - r.y).toDouble() / r.height
        if (fx !in 0.0..1.0 || fy !in 0.0..1.0) return null
        val v = view
        val (origin, direction) = cursorRay(v, fx, fy)

        var bestT = Double.POSITIVE_INFINITY
        // The hit surface's normal, which way it faces.
        var normal = doubleArrayOf(0.0, 1.0, 0.0)
        val ray = Ray(
            Point(origin[0].toFloat(), origin[1].toFloat(), origin[2].toFloat()),
            Vector(direction[0].toFloat(), direction[1].toFloat(), direction[2].toFloat())
        )
        sceneWithoutModels(except = model).hit(ray, Interval(0.001f, Float.POSITIVE_INFINITY))?.let { hit ->
            bestT = hit.t.toDouble()
            val length = sqrt(hit.normal.x * hit.normal.x + hit.normal.y * hit.normal.y + hit.normal.z * hit.normal.z)
            if (length > 0) normal = doubleArrayOf((hit.normal.x / length).toDouble(), (hit.normal.y / length).toDouble(), (hit.normal.z / length).toDouble())
        }
        for (other in objects) {
            if (other === model) continue
            val b = other.localBounds()
            val min = doubleArrayOf(other.x + b[0], other.y + b[1], other.z + b[2])
            val max = doubleArrayOf(other.x + b[3], other.y + b[4], other.z + b[5])
            boxHit(origin, direction, min, max)?.let { (t, faceNormal) ->
                if (t < bestT) {
                    bestT = t
                    normal = faceNormal
                }
            }
        }

        if (!bestT.isFinite()) return floorPoint(v, fx, fy, 0.0)
        val hit = DoubleArray(3) { origin[it] + bestT * direction[it] }
        if (normal[1] > UPWARD) return hit

        // A wall or a side: stand at its foot on the floor, moved out along it by half the
        // model's footprint so the two do not overlap.
        val b = model.localBounds()
        val halfFootprint = maxOf(-b[0], b[3], -b[2], b[5])
        val flat = sqrt(normal[0] * normal[0] + normal[2] * normal[2]).coerceAtLeast(1e-9)
        return doubleArrayOf(hit[0] + normal[0] / flat * halfFootprint, 0.0, hit[2] + normal[2] / flat * halfFootprint)
    }

    /** Entry distance of a ray into a box and the normal of the face it enters by; null on a miss. */
    private fun boxHit(origin: DoubleArray, direction: DoubleArray, min: DoubleArray, max: DoubleArray): Pair<Double, DoubleArray>? {
        var near = 0.0
        var far = Double.POSITIVE_INFINITY
        var nearAxis = -1
        for (a in 0 until 3) {
            if (kotlin.math.abs(direction[a]) < 1e-12) {
                if (origin[a] < min[a] || origin[a] > max[a]) return null
                continue
            }
            var t0 = (min[a] - origin[a]) / direction[a]
            var t1 = (max[a] - origin[a]) / direction[a]
            if (t0 > t1) t0 = t1.also { t1 = t0 }
            if (t0 > near) {
                near = t0
                nearAxis = a
            }
            if (t1 < far) far = t1
            if (far < near) return null
        }
        // The entry face is across the axis entered last, facing back towards the ray.
        val normal = DoubleArray(3)
        if (nearAxis >= 0) normal[nearAxis] = if (direction[nearAxis] > 0) -1.0 else 1.0
        return near to normal
    }

    /** Where a world point lands in the image, in image pixels; null when it is behind the camera. */
    private fun project(s: CameraSetup, p: DoubleArray): DoubleArray? {
        val c = doubleArrayOf(s.cameraCenter.x.toDouble(), s.cameraCenter.y.toDouble(), s.cameraCenter.z.toDouble())
        val du = doubleArrayOf(s.pixelDeltaU.x.toDouble(), s.pixelDeltaU.y.toDouble(), s.pixelDeltaU.z.toDouble())
        val dv = doubleArrayOf(s.pixelDeltaV.x.toDouble(), s.pixelDeltaV.y.toDouble(), s.pixelDeltaV.z.toDouble())
        val p00 = doubleArrayOf(s.pixel00.x.toDouble(), s.pixel00.y.toDouble(), s.pixel00.z.toDouble())
        fun dot(a: DoubleArray, b: DoubleArray) = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        // The image plane holds pixel00 and is spanned by the pixel steps; meet it on the way to p.
        val normal = doubleArrayOf(du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0])
        val d = DoubleArray(3) { p[it] - c[it] }
        val denominator = dot(normal, d)
        if (kotlin.math.abs(denominator) < 1e-12) return null
        val t = dot(normal, DoubleArray(3) { p00[it] - c[it] }) / denominator
        if (t <= 0) return null
        val onPlane = DoubleArray(3) { c[it] + t * d[it] - p00[it] }
        // pixel00 is the centre of the first pixel, half a pixel in from the corner.
        return doubleArrayOf(dot(onPlane, du) / dot(du, du) + 0.5, dot(onPlane, dv) / dot(dv, dv) + 0.5)
    }

    private fun show(image: BufferedImage, text: String) {
        shown = image
        status = text
        panel.repaint()
    }

    /** Linear values times [scale] -> sRGB-ish 8 bit, the same sqrt gamma and clamp as the files. */
    private fun toImage(linear: FloatArray, width: Int, height: Int, scale: Float): BufferedImage {
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        val out = (image.raster.dataBuffer as DataBufferInt).data
        for (pixel in 0 until width * height) {
            var rgb = 0
            for (channel in 0 until 3) {
                val value = linear[pixel * 3 + channel] * scale
                val gamma = if (value > 0f) sqrt(value).coerceAtMost(0.999f) else 0f
                rgb = (rgb shl 8) or (256f * gamma).toInt()
            }
            out[pixel] = rgb
        }
        return image
    }

    // ------------------------------------------------------------------ mouse and keys

    private fun installControls(frame: JFrame) {
        val mouse = object : MouseAdapter() {
            private var lastX = 0
            private var lastY = 0
            private var placing = false

            override fun mousePressed(e: MouseEvent) {
                panel.requestFocusInWindow()
                lastX = e.x
                lastY = e.y
                placing = SwingUtilities.isLeftMouseButton(e) && e.isControlDown && selectedMovable != null
                if (placing) {
                    updateGhost(e.x, e.y)
                    return
                }
                dragging = true
                touch()
            }

            override fun mouseReleased(e: MouseEvent) {
                if (placing) {
                    placing = false
                    updateGhost(e.x, e.y)
                    dropGhost()
                    return
                }
                dragging = false
                touch()
            }

            override fun mouseDragged(e: MouseEvent) {
                if (placing) {
                    updateGhost(e.x, e.y)
                    return
                }
                val dx = e.x - lastX
                val dy = e.y - lastY
                lastX = e.x
                lastY = e.y
                val v = view
                view = when {
                    SwingUtilities.isMiddleMouseButton(e) || (SwingUtilities.isLeftMouseButton(e) && e.isShiftDown) -> pan(v, dx, dy)
                    SwingUtilities.isRightMouseButton(e) -> v.with(radius = dolly(v.radius, dy * DRAG_ZOOM))
                    else -> v.with(
                        yaw = v.yaw - dx * ORBIT_SPEED,
                        pitch = (v.pitch + dy * ORBIT_SPEED).coerceIn(-MAX_PITCH, MAX_PITCH)
                    )
                }
                touch()
            }

            override fun mouseWheelMoved(e: MouseWheelEvent) {
                view = view.with(radius = dolly(view.radius, e.preciseWheelRotation))
                touch()
            }
        }
        panel.addMouseListener(mouse)
        panel.addMouseMotionListener(mouse)
        panel.addMouseWheelListener(mouse)

        // Shortcuts work wherever the focus is in the window - on the image, a button, the object
        // list - except in a field being typed into, where keys are text. On the list the arrows
        // keep choosing rows.
        java.awt.KeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher { e ->
            if (e.id != KeyEvent.KEY_PRESSED || !frame.isActive) return@addKeyEventDispatcher false
            val owner = java.awt.KeyboardFocusManager.getCurrentKeyboardFocusManager().focusOwner
            if (owner is javax.swing.text.JTextComponent) return@addKeyEventDispatcher false
            val arrow = e.keyCode in intArrayOf(KeyEvent.VK_LEFT, KeyEvent.VK_RIGHT, KeyEvent.VK_UP, KeyEvent.VK_DOWN)
            if (arrow && owner is javax.swing.JList<*>) return@addKeyEventDispatcher false
            // Space and Enter press the focused button, Esc closes an open drop-down: leave those
            // to the control. Esc closes the window only from the image.
            val controlKey = e.keyCode == KeyEvent.VK_SPACE || e.keyCode == KeyEvent.VK_ENTER || e.keyCode == KeyEvent.VK_ESCAPE
            if (controlKey && owner !== panel) return@addKeyEventDispatcher false
            shortcut(e, frame)
        }
    }

    /** Runs the shortcut for [e]; false when the key is not one. */
    private fun shortcut(e: KeyEvent, frame: JFrame): Boolean {
        when (e.keyCode) {
            KeyEvent.VK_R -> { view = initialView; touch() }
            KeyEvent.VK_D -> { denoiseOn = !denoiseOn; controls.syncDenoise() }
            KeyEvent.VK_T -> { raytrace = !raytrace; controls.syncMode() }
            KeyEvent.VK_S -> if (e.isControlDown) controls.chooseSceneToSave() else chooseSave(frame)
            KeyEvent.VK_O -> if (e.isControlDown) controls.chooseSceneToLoad() else return false
            KeyEvent.VK_PLUS, KeyEvent.VK_ADD, KeyEvent.VK_EQUALS -> { view = view.with(vfov = (view.vfov - 2).coerceAtLeast(5.0)); touch() }
            KeyEvent.VK_MINUS, KeyEvent.VK_SUBTRACT -> { view = view.with(vfov = (view.vfov + 2).coerceAtMost(120.0)); touch() }
            KeyEvent.VK_DELETE -> selected?.let { removeObject(it); controls.refreshObjects() }
            KeyEvent.VK_F1, KeyEvent.VK_H -> HelpDialog.show(frame)
            KeyEvent.VK_LEFT, KeyEvent.VK_RIGHT, KeyEvent.VK_UP, KeyEvent.VK_DOWN,
            KeyEvent.VK_PAGE_UP, KeyEvent.VK_PAGE_DOWN, KeyEvent.VK_Q, KeyEvent.VK_E,
            KeyEvent.VK_OPEN_BRACKET, KeyEvent.VK_CLOSE_BRACKET -> nudge(e.keyCode, e.isShiftDown)
            KeyEvent.VK_ESCAPE -> frame.dispose()
            else -> return false
        }
        return true
    }

    /** Ctrl + drag: where the selected model would land, shown as its box until the button is let go. */
    private fun updateGhost(mouseX: Int, mouseY: Int) {
        placeMarker = java.awt.Point(mouseX, mouseY)
        ghost = selectedMovable?.let { dropPoint(it, mouseX, mouseY) }
        panel.repaint()
    }

    /** Ctrl + release: the selected model moves to where its box was shown. */
    private fun dropGhost() {
        val o = selectedMovable
        val target = ghost
        ghost = null
        placeMarker = null
        panel.repaint()
        if (o == null || target == null) return
        o.x = target[0]
        o.y = target[1]
        o.z = target[2]
        sceneChanged()
        controls.refreshSelection()
    }

    /**
     * Arrow keys move the selected model across the floor as the camera sees it - left, right,
     * away, closer - PageUp / PageDown up and down, Q / E or [ / ] turn it; Shift takes five
     * times the step.
     */
    private fun nudge(key: Int, big: Boolean) {
        val o = selectedMovable ?: return
        val step = defaultObjectHeight / 20 * (if (big) 5 else 1)
        val v = view
        // The camera's right and its forward direction, both flattened onto the floor.
        val w = doubleArrayOf(cos(v.pitch) * sin(v.yaw), cos(v.pitch) * cos(v.yaw))
        val length = sqrt(w[0] * w[0] + w[1] * w[1]).coerceAtLeast(1e-9)
        val rightX = w[1] / length
        val rightZ = -w[0] / length
        val forwardX = -w[0] / length
        val forwardZ = -w[1] / length
        when (key) {
            KeyEvent.VK_LEFT -> { o.x -= rightX * step; o.z -= rightZ * step }
            KeyEvent.VK_RIGHT -> { o.x += rightX * step; o.z += rightZ * step }
            KeyEvent.VK_UP -> { o.x += forwardX * step; o.z += forwardZ * step }
            KeyEvent.VK_DOWN -> { o.x -= forwardX * step; o.z -= forwardZ * step }
            KeyEvent.VK_PAGE_UP -> o.y += step
            KeyEvent.VK_PAGE_DOWN -> o.y -= step
            KeyEvent.VK_Q, KeyEvent.VK_OPEN_BRACKET -> o.rotation = ((o.rotation - TURN_STEP + 540) % 360) - 180
            KeyEvent.VK_E, KeyEvent.VK_CLOSE_BRACKET -> o.rotation = ((o.rotation + TURN_STEP + 540) % 360) - 180
        }
        sceneChanged()
        controls.refreshSelection()
    }

    private fun touch() {
        lastInput = System.currentTimeMillis()
    }

    /** Moves target and camera together so the point under the cursor follows it. */
    private fun pan(v: View, dx: Int, dy: Int): View {
        val w = doubleArrayOf(cos(v.pitch) * sin(v.yaw), sin(v.pitch), cos(v.pitch) * cos(v.yaw))
        // u = up x w, v = w x u: the screen's right and up directions in the world.
        val ux = w[2]
        val uz = -w[0]
        val uLength = sqrt(ux * ux + uz * uz).coerceAtLeast(1e-9)
        val u = doubleArrayOf(ux / uLength, 0.0, uz / uLength)
        val up = doubleArrayOf(w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0])
        val shownHeight = panel.imageBounds().height.coerceAtLeast(1)
        val unitsPerPixel = 2.0 * v.radius * tan(Math.toRadians(v.vfov / 2)) / shownHeight
        val target = DoubleArray(3) { v.target[it] + (-dx * u[it] + dy * up[it]) * unitsPerPixel }
        return v.with(target = target)
    }

    private fun dolly(radius: Double, steps: Double): Double = (radius * Math.pow(ZOOM_STEP, steps)).coerceIn(1.0, 1e5)

    /** Draws the latest image, scaled to fit, and the placement marker. */
    private inner class ImagePanel : JPanel() {
        init {
            background = Theme.canvas
            isFocusable = true
        }

        /** Where the image goes: as large as fits, centred, proportions kept. */
        fun imageBounds(): java.awt.Rectangle {
            val available = height.coerceAtLeast(1)
            val aspect = resolution.width.toDouble() / resolution.height
            val w = minOf(width.toDouble(), available * aspect)
            val h = w / aspect
            return java.awt.Rectangle(((width - w) / 2).toInt(), ((available - h) / 2).toInt(), w.toInt(), h.toInt())
        }

        /** The dragged model's box, where it would land, drawn over the image in perspective. */
        private fun drawGhost(g2: Graphics2D) {
            val at = ghost ?: return
            val model = selectedMovable ?: return
            val b = model.localBounds()
            val s = setup(view, resolution)
            val r = imageBounds()
            val corners = (0 until 8).map { i ->
                val p = doubleArrayOf(
                    at[0] + if (i and 1 == 0) b[0] else b[3],
                    at[1] + if (i and 2 == 0) b[1] else b[4],
                    at[2] + if (i and 4 == 0) b[2] else b[5]
                )
                project(s, p)?.let { px ->
                    java.awt.Point(
                        (r.x + px[0] / resolution.width * r.width).toInt(),
                        (r.y + px[1] / resolution.height * r.height).toInt()
                    )
                }
            }
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            g2.color = Theme.accent
            g2.stroke = BasicStroke(1.6f)
            // The 12 edges: corners that differ in exactly one of the three bits.
            for (i in 0 until 8) for (bit in intArrayOf(1, 2, 4)) {
                val j = i or bit
                if (j == i) continue
                val a = corners[i] ?: continue
                val c = corners[j] ?: continue
                g2.drawLine(a.x, a.y, c.x, c.y)
            }
        }

        override fun paintComponent(g: Graphics) {
            super.paintComponent(g)
            val g2 = g as Graphics2D
            shown?.let { image ->
                val r = imageBounds()
                g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
                g2.drawImage(image, r.x, r.y, r.width, r.height, null)
            }
            drawGhost(g2)
            placeMarker?.let { p ->
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
                g2.color = Theme.accent
                g2.stroke = BasicStroke(2f)
                g2.drawOval(p.x - 10, p.y - 10, 20, 20)
                g2.drawLine(p.x - 16, p.y, p.x + 16, p.y)
                g2.drawLine(p.x, p.y - 16, p.x, p.y + 16)
            }
        }
    }

    companion object {
        private const val PREVIEW_DIVISOR = 4
        private const val PREVIEW_SAMPLES = 1
        /** How long after the last mouse or key move the full-size render takes over. */
        private const val SETTLE_MS = 150L
        /** How long after an object edit the scene is rebuilt, so a run of edits costs one rebuild. */
        private const val REBUILD_DELAY_MS = 80L
        private const val FRAME_MS = 16L
        private const val LAUNCH_MS = 40.0
        private const val DISPLAY_INTERVAL_NS = 100_000_000L
        /** A read of the picture may cost the GPU at most 1 / this of the time between two of them. */
        private const val DISPLAY_READ_SHARE = 8L
        private const val DEFAULT_TARGET = 1024
        private const val FIRST_DENOISE = 16
        /** Rays per pixel the quick look gathers once the view holds still, for smooth edges. */
        private const val QUICK_SAMPLES = 16
        private const val GUIDE_SAMPLES = 16
        private const val ORBIT_SPEED = 0.005
        private const val MAX_PITCH = 1.55
        private const val ZOOM_STEP = 1.1
        private const val DRAG_ZOOM = 0.02
        /** How much a model turns per Q / E press, in degrees. */
        private const val TURN_STEP = 15.0
        /** A surface counts as one to stand on when its normal is within ~45 degrees of up. */
        private const val UPWARD = 0.7
        private const val HELP =
            "LPM obrót · Shift+LPM przesuń · kółko zoom · Ctrl+LPM postaw · T tryb · S zapis · F1 skróty"
    }
}
