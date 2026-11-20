package rayTraceTypescript.gui

import java.io.File
import java.util.Properties

/** An added object as a scene file keeps it; a model keeps the path of its file, not the mesh. */
class SavedObject(
    val name: String,
    /** "model", "ball" or "block". */
    val shape: String,
    val file: File?,
    val x: Double,
    val y: Double,
    val z: Double,
    val rotation: Double,
    val height: Double,
    val width: Double,
    val depth: Double,
    val zUp: Boolean,
    val material: MaterialSpec,
    /** Made of the materials its file names rather than [material]. */
    val fileColours: Boolean = false,
    /** What its surface is turned into, and the sponge's hole spacing. */
    val form: Form = Form.SOLID,
    val latticePitch: Double = 0.035
)

/** One of the scene's own objects or lights: whether it is in, and where it stands (null: where the scene put it). */
class SavedPart(
    val name: String,
    val enabled: Boolean,
    val position: DoubleArray? = null,
    val rotation: Double = 0.0,
    /** Its material when changed from the scene's; null keeps the scene's. */
    val material: MaterialSpec? = null,
    /** For a light: how many times its own strength it shines with. */
    val power: Double = 1.0
)

/** A wall whose material was changed. */
class SavedWall(val name: String, val material: MaterialSpec)

/** Everything the viewer lets one change, as a scene file keeps it. */
class SavedScene(
    /** The scene the window was opened with (-Drt.scene). */
    val scene: String,
    /** The switches the scene was built with ([SceneFile.SCENE_PROPERTIES]) that were set. */
    val properties: Map<String, String>,
    /** Where the camera looks (x, y, z), then yaw, pitch, distance and field of view. */
    val camera: DoubleArray,
    val width: Int,
    val height: Int,
    val samples: Int,
    val raytrace: Boolean,
    val denoise: Boolean,
    /** The scene's own objects and lights, whether each is in, and any that were moved. */
    val parts: List<SavedPart>,
    val objects: List<SavedObject>,
    /** How many bounces a path may take; 0 keeps the scene's own. */
    val depth: Int = 0,
    /** Walls made of something other than the scene's own material. */
    val walls: List<SavedWall> = emptyList()
)

/**
 * A viewer scene as a text file (.rtscene, Java properties in UTF-8), readable and editable by
 * hand. Model paths are written relative to the scene file when they can be, so a scene saved
 * inside the repository opens from any checkout.
 */
object SceneFile {
    const val EXTENSION = "rtscene"

    /** Switches that change what a scene is built from, so a load has to rebuild it the same way. */
    val SCENE_PROPERTIES = listOf("rt.block", "rt.fill", "rt.model", "rt.model.turn", "rt.model.material")

    fun write(file: File, s: SavedScene) {
        val lines = mutableListOf(
            "# Scena okna ray tracera - wczytaj w oknie (Wczytaj scenę…) albo: mvn exec:java -Drt.gui \"-Drt.gui.load=${file.name}\"",
            "format=1",
            "scene=${s.scene}"
        )
        s.properties.forEach { (key, value) -> lines += "property.$key=${escape(value)}" }
        lines += "camera.target=${s.camera[0]},${s.camera[1]},${s.camera[2]}"
        lines += "camera.yaw=${s.camera[3]}"
        lines += "camera.pitch=${s.camera[4]}"
        lines += "camera.distance=${s.camera[5]}"
        lines += "camera.vfov=${s.camera[6]}"
        lines += "image.width=${s.width}"
        lines += "image.height=${s.height}"
        lines += "image.samples=${s.samples}"
        if (s.depth > 0) lines += "image.depth=${s.depth}"
        lines += "mode=${if (s.raytrace) "raytrace" else "quick"}"
        lines += "denoise=${s.denoise}"
        s.parts.forEachIndexed { i, part ->
            lines += "part.$i.name=${escape(part.name)}"
            lines += "part.$i.enabled=${part.enabled}"
            part.position?.let { p ->
                lines += "part.$i.position=${p[0]},${p[1]},${p[2]}"
                lines += "part.$i.rotation=${part.rotation}"
            }
            part.material?.let { m ->
                lines += "part.$i.material=${m.kind.name}"
                lines += "part.$i.colour=${m.red},${m.green},${m.blue}"
                lines += "part.$i.parameter=${m.parameter}"
            }
            if (part.power != 1.0) lines += "part.$i.power=${part.power}"
        }
        s.walls.forEachIndexed { i, w ->
            lines += "wall.$i.name=${escape(w.name)}"
            lines += "wall.$i.material=${w.material.kind.name}"
            lines += "wall.$i.colour=${w.material.red},${w.material.green},${w.material.blue}"
            lines += "wall.$i.parameter=${w.material.parameter}"
        }
        s.objects.forEachIndexed { i, o ->
            val p = "object.$i"
            lines += "$p.name=${escape(o.name)}"
            lines += "$p.shape=${o.shape}"
            o.file?.let { lines += "$p.file=${escape(relativePath(file, it))}" }
            lines += "$p.position=${o.x},${o.y},${o.z}"
            lines += "$p.rotation=${o.rotation}"
            lines += "$p.height=${o.height}"
            lines += "$p.width=${o.width}"
            lines += "$p.depth=${o.depth}"
            lines += "$p.zUp=${o.zUp}"
            lines += "$p.material=${o.material.kind.name}"
            lines += "$p.colour=${o.material.red},${o.material.green},${o.material.blue}"
            lines += "$p.parameter=${o.material.parameter}"
            if (o.fileColours) lines += "$p.fileColours=true"
            if (o.form != Form.SOLID) {
                lines += "$p.form=${o.form.name}"
                lines += "$p.lattice=${o.latticePitch}"
            }
        }
        file.absoluteFile.parentFile?.mkdirs()
        file.writeText(lines.joinToString("\n", postfix = "\n"), Charsets.UTF_8)
    }

    fun read(file: File): SavedScene {
        val p = Properties().apply { file.reader(Charsets.UTF_8).use { load(it) } }
        fun text(key: String): String = p.getProperty(key) ?: throw IllegalArgumentException("brak pola $key")
        fun number(key: String): Double = text(key).toDoubleOrNull() ?: throw IllegalArgumentException("$key nie jest liczbą")
        fun numbers(key: String, count: Int): List<Double> = text(key).split(',').map {
            it.trim().toDoubleOrNull() ?: throw IllegalArgumentException("$key: \"$it\" nie jest liczbą")
        }.also { require(it.size == count) { "$key: oczekiwano $count liczb" } }

        fun material(prefix: String): MaterialSpec {
            val colour = numbers("$prefix.colour", 3)
            val kind = MaterialKind.values().firstOrNull { it.name == text("$prefix.material") }
                ?: throw IllegalArgumentException("$prefix.material: nieznany materiał ${text("$prefix.material")}")
            return MaterialSpec(kind, colour[0].toFloat(), colour[1].toFloat(), colour[2].toFloat(), number("$prefix.parameter"))
        }
        val target = numbers("camera.target", 3)
        val parts = mutableListOf<SavedPart>()
        var i = 0
        while (p.getProperty("part.$i.name") != null) {
            val position = p.getProperty("part.$i.position")?.let { numbers("part.$i.position", 3).toDoubleArray() }
            parts += SavedPart(
                text("part.$i.name"), text("part.$i.enabled").toBoolean(), position,
                p.getProperty("part.$i.rotation")?.toDoubleOrNull() ?: 0.0,
                p.getProperty("part.$i.material")?.let { material("part.$i") },
                p.getProperty("part.$i.power")?.toDoubleOrNull() ?: 1.0
            )
            i++
        }
        val walls = mutableListOf<SavedWall>()
        i = 0
        while (p.getProperty("wall.$i.name") != null) {
            walls += SavedWall(text("wall.$i.name"), material("wall.$i"))
            i++
        }
        val objects = mutableListOf<SavedObject>()
        i = 0
        while (p.getProperty("object.$i.shape") != null) {
            val o = "object.$i"
            val position = numbers("$o.position", 3)
            val colour = numbers("$o.colour", 3)
            val kind = MaterialKind.values().firstOrNull { it.name == text("$o.material") }
                ?: throw IllegalArgumentException("$o.material: nieznany materiał ${text("$o.material")}")
            objects += SavedObject(
                name = p.getProperty("$o.name") ?: "Obiekt ${i + 1}",
                shape = text("$o.shape"),
                file = p.getProperty("$o.file")?.let { resolve(file, it) },
                x = position[0], y = position[1], z = position[2],
                rotation = number("$o.rotation"),
                height = number("$o.height"),
                width = number("$o.width"),
                depth = number("$o.depth"),
                zUp = text("$o.zUp").toBoolean(),
                material = MaterialSpec(kind, colour[0].toFloat(), colour[1].toFloat(), colour[2].toFloat(), number("$o.parameter")),
                fileColours = p.getProperty("$o.fileColours")?.toBoolean() ?: false,
                // A file without a form (an older one) holds solids; so does one with the honeycomb, removed.
                form = Form.values().firstOrNull { it.name == p.getProperty("$o.form") } ?: Form.SOLID,
                latticePitch = p.getProperty("$o.lattice")?.split(',')?.getOrNull(0)?.toDoubleOrNull() ?: 0.035
            )
            i++
        }
        return SavedScene(
            scene = text("scene"),
            properties = SCENE_PROPERTIES.mapNotNull { key -> p.getProperty("property.$key")?.let { key to it } }.toMap(),
            camera = doubleArrayOf(
                target[0], target[1], target[2],
                number("camera.yaw"), number("camera.pitch"), number("camera.distance"), number("camera.vfov")
            ),
            width = number("image.width").toInt(),
            height = number("image.height").toInt(),
            samples = number("image.samples").toInt(),
            raytrace = p.getProperty("mode") == "raytrace",
            denoise = p.getProperty("denoise")?.toBoolean() ?: true,
            parts = parts,
            objects = objects,
            depth = p.getProperty("image.depth")?.toIntOrNull() ?: 0,
            walls = walls
        )
    }

    /** The model's path from the scene file's folder when there is one (same drive), else absolute. */
    private fun relativePath(sceneFile: File, model: File): String {
        val base = sceneFile.absoluteFile.parentFile.toPath().normalize()
        val target = model.absoluteFile.toPath().normalize()
        val path = runCatching { base.relativize(target) }.getOrNull() ?: target
        return path.toString().replace('\\', '/')
    }

    private fun resolve(sceneFile: File, path: String): File {
        val f = File(path)
        return if (f.isAbsolute) f else File(sceneFile.absoluteFile.parentFile, path).normalize()
    }

    /** Properties treat a backslash as an escape and a line break as the end of the value. */
    private fun escape(value: String) = value.replace("\\", "\\\\").replace("\n", "\\n")
}
