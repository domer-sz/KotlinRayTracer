package rayTraceTypescript.gui

import rayTraceTypescript.Color
import rayTraceTypescript.Vector
import rayTraceTypescript.gpu.MeshBlock
import rayTraceTypescript.gpu.PlacedMesh
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.DiffuseLight
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Material
import rayTraceTypescript.materials.Metal
import rayTraceTypescript.Point
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.objects.RotateY
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.objects.Translate
import rayTraceTypescript.objects.box
import rayTraceTypescript.scenes.ScenePart

/** An object listed in the viewer: part of the scene itself, or a model added there. */
sealed interface SceneEntry

/** Something whose material the viewer lets one change: an added object, a wall or a scene object. */
sealed interface HasMaterial : SceneEntry {
    var material: MaterialSpec
}

/** A scene's own object or wall: what the scene made it of, to go back to. */
sealed interface SceneMaterial : HasMaterial {
    val original: MaterialSpec
    val changed: Boolean get() = material != original
}

/**
 * One of the room's walls (a [ScenePart] marked as a surface). It stays where it is; its
 * material can be changed, and a changed wall is rebuilt as the same quad made of that.
 */
class WallObject(val part: ScenePart) : SceneMaterial {
    private val quad = part.hittable as? Quad
    /** What the scene made it of, when that is a plain colour the editor can show. */
    override val original: MaterialSpec = quad?.let { MaterialSwap.specOf(it.material) } ?: MaterialPreset.WHITE.spec
    @Volatile override var material: MaterialSpec = original

    /** Whether the editor can remake it: a quad, which every wall of the Cornell box is. */
    val editable: Boolean get() = quad != null

    fun hittable(): Hittable {
        val q = quad
        if (!changed || q == null) return part.hittable
        return Quad(q.q, q.u, q.v, material.material())
    }

    override fun toString() = part.name
}

/**
 * An object that can be moved and turned: [x], [y], [z] are the middle of its footprint and its
 * lowest point, [rotation] its turn about the vertical in degrees.
 */
sealed interface Movable : SceneEntry {
    var x: Double
    var y: Double
    var z: Double
    var rotation: Double

    /** Its box relative to (x, y, z): min x, y, z, max x, y, z. */
    fun localBounds(): FloatArray
}

/**
 * An object or light the scene came with. The viewer can take it out (switch it off), put it
 * back, and move and turn it: it then stands wrapped in a turn about its footprint's middle and
 * a move, and so does its stand-in for light sampling, so the renderer still aims at it.
 * An object's material can be changed - it is then rebuilt, same shapes, in the new one - and a
 * light's [power] scales what it gives off.
 */
class BuiltInObject(val part: ScenePart) : Movable, SceneMaterial {
    @Volatile var enabled = part.enabled

    /**
     * What the scene made it of, as the editor shows it; plain white matte for what it cannot
     * show (the checkered column), which stays checkered until another material is chosen.
     */
    override val original: MaterialSpec =
        MaterialSwap.firstMaterial(part.hittable)?.let { MaterialSwap.specOf(it) } ?: MaterialPreset.WHITE.spec
    @Volatile override var material: MaterialSpec = original

    /** For a light: how many times its own strength it shines with. */
    @Volatile var power = 1.0

    /** Where the scene put it: the middle of its footprint, at its lowest point. */
    private val home: DoubleArray
    /** Its box around [home], unturned. */
    private val homeBounds: FloatArray

    init {
        val b = part.hittable.aabbBoundingBox()
        home = doubleArrayOf(((b.x.min + b.x.max) / 2).toDouble(), b.y.min.toDouble(), ((b.z.min + b.z.max) / 2).toDouble())
        homeBounds = floatArrayOf(
            (b.x.min - home[0]).toFloat(), 0f, (b.z.min - home[2]).toFloat(),
            (b.x.max - home[0]).toFloat(), b.y.max - b.y.min, (b.z.max - home[2]).toFloat()
        )
    }

    @Volatile override var x = home[0]
    @Volatile override var y = home[1]
    @Volatile override var z = home[2]
    @Volatile override var rotation = 0.0

    /** Whether it stands anywhere but where the scene put it. */
    val moved: Boolean get() = x != home[0] || y != home[1] || z != home[2] || rotation != 0.0

    /** Back where the scene put it. */
    fun returnHome() {
        x = home[0]
        y = home[1]
        z = home[2]
        rotation = 0.0
    }

    /** The part as it now stands, in its material or at its power. */
    fun hittable(): Hittable = placed(dressed(part.hittable))

    private fun dressed(h: Hittable): Hittable = when {
        part.isLight -> if (power == 1.0) h else MaterialSwap.rebuild(h) { MaterialSwap.brighter(it, power) }
        changed -> material.material().let { m -> MaterialSwap.rebuild(h) { m } }
        else -> h
    }

    /** Its stand-in for light sampling, moved along with it. */
    fun lightProxy(): Hittable? = part.lightProxy?.let { placed(it) }

    private fun placed(h: Hittable): Hittable {
        if (!moved) return h
        // A part that is a placed mesh stays one, so moving it copies its tree in instead of rebuilding it.
        return rayTraceTypescript.gpu.Placement.moved(
            h,
            Vector(home[0].toFloat(), home[1].toFloat(), home[2].toFloat()),
            rotation.toFloat(),
            Vector(x.toFloat(), y.toFloat(), z.toFloat())
        )
    }

    override fun localBounds(): FloatArray {
        val b = homeBounds
        val a = Math.toRadians(rotation)
        val c = kotlin.math.cos(a)
        val s = kotlin.math.sin(a)
        var minX = Double.POSITIVE_INFINITY; var maxX = Double.NEGATIVE_INFINITY
        var minZ = Double.POSITIVE_INFINITY; var maxZ = Double.NEGATIVE_INFINITY
        for (px in floatArrayOf(b[0], b[3])) for (pz in floatArrayOf(b[2], b[5])) {
            val tx = c * px + s * pz
            val tz = -s * px + c * pz
            minX = minOf(minX, tx); maxX = maxOf(maxX, tx)
            minZ = minOf(minZ, tz); maxZ = maxOf(maxZ, tz)
        }
        return floatArrayOf(minX.toFloat(), b[1], minZ.toFloat(), maxX.toFloat(), b[4], maxZ.toFloat())
    }

    override fun toString() = part.name
}

/**
 * The kinds of material a placed model can be made of, and what their one number means:
 * [parameterLabel] with its range and step, or null when the kind has none.
 */
enum class MaterialKind(
    val label: String,
    val parameterLabel: String?,
    val min: Double = 0.0,
    val max: Double = 1.0,
    val step: Double = 0.05
) {
    MATTE("Matowy", null),
    METAL("Metal", "Chropowatość", 0.0, 1.0, 0.05),
    GLASS("Szkło", "Wsp. załamania", 1.0, 3.0, 0.05),
    LIGHT("Świecący", "Moc", 0.1, 100.0, 0.5);

    override fun toString() = label
}

/**
 * What a placed model is made of: a [kind], a colour ([red], [green], [blue] as albedo or
 * emission, 0..1, linear) and the kind's [parameter] - roughness for metal, refractive index for
 * glass, strength for a light. Glass ignores the colour, as the renderer's glass is clear.
 */
data class MaterialSpec(
    val kind: MaterialKind,
    val red: Float,
    val green: Float,
    val blue: Float,
    val parameter: Double = 0.0
) {
    fun material(): Material = when (kind) {
        MaterialKind.MATTE -> Lambertian(Color(red, green, blue))
        MaterialKind.METAL -> Metal(Color(red, green, blue), parameter.toFloat().coerceIn(0f, 1f))
        MaterialKind.GLASS -> Dielectric(parameter.toFloat().coerceAtLeast(1f))
        MaterialKind.LIGHT -> DiffuseLight(Color(red * parameter.toFloat(), green * parameter.toFloat(), blue * parameter.toFloat()))
    }

    /** The colour as the screen shows it: the renderer displays linear values through a square root. */
    fun screenColor(): java.awt.Color = java.awt.Color(toScreen(red), toScreen(green), toScreen(blue))

    fun withScreenColor(c: java.awt.Color) = copy(red = toLinear(c.red), green = toLinear(c.green), blue = toLinear(c.blue))

    private fun toScreen(v: Float) = kotlin.math.sqrt(v.coerceIn(0f, 1f))
    private fun toLinear(v: Int) = (v / 255f).let { it * it }
}

/** Ready-made materials to start from; any of them can be changed further. */
enum class MaterialPreset(val label: String, val spec: MaterialSpec) {
    GREY("Szary mat", MaterialSpec(MaterialKind.MATTE, 0.45f, 0.45f, 0.45f)),
    WHITE("Biały mat", MaterialSpec(MaterialKind.MATTE, 0.73f, 0.73f, 0.73f)),
    RED("Czerwony mat", MaterialSpec(MaterialKind.MATTE, 0.65f, 0.08f, 0.06f)),
    BLUE("Niebieski mat", MaterialSpec(MaterialKind.MATTE, 0.08f, 0.18f, 0.6f)),
    GLASS("Szkło", MaterialSpec(MaterialKind.GLASS, 1f, 1f, 1f, 1.5)),
    DIAMOND("Diament", MaterialSpec(MaterialKind.GLASS, 1f, 1f, 1f, 2.42)),
    MIRROR("Lustro", MaterialSpec(MaterialKind.METAL, 0.9f, 0.9f, 0.9f, 0.0)),
    GOLD("Złoto", MaterialSpec(MaterialKind.METAL, 0.9f, 0.7f, 0.3f, 0.05)),
    COPPER("Miedź", MaterialSpec(MaterialKind.METAL, 0.95f, 0.64f, 0.54f, 0.15)),
    BRUSHED("Stal szczotkowana", MaterialSpec(MaterialKind.METAL, 0.7f, 0.7f, 0.72f, 0.35)),
    LAMP("Świecący biały", MaterialSpec(MaterialKind.LIGHT, 1f, 1f, 1f, 4.0));

    override fun toString() = label
}

/** What an added object is: a loaded model, or one of the shapes the scene's own objects have. */
sealed interface Shape {
    /** A mesh from a file. */
    class Model(val mesh: Mesh) : Shape

    /** A true sphere, [height] across. */
    object Ball : Shape

    /** A box with flat sides, like the Cornell box's column: [width] by [height] by [depth]. */
    object Block : Shape
}

/**
 * An object added in the viewer and where and how it stands. [x], [y], [z] are the middle of
 * its footprint, [height] its size, [rotation] its turn about the vertical in degrees. A block
 * also has its own [width] and [depth]; a model is scaled so its height is [height]. A model
 * remembers its [source] file, which a saved scene points to.
 */
class PlacedObject(val name: String, val shape: Shape, val source: java.io.File? = null) : Movable, HasMaterial {
    @Volatile override var x = 0.0
    @Volatile override var y = 0.0
    @Volatile override var z = 0.0
    @Volatile override var rotation = 0.0
    @Volatile var height = 1.0
    @Volatile var width = 0.5
    @Volatile var depth = 0.5
    @Volatile var zUp = false
    @Volatile override var material = MaterialPreset.GREY.spec

    val isModel: Boolean get() = shape is Shape.Model

    /** Whether the model's file coloured its faces (an OBJ with a .mtl). */
    val hasFileColours: Boolean get() = (shape as? Shape.Model)?.mesh?.hasColours == true

    /** Made of the file's own materials rather than [material]; on for a coloured file from the start. */
    @Volatile var fileColours: Boolean = hasFileColours

    /** A line about what the object is, for the panel. */
    val description: String
        get() = when (shape) {
            is Shape.Model -> "%,d trójkątów".format(shape.mesh.triangleCount)
            Shape.Ball -> "Kula"
            Shape.Block -> "Graniastosłup"
        }

    /** The object as the renderer takes it, in its place and made of its material. */
    fun hittable(): Hittable {
        val m = material.material()
        return when (shape) {
            is Shape.Model -> placedMesh(shape.mesh, m)
            Shape.Ball -> Sphere(Point(x.toFloat(), (y + height / 2).toFloat(), z.toFloat()), (height / 2).toFloat(), m)
            Shape.Block -> Translate(
                RotateY(
                    box(
                        Point((-width / 2).toFloat(), 0f, (-depth / 2).toFloat()),
                        Point((width / 2).toFloat(), height.toFloat(), (depth / 2).toFloat()),
                        m
                    ),
                    rotation.toFloat()
                ),
                Vector(x.toFloat(), y.toFloat(), z.toFloat())
            )
        }
    }

    @Volatile private var blockZUp: Boolean? = null
    @Volatile private var block: MeshBlock? = null

    /**
     * Builds a model's GPU tree now, on the calling thread - the loader's - so the render thread
     * does not stall on it when the object first joins the scene.
     */
    fun prepare() {
        if (shape is Shape.Model) block(shape.mesh)
    }

    /** The model stood up, with its own GPU tree; built again only when [zUp] changes. */
    private fun block(mesh: Mesh): MeshBlock {
        block?.let { if (blockZUp == zUp) return it }
        return MeshBlock(if (zUp) mesh.zUpToYUp() else mesh).also {
            block = it
            blockZUp = zUp
        }
    }

    /**
     * The model placed as [shapedMesh] places it - turned, scaled to [height], the turned
     * footprint centred on (x, z) and its lowest point at y - but as a transform of the cached
     * [MeshBlock], so moving, turning or resizing it builds no new tree.
     */
    private fun placedMesh(mesh: Mesh, material: Material): Hittable {
        val b = block(mesh)
        val radians = Math.toRadians(rotation)
        val c = kotlin.math.cos(radians).toFloat()
        val s = kotlin.math.sin(radians).toFloat()
        var minX = Float.POSITIVE_INFINITY; var maxX = Float.NEGATIVE_INFINITY
        var minZ = Float.POSITIVE_INFINITY; var maxZ = Float.NEGATIVE_INFINITY
        val v = b.mesh.vertices
        var i = 0
        while (i < v.size) {
            val tx = c * v[i] + s * v[i + 2]
            val tz = -s * v[i] + c * v[i + 2]
            if (tx < minX) minX = tx; if (tx > maxX) maxX = tx
            if (tz < minZ) minZ = tz; if (tz > maxZ) maxZ = tz
            i += 3
        }
        val minY = b.bounds.y.min
        val tall = b.bounds.y.max - minY
        val scale = if (tall > 0f) (height / tall).toFloat() else 1.0f
        val offset = Vector(
            x.toFloat() - (minX + maxX) / 2 * scale,
            y.toFloat() - minY * scale,
            z.toFloat() - (minZ + maxZ) / 2 * scale
        )
        val palette = if (fileColours && hasFileColours) b.mesh.palette.map { it.material() } else null
        return PlacedMesh(b, scale, rotation.toFloat(), offset, material, palette)
    }

    /** The mesh stood up, turned and scaled, its footprint centred on the origin at y = 0. */
    private fun shapedMesh(mesh: Mesh): Mesh = (if (zUp) mesh.zUpToYUp() else mesh)
        .rotatedY(rotation.toFloat())
        .standingOnFloor(targetHeight = height.toFloat())

    @Volatile private var boundsFor: List<Any>? = null
    @Volatile private var bounds = FloatArray(6)

    /**
     * The object's box relative to its position (min x, y, z, max x, y, z), for drawing it while
     * it is dragged and for standing other objects on it. Worked out again only when the object
     * is turned, resized or stood up differently.
     */
    override fun localBounds(): FloatArray {
        val key = listOf(zUp, rotation, height, width, depth)
        if (key != boundsFor) {
            bounds = when (shape) {
                is Shape.Model -> shapedMesh(shape.mesh).bounds().let { box ->
                    floatArrayOf(box.x.min, box.y.min, box.z.min, box.x.max, box.y.max, box.z.max)
                }
                Shape.Ball -> (height / 2).toFloat().let { r -> floatArrayOf(-r, 0f, -r, r, 2 * r, r) }
                Shape.Block -> {
                    // The turned footprint's reach along x and z.
                    val a = Math.toRadians(rotation)
                    val c = kotlin.math.abs(kotlin.math.cos(a))
                    val s = kotlin.math.abs(kotlin.math.sin(a))
                    val halfX = (width / 2 * c + depth / 2 * s).toFloat()
                    val halfZ = (width / 2 * s + depth / 2 * c).toFloat()
                    floatArrayOf(-halfX, 0f, -halfZ, halfX, height.toFloat(), halfZ)
                }
            }
            boundsFor = key
        }
        return bounds
    }

    override fun toString() = name
}
