package rayTraceTypescript.optics

import rayTraceTypescript.Point
import rayTraceTypescript.Vector
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.objects.Lens
import java.nio.charset.Charset
import java.nio.file.Files
import java.nio.file.Path


/**
 * A lens from a Zemax OpticStudio file (.zmx, the text format lens makers publish their stock
 * lenses in - Thorlabs' aspheric collimators among them): its surfaces in order, each STANDARD
 * (sphere or conic) or EVENASPH (conic plus a polynomial in rho^2), the gaps between them and what
 * fills each gap. Glass that [GlassCatalog] knows is dispersive from its formula; any other is
 * Zemax's model glass from the n_d and Abbe number the file gives, written as Cauchy's
 * n = A + B / lambda^2 (squared, Schott's form) - right in the visible, rough far beyond it.
 *
 * The groups of surfaces with glass between them become [Lens]es: an asphere and the window in
 * front of a laser diode are two, with air between. The object surface (the first) and the image
 * surface (the last) are not glass; the image surface's place is where the file's design brings the
 * light to a focus - a collimator's source, read backwards.
 */
class ZemaxLens private constructor(
    val name: String,
    /** The surfaces from the first after the object up to the image, each with where its vertex is along the axis, in millimetres. */
    private val surfaces: List<Surface>,
    /** Where the image surface is along the axis, in millimetres from the first surface's vertex. */
    val imageAt: Float,
    val wavelength: Float
) {
    class Surface(
        val z: Float,
        val curvature: Float,
        val conic: Float,
        val aspherics: FloatArray,
        val semiDiameter: Float,
        /** The glass after this surface, null for air. */
        val glass: Dielectric?
    )

    /** How many separate pieces of glass. */
    val elementGroups: List<IntRange> = buildList {
        var start = -1
        for (k in surfaces.indices) {
            val inGlassBefore = k > 0 && surfaces[k - 1].glass != null
            val inGlassAfter = surfaces[k].glass != null
            if (!inGlassBefore && inGlassAfter) start = k
            if (inGlassBefore && !inGlassAfter) add(start..k)
        }
    }

    /**
     * The lenses, the first surface's vertex at [vertex], light coming in along [axis], scaled by
     * [unitsPerMm]. Each group's rim is its widest surface's semi-diameter - which Zemax works out as
     * the beam's footprint there, not the glass's size. A flat window (a laser diode's, in front of
     * a collimator) is cut wider than the beam crossing it; held to the footprint, its rim caught
     * every ray outside the paraxial ones, so a window takes the widest semi-diameter in the file.
     */
    fun build(vertex: Point, axis: Vector, unitsPerMm: Float = 1f): List<Lens> {
        val a = axis.unit()
        val widest = surfaces.maxOf { it.semiDiameter }
        return elementGroups.map { group ->
            val members = group.map { surfaces[it] }
            val first = members.first().z
            val flat = members.all { it.curvature == 0f && it.aspherics.all { term -> term == 0f } }
            val rim = (if (flat) widest else members.maxOf { it.semiDiameter }) * unitsPerMm
            val aspherics = FloatArray(Lens.ASPHERIC_TERMS * members.size)
            members.forEachIndexed { k, s ->
                // a_i rho^(2i) in millimetres becomes a_i / scale^(2i - 1) in scene units.
                for (i in 0 until Lens.ASPHERIC_TERMS) {
                    aspherics[Lens.ASPHERIC_TERMS * k + i] = (s.aspherics[i] / Math.pow(unitsPerMm.toDouble(), 2.0 * (i + 1) - 1)).toFloat()
                }
            }
            Lens(
                vertex = (vertex + a * (first * unitsPerMm)).toPoint(),
                axis = a,
                semiDiameter = rim,
                vertices = FloatArray(members.size) { (members[it].z - first) * unitsPerMm },
                curvatures = FloatArray(members.size) { members[it].curvature / unitsPerMm },
                media = listOf<Dielectric?>(null) + members.dropLast(1).map { it.glass } + listOf(null),
                conics = FloatArray(members.size) { members[it].conic },
                aspherics = aspherics
            )
        }
    }

    companion object {
        fun read(path: Path): ZemaxLens = parse(decode(Files.readAllBytes(path)), path.fileName.toString())

        /** Zemax writes UTF-16 with a byte-order mark, older files plain bytes. */
        internal fun decode(bytes: ByteArray): String = when {
            bytes.size >= 2 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xFE.toByte() -> String(bytes, 2, bytes.size - 2, Charsets.UTF_16LE)
            bytes.size >= 2 && bytes[0] == 0xFE.toByte() && bytes[1] == 0xFF.toByte() -> String(bytes, 2, bytes.size - 2, Charsets.UTF_16BE)
            else -> String(bytes, Charset.forName("ISO-8859-1"))
        }

        internal fun parse(text: String, fallbackName: String): ZemaxLens {
            var name = fallbackName
            var scale = 1f
            var wavelength = GlassCatalog.D_LINE
            class Raw {
                var type = "STANDARD"; var curvature = 0f; var conic = 0f; var gap = 0f; var infinite = false
                val parms = FloatArray(9); var semi = 0f; var glassName: String? = null; var nd = 0f; var vd = 0f
            }
            val raws = mutableListOf<Raw>()
            var current: Raw? = null
            for (rawLine in text.lineSequence()) {
                val line = rawLine.trim()
                if (line.isEmpty()) continue
                val f = line.split(Regex("\\s+"))
                val indented = rawLine.startsWith(" ")
                if (!indented) {
                    current = null
                    when (f[0]) {
                        "NAME" -> name = line.removePrefix("NAME").trim().ifEmpty { fallbackName }
                        "UNIT" -> scale = when (f.getOrNull(1)) { "IN" -> 25.4f; "CM" -> 10f; "METER" -> 1000f; else -> 1f }
                        "WAVM" -> if (f.getOrNull(1) == "1") wavelength = f[2].toFloat()
                        "SURF" -> current = Raw().also { raws += it }
                    }
                    continue
                }
                val s = current ?: continue
                when (f[0]) {
                    "TYPE" -> s.type = f[1]
                    "CURV" -> s.curvature = f[1].toFloat()
                    "CONI" -> s.conic = f[1].toFloat()
                    "PARM" -> f[1].toInt().takeIf { it in 1..8 }?.let { s.parms[it] = f[2].toFloat() }
                    "DISZ" -> if (f[1] == "INFINITY") s.infinite = true else s.gap = f[1].toFloat()
                    "DIAM" -> s.semi = f[1].toFloat()
                    "GLAS" -> {
                        s.glassName = f[1]
                        s.nd = f.getOrNull(4)?.toFloatOrNull() ?: 0f
                        s.vd = f.getOrNull(5)?.toFloatOrNull() ?: 0f
                    }
                }
            }
            require(raws.size >= 3) { "$fallbackName: no lens in it (${raws.size} surfaces)" }
            // Surface 0 is the object, the last the image; positions from the first real surface.
            val real = raws.subList(1, raws.size - 1)
            var z = 0f
            val surfaces = real.map { r ->
                require(r.type == "STANDARD" || r.type == "EVENASPH") {
                    "$fallbackName: surfaces of type ${r.type} are not supported (only STANDARD and EVENASPH)"
                }
                val aspherics = FloatArray(Lens.ASPHERIC_TERMS) { i -> if (r.type == "EVENASPH") r.parms[i + 1] * Math.pow(scale.toDouble(), -(2.0 * (i + 1) - 1)).toFloat() else 0f }
                val surface = Surface(z * scale, r.curvature / scale, r.conic, aspherics, r.semi * scale, glassOf(r.glassName, r.nd, r.vd))
                z += r.gap
                surface
            }
            return ZemaxLens(name, surfaces, z * scale, wavelength)
        }

        /** A glass the catalog knows, else Zemax's model glass from n_d and V_d; null for air or a mirror. */
        private fun glassOf(name: String?, nd: Float, vd: Float): Dielectric? {
            if (name == null || name == "MIRROR") return null
            GlassCatalog.all[name.uppercase().replace("-", "").replace("_", "")]?.let { return Dielectric(it) }
            require(nd > 1f) { "glass $name is not in the catalog and the file gives no index for it" }
            return Dielectric(modelGlass(name, nd, if (vd > 0f) vd else 50f))
        }

        /** Zemax's model glass from n_d and V_d ([GlassCatalog.model]). */
        internal fun modelGlass(name: String, nd: Float, vd: Float): GlassCatalog.Glass = GlassCatalog.model(name, nd, vd)
    }

    override fun toString() = "$name (${elementGroups.size} element group(s), image ${"%.3f".format(imageAt)} mm from the first vertex)"

    /** The surfaces, for tests. */
    internal val surfaceList: List<Surface> get() = surfaces

}
