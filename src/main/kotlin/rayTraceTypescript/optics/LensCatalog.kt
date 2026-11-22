package rayTraceTypescript.optics

import rayTraceTypescript.Point
import rayTraceTypescript.Vector
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.objects.Lens

/**
 * Stock lenses - Thorlabs' achromatic doublets and plano-convex singlets, four Edmund doublets -
 * from `optics/lenses.csv` (DCC-Lab/RayTracing's catalog, MIT; converted by tools/dcc-lenses.mjs),
 * in millimetres. Each turns into a [Lens] of its real surfaces and glasses.
 */
object LensCatalog {

    class Spec(
        val name: String,
        val doublet: Boolean,
        /** The effective focal length and the back focal length (last vertex to focus) the maker quotes, at [reference]. */
        val focalLength: Float,
        val backFocalLength: Float,
        val radii: FloatArray,
        val thicknesses: FloatArray,
        val edgeThickness: Float,
        val diameter: Float,
        /** Each element's catalog glass, or null where the catalog gives only a fixed index ([indices]). */
        val glasses: List<GlassCatalog.Glass?>,
        val indices: List<Float?>,
        /** The wavelength the focal lengths are quoted at, in microns. */
        val reference: Float,
        val url: String
    ) {
        /** Element [k]'s glass: dispersive from the catalog, or of fixed index. */
        fun material(k: Int): Dielectric = glasses[k]?.let { Dielectric(it) } ?: Dielectric(indices[k]!!)

        /**
         * The lens with its first vertex at [vertex], light meant to come in along [axis], scaled by
         * [unitsPerMm] (1 lays the scene out in millimetres).
         */
        fun build(vertex: Point, axis: Vector, unitsPerMm: Float = 1f): Lens {
            val r = radii.map { it * unitsPerMm }
            val t = thicknesses.map { it * unitsPerMm }
            val d = diameter * unitsPerMm
            return if (doublet) Lens.doublet(vertex, axis, r[0], r[1], r[2], t[0], t[1], d, material(0), material(1))
            else Lens.singlet(vertex, axis, r[0], r[1], t[0], d, material(0))
        }

        /** How long the lens is along its axis, vertex to vertex, in millimetres. */
        val length: Float get() = thicknesses.sum()

        override fun toString() = "$name (f = $focalLength mm, ø $diameter mm, ${glasses.mapIndexed { k, g -> g?.name ?: "n=${indices[k]}" }.joinToString(" + ")})"
    }

    val all: Map<String, Spec> by lazy {
        val text = LensCatalog::class.java.getResourceAsStream("/optics/lenses.csv")
            ?.bufferedReader()?.readText() ?: error("optics/lenses.csv is missing from the classpath")
        val rows = text.lineSequence().map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("#") }.toList()
        val columns = rows.first().split(',')
        rows.drop(1).map { line ->
            val values = line.split(',')
            val row = columns.zip(values).toMap()
            fun f(key: String) = row.getValue(key).let { if (it == "inf") Float.POSITIVE_INFINITY else if (it == "-inf") Float.NEGATIVE_INFINITY else it.toFloat() }
            fun glass(key: String) = row.getValue(key).takeIf { it.isNotBlank() }?.let { GlassCatalog.glass(it) }
            fun index(key: String) = row.getValue(key).takeIf { it.isNotBlank() }?.toFloat()
            val doublet = row.getValue("kind") == "doublet"
            Spec(
                name = row.getValue("name"),
                doublet = doublet,
                focalLength = f("f"),
                backFocalLength = f("fb"),
                radii = if (doublet) floatArrayOf(f("R1"), f("R2"), f("R3")) else floatArrayOf(f("R1"), f("R2")),
                thicknesses = if (doublet) floatArrayOf(f("tc1"), f("tc2")) else floatArrayOf(f("tc1")),
                edgeThickness = f("te"),
                diameter = f("diameter"),
                glasses = if (doublet) listOf(glass("glass1"), glass("glass2")) else listOf(glass("glass1")),
                indices = if (doublet) listOf(index("n1"), index("n2")) else listOf(index("n1")),
                reference = f("reference"),
                url = row["url"] ?: ""
            )
        }.associateBy { key(it.name) }
    }

    /** The lens called [name]: "AC254-100-A", "ac254_100_a" and "AC254100A" are the same. */
    fun lens(name: String): Spec = all[key(name)]
        ?: throw IllegalArgumentException("no lens $name in the catalog (${all.size} lenses; -Drt.lens.list prints them)")

    private fun key(name: String) = name.uppercase().replace("-", "").replace("_", "").replace(" ", "").replace("#", "")
}
