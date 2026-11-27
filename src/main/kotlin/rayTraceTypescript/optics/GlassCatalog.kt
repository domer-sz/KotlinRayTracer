package rayTraceTypescript.optics

import kotlin.math.sqrt

/**
 * Optical glasses and their dispersion, from `optics/glasses.csv` (DCC-Lab/RayTracing's
 * materials, MIT; converted by tools/dcc-lenses.mjs): each glass gives its refractive index for a
 * wavelength in microns by one of two formulas, both written as n squared -
 *  - Sellmeier: 1 + k0/(1 - k3/x^2) + k1/(1 - k4/x^2) + k2/(1 - k5/x^2)
 *  - Schott:    k0 + k1 x^2 + k2 x^-2 + k3 x^-4 + k4 x^-6 + k5 x^-8
 * The kernel evaluates the same formulas from the same six numbers ([Glass.packed]).
 */
object GlassCatalog {

    /** The wavelengths glass makers quote: helium d (587.6 nm), hydrogen F (486.1) and C (656.3). */
    const val D_LINE = 0.5876f
    const val F_LINE = 0.4861f
    const val C_LINE = 0.6563f

    /**
     * A glass: its formula ([schott] or Sellmeier) and six coefficients. A model glass
     * ([GlassCatalog.model]) also keeps the [modelIndex] and [modelAbbe] it was made from, so its
     * spread of colours can be scaled again ([withDispersion]).
     */
    class Glass(
        val name: String, val schott: Boolean, val k: FloatArray, val catalogAbbe: Float?,
        val modelName: String? = null, val modelIndex: Float = 0f, val modelAbbe: Float = 0f
    ) {
        init {
            require(k.size == 6) { "a glass needs six coefficients" }
        }

        /** The refractive index at [microns]. */
        fun n(microns: Float): Float {
            val x2 = microns.toDouble() * microns
            val n2 = if (schott) {
                k[0] + k[1] * x2 + k[2] / x2 + k[3] / (x2 * x2) + k[4] / (x2 * x2 * x2) + k[5] / (x2 * x2 * x2 * x2)
            } else {
                1.0 + k[0] / (1.0 - k[3] / x2) + k[1] / (1.0 - k[4] / x2) + k[2] / (1.0 - k[5] / x2)
            }
            return sqrt(n2).toFloat()
        }

        /** Abbe's number, (n_d - 1) / (n_F - n_C): how little the glass spreads colours. */
        fun abbe(): Float = (n(D_LINE) - 1f) / (n(F_LINE) - n(C_LINE))

        /** Eight floats for the kernel: the formula (0 Sellmeier, 1 Schott), the six coefficients, n at the d line. */
        fun packed(): FloatArray = floatArrayOf(if (schott) 1f else 0f, k[0], k[1], k[2], k[3], k[4], k[5], n(D_LINE))

        /** The same model glass with its spread of colours scaled by [scale] instead (0 none, 1 its own). */
        fun withDispersion(scale: Float): Glass {
            val base = requireNotNull(modelName) { "only a model glass (n_d and V_d) can have its dispersion scaled, not $name" }
            return model(base, modelIndex, modelAbbe, scale)
        }

        override fun toString() = name
    }

    val all: Map<String, Glass> by lazy {
        val text = GlassCatalog::class.java.getResourceAsStream("/optics/glasses.csv")
            ?.bufferedReader()?.readText() ?: error("optics/glasses.csv is missing from the classpath")
        text.lineSequence().map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("#") && !it.startsWith("name,") }
            .map { line ->
                val f = line.split(',')
                Glass(f[0], f[1] == "schott", FloatArray(6) { f[2 + it].toFloat() }, f.getOrNull(8)?.takeIf { it.isNotBlank() }?.toFloat())
            }
            .associateBy { key(it.name) }
    }

    /** The glass called [name], however it is written: "N-BK7", "n_bk7" and "NBK7" are the same. */
    fun glass(name: String): Glass = all[key(name)]
        ?: throw IllegalArgumentException("unknown glass $name; known: ${all.values.joinToString { it.name }}")

    /**
     * A glass known only by its index at the d line [nd] and its Abbe number [abbe] - Zemax's model
     * glass, or a gem such as diamond - written as Cauchy's n = A + B / lambda^2 through n_d with
     * (n_F - n_C) = [scale] (n_d - 1) / V_d, in Schott's form n^2 = A^2 + 2AB/x^2 + B^2/x^4.
     * [scale] spreads the colours as Blender's dispersion scale does, about a d-line index that
     * stays put: 0 bends every wavelength alike, 1 is the glass itself, more exaggerates it.
     */
    fun model(name: String, nd: Float, abbe: Float, scale: Float = 1f): Glass {
        require(nd > 1f && abbe > 0f && scale >= 0f) { "a model glass needs n_d > 1, V_d > 0 and a scale of 0 or more, not $nd, $abbe, $scale" }
        val f = F_LINE.toDouble(); val c = C_LINE.toDouble(); val d = D_LINE.toDouble()
        val b = (nd - 1.0) * scale / abbe / (1.0 / (f * f) - 1.0 / (c * c))
        val a = nd - b / (d * d)
        val label = if (scale == 1f) "$name (model n_d $nd, V_d $abbe)" else "$name (model n_d $nd, V_d $abbe, dispersion x$scale)"
        return Glass(label, true, floatArrayOf((a * a).toFloat(), 0f, (2 * a * b).toFloat(), (b * b).toFloat(), 0f, 0f), if (scale > 0f) abbe / scale else null, name, nd, abbe)
    }

    /** Diamond, n_d 2.417 and V_d 55.3 (n_F - n_C 0.025; the "fire" of 0.044 jewellers quote is n_G - n_B). */
    fun diamond(scale: Float = 1f): Glass = model("diamond", 2.417f, 55.3f, scale)

    /** Plain crown glass, the studio's index of 1.5 with N-BK7's spread of colours (V_d 64.2). */
    fun crown(scale: Float = 1f): Glass = model("crown", 1.5f, 64.17f, scale)

    private fun key(name: String) = name.uppercase().replace("-", "").replace("_", "").replace(" ", "")
}
