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

    class Glass(val name: String, val schott: Boolean, val k: FloatArray, val catalogAbbe: Float?) {
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

    private fun key(name: String) = name.uppercase().replace("-", "").replace("_", "").replace(" ", "")
}
