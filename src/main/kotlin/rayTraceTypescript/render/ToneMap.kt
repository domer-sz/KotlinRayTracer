package rayTraceTypescript.render

import kotlin.math.pow
import kotlin.math.sqrt

/**
 * How linear light becomes the 8-bit pixels of a file, the viewer and a film - one place, so the
 * three always agree.
 *
 * `-Drt.tonemap=sqrt` (the default) is the renderer's old look: a square root as the gamma and
 * everything over one clipped, as the books do. `aces` is the filmic curve games use (Krzysztof
 * Narkowicz's fit of ACES): highlights roll off instead of clipping, shadows keep their depth, and
 * the result is encoded as sRGB. `-Drt.exposure=<EV>` scales the light first, in stops.
 */
class ToneMap private constructor(private val aces: Boolean, private val scale: Float) {

    /** One linear channel value as a display code value in [0, 1). */
    fun encode(linear: Float): Float {
        val x = linear * scale
        if (!(x > 0f)) return 0f
        if (!aces) return sqrt(x).coerceAtMost(0.999f)
        val mapped = ((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f)).coerceIn(0f, 1f)
        val srgb = if (mapped <= 0.0031308f) 12.92f * mapped else 1.055f * mapped.pow(1f / 2.4f) - 0.055f
        return srgb.coerceAtMost(0.999f)
    }

    /** One linear channel value as a byte, 0..255. */
    fun byte(linear: Float): Int = (256f * encode(linear)).toInt()

    companion object {
        /** The mapping the properties ask for now; read once per image, not per pixel. */
        fun current(): ToneMap {
            val mode = System.getProperty("rt.tonemap")?.trim()?.lowercase()
            val stops = System.getProperty("rt.exposure")?.toFloatOrNull() ?: 0f
            return ToneMap(mode == "aces" || mode == "filmic", 2f.pow(stops))
        }
    }
}
