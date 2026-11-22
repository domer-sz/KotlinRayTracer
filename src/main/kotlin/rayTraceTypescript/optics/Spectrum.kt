package rayTraceTypescript.optics

import rayTraceTypescript.Color
import kotlin.math.exp
import kotlin.math.max

/**
 * The colour one wavelength carries, as the kernel's spectral_weight gives it: a path that meets
 * a catalog glass draws its wavelength evenly over [FROM]-[TO] and from then on carries only that
 * colour's share of its light. The CIE 1931 observer (Wyman, Sloan and Shirley's multi-lobe fit,
 * 2013), turned into linear sRGB, negative lobes cut off, each channel divided by its mean over
 * the band - so white light, split and gathered again, is white.
 */
object Spectrum {
    /** The band drawn from, in microns. */
    const val FROM = 0.38f
    const val TO = 0.73f

    /** Each channel's mean over the band, before the division: the kernel divides by the same numbers. */
    private val MEANS = floatArrayOf(0.5033789f, 0.3296633f, 0.3122747f)

    fun weight(microns: Float): Color {
        val l = microns * 1000f
        val x = 1.056f * lobe(l, 599.8f, 37.9f, 31.0f) + 0.362f * lobe(l, 442.0f, 16.0f, 26.7f) - 0.065f * lobe(l, 501.1f, 20.4f, 26.2f)
        val y = 0.821f * lobe(l, 568.8f, 46.9f, 40.5f) + 0.286f * lobe(l, 530.9f, 16.3f, 31.1f)
        val z = 1.217f * lobe(l, 437.0f, 11.8f, 36.0f) + 0.681f * lobe(l, 459.0f, 26.0f, 13.8f)
        val r = max(0f, 3.2406f * x - 1.5372f * y - 0.4986f * z)
        val g = max(0f, -0.9689f * x + 1.8758f * y + 0.0415f * z)
        val b = max(0f, 0.0557f * x - 0.2040f * y + 1.0570f * z)
        return Color(r / MEANS[0], g / MEANS[1], b / MEANS[2])
    }

    private fun lobe(l: Float, mu: Float, below: Float, above: Float): Float {
        val t = (l - mu) / (if (l < mu) below else above)
        return exp(-0.5f * t * t)
    }
}
