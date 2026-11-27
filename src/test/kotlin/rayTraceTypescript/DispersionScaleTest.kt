package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import rayTraceTypescript.gpu.TurntableSettings
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.optics.GlassCatalog

/** A model glass's spread of colours scaled as Blender's dispersion scale does, and animated in a film. */
class DispersionScaleTest {

    @Test
    fun `a model glass keeps its d-line index and its abbe number`() {
        val diamond = GlassCatalog.diamond()
        assertEquals(2.417f, diamond.n(GlassCatalog.D_LINE), 1e-4f)
        assertEquals(55.3f, diamond.abbe(), 0.2f)
        // Blue bends more than red.
        assertTrue(diamond.n(0.45f) > diamond.n(0.65f))
    }

    @Test
    fun `the scale spreads colours about a d-line index that stays put`() {
        val none = GlassCatalog.diamond(0f)
        for (microns in listOf(0.38f, 0.5f, 0.5876f, 0.73f)) assertEquals(2.417f, none.n(microns), 1e-5f, "at $microns um")
        val real = GlassCatalog.diamond(1f)
        val doubled = GlassCatalog.diamond(2f)
        assertEquals(2.417f, doubled.n(GlassCatalog.D_LINE), 1e-4f)
        assertEquals(real.abbe() / 2f, doubled.abbe(), 0.2f)
        val spread = real.n(GlassCatalog.F_LINE) - real.n(GlassCatalog.C_LINE)
        assertEquals(2f * spread, doubled.n(GlassCatalog.F_LINE) - doubled.n(GlassCatalog.C_LINE), spread * 0.01f)
    }

    @Test
    fun `a glass rescaled is the model it came from at the new scale`() {
        val crown = Dielectric(GlassCatalog.crown())
        val half = crown.withDispersion(0.5f)
        assertEquals(1.5f, half.ri, 1e-4f)
        assertEquals(GlassCatalog.crown(0.5f).n(0.4f), half.glass!!.n(0.4f), 1e-6f)
        // Back to the full scale from the half one, not half of half.
        assertEquals(GlassCatalog.crown().n(0.4f), half.withDispersion(1f).glass!!.n(0.4f), 1e-6f)
        // A catalog glass has no n_d and V_d of its own to scale from.
        assertThrows<IllegalArgumentException> { Dielectric(GlassCatalog.glass("N-BK7")).withDispersion(2f) }
    }

    @Test
    fun `a film's dispersion goes there and back over a turn and loops`() {
        val film = TurntableSettings(seconds = 4.0, fps = 10, dispersion = 0f to 1f)
        val n = film.framesPerTurn
        assertEquals(0f, film.dispersionScale(0)!!, 1e-6f)
        assertEquals(1f, film.dispersionScale(n / 2)!!, 1e-6f)
        // Symmetric, so the frame after the last is the first again.
        assertEquals(film.dispersionScale(1)!!, film.dispersionScale(n - 1)!!, 1e-6f)
        // A swing goes out only: its return is the frames reversed.
        val swing = TurntableSettings(seconds = 4.0, fps = 10, swing = true, dispersion = 0f to 2f)
        assertEquals(0f, swing.dispersionScale(0)!!, 1e-6f)
        assertEquals(2f, swing.dispersionScale(swing.framesPerTurn - 1)!!, 1e-6f)
        assertNull(TurntableSettings(seconds = 4.0, fps = 10).dispersionScale(3))
    }
}
