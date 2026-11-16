package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class CameraSetupTest {

    @Test
    fun `a halved view covers each 2x2 block of the full one with one pixel at its middle`() {
        val full = Camera().apply {
            aspectRatio = 9.0f / 16.0f
            imageWidth = 1080
            lookFrom = Point(0.0f, 1.0f, 5.0f)
            lookAt = Point(0.0f, 1.0f, 0.0f)
        }.initialize()
        val half = full.halved()

        assertEquals(540, half.imageWidth)
        assertEquals(960, half.imageHeight)
        // The half view's pixel (i, j) sits where the full view's (2i + 0.5, 2j + 0.5) does.
        for ((i, j) in listOf(0 to 0, 539 to 0, 0 to 959, 271 to 480)) {
            val halfCentre = half.pixel00 + half.pixelDeltaU * i.toFloat() + half.pixelDeltaV * j.toFloat()
            val fullCentre = full.pixel00 + full.pixelDeltaU * (2 * i + 0.5f) + full.pixelDeltaV * (2 * j + 0.5f)
            assertEquals(fullCentre.x, halfCentre.x, 1e-4f)
            assertEquals(fullCentre.y, halfCentre.y, 1e-4f)
            assertEquals(fullCentre.z, halfCentre.z, 1e-4f)
        }
        assertEquals(full.cameraCenter, half.cameraCenter)
        assertEquals(full.samplesPerPixel, half.samplesPerPixel)
    }

    @Test
    fun `an image of odd size cannot be halved`() {
        val odd = Camera().apply { aspectRatio = 1.0f; imageWidth = 101 }.initialize()
        assertThrows<IllegalArgumentException> { odd.halved() }
    }
}
