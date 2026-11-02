package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.SceneFlattener
import rayTraceTypescript.scenes.BookScenes

class BookScenesTest {

    /** The model scenes want a file on disk; every other scene has to stand on its own. */
    private val selfContained = BookScenes.names.filterNot { it == "model" || it == "cornell-model" || it == "studio-model" }

    @Test
    fun `every scene builds and flattens for the gpu`() {
        selfContained.forEach { name ->
            val scene = BookScenes.byName(name)
            val buffers = SceneFlattener.flatten(scene.world, scene.camera.lights)
            assertTrue(
                buffers.sphereCount + buffers.quadCount + buffers.triangleCount + buffers.mediumCount > 0,
                "$name produced no primitives"
            )
            assertTrue(scene.camera.imageWidth > 0, "$name has no image size")
        }
    }

    @Test
    fun `the book three scene names the shapes worth aiming at`() {
        val scene = BookScenes.byName("cornell-final")
        val lights = requireNotNull(scene.camera.lights) { "cornell-final has no lights to sample" }
        val buffers = SceneFlattener.flatten(scene.world, lights)
        assertEquals(2, buffers.lightCount, "the lamp and the glass sphere")
    }

    @Test
    fun `the old numeric scene names still work`() {
        assertTrue(BookScenes.byName("1").name == "bouncing-spheres")
        assertTrue(BookScenes.byName("2").name == "earth")
    }

    @Test
    fun `an unknown scene lists what is available`() {
        val failure = assertThrows(IllegalArgumentException::class.java) { BookScenes.byName("nope") }
        assertTrue(failure.message!!.contains("cornell-box"), failure.message)
    }
}
