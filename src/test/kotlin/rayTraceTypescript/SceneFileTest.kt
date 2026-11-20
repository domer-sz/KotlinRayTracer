package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import rayTraceTypescript.gui.Form
import rayTraceTypescript.gui.MaterialKind
import rayTraceTypescript.gui.MaterialSpec
import rayTraceTypescript.gui.SavedObject
import rayTraceTypescript.gui.SavedPart
import rayTraceTypescript.gui.SavedScene
import rayTraceTypescript.gui.SavedWall
import rayTraceTypescript.gui.SceneFile
import java.io.File

class SceneFileTest {

    @Test
    fun `a saved scene reads back the same`(@TempDir dir: File) {
        val model = File(dir, "models/ork zbrojny.stl").apply { parentFile.mkdirs(); writeText("solid") }
        val saved = SavedScene(
            scene = "cornell-final",
            properties = mapOf("rt.block" to "checker"),
            camera = doubleArrayOf(278.0, 278.0, 0.0, 0.25, -0.1, 800.0, 40.0),
            width = 800,
            height = 600,
            samples = 2048,
            raytrace = true,
            denoise = false,
            parts = listOf(
                SavedPart("Lampa sufitowa", true), SavedPart("Światło przednie", false),
                SavedPart("Szklana kula", true, doubleArrayOf(120.0, 0.0, 250.5), -30.0, MaterialSpec(MaterialKind.METAL, 0.9f, 0.7f, 0.3f, 0.05)),
                SavedPart("Lampa 2", true, power = 2.5)
            ),
            objects = listOf(
                SavedObject("Ork\\z mieczem", "model", model, 292.0, 0.0, 277.5, 45.0, 145.6, 0.5, 0.5, true,
                    MaterialSpec(MaterialKind.METAL, 0.9f, 0.7f, 0.3f, 0.05), form = Form.SPONGE, latticePitch = 0.05),
                SavedObject("Kula 1", "ball", null, 100.0, 20.0, 50.0, 0.0, 80.0, 40.0, 40.0, false,
                    MaterialSpec(MaterialKind.GLASS, 1f, 1f, 1f, 1.5))
            ),
            depth = 12,
            walls = listOf(SavedWall("Ściana lewa", MaterialSpec(MaterialKind.METAL, 0.9f, 0.9f, 0.9f, 0.0)))
        )
        val file = File(dir, "scena.rtscene")
        SceneFile.write(file, saved)
        // Paths inside the folder stay relative, so the scene moves with it.
        assertTrue(file.readText().contains("object.0.file=models/ork zbrojny.stl"), file.readText())

        val read = SceneFile.read(file)
        assertEquals("cornell-final", read.scene)
        assertEquals(mapOf("rt.block" to "checker"), read.properties)
        assertArrayEquals(saved.camera, read.camera, 1e-9)
        assertEquals(listOf(800, 600, 2048, 12), listOf(read.width, read.height, read.samples, read.depth))
        assertTrue(read.raytrace)
        assertEquals(false, read.denoise)
        assertEquals(saved.parts.map { it.name to it.enabled }, read.parts.map { it.name to it.enabled })
        assertEquals(null, read.parts[0].position)
        assertArrayEquals(doubleArrayOf(120.0, 0.0, 250.5), read.parts[2].position, 1e-9)
        assertEquals(-30.0, read.parts[2].rotation)
        assertEquals(saved.parts[2].material, read.parts[2].material)
        assertEquals(null, read.parts[0].material)
        assertEquals(2.5, read.parts[3].power)
        assertEquals(1.0, read.parts[0].power)
        assertEquals(listOf("Ściana lewa"), read.walls.map { it.name })
        assertEquals(saved.walls[0].material, read.walls[0].material)
        assertEquals(2, read.objects.size)
        val ork = read.objects[0]
        assertEquals("Ork\\z mieczem", ork.name)
        assertEquals(model.canonicalFile, ork.file!!.canonicalFile)
        assertEquals(listOf(292.0, 0.0, 277.5, 45.0, 145.6), listOf(ork.x, ork.y, ork.z, ork.rotation, ork.height))
        assertTrue(ork.zUp)
        assertEquals(saved.objects[0].material, ork.material)
        assertEquals(Form.SPONGE, ork.form)
        assertEquals(0.05, ork.latticePitch)
        assertEquals("ball", read.objects[1].shape)
        assertEquals(null, read.objects[1].file)
        assertEquals(saved.objects[1].material, read.objects[1].material)
        // A form left at solid writes nothing, and a file without one reads as a solid.
        assertEquals(Form.SOLID, read.objects[1].form)
        assertTrue(!file.readText().contains("object.1.form"), file.readText())
    }
}
