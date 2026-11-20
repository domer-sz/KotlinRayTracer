package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.PlacedMesh
import rayTraceTypescript.gui.Form
import rayTraceTypescript.gui.PlacedObject
import rayTraceTypescript.gui.Shape
import rayTraceTypescript.objects.Sphere

/**
 * The viewer's forms for an added object: it shows as a solid until the converted mesh is built, then
 * as a placed mesh of its own proportions; the added ball and block convert like a loaded model.
 */
class PlacedObjectFormTest {

    private fun ball() = PlacedObject("Kula 1", Shape.Ball).apply { height = 2.0; latticePitch = 0.25 }

    @Test
    fun aSolidIsShownAsItsOwnShapeAndNeedsNoBuild() {
        val o = ball()
        assertTrue(o.formReady)
        assertTrue(o.hittable() is Sphere)
    }

    @Test
    fun aSpongeShowsAsASolidUntilItIsBuilt() {
        val o = ball().apply { form = Form.SPONGE }
        assertFalse(o.formReady)
        assertTrue(o.hittable() is Sphere, "before the build the ball still shows")
        o.prepareForm()
        assertTrue(o.formReady)
        val placed = o.hittable() as PlacedMesh
        // The converted mesh is placed to the object's height, standing on its base.
        assertEquals(2.0f, placed.block.bounds.y.max * placed.scale + placed.offset.y - (placed.block.bounds.y.min * placed.scale + placed.offset.y), 1e-3f)
    }

    @Test
    fun changingTheSpacingNeedsANewBuild() {
        val o = ball().apply { form = Form.SPONGE }
        o.prepareForm()
        assertTrue(o.formReady)
        o.latticePitch = 0.3
        assertFalse(o.formReady)
        assertTrue(o.hittable() is Sphere)
    }

    @Test
    fun aBlockAndASpongeConvertToo() {
        val block = PlacedObject("Graniastosłup 1", Shape.Block).apply { height = 2.0; width = 1.0; depth = 1.0; form = Form.SPONGE; latticePitch = 0.2 }
        block.prepareForm()
        assertTrue(block.hittable() is PlacedMesh)
        val pyramids = PlacedObject("Kula 2", Shape.Ball).apply { height = 2.0; form = Form.PYRAMIDS }
        pyramids.prepareForm()
        assertTrue(pyramids.hittable() is PlacedMesh)
    }
}
