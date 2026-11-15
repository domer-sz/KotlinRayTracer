package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.GpuSession
import rayTraceTypescript.gpu.MultiGpuSession
import rayTraceTypescript.gpu.GpuScene
import rayTraceTypescript.gpu.SceneFlattener
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.render.CameraSetup
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * How several cards share one image's samples, on made-up cards that only take their time: no
 * GPU is needed, so this runs everywhere.
 */
class MultiGpuSessionTest {
    private val setup: CameraSetup = Camera().apply { imageWidth = 2; aspectRatio = 1.0f }.initialize()
    private val pixels = setup.totalPixels

    @Test
    fun everySampleIsTracedExactlyOnceAndTheImageIsTheSumOfTheCards() {
        val fast = FakeCard("fast", msPerSample = 1)
        val slow = FakeCard("slow", msPerSample = 3)
        session(fast, slow).use { session ->
            val samples = 200
            session.traceSliced(GpuSession.MODE_IMAGE, setup, samples, GpuSession.IMAGE, 20.0)

            assertEquals((0 until samples).toList(), (fast.traced + slow.traced).sorted())
            assertTrue(fast.traced.size > slow.traced.size, "the quicker card took ${fast.traced.size}, the slower ${slow.traced.size}")
            // Sample s adds (s + 1) / samples to every pixel, so all of them add up to (samples + 1) / 2.
            val image = FloatArray(pixels).also { session.read(GpuSession.IMAGE, it) }
            image.forEach { assertEquals((samples + 1) / 2.0f, it, 0.01f) }
        }
    }

    @Test
    fun aSlowCardDoesNotMakeALaunchTakeLonger() {
        val fast = FakeCard("fast", msPerSample = 1)
        val slow = FakeCard("slow", msPerSample = 300)
        session(fast, slow).use { session ->
            // The first launch times both cards, one sample each.
            session.launch(GpuSession.MODE_IMAGE, setup, 0, 2, 1.0f)
            val before = slow.traced.size

            val start = System.nanoTime()
            session.launch(GpuSession.MODE_IMAGE, setup, 2, 20, 1.0f)
            val ms = (System.nanoTime() - start) / 1_000_000

            assertEquals(before, slow.traced.size, "the slow card should have left these samples alone")
            assertTrue(ms < 200, "20 samples took $ms ms, as if they had waited for the slow card")
        }
    }

    @Test
    fun shortLaunchesInARowNeverFallToTheSlowCard() {
        // What a viewer does: many launches of a few samples. Whichever card's thread wakes first
        // must not see itself alone and take them - the slow card would then set the pace.
        val fast = FakeCard("fast", msPerSample = 1)
        val slow = FakeCard("slow", msPerSample = 60)
        session(fast, slow).use { session ->
            session.launch(GpuSession.MODE_IMAGE, setup, 0, 2, 1.0f)
            val before = slow.traced.size
            var sample = 2
            repeat(60) {
                session.launch(GpuSession.MODE_IMAGE, setup, sample, 4, 1.0f)
                sample += 4
            }
            assertEquals(before, slow.traced.size, "the slow card took samples of launches too short for it")
            assertEquals((0 until sample).toList(), (fast.traced + slow.traced).sorted())
        }
    }

    @Test
    fun aCardThatIsTimedLateIsStillTimedAndThenTakesTheLead() {
        // The fast card is still building its kernel when the first samples are given out, so the
        // slow one is timed first - and nobody waits for a compiler, so it traces them alone.
        // Once the fast card is ready, the slow one must not pass for the only card there is.
        val fast = FakeCard("fast", msPerSample = 1, prepareMs = 150)
        val slow = FakeCard("slow", msPerSample = 30)
        session(fast, slow).use { session ->
            var sample = 0
            repeat(12) { session.launch(GpuSession.MODE_IMAGE, setup, sample++, 1, 1.0f) }
            assertTrue(fast.traced.isNotEmpty(), "the fast card was never given a sample to be timed with")

            val before = slow.traced.size
            repeat(30) {
                session.launch(GpuSession.MODE_IMAGE, setup, sample, 4, 1.0f)
                sample += 4
            }
            assertEquals(before, slow.traced.size, "the slow card kept taking samples once the fast one was timed")
        }
    }

    @Test
    fun aShortRangeIsOneLaunchForEachCardThatTakesPart() {
        // A launch costs something whatever it traces, so a range that fits in one turn must not
        // be cut finer: the quick card takes its part at once, and the slow one what is left for it.
        val fast = FakeCard("fast", msPerSample = 1)
        val slow = FakeCard("slow", msPerSample = 4)
        session(fast, slow).use { session ->
            // The cards are timed, and their turns grow to the size of the range.
            session.launch(GpuSession.MODE_IMAGE, setup, 0, 2, 1.0f)
            var sample = 2
            repeat(4) {
                session.launch(GpuSession.MODE_IMAGE, setup, sample, 27, 1.0f)
                sample += 27
            }
            val fastLaunches = fast.launches.get()
            val slowLaunches = slow.launches.get()
            val slowSamples = slow.traced.size

            session.launch(GpuSession.MODE_IMAGE, setup, sample, 27, 1.0f)

            assertEquals(1, fast.launches.get() - fastLaunches, "launches of the fast card")
            assertEquals(1, slow.launches.get() - slowLaunches, "launches of the slow card")
            // A fifth of the speed between them is the slow card's: 5 of 27 samples, give or take the timing.
            assertTrue(slow.traced.size - slowSamples in 4..6, "the slow card took ${slow.traced.size - slowSamples} of 27 samples")
        }
    }

    @Test
    fun aNewAccumulationLeavesOutWhatACardHoldsFromTheOldOne() {
        val a = FakeCard("a", msPerSample = 2)
        val b = FakeCard("b", msPerSample = 2)
        session(a, b).use { session ->
            session.launch(GpuSession.MODE_IMAGE, setup, 0, 40, 1.0f)
            assertTrue(a.traced.isNotEmpty() && b.traced.isNotEmpty(), "both cards should have traced some of 40 samples")

            // One sample from the start again: a single card traces it, the other's output is stale.
            session.launch(GpuSession.MODE_IMAGE, setup, 0, 1, 1.0f)
            val image = FloatArray(pixels).also { session.read(GpuSession.IMAGE, it) }
            image.forEach { assertEquals(1.0f, it, 1e-6f) }
        }
    }

    @Test
    fun aCardAddingToAnAccumulationItDidNotStartEmptiesItsOutputFirst() {
        val a = FakeCard("a", msPerSample = 2)
        val b = FakeCard("b", msPerSample = 2)
        session(a, b).use { session ->
            // Two accumulations in a row: whichever card does not get sample 0 of the second
            // still holds its part of the first.
            session.launch(GpuSession.MODE_IMAGE, setup, 0, 40, 1.0f)
            session.launch(GpuSession.MODE_IMAGE, setup, 0, 40, 1.0f)
            val image = FloatArray(pixels).also { session.read(GpuSession.IMAGE, it) }
            image.forEach { assertEquals(40 * 41 / 2.0f, it, 0.01f) }
        }
    }

    @Test
    fun aCardStillLoadingASceneJoinsLaterAndHoldsNothingUp() {
        val ready = FakeCard("ready", msPerSample = 1)
        val loading = FakeCard("loading", msPerSample = 1, loadMs = 600)
        session(ready, loading).use { session ->
            session.replaceScene(scene())
            val start = System.nanoTime()
            session.traceSliced(GpuSession.MODE_IMAGE, setup, 60, GpuSession.IMAGE, 20.0)
            val ms = (System.nanoTime() - start) / 1_000_000

            assertTrue(ms < 400, "60 samples took $ms ms, as if they had waited for the other card's scene")
            assertEquals((0 until 60).toList(), ready.traced.sorted())
        }
    }

    @Test
    fun aCardFailingInALaunchFailsItAndIsLeftOutOfTheNext() {
        val good = FakeCard("good", msPerSample = 1)
        val bad = FakeCard("bad", msPerSample = 1, failing = true)
        session(good, bad).use { session ->
            assertThrows(IllegalStateException::class.java) {
                session.traceSliced(GpuSession.MODE_IMAGE, setup, 50, GpuSession.IMAGE, 20.0)
            }
            good.traced.clear()

            session.traceSliced(GpuSession.MODE_IMAGE, setup, 50, GpuSession.IMAGE, 20.0)
            assertEquals((0 until 50).toList(), good.traced.sorted())
            val image = FloatArray(pixels).also { session.read(GpuSession.IMAGE, it) }
            image.forEach { assertEquals(51 / 2.0f, it, 0.01f) }
        }
    }

    @Test
    fun aCardThatCannotStartIsLeftOutOnlyWhenAskedToBeLenient() {
        val good = FakeCard("good", msPerSample = 1)
        val openers = listOf<Pair<String, () -> GpuSession>>("good" to { good }, "broken" to { throw IllegalStateException("no such card") })

        MultiGpuSession.of(openers, lenient = true).use { session ->
            session.traceSliced(GpuSession.MODE_IMAGE, setup, 10, GpuSession.IMAGE, 20.0)
            assertEquals(10, good.traced.size)
        }
        assertThrows(IllegalStateException::class.java) { MultiGpuSession.of(openers, lenient = false) }
    }

    private fun session(vararg cards: FakeCard): GpuSession = MultiGpuSession.of(cards.map { card -> card.name to { card as GpuSession } })

    private fun scene(): GpuScene = SceneFlattener.scene(Sphere(Point(0f, 0f, 0f), 1f, Lambertian(Color(0.5f, 0.5f, 0.5f))))

    /** A card that takes [msPerSample] for a sample and adds (sample + 1) * scale to every pixel. */
    private inner class FakeCard(
        val name: String, private val msPerSample: Long, private val loadMs: Long = 0, private val failing: Boolean = false,
        private val prepareMs: Long = 0
    ) : GpuSession {
        val traced = ConcurrentLinkedQueue<Int>()
        val launches = java.util.concurrent.atomic.AtomicInteger()
        private val outputs = Array(2) { FloatArray(pixels) }
        private var prepared = false
        override var seed = 0L
        override val devices = listOf(name)

        override fun replaceScene(next: GpuScene) = Thread.sleep(loadMs)

        override fun setCamera(setup: CameraSetup) {}

        override fun launch(mode: Int, setup: CameraSetup, sampleStart: Int, count: Int, scale: Float, target: Int) {
            if (failing) throw IllegalStateException("$name lost its card")
            launches.incrementAndGet()
            Thread.sleep(msPerSample * count)
            // As the kernels do: the launch holding sample 0 overwrites, any other adds.
            if (sampleStart == 0) outputs[target].fill(0.0f)
            for (sample in sampleStart until sampleStart + count) {
                traced += sample
                for (i in 0 until pixels) outputs[target][i] += (sample + 1) * scale
            }
        }

        override fun read(target: Int, into: FloatArray) {
            System.arraycopy(outputs[target], 0, into, 0, into.size)
        }

        override fun clear(target: Int) = outputs[target].fill(0.0f)

        // The first time, the kernel is "built".
        override fun prepare(mode: Int) {
            if (!prepared) Thread.sleep(prepareMs)
            prepared = true
        }

        override fun close() {}
    }
}
