package rayTraceTypescript.gpu

import rayTraceTypescript.render.CameraSetup
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * Several cards rendering one image together.
 *
 * Every random stream is keyed by its pixel and its sample, so the samples of an image can be
 * traced in any split: each card traces some of them for the whole image, into an output of its
 * own, and the image is the sum of the outputs. Nothing is tiled, so there are no seams, and the
 * cards need not be alike - an RTX card going through OptiX beside an older one on the CUDA
 * kernel.
 *
 * Each card has a thread of its own, the only one to touch its session. A launch is a range of
 * samples the threads share out: a card takes the next few - as many as keep it busy for about
 * one slice - traces them, and comes back for more, so a faster card simply takes more turns.
 * A card takes no more than its share of what is left, by speed, and none at all once the others
 * would be done with the rest before it could finish one sample; a slow card can therefore not
 * make a launch take longer than the fast one alone would. A single sample goes to the quickest.
 *
 * A new scene goes to every card at once; a card still loading it joins the launch in progress
 * when it is ready and does not hold the others up, and one that cannot load it sits that scene
 * out. A card failing in the middle of a launch fails the launch - what it had traced is lost -
 * and is left out until a scene loads on it again.
 */
internal class MultiGpuSession private constructor(private val cards: List<Card>) : GpuSession {
    override var seed: Long = GpuSession.initialSeed()

    /** The scene the cards should hold, and its number: a card loads the newest when it gets to it. */
    @Volatile private var wanted: Pair<Int, SceneBuffers>? = null
    private var sceneNumber = 0
    private var camera: CameraSetup? = null

    /** Which accumulation each output is on. A card's output counts only while it is on the same one. */
    private val epochs = IntArray(TARGETS)

    override val devices: List<String> get() = cards.flatMap { it.devices() }

    override fun replaceScene(next: SceneBuffers) {
        val number = ++sceneNumber
        wanted = number to next
        cards.forEach { card -> card.post { card.loadScene(wanted) } }
    }

    override fun setCamera(setup: CameraSetup) {
        // Carried by the next launch: a card sets it when it starts on that launch's samples.
        camera = setup
    }

    override fun launch(mode: Int, setup: CameraSetup, sampleStart: Int, count: Int, scale: Float, target: Int) {
        share(mode, setup, sampleStart, count, scale, target, GpuSession.SLICE_MS, null)
    }

    override fun traceSliced(mode: Int, setup: CameraSetup, samples: Int, target: Int, sliceMs: Double, onProgress: ((Int) -> Unit)?): Int =
        share(mode, setup, 0, samples, 1.0f / samples, target, sliceMs, onProgress)

    override fun read(target: Int, into: FloatArray) {
        val epoch = epochs[target]
        val done = CountDownLatch(cards.size)
        val parts = arrayOfNulls<FloatArray>(cards.size)
        val failures = arrayOfNulls<Throwable>(cards.size)
        cards.forEachIndexed { i, card ->
            card.post {
                try {
                    parts[i] = card.readIfOn(target, epoch, into.size)
                } catch (t: Throwable) {
                    failures[i] = t
                } finally {
                    done.countDown()
                }
            }
        }
        done.await()
        failures.firstOrNull { it != null }?.let { throw IllegalStateException("reading the image back failed: ${it.message}", it) }
        into.fill(0.0f)
        for (part in parts) {
            if (part == null) continue
            for (i in into.indices) into[i] += part[i]
        }
    }

    override fun clear(target: Int) {
        // Nothing to do on the cards now: each empties its output when it first adds to the new accumulation.
        epochs[target]++
    }

    override fun close() {
        cards.forEach { it.post { it.shutDown() } }
        cards.forEach { it.join() }
    }

    /**
     * Has the cards trace samples [first] until [first] + [count] between them and returns the
     * number of launches that took. A range starting at sample 0 starts a new accumulation.
     */
    private fun share(
        mode: Int, setup: CameraSetup, first: Int, count: Int, scale: Float, target: Int,
        sliceMs: Double, onProgress: ((Int) -> Unit)?
    ): Int {
        if (count <= 0) return 0
        if (first == 0) epochs[target]++
        val range = Range(mode, camera ?: setup, setup, first, first + count, scale, target, seed, sceneNumber, epochs[target], sliceMs, onProgress, cards.size)
        cards.forEach { card -> card.post { card.trace(range) } }
        return range.await()
    }

    /** A range of samples being shared out: what is left, who is on it, and how it ended. */
    private class Range(
        val mode: Int, val camera: CameraSetup, val setup: CameraSetup, val first: Int, val end: Int, val scale: Float,
        val target: Int, val seed: Long, val sceneNumber: Int, val epoch: Int, val sliceMs: Double,
        val onProgress: ((Int) -> Unit)?, cards: Int
    ) {
        private val lock = Object()
        private var next = first
        private var inFlight = 0
        private var finished = 0
        private var launches = 0
        /** Cards that have not come to this range yet, or are on it. */
        private var pending = cards
        private val active = HashSet<Card>()
        val rateKey: Long = mode.toLong() shl 40 or setup.totalPixels.toLong()

        fun join(card: Card) = synchronized(lock) { active += card }

        /** [card] is through with this range - done, dropped out, or never able to start. */
        fun leave(card: Card) = synchronized(lock) {
            active -= card
            pending--
            lock.notifyAll()
        }

        /**
         * The samples [card] should trace next, as (first, count), or null when it should stop:
         * nothing is left, or the rest is better left to the others.
         */
        fun claim(card: Card): IntArray? = synchronized(lock) {
            val left = end - next
            if (left <= 0 || failure != null) return null
            val rate = card.rate(rateKey)
            val size = if (rate == null) {
                // Not timed yet at this size and mode: one sample, to find out.
                1
            } else {
                val others = active.filter { it !== card }.mapNotNull { it.rate(rateKey) }
                val total = rate.samplesPerMs + others.sumOf { it.samplesPerMs }
                // Its share of what is left: more would finish after the others had done the rest.
                val share = (left * rate.samplesPerMs / total).toInt()
                val quickest = others.all { it.samplesPerMs <= rate.samplesPerMs }
                val slice = (rate.samplesPerMs * sliceMs).toInt().coerceIn(1, rate.lastCount * GROWTH)
                // The last odd sample is the quickest card's, so that someone takes it.
                minOf(slice, maxOf(share, if (quickest) 1 else 0), left)
            }
            if (size <= 0) return null
            val start = next
            next += size
            inFlight++
            launches++
            intArrayOf(start, size)
        }

        fun finish(count: Int) = synchronized(lock) {
            inFlight--
            finished += count
            onProgress?.invoke(finished)
            lock.notifyAll()
        }

        /**
         * A card failed in the middle of its samples. What it had added to this accumulation is
         * gone with it, so the range cannot be completed by the others: it ends here.
         */
        fun fail(cause: Throwable) = synchronized(lock) {
            inFlight--
            if (failure == null) failure = cause
            lock.notifyAll()
        }

        private var failure: Throwable? = null

        /** Waits for every sample to be traced; throws when a card failed or none could start. */
        fun await(): Int = synchronized(lock) {
            while (true) {
                if (failure != null && inFlight == 0) break
                if (next >= end && inFlight == 0) return launches
                if (pending == 0) break
                lock.wait()
            }
            val cause = failure
            throw IllegalStateException(
                if (cause != null) "a card failed while tracing: ${cause.message ?: cause}" else "no card holds the scene to trace it", cause
            )
        }
    }

    /** How fast a card traced last time, at one size and mode. */
    private class Rate(val samplesPerMs: Double, val lastCount: Int)

    /** One card: its session, the thread that owns it, and what that thread knows. */
    private class Card(private val label: String, private val open: () -> GpuSession) {
        private val commands = LinkedBlockingQueue<() -> Unit>()
        private val thread = Thread(::run, "gpu-$label").apply { isDaemon = true }
        private val rates = ConcurrentHashMap<Long, Rate>()
        @Volatile private var session: GpuSession? = null
        /** Why this card cannot be used, once it cannot. */
        @Volatile var broken: Throwable? = null
            private set
        @Volatile private var running = true

        // Owned by the card's thread.
        private var loadedScene = 0
        private var camera: CameraSetup? = null
        private val epochs = IntArray(TARGETS) { -1 }
        private var scratch = FloatArray(0)

        fun start(opened: CountDownLatch) {
            post {
                try {
                    session = open()
                } catch (t: Throwable) {
                    broken = t
                } finally {
                    opened.countDown()
                }
            }
            thread.start()
        }

        fun post(command: () -> Unit) = commands.put(command)

        fun join() = thread.join(CLOSE_WAIT_MS)

        fun devices(): List<String> = session?.devices ?: listOf(label)

        fun rate(key: Long): Rate? = rates[key]

        private fun run() {
            while (running) {
                val command = commands.poll(1, TimeUnit.SECONDS) ?: continue
                try {
                    command()
                } catch (t: Throwable) {
                    // A command reports its own failures; this is a last line so the thread lives on.
                    System.err.println("[gpu] $label: ${t.message ?: t}")
                }
            }
        }

        /** Loads [wanted] unless this card already holds it; a card that cannot load it sits the scene out. */
        fun loadScene(wanted: Pair<Int, SceneBuffers>?) {
            val (number, scene) = wanted ?: return
            val held = session ?: return
            if (number == loadedScene) return
            try {
                held.replaceScene(scene)
                loadedScene = number
                broken = null
            } catch (t: Throwable) {
                loadedScene = -1
                broken = t
                System.err.println("[gpu] $label sits this scene out: ${t.message ?: t}")
            }
            // The scene is new to the session, and so, as far as its outputs go, is everything on them.
            epochs.fill(-1)
        }

        fun trace(range: Range) {
            val held = session
            // A range for a scene this card does not hold: an older one, or one it could not load.
            if (held == null || broken != null || loadedScene != range.sceneNumber) {
                range.leave(this)
                return
            }
            range.join(this)
            try {
                if (camera !== range.camera) {
                    held.setCamera(range.camera)
                    camera = range.camera
                }
                held.seed = range.seed
                while (true) {
                    val chunk = range.claim(this) ?: break
                    try {
                        if (epochs[range.target] != range.epoch) {
                            // The first of its samples in this accumulation: a launch at sample 0
                            // overwrites the output by itself, any other adds and needs it empty.
                            if (chunk[0] > 0) held.clear(range.target)
                            epochs[range.target] = range.epoch
                        }
                        val start = System.nanoTime()
                        held.launch(range.mode, range.setup, chunk[0], chunk[1], range.scale, range.target)
                        val ms = maxOf((System.nanoTime() - start) / 1_000_000.0, 0.05)
                        rates[range.rateKey] = Rate(chunk[1] / ms, chunk[1])
                    } catch (t: Throwable) {
                        broken = t
                        epochs[range.target] = -1
                        System.err.println("[gpu] $label failed while tracing: ${t.message ?: t}")
                        range.fail(t)
                        return
                    }
                    range.finish(chunk[1])
                }
            } catch (t: Throwable) {
                broken = t
                System.err.println("[gpu] $label dropped out of the render: ${t.message ?: t}")
            } finally {
                range.leave(this)
            }
        }

        /** This card's output for [target] when it belongs to accumulation [epoch], else null. */
        fun readIfOn(target: Int, epoch: Int, floats: Int): FloatArray? {
            val held = session ?: return null
            if (broken != null || epochs[target] != epoch) return null
            if (scratch.size != floats) scratch = FloatArray(floats)
            held.read(target, scratch)
            return scratch
        }

        fun shutDown() {
            running = false
            try {
                session?.close()
            } finally {
                session = null
            }
        }
    }

    companion object {
        private const val TARGETS = 2
        /** How many times a card's turn may grow from one launch to the next. */
        private const val GROWTH = 4
        private const val CLOSE_WAIT_MS = 10_000L

        /**
         * Sessions on [picks], each opened on its card's own thread, with [scene] loaded. With
         * [lenient], a card that cannot be opened is left out with a message as long as another
         * can; otherwise the first failure is thrown.
         */
        fun open(picks: List<GpuPick>, scene: SceneBuffers, maxPixels: Int, lenient: Boolean): GpuSession =
            of(picks.map { pick -> pick.toString() to { GpuSession.openOn(pick, scene, maxPixels) } }, lenient)

        /** Cards made by [openers], each called on the thread that will own its session - also how tests put fakes in. */
        fun of(openers: List<Pair<String, () -> GpuSession>>, lenient: Boolean = false): GpuSession {
            val opened = CountDownLatch(openers.size)
            val cards = openers.map { (label, open) -> Card(label, open).also { it.start(opened) } }
            opened.await()
            val failed = cards.filter { it.broken != null }
            val usable = cards - failed.toSet()
            if (failed.isNotEmpty() && (!lenient || usable.isEmpty())) {
                cards.forEach { it.post { it.shutDown() } }
                cards.forEach { it.join() }
                val first = failed.first().broken!!
                throw IllegalStateException(first.message ?: first.toString(), first)
            }
            failed.forEach { card ->
                System.err.println("[gpu] leaving out ${card.devices().joinToString()}: ${card.broken?.message}")
                card.post { card.shutDown() }
            }
            return MultiGpuSession(usable)
        }
    }
}
