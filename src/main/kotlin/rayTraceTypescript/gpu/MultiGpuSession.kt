package rayTraceTypescript.gpu

import rayTraceTypescript.render.CameraSetup
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
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
 * How fast each card is, is measured as it traces. A card takes no more than its share of what
 * is left, by speed, and none of a range too short for it to finish a sample before the others
 * finish the rest: a slow card never makes a launch take longer than the fast one alone would,
 * and of a short launch - a viewer's, a single sample - it hears nothing at all.
 *
 * A new scene goes to every card at once; a card still loading it joins the launch in progress
 * when it is ready and does not hold the others up, and one that cannot load it sits that scene
 * out. A card failing in the middle of a launch fails the launch - what it had traced is lost -
 * and is left out until a scene loads on it again.
 */
internal class MultiGpuSession private constructor(private val cards: List<Card>) : GpuSession {
    override var seed: Long = GpuSession.initialSeed()

    /** The scene the cards should hold, and its number: a card loads the newest when it gets to it. */
    @Volatile private var wanted: Pair<Int, GpuScene>? = null
    private var sceneNumber = 0
    private var camera: CameraSetup? = null

    /** Which accumulation each output is on, and the cards whose outputs add up to it. */
    private val epochs = IntArray(TARGETS)
    private val holders = Array(TARGETS) { LinkedHashSet<Card>() }

    override val devices: List<String> get() = cards.flatMap { it.devices() }

    override fun replaceScene(next: GpuScene) {
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
        val from = holders[target].toList()
        if (from.isEmpty()) {
            into.fill(0.0f)
            return
        }
        // One card's output is the image; several are read side by side and added up.
        val rest = from.drop(1).map { card -> card.ask { card.readScratch(it, target, into.size) } }
        from[0].now { it.read(target, into) }
        for (part in rest) {
            val values = part.await()
            for (i in into.indices) into[i] += values[i]
        }
    }

    override fun clear(target: Int) {
        // Nothing to do on the cards now: each empties its output when it first adds to the new accumulation.
        epochs[target]++
        holders[target].clear()
    }

    override fun prepare(mode: Int) {
        // Each card builds what it needs when it comes to a launch in that mode.
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
        if (first == 0) {
            epochs[target]++
            holders[target].clear()
        }
        val key = mode.toLong() shl 40 or setup.totalPixels.toLong()
        val now = System.nanoTime()
        // The cards holding the scene can start at once. Of those, only the ones with something
        // to gain are told of the range; a card still loading the scene hears of it and joins late.
        val ready = cards.filter { it.holds(sceneNumber) }
        val takers = ready.filter { card -> ready.size == 1 || card.wouldTake(key, count, ready, now) }
        val told = cards.filter { it !in ready || it in takers }
        val range = Range(
            mode, camera ?: setup, setup, first, first + count, scale, target, seed, sceneNumber, epochs[target],
            sliceMs, onProgress, key, told.size, takers
        )
        // The card likely to do most of the work traces on this thread, which would only wait
        // otherwise: a hand-over to its own thread and back costs about a millisecond, a lot
        // for a viewer's launches of a few samples. The others are told on theirs.
        val leader = takers.maxByOrNull { it.rate(key)?.samplesPerMs ?: 0.0 }
        told.forEach { card -> if (card !== leader) card.post { card.trace(range) } }
        leader?.now { leader.trace(range) }
        val launches = range.await()
        holders[target] += range.tracers()
        if (LOG) report(System.nanoTime() - now, count)
        return launches
    }

    // -Drt.gpus.log: every couple of seconds, what each card traced and how long it was busy.
    private var reportedAt = System.nanoTime()
    private var ranges = 0
    private var rangeSamples = 0
    private var rangeNs = 0L

    private fun report(ns: Long, samples: Int) {
        ranges++
        rangeSamples += samples
        rangeNs += ns
        val now = System.nanoTime()
        if (now - reportedAt < REPORT_NS) return
        System.err.println(
            "[gpus] $ranges ranges, $rangeSamples samples in ${rangeNs / 1_000_000} ms; " +
                cards.joinToString("; ") { it.report() }
        )
        reportedAt = now
        ranges = 0
        rangeSamples = 0
        rangeNs = 0L
    }

    /** A range of samples being shared out: what is left, who is on it, and how it ended. */
    private class Range(
        val mode: Int, val camera: CameraSetup, val setup: CameraSetup, val first: Int, val end: Int, val scale: Float,
        val target: Int, val seed: Long, val sceneNumber: Int, val epoch: Int, val sliceMs: Double,
        val onProgress: ((Int) -> Unit)?, val rateKey: Long, told: Int, ready: List<Card>
    ) {
        private val lock = Object()
        private var next = first
        private var inFlight = 0
        private var finished = 0
        private var launches = 0
        private var failure: Throwable? = null
        /** Cards told of this range that have not come to it yet, or are on it. */
        private var pending = told
        /**
         * The cards sharing the range: those ready for it from the start, not only those whose
         * thread has got to it. Otherwise whichever thread woke first would see itself alone and
         * take a whole turn - on a short range, a slow card taking all of it.
         */
        private val active = HashSet<Card>(ready)
        /** Cards that have taken samples of this range. */
        private val started = LinkedHashSet<Card>()
        /** Cards whose thread has come to this range with its kernel built. */
        private val arrived = HashSet<Card>()

        /** [card] is here and can trace at once. */
        fun arrive(card: Card) = synchronized(lock) {
            active += card
            arrived += card
            lock.notifyAll()
        }

        /**
         * Whether [card] counts when the others work out their shares: it is here, or about to
         * be. A card still building its kernel does not - nobody waits for a compiler.
         */
        private fun counts(card: Card) = card in arrived || card.isPrepared(mode)

        /** Cards tracing samples of this range right now. */
        private val flying = HashSet<Card>()
        /** Samples the quickest card left for another, which that card takes when it comes for them. */
        private val kept = HashMap<Card, Int>()

        /** [card] is through with this range - done, dropped out, or never able to start. */
        fun leave(card: Card) = synchronized(lock) {
            active -= card
            flying -= card
            kept -= card
            pending--
            lock.notifyAll()
        }

        fun tracers(): List<Card> = synchronized(lock) { started.toList() }

        /**
         * The samples [card] should trace next, as (first, count), or null when it should stop:
         * nothing is left, or the rest is better left to the others.
         *
         * The quickest card takes what is left, less what the cards that are free at that moment
         * would take: a sample for one that is not timed yet - without it, it never would be, and
         * a card that happens to be timed would pass for the only one, however slow - and for a
         * slower card its share by speed, rounded down. Any other card takes what was left for
         * it, or that share. Rounded down, because with more a card would still be tracing when
         * the others had done the rest; a range too short for a card is thus all the quickest
         * one's, in a single launch.
         */
        fun claim(card: Card): IntArray? = synchronized(lock) {
            var chunk: IntArray? = null
            var waits = 0
            while (true) {
                val left = end - next
                if (left <= 0 || failure != null) break
                val now = System.nanoTime()
                fun timing(of: Card): Rate? = of.rate(rateKey)?.takeUnless { it.stale(now) }
                val rate = timing(card)
                val size = if (rate == null) {
                    // Not timed at this size and mode, or not lately: one sample, to find out.
                    1
                } else {
                    val others = active.filter { it !== card && counts(it) }
                    val timed = others.mapNotNull { timing(it) }
                    val total = rate.samplesPerMs + timed.sumOf { it.samplesPerMs }
                    val slice = (rate.samplesPerMs * sliceMs).toInt().coerceIn(1, rate.lastCount * GROWTH)
                    if (timed.all { it.samplesPerMs <= rate.samplesPerMs }) {
                        // A card in the middle of its samples asks again when it is through.
                        val parts = if (waits >= KEEP_WAITS) emptyMap() else others.filter { it !in flying }.associateWith { other ->
                            timing(other)?.let { (left * it.samplesPerMs / total).toInt() } ?: 1
                        }.filterValues { it > 0 }
                        val free = left - parts.values.sum()
                        if (free <= 0) {
                            // All that is left is another card's, a thread's wake-up away.
                            waits++
                            lock.wait(KEEP_WAIT_MS)
                            continue
                        }
                        kept.clear()
                        kept += parts
                        minOf(slice, free)
                    } else {
                        minOf(slice, maxOf(kept[card] ?: 0, (left * rate.samplesPerMs / total).toInt()), left)
                    }
                }
                if (size > 0) {
                    chunk = intArrayOf(next, size)
                    next += size
                    inFlight++
                    launches++
                    kept -= card
                    flying += card
                    started += card
                    lock.notifyAll()
                }
                break
            }
            chunk
        }

        fun finish(card: Card, count: Int) = synchronized(lock) {
            inFlight--
            flying -= card
            finished += count
            onProgress?.invoke(finished)
            lock.notifyAll()
        }

        /**
         * A card failed in the middle of its samples. What it had added to this accumulation is
         * gone with it, so the range cannot be completed by the others: it ends here.
         */
        fun fail(card: Card, cause: Throwable) = synchronized(lock) {
            inFlight--
            flying -= card
            if (failure == null) failure = cause
            lock.notifyAll()
        }

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

    /** How fast a card traced, at one size and mode, and when that was measured. */
    private class Rate(val samplesPerMs: Double, val lastCount: Int, private val at: Long) {
        /**
         * Whether the card should be timed again: it has not traced for a while. A card left out
         * because it measured slow would otherwise stay left out on one bad measurement. The
         * while is longer for a card to which one sample is a long time.
         */
        fun stale(now: Long): Boolean = (now - at) / 1_000_000.0 > maxOf(REFRESH_MS, REFRESH_SAMPLES / samplesPerMs)
    }

    /** What a card is asked for and answers later; [await] rethrows what went wrong on its thread. */
    private class Answer<T>(private val future: CompletableFuture<T>) {
        fun await(): T = try {
            future.get()
        } catch (e: ExecutionException) {
            throw IllegalStateException(e.cause?.message ?: e.toString(), e.cause ?: e)
        }
    }

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
        /** The number of the scene the session holds; written by the card's thread only. */
        @Volatile private var loadedScene = 0

        // Owned by the card's thread.
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

        /**
         * Held by whoever is using the session: the card's own thread for each command, or the
         * session's caller doing a card's work itself ([now]). The engines bind the card's
         * context to whichever thread calls them, so either may - one at a time.
         */
        private val busy = java.util.concurrent.locks.ReentrantLock()

        /** Does [work] with the session on the calling thread, once the card's own is not using it. */
        fun <T> now(work: (GpuSession) -> T): T {
            busy.lock()
            try {
                return work(session ?: throw IllegalStateException("$label has no session"))
            } finally {
                busy.unlock()
            }
        }

        /** Has the card's thread do [work] with its session. */
        fun <T> ask(work: (GpuSession) -> T): Answer<T> {
            val future = CompletableFuture<T>()
            post {
                try {
                    future.complete(work(session ?: throw IllegalStateException("$label has no session")))
                } catch (t: Throwable) {
                    future.completeExceptionally(t)
                }
            }
            return Answer(future)
        }

        fun join() = thread.join(CLOSE_WAIT_MS)

        private val tracedSamples = java.util.concurrent.atomic.AtomicInteger()
        private val tracedLaunches = java.util.concurrent.atomic.AtomicInteger()
        private val tracedNs = java.util.concurrent.atomic.AtomicLong()

        /** What this card traced since it was last asked, for -Drt.gpus.log. */
        fun report(): String =
            "$label ${tracedSamples.getAndSet(0)} samples in ${tracedLaunches.getAndSet(0)} launches, ${tracedNs.getAndSet(0) / 1_000_000} ms"

        fun devices(): List<String> = session?.devices ?: listOf(label)

        fun rate(key: Long): Rate? = rates[key]

        /** Whether this card can start on scene number [scene] right away. */
        fun holds(scene: Int): Boolean = session != null && broken == null && loadedScene == scene

        /** The modes whose kernels this card has built for the scene it holds. */
        private val prepared = ConcurrentHashMap.newKeySet<Int>()

        fun isPrepared(mode: Int): Boolean = mode in prepared

        /**
         * Whether a range of [count] samples has anything for this card, [ready] being the cards
         * it would share it with: it is to be timed, or it is the quickest, or its share by
         * speed comes to a sample at least.
         */
        fun wouldTake(key: Long, count: Int, ready: List<Card>, now: Long): Boolean {
            val mine = rates[key]?.takeUnless { it.stale(now) } ?: return true
            val others = ready.filter { it !== this }.mapNotNull { other -> other.rates[key]?.takeUnless { it.stale(now) } }
            if (others.all { it.samplesPerMs <= mine.samplesPerMs }) return true
            return (count * mine.samplesPerMs / (mine.samplesPerMs + others.sumOf { it.samplesPerMs })).toInt() >= 1
        }

        private fun run() {
            while (running) {
                val command = commands.poll(1, TimeUnit.SECONDS) ?: continue
                busy.lock()
                try {
                    command()
                } catch (t: Throwable) {
                    // A command reports its own failures; this is a last line so the thread lives on.
                    System.err.println("[gpu] $label: ${t.message ?: t}")
                } finally {
                    busy.unlock()
                }
            }
        }

        /** Loads [wanted] unless this card already holds it; a card that cannot load it sits the scene out. */
        fun loadScene(wanted: Pair<Int, GpuScene>?) {
            val (number, scene) = wanted ?: return
            val held = session ?: return
            if (number == loadedScene) return
            // Another scene may need other kernels.
            prepared.clear()
            try {
                held.replaceScene(scene)
                broken = null
                loadedScene = number
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
            try {
                if (camera !== range.camera) {
                    held.setCamera(range.camera)
                    camera = range.camera
                }
                held.seed = range.seed
                // Kernels are built before any sample is timed: a card's speed must not look
                // like its compiler's. Until then the others share the range without it.
                held.prepare(range.mode)
                prepared += range.mode
                range.arrive(this)
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
                        val end = System.nanoTime()
                        rates[range.rateKey] = Rate(chunk[1] / maxOf((end - start) / 1_000_000.0, 0.05), chunk[1], end)
                        tracedSamples.addAndGet(chunk[1])
                        tracedLaunches.incrementAndGet()
                        tracedNs.addAndGet(end - start)
                    } catch (t: Throwable) {
                        broken = t
                        epochs[range.target] = -1
                        System.err.println("[gpu] $label failed while tracing: ${t.message ?: t}")
                        range.fail(this, t)
                        return
                    }
                    range.finish(this, chunk[1])
                }
            } catch (t: Throwable) {
                broken = t
                System.err.println("[gpu] $label dropped out of the render: ${t.message ?: t}")
            } finally {
                range.leave(this)
            }
        }

        /** This card's output for [target], in an array of its own that the next read reuses. */
        fun readScratch(held: GpuSession, target: Int, floats: Int): FloatArray {
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
        /** How long, in waits of so many milliseconds, the quickest card leaves the last samples to the card they are kept for. */
        private const val KEEP_WAITS = 5
        private const val KEEP_WAIT_MS = 2L
        /** A card that has not traced for this long, or for the time it needs for this many samples, is timed again. */
        private const val REFRESH_MS = 3000.0
        private const val REFRESH_SAMPLES = 50.0
        private const val CLOSE_WAIT_MS = 10_000L
        private val LOG = System.getProperty("rt.gpus.log") != null
        private const val REPORT_NS = 2_000_000_000L

        /**
         * Sessions on [picks], each opened on its card's own thread, with [scene] loaded. With
         * [lenient], a card that cannot be opened is left out with a message as long as another
         * can; otherwise the first failure is thrown.
         */
        fun open(picks: List<GpuPick>, scene: GpuScene, maxPixels: Int, lenient: Boolean): GpuSession =
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
