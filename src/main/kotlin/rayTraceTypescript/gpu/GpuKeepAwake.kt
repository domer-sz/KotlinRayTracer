package rayTraceTypescript.gpu

import org.lwjgl.glfw.GLFW.GLFW_FALSE
import org.lwjgl.glfw.GLFW.GLFW_VISIBLE
import org.lwjgl.glfw.GLFW.glfwCreateWindow
import org.lwjgl.glfw.GLFW.glfwDestroyWindow
import org.lwjgl.glfw.GLFW.glfwInit
import org.lwjgl.glfw.GLFW.glfwMakeContextCurrent
import org.lwjgl.glfw.GLFW.glfwTerminate
import org.lwjgl.glfw.GLFW.glfwWindowHint
import org.lwjgl.opengl.GL
import org.lwjgl.opengl.GL11.GL_COLOR_BUFFER_BIT
import org.lwjgl.opengl.GL11.glClear
import org.lwjgl.opengl.GL11.glFinish

/**
 * Keeps the NVIDIA driver on Windows from clocking the card down in the middle of a render.
 *
 * The driver judges idleness by the graphics engine, not by compute work: a few
 * seconds into a render it drops a GTX 1070 from P2 (1582-1936 MHz core, 3802 MHz memory) to
 * P5 (~700 / 810 MHz) with the reason "Idle" while the card is 100% busy, halving throughput.
 * Launch length (16 ms to 1 s) and queue depth make no difference, and GeForce cards refuse
 * nvidia-smi clock locks. A hidden OpenGL window clearing itself every 16 ms is enough to keep
 * the card in P2 for the whole render; it costs next to nothing.
 *
 * Windows only (GLFW wants the main thread on macOS), off with -Drt.keepawake=off, and never
 * allowed to break a render: if OpenGL cannot start, rendering just goes on without it.
 */
internal class GpuKeepAwake private constructor() : AutoCloseable {
    @Volatile
    private var running = true
    private val thread = Thread(::run, "gpu-keep-awake").apply { isDaemon = true }

    private fun run() {
        try {
            if (!glfwInit()) return
            try {
                glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE)
                val window = glfwCreateWindow(16, 16, "gpu-keep-awake", 0L, 0L)
                if (window == 0L) return
                try {
                    glfwMakeContextCurrent(window)
                    GL.createCapabilities()
                    while (running) {
                        glClear(GL_COLOR_BUFFER_BIT)
                        glFinish()
                        Thread.sleep(TICK_MS)
                    }
                } finally {
                    glfwDestroyWindow(window)
                }
            } finally {
                glfwTerminate()
            }
        } catch (_: InterruptedException) {
        } catch (t: Throwable) {
            System.err.println("[gpu] keep-awake unavailable: ${t.message}")
        }
    }

    override fun close() {
        running = false
        thread.join(1000)
    }

    companion object {
        private const val TICK_MS = 16L

        /** Starts the keep-alive where it helps and is allowed; null otherwise. */
        fun startIfUseful(): GpuKeepAwake? {
            if (System.getProperty("rt.keepawake") == "off") return null
            if (!System.getProperty("os.name", "").lowercase().contains("windows")) return null
            return GpuKeepAwake().also { it.thread.start() }
        }
    }
}
