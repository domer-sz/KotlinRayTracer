package rayTraceTypescript.gpu

import org.lwjgl.cuda.CU.cuMemAlloc
import org.lwjgl.cuda.CU.cuMemAllocHost
import org.lwjgl.cuda.CU.CUDA_SUCCESS
import org.lwjgl.cuda.CU.cuMemFreeHost
import org.lwjgl.cuda.CU.cuMemFree
import org.lwjgl.cuda.CU.cuMemcpyHtoD
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil
import java.nio.ByteBuffer
import java.nio.FloatBuffer

/**
 * Memory on the card whose CUDA context is current: allocation and copies from the host. The
 * caller keeps the pointers and frees them.
 */
internal object DeviceMemory {

    /** [bytes] of device memory; an empty request still gets a few, so the pointer is valid. */
    fun allocate(bytes: Long): Long = MemoryStack.stackPush().use { stack ->
        val pointer = stack.mallocPointer(1)
        CudaContext.check(cuMemAlloc(pointer, maxOf(bytes, 4L)), "cuMemAlloc($bytes bytes)")
        pointer.get(0)
    }

    fun upload(data: FloatArray): Long = upload(data.size * 4L) { it.asFloatBuffer().put(data) }

    fun upload(data: IntArray): Long = upload(data.size * 4L) { it.asIntBuffer().put(data) }

    fun upload(data: ByteArray): Long = upload(data.size.toLong()) { it.put(data) }

    /** A copy of [data] from its start to its capacity, whatever its position. */
    fun upload(data: ByteBuffer): Long {
        val device = allocate(data.capacity().toLong())
        if (data.capacity() > 0) write(device, data)
        return device
    }

    /** Overwrites the start of [device] with all of [data]. */
    fun write(device: Long, data: ByteBuffer) {
        // LWJGL copies what lies between position and limit.
        val whole = data.duplicate().order(data.order())
        whole.clear()
        CudaContext.check(cuMemcpyHtoD(device, whole), "cuMemcpyHtoD(${whole.remaining()} bytes)")
    }

    fun free(device: Long) {
        if (device != 0L) cuMemFree(device)
    }

    /**
     * [floats] of host memory for copies to and from the card, page-locked when the driver allows:
     * the card then copies straight into it, instead of through a driver buffer at a fraction of
     * the speed (a 1080x1920 image came back in 17 ms from ordinary memory).
     */
    fun allocateHost(floats: Int): HostBuffer = MemoryStack.stackPush().use { stack ->
        val pointer = stack.mallocPointer(1)
        val bytes = floats * 4L
        if (cuMemAllocHost(pointer, bytes) == CUDA_SUCCESS) {
            HostBuffer(MemoryUtil.memFloatBuffer(pointer.get(0), floats), pinned = true)
        } else {
            HostBuffer(MemoryUtil.memAllocFloat(floats), pinned = false)
        }
    }

    /** Host memory from [allocateHost]; free it while its context is current. */
    class HostBuffer(val floats: FloatBuffer, private val pinned: Boolean) {
        fun free() {
            if (pinned) cuMemFreeHost(MemoryUtil.memByteBuffer(MemoryUtil.memAddress0(floats), floats.capacity() * 4)) else MemoryUtil.memFree(floats)
        }
    }

    private fun upload(bytes: Long, fill: (ByteBuffer) -> Unit): Long {
        val device = allocate(bytes)
        if (bytes == 0L) return device
        val host = MemoryUtil.memAlloc(bytes.toInt())
        try {
            fill(host)
            write(device, host)
        } finally {
            MemoryUtil.memFree(host)
        }
        return device
    }
}
