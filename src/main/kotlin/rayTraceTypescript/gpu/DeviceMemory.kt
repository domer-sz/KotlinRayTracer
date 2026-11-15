package rayTraceTypescript.gpu

import org.lwjgl.cuda.CU.cuMemAlloc
import org.lwjgl.cuda.CU.cuMemFree
import org.lwjgl.cuda.CU.cuMemcpyHtoD
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil
import java.nio.ByteBuffer

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
