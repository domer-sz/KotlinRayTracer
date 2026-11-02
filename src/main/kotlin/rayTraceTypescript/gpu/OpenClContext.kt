package rayTraceTypescript.gpu

import org.lwjgl.PointerBuffer
import org.lwjgl.opencl.CL
import org.lwjgl.opencl.CL10.CL_CONTEXT_PLATFORM
import org.lwjgl.opencl.CL10.CL_DEVICE_EXTENSIONS
import org.lwjgl.opencl.CL10.CL_DEVICE_NAME
import org.lwjgl.opencl.CL10.CL_DEVICE_TYPE_ALL
import org.lwjgl.opencl.CL10.CL_DEVICE_TYPE_GPU
import org.lwjgl.opencl.CL10.CL_PROGRAM_BUILD_LOG
import org.lwjgl.opencl.CL10.CL_SUCCESS
import org.lwjgl.opencl.CL10.clBuildProgram
import org.lwjgl.opencl.CL10.clCreateCommandQueue
import org.lwjgl.opencl.CL10.clCreateContext
import org.lwjgl.opencl.CL10.clCreateProgramWithSource
import org.lwjgl.opencl.CL10.clGetDeviceIDs
import org.lwjgl.opencl.CL10.clGetDeviceInfo
import org.lwjgl.opencl.CL10.clGetPlatformIDs
import org.lwjgl.opencl.CL10.clGetProgramBuildInfo
import org.lwjgl.opencl.CL10.clGetProgramInfo
import org.lwjgl.opencl.CL10.clReleaseCommandQueue
import org.lwjgl.opencl.CL10.clReleaseContext
import org.lwjgl.system.Configuration
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil

/** An OpenCL device with a context and a command queue, plus the program build helper. */
class OpenClContext private constructor(
    val device: Long,
    val context: Long,
    val queue: Long,
    val deviceName: String
) : AutoCloseable {

    /** The extensions the device reports, e.g. cl_nv_compiler_options for NVIDIA's build flags. */
    val extensions: String by lazy { deviceString(device, CL_DEVICE_EXTENSIONS) }

    /** Compiles a kernel program; throws with the device's build log when compilation fails. */
    fun buildProgram(source: String, options: String): Long {
        MemoryStack.stackPush().use { stack ->
            val errcode = stack.mallocInt(1)
            val program = clCreateProgramWithSource(context, source, errcode)
            check(errcode, "clCreateProgramWithSource")

            val status = clBuildProgram(program, device, options, null, 0)
            if (status != CL_SUCCESS) {
                throw OpenClException("Kernel build failed (error $status):\n${buildLog(program)}")
            }
            // -Drt.cl.log shows what the compiler said even on success - with NVIDIA's
            // -cl-nv-verbose that is the register and spill count, the kernel's key limits.
            if (System.getProperty("rt.cl.log") != null) System.err.println("[gpu] build log:\n${buildLog(program)}")
            // -Drt.cl.dump=<file> keeps the program binary - PTX text on NVIDIA.
            System.getProperty("rt.cl.dump")?.let { path ->
                val sizes = stack.mallocPointer(1)
                clGetProgramInfo(program, org.lwjgl.opencl.CL10.CL_PROGRAM_BINARY_SIZES, sizes, null)
                val binary = MemoryUtil.memAlloc(sizes.get(0).toInt())
                try {
                    val pointers = stack.mallocPointer(1).put(0, MemoryUtil.memAddress(binary))
                    clGetProgramInfo(program, org.lwjgl.opencl.CL10.CL_PROGRAM_BINARIES, pointers, null)
                    java.nio.file.Files.write(java.nio.file.Paths.get(path), ByteArray(binary.remaining()).also { binary.get(it) })
                } finally {
                    MemoryUtil.memFree(binary)
                }
            }
            return program
        }
    }

    private fun buildLog(program: Long): String {
        MemoryStack.stackPush().use { stack ->
            val size = stack.mallocPointer(1)
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, null as java.nio.ByteBuffer?, size)
            val log = MemoryUtil.memAlloc(size.get(0).toInt().coerceAtLeast(1))
            try {
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log, null)
                return MemoryUtil.memUTF8(log, log.capacity() - 1).trim()
            } finally {
                MemoryUtil.memFree(log)
            }
        }
    }

    override fun close() {
        clReleaseCommandQueue(queue)
        clReleaseContext(context)
    }

    companion object {
        /** Device name filter, e.g. RT_OPENCL_DEVICE=NVIDIA. */
        private const val DEVICE_FILTER_ENV = "RT_OPENCL_DEVICE"

        private var libraryLoaded = false

        fun create(): OpenClContext {
            loadLibrary()
            MemoryStack.stackPush().use { stack ->
                val device = selectDevice(stack) ?: throw OpenClException("no OpenCL device found")
                val properties = stack.mallocPointer(3)
                properties.put(0, CL_CONTEXT_PLATFORM.toLong()).put(1, device.platform).put(2, 0)

                val errcode = stack.mallocInt(1)
                val context = clCreateContext(properties, device.id, null, 0, errcode)
                check(errcode, "clCreateContext")
                val queue = clCreateCommandQueue(context, device.id, 0, errcode)
                check(errcode, "clCreateCommandQueue")
                return OpenClContext(device.id, context, queue, device.name)
            }
        }

        /** Null when a device is usable, otherwise a short reason for the fallback message. */
        fun unavailabilityReason(): String? = try {
            create().use { null }
        } catch (t: Throwable) {
            t.message ?: t::class.java.simpleName
        }

        private fun loadLibrary() {
            if (libraryLoaded) return
            // Must happen before the CL class initializes: it loads the library once, and a
            // failed initialization cannot be retried in the same JVM.
            if (Configuration.OPENCL_LIBRARY_NAME.get() == null) {
                findIcdLoader()?.let { Configuration.OPENCL_LIBRARY_NAME.set(it) }
            }
            try {
                CL.create()
            } catch (_: IllegalStateException) {
                // Already created by the class initializer or an earlier renderer.
            } catch (t: Throwable) {
                throw OpenClException("could not load the OpenCL ICD loader: ${t.message}", t)
            }
            libraryLoaded = true
        }

        /**
         * LWJGL looks for `libOpenCL.so`, which only the -dev packages install. When just the
         * versioned runtime library is present, point LWJGL at it.
         */
        private fun findIcdLoader(): String? {
            if (!System.getProperty("os.name", "").lowercase().contains("linux")) return null
            val directories = listOf(
                "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/lib"
            ) + (System.getenv("LD_LIBRARY_PATH")?.split(':')?.filter { it.isNotBlank() } ?: emptyList())

            if (directories.any { java.io.File(it, "libOpenCL.so").exists() }) return null
            return directories
                .map { java.io.File(it, "libOpenCL.so.1") }
                .firstOrNull { it.exists() }
                ?.absolutePath
        }

        private class DeviceHandle(val platform: Long, val id: Long, val name: String)

        private fun selectDevice(stack: MemoryStack): DeviceHandle? {
            val count = stack.mallocInt(1)
            if (clGetPlatformIDs(null as PointerBuffer?, count) != CL_SUCCESS || count.get(0) == 0) return null
            val platforms = stack.mallocPointer(count.get(0))
            check(clGetPlatformIDs(platforms, null as java.nio.IntBuffer?), "clGetPlatformIDs")

            val filter = System.getenv(DEVICE_FILTER_ENV)
            val candidates = mutableListOf<Pair<DeviceHandle, Boolean>>()

            for (p in 0 until platforms.capacity()) {
                val platform = platforms.get(p)
                // CL_DEVICE_TYPE_ALL is 0xFFFFFFFF; LWJGL exposes it as a negative Int.
                val gpuOnly = CL_DEVICE_TYPE_GPU.toLong()
                val anyDevice = CL_DEVICE_TYPE_ALL.toLong() and 0xFFFFFFFFL
                for (type in listOf(gpuOnly, anyDevice)) {
                    if (clGetDeviceIDs(platform, type, null as PointerBuffer?, count) != CL_SUCCESS || count.get(0) == 0) continue
                    val devices = stack.mallocPointer(count.get(0))
                    if (clGetDeviceIDs(platform, type, devices, null as java.nio.IntBuffer?) != CL_SUCCESS) continue
                    for (d in 0 until devices.capacity()) {
                        val id = devices.get(d)
                        val handle = DeviceHandle(platform, id, deviceName(id))
                        if (candidates.none { it.first.id == id }) {
                            candidates += handle to (type == gpuOnly)
                        }
                    }
                }
            }

            val matching = if (filter.isNullOrBlank()) candidates
            else candidates.filter { it.first.name.contains(filter, ignoreCase = true) }
            // Prefer a real GPU; fall back to whatever else the platform exposes.
            return (matching.firstOrNull { it.second } ?: matching.firstOrNull())?.first
        }

        private fun deviceName(device: Long): String = deviceString(device, CL_DEVICE_NAME)

        private fun deviceString(device: Long, param: Int): String {
            MemoryStack.stackPush().use { stack ->
                val size = stack.mallocPointer(1)
                clGetDeviceInfo(device, param, null as java.nio.ByteBuffer?, size)
                val name = MemoryUtil.memAlloc(size.get(0).toInt().coerceAtLeast(1))
                try {
                    clGetDeviceInfo(device, param, name, null)
                    return MemoryUtil.memUTF8(name, name.capacity() - 1).trim()
                } finally {
                    MemoryUtil.memFree(name)
                }
            }
        }

        fun check(errcode: java.nio.IntBuffer, operation: String) = check(errcode.get(0), operation)

        fun check(errcode: IntArray, operation: String) = check(errcode[0], operation)

        fun check(status: Int, operation: String) {
            if (status != CL_SUCCESS) throw OpenClException("$operation failed with error $status")
        }
    }
}

class OpenClException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
