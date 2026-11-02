package rayTraceTypescript.gpu

import org.lwjgl.cuda.CU.CUDA_SUCCESS
import org.lwjgl.cuda.CU.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
import org.lwjgl.cuda.CU.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
import org.lwjgl.cuda.CU.cuCtxSetCurrent
import org.lwjgl.cuda.CU.cuDeviceGet
import org.lwjgl.cuda.CU.cuDeviceGetAttribute
import org.lwjgl.cuda.CU.cuDeviceGetCount
import org.lwjgl.cuda.CU.cuDeviceGetName
import org.lwjgl.cuda.CU.cuDevicePrimaryCtxRelease
import org.lwjgl.cuda.CU.cuDevicePrimaryCtxRetain
import org.lwjgl.cuda.CU.cuGetErrorString
import org.lwjgl.cuda.CU.cuInit
import org.lwjgl.cuda.CU.cuModuleGetFunction
import org.lwjgl.cuda.CU.cuModuleLoadData
import org.lwjgl.cuda.NVRTC.NVRTC_SUCCESS
import org.lwjgl.cuda.NVRTC.nvrtcCompileProgram
import org.lwjgl.cuda.NVRTC.nvrtcCreateProgram
import org.lwjgl.cuda.NVRTC.nvrtcDestroyProgram
import org.lwjgl.cuda.NVRTC.nvrtcGetCUBIN
import org.lwjgl.cuda.NVRTC.nvrtcGetCUBINSize
import org.lwjgl.cuda.NVRTC.nvrtcGetErrorString
import org.lwjgl.cuda.NVRTC.nvrtcGetNumSupportedArchs
import org.lwjgl.cuda.NVRTC.nvrtcGetPTX
import org.lwjgl.cuda.NVRTC.nvrtcGetPTXSize
import org.lwjgl.cuda.NVRTC.nvrtcGetProgramLog
import org.lwjgl.cuda.NVRTC.nvrtcGetProgramLogSize
import org.lwjgl.cuda.NVRTC.nvrtcGetSupportedArchs
import org.lwjgl.system.Configuration
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil
import java.io.File

/**
 * A CUDA device and its primary context, plus the kernel compiler.
 *
 * The driver API comes from `nvcuda.dll`, which the NVIDIA driver installs. The kernel is CUDA
 * C++ compiled at run time by NVRTC - for the card found, with the scene's features switched in -
 * from the copy in native/cuda (see its README), so no CUDA Toolkit is needed.
 *
 * A CUDA context is current on one thread at a time; [makeCurrent] binds it to the caller, and
 * the session calls it on every entry so whichever thread drives it is bound.
 */
class CudaContext private constructor(
    val device: Int,
    val context: Long,
    val deviceName: String,
    /** Compute capability as major * 10 + minor, e.g. 61 for a GTX 1070. */
    val computeCapability: Int
) : AutoCloseable {

    fun makeCurrent() = check(cuCtxSetCurrent(context), "cuCtxSetCurrent")

    /**
     * Compiles [source] with [options] and loads it, returning the module and its `render`
     * function. Code for the card itself (a cubin) when NVRTC knows the card; otherwise PTX for
     * the newest architecture it knows, which the driver finishes compiling.
     */
    fun buildKernel(source: String, options: List<String>): Pair<Long, Long> {
        makeCurrent()
        MemoryStack.stackPush().use { stack ->
            val programOut = stack.mallocPointer(1)
            checkNvrtc(nvrtcCreateProgram(programOut, source, "raytracer.cu", null, null), "nvrtcCreateProgram")
            val program = programOut.get(0)
            try {
                val supported = supportedArchitectures()
                // -Drt.cuda.ptx hands the driver PTX to finish instead of a ready cubin.
                val native = computeCapability in supported && System.getProperty("rt.cuda.ptx") == null
                val arch = if (native) "--gpu-architecture=sm_$computeCapability"
                else "--gpu-architecture=compute_${supported.filter { it <= computeCapability }.maxOrNull() ?: supported.min()}"
                val all = listOf(arch) + options
                val pointers = stack.mallocPointer(all.size)
                all.forEachIndexed { i, option -> pointers.put(i, stack.UTF8(option)) }
                val status = nvrtcCompileProgram(program, pointers)
                val log = programLog(program)
                if (status != NVRTC_SUCCESS) {
                    throw CudaException("Kernel build failed (${nvrtcGetErrorString(status)}):\n$log")
                }
                // -Drt.cuda.log shows what the compiler said even on success.
                if (System.getProperty("rt.cuda.log") != null && log.isNotBlank()) System.err.println("[gpu] build log:\n$log")

                val image = if (native) {
                    val size = stack.mallocPointer(1)
                    checkNvrtc(nvrtcGetCUBINSize(program, size), "nvrtcGetCUBINSize")
                    MemoryUtil.memAlloc(size.get(0).toInt()).also { checkNvrtc(nvrtcGetCUBIN(program, it), "nvrtcGetCUBIN") }
                } else {
                    val size = stack.mallocPointer(1)
                    checkNvrtc(nvrtcGetPTXSize(program, size), "nvrtcGetPTXSize")
                    MemoryUtil.memAlloc(size.get(0).toInt()).also { checkNvrtc(nvrtcGetPTX(program, it), "nvrtcGetPTX") }
                }
                try {
                    // -Drt.cuda.dump=<file> keeps what NVRTC produced, for reading PTX.
                    System.getProperty("rt.cuda.dump")?.let { path ->
                        val bytes = ByteArray(image.remaining()).also { image.duplicate().get(it) }
                        java.nio.file.Files.write(java.nio.file.Paths.get(path), bytes)
                    }
                    val module = stack.mallocPointer(1)
                    // PTX is finished by the driver's compiler, which needs the register cap again.
                    val cap = options.firstOrNull { it.startsWith("--maxrregcount=") }?.substringAfter('=')?.toLongOrNull()
                    if (!native && cap != null) {
                        val keys = stack.ints(org.lwjgl.cuda.CU.CU_JIT_MAX_REGISTERS)
                        val values = stack.mallocPointer(1).put(0, cap)
                        check(org.lwjgl.cuda.CU.cuModuleLoadDataEx(module, image, keys, values), "cuModuleLoadDataEx")
                    } else {
                        check(cuModuleLoadData(module, image), "cuModuleLoadData")
                    }
                    val function = stack.mallocPointer(1)
                    check(cuModuleGetFunction(function, module.get(0), "render"), "cuModuleGetFunction")
                    return module.get(0) to function.get(0)
                } finally {
                    MemoryUtil.memFree(image)
                }
            } finally {
                programOut.put(0, program)
                nvrtcDestroyProgram(programOut)
            }
        }
    }

    private fun programLog(program: Long): String = MemoryStack.stackPush().use { stack ->
        val size = stack.mallocPointer(1)
        if (nvrtcGetProgramLogSize(program, size) != NVRTC_SUCCESS || size.get(0) <= 1) return ""
        val log = MemoryUtil.memAlloc(size.get(0).toInt())
        try {
            nvrtcGetProgramLog(program, log)
            MemoryUtil.memUTF8(log, log.capacity() - 1).trim()
        } finally {
            MemoryUtil.memFree(log)
        }
    }

    override fun close() {
        cuDevicePrimaryCtxRelease(device)
    }

    companion object {
        /** Device name filter, e.g. RT_GPU_DEVICE=1070 (RT_OPENCL_DEVICE is still read). */
        private const val DEVICE_FILTER_ENV = "RT_GPU_DEVICE"
        /** Where the repository keeps NVRTC, relative to the project root. */
        private const val BUNDLED_NVRTC = "native/cuda/windows-x64"
        private const val NVRTC_DLL = "nvrtc64_120_0.dll"
        private val NVRTC_BUILTINS_DLL = Regex("nvrtc-builtins64_\\d+\\.dll", RegexOption.IGNORE_CASE)

        @Volatile private var initialised = false

        fun create(): CudaContext {
            initialise()
            MemoryStack.stackPush().use { stack ->
                val count = stack.mallocInt(1)
                check(cuDeviceGetCount(count), "cuDeviceGetCount")
                if (count.get(0) == 0) throw CudaException("no CUDA device found")
                val filter = System.getenv(DEVICE_FILTER_ENV) ?: System.getenv("RT_OPENCL_DEVICE")
                val handle = stack.mallocInt(1)
                val devices = (0 until count.get(0)).map { ordinal ->
                    check(cuDeviceGet(handle, ordinal), "cuDeviceGet")
                    handle.get(0) to deviceName(handle.get(0))
                }
                val (device, name) = devices.firstOrNull { filter.isNullOrBlank() || it.second.contains(filter, ignoreCase = true) }
                    ?: throw CudaException("no CUDA device matches $DEVICE_FILTER_ENV=$filter (found ${devices.joinToString { it.second }})")
                val value = stack.mallocInt(1)
                check(cuDeviceGetAttribute(value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute")
                val major = value.get(0)
                check(cuDeviceGetAttribute(value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute")
                val minor = value.get(0)
                val context = stack.mallocPointer(1)
                check(cuDevicePrimaryCtxRetain(context, device), "cuDevicePrimaryCtxRetain")
                return CudaContext(device, context.get(0), name, major * 10 + minor).also { it.makeCurrent() }
            }
        }

        /** Null when a CUDA device and NVRTC are usable, otherwise a short reason. */
        fun unavailabilityReason(): String? = try {
            create().use { supportedArchitectures(); null }
        } catch (t: Throwable) {
            t.message ?: t::class.java.simpleName
        }

        /** The folder NVRTC comes from: -Drt.nvrtc, NVRTC_DIR, or the copy in the repository. */
        fun nvrtcFolder(): File = File(System.getProperty("rt.nvrtc") ?: System.getenv("NVRTC_DIR") ?: BUNDLED_NVRTC)

        @Synchronized
        private fun initialise() {
            if (initialised) return
            // Before the NVRTC class initialises: it loads the library once, and a failed
            // initialisation cannot be retried in the same JVM. Both by full path - the compiler
            // loads its builtins by name, which Windows would not look for beside it.
            val folder = nvrtcFolder()
            val compiler = File(folder, NVRTC_DLL)
            // The builtins carry the minor version in their name (nvrtc-builtins64_126.dll for 12.6).
            val builtins = folder.listFiles { f -> f.name.matches(NVRTC_BUILTINS_DLL) }?.maxByOrNull { it.name }
            if (!compiler.isFile || builtins == null) {
                throw CudaException("NVRTC not found: expected $NVRTC_DLL and nvrtc-builtins64_12x.dll in ${folder.absolutePath} (-Drt.nvrtc / NVRTC_DIR)")
            }
            if (Configuration.CUDA_NVRTC_LIBRARY_NAME.get() == null) Configuration.CUDA_NVRTC_LIBRARY_NAME.set(compiler.absolutePath)
            if (Configuration.CUDA_NVRTC_BUILTINS_LIBRARY_NAME.get() == null) {
                Configuration.CUDA_NVRTC_BUILTINS_LIBRARY_NAME.set(builtins.absolutePath)
            }
            try {
                check(cuInit(0), "cuInit")
            } catch (e: CudaException) {
                throw e
            } catch (t: Throwable) {
                throw CudaException("could not load the CUDA driver (nvcuda): ${t.message}", t)
            }
            initialised = true
        }

        /** The architectures NVRTC can compile for, as major * 10 + minor. */
        private fun supportedArchitectures(): Set<Int> = MemoryStack.stackPush().use { stack ->
            val count = stack.mallocInt(1)
            checkNvrtc(nvrtcGetNumSupportedArchs(count), "nvrtcGetNumSupportedArchs")
            val archs = stack.mallocInt(count.get(0))
            checkNvrtc(nvrtcGetSupportedArchs(archs), "nvrtcGetSupportedArchs")
            (0 until count.get(0)).map { archs.get(it) }.toSet()
        }

        private fun deviceName(device: Int): String = MemoryStack.stackPush().use { stack ->
            val name = stack.malloc(256)
            check(cuDeviceGetName(name, device), "cuDeviceGetName")
            MemoryUtil.memASCII(name, (0 until 256).first { name.get(it) == 0.toByte() }).trim()
        }

        fun check(status: Int, operation: String) {
            if (status == CUDA_SUCCESS) return
            val text = MemoryStack.stackPush().use { stack ->
                val message = stack.mallocPointer(1)
                if (cuGetErrorString(status, message) == CUDA_SUCCESS && message.get(0) != 0L) MemoryUtil.memUTF8(message.get(0)) else "error $status"
            }
            throw CudaException("$operation failed: $text ($status)")
        }

        private fun checkNvrtc(status: Int, operation: String) {
            if (status != NVRTC_SUCCESS) throw CudaException("$operation failed: ${nvrtcGetErrorString(status)}")
        }
    }
}

class CudaException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
