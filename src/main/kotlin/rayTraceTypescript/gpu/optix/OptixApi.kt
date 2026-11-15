package rayTraceTypescript.gpu.optix

import org.lwjgl.system.CallbackI
import org.lwjgl.system.Library
import org.lwjgl.system.MemoryUtil
import org.lwjgl.system.Pointer
import org.lwjgl.system.libffi.FFICIF
import org.lwjgl.system.libffi.LibFFI
import org.lwjgl.system.windows.WinBase
import rayTraceTypescript.utils.NativeFunction
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * NVIDIA OptiX - what reaches a card's RT cores - without a binding library.
 *
 * OptiX is two things: `nvoptix.dll`, which the NVIDIA driver installs, and the SDK's headers.
 * The library exports one function, which fills a table of function pointers for the ABI version
 * asked for; each is then called through libffi ([NativeFunction]). The headers are needed when
 * NVRTC compiles the kernel and to learn how OptiX's structures are laid out ([OptixLayout]);
 * they are NVIDIA's and not in the repository, see native/optix/README.md.
 *
 * The table's order and [ABI_VERSION] are OptiX 9.0's: a driver of the 570 series or newer, and
 * the newest version a driver that still runs a GTX 10 card knows.
 */
internal object OptixApi {
    const val ABI_VERSION = 105

    /** Where native/optix/fetch.ps1 puts the headers, relative to the project root. */
    private const val BUNDLED_HEADERS = "native/optix/include"

    /** The functions of OptixFunctionTable in its order, from optix_function_table.h of 9.0.0. */
    private val TABLE = listOf(
        "optixGetErrorName", "optixGetErrorString",
        "optixDeviceContextCreate", "optixDeviceContextDestroy", "optixDeviceContextGetProperty",
        "optixDeviceContextSetLogCallback", "optixDeviceContextSetCacheEnabled", "optixDeviceContextSetCacheLocation",
        "optixDeviceContextSetCacheDatabaseSizes", "optixDeviceContextGetCacheEnabled",
        "optixDeviceContextGetCacheLocation", "optixDeviceContextGetCacheDatabaseSizes",
        "optixModuleCreate", "optixModuleCreateWithTasks", "optixModuleGetCompilationState", "optixModuleDestroy",
        "optixBuiltinISModuleGet", "optixTaskExecute",
        "optixProgramGroupCreate", "optixProgramGroupDestroy", "optixProgramGroupGetStackSize",
        "optixPipelineCreate", "optixPipelineDestroy", "optixPipelineSetStackSize",
        "optixAccelComputeMemoryUsage", "optixAccelBuild", "optixAccelGetRelocationInfo",
        "optixCheckRelocationCompatibility", "optixAccelRelocate", "optixAccelCompact", "optixAccelEmitProperty",
        "optixConvertPointerToTraversableHandle",
        "optixOpacityMicromapArrayComputeMemoryUsage", "optixOpacityMicromapArrayBuild",
        "optixOpacityMicromapArrayGetRelocationInfo", "optixOpacityMicromapArrayRelocate",
        "optixDisplacementMicromapArrayComputeMemoryUsage", "optixDisplacementMicromapArrayBuild",
        "optixClusterAccelComputeMemoryUsage", "optixClusterAccelBuild",
        "optixSbtRecordPackHeader", "optixLaunch",
        "optixCoopVecMatrixConvert", "optixCoopVecMatrixComputeSize",
        "optixDenoiserCreate", "optixDenoiserDestroy", "optixDenoiserComputeMemoryResources", "optixDenoiserSetup",
        "optixDenoiserInvoke", "optixDenoiserComputeIntensity", "optixDenoiserComputeAverageColor",
        "optixDenoiserCreateWithUserModel"
    )

    /** The functions the renderer calls, as [NativeFunction] signatures. */
    private val SIGNATURES = mapOf(
        "optixGetErrorName" to "pi",
        "optixDeviceContextCreate" to "ippp",
        "optixDeviceContextDestroy" to "ip",
        "optixDeviceContextGetProperty" to "ipipl",
        "optixModuleCreate" to "ipppplppp",
        "optixModuleDestroy" to "ip",
        "optixProgramGroupCreate" to "ippupppp",
        "optixProgramGroupDestroy" to "ip",
        "optixProgramGroupGetStackSize" to "ippp",
        "optixPipelineCreate" to "ippppuppp",
        "optixPipelineDestroy" to "ip",
        "optixPipelineSetStackSize" to "ipuuuu",
        "optixAccelComputeMemoryUsage" to "ipppup",
        "optixAccelBuild" to "ippppullllppu",
        "optixAccelCompact" to "ipplllp",
        "optixSbtRecordPackHeader" to "ipp",
        "optixLaunch" to "ippllpuuu"
    )

    private val table: Long by lazy { load() }
    private val functions = ConcurrentHashMap<String, NativeFunction>()

    /** The folder the headers are in: -Drt.optix.include, OPTIX_INCLUDE, or where fetch.ps1 puts them. */
    fun includeFolder(): File = File(System.getProperty("rt.optix.include") ?: System.getenv("OPTIX_INCLUDE") ?: BUNDLED_HEADERS)

    /** The NVRTC option that lets a kernel include the headers. */
    fun includeOption(): String = "--include-path=${includeFolder().absolutePath}"

    /**
     * Null when OptiX can be used here, otherwise why not. Call it once a CUDA context exists:
     * the driver's library is found beside the CUDA one, which is loaded by then.
     */
    fun unavailabilityReason(): String? = try {
        if (!File(includeFolder(), "optix.h").isFile) {
            "no OptiX headers in ${includeFolder().absolutePath} (see native/optix/README.md)"
        } else {
            table
            null
        }
    } catch (t: Throwable) {
        t.message ?: t::class.java.simpleName
    }

    /** Calls the OptiX function [name] and throws unless it reports success. */
    fun call(name: String, vararg args: Any?) {
        val result = function(name)(*args).toInt()
        if (result != 0) throw OptixException("$name failed: ${errorName(result)} ($result)")
    }

    /**
     * Calls [name], one of the functions that also write a log: [arguments] gets the log buffer
     * and the address of its size and returns the full argument list. A failure carries the
     * log; -Drt.optix.log prints it on success too.
     */
    fun callWithLog(name: String, arguments: (log: Long, logSize: Long) -> Array<Any?>) {
        val log = MemoryUtil.memCalloc(LOG_BYTES)
        val size = MemoryUtil.memAllocLong(1).put(0, LOG_BYTES.toLong())
        try {
            val result = function(name)(*arguments(MemoryUtil.memAddress(log), MemoryUtil.memAddress(size))).toInt()
            val text = MemoryUtil.memUTF8(log, minOf(size.get(0), LOG_BYTES.toLong()).toInt().coerceAtLeast(0)).trimEnd('\u0000', '\n', ' ')
            if (result != 0) throw OptixException("$name failed: ${errorName(result)} ($result)" + if (text.isEmpty()) "" else "\n$text")
            if (logLevel() >= 3 && text.isNotEmpty()) System.err.println("[optix] $name: $text")
        } finally {
            MemoryUtil.memFree(log)
            MemoryUtil.memFree(size)
        }
    }

    /** How much OptiX says through its log callback: -Drt.optix.log=1..4, by default errors only (2). */
    fun logLevel(): Int = System.getProperty("rt.optix.log")?.let { it.toIntOrNull() ?: 4 } ?: 2

    /** The address of the function OptiX reports through, for OptixDeviceContextOptions. */
    val logCallback: Long by lazy { LogCallback().address() }

    private fun errorName(code: Int): String =
        function("optixGetErrorName")(code).takeIf { it != 0L }?.let { MemoryUtil.memUTF8(it) } ?: "unknown error"

    private fun function(name: String): NativeFunction = functions.getOrPut(name) {
        val index = TABLE.indexOf(name)
        val signature = SIGNATURES[name]
        require(index >= 0 && signature != null) { "$name is not an OptiX function the renderer knows how to call" }
        NativeFunction(name, MemoryUtil.memGetAddress(table + index * Pointer.POINTER_SIZE.toLong()), signature)
    }

    /** Loads the driver's library and has it fill the function table; returns the table's address. */
    private fun load(): Long {
        val file = libraryFile()
        if (!file.isFile) throw OptixException("the NVIDIA driver's nvoptix.dll was not found (looked for ${file.absolutePath}; -Drt.optix.dll / OPTIX_DLL point at it)")
        val library = try {
            Library.loadNative("rayTraceTypescript", file.absolutePath)
        } catch (t: Throwable) {
            throw OptixException("could not load ${file.absolutePath}: ${t.message}", t)
        }
        val query = library.getFunctionAddress("optixQueryFunctionTable")
        if (query == 0L) throw OptixException("${file.name} has no optixQueryFunctionTable")
        val bytes = TABLE.size * Pointer.POINTER_SIZE.toLong()
        val memory = MemoryUtil.nmemCalloc(1, bytes)
        // int abiId, unsigned numOptions, option keys, option values, the table, its size
        val result = NativeFunction("optixQueryFunctionTable", query, "iiupppl")(ABI_VERSION, 0, null, null, memory, bytes).toInt()
        if (result != 0) {
            MemoryUtil.nmemFree(memory)
            throw OptixException(
                when (result) {
                    7801 -> "the NVIDIA driver does not know OptiX ABI $ABI_VERSION (OptiX 9.0 needs a driver of the 570 series or newer)"
                    7802 -> "the driver's OptiX function table has another size than OptiX 9.0's"
                    else -> "optixQueryFunctionTable failed ($result)"
                }
            )
        }
        return memory
    }

    private fun libraryFile(): File {
        (System.getProperty("rt.optix.dll") ?: System.getenv("OPTIX_DLL"))?.let { return File(it) }
        if (!System.getProperty("os.name", "").lowercase().contains("windows")) {
            throw OptixException("OptiX is only wired up for Windows here")
        }
        // The driver keeps nvoptix.dll in its own folder of the driver store, beside the CUDA
        // library that cuInit loaded from there.
        val cuda = WinBase.GetModuleHandle("nvcuda64.dll")
        if (cuda != 0L) {
            val beside = File(WinBase.GetModuleFileName(cuda, 1024)).resolveSibling("nvoptix.dll")
            if (beside.isFile) return beside
        }
        // Drivers before the driver store kept it in System32.
        return File(System.getenv("SystemRoot") ?: "C:\\Windows", "System32\\nvoptix.dll")
    }

    /** void callback(unsigned int level, const char* tag, const char* message, void* data) */
    private class LogCallback : CallbackI {
        private val cif: FFICIF = FFICIF.calloc().also { cif ->
            val types = MemoryUtil.memAllocPointer(4)
            types.put(0, LibFFI.ffi_type_uint32).put(1, LibFFI.ffi_type_pointer)
                .put(2, LibFFI.ffi_type_pointer).put(3, LibFFI.ffi_type_pointer)
            check(LibFFI.ffi_prep_cif(cif, LibFFI.FFI_DEFAULT_ABI, LibFFI.ffi_type_void, types) == LibFFI.FFI_OK) { "ffi_prep_cif(log callback)" }
        }

        override fun getCallInterface(): FFICIF = cif

        // args is an array of pointers, one to each argument's value.
        override fun callback(ret: Long, args: Long) {
            fun text(index: Int): String =
                MemoryUtil.memGetAddress(MemoryUtil.memGetAddress(args + index * Pointer.POINTER_SIZE.toLong()))
                    .takeIf { it != 0L }?.let { MemoryUtil.memUTF8(it) } ?: ""
            System.err.println("[optix] ${text(1)}: ${text(2).trimEnd()}")
        }
    }

    private const val LOG_BYTES = 16 * 1024
}

open class OptixException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/** A scene holding something the OptiX engine does not trace; the CUDA kernel renders it. */
class OptixUnsupportedScene(message: String) : OptixException(message)
