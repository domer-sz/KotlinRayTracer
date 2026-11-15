package rayTraceTypescript.utils

import org.lwjgl.PointerBuffer
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil
import org.lwjgl.system.libffi.FFICIF
import org.lwjgl.system.libffi.FFIType
import org.lwjgl.system.libffi.LibFFI

/**
 * One C function at an address, called through LWJGL's libffi - for libraries LWJGL has no
 * binding for.
 *
 * [signature] names the types, the return type first: `i` int, `u` unsigned int, `l` a 64-bit
 * integer (size_t, a device pointer), `p` a pointer, `f` float, `b` a byte-sized bool, and `v`
 * for a function that returns nothing. So `"ippl"` is `int f(void*, void*, size_t)`.
 */
class NativeFunction(val name: String, private val address: Long, signature: String) {
    private val returnKind = signature[0]
    private val kinds = signature.substring(1)
    private val types: PointerBuffer? = if (kinds.isEmpty()) null else MemoryUtil.memAllocPointer(kinds.length).also { buffer ->
        kinds.forEachIndexed { i, kind -> buffer.put(i, type(kind)) }
    }
    private val cif = FFICIF.calloc().also {
        require(address != 0L) { "$name has no address" }
        check(LibFFI.ffi_prep_cif(it, LibFFI.FFI_DEFAULT_ABI, type(returnKind), types) == LibFFI.FFI_OK) { "ffi_prep_cif($name)" }
    }

    /**
     * Calls the function. Arguments follow the signature: an Int or Long for the integers and
     * pointers (null is a null pointer), a Float, a Boolean, or a String, which is passed as
     * UTF-8. Returns the result widened to a Long, 0 for a void function.
     */
    operator fun invoke(vararg args: Any?): Long {
        require(args.size == kinds.length) { "$name takes ${kinds.length} arguments, got ${args.size}" }
        MemoryStack.stackPush().use { stack ->
            val values = stack.mallocPointer(maxOf(args.size, 1))
            args.forEachIndexed { i, arg ->
                val cell = stack.malloc(8, 8)
                when (val kind = kinds[i]) {
                    'i', 'u' -> cell.putInt(0, (arg as Number).toInt())
                    'l' -> cell.putLong(0, (arg as Number).toLong())
                    'p' -> cell.putLong(0, when (arg) {
                        null -> 0L
                        is String -> MemoryUtil.memAddress(stack.UTF8(arg))
                        else -> (arg as Number).toLong()
                    })
                    'f' -> cell.putFloat(0, (arg as Number).toFloat())
                    'b' -> cell.put(0, if (arg as Boolean) 1 else 0)
                    else -> throw IllegalArgumentException("$name: no argument of kind '$kind'")
                }
                values.put(i, MemoryUtil.memAddress(cell))
            }
            // libffi widens small results to a whole register.
            val result = stack.calloc(8, 8)
            LibFFI.ffi_call(cif, address, result, values)
            return when (returnKind) {
                'v' -> 0L
                'i' -> result.getInt(0).toLong()
                'u' -> result.getInt(0).toLong() and 0xFFFFFFFFL
                else -> result.getLong(0)
            }
        }
    }

    private companion object {
        fun type(kind: Char): FFIType = when (kind) {
            'v' -> LibFFI.ffi_type_void
            'i' -> LibFFI.ffi_type_sint32
            'u' -> LibFFI.ffi_type_uint32
            'l' -> LibFFI.ffi_type_uint64
            'p' -> LibFFI.ffi_type_pointer
            'f' -> LibFFI.ffi_type_float
            'b' -> LibFFI.ffi_type_uint8
            else -> throw IllegalArgumentException("No C type for '$kind'")
        }
    }
}
