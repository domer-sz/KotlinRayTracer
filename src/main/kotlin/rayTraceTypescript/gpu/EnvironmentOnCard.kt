package rayTraceTypescript.gpu

import rayTraceTypescript.textures.Environment

/**
 * A session's copy of the camera's environment map on its card: uploaded when a launch first
 * names a map, again only when the map's pixels change (a brighter or turned copy shares them).
 * Call from the session's own thread, with its context current.
 */
internal class EnvironmentOnCard : AutoCloseable {
    private var pixels: FloatArray? = null
    private var buffer = 0L

    /** The device buffer holding [map]'s pixels, 0 for none. */
    fun bufferFor(map: Environment?): Long {
        if (map?.rgb !== pixels) {
            DeviceMemory.free(buffer)
            buffer = map?.let { DeviceMemory.upload(it.rgb) } ?: 0L
            pixels = map?.rgb
        }
        return buffer
    }

    override fun close() {
        DeviceMemory.free(buffer)
        buffer = 0L
        pixels = null
    }
}
