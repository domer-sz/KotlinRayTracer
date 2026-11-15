package rayTraceTypescript.objects

import rayTraceTypescript.objects.formats.GltfFormat
import rayTraceTypescript.objects.formats.ObjFormat
import rayTraceTypescript.objects.formats.OffFormat
import rayTraceTypescript.objects.formats.PlyFormat
import rayTraceTypescript.objects.formats.StlFormat
import rayTraceTypescript.objects.formats.ThreeMfFormat
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

/**
 * Reads a 3D model file into a [Mesh], picking the parser by file extension. A model from a file
 * is shaded smooth ([Mesh.smooth]) unless -Drt.smooth=off asks for its flat facets.
 */
object MeshLoader {

    val supportedExtensions: List<String> = listOf("stl", "obj", "ply", "off", "gltf", "glb", "3mf")

    fun load(path: String): Mesh = load(Paths.get(path))

    fun load(path: Path): Mesh = parse(path).smoothed(System.getProperty("rt.smooth") != "off")

    private fun parse(path: Path): Mesh {
        require(Files.isRegularFile(path)) { "Model file not found: $path" }
        val extension = path.fileName.toString().substringAfterLast('.', "").lowercase()
        val bytes = Files.readAllBytes(path)
        return when (extension) {
            "stl" -> StlFormat.parse(bytes)
            "obj" -> ObjFormat.parse(bytes, path.toAbsolutePath().parent)
            "ply" -> PlyFormat.parse(bytes)
            "off" -> OffFormat.parse(bytes)
            "gltf", "glb" -> GltfFormat.parse(bytes, path.toAbsolutePath().parent)
            "3mf" -> ThreeMfFormat.parse(bytes)
            else -> throw IllegalArgumentException(
                "Unsupported model format '.$extension'; this renderer reads ${supportedExtensions.joinToString()}"
            )
        }
    }
}
