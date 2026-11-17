package rayTraceTypescript.objects.formats

import rayTraceTypescript.Color
import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.objects.MeshMaterial
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

/**
 * Wavefront OBJ: `v` vertices, `vt` texture coordinates and `f` faces, with the normal part of a
 * face corner ignored. Faces may be polygons and may index from the end with negative numbers.
 *
 * When the file names a material library (`mtllib`) found beside it, `usemtl` gives the faces
 * after it that material; see [MtlFormat] for what of a material is read.
 */
internal object ObjFormat {

    fun parse(bytes: ByteArray, folder: Path? = null): Mesh {
        val positions = FloatList()
        val texCoords = FloatList()
        val builder = MeshBuilder()
        val library = LinkedHashMap<String, MeshMaterial>()
        val palette = ArrayList<MeshMaterial>()
        val whitespace = Regex("\\s+")

        bytes.toString(StandardCharsets.UTF_8).lineSequence().forEach { rawLine ->
            val line = rawLine.substringBefore('#').trim()
            if (line.isEmpty()) return@forEach
            val parts = line.split(whitespace)
            when (parts[0]) {
                "v" -> {
                    require(parts.size >= 4) { "Vertex needs three coordinates: '$line'" }
                    positions.add(parts[1].toFloat(), parts[2].toFloat(), parts[3].toFloat())
                }
                "vt" -> {
                    require(parts.size >= 2) { "Texture coordinate needs u: '$line'" }
                    texCoords.add(parts[1].toFloat())
                    texCoords.add(parts.getOrNull(2)?.toFloatOrNull() ?: 0.0f)
                }
                "f" -> {
                    val corners = parts.drop(1)
                    if (corners.size >= 3) {
                        val uvIndices = if (texCoords.size == 0) null else IntArray(corners.size) { textureIndex(corners[it], texCoords.size / 2) }
                        builder.polygon(positions, IntArray(corners.size) { resolve(corners[it], positions.tripleCount) }, texCoords, uvIndices)
                    }
                }
                "mtllib" -> if (folder != null) {
                    val file = folder.resolve(line.substringAfter("mtllib").trim())
                    if (Files.isRegularFile(file)) MtlFormat.parse(Files.readAllBytes(file), file.parent).forEach { library[it.name] = it }
                }
                "usemtl" -> {
                    val material = library[line.substringAfter("usemtl").trim()]
                    builder.material = if (material == null) -1 else {
                        val at = palette.indexOf(material)
                        if (at >= 0) at else { palette += material; palette.size - 1 }
                    }
                }
            }
        }

        require(builder.triangleCount > 0) { "No faces found; is this an OBJ file?" }
        return builder.build(palette)
    }

    /** `f 1/2/3` - only the vertex index matters; negative indices count back from the end. */
    private fun resolve(corner: String, vertexCount: Int): Int {
        val index = corner.substringBefore('/').toIntOrNull()
            ?: throw IllegalArgumentException("Malformed face corner '$corner'")
        return if (index > 0) index - 1 else vertexCount + index
    }

    /** `f 1/2/3` - the texture coordinate's index, -1 when the corner names none. */
    private fun textureIndex(corner: String, count: Int): Int {
        val index = corner.split('/').getOrNull(1)?.toIntOrNull() ?: return -1
        return if (index > 0) index - 1 else count + index
    }
}

/**
 * A Wavefront material library (.mtl), read for what this renderer can draw: `Kd` the colour
 * (linear), the PBR extension's `Pm` metallic and `Pr` roughness, `Ke` light given off,
 * `d` (or `Tr`) with `Ni` for glass, and `map_Kd`, an image for the colour, found beside the
 * library ([folder]).
 */
internal object MtlFormat {

    fun parse(bytes: ByteArray, folder: Path? = null): List<MeshMaterial> {
        val materials = ArrayList<MeshMaterial>()
        var name: String? = null
        var kd = Color(0.8f, 0.8f, 0.8f)
        var pm = 0.0f
        var pr = 0.0f
        var ke: Color? = null
        var d = 1.0f
        var ni = 1.5f
        var mapKd: Path? = null
        var mapPr: Path? = null
        var mapKe: Path? = null
        var norm: Path? = null
        var tf: Color? = null
        var tfDistance = 1.0f
        fun finish() {
            name?.let { materials += MeshMaterial(it, kd, pm, pr, ke, d, ni, mapKd, mapPr, mapKe, norm, tf, tfDistance) }
        }
        val whitespace = Regex("\\s+")
        bytes.toString(StandardCharsets.UTF_8).lineSequence().forEach { rawLine ->
            val line = rawLine.substringBefore('#').trim()
            if (line.isEmpty()) return@forEach
            val p = line.split(whitespace)
            fun f(i: Int) = p.getOrNull(i)?.toFloatOrNull() ?: 0.0f
            fun colour() = Color(f(1), if (p.size > 2) f(2) else f(1), if (p.size > 3) f(3) else f(1))
            when (p[0]) {
                "newmtl" -> {
                    finish()
                    name = line.substringAfter("newmtl").trim()
                    kd = Color(0.8f, 0.8f, 0.8f); pm = 0.0f; pr = 0.0f; ke = null; d = 1.0f; ni = 1.5f; mapKd = null; mapPr = null; mapKe = null; norm = null; tf = null; tfDistance = 1.0f
                }
                "Kd" -> kd = colour()
                "Pm" -> pm = f(1)
                "Pr" -> pr = f(1)
                "Ke" -> ke = colour()
                "d" -> d = f(1)
                "Tr" -> d = 1.0f - f(1)
                "Ni" -> ni = f(1)
                // Glass coloured through its body: what is left after a distance through it - one
                // model unit, or the fourth number (not in the format, which has no distance).
                "Tf" -> if (p.size >= 4) { tf = colour(); tfDistance = p.getOrNull(4)?.toFloatOrNull()?.takeIf { it > 0f } ?: 1.0f }
                // Options (-bm 1, -s ...) come before the file name, which is last.
                "map_Kd" -> mapKd = folder?.resolve(p.last())?.takeIf { Files.isRegularFile(it) }
                "map_Pr" -> mapPr = folder?.resolve(p.last())?.takeIf { Files.isRegularFile(it) }
                "map_Ke" -> mapKe = folder?.resolve(p.last())?.takeIf { Files.isRegularFile(it) }
                "norm", "map_Bump", "bump" -> folder?.resolve(p.last())?.takeIf { Files.isRegularFile(it) }?.let { norm = it }
            }
        }
        finish()
        return materials
    }
}
