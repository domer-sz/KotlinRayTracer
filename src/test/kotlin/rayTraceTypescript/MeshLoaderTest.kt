package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.objects.MeshLoader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.util.Base64
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * One tetrahedron, written out in every supported format: whatever the container, the loader
 * has to come back with the same four triangles in the same place.
 */
class MeshLoaderTest {

    private val vertices = arrayOf(
        floatArrayOf(0f, 0f, 0f),
        floatArrayOf(1f, 0f, 0f),
        floatArrayOf(0f, 2f, 0f),
        floatArrayOf(0f, 0f, 3f)
    )
    private val faces = arrayOf(
        intArrayOf(0, 1, 2), intArrayOf(0, 1, 3), intArrayOf(0, 2, 3), intArrayOf(1, 2, 3)
    )

    @Test
    fun `every supported format loads the same mesh`() {
        val files = mapOf(
            "stl (binary)" to binaryStl(),
            "stl (ascii)" to asciiStl(),
            "obj" to obj(),
            "ply (ascii)" to plyAscii(),
            "ply (binary little endian)" to plyBinary(ByteOrder.LITTLE_ENDIAN),
            "ply (binary big endian)" to plyBinary(ByteOrder.BIG_ENDIAN),
            "off" to off(),
            "glb" to glb(),
            "gltf" to gltf(),
            "3mf" to threeMf()
        )

        files.forEach { (label, file) ->
            val mesh = MeshLoader.load(file)
            assertEquals(4, mesh.triangleCount, "$label: triangle count")
            val box = mesh.bounds()
            assertEquals(0f, box.x.min, 1e-4f, "$label: min x")
            assertEquals(1f, box.x.max, 1e-4f, "$label: max x")
            assertEquals(2f, box.y.max, 1e-4f, "$label: max y")
            assertEquals(3f, box.z.max, 1e-4f, "$label: max z")
        }
    }

    @Test
    fun `obj triangulates polygons and resolves negative indices`() {
        val file = write(
            "quad.obj",
            """
            v 0 0 0
            v 1 0 0
            v 1 1 0
            v 0 1 0
            f 1 2 3 4
            f -4 -3 -2
            """.trimIndent()
        )
        // The quad becomes two triangles, the relative face one more.
        assertEquals(3, MeshLoader.load(file).triangleCount)
    }

    @Test
    fun `obj reads the colours its material library gives the faces`() {
        val obj = write(
            "painted.obj",
            """
            mtllib painted.mtl
            v 0 0 0
            v 1 0 0
            v 1 1 0
            v 0 1 0
            f 1 2 3
            usemtl blade
            f 1 3 4
            usemtl robe
            f 2 3 4
            usemtl unknown
            f 1 2 4
            """.trimIndent()
        )
        Files.writeString(
            obj.resolveSibling("painted.mtl"),
            """
            newmtl robe
            Kd 0.1 0.2 0.6
            newmtl blade
            Kd 0.8 0.8 0.8
            Pm 1.0
            Pr 0.1
            """.trimIndent()
        )
        val mesh = MeshLoader.load(obj)
        assertTrue(mesh.hasColours)
        // In the order the faces first use them; faces before any material, or naming one the
        // library lacks, take the first entry.
        assertEquals(listOf("blade", "robe"), mesh.palette.map { it.name })
        assertEquals(listOf(0, 0, 1, 0), mesh.faceMaterials!!.toList())
        assertTrue(mesh.palette[0].material() is rayTraceTypescript.materials.Metal)
        assertTrue(mesh.palette[1].material() is rayTraceTypescript.materials.Lambertian)
        assertEquals(0.6f, mesh.palette[1].colour.b, 1e-6f)
        // Moving the mesh keeps its colours.
        assertEquals(mesh.faceMaterials!!.toList(), mesh.zUpToYUp().rotatedY(30f).standingOnFloor().faceMaterials!!.toList())
    }

    @Test
    fun `obj reads texture coordinates and the colour image its library names`() {
        val obj = write(
            "textured.obj",
            """
            mtllib textured.mtl
            v 0 0 0
            v 1 0 0
            v 1 1 0
            v 0 1 0
            vt 0 0
            vt 2 0
            vt 2 1
            vt 0 1
            usemtl wall
            f 1/1 2/2 3/3 4/4
            f 1 3 4
            """.trimIndent()
        )
        Files.writeString(obj.resolveSibling("textured.mtl"), "newmtl wall\nKd 0.5 0.5 0.5\nmap_Kd -bm 1 wall.png\n")
        val image = java.awt.image.BufferedImage(4, 2, java.awt.image.BufferedImage.TYPE_INT_RGB)
        javax.imageio.ImageIO.write(image, "png", obj.resolveSibling("wall.png").toFile())

        val mesh = MeshLoader.load(obj)
        val uvs = mesh.uvs!!
        // The quad fans into two triangles, each corner with its own coordinates.
        assertEquals(listOf(0f, 0f, 2f, 0f, 2f, 1f), uvs.copyOfRange(0, 6).toList())
        assertEquals(listOf(0f, 0f, 2f, 1f, 0f, 1f), uvs.copyOfRange(6, 12).toList())
        // A face whose corners name none has none.
        assertTrue(uvs[12].isNaN())
        assertEquals(obj.resolveSibling("wall.png"), mesh.palette[0].texture)
        val material = mesh.palette[0].material() as rayTraceTypescript.materials.Lambertian
        assertTrue(material.texture is rayTraceTypescript.textures.ImageTexture)
        // Moving the mesh keeps its coordinates.
        assertEquals(uvs.toList(), mesh.zUpToYUp().rotatedY(30f).standingOnFloor().uvs!!.toList())
    }

    @Test
    fun `tinted glass keeps its colour by the distance through it`() {
        val obj = write("amber.obj", "mtllib amber.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl amber\nf 1 2 3\n")
        Files.writeString(obj.resolveSibling("amber.mtl"), "newmtl amber\nd 0\nNi 1.52\nTf 0.9 0.5 0.1 0.2\n")
        val glass = MeshLoader.load(obj).palette[0].material() as rayTraceTypescript.materials.Dielectric
        assertEquals(1.52f, glass.ri)
        assertEquals(0.2f, glass.tintDistance)
        // 0.2 through it leaves the tint itself, 0.4 its square, nothing through it all the light.
        val once = glass.transmitted(0.2f)
        assertEquals(0.5f, once.g, 1e-5f)
        val twice = glass.transmitted(0.4f)
        assertEquals(0.81f, twice.r, 1e-5f)
        assertEquals(0.01f, twice.b, 1e-5f)
        assertEquals(1.0f, glass.transmitted(0f).b, 1e-6f)
        // Clear glass stays clear.
        Files.writeString(obj.resolveSibling("amber.mtl"), "newmtl amber\nd 0\nTf 1 1 1\n")
        assertEquals(null, (MeshLoader.load(obj).palette[0].material() as rayTraceTypescript.materials.Dielectric).tint)
    }

    @Test
    fun `an obj without a material library has no colours`() {
        val obj = write("plain.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl robe\nf 1 2 3\n")
        assertTrue(!MeshLoader.load(obj).hasColours)
    }

    @Test
    fun `gltf bakes node transforms into the geometry`() {
        val moved = MeshLoader.load(glb(translation = floatArrayOf(10f, 0f, 0f)))
        assertEquals(10f, moved.bounds().x.min, 1e-4f, "the node translation moved the mesh")
        assertEquals(11f, moved.bounds().x.max, 1e-4f)
    }

    @Test
    fun `unknown extensions name the formats that work`() {
        val file = write("model.fbx", "not really a model")
        val failure = assertThrows<IllegalArgumentException> { MeshLoader.load(file) }
        assertTrue(failure.message!!.contains("stl"), failure.message)
        assertTrue(failure.message!!.contains("gltf"), failure.message)
    }

    // ---------------------------------------------------------------- file writers

    private fun write(name: String, text: String): Path = write(name, text.toByteArray(StandardCharsets.UTF_8))

    private fun write(name: String, bytes: ByteArray): Path {
        val file = Files.createTempDirectory("meshes").resolve(name)
        Files.write(file, bytes)
        return file
    }

    private fun binaryStl(): Path {
        val buffer = ByteBuffer.allocate(84 + faces.size * 50).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put(ByteArray(80))
        buffer.putInt(faces.size)
        faces.forEach { face ->
            repeat(3) { buffer.putFloat(0f) }
            face.forEach { index -> vertices[index].forEach { buffer.putFloat(it) } }
            buffer.putShort(0)
        }
        return write("model.stl", buffer.array())
    }

    private fun asciiStl(): Path = write("ascii.stl", buildString {
        append("solid test\n")
        faces.forEach { face ->
            append("  facet normal 0 0 0\n    outer loop\n")
            face.forEach { index ->
                val v = vertices[index]
                append("      vertex ${v[0]} ${v[1]} ${v[2]}\n")
            }
            append("    endloop\n  endfacet\n")
        }
        append("endsolid test\n")
    })

    private fun obj(): Path = write("model.obj", buildString {
        append("# tetrahedron\n")
        vertices.forEach { append("v ${it[0]} ${it[1]} ${it[2]}\n") }
        faces.forEach { append("f ${it[0] + 1}//1 ${it[1] + 1}//1 ${it[2] + 1}//1\n") }
    })

    private fun plyHeader(format: String) = buildString {
        append("ply\nformat $format 1.0\n")
        append("element vertex ${vertices.size}\n")
        append("property float x\nproperty float y\nproperty float z\n")
        append("element face ${faces.size}\n")
        append("property list uchar int vertex_indices\n")
        append("end_header\n")
    }

    private fun plyAscii(): Path = write("model.ply", buildString {
        append(plyHeader("ascii"))
        vertices.forEach { append("${it[0]} ${it[1]} ${it[2]}\n") }
        faces.forEach { append("3 ${it[0]} ${it[1]} ${it[2]}\n") }
    })

    private fun plyBinary(order: ByteOrder): Path {
        val header = plyHeader(
            if (order == ByteOrder.LITTLE_ENDIAN) "binary_little_endian" else "binary_big_endian"
        ).toByteArray(StandardCharsets.US_ASCII)
        val body = ByteBuffer.allocate(vertices.size * 12 + faces.size * 13).order(order)
        vertices.forEach { v -> v.forEach { body.putFloat(it) } }
        faces.forEach { face ->
            body.put(3)
            face.forEach { body.putInt(it) }
        }
        return write("binary-${order.toString().lowercase().replace("_", "-")}.ply", header + body.array())
    }

    private fun off(): Path = write("model.off", buildString {
        append("OFF\n# a tetrahedron\n${vertices.size} ${faces.size} 0\n")
        vertices.forEach { append("${it[0]} ${it[1]} ${it[2]}\n") }
        faces.forEach { append("3 ${it[0]} ${it[1]} ${it[2]}\n") }
    })

    /** Positions as float VEC3 followed by ushort indices, the layout most exporters emit. */
    private fun gltfBuffer(): ByteArray {
        val buffer = ByteBuffer.allocate(vertices.size * 12 + faces.size * 6).order(ByteOrder.LITTLE_ENDIAN)
        vertices.forEach { v -> v.forEach { buffer.putFloat(it) } }
        faces.forEach { face -> face.forEach { buffer.putShort(it.toShort()) } }
        return buffer.array()
    }

    private fun gltfJson(bufferLength: Int, uri: String?, translation: FloatArray?): String {
        val node = if (translation == null) """{"mesh":0}"""
        else """{"mesh":0,"translation":[${translation[0]},${translation[1]},${translation[2]}]}"""
        val buffer = if (uri == null) """{"byteLength":$bufferLength}"""
        else """{"byteLength":$bufferLength,"uri":"$uri"}"""
        return """
            {"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
             "nodes":[$node],
             "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
             "accessors":[
               {"bufferView":0,"componentType":5126,"count":${vertices.size},"type":"VEC3"},
               {"bufferView":1,"componentType":5123,"count":${faces.size * 3},"type":"SCALAR"}],
             "bufferViews":[
               {"buffer":0,"byteOffset":0,"byteLength":${vertices.size * 12}},
               {"buffer":0,"byteOffset":${vertices.size * 12},"byteLength":${faces.size * 6}}],
             "buffers":[$buffer]}
        """.trimIndent()
    }

    private fun glb(translation: FloatArray? = null): Path {
        val binary = gltfBuffer()
        val json = gltfJson(binary.size, uri = null, translation = translation)
            .toByteArray(StandardCharsets.UTF_8)
            .let { it + ByteArray((4 - it.size % 4) % 4) { ' '.code.toByte() } }
        val padded = binary + ByteArray((4 - binary.size % 4) % 4)

        val glb = ByteBuffer.allocate(12 + 8 + json.size + 8 + padded.size).order(ByteOrder.LITTLE_ENDIAN)
        glb.putInt(0x46546C67)                       // "glTF"
        glb.putInt(2)
        glb.putInt(glb.capacity())
        glb.putInt(json.size); glb.putInt(0x4E4F534A); glb.put(json)
        glb.putInt(padded.size); glb.putInt(0x004E4942); glb.put(padded)
        return write("model.glb", glb.array())
    }

    private fun gltf(): Path {
        val uri = "data:application/octet-stream;base64," + Base64.getEncoder().encodeToString(gltfBuffer())
        return write("model.gltf", gltfJson(gltfBuffer().size, uri, translation = null))
    }

    private fun threeMf(): Path {
        val model = buildString {
            append("""<?xml version="1.0" encoding="UTF-8"?>""")
            append("""<model unit="millimeter"><resources><object id="1" type="model"><mesh><vertices>""")
            vertices.forEach { append("""<vertex x="${it[0]}" y="${it[1]}" z="${it[2]}"/>""") }
            append("</vertices><triangles>")
            faces.forEach { append("""<triangle v1="${it[0]}" v2="${it[1]}" v3="${it[2]}"/>""") }
            append("</triangles></mesh></object></resources><build><item objectid=\"1\"/></build></model>")
        }
        val bytes = java.io.ByteArrayOutputStream()
        ZipOutputStream(bytes).use { zip ->
            zip.putNextEntry(ZipEntry("3D/3dmodel.model"))
            zip.write(model.toByteArray(StandardCharsets.UTF_8))
            zip.closeEntry()
        }
        return write("model.3mf", bytes.toByteArray())
    }
}
