package rayTraceTypescript.gpu

import rayTraceTypescript.Color
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.DiffuseLight
import rayTraceTypescript.materials.Isotropic
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Material
import rayTraceTypescript.materials.Metal
import rayTraceTypescript.textures.CheckerTexture
import rayTraceTypescript.textures.ImageTexture
import rayTraceTypescript.textures.NoiseTexture
import rayTraceTypescript.textures.Perlin
import rayTraceTypescript.textures.SolidColorTexture
import rayTraceTypescript.textures.Texture
import rayTraceTypescript.textures.UvCheckerTexture
import java.io.ByteArrayOutputStream
import java.util.IdentityHashMap

/**
 * The materials and textures of a scene as the kernel reads them (see [SceneBuffers] for the
 * strides): each is written once, the first time something that wears it is laid out, and
 * referred to by its index from then on.
 */
internal class MaterialTable {
    val materialInts = IntBuffer()
    val materialFloats = FloatBuffer()
    val textureInts = IntBuffer()
    val textureFloats = FloatBuffer()
    val images = ByteArrayOutputStream()
    val perlinVectors = FloatBuffer()
    val perlinPermutations = IntBuffer()

    private val materialIndices = IdentityHashMap<Material, Int>()
    private val textureIndices = IdentityHashMap<Texture, Int>()
    private val colorTextures = IdentityHashMap<Color, Int>()
    private val perlinBlocks = IdentityHashMap<Perlin, Int>()

    /** The index of [material], writing it out first if it is new. */
    fun register(material: Material): Int = materialIndices.getOrPut(material) {
        val index = materialInts.size / SceneBuffers.MATERIAL_INT_STRIDE
        when (material) {
            is Lambertian -> {
                materialInts.add(0)
                materialInts.add(registerTexture(material.texture))
                materialFloats.add(0.0f)
                materialFloats.add(0.0f)
            }
            is Metal -> {
                materialInts.add(1)
                materialInts.add(registerColor(material.albedo))
                materialFloats.add(material.fuzz)
                materialFloats.add(0.0f)
            }
            is DiffuseLight -> {
                materialInts.add(3)
                materialInts.add(registerTexture(material.texture))
                materialFloats.add(0.0f)
                materialFloats.add(0.0f)
            }
            is Isotropic -> {
                materialInts.add(4)
                materialInts.add(registerTexture(material.texture))
                materialFloats.add(0.0f)
                materialFloats.add(0.0f)
            }
            is Dielectric -> {
                materialInts.add(2)
                materialInts.add(0)
                materialFloats.add(0.0f)
                materialFloats.add(material.ri)
            }
            else -> throw IllegalArgumentException("Unsupported material for GPU rendering: ${material::class.java.name}")
        }
        index
    }

    private fun registerTexture(texture: Texture): Int = textureIndices.getOrPut(texture) {
        when (texture) {
            is SolidColorTexture -> solidTexture(texture.albedo)
            is CheckerTexture -> {
                // Children first: their indices are payload of the checker entry.
                val even = registerTexture(texture.even)
                val odd = registerTexture(texture.odd)
                textureEntry(intArrayOf(1, even, odd, 0), floatArrayOf(texture.invertedScale, 0.0f, 0.0f, 0.0f))
            }
            is ImageTexture -> imageTexture(texture)
            is NoiseTexture -> textureEntry(
                intArrayOf(4, registerPerlin(texture.perlin), 0, 0),
                floatArrayOf(texture.scale, 0.0f, 0.0f, 0.0f)
            )
            is UvCheckerTexture -> {
                val even = registerTexture(texture.even)
                val odd = registerTexture(texture.odd)
                textureEntry(intArrayOf(5, even, odd, 0), floatArrayOf(texture.cellsU, texture.cellsV, 0.0f, 0.0f))
            }
            else -> throw IllegalArgumentException("Unsupported texture for GPU rendering: ${texture::class.java.name}")
        }
    }

    /** Copies one Perlin instance's lattice into the shared buffers and returns its block. */
    private fun registerPerlin(perlin: Perlin): Int = perlinBlocks.getOrPut(perlin) {
        val block = perlinVectors.size / SceneBuffers.PERLIN_BLOCK
        perlin.randomVectors.forEach { vector ->
            perlinVectors.add(vector.x)
            perlinVectors.add(vector.y)
            perlinVectors.add(vector.z)
        }
        listOf(perlin.permX, perlin.permY, perlin.permZ).forEach { table ->
            table.forEach { perlinPermutations.add(it) }
        }
        block
    }

    private fun registerColor(color: Color): Int = colorTextures.getOrPut(color) { solidTexture(color) }

    private fun solidTexture(color: Color): Int =
        textureEntry(intArrayOf(0, 0, 0, 0), floatArrayOf(color.r, color.g, color.b, 0.0f))

    private fun imageTexture(texture: ImageTexture): Int {
        val image = texture.image
        val bytes = image.rgbBytes()
        // Matches ImageTexture.value(): an image that failed to load renders as cyan.
        if (bytes == null || image.height() <= 0) return textureEntry(intArrayOf(3, 0, 0, 0), FloatArray(4))
        val offset = images.size()
        images.write(bytes)
        return textureEntry(intArrayOf(2, offset, image.width(), image.height()), FloatArray(4))
    }

    private fun textureEntry(ints: IntArray, floats: FloatArray): Int {
        val index = textureInts.size / SceneBuffers.TEXTURE_INT_STRIDE
        ints.forEach { textureInts.add(it) }
        floats.forEach { textureFloats.add(it) }
        return index
    }
}
