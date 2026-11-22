package rayTraceTypescript.gpu

import rayTraceTypescript.Color
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.DiffuseLight
import rayTraceTypescript.materials.Isotropic
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Material
import rayTraceTypescript.materials.Metal
import rayTraceTypescript.objects.Lens
import rayTraceTypescript.optics.GlassCatalog
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
    /** Lens records and catalog glasses, by offset ([SceneBuffers.optics]). */
    val optics = FloatBuffer()

    private val materialIndices = IdentityHashMap<Material, Int>()
    private val textureIndices = IdentityHashMap<Texture, Int>()
    private val colorTextures = IdentityHashMap<Color, Int>()
    private val perlinBlocks = IdentityHashMap<Perlin, Int>()
    private val glassOffsets = IdentityHashMap<GlassCatalog.Glass, Int>()

    /** The index of [material], writing it out first if it is new. */
    fun register(material: Material): Int = materialIndices.getOrPut(material) {
        val index = materialInts.size / SceneBuffers.MATERIAL_INT_STRIDE
        // Four ints a material - its kind, its colour's texture, a texture of light it gives off on
        // top (-1 for none) and one of its roughness (-1 for none) - and four floats: the metal's
        // fuzz, glass's refractive index or the metal's Fresnel switch, how bright that light is, unused.
        // The fourth float is a normal map's texture plus one, 0 for none.
        fun entry(
            kind: Int, texture: Int, first: Float = 0f, second: Float = 0f,
            glow: Texture? = null, glowScale: Float = 0f, roughness: Texture? = null, normalMap: Texture? = null,
            glass: GlassCatalog.Glass? = null
        ) {
            materialInts.add(kind); materialInts.add(texture)
            // The fourth int: a metal's roughness texture, or a catalog glass's offset in [optics].
            materialInts.add(glow?.let { registerTexture(it) } ?: -1)
            materialInts.add(roughness?.let { registerTexture(it) } ?: glass?.let { registerGlass(it) } ?: -1)
            materialFloats.add(first); materialFloats.add(second); materialFloats.add(glowScale)
            materialFloats.add(normalMap?.let { registerTexture(it) + 1f } ?: 0f)
        }
        when (material) {
            is Lambertian -> entry(0, registerTexture(material.texture), glow = material.glow, glowScale = material.glowScale, normalMap = material.normalMap)
            // The second float, unused by the book's metal, switches its Fresnel term on.
            is Metal -> entry(
                1, material.texture?.let { registerTexture(it) } ?: registerColor(material.albedo),
                material.fuzz, if (material.fresnel) 1.0f else 0.0f, material.glow, material.glowScale, material.roughness, material.normalMap
            )
            is DiffuseLight -> entry(3, registerTexture(material.texture))
            is Isotropic -> entry(4, registerTexture(material.texture))
            // Glass: its tint's colour and, in the first float, the distance through it that leaves
            // that colour (0 for clear glass, whose texture slot is then unused).
            // A catalog glass also gets its dispersion ([registerGlass]).
            is Dielectric -> material.tint?.let { entry(2, registerColor(it), material.tintDistance, material.ri, glass = material.glass) }
                ?: entry(2, 0, 0f, material.ri, glass = material.glass)
            else -> throw IllegalArgumentException("Unsupported material for GPU rendering: ${material::class.java.name}")
        }
        index
    }

    /** Where a catalog glass's formula sits in [optics] ([GlassCatalog.Glass.packed]). */
    private fun registerGlass(glass: GlassCatalog.Glass): Int = glassOffsets.getOrPut(glass) {
        val offset = optics.size
        glass.packed().forEach { optics.add(it) }
        offset
    }

    /**
     * Writes [lens], moved by [transform], as the kernel's lens_t reads it and returns its offset in
     * [optics]: vertex (3), axis (3), semi-diameter, surface count, each surface's vertex along the
     * axis and curvature (2 x 3), their rims along the axis (3), each element's material (2), which
     * surfaces have a polynomial (bits), then each surface's conic constant and 8 terms (9 x 3).
     */
    fun addLens(lens: Lens, transform: Transform): Int {
        val vertex = transform.point(lens.vertex)
        val axis = transform.vector(lens.axis)
        val elements = (1 until lens.media.size - 1).map { register(lens.media[it]!!) }
        val offset = optics.size
        val record = FloatArray(SceneBuffers.LENS_STRIDE)
        record[0] = vertex.x; record[1] = vertex.y; record[2] = vertex.z
        record[3] = axis.x; record[4] = axis.y; record[5] = axis.z
        record[6] = lens.semiDiameter
        record[7] = lens.surfaceCount.toFloat()
        for (k in 0 until lens.surfaceCount) {
            record[8 + 2 * k] = lens.vertices[k]
            record[9 + 2 * k] = lens.curvatures[k]
            record[14 + k] = lens.rims[k]
        }
        elements.forEachIndexed { j, material -> record[17 + j] = material.toFloat() }
        // Which surfaces have a polynomial (bits), then each surface's conic constant and terms.
        record[19] = (0 until lens.surfaceCount).filter { lens.polynomial(it) }.sumOf { 1 shl it }.toFloat()
        for (k in 0 until lens.surfaceCount) {
            record[20 + 9 * k] = lens.conics[k]
            for (i in 0 until Lens.ASPHERIC_TERMS) record[21 + 9 * k + i] = lens.aspherics[Lens.ASPHERIC_TERMS * k + i]
        }
        record.forEach { optics.add(it) }
        return offset
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
