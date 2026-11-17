package rayTraceTypescript.objects

import rayTraceTypescript.Color
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.DiffuseLight
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Material
import rayTraceTypescript.materials.Metal
import rayTraceTypescript.textures.ImageTexture
import rayTraceTypescript.textures.RtwImage
import rayTraceTypescript.textures.SolidColorTexture

/**
 * A material a model file names for some of its faces - an entry of an OBJ's .mtl - reduced
 * to what this renderer can draw: a [colour] (linear, as `Kd` holds it), how [metallic] and how
 * rough the surface is, light it gives off, and glass.
 */
class MeshMaterial(
    val name: String,
    val colour: Color,
    val metallic: Float = 0.0f,
    val roughness: Float = 0.0f,
    val emission: Color? = null,
    /** Below one for glass (the .mtl's `d`), with [refractiveIndex] as its `Ni`. */
    val opacity: Float = 1.0f,
    val refractiveIndex: Float = 1.5f,
    /** An image for the colour (the .mtl's `map_Kd`), over the mesh's texture coordinates; it takes [colour]'s place on a matte surface. */
    val texture: java.nio.file.Path? = null,
    /** Roughness a texel at a time (`map_Pr`, its red channel, read as data), in place of [roughness]. */
    val roughnessMap: java.nio.file.Path? = null,
    /** Where it glows (`map_Ke`), times [emission] - or times one without - on top of what it reflects. */
    val glowMap: java.nio.file.Path? = null,
    /** A tangent-space normal map (`norm` or `map_Bump`), DirectX-style, read as data. */
    val normalMap: java.nio.file.Path? = null,
    /** Glass coloured through its body (`Tf`): what is left of the light after [tintDistance] through it. */
    val tint: Color? = null,
    val tintDistance: Float = 1.0f
) {
    fun material(): Material = when {
        glowMap == null && emission != null && (emission.r > 0f || emission.g > 0f || emission.b > 0f) -> DiffuseLight(emission)
        opacity < 0.5f -> Dielectric(refractiveIndex.coerceAtLeast(1.0f), tint?.takeIf { it.r < 1f || it.g < 1f || it.b < 1f }, tintDistance)
        // A file's metal is physically based: its image colours it texel by texel (its average on the
        // CPU, as exporters leave Kd out then), its roughness blurs as GGX's alpha, roughness
        // squared, does - the book's fuzz taken straight would make a 0.5-rough metal near matte -
        // and Fresnel brightens its edges.
        metallic >= 0.5f -> {
            val image = texture?.let { imageOf(it) }
            Metal(
                image?.image?.averageColour() ?: colour, (roughness * roughness).coerceIn(0.0f, 1.0f), image, fresnel = true,
                glow = glow, glowScale = glowScale, roughness = roughnessMap?.let { imageOf(it, data = true) }, normalMap = normals
            )
        }
        texture != null -> imageOf(texture)?.let { Lambertian(it, glow, glowScale, normals) } ?: Lambertian(SolidColorTexture(colour), glow, glowScale, normals)
        else -> Lambertian(SolidColorTexture(colour), glow, glowScale, normals)
    }

    private val glow get() = glowMap?.let { imageOf(it) }
    private val normals get() = normalMap?.let { imageOf(it, data = true) }
    private val glowScale: Float get() = emission?.let { maxOf(it.r, it.g, it.b) }?.takeIf { it > 0f } ?: 1f

    companion object {
        /**
         * One texture per image file, however many materials and placements name it: the GPU
         * tables then hold its pixels once. Null for a file that would not load.
         */
        private val images = java.util.concurrent.ConcurrentHashMap<Pair<java.nio.file.Path, Boolean>, java.util.Optional<ImageTexture>>()

        private fun imageOf(path: java.nio.file.Path, data: Boolean = false): ImageTexture? = images.computeIfAbsent(path.toAbsolutePath().normalize() to data) { (file, raw) ->
            val image = RtwImage()
            java.util.Optional.ofNullable(if (image.load(file.toString(), raw)) ImageTexture(image) else null)
        }.orElse(null)
    }
}
