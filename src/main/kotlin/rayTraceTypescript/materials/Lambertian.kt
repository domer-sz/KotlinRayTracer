package rayTraceTypescript.materials

import rayTraceTypescript.Color
import rayTraceTypescript.Ray
import rayTraceTypescript.Vector
import rayTraceTypescript.objects.Hit
import rayTraceTypescript.pdf.CosinePdf
import rayTraceTypescript.textures.SolidColorTexture
import rayTraceTypescript.textures.Texture
import rayTraceTypescript.utils.pi

/**
 * A matte surface coloured by [texture]. A model file's may also glow: [glow] (its `map_Ke`) times
 * [glowScale] is light it gives off on top of what it reflects - the embers in a fire pit's bowl.
 */
/** [normalMap]: a tangent-space normal map read as data (DirectX green), bending the shading normal on the GPU. */
class Lambertian(val texture: Texture, val glow: Texture? = null, val glowScale: Float = 0f, val normalMap: Texture? = null) : Material {
    constructor(color: Color) : this(SolidColorTexture(color))

    override fun scatter(rayIn: Ray, hit: Hit): ScatterRecord =
        ScatterRecord(texture.value(hit.u, hit.v, hit.point), CosinePdf(hit.normal))

    override fun scatteringPdf(rayIn: Ray, hit: Hit, scattered: Ray): Float {
        val cosine = Vector.dotProduct(hit.normal, Vector.unit(scattered.direction))
        return if (cosine < 0.0f) 0.0f else cosine / pi
    }
}
