package rayTraceTypescript.materials

import rayTraceTypescript.Color
import rayTraceTypescript.Ray
import rayTraceTypescript.Vector
import rayTraceTypescript.objects.Hit

/**
 * A mirror blurred by [fuzz], tinted [albedo] - or, on the GPU, by [texture] when it has one, a
 * texel at a time. With [fresnel] its reflectance rises towards white at grazing angles (Schlick),
 * as a physically based metal's does: what a model file's metals get; the book's keep a flat tint.
 */
class Metal(
    val albedo: Color,
    val fuzz: Float,
    val texture: rayTraceTypescript.textures.Texture? = null,
    val fresnel: Boolean = false,
    /** Light given off on top of the reflection, times [glowScale], as [Lambertian.glow]. */
    val glow: rayTraceTypescript.textures.Texture? = null,
    val glowScale: Float = 0f,
    /**
     * Roughness a texel at a time (its red channel, 0..1, read as data): the fuzz is then its
     * square, as for [fuzz] from a file, instead of [fuzz]. On the GPU only.
     */
    val roughness: rayTraceTypescript.textures.Texture? = null,
    /** A tangent-space normal map, as [Lambertian.normalMap]. */
    val normalMap: rayTraceTypescript.textures.Texture? = null
) : Material {
    constructor(albedo: Color, fuzz: Double) : this(albedo, fuzz.toFloat())

    override fun scatter(rayIn: Ray, hit: Hit): ScatterRecord {
        // Reflect, then blur the reflection by nudging it towards a random direction.
        val reflected = Vector.unit(Vector.reflect(rayIn.direction, hit.normal)) +
            Vector.randomUnitVector() * fuzz
        return ScatterRecord(albedo, skipPdfRay = Ray(hit.point, reflected, rayIn.time))
    }
}
