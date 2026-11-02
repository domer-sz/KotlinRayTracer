package rayTraceTypescript.gui

import rayTraceTypescript.Color
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.DiffuseLight
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Material
import rayTraceTypescript.materials.Metal
import rayTraceTypescript.objects.BvhNode
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.objects.RotateY
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.objects.Translate
import rayTraceTypescript.objects.Triangle
import rayTraceTypescript.textures.SolidColorTexture
import kotlin.math.atan2

/**
 * Remakes a scene's own objects in other materials: the same shapes in the same places, each
 * surface's material passed through a function. What it does not know (volumes) stays as it is.
 */
internal object MaterialSwap {

    fun rebuild(h: Hittable, swap: (Material) -> Material): Hittable = when (h) {
        is Quad -> Quad(h.q, h.u, h.v, swap(h.material))
        is Sphere -> Sphere(h.center, h.radius, swap(h.material))
        is Triangle -> Triangle(h.v0, h.v1, h.v2, swap(h.material))
        is Translate -> Translate(rebuild(h.obj, swap), h.offset)
        is RotateY -> RotateY(rebuild(h.obj, swap), Math.toDegrees(atan2(h.sinTheta, h.cosTheta).toDouble()).toFloat())
        is BvhNode -> BvhNode(HittableList(leaves(h).map { rebuild(it, swap) }.toMutableList()))
        is HittableList -> HittableList(h.objects.map { rebuild(it, swap) }.toMutableList())
        is rayTraceTypescript.gpu.PlacedMesh -> rayTraceTypescript.gpu.PlacedMesh(
            h.block, h.scale, h.degrees, h.offset, swap(h.material), h.palette?.map(swap)
        )
        else -> h
    }

    /** The first surface material in [h], depth first. */
    fun firstMaterial(h: Hittable): Material? = when (h) {
        is Quad -> h.material
        is Sphere -> h.material
        is Triangle -> h.material
        is Translate -> firstMaterial(h.obj)
        is RotateY -> firstMaterial(h.obj)
        is BvhNode -> leaves(h).firstNotNullOfOrNull { firstMaterial(it) }
        is HittableList -> h.objects.firstNotNullOfOrNull { firstMaterial(it) }
        is rayTraceTypescript.gpu.PlacedMesh -> h.material
        else -> null
    }

    /** [material] as the editor shows it; null for what it cannot (textures other than a plain colour). */
    fun specOf(material: Material): MaterialSpec? = when (material) {
        is Lambertian -> (material.texture as? SolidColorTexture)?.albedo?.let { MaterialSpec(MaterialKind.MATTE, it.r, it.g, it.b) }
        is Metal -> MaterialSpec(MaterialKind.METAL, material.albedo.r, material.albedo.g, material.albedo.b, material.fuzz.toDouble())
        is Dielectric -> MaterialSpec(MaterialKind.GLASS, 1f, 1f, 1f, material.ri.toDouble())
        is DiffuseLight -> (material.texture as? SolidColorTexture)?.albedo?.let { c ->
            // Stored as a colour of at most one and a strength, as the editor sets a light.
            val strength = maxOf(c.r, c.g, c.b, 1e-6f)
            MaterialSpec(MaterialKind.LIGHT, c.r / strength, c.g / strength, c.b / strength, strength.toDouble())
        }
        else -> null
    }

    /** A light [factor] times as strong; other materials unchanged. */
    fun brighter(material: Material, factor: Double): Material {
        val light = material as? DiffuseLight ?: return material
        val c = (light.texture as? SolidColorTexture)?.albedo ?: return material
        val k = factor.toFloat()
        return DiffuseLight(Color(c.r * k, c.g * k, c.b * k))
    }

    /** The objects a BVH node holds, each once. */
    private fun leaves(node: BvhNode): List<Hittable> {
        val out = ArrayList<Hittable>()
        fun walk(h: Hittable) {
            if (h is BvhNode) {
                walk(h.left)
                if (h.right !== h.left) walk(h.right)
            } else out += h
        }
        walk(node)
        return out
    }
}
