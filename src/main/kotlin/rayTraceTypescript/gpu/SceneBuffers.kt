package rayTraceTypescript.gpu

/**
 * The scene, flattened into primitive arrays the CUDA kernel can read.
 *
 * Layout (see raytracer.cu, which mirrors these strides):
 *  - [nodes]: 16 ints per inner node, floats as raw bits - both children's boxes and codes,
 *    laid out as [BvhBuilder] describes
 *  - [leaves]: 2 ints per leaf - first entry in [refs], count
 *  - [refs]: 2 ints per primitive, in leaf order - its kind ([LEAF_SPHERE], [LEAF_TRIANGLE],
 *    [LEAF_QUAD] or [LEAF_MEDIUM]) and its index in that kind's buffer
 *  - [spheres]: 8 floats per sphere - center at t=0 (3), center movement (3), radius, unused
 *  - [triangles]: 12 floats per facet, in leaf order - first vertex, two edges, each padded to a
 *    float4
 *  - [quads]: 16 floats - corner (3), the two edges (3 + 3), the basis vector w (3),
 *    the plane normal (3) and its offset
 *  - [materialInts]: 2 ints per material - type (0 Lambertian, 1 Metal, 2 Dielectric,
 *    3 DiffuseLight, 4 Isotropic), texture index
 *  - [mediumInts] / [mediumFloats]: per volume - the node its boundary starts at, the phase
 *    function's material, and the negated inverse density
 *  - [lights]: 2 ints per light - which primitive buffer it lives in and its index there
 *  - [materialFloats]: 2 floats per material - fuzz, refraction index
 *  - [textureInts]: 4 ints per texture - type (0 solid, 1 checker, 2 image, 3 missing image,
 *    4 noise, 5 u/v checker) + payload
 *  - [textureFloats]: 4 floats per texture - solid rgb, the checker's inverted scale, or a
 *    noise texture's scale
 *  - [perlinVectors] / [perlinPermutations]: one 256-entry lattice per Perlin instance, copied
 *    from the CPU tables so both renderers sample the same noise field
 */
class SceneBuffers(
    val nodes: IntArray,
    val leaves: IntArray,
    val refs: IntArray,
    val spheres: FloatArray,
    val sphereMaterials: IntArray,
    val triangles: FloatArray,
    val triangleMaterials: IntArray,
    val quads: FloatArray,
    val quadMaterials: IntArray,
    val mediumInts: IntArray,
    val mediumFloats: FloatArray,
    val lights: IntArray,
    val materialInts: IntArray,
    val materialFloats: FloatArray,
    val textureInts: IntArray,
    val textureFloats: FloatArray,
    val textureImages: ByteArray,
    val perlinVectors: FloatArray,
    val perlinPermutations: IntArray,
    val rootNode: Int,
    val maxDepth: Int
) {
    val nodeCount: Int get() = nodes.size / NODE_INTS
    val sphereCount: Int get() = sphereMaterials.size
    val triangleCount: Int get() = triangleMaterials.size
    val quadCount: Int get() = quadMaterials.size
    val mediumCount: Int get() = mediumFloats.size
    val lightCount: Int get() = lights.size / 2

    /**
     * The kernel's feature switches for this scene (see the top of raytracer.cu): a scene
     * without volumes, textures or moving spheres gets a kernel without that code, which needs
     * fewer registers and so keeps more of the GPU busy.
     */
    fun kernelDefines(): String {
        val kinds = refs.filterIndexed { i, _ -> i % 2 == 0 }.toSet()
        val materials = materialInts.filterIndexed { i, _ -> i % MATERIAL_INT_STRIDE == 0 }.toSet()
        val textures = textureInts.filterIndexed { i, _ -> i % TEXTURE_INT_STRIDE == 0 }.toSet()
        val moving = (0 until sphereCount).any { s ->
            (3..5).any { spheres[s * SPHERE_STRIDE + it] != 0.0f }
        }
        val features = linkedMapOf(
            "HAS_SPHERES" to (LEAF_SPHERE in kinds),
            "HAS_TRIANGLES" to (LEAF_TRIANGLE in kinds),
            "HAS_QUADS" to (LEAF_QUAD in kinds),
            "HAS_MEDIA" to (LEAF_MEDIUM in kinds),
            "HAS_MOTION" to moving,
            "HAS_LIGHTS" to (lightCount > 0),
            "TEX_CHECKER" to (1 in textures),
            "TEX_IMAGE" to (2 in textures),
            "TEX_NOISE" to (4 in textures),
            "TEX_UVCHECKER" to (5 in textures),
            "MAT_METAL" to (1 in materials),
            "MAT_DIELECTRIC" to (2 in materials),
            "MAT_EMISSIVE" to (3 in materials),
            "MAT_ISOTROPIC" to (4 in materials)
        )
        return features.entries.joinToString(" ") { (name, on) -> "-D$name=${if (on) 1 else 0}" }
    }

    companion object {
        /** Ints per inner BVH node, see [BvhBuilder]. */
        const val NODE_INTS = 16
        const val SPHERE_STRIDE = 8
        const val TRIANGLE_STRIDE = 12
        const val QUAD_STRIDE = 16
        const val MEDIUM_INT_STRIDE = 2
        const val MATERIAL_INT_STRIDE = 2
        const val MATERIAL_FLOAT_STRIDE = 2
        const val TEXTURE_INT_STRIDE = 4
        const val TEXTURE_FLOAT_STRIDE = 4

        /** 256 lattice vectors of x/y/z, and three 256-entry permutations, per Perlin instance. */
        const val PERLIN_BLOCK = 768

        /** Primitive kinds, as written into [refs]. */
        const val LEAF_SPHERE = -1
        const val LEAF_TRIANGLE = -2
        const val LEAF_QUAD = -3
        const val LEAF_MEDIUM = -4

        /** Depth of the traversal stack reserved in the kernel. */
        const val MAX_TRAVERSAL_DEPTH = 64
    }
}
