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
 *  - [triangleNormals]: 9 floats per facet, in the same order - the normals at its three corners
 *    for smooth shading, zeros for a flat facet; empty when no facet of the scene is smooth
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
    val maxDepth: Int,
    val triangleNormals: FloatArray = FloatArray(0),
    /** 6 floats per facet, in the same order: the texture coordinates at its corners, NaN first for one without; empty when no facet has any. */
    val triangleUvs: FloatArray = FloatArray(0)
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
        return defines(
            LEAF_SPHERE in kinds, LEAF_TRIANGLE in kinds, LEAF_QUAD in kinds, LEAF_MEDIUM in kinds,
            spheres, lightCount > 0, materialInts, textureInts, smooth = triangleNormals.isNotEmpty(),
            uvs = triangleUvs.isNotEmpty(), materialFloats = materialFloats
        )
    }

    companion object {
        /**
         * The kernel's feature switches for a scene that traces [spheres], [triangles], [quads]
         * and [media] or not, whatever its layout: the rest is read off the tables themselves.
         */
        internal fun defines(
            spheres: Boolean, triangles: Boolean, quads: Boolean, media: Boolean,
            sphereData: FloatArray, lights: Boolean, materialInts: IntArray, textureInts: IntArray,
            smooth: Boolean,
            uvs: Boolean = false,
            materialFloats: FloatArray = FloatArray(0)
        ): String {
            val materials = materialInts.filterIndexed { i, _ -> i % MATERIAL_INT_STRIDE == 0 }.toSet()
            // A surface that glows from a texture needs the emission code as much as a lamp does.
            val glows = (0 until materialInts.size / MATERIAL_INT_STRIDE).any { materialInts[it * MATERIAL_INT_STRIDE + 2] >= 0 }
            val textures = textureInts.filterIndexed { i, _ -> i % TEXTURE_INT_STRIDE == 0 }.toSet()
            val moving = (0 until sphereData.size / SPHERE_STRIDE).any { s ->
                (3..5).any { sphereData[s * SPHERE_STRIDE + it] != 0.0f }
            }
            val features = linkedMapOf(
                "HAS_SPHERES" to spheres,
                "HAS_TRIANGLES" to triangles,
                "HAS_QUADS" to quads,
                "HAS_MEDIA" to media,
                "HAS_MOTION" to moving,
                "HAS_LIGHTS" to lights,
                "HAS_SMOOTH" to smooth,
                "HAS_UV" to uvs,
                // Normal maps bend only textured triangles' normals.
                "HAS_NORMALMAP" to (uvs && (0 until materialFloats.size / MATERIAL_FLOAT_STRIDE).any { materialFloats[it * MATERIAL_FLOAT_STRIDE + 3] > 0f }),
                "TEX_CHECKER" to (1 in textures),
                "TEX_IMAGE" to (2 in textures),
                "TEX_NOISE" to (4 in textures),
                "TEX_UVCHECKER" to (5 in textures),
                "MAT_METAL" to (1 in materials),
                "MAT_DIELECTRIC" to (2 in materials),
                "MAT_EMISSIVE" to (3 in materials || glows),
                "MAT_ISOTROPIC" to (4 in materials)
            )
            return features.entries.joinToString(" ") { (name, on) -> "-D$name=${if (on) 1 else 0}" }
        }

        /** Ints per inner BVH node, see [BvhBuilder]. */
        const val NODE_INTS = 16
        const val SPHERE_STRIDE = 8
        const val TRIANGLE_STRIDE = 12
        const val QUAD_STRIDE = 16
        const val MEDIUM_INT_STRIDE = 2
        const val MATERIAL_INT_STRIDE = 4
        const val MATERIAL_FLOAT_STRIDE = 4
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
