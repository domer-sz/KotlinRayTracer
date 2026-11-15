package rayTraceTypescript.gpu

import rayTraceTypescript.Point
import rayTraceTypescript.materials.Material
import rayTraceTypescript.objects.BvhNode
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.ConstantMedium
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.objects.RotateY
import rayTraceTypescript.objects.Translate
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.objects.Triangle

/**
 * Walks the CPU scene graph once and lays it out as flat arrays for the GPU.
 *
 * The CPU tree is not copied: its shape is whatever the scene was assembled from (lists split by
 * index, the books' median-split BvhNode), which is fine for a CPU but costs the GPU dearly on
 * big meshes. Instead every primitive is collected with its instance transforms folded in, and
 * a fresh tree is built over them with the surface area heuristic (see [BvhBuilder]). The closest
 * hit along a ray does not depend on the tree, so both renderers still see the same scene.
 */
class SceneFlattener private constructor() {

    private val spheres = FloatBuffer()
    private val sphereMaterials = IntBuffer()
    private val triangles = FloatBuffer()
    private val triangleMaterials = IntBuffer()
    private val quads = FloatBuffer()
    private val quadMaterials = IntBuffer()
    private val mediumInts = IntBuffer()
    private val mediumFloats = FloatBuffer()
    private val lights = IntBuffer()
    private val materials = MaterialTable()

    private val bvh = BvhBuilder()

    companion object {
        fun flatten(world: Hittable, lights: Hittable? = null): SceneBuffers =
            SceneFlattener().build(world, lights)
    }

    private fun build(world: Hittable, lightSources: Hittable?): SceneBuffers {
        val refs = PrimitiveRefs()
        collect(world, Transform.IDENTITY, refs)
        require(refs.size > 0) { "Cannot render an empty world" }
        val root = bvh.build(refs)
        lightSources?.let { collectLights(it, Transform.IDENTITY) }
        require(bvh.maxDepth <= SceneBuffers.MAX_TRAVERSAL_DEPTH) {
            "Scene tree is ${bvh.maxDepth} levels deep, kernel stack holds ${SceneBuffers.MAX_TRAVERSAL_DEPTH}"
        }
        val (orderedTriangles, orderedTriangleMaterials) = trianglesInLeafOrder()
        return SceneBuffers(
            nodes = bvh.nodes.toArray(),
            leaves = bvh.leaves.toArray(),
            refs = bvh.refs.toArray(),
            spheres = spheres.toArray(),
            sphereMaterials = sphereMaterials.toArray(),
            triangles = orderedTriangles,
            triangleMaterials = orderedTriangleMaterials,
            quads = quads.toArray(),
            quadMaterials = quadMaterials.toArray(),
            mediumInts = mediumInts.toArray(),
            mediumFloats = mediumFloats.toArray(),
            lights = lights.toArray(),
            materialInts = materials.materialInts.toArray(),
            materialFloats = materials.materialFloats.toArray(),
            textureInts = materials.textureInts.toArray(),
            textureFloats = materials.textureFloats.toArray(),
            textureImages = materials.images.toByteArray(),
            perlinVectors = materials.perlinVectors.toArray(),
            perlinPermutations = materials.perlinPermutations.toArray(),
            rootNode = root,
            maxDepth = bvh.maxDepth
        )
    }

    /** Appends every primitive below [hittable] to [refs], instance transforms applied. */
    private fun collect(hittable: Hittable, transform: Transform, refs: PrimitiveRefs) {
        when (hittable) {
            is Sphere -> {
                val (index, start, end) = appendSphere(hittable, transform)
                val r = hittable.radius
                refs.add(
                    SceneBuffers.LEAF_SPHERE, index,
                    minOf(start.x, end.x) - r, minOf(start.y, end.y) - r, minOf(start.z, end.z) - r,
                    maxOf(start.x, end.x) + r, maxOf(start.y, end.y) + r, maxOf(start.z, end.z) + r
                )
            }
            is Triangle -> {
                val triangle = if (transform === Transform.IDENTITY) hittable
                else Triangle(
                    transform.point(hittable.v0), transform.point(hittable.v1), transform.point(hittable.v2),
                    hittable.material
                )
                val index = triangleMaterials.size
                for (v in listOf(triangle.v0, triangle.edge1, triangle.edge2)) {
                    triangles.add(v.x); triangles.add(v.y); triangles.add(v.z); triangles.add(0.0f)
                }
                triangleMaterials.add(registerMaterial(triangle.material))
                val box = triangle.aabbBoundingBox()
                refs.add(SceneBuffers.LEAF_TRIANGLE, index, box.x.min, box.y.min, box.z.min, box.x.max, box.y.max, box.z.max)
            }
            is Quad -> {
                val (index, quad) = appendQuad(hittable, transform)
                val box = quad.aabbBoundingBox()
                refs.add(SceneBuffers.LEAF_QUAD, index, box.x.min, box.y.min, box.z.min, box.x.max, box.y.max, box.z.max)
            }
            is PlacedMesh -> {
                require(transform === Transform.IDENTITY) { "A placed mesh carries its own placement" }
                collectPlaced(hittable, refs)
            }
            is ConstantMedium -> collectMedium(hittable, transform, refs)
            is Translate -> collect(hittable.obj, transform.after(Transform.translation(hittable.offset)), refs)
            is RotateY -> collect(hittable.obj, transform.after(Transform.rotation(hittable.cosTheta, hittable.sinTheta)), refs)
            is BvhNode -> {
                collect(hittable.left, transform, refs)
                // A one-object BVH node stores the same child twice; it is one primitive.
                if (hittable.left !== hittable.right) collect(hittable.right, transform, refs)
            }
            is HittableList -> hittable.objects.forEach { collect(it, transform, refs) }
            else -> throw IllegalArgumentException("Unsupported hittable for GPU rendering: ${hittable::class.java.name}")
        }
    }

    /**
     * A placed [MeshBlock]: its triangles copied in moved, and its prebuilt tree grafted in whole
     * as one primitive of the scene's tree - no tree is built over its triangles again.
     */
    private fun collectPlaced(placed: PlacedMesh, refs: PrimitiveRefs) {
        val block = placed.block
        val base = triangleMaterials.size
        val material = registerMaterial(placed.material)
        // The file's own materials, when its colours are used: registered once each.
        val faces = block.faceMaterials
        val paletteIds = placed.palette?.takeIf { faces != null }?.map { registerMaterial(it) }
        val source = block.triangles
        val stride = SceneBuffers.TRIANGLE_STRIDE
        val c = placed.cos
        val s = placed.sin
        val k = placed.scale
        val o = placed.offset
        for (t in 0 until block.triangleCount) {
            val at = t * stride
            // v0 is a point, the two edges are directions: only v0 moves by the offset.
            for (part in 0 until 3) {
                val x = source[at + part * 4]
                val y = source[at + part * 4 + 1]
                val z = source[at + part * 4 + 2]
                val point = part == 0
                triangles.add((c * x + s * z) * k + if (point) o.x else 0.0f)
                triangles.add(y * k + if (point) o.y else 0.0f)
                triangles.add((-s * x + c * z) * k + if (point) o.z else 0.0f)
                triangles.add(0.0f)
            }
            triangleMaterials.add(if (paletteIds != null) paletteIds.getOrElse(faces!![t]) { material } else material)
        }
        val root = bvh.graft(placed, base)
        // Its box is what its root's two children cover, so the tree above holds them whole.
        val at = root * BvhBuilder.NODE_INTS
        fun f(k: Int) = java.lang.Float.intBitsToFloat(bvh.nodes[at + k])
        refs.add(
            BvhBuilder.SUBTREE, root,
            minOf(f(0), f(4)), minOf(f(2), f(6)), minOf(f(8), f(10)),
            maxOf(f(1), f(5)), maxOf(f(3), f(7)), maxOf(f(9), f(11))
        )
    }

    /**
     * A volume gets a tree of its own over its boundary, and one opaque leaf in the tree that
     * contains it: the kernel runs a second, nested traversal over the boundary to find where a
     * ray enters and leaves the fog.
     */
    private fun collectMedium(medium: ConstantMedium, transform: Transform, refs: PrimitiveRefs) {
        val boundary = PrimitiveRefs()
        collect(medium.boundary, transform, boundary)
        require(boundary.kinds.toArray().none { it == SceneBuffers.LEAF_MEDIUM }) { "A volume cannot contain a volume" }
        val boundaryRoot = bvh.build(boundary)

        val index = mediumFloats.size
        mediumInts.add(boundaryRoot)
        mediumInts.add(registerMaterial(medium.phaseFunction))
        mediumFloats.add(medium.negativeInverseDensity)
        val b = boundary.totalBounds()
        refs.add(SceneBuffers.LEAF_MEDIUM, index, b[0], b[1], b[2], b[3], b[4], b[5])
    }

    /**
     * Lights are written a second time, as bare geometry outside the tree: the renderer samples
     * them directly, which is a different job from finding them along a ray.
     */
    private fun collectLights(hittable: Hittable, transform: Transform) {
        when (hittable) {
            is Quad -> {
                val (index, _) = appendQuad(hittable, transform)
                lights.add(SceneBuffers.LEAF_QUAD)
                lights.add(index)
            }
            is Sphere -> {
                val (index, _, _) = appendSphere(hittable, transform)
                lights.add(SceneBuffers.LEAF_SPHERE)
                lights.add(index)
            }
            is HittableList -> hittable.objects.forEach { collectLights(it, transform) }
            is Translate -> collectLights(hittable.obj, transform.after(Transform.translation(hittable.offset)))
            is RotateY -> collectLights(hittable.obj, transform.after(Transform.rotation(hittable.cosTheta, hittable.sinTheta)))
            is BvhNode -> {
                collectLights(hittable.left, transform)
                if (hittable.left !== hittable.right) collectLights(hittable.right, transform)
            }
            else -> throw IllegalArgumentException(
                "Only quads and spheres can be sampled as lights, not ${hittable::class.java.name}"
            )
        }
    }

    /**
     * Rewrites the triangle buffer in the order the leaves reference it, so the facets of one
     * leaf - and of neighbouring leaves - sit next to each other in memory.
     */
    private fun trianglesInLeafOrder(): Pair<FloatArray, IntArray> {
        val source = triangles.toArray()
        val sourceMaterials = triangleMaterials.toArray()
        val remap = IntArray(sourceMaterials.size) { -1 }
        val ordered = FloatArray(source.size)
        val orderedMaterials = IntArray(sourceMaterials.size)
        var next = 0
        val refs = bvh.refs
        for (i in 0 until refs.size step 2) {
            if (refs[i] != SceneBuffers.LEAF_TRIANGLE) continue
            val old = refs[i + 1]
            if (remap[old] < 0) {
                remap[old] = next
                System.arraycopy(source, old * SceneBuffers.TRIANGLE_STRIDE, ordered, next * SceneBuffers.TRIANGLE_STRIDE, SceneBuffers.TRIANGLE_STRIDE)
                orderedMaterials[next] = sourceMaterials[old]
                next++
            }
            refs[i + 1] = remap[old]
        }
        return ordered to orderedMaterials
    }

    /** Writes a quad's geometry and returns its index; the caller decides where it is referenced. */
    private fun appendQuad(original: Quad, transform: Transform): Pair<Int, Quad> {
        // Rebuilding the quad from transformed corners lets its own constructor work out the
        // plane, the uv basis and the padded box again.
        val quad = if (transform === Transform.IDENTITY) original
        else Quad(transform.point(original.q), transform.vector(original.u), transform.vector(original.v), original.material)

        val index = quadMaterials.size
        for (vector in listOf(quad.q, quad.u, quad.v, quad.w, quad.normal)) {
            quads.add(vector.x)
            quads.add(vector.y)
            quads.add(vector.z)
        }
        quads.add(quad.d)
        quadMaterials.add(registerMaterial(quad.material))
        return index to quad
    }

    /** Writes a sphere's geometry and returns its index. */
    private fun appendSphere(sphere: Sphere, transform: Transform): Triple<Int, Point, Point> {
        val sphereIndex = sphereMaterials.size
        // A sphere is round: a rotation only moves its centre, so the radius survives untouched.
        val start = transform.point(sphere.center.origin)
        val end = transform.point(sphere.center.at(1.0f))

        spheres.add(start.x)
        spheres.add(start.y)
        spheres.add(start.z)
        spheres.add(end.x - start.x)
        spheres.add(end.y - start.y)
        spheres.add(end.z - start.z)
        spheres.add(sphere.radius)
        spheres.add(0.0f)
        sphereMaterials.add(registerMaterial(sphere.material))
        return Triple(sphereIndex, start, end)
    }

    private fun registerMaterial(material: Material): Int = materials.register(material)
}

/** A growable FloatArray: the scene can run to millions of floats, too many to box one by one. */
internal class FloatBuffer(capacity: Int = 64) {
    private var data = FloatArray(capacity)
    var size = 0
        private set

    fun add(value: Float) {
        if (size == data.size) data = data.copyOf(data.size * 2)
        data[size++] = value
    }

    operator fun get(index: Int): Float = data[index]
    operator fun set(index: Int, value: Float) { data[index] = value }
    fun toArray(): FloatArray = data.copyOf(size)
}

/** The Int counterpart of [FloatBuffer]. */
internal class IntBuffer(capacity: Int = 64) {
    private var data = IntArray(capacity)
    var size = 0
        private set

    fun add(value: Int) {
        if (size == data.size) data = data.copyOf(data.size * 2)
        data[size++] = value
    }

    operator fun get(index: Int): Int = data[index]
    operator fun set(index: Int, value: Int) { data[index] = value }
    fun toArray(): IntArray = data.copyOf(size)
}

/** Primitives waiting for a tree: what they are, where their data lives, and their box. */
internal class PrimitiveRefs {
    val kinds = IntBuffer()
    val indices = IntBuffer()
    val bounds = FloatBuffer()
    val size: Int get() = kinds.size

    fun add(kind: Int, index: Int, minX: Float, minY: Float, minZ: Float, maxX: Float, maxY: Float, maxZ: Float) {
        kinds.add(kind)
        indices.add(index)
        bounds.add(minX); bounds.add(minY); bounds.add(minZ)
        bounds.add(maxX); bounds.add(maxY); bounds.add(maxZ)
    }

    fun totalBounds(): FloatArray {
        val b = floatArrayOf(
            Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY,
            Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY
        )
        for (i in 0 until size) for (a in 0 until 3) {
            b[a] = minOf(b[a], bounds[i * 6 + a])
            b[a + 3] = maxOf(b[a + 3], bounds[i * 6 + a + 3])
        }
        return b
    }
}
