package rayTraceTypescript.gpu

import rayTraceTypescript.Vector
import rayTraceTypescript.objects.BvhNode
import rayTraceTypescript.objects.ConstantMedium
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.objects.RotateY
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.objects.Translate
import rayTraceTypescript.objects.Triangle

/**
 * A scene made ready for the GPU, in the two layouts its engines trace.
 *
 * [instanced] keeps each placed mesh as what it is - a mesh and where it stands - which is how
 * OptiX takes a scene: a tree per mesh, built once, and a small tree of instances above them, so
 * that moving a model changes a matrix and nothing else. [buffers] is the CUDA kernel's layout,
 * one tree over everything with the meshes pasted in where they stand ([SceneBuffers]); pasting
 * a large model costs a quarter of a second, so it is only worked out when an engine asks.
 */
class GpuScene internal constructor(val instanced: InstancedScene, kernelLayout: () -> SceneBuffers) {
    val buffers: SceneBuffers by lazy(kernelLayout)
}

/** A placed mesh as an instance: the mesh, its placement (see [PlacedMesh]) and what it is made of. */
class MeshInstance internal constructor(
    val block: MeshBlock,
    val scale: Float,
    val cos: Float,
    val sin: Float,
    val offset: Vector,
    /** The material of the whole mesh, and of a face whose own is missing from [palette]. */
    val material: Int,
    /** The material of each entry of the mesh's own palette, when the file's colours are used. */
    val palette: IntArray?
)

/**
 * The scene with its placed meshes as instances.
 *
 * The tables - spheres, quads, triangles, volumes, lights, materials, textures - are laid out as
 * in [SceneBuffers], which documents the strides, and read by the same kernel code. They hold
 * more than the scene's surfaces: the lights' stand-ins, which are sampled and never hit, and
 * the boundaries of the volumes. So the surfaces are named apart: the first [sceneTriangles]
 * triangles, [sceneQuads], [sceneSpheres] and the [instances].
 *
 * A volume's boundary keeps a tree of its own, in the CUDA kernel's node layout
 * ([boundaryNodes], [boundaryLeaves], [boundaryRefs]): finding where a ray enters and leaves a
 * box or a ball of fog is a small search the kernel does itself.
 */
class InstancedScene internal constructor(
    val spheres: FloatArray,
    val sphereMaterials: IntArray,
    val quads: FloatArray,
    val quadMaterials: IntArray,
    val triangles: FloatArray,
    val triangleMaterials: IntArray,
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
    val sceneTriangles: Int,
    val sceneQuads: IntArray,
    val sceneSpheres: IntArray,
    val instances: List<MeshInstance>,
    val boundaryNodes: IntArray,
    val boundaryLeaves: IntArray,
    val boundaryRefs: IntArray
) {
    val mediumCount: Int get() = mediumFloats.size
    val lightCount: Int get() = lights.size / 2

    /** The kernel's feature switches for this scene, as [SceneBuffers.kernelDefines] gives them for its layout. */
    fun kernelDefines(): String {
        val boundaries = boundaryRefs.filterIndexed { i, _ -> i % 2 == 0 }.toSet()
        return SceneBuffers.defines(
            spheres = sceneSpheres.isNotEmpty() || SceneBuffers.LEAF_SPHERE in boundaries,
            triangles = sceneTriangles > 0 || instances.isNotEmpty() || SceneBuffers.LEAF_TRIANGLE in boundaries,
            quads = sceneQuads.isNotEmpty() || SceneBuffers.LEAF_QUAD in boundaries,
            media = mediumCount > 0,
            sphereData = spheres, lights = lightCount > 0, materialInts = materialInts, textureInts = textureInts
        )
    }

    companion object {
        internal fun of(world: Hittable, lights: Hittable?): InstancedScene = Builder().build(world, lights)
    }

    /** Walks the scene graph once, as [SceneFlattener] does, keeping placed meshes whole. */
    private class Builder {
        private val materials = MaterialTable()
        private val spheres = FloatBuffer()
        private val sphereMaterials = IntBuffer()
        private val quads = FloatBuffer()
        private val quadMaterials = IntBuffer()
        private val triangles = FloatBuffer()
        private val triangleMaterials = IntBuffer()
        private val mediumInts = IntBuffer()
        private val mediumFloats = FloatBuffer()
        private val lights = IntBuffer()
        private val sceneQuads = IntBuffer()
        private val sceneSpheres = IntBuffer()
        private val instances = ArrayList<MeshInstance>()
        private val volumes = ArrayList<Pair<ConstantMedium, Transform>>()
        private val boundaries = BvhBuilder()

        fun build(world: Hittable, lightSources: Hittable?): InstancedScene {
            surfaces(world, Transform.IDENTITY)
            val sceneTriangles = triangleMaterials.size
            require(sceneTriangles + sceneQuads.size + sceneSpheres.size + instances.size + volumes.size > 0) { "Cannot render an empty world" }
            // Boundaries after every surface, so that the surfaces' triangles come first in their table.
            volumes.forEach { (medium, transform) -> volume(medium, transform) }
            lightSources?.let { lightsOf(it, Transform.IDENTITY) }
            return InstancedScene(
                spheres = spheres.toArray(),
                sphereMaterials = sphereMaterials.toArray(),
                quads = quads.toArray(),
                quadMaterials = quadMaterials.toArray(),
                triangles = triangles.toArray(),
                triangleMaterials = triangleMaterials.toArray(),
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
                sceneTriangles = sceneTriangles,
                sceneQuads = sceneQuads.toArray(),
                sceneSpheres = sceneSpheres.toArray(),
                instances = instances,
                boundaryNodes = boundaries.nodes.toArray(),
                boundaryLeaves = boundaries.leaves.toArray(),
                boundaryRefs = boundaries.refs.toArray()
            )
        }

        /** Writes out every surface below [hittable], instance transforms applied; volumes wait their turn. */
        private fun surfaces(hittable: Hittable, transform: Transform) {
            when (hittable) {
                is Sphere -> sceneSpheres.add(sphere(hittable, transform))
                is Triangle -> triangle(hittable, transform)
                is Quad -> sceneQuads.add(quad(hittable, transform).first)
                is PlacedMesh -> {
                    require(transform === Transform.IDENTITY) { "A placed mesh carries its own placement" }
                    val block = hittable.block
                    instances += MeshInstance(
                        block, hittable.scale, hittable.cos, hittable.sin, hittable.offset,
                        materials.register(hittable.material),
                        hittable.palette?.takeIf { block.mesh.hasColours }?.map { materials.register(it) }?.toIntArray()
                    )
                }
                is ConstantMedium -> volumes += hittable to transform
                is Translate -> surfaces(hittable.obj, transform.after(Transform.translation(hittable.offset)))
                is RotateY -> surfaces(hittable.obj, transform.after(Transform.rotation(hittable.cosTheta, hittable.sinTheta)))
                is BvhNode -> {
                    surfaces(hittable.left, transform)
                    // A one-object BVH node stores the same child twice; it is one primitive.
                    if (hittable.left !== hittable.right) surfaces(hittable.right, transform)
                }
                is HittableList -> hittable.objects.forEach { surfaces(it, transform) }
                else -> throw IllegalArgumentException("Unsupported hittable for GPU rendering: ${hittable::class.java.name}")
            }
        }

        /** A volume: its boundary's primitives, and the tree over them the kernel searches for the way in and out. */
        private fun volume(medium: ConstantMedium, transform: Transform) {
            val boundary = PrimitiveRefs()
            boundaryOf(medium.boundary, transform, boundary)
            mediumInts.add(boundaries.build(boundary))
            mediumInts.add(materials.register(medium.phaseFunction))
            mediumFloats.add(medium.negativeInverseDensity)
        }

        private fun boundaryOf(hittable: Hittable, transform: Transform, refs: PrimitiveRefs) {
            when (hittable) {
                is Sphere -> {
                    val index = sphere(hittable, transform)
                    val start = transform.point(hittable.center.origin)
                    val end = transform.point(hittable.center.at(1.0f))
                    val r = hittable.radius
                    refs.add(
                        SceneBuffers.LEAF_SPHERE, index,
                        minOf(start.x, end.x) - r, minOf(start.y, end.y) - r, minOf(start.z, end.z) - r,
                        maxOf(start.x, end.x) + r, maxOf(start.y, end.y) + r, maxOf(start.z, end.z) + r
                    )
                }
                is Triangle -> {
                    val (index, moved) = triangle(hittable, transform)
                    val box = moved.aabbBoundingBox()
                    refs.add(SceneBuffers.LEAF_TRIANGLE, index, box.x.min, box.y.min, box.z.min, box.x.max, box.y.max, box.z.max)
                }
                is Quad -> {
                    val (index, moved) = quad(hittable, transform)
                    val box = moved.aabbBoundingBox()
                    refs.add(SceneBuffers.LEAF_QUAD, index, box.x.min, box.y.min, box.z.min, box.x.max, box.y.max, box.z.max)
                }
                is PlacedMesh -> {
                    // A boundary is searched by the kernel, triangle by triangle: the mesh is spelt out where it stands.
                    val placed = hittable.block.mesh.rotatedY(hittable.degrees).transformed(hittable.scale, Vector(0f, 0f, 0f), hittable.offset)
                    placed.toHittables(hittable.material).forEach { boundaryOf(it, transform, refs) }
                }
                is ConstantMedium -> throw IllegalArgumentException("A volume cannot contain a volume")
                is Translate -> boundaryOf(hittable.obj, transform.after(Transform.translation(hittable.offset)), refs)
                is RotateY -> boundaryOf(hittable.obj, transform.after(Transform.rotation(hittable.cosTheta, hittable.sinTheta)), refs)
                is BvhNode -> {
                    boundaryOf(hittable.left, transform, refs)
                    if (hittable.left !== hittable.right) boundaryOf(hittable.right, transform, refs)
                }
                is HittableList -> hittable.objects.forEach { boundaryOf(it, transform, refs) }
                else -> throw IllegalArgumentException("Unsupported hittable for GPU rendering: ${hittable::class.java.name}")
            }
        }

        /** The lights' stand-ins: bare geometry the renderer aims samples at, apart from the surfaces rays hit. */
        private fun lightsOf(hittable: Hittable, transform: Transform) {
            when (hittable) {
                is Quad -> {
                    lights.add(SceneBuffers.LEAF_QUAD)
                    lights.add(quad(hittable, transform).first)
                }
                is Sphere -> {
                    lights.add(SceneBuffers.LEAF_SPHERE)
                    lights.add(sphere(hittable, transform))
                }
                is HittableList -> hittable.objects.forEach { lightsOf(it, transform) }
                is Translate -> lightsOf(hittable.obj, transform.after(Transform.translation(hittable.offset)))
                is RotateY -> lightsOf(hittable.obj, transform.after(Transform.rotation(hittable.cosTheta, hittable.sinTheta)))
                is BvhNode -> {
                    lightsOf(hittable.left, transform)
                    if (hittable.left !== hittable.right) lightsOf(hittable.right, transform)
                }
                else -> throw IllegalArgumentException(
                    "Only quads and spheres can be sampled as lights, not ${hittable::class.java.name}"
                )
            }
        }

        private fun sphere(sphere: Sphere, transform: Transform): Int {
            val index = sphereMaterials.size
            // A sphere is round: a rotation only moves its centre, so the radius survives untouched.
            spheres.addSphere(transform.point(sphere.center.origin), transform.point(sphere.center.at(1.0f)), sphere.radius)
            sphereMaterials.add(materials.register(sphere.material))
            return index
        }

        private fun quad(original: Quad, transform: Transform): Pair<Int, Quad> {
            // Rebuilt from its moved corner and edges, so its own constructor works the plane out again.
            val quad = if (transform === Transform.IDENTITY) original
            else Quad(transform.point(original.q), transform.vector(original.u), transform.vector(original.v), original.material)
            val index = quadMaterials.size
            quads.addQuad(quad)
            quadMaterials.add(materials.register(quad.material))
            return index to quad
        }

        private fun triangle(original: Triangle, transform: Transform): Pair<Int, Triangle> {
            val triangle = if (transform === Transform.IDENTITY) original
            else Triangle(transform.point(original.v0), transform.point(original.v1), transform.point(original.v2), original.material)
            val index = triangleMaterials.size
            triangles.addTriangle(triangle)
            triangleMaterials.add(materials.register(triangle.material))
            return index to triangle
        }
    }
}
