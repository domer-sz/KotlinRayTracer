package rayTraceTypescript.gpu

import rayTraceTypescript.Aabb
import rayTraceTypescript.Interval
import rayTraceTypescript.Point
import rayTraceTypescript.Ray
import rayTraceTypescript.Vector
import rayTraceTypescript.materials.Material
import rayTraceTypescript.objects.BvhNode
import rayTraceTypescript.objects.Hit
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Mesh

/**
 * A mesh laid out for the GPU once: its triangles and its own SAH tree, in its own space. A
 * scene places it ([PlacedMesh]) by copying that tree in scaled, turned about the vertical and
 * moved, instead of building a tree again - so moving, turning or resizing a model of hundreds
 * of thousands of triangles in the viewer costs a copy rather than a rebuild.
 *
 * The copied boxes are the boxes of the turned boxes, a little looser than a tree built for the
 * turned mesh would have; every hit is still found, only traversal gets slightly slower.
 *
 * That tree is the CUDA kernel's, and is built the first time the kernel's layout asks for it:
 * OptiX builds a tree of its own from the mesh's vertices, on the card, and a model that only
 * OptiX ever traces never pays for this one - a couple of seconds for a large model.
 */
class MeshBlock(val mesh: Mesh) {
    val triangleCount: Int = mesh.triangleCount
    val bounds: Aabb = mesh.bounds()

    private val tree: Tree by lazy { Tree(mesh) }

    /** [SceneBuffers.TRIANGLE_STRIDE] floats per triangle (v0, edge1, edge2), in leaf order. */
    internal val triangles: FloatArray get() = tree.triangles
    internal val nodes: IntArray get() = tree.nodes
    internal val leaves: IntArray get() = tree.leaves
    /** (kind, triangle) pairs, triangles counted from 0 in [triangles]. */
    internal val refs: IntArray get() = tree.refs
    internal val maxDepth: Int get() = tree.maxDepth
    /** Each triangle's entry in the mesh's palette, in leaf order; null when the file gave none. */
    internal val faceMaterials: IntArray? get() = tree.faceMaterials

    /** The mesh's triangles and SAH tree as the CUDA kernel reads them. */
    private class Tree(mesh: Mesh) {
        val triangles: FloatArray
        val nodes: IntArray
        val leaves: IntArray
        val refs: IntArray
        val maxDepth: Int
        val faceMaterials: IntArray?

        init {
            val triangleCount = mesh.triangleCount
            val stride = SceneBuffers.TRIANGLE_STRIDE
            val source = FloatArray(triangleCount * stride)
            val prims = PrimitiveRefs()
            val v = mesh.vertices
            for (t in 0 until triangleCount) {
                val at = t * Mesh.VERTEX_FLOATS
                val out = t * stride
                for (a in 0 until 3) {
                    source[out + a] = v[at + a]
                    source[out + 4 + a] = v[at + 3 + a] - v[at + a]
                    source[out + 8 + a] = v[at + 6 + a] - v[at + a]
                }
                // The same padded box a Triangle gives itself.
                val box = Aabb.fromPoints(
                    Point(minOf(v[at], v[at + 3], v[at + 6]), minOf(v[at + 1], v[at + 4], v[at + 7]), minOf(v[at + 2], v[at + 5], v[at + 8])),
                    Point(maxOf(v[at], v[at + 3], v[at + 6]), maxOf(v[at + 1], v[at + 4], v[at + 7]), maxOf(v[at + 2], v[at + 5], v[at + 8]))
                ).padded()
                prims.add(SceneBuffers.LEAF_TRIANGLE, t, box.x.min, box.y.min, box.z.min, box.x.max, box.y.max, box.z.max)
            }
            val builder = BvhBuilder()
            builder.build(prims)

            // Triangles in the order the leaves read them, as SceneFlattener lays a scene out.
            val ordered = FloatArray(source.size)
            val treeRefs = builder.refs.toArray()
            val sourceMaterials = mesh.faceMaterials?.takeIf { mesh.hasColours }
            val orderedMaterials = sourceMaterials?.let { IntArray(it.size) }
            for (i in treeRefs.indices step 2) {
                val old = treeRefs[i + 1]
                val next = i / 2
                System.arraycopy(source, old * stride, ordered, next * stride, stride)
                if (orderedMaterials != null) orderedMaterials[next] = sourceMaterials[old]
                treeRefs[i + 1] = next
            }
            triangles = ordered
            faceMaterials = orderedMaterials
            nodes = builder.nodes.toArray()
            leaves = builder.leaves.toArray()
            refs = treeRefs
            maxDepth = builder.maxDepth
        }
    }
}

/**
 * Where and how a [MeshBlock] stands: each vertex p goes to scale * turn(p) + offset, the turn
 * being [degrees] about the vertical, as [Mesh.rotatedY] turns. On the GPU the block's tree is
 * copied in; the CPU (picking, the CPU renderer) gets an ordinary tree over the moved mesh,
 * built only if it is ever asked.
 */
class PlacedMesh(
    val block: MeshBlock,
    val scale: Float,
    val degrees: Float,
    val offset: Vector,
    val material: Material,
    /** The file's own materials, one per palette entry, when its colours are used; else [material] throughout. */
    val palette: List<Material>? = null
) : Hittable {
    val cos: Float = kotlin.math.cos(Math.toRadians(degrees.toDouble())).toFloat()
    val sin: Float = kotlin.math.sin(Math.toRadians(degrees.toDouble())).toFloat()

    private val cpu: Hittable by lazy {
        val moved = block.mesh.rotatedY(degrees).transformed(scale, Vector(0f, 0f, 0f), offset)
        val faces = if (palette != null) moved.toColouredHittables(material) else moved.toHittables(material)
        BvhNode(HittableList(faces.toMutableList()))
    }

    override fun hit(ray: Ray, rayT: Interval): Hit? = cpu.hit(ray, rayT)

    /** The moved block's box, from its own box's turned corners - lists ask for it eagerly. */
    private val box: Aabb = run {
        val b = block.bounds
        var minX = Float.POSITIVE_INFINITY; var maxX = Float.NEGATIVE_INFINITY
        var minZ = Float.POSITIVE_INFINITY; var maxZ = Float.NEGATIVE_INFINITY
        for (x in floatArrayOf(b.x.min, b.x.max)) for (z in floatArrayOf(b.z.min, b.z.max)) {
            val tx = (cos * x + sin * z) * scale + offset.x
            val tz = (-sin * x + cos * z) * scale + offset.z
            minX = minOf(minX, tx); maxX = maxOf(maxX, tx)
            minZ = minOf(minZ, tz); maxZ = maxOf(maxZ, tz)
        }
        Aabb.fromPoints(
            Point(minX, b.y.min * scale + offset.y, minZ),
            Point(maxX, b.y.max * scale + offset.y, maxZ)
        ).padded()
    }

    override fun aabbBoundingBox(): Aabb = box
}
