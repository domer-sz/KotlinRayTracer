package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.GpuRenderer
import rayTraceTypescript.gpu.MeshBlock
import rayTraceTypescript.gpu.PlacedMesh
import rayTraceTypescript.gpu.optix.OptixApi
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.DiffuseLight
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Metal
import rayTraceTypescript.objects.ConstantMedium
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.objects.MeshMaterial
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.render.PpmWriter
import rayTraceTypescript.utils.RandomSource
import java.nio.file.Files

/**
 * Scenes the OptiX engine traces that no stored CPU render covers - placed meshes on a quad floor,
 * turned and scaled every way - held to the CUDA kernel instead, which [GpuCpuComparisonTest]
 * holds to the CPU renders. The yardstick is the same: OptiX may differ from the kernel by no
 * more than two renders of the kernel at different seeds differ from each other.
 */
class OptixCudaComparisonTest {

    @Test
    fun `optix places meshes turned and scaled as the kernel does`() {
        assumeOptix()
        val block = MeshBlock(octahedron())
        val world = HittableList(
            mutableListOf(
                floor(),
                PlacedMesh(block, 1.0f, 0.0f, Vector(-1.4f, 0.0f, 0.2f), Lambertian(Color(0.72f, 0.38f, 0.22f))),
                PlacedMesh(block, 0.6f, 37.0f, Vector(0.3f, 0.0f, 0.9f), Metal(Color(0.7f, 0.7f, 0.75f), 0.1f)),
                PlacedMesh(block, 1.4f, -110.0f, Vector(1.2f, 0.3f, -0.8f), Lambertian(Color(0.2f, 0.4f, 0.7f)))
            )
        )
        assertLooksTheSame(world)
    }

    @Test
    fun `optix lights placed meshes from a lamp as the kernel does`() {
        assumeOptix()
        val block = MeshBlock(octahedron())
        val light = lamp()
        val world = HittableList(
            mutableListOf(
                floor(),
                PlacedMesh(block, 1.0f, 45.0f, Vector(-0.8f, 0.0f, 0.2f), Lambertian(Color(0.72f, 0.38f, 0.22f))),
                PlacedMesh(block, 0.7f, 190.0f, Vector(0.9f, 0.0f, -0.3f), Lambertian(Color(0.2f, 0.4f, 0.7f))),
                light
            )
        )
        assertLooksTheSame(world, lights = light)
    }

    @Test
    fun `optix colours a mesh by its own palette as the kernel does`() {
        assumeOptix()
        // Each face of the octahedron names an entry of the file's palette; one names an entry
        // past its end and falls back to the mesh's own material.
        val palette = listOf(
            MeshMaterial("red", Color(0.8f, 0.15f, 0.1f)),
            MeshMaterial("green", Color(0.1f, 0.7f, 0.2f)),
            MeshMaterial("blue", Color(0.15f, 0.25f, 0.85f))
        )
        val shape = octahedron()
        val coloured = Mesh(shape.vertices, intArrayOf(0, 1, 2, 0, 1, 2, 0, 7), palette)
        val block = MeshBlock(coloured)
        val world = HittableList(
            mutableListOf(
                floor(),
                PlacedMesh(block, 1.2f, 30.0f, Vector(0.0f, 0.0f, 0.0f), Lambertian(Color(0.9f, 0.9f, 0.9f)), palette.map { it.material() })
            )
        )
        assertLooksTheSame(world)
    }

    @Test
    fun `optix shades a smooth placed mesh as the kernel does`() {
        // The two engines read a smooth mesh's normals from different layouts - the kernel's
        // turned and in leaf order, OptiX's in the mesh's own space, turned per hit - so a turn
        // or an order gone wrong in either shows here.
        assumeOptix()
        val block = MeshBlock(octahedron().smoothed())
        val light = lamp()
        val world = HittableList(
            mutableListOf(
                floor(),
                PlacedMesh(block, 1.0f, 30.0f, Vector(-1.1f, 0.0f, 0.3f), Metal(Color(0.9f, 0.7f, 0.3f), 0.05f)),
                PlacedMesh(block, 0.8f, -75.0f, Vector(0.9f, 0.0f, -0.4f), Lambertian(Color(0.2f, 0.4f, 0.7f))),
                light
            )
        )
        assertLooksTheSame(world, lights = light)
    }

    @Test
    fun `optix shades loose smooth triangles as the kernel does`() {
        assumeOptix()
        val facets = octahedron().smoothed().rotatedY(20.0f).toHittables(Metal(Color(0.8f, 0.8f, 0.85f), 0.02f))
        val world = HittableList((listOf(floor()) + facets).toMutableList())
        assertLooksTheSame(world)
    }

    @Test
    fun `optix places a model on a sphere floor as the kernel does`() {
        // The model scene's layout: a mesh standing on the top of a huge sphere.
        assumeOptix()
        val world = HittableList(
            mutableListOf(
                Sphere(Point(0.0f, -1000.0f, 0.0f), 1000.0f, Lambertian(Color(0.5f, 0.5f, 0.5f))),
                PlacedMesh(MeshBlock(octahedron().smoothed()), 1.2f, 25.0f, Vector(0.0f, 0.0f, 0.0f), Metal(Color(0.9f, 0.7f, 0.3f), 0.05f))
            )
        )
        assertLooksTheSame(world)
    }

    @Test
    fun `optix moves a moving sphere as the kernel does`() {
        // A sphere that travels during the shutter: its box must cover the whole path, and the
        // intersection program must be handed each ray's time.
        assumeOptix()
        val world = HittableList(
            mutableListOf(
                floor(),
                Sphere(Point(-1.2f, 0.6f, 0.0f), Point(0.4f, 0.9f, 0.3f), 0.6f, Lambertian(Color(0.7f, 0.3f, 0.1f))),
                Sphere(Point(1.2f, 0.5f, -0.6f), 0.5f, Metal(Color(0.8f, 0.8f, 0.85f), 0.1f))
            )
        )
        assertLooksTheSame(world)
    }

    @Test
    fun `optix sees through a glass sphere as the kernel does`() {
        // A refracted ray starts inside the sphere and must find its far side: the second root.
        assumeOptix()
        val world = HittableList(
            mutableListOf(
                floor(),
                Sphere(Point(0.0f, 0.8f, 0.0f), 0.8f, Dielectric(1.5f)),
                Sphere(Point(0.0f, 0.8f, 0.0f), -0.7f, Dielectric(1.5f)),
                PlacedMesh(MeshBlock(octahedron()), 0.8f, 10.0f, Vector(-1.6f, 0.0f, -1.0f), Lambertian(Color(0.2f, 0.4f, 0.7f)))
            )
        )
        assertLooksTheSame(world)
    }

    @Test
    fun `optix aims at a sphere light as the kernel does`() {
        assumeOptix()
        val light = Sphere(Point(0.0f, 3.0f, 0.0f), 0.7f, DiffuseLight(Color(8.0f, 8.0f, 8.0f)))
        val world = HittableList(
            mutableListOf(
                floor(),
                Sphere(Point(-0.9f, 0.6f, 0.2f), 0.6f, Lambertian(Color(0.72f, 0.38f, 0.22f))),
                PlacedMesh(MeshBlock(octahedron()), 0.9f, 45.0f, Vector(0.9f, 0.0f, -0.3f), Lambertian(Color(0.2f, 0.4f, 0.7f))),
                light
            )
        )
        assertLooksTheSame(world, lights = light)
    }

    @Test
    fun `optix traces fog around the camera and in a box as the kernel does`() {
        // A thin haze the camera stands in, as in the final scene of the second book, and a dense
        // ball of fog: every ray starts inside the first, and many pass through the second.
        assumeOptix()
        val world = HittableList(
            mutableListOf(
                floor(),
                Sphere(Point(1.0f, 0.6f, -0.4f), 0.6f, Metal(Color(0.8f, 0.8f, 0.85f), 0.05f)),
                ConstantMedium(Sphere(Point(-0.8f, 0.7f, 0.3f), 0.7f, Dielectric(1.5f)), 1.5f, Color(0.2f, 0.4f, 0.9f)),
                ConstantMedium(Sphere(Point(0.0f, 0.0f, 0.0f), 50.0f, Dielectric(1.5f)), 0.02f, Color(1.0f, 1.0f, 1.0f))
            )
        )
        assertLooksTheSame(world)
    }

    private fun assertLooksTheSame(world: Hittable, lights: Hittable? = null) {
        val cuda = render(world, lights, "cuda", SEED)
        val cudaOtherSeed = render(world, lights, "cuda", OTHER_SEED)
        val optix = render(world, lights, "optix", SEED)

        val noise = PpmSupport.diff(cuda, cudaOtherSeed, HIGH_DIFF_THRESHOLD)
        val actual = PpmSupport.diff(cuda, optix, HIGH_DIFF_THRESHOLD)
        val allowedMean = noise.meanAbsDiff * NOISE_TOLERANCE
        assertTrue(
            actual.meanAbsDiff <= allowedMean,
            "OptiX differs from the CUDA kernel by ${actual.meanAbsDiff} on average (max ${actual.maxDiff}); " +
                "two renders of the kernel differ by ${noise.meanAbsDiff}, so anything above $allowedMean is a real difference"
        )
        assertTrue(
            actual.highDiffCount <= maxOf(noise.highDiffCount * 2, MIN_HIGH_DIFF_CHANNELS),
            "${actual.highDiffCount} channels differ by >= $HIGH_DIFF_THRESHOLD (kernel-to-kernel noise produces ${noise.highDiffCount})"
        )
    }

    private fun render(world: Hittable, lights: Hittable?, backend: String, seed: Long): PpmSupport.PpmImage {
        val camera = Camera().apply {
            aspectRatio = 16.0f / 9.0f
            imageWidth = 96
            samplesPerPixel = 64
            maxReflectionDepth = 12
            vfov = 25.0f
            lookFrom = Point(6.0f, 2.0f, 6.0f)
            lookAt = Point(0.0f, 0.5f, 0.0f)
            vUp = Vector(0.0f, 1.0f, 0.0f)
            if (lights != null) background = Color(0.05f, 0.05f, 0.06f)
            this.lights = lights
        }
        val target = Files.createTempFile("raytracer-optix-test", ".ppm")
        val before = mapOf("rt.gpus" to System.getProperty("rt.gpus"), "rt.backend" to System.getProperty("rt.backend"))
        System.setProperty("rt.gpus", "one")
        System.setProperty("rt.backend", backend)
        RandomSource.withSeed(seed)
        return try {
            GpuRenderer().render(world, camera.initialize(), target)
            PpmSupport.parse(Files.readAllBytes(target))
        } finally {
            Files.deleteIfExists(target)
            Files.deleteIfExists(PpmWriter.pngPath(target))
            RandomSource.reset()
            before.forEach { (key, value) -> if (value == null) System.clearProperty(key) else System.setProperty(key, value) }
        }
    }

    private fun floor() = Quad(
        Point(-6.0f, 0.0f, -6.0f), Vector(12.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 12.0f),
        Lambertian(Color(0.6f, 0.6f, 0.62f))
    )

    private fun lamp() = Quad(
        Point(-1.5f, 3.0f, -1.5f), Vector(3.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 3.0f),
        DiffuseLight(Color(6.0f, 6.0f, 6.0f))
    )

    /** A unit octahedron standing on y = 0. */
    private fun octahedron(): Mesh = Mesh(
        floatArrayOf(
            1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f,
            0f, 1f, 0f, -1f, 0f, 0f, 0f, 0f, 1f,
            -1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, 1f,
            0f, -1f, 0f, 1f, 0f, 0f, 0f, 0f, 1f,
            0f, 1f, 0f, 1f, 0f, 0f, 0f, 0f, -1f,
            -1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, -1f,
            0f, -1f, 0f, -1f, 0f, 0f, 0f, 0f, -1f,
            1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, -1f
        )
    ).standingOnFloor()

    private fun assumeOptix() {
        val gpu = GpuRenderer.unavailabilityReason()
        assumeTrue(gpu == null, "No CUDA device available: $gpu")
        val optix = OptixApi.unavailabilityReason()
        assumeTrue(optix == null, "OptiX is not available: $optix")
    }

    private companion object {
        const val SEED = 4242L
        const val OTHER_SEED = 1337L
        const val HIGH_DIFF_THRESHOLD = 64
        const val NOISE_TOLERANCE = 1.5f
        const val MIN_HIGH_DIFF_CHANNELS = 8
    }
}
