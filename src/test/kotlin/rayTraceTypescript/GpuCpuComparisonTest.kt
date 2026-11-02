package rayTraceTypescript

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import rayTraceTypescript.gpu.GpuRenderer
import rayTraceTypescript.gpu.MeshBlock
import rayTraceTypescript.gpu.PlacedMesh
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.DiffuseLight
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Metal
import rayTraceTypescript.objects.BvhNode
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.ConstantMedium
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.objects.RotateY
import rayTraceTypescript.objects.Translate
import rayTraceTypescript.objects.box
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.render.PpmWriter
import rayTraceTypescript.render.Renderer
import rayTraceTypescript.textures.CheckerTexture
import rayTraceTypescript.textures.ImageTexture
import rayTraceTypescript.textures.NoiseTexture
import rayTraceTypescript.textures.RtwImage
import rayTraceTypescript.utils.RandomSource
import java.awt.image.BufferedImage
import java.nio.file.Files
import java.nio.file.Path
import javax.imageio.ImageIO

/**
 * The GPU render must look like the CPU render. The CPU path tracer - the ground truth, a
 * straight copy of the books - rendered every scene here at two seeds before it was removed, and
 * those renders are stored in src/test/resources/cpu-reference. The GPU traces the same scene
 * with the same sample count; only the random streams differ, so the images are compared
 * statistically.
 */
class GpuCpuComparisonTest {

    @Test
    fun `gpu matches the cpu render for every material`() {
        assumeGpu()
        assertLooksTheSame("materials", materialsScene())
    }

    @Test
    fun `gpu matches the cpu render when samples are aimed at the lights`() {
        assumeGpu()
        val light = Quad(
            Point(-1.5f, 3.0f, -1.5f), Vector(3.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 3.0f),
            DiffuseLight(Color(6.0f, 6.0f, 6.0f))
        )
        val world = HittableList(
            mutableListOf(
                Sphere(Point(0.0f, -1000.0f, 0.0f), 1000.0f, Lambertian(Color(0.6f, 0.6f, 0.62f))),
                Sphere(Point(0.0f, 0.8f, 0.0f), 0.8f, Lambertian(Color(0.75f, 0.35f, 0.2f))),
                light
            )
        )
        assertLooksTheSame("aimed-at-lights", world, lit = true, lights = light)
    }

    @Test
    fun `gpu matches the cpu render for a volume`() {
        assumeGpu()
        assertLooksTheSame("volume", fogScene(), lit = true)
    }

    @Test
    fun `gpu matches the cpu render for rotated and translated instances`() {
        assumeGpu()
        assertLooksTheSame("instances", instanceScene(), lit = true)
    }

    @Test
    fun `gpu matches the cpu render for an emissive scene`() {
        assumeGpu()
        assertLooksTheSame("emissive", lightScene(), lit = true)
    }

    @Test
    fun `gpu matches the cpu render for quads`() {
        assumeGpu()
        assertLooksTheSame("quads", quadScene())
    }

    @Test
    fun `gpu matches the cpu render for perlin noise`() {
        assumeGpu()
        assertLooksTheSame("perlin", noiseScene())
    }

    @Test
    fun `gpu matches the cpu render for an stl mesh`() {
        assumeGpu()
        assertLooksTheSame("stl-mesh", meshScene())
    }

    @Test
    fun `gpu matches the cpu render for placed mesh blocks`() {
        assumeGpu()
        // One prebuilt tree used three times - moved, turned and scaled - grafted into the
        // scene's tree on the GPU, and triangles moved the ordinary way on the CPU.
        val block = MeshBlock(octahedron())
        val ground = Sphere(Point(0.0f, -1000.0f, 0.0f), 1000.0f, Lambertian(Color(0.6f, 0.6f, 0.62f)))
        val world = HittableList(
            mutableListOf(
                ground,
                PlacedMesh(block, 1.0f, 0.0f, Vector(-1.4f, 0.0f, 0.2f), Lambertian(Color(0.72f, 0.38f, 0.22f))),
                PlacedMesh(block, 0.6f, 37.0f, Vector(0.3f, 0.0f, 0.9f), Metal(Color(0.7f, 0.7f, 0.75f), 0.1f)),
                PlacedMesh(block, 1.4f, -110.0f, Vector(1.2f, 0.3f, -0.8f), Lambertian(Color(0.2f, 0.4f, 0.7f)))
            )
        )
        assertLooksTheSame("placed-blocks", world)
    }

    @Test
    fun `gpu matches the cpu render for placed mesh blocks lit by a lamp`() {
        assumeGpu()
        val block = MeshBlock(octahedron())
        val light = Quad(
            Point(-1.5f, 3.0f, -1.5f), Vector(3.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 3.0f),
            DiffuseLight(Color(6.0f, 6.0f, 6.0f))
        )
        val world = HittableList(
            mutableListOf(
                Sphere(Point(0.0f, -1000.0f, 0.0f), 1000.0f, Lambertian(Color(0.6f, 0.6f, 0.62f))),
                PlacedMesh(block, 1.0f, 45.0f, Vector(-0.8f, 0.0f, 0.2f), Lambertian(Color(0.72f, 0.38f, 0.22f))),
                PlacedMesh(block, 0.7f, 10.0f, Vector(0.9f, 0.0f, -0.3f), Lambertian(Color(0.2f, 0.4f, 0.7f))),
                light
            )
        )
        assertLooksTheSame("placed-blocks-lit", world, lit = true, lights = light)
    }

    @Test
    fun `gpu matches the cpu render for a lone placed mesh block`() {
        assumeGpu()
        val world = PlacedMesh(MeshBlock(octahedron()), 1.2f, 20.0f, Vector(0.0f, 0.0f, 0.0f), Lambertian(Color(0.72f, 0.38f, 0.22f)))
        assertLooksTheSame("lone-placed-block", world)
    }

    @Test
    fun `gpu matches the cpu render for an image texture`() {
        assumeGpu()
        assertLooksTheSame("image-texture", imageTextureScene())
    }

    /**
     * Two path traced images of one scene never match pixel for pixel - the sampling noise
     * differs. The yardstick is therefore the noise itself: how far the GPU image sits from a
     * CPU image must be no worse than how far two CPU images with different seeds sit apart.
     */
    private fun assertLooksTheSame(name: String, world: Hittable, lit: Boolean = false, lights: Hittable? = null) {
        val camera = testCamera().apply {
            if (lit) background = Color(0f, 0f, 0f)
            this.lights = lights
        }
        val cpu = cpuReference(name, world, camera, SEED)
        val cpuOtherSeed = cpuReference(name, world, camera, OTHER_SEED)
        val gpu = render(GpuRenderer(), world, camera, SEED)

        val noise = PpmSupport.diff(cpu, cpuOtherSeed, HIGH_DIFF_THRESHOLD)
        val actual = PpmSupport.diff(cpu, gpu, HIGH_DIFF_THRESHOLD)
        val allowedMean = noise.meanAbsDiff * NOISE_TOLERANCE

        assertTrue(
            actual.meanAbsDiff <= allowedMean,
            "GPU differs from the CPU render by ${actual.meanAbsDiff} on average (max ${actual.maxDiff}); " +
                "two CPU renders differ by ${noise.meanAbsDiff}, so anything above $allowedMean is a real difference"
        )
        assertTrue(
            actual.highDiffCount <= maxOf(noise.highDiffCount * 2, MIN_HIGH_DIFF_CHANNELS),
            "${actual.highDiffCount} channels differ by >= $HIGH_DIFF_THRESHOLD " +
                "(CPU-to-CPU noise produces ${noise.highDiffCount})"
        )
    }

    /**
     * The CPU render of scene [name] at [seed], as stored in src/test/resources/cpu-reference.
     * They were rendered by the CPU path tracer, the ground truth, before it was removed; they
     * cannot be made again, so a new scene needs references from elsewhere.
     */
    private fun cpuReference(name: String, world: Hittable, camera: Camera, seed: Long): PpmSupport.PpmImage {
        val resource = "/cpu-reference/$name-$seed.ppm.gz"
        val stored = requireNotNull(javaClass.getResourceAsStream(resource)) { "Missing CPU reference $resource" }
        return PpmSupport.parse(java.util.zip.GZIPInputStream(stored).use { it.readBytes() })
    }

    private fun render(renderer: Renderer, world: Hittable, camera: Camera, seed: Long): PpmSupport.PpmImage {
        val target: Path = Files.createTempFile("raytracer-backend-test", ".ppm")
        RandomSource.withSeed(seed)
        return try {
            renderer.render(world, camera.initialize(), target)
            PpmSupport.parse(Files.readAllBytes(target))
        } finally {
            Files.deleteIfExists(target)
            Files.deleteIfExists(PpmWriter.pngPath(target))
            RandomSource.reset()
        }
    }

    private fun testCamera() = Camera().apply {
        aspectRatio = 16.0f / 9.0f
        imageWidth = 96
        samplesPerPixel = 64
        maxReflectionDepth = 12
        vfov = 25.0f
        lookFrom = Point(6.0f, 2.0f, 6.0f)
        lookAt = Point(0.0f, 0.5f, 0.0f)
        vUp = Vector(0.0f, 1.0f, 0.0f)
        defocusAngle = 0.4f
        focusDistance = 8.0f
    }

    /** Checker ground, a moving Lambertian sphere, a fuzzy Metal and a Dielectric, behind a BVH. */
    private fun materialsScene(): Hittable {
        val ground = Sphere(
            Point(0.0f, -1000.0f, 0.0f),
            1000.0f,
            Lambertian(CheckerTexture(0.32f, Color(0.2f, 0.3f, 0.1f), Color(0.9f, 0.9f, 0.9f)))
        )
        val moving = Sphere(
            Point(-1.6f, 0.5f, 0.0f),
            Point(-1.6f, 0.9f, 0.0f),
            0.5f,
            Lambertian(Color(0.8f, 0.3f, 0.3f))
        )
        val metal = Sphere(Point(0.0f, 0.6f, 0.0f), 0.6f, Metal(Color(0.7f, 0.6f, 0.5f), 0.15f))
        val glass = Sphere(Point(1.6f, 0.5f, 0.4f), 0.5f, Dielectric(1.5f))
        return HittableList(mutableListOf(BvhNode(HittableList(mutableListOf(ground, moving, metal, glass)))))
    }

    /** A block of fog, which the kernel resolves with a nested traversal of its boundary. */
    private fun fogScene(): Hittable = HittableList(
        mutableListOf(
            Sphere(Point(0.0f, -1000.0f, 0.0f), 1000.0f, Lambertian(Color(0.6f, 0.6f, 0.62f))),
            ConstantMedium(
                box(Point(-0.8f, 0.0f, -0.8f), Point(0.8f, 1.6f, 0.8f), Lambertian(Color(1f, 1f, 1f))),
                1.2f,
                Color(0.9f, 0.9f, 0.9f)
            ),
            Quad(
                Point(-1.5f, 3.0f, -1.5f), Vector(3.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 3.0f),
                DiffuseLight(Color(6.0f, 6.0f, 6.0f))
            )
        )
    )

    /**
     * A block spun and moved by the instance wrappers. The CPU transforms rays at traversal
     * time and the GPU bakes the same transform into the quads, so this is where the two
     * approaches have to agree.
     */
    private fun instanceScene(): Hittable = HittableList(
        mutableListOf(
            Sphere(Point(0.0f, -1000.0f, 0.0f), 1000.0f, Lambertian(Color(0.6f, 0.6f, 0.62f))),
            Translate(
                RotateY(box(Point(0f, 0f, 0f), Point(1.2f, 1.2f, 1.2f), Lambertian(Color(0.75f, 0.35f, 0.2f))), 27.0f),
                Vector(-0.4f, 0.0f, -0.3f)
            ),
            Quad(
                Point(-1.5f, 3.0f, -1.5f), Vector(3.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 3.0f),
                DiffuseLight(Color(6.0f, 6.0f, 6.0f))
            )
        )
    )

    /** A dark room where the only light is an emissive quad above a diffuse sphere. */
    private fun lightScene(): Hittable = HittableList(
        mutableListOf(
            Sphere(Point(0.0f, -1000.0f, 0.0f), 1000.0f, Lambertian(Color(0.6f, 0.6f, 0.62f))),
            Sphere(Point(0.0f, 0.8f, 0.0f), 0.8f, Lambertian(Color(0.75f, 0.35f, 0.2f))),
            Quad(
                Point(-1.0f, 2.6f, -1.0f), Vector(2.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 2.0f),
                DiffuseLight(Color(6.0f, 6.0f, 6.0f))
            )
        )
    )

    /**
     * A quad floor with an upright quad standing on it, both seen at a glancing angle.
     * The floor sits at y = 0.2 rather than y = 0 on purpose: a checker cell boundary runs
     * through y = 0, and there the pattern's parity is decided by the last bit of the hit
     * point, which two renderers have no reason to agree on.
     */
    private fun quadScene(): Hittable = HittableList(
        mutableListOf(
            Quad(
                Point(-6.0f, 0.2f, -6.0f), Vector(12.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 12.0f),
                Lambertian(CheckerTexture(0.5f, Color(0.2f, 0.3f, 0.1f), Color(0.9f, 0.9f, 0.9f)))
            ),
            Quad(
                Point(-1.0f, 0.2f, 0.0f), Vector(2.0f, 0.0f, 0.0f), Vector(0.0f, 2.0f, 0.0f),
                Lambertian(Color(0.8f, 0.3f, 0.2f))
            ),
            Sphere(Point(1.6f, 0.6f, 1.2f), 0.6f, Metal(Color(0.8f, 0.8f, 0.85f), 0.1f))
        )
    )

    /** The marble texture, which the GPU reads from the same lattice the CPU built. */
    private fun noiseScene(): Hittable {
        // Perlin draws its tables when built: a pinned seed makes them match the stored CPU render.
        RandomSource.withSeed(SCENE_SEED)
        val marble = try {
            NoiseTexture(4.0f)
        } finally {
            RandomSource.reset()
        }
        return HittableList(
            mutableListOf(
                Sphere(Point(0.0f, -1000.0f, 0.0f), 1000.0f, Lambertian(marble)),
                Sphere(Point(0.0f, 1.0f, 0.0f), 1.0f, Lambertian(marble))
            )
        )
    }

    /** An octahedron standing on the checkered floor, the default model placement. */
    private fun meshScene(): Hittable {
        val ground = Sphere(
            Point(0.0f, -1000.0f, 0.0f),
            1000.0f,
            Lambertian(CheckerTexture(0.32f, Color(0.2f, 0.3f, 0.1f), Color(0.9f, 0.9f, 0.9f)))
        )
        val facets = octahedron().toHittables(Lambertian(Color(0.72f, 0.38f, 0.22f)))
        return HittableList(mutableListOf(BvhNode(HittableList((listOf(ground) + facets).toMutableList()))))
    }

    /** A unit octahedron standing on y = 0, the default model placement. */
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

    private fun imageTextureScene(): Hittable {
        val globe = Sphere(Point(0.0f, 0.5f, 0.0f), 1.2f, Lambertian(ImageTexture(syntheticImage())))
        return HittableList(mutableListOf(globe))
    }

    /** A tiny generated texture; the repository ships no image assets. */
    private fun syntheticImage(): RtwImage {
        val file = Files.createTempFile("raytracer-texture", ".png")
        val image = BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB)
        for (y in 0 until 8) {
            for (x in 0 until 8) {
                val rgb = if ((x + y) % 2 == 0) 0xC03040 else 0x2050B0
                image.setRGB(x, y, rgb)
            }
        }
        ImageIO.write(image, "png", file.toFile())
        return try {
            RtwImage().apply { check(load(file.toString())) { "Could not load the generated texture" } }
        } finally {
            Files.deleteIfExists(file)
        }
    }

    private fun assumeGpu() {
        val reason = GpuRenderer.unavailabilityReason()
        assumeTrue(reason == null, "No CUDA device available: $reason")
    }

    companion object {
        private const val SEED = 4242L
        private const val OTHER_SEED = 1337L
        /** Seeds the random parts of a scene itself (Perlin tables), so stored renders stay valid. */
        private const val SCENE_SEED = 77L

        private const val HIGH_DIFF_THRESHOLD = 64
        /** Headroom over the CPU-to-CPU noise floor, for float differences between devices. */
        private const val NOISE_TOLERANCE = 1.5f
        private const val MIN_HIGH_DIFF_CHANNELS = 8
    }
}
