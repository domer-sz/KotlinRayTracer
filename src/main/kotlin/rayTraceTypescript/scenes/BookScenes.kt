package rayTraceTypescript.scenes

import rayTraceTypescript.Camera
import rayTraceTypescript.Color
import rayTraceTypescript.Point
import rayTraceTypescript.Vector
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.DiffuseLight
import rayTraceTypescript.materials.Metal
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Material
import rayTraceTypescript.objects.BvhNode
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.Mesh
import rayTraceTypescript.objects.MeshFill
import rayTraceTypescript.objects.MeshLoader
import rayTraceTypescript.objects.ConstantMedium
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.objects.RotateY
import rayTraceTypescript.objects.Translate
import rayTraceTypescript.objects.box
import rayTraceTypescript.objects.checkeredBox
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.randomWorld
import rayTraceTypescript.utils.randomFloat
import rayTraceTypescript.textures.CheckerTexture
import rayTraceTypescript.textures.ImageTexture
import rayTraceTypescript.textures.NoiseTexture

/**
 * The scenes from the books, each built the way its chapter describes. Pick one with
 * `-Drt.scene=<name>`; [byName] also accepts the numbers the earlier scene switch used.
 */
object BookScenes {

    private val scenes: Map<String, () -> SceneDefinition> = linkedMapOf(
        "bouncing-spheres" to { bouncingSpheres() },
        "earth" to { earth() },
        "perlin-spheres" to { perlinSpheres() },
        "quads" to { quads() },
        "simple-light" to { simpleLight() },
        "cornell-box" to { cornellBox() },
        "cornell-blocks" to { cornellBlocks() },
        "cornell-smoke" to { cornellSmoke() },
        "final-week" to { finalWeek() },
        "cornell-final" to { cornellFinal() },
        "cornell-model" to { cornellModel() },
        "showcase" to { showcase() },
        "model" to { model() },
        "studio-model" to { studioModel() },
        "file" to { fileScene() }
    )

    val names: List<String> get() = scenes.keys.toList()

    fun byName(name: String): SceneDefinition {
        val key = name.trim().lowercase()
        val builder = scenes[key]
            ?: throw IllegalArgumentException("Unknown scene '$name'; available: ${names.joinToString()}")
        return builder()
    }

    /** The Next Week, chapter 2: the One Weekend finale with motion blur and a checkered floor. */
    fun bouncingSpheres() = SceneDefinition(
        name = "bouncing-spheres",
        world = HittableList(mutableListOf(BvhNode(randomWorld()))),
        camera = Camera().apply {
            aspectRatio = 16.0f / 9.0f
            imageWidth = 500
            samplesPerPixel = 90
            maxReflectionDepth = 20
            vfov = 20.0f
            lookFrom = Point(13.0f, 2.0f, 3.0f)
            lookAt = Point(0.0f, 0.0f, 0.0f)
            vUp = Vector(0.0f, 1.0f, 0.0f)
            defocusAngle = 0.6f
            focusDistance = 10.0f
        }
    )

    /** The Next Week, chapter 5: two spheres sharing one marble texture built from turbulence. */
    fun perlinSpheres(): SceneDefinition {
        val marble = NoiseTexture(4.0f)
        return SceneDefinition(
            name = "perlin-spheres",
            world = HittableList(
                mutableListOf(
                    Sphere(Point(0f, -1000f, 0f), 1000f, Lambertian(marble)),
                    Sphere(Point(0f, 2f, 0f), 2f, Lambertian(marble))
                )
            ),
            camera = Camera().apply {
                aspectRatio = 16.0f / 9.0f
                imageWidth = 400
                samplesPerPixel = 100
                maxReflectionDepth = 50
                vfov = 20.0f
                lookFrom = Point(13f, 2f, 3f)
                lookAt = Point(0f, 0f, 0f)
                vUp = Vector(0f, 1f, 0f)
                defocusAngle = 0f
            }
        )
    }

    /** The Next Week, chapter 6: five quads seen from inside the box they half enclose. */
    fun quads(): SceneDefinition {
        val red = Lambertian(Color(1.0f, 0.2f, 0.2f))
        val green = Lambertian(Color(0.2f, 1.0f, 0.2f))
        val blue = Lambertian(Color(0.2f, 0.2f, 1.0f))
        val orange = Lambertian(Color(1.0f, 0.5f, 0.0f))
        val teal = Lambertian(Color(0.2f, 0.8f, 0.8f))

        return SceneDefinition(
            name = "quads",
            world = HittableList(
                mutableListOf(
                    Quad(Point(-3f, -2f, 5f), Vector(0f, 0f, -4f), Vector(0f, 4f, 0f), red),
                    Quad(Point(-2f, -2f, 0f), Vector(4f, 0f, 0f), Vector(0f, 4f, 0f), green),
                    Quad(Point(3f, -2f, 1f), Vector(0f, 0f, 4f), Vector(0f, 4f, 0f), blue),
                    Quad(Point(-2f, 3f, 1f), Vector(4f, 0f, 0f), Vector(0f, 0f, 4f), orange),
                    Quad(Point(-2f, -3f, 5f), Vector(4f, 0f, 0f), Vector(0f, 0f, -4f), teal)
                )
            ),
            camera = Camera().apply {
                aspectRatio = 1.0f
                imageWidth = 400
                samplesPerPixel = 100
                maxReflectionDepth = 50
                vfov = 80.0f
                lookFrom = Point(0f, 0f, 9f)
                lookAt = Point(0f, 0f, 0f)
                vUp = Vector(0f, 1f, 0f)
                defocusAngle = 0f
            }
        )
    }

    /** The Next Week, chapter 7: the first scene lit by its own geometry rather than the sky. */
    fun simpleLight(): SceneDefinition {
        val marble = Lambertian(NoiseTexture(4.0f))
        val light = DiffuseLight(Color(4f, 4f, 4f))     // brighter than white, so it lights the room
        return SceneDefinition(
            name = "simple-light",
            world = HittableList(
                mutableListOf(
                    Sphere(Point(0f, -1000f, 0f), 1000f, marble),
                    Sphere(Point(0f, 2f, 0f), 2f, marble),
                    Sphere(Point(0f, 7f, 0f), 2f, light),
                    Quad(Point(3f, 1f, 0f), Vector(2f, 0f, 0f), Vector(0f, 2f, 0f), light)
                )
            ),
            camera = Camera().apply {
                aspectRatio = 16.0f / 9.0f
                imageWidth = 400
                samplesPerPixel = 100
                maxReflectionDepth = 50
                background = Color(0f, 0f, 0f)
                vfov = 20.0f
                lookFrom = Point(26f, 3f, 6f)
                lookAt = Point(0f, 2f, 0f)
                vUp = Vector(0f, 1f, 0f)
                defocusAngle = 0f
            }
        )
    }

    /** The Next Week, chapter 7.4: the Cornell box, empty for now. */
    fun cornellBox(
        contents: List<Hittable> = emptyList(),
        samples: Int = 200,
        width: Int = 600
    ): SceneDefinition {
        val red = Lambertian(Color(0.65f, 0.05f, 0.05f))
        val white = Lambertian(Color(0.73f, 0.73f, 0.73f))
        val green = Lambertian(Color(0.12f, 0.45f, 0.15f))
        val light = DiffuseLight(Color(15f, 15f, 15f))

        val lamp = Quad(Point(343f, 554f, 332f), Vector(-130f, 0f, 0f), Vector(0f, 0f, -105f), light)
        // Named as the camera sees them: it looks down +z, so x = 555 is on the left.
        val left = Quad(Point(555f, 0f, 0f), Vector(0f, 555f, 0f), Vector(0f, 0f, 555f), green)
        val right = Quad(Point(0f, 0f, 0f), Vector(0f, 555f, 0f), Vector(0f, 0f, 555f), red)
        val floor = Quad(Point(0f, 0f, 0f), Vector(555f, 0f, 0f), Vector(0f, 0f, 555f), white)
        val ceiling = Quad(Point(555f, 555f, 555f), Vector(-555f, 0f, 0f), Vector(0f, 0f, -555f), white)
        val back = Quad(Point(0f, 0f, 555f), Vector(555f, 0f, 0f), Vector(0f, 555f, 0f), white)
        val walls = mutableListOf<Hittable>(left, right, lamp, floor, ceiling, back)
        walls.addAll(contents)

        return SceneDefinition(
            name = "cornell-box",
            parts = listOf(
                ScenePart("Lampa sufitowa", lamp, isLight = true),
                ScenePart("Ściana lewa", left, isSurface = true),
                ScenePart("Ściana prawa", right, isSurface = true),
                ScenePart("Ściana tylna", back, isSurface = true),
                ScenePart("Podłoga", floor, isSurface = true),
                ScenePart("Sufit", ceiling, isSurface = true)
            ),
            world = HittableList(walls),
            camera = Camera().apply {
                aspectRatio = 1.0f
                imageWidth = width
                samplesPerPixel = samples
                maxReflectionDepth = 50
                background = Color(0f, 0f, 0f)
                vfov = 40.0f
                lookFrom = Point(278f, 278f, -800f)
                lookAt = Point(278f, 278f, 0f)
                vUp = Vector(0f, 1f, 0f)
                defocusAngle = 0f
            }
        )
    }

    /** The Next Week, chapter 8.2: the Cornell box with two rotated, translated blocks. */
    fun cornellBlocks(samples: Int = 200, width: Int = 600): SceneDefinition {
        val white = Lambertian(Color(0.73f, 0.73f, 0.73f))
        val tall = Translate(
            RotateY(box(Point(0f, 0f, 0f), Point(165f, 330f, 165f), white), 15.0f),
            Vector(265f, 0f, 295f)
        )
        val short = Translate(
            RotateY(box(Point(0f, 0f, 0f), Point(165f, 165f, 165f), white), -18.0f),
            Vector(130f, 0f, 65f)
        )
        return SceneDefinition(
            name = "cornell-blocks",
            world = cornellBox(listOf(tall, short), samples, width).world,
            camera = cornellBox(listOf(), samples, width).camera
        )
    }

    /** The Next Week, chapter 9.2: the same room, with the blocks replaced by smoke. */
    fun cornellSmoke(samples: Int = 200, width: Int = 600): SceneDefinition {
        val red = Lambertian(Color(0.65f, 0.05f, 0.05f))
        val white = Lambertian(Color(0.73f, 0.73f, 0.73f))
        val green = Lambertian(Color(0.12f, 0.45f, 0.15f))
        val light = DiffuseLight(Color(7f, 7f, 7f))

        val tall = Translate(
            RotateY(box(Point(0f, 0f, 0f), Point(165f, 330f, 165f), white), 15.0f),
            Vector(265f, 0f, 295f)
        )
        val short = Translate(
            RotateY(box(Point(0f, 0f, 0f), Point(165f, 165f, 165f), white), -18.0f),
            Vector(130f, 0f, 65f)
        )

        return SceneDefinition(
            name = "cornell-smoke",
            world = HittableList(
                mutableListOf(
                    Quad(Point(555f, 0f, 0f), Vector(0f, 555f, 0f), Vector(0f, 0f, 555f), green),
                    Quad(Point(0f, 0f, 0f), Vector(0f, 555f, 0f), Vector(0f, 0f, 555f), red),
                    // A wider, dimmer lamp than the solid box scene uses.
                    Quad(Point(113f, 554f, 127f), Vector(330f, 0f, 0f), Vector(0f, 0f, 305f), light),
                    Quad(Point(0f, 555f, 0f), Vector(555f, 0f, 0f), Vector(0f, 0f, 555f), white),
                    Quad(Point(0f, 0f, 0f), Vector(555f, 0f, 0f), Vector(0f, 0f, 555f), white),
                    Quad(Point(0f, 0f, 555f), Vector(555f, 0f, 0f), Vector(0f, 555f, 0f), white),
                    ConstantMedium(tall, 0.01f, Color(0f, 0f, 0f)),
                    ConstantMedium(short, 0.01f, Color(1f, 1f, 1f))
                )
            ),
            camera = cornellBox(listOf(), samples, width).camera
        )
    }

    /**
     * The Next Week, chapter 10: every feature of the book in one room - a floor of boxes of
     * random height, a moving sphere, glass, metal, a glass ball filled with blue fog, a thin
     * white haze over the whole scene, the earth, marble, and a block of a thousand spheres
     * rotated into place.
     *
     * The book renders this at 10000 samples; the default here is what a machine can finish in
     * a sitting. Without ./earthmap.jpg (which this repository does not carry) the globe falls
     * back to the cyan ImageTexture uses for a missing file.
     */
    fun finalWeek(samples: Int = 400, width: Int = 600): SceneDefinition {
        val ground = Lambertian(Color(0.48f, 0.83f, 0.53f))
        val floorBoxes = mutableListOf<Hittable>()
        val boxesPerSide = 20
        for (i in 0 until boxesPerSide) {
            for (j in 0 until boxesPerSide) {
                val w = 100.0f
                val x0 = -1000.0f + i * w
                val z0 = -1000.0f + j * w
                val y1 = randomFloat(1.0f, 101.0f)
                floorBoxes.add(box(Point(x0, 0.0f, z0), Point(x0 + w, y1, z0 + w), ground))
            }
        }

        val spheresInBlock = mutableListOf<Hittable>()
        val white = Lambertian(Color(0.73f, 0.73f, 0.73f))
        repeat(1000) {
            spheresInBlock.add(Sphere(Vector.random(0.0f, 165.0f).toPoint(), 10.0f, white))
        }

        val blueFogBoundary = Sphere(Point(360f, 150f, 145f), 70f, Dielectric(1.5f))
        val hazeBoundary = Sphere(Point(0f, 0f, 0f), 5000f, Dielectric(1.5f))

        val world = HittableList(
            mutableListOf(
                BvhNode(HittableList(floorBoxes)),
                Quad(Point(123f, 554f, 147f), Vector(300f, 0f, 0f), Vector(0f, 0f, 265f), DiffuseLight(Color(7f, 7f, 7f))),
                Sphere(Point(400f, 400f, 200f), Point(430f, 400f, 200f), 50f, Lambertian(Color(0.7f, 0.3f, 0.1f))),
                Sphere(Point(260f, 150f, 45f), 50f, Dielectric(1.5f)),
                Sphere(Point(0f, 150f, 145f), 50f, Metal(Color(0.8f, 0.8f, 0.9f), 1.0f)),
                blueFogBoundary,
                ConstantMedium(blueFogBoundary, 0.2f, Color(0.2f, 0.4f, 0.9f)),
                ConstantMedium(hazeBoundary, 0.0001f, Color(1f, 1f, 1f)),
                Sphere(Point(400f, 200f, 400f), 100f, Lambertian(ImageTexture("earthmap.jpg"))),
                Sphere(Point(220f, 280f, 300f), 80f, Lambertian(NoiseTexture(0.2f))),
                Translate(RotateY(BvhNode(HittableList(spheresInBlock)), 15.0f), Vector(-100f, 270f, 395f))
            )
        )

        return SceneDefinition(
            name = "final-week",
            world = world,
            camera = Camera().apply {
                aspectRatio = 1.0f
                imageWidth = width
                samplesPerPixel = samples
                maxReflectionDepth = 40
                background = Color(0f, 0f, 0f)
                vfov = 40.0f
                lookFrom = Point(478f, 278f, -600f)
                lookAt = Point(278f, 278f, 0f)
                vUp = Vector(0f, 1f, 0f)
                defocusAngle = 0f
            }
        )
    }

    /**
     * The Rest of Your Life, chapter 13: the Cornell box with the tall block and a glass
     * sphere, and both the lamp and that sphere named as things worth aiming samples at.
     *
     * The book renders it at 1000 samples. Turning the lights off (-Drt.lights=off) leaves
     * exactly the same scene sampled the way The Next Week did it, which is the comparison the
     * whole book is about.
     *
     * -Drt.block=checker covers the tall block in a black and white checkerboard.
     */
    fun cornellFinal(samples: Int = 200, width: Int = 600): SceneDefinition {
        val white = Lambertian(Color(0.73f, 0.73f, 0.73f))
        val glass = Dielectric(1.5f)
        val block = if (System.getProperty("rt.block") == "checker") {
            // 27.5 units a square: 6 across the block, 12 up it.
            checkeredBox(Point(0f, 0f, 0f), Point(165f, 330f, 165f), 27.5f, Color(0.73f, 0.73f, 0.73f), Color(0.03f, 0.03f, 0.03f))
        } else {
            box(Point(0f, 0f, 0f), Point(165f, 330f, 165f), white)
        }

        val tall = Translate(
            RotateY(block, 15.0f),
            Vector(265f, 0f, 295f)
        )
        val glassSphere = Sphere(Point(190f, 90f, 190f), 90f, glass)

        val scene = cornellBox(listOf(tall, glassSphere), samples, width)
        val world = scene.world as HittableList

        // The material here is never asked for: these stand in only to be sampled towards.
        val lampProxy = Quad(Point(343f, 554f, 332f), Vector(-130f, 0f, 0f), Vector(0f, 0f, -105f), white)
        val sphereProxy = Sphere(Point(190f, 90f, 190f), 90f, white)
        val lights = mutableListOf<Hittable>(lampProxy, sphereProxy)

        // The lamp hangs above the room and slightly behind what stands in it, which leaves
        // fronts dark. -Drt.fill=<f> switches on a fill light like a photographer's: a panel
        // outside the box, above the camera's view (see fillLight). A viewer can switch it on
        // either way, at DEFAULT_FILL when no -Drt.fill is given.
        val fillStrength = System.getProperty("rt.fill")?.toFloatOrNull() ?: 0.0f
        val fill = fillLight(if (fillStrength > 0.0f) fillStrength else DEFAULT_FILL)
        if (fillStrength > 0.0f) {
            world.add(fill)
            lights.add(fill)
        }

        val lamp = scene.parts.first()
        return SceneDefinition(
            name = "cornell-final",
            world = world,
            camera = scene.camera.apply { this.lights = HittableList(lights) },
            parts = listOf(
                ScenePart(lamp.name, lamp.hittable, lampProxy, isLight = true),
                ScenePart("Światło przednie", fill, fill, isLight = true, enabled = fillStrength > 0.0f),
                ScenePart("Wysoki blok", tall),
                ScenePart("Szklana kula", glassSphere, sphereProxy)
            ) + scene.parts.filter { it.isSurface }
        )
    }

    /**
     * Not from the books: [cornellFinal] with a model standing on the floor beside the glass
     * sphere, in front of the tall block. The path comes from -Drt.model=<path> or MODEL_FILE;
     * the model is taken to be z-up, as files from 3D-printing tools are, and -Drt.model.turn
     * spins it about the vertical axis (degrees) to face the camera.
     */
    fun cornellModel(
        path: String = System.getProperty("rt.model") ?: System.getenv("MODEL_FILE") ?: "model.stl",
        turnDegrees: Float = System.getProperty("rt.model.turn")?.toFloatOrNull() ?: 0.0f,
        samples: Int = 200,
        width: Int = 600
    ): SceneDefinition {
        val scene = cornellFinal(samples, width)
        // -Drt.model.material=glass makes the model glass, like the sphere; =grey matte grey.
        // Without it a file that colours its faces (an OBJ with a .mtl) keeps its colours.
        val chosen = System.getProperty("rt.model.material")
        val material: Material = when (chosen) {
            "glass" -> Dielectric(1.5f)
            else -> Lambertian(Color(0.45f, 0.45f, 0.45f))
        }
        val loaded = MeshLoader.load(path)
        // Files for 3D printing (STL, 3MF) stand on z; the other formats use y as up.
        val zUp = path.substringAfterLast('.').lowercase() in setOf("stl", "3mf")
        val mesh = (if (zUp) loaded.zUpToYUp() else loaded)
            .rotatedY(turnDegrees)
            .standingOnFloor(targetHeight = MODEL_HEIGHT)
            .transformed(1.0f, Vector(0f, 0f, 0f), Vector(360f, 0f, 170f))
        println("Loaded ${mesh.triangleCount} triangles from $path")

        // The fill light (-Drt.fill) comes with cornellFinal and is aimed at this model's face.
        val world = scene.world as HittableList
        val faces = if (chosen == null) mesh.toColouredHittables(material) else mesh.toHittables(material)
        val model = filling(mesh, material) ?: BvhNode(HittableList(faces.toMutableList()))
        world.add(model)
        val modelName = java.io.File(path).nameWithoutExtension
        return SceneDefinition(name = "cornell-model", world = world, camera = scene.camera, parts = scene.parts + ScenePart(modelName, model))
    }

    /**
     * A 150x150 panel centred at (330, 560, -200) - out of frame, since the camera at z=-800
     * sees only up to y~480 there - facing the model's face at about (360, 150, 170). Its
     * brightness is set so that it lights that point with [strength] times the light the lamp
     * gives it, going by area, distance and the panels' angles to it.
     */
    /**
     * What -Drt.infill puts in place of [mesh] (threads or pyramids packed into it, the
     * mesh itself left out; see [MeshFill]), in [material] or -Drt.infill.color; null without it.
     */
    fun filling(mesh: Mesh, material: Material): Hittable? {
        val settings = MeshFill.Settings.fromProperties() ?: return null
        val chosen = MeshFill.Settings.materialOverride() ?: material
        val filling = MeshFill.fill(mesh, chosen, settings)
        println(filling.summary)
        filling.mesh?.let { return rayTraceTypescript.gpu.PlacedMesh(rayTraceTypescript.gpu.MeshBlock(it), 1f, 0f, Vector(0f, 0f, 0f), chosen) }
        // A tree over the spheres for the CPU's own hit tests (the viewer's picking); the GPU builds its own.
        return BvhNode(HittableList(filling.spheres!!.toMutableList<Hittable>()))
    }

    private fun fillLight(strength: Float): Quad {
        val target = Vector(360f, 150f, 170f)
        val centre = Vector(330f, 560f, -200f)
        val towards = (target - centre).unit()
        // Edges chosen so u x v points along [towards]: u across the room, v in the y/z plane.
        val u = Vector(150f, 0f, 0f)
        val v = Vector(0f, towards.z, -towards.y).unit() * 150f
        val corner = centre - u * 0.5f - v * 0.5f

        fun reach(position: Vector, normal: Vector, area: Float): Float {
            val toTarget = target - position
            val distance = toTarget.length()
            return area * kotlin.math.abs(Vector.dotProduct(normal, toTarget)) / (distance * distance * distance)
        }
        val lamp = reach(Vector(278f, 554f, 279.5f), Vector(0f, -1f, 0f), 130f * 105f)
        val panel = reach(centre, towards, 150f * 150f)
        val radiance = 15f * strength * lamp / panel
        return Quad(corner.toPoint(), u, v, DiffuseLight(Color(radiance, radiance, radiance)))
    }

    /** Height the model in [cornellModel] is scaled to: a little over half the tall block. */
    private const val MODEL_HEIGHT = 190.0f

    /** Fill light strength a viewer switches on when no -Drt.fill set one. */
    private const val DEFAULT_FILL = 0.4f

    /**
     * Not from the books: one widescreen room holding what the two books added - quads, a
     * marble sphere built from Perlin turbulence, glass, a block of fog and a metal block both
     * spun into place - lit by a quad the renderer aims its samples at.
     *
     * The floor is a huge sphere rather than a quad on purpose: a checkered plane lying exactly
     * on a cell boundary leaves the pattern's parity to the last bit of the hit point.
     */
    fun showcase(samples: Int = 500, width: Int = 800): SceneDefinition {
        val white = Lambertian(Color(0.73f, 0.73f, 0.73f))
        val lamp = Quad(
            Point(-2.2f, 5.4f, -2.2f), Vector(4.4f, 0f, 0f), Vector(0f, 0f, 4.4f),
            DiffuseLight(Color(6f, 6f, 6f))
        )

        val fogBlock = ConstantMedium(
            Translate(
                RotateY(box(Point(0f, 0f, 0f), Point(1.5f, 2.6f, 1.5f), white), 22.0f),
                Vector(-3.4f, 0.01f, -0.6f)
            ),
            0.9f,
            Color(0.85f, 0.87f, 0.95f)
        )
        val metalBlock = Translate(
            RotateY(box(Point(0f, 0f, 0f), Point(1.4f, 1.4f, 1.4f), Metal(Color(0.8f, 0.75f, 0.55f), 0.05f)), -17.0f),
            Vector(2.4f, 0.01f, -1.2f)
        )

        return SceneDefinition(
            name = "showcase",
            world = HittableList(
                mutableListOf(
                    Sphere(
                        Point(0f, -1000f, 0f), 1000f,
                        Lambertian(CheckerTexture(0.7f, Color(0.16f, 0.25f, 0.12f), Color(0.86f, 0.86f, 0.82f)))
                    ),
                    Quad(Point(-7f, 0f, -4.2f), Vector(14f, 0f, 0f), Vector(0f, 6.5f, 0f), white),
                    Sphere(Point(-1.1f, 1.15f, 0.9f), 1.15f, Lambertian(NoiseTexture(2.6f))),
                    Sphere(Point(1.5f, 1.0f, 1.6f), 1.0f, Dielectric(1.5f)),
                    Sphere(Point(-3.3f, 0.55f, 1.9f), 0.55f, Lambertian(Color(0.75f, 0.3f, 0.2f))),
                    fogBlock,
                    metalBlock,
                    lamp
                )
            ),
            camera = Camera().apply {
                aspectRatio = 16.0f / 9.0f
                imageWidth = width
                samplesPerPixel = samples
                maxReflectionDepth = 30
                background = Color(0.02f, 0.02f, 0.03f)
                vfov = 32.0f
                lookFrom = Point(0.6f, 3.2f, 9.5f)
                lookAt = Point(-0.2f, 1.3f, 0f)
                vUp = Vector(0f, 1f, 0f)
                defocusAngle = 0.3f
                focusDistance = 9.0f
                lights = lamp
            }
        )
    }

    /** The Next Week, chapter 4: an image texture wrapped around a globe. */
    fun earth() = SceneDefinition(
        name = "earth",
        world = HittableList(mutableListOf(Sphere(Point(0f, 0f, 0f), 2f, Lambertian(ImageTexture("earthmap.jpg"))))),
        camera = Camera().apply {
            aspectRatio = 16.0f / 9.0f
            imageWidth = 400
            samplesPerPixel = 100
            maxReflectionDepth = 50
            vfov = 20f
            lookFrom = Point(0f, 0f, 12f)
            lookAt = Point(0f, 0f, 0f)
            vUp = Vector(0f, 1f, 0f)
            defocusAngle = 0f
        }
    )

    /**
     * Not from the books: a model file standing on the checkered floor. The path comes from
     * -Drt.model=<path>, the MODEL_FILE environment variable, or ./model.stl.
     */
    fun model(
        path: String = System.getProperty("rt.model") ?: System.getenv("MODEL_FILE") ?: "model.stl",
        material: Material = Lambertian(Color(0.72f, 0.38f, 0.22f))
    ) = SceneDefinition(
        name = "model",
        world = modelWorld(path, material),
        camera = Camera().apply {
            aspectRatio = 16.0f / 9.0f
            imageWidth = 800
            samplesPerPixel = 150
            maxReflectionDepth = 20
            vfov = 24.0f
            lookFrom = Point(3.2f, 2.2f, 6.0f)
            lookAt = Point(0.0f, 0.9f, 0.0f)
            vUp = Vector(0.0f, 1.0f, 0.0f)
            defocusAngle = 0.2f
            focusDistance = 6.8f
        }
    )

    /**
     * The default model setup: the checkered ground of [bouncingSpheres], with the mesh scaled
     * to a usable size and resting on it. Every facet goes through a BVH, otherwise a mesh of
     * any size would be traversed one triangle at a time.
     */
    fun modelWorld(
        path: String,
        material: Material = Lambertian(Color(0.72f, 0.38f, 0.22f))
    ): HittableList {
        val checker = CheckerTexture(.32F, Color(.2, .3, .1), Color(.9, .9, .9))
        val objects: MutableList<Hittable> = mutableListOf(
            Sphere(Point(0.0f, -1000.0f, 0.0f), 1000.0F, Lambertian(checker))
        )
        val mesh = MeshLoader.load(path).standingOnFloor()
        println("Loaded ${mesh.triangleCount} triangles from $path")
        val filling = filling(mesh, material)
        if (filling != null) objects.add(filling) else objects.addAll(mesh.toHittables(material))
        return HittableList(mutableListOf(BvhNode(HittableList(objects))))
    }

    /**
     * Not from the books: a port of StlViewer's "gold-reel" studio, so the same model can be
     * timed and compared against that browser renderer. A gold figure on a glossy black floor in
     * front of a navy wall, three softbox lamps (key, fill, rim) aimed at the figure's chest,
     * a flat dark backdrop, a 1080x1920 portrait frame. The figure is scaled so its bounding
     * box has a half diagonal of 1 / 0.6162 (height 1.6226), which is what the lamps and the
     * camera there are laid out around. Meant for z-up STL files: -Drt.model=<path>.
     * -Drt.studio.turn=<degrees> spins the figure about the vertical axis; -Drt.studio.height=<h> sizes it otherwise.
     * -Drt.studio.fog=<density> fills the studio with grey fog, lit by the same lamps (0.05 is a
     * haze); -Drt.studio.fog.radius=<r> keeps it to a ball of that radius round the figure instead,
     * whose edge shows. It is part of the room, so a turntable turns the figure inside it.
     */
    fun studioModel(): SceneDefinition {
        val path = System.getProperty("rt.model") ?: System.getenv("MODEL_FILE") ?: "model.stl"
        val turn = System.getProperty("rt.studio.turn")?.toFloatOrNull() ?: 0.0f
        // -Drt.studio.height sizes a figure the camera is not framed for, such as a block as wide as it is tall.
        val height = System.getProperty("rt.studio.height")?.toFloatOrNull()?.takeIf { it > 0f } ?: STUDIO_HEIGHT
        // -0.4 EV on the emitters (lamps and backdrop): everything else only reflects them.
        val exposure = Math.pow(2.0, -0.4).toFloat()

        // -Drt.studio.material=glass|diamond makes the figure clear (index 1.5, or diamond's 2.42) instead of gold.
        val gold: Material = when (val chosen = System.getProperty("rt.studio.material")?.trim()?.lowercase()) {
            null, "", "gold" -> Metal(Color(0.9f, 0.7f, 0.3f), 0.05f)
            "glass" -> Dielectric(1.5f)
            "diamond" -> Dielectric(2.42f)
            else -> throw IllegalArgumentException("-Drt.studio.material=$chosen: expected gold, glass or diamond")
        }
        val mesh = MeshLoader.load(path).zUpToYUp().rotatedY(turn).standingOnFloor(targetHeight = height)
        println("Loaded ${mesh.triangleCount} triangles from $path")
        // A placed mesh: its tree is built once, so turning it (the viewer, a turntable film)
        // copies that tree in instead of building it again.
        // -Drt.infill: what fills the figure stands in for it (MeshFill).
        val figure: Hittable = filling(mesh, gold)
            ?: rayTraceTypescript.gpu.PlacedMesh(rayTraceTypescript.gpu.MeshBlock(mesh), 1f, 0f, Vector(0f, 0f, 0f), gold)

        // Floor, wall, lamps and camera exactly as StlViewer dumped them for this figure at
        // angle 0 (renders/yenn-scene-dump.json there): the camera stands 45 degrees round the
        // figure, and the wall and lamps are laid out round it.
        // -Drt.studio.floor=matte[:r,g,b] lays a matte floor (light grey by default) instead of the
        // black mirror, so caustics have somewhere to land.
        val floorSetting = System.getProperty("rt.studio.floor")?.trim()?.lowercase()
        val floorMaterial: Material = when {
            floorSetting == null || floorSetting == "mirror" -> Metal(Color(0.22f, 0.22f, 0.25f), 0.004f)
            floorSetting.startsWith("matte") -> {
                val c = floorSetting.substringAfter(':', "").split(',').mapNotNull { it.trim().toFloatOrNull() }
                Lambertian(if (c.size == 3) Color(c[0], c[1], c[2]) else Color(0.75f, 0.75f, 0.75f))
            }
            else -> throw IllegalArgumentException("-Drt.studio.floor=$floorSetting: expected matte[:r,g,b] or mirror")
        }
        val floor = Quad(
            Point(-48.6785f, 0f, -48.6785f), Vector(0f, 0f, 97.3571f), Vector(97.3571f, 0f, 0f),
            floorMaterial
        )
        val wall = Quad(
            Point(37.863f, 64.9047f, -53.9261f), Vector(-91.7891f, 0f, 91.7891f), Vector(0f, -64.9047f, 0f),
            Lambertian(Color(0.009696f, 0.038473f, 0.173439f))
        )
        // -Drt.studio.softboxes=<k> dims (or brightens) the three softboxes, e.g. under a lamp that should lead.
        val softboxes = System.getProperty("rt.studio.softboxes")?.toFloatOrNull()?.takeIf { it >= 0f } ?: 1f
        fun softbox(corner: Point, u: Vector, v: Vector, power: Float): Quad {
            val level = power * exposure * softboxes
            return Quad(corner, u, v, DiffuseLight(Color(level, level, level)))
        }
        val key = softbox(Point(-1.8395f, 4.4035f, -0.3628f), Vector(1.8774f, 0f, 2.2374f), Vector(-1.7139f, -1.8774f, 1.4382f), 8f)
        val fill = softbox(Point(2.855f, 3.3234f, 1.4604f), Vector(0f, 0f, -2.9207f), Vector(0.9989f, -2.7446f, 0f), 4f)
        val rim = softbox(Point(1.5331f, 3.2739f, -2.3362f), Vector(-1.6927f, 0f, -0.4536f), Vector(0.2268f, -1.5176f, -0.8464f), 12f)

        val target = Point(0.0011f, 0.7258f, 0.0002f)
        val eye = Point(2.3622f, 1.3017f, 2.3613f)
        val sky = Color(0.000805f, 0.002263f, 0.010398f) * exposure

        // -Drt.studio.wall=off takes the navy wall away: only the dark sky behind.
        val withWall = System.getProperty("rt.studio.wall")?.trim()?.lowercase() != "off"
        // With -Drt.studio.softboxes=0 they are left out altogether, not aimed at for nothing.
        val boxes = if (softboxes > 0f) listOf(key, fill, rim) else emptyList()
        val world = HittableList((listOf(floor) + (if (withWall) listOf(wall) else emptyList()) + boxes + figure).toMutableList())
        val lamps = boxes.toMutableList<Hittable>()
        // -Drt.studio.lamp=x,y,z[,irradiance[,radius]]: a small glowing ball, hard light for sharp
        // shadows and caustics. Its brightness is given as the light it casts on the figure's middle
        // (4 by default, about twice the key softbox), its radius 0.04 by default.
        System.getProperty("rt.studio.lamp")?.let { setting ->
            val n = setting.split(',').mapNotNull { it.trim().toFloatOrNull() }
            require(n.size >= 3) { "-Drt.studio.lamp=$setting: expected x,y,z[,irradiance[,radius]]" }
            val at = Point(n[0], n[1], n[2])
            val irradiance = n.getOrNull(3) ?: 4f
            val radius = n.getOrNull(4) ?: 0.04f
            val middle = Point(0f, height * 0.5f, 0f)
            val d2 = (at - middle).let { it.x * it.x + it.y * it.y + it.z * it.z }
            // A ball of radiance L, radius r, seen from d away, lights a facing point with L pi r^2 / d^2.
            val radiance = irradiance * d2 / (Math.PI.toFloat() * radius * radius)
            val lamp = Sphere(at, radius, DiffuseLight(Color(radiance, radiance, radiance)))
            world.add(lamp)
            lamps.add(lamp)
            // -Drt.studio.lamp.cone=<degrees>: a black shade round the lamp, open towards the figure,
            // so it lights a pool of that half-angle and the rest stays dark (a tube of 16 panels
            // and a back, its rim three lamp radii out).
            System.getProperty("rt.studio.lamp.cone")?.toFloatOrNull()?.takeIf { it in 1f..80f }?.let { cone ->
                val axis = (middle - at).unit()
                val side = Vector.cross(if (kotlin.math.abs(axis.y) < 0.9f) Vector(0f, 1f, 0f) else Vector(1f, 0f, 0f), axis).unit()
                val up = Vector.cross(axis, side)
                val rim = 3f * radius
                val ahead = rim / kotlin.math.tan(Math.toRadians(cone.toDouble())).toFloat()
                val back = (at - axis * (rim * 0.5f)).toPoint()
                val black = Lambertian(Color(0.02f, 0.02f, 0.02f))
                fun ring(k: Int): Point {
                    val phi = 2.0 * Math.PI * k / 16
                    return (back + side * (rim * kotlin.math.cos(phi).toFloat()) + up * (rim * kotlin.math.sin(phi).toFloat())).toPoint()
                }
                for (k in 0 until 16) world.add(Quad(ring(k), ring(k + 1) - ring(k), axis * (ahead + rim * 0.5f), black))
                world.add(Quad((back - side * rim - up * rim).toPoint(), side * (2f * rim), up * (2f * rim), black))
                println("  under a shade: a pool of %.0f degrees".format(cone))
            }
            println("Studio lamp at (%.2f, %.2f, %.2f), radius %.3f, %.1f on the figure".format(at.x, at.y, at.z, radius, irradiance))
        }
        System.getProperty("rt.studio.fog")?.toFloatOrNull()?.takeIf { it > 0f }?.let { density ->
            // A ball round the figure: by default one taking in the room, camera and lamps too, so it
            // has no edge in sight. Grey: it scatters every colour alike.
            val radius = System.getProperty("rt.studio.fog.radius")?.toFloatOrNull()?.takeIf { it > 0f } ?: STUDIO_ROOM_RADIUS
            val ball = Sphere(Point(0f, height * 0.5f, 0f), radius, Dielectric(1.5f))
            world.add(ConstantMedium(ball, density, Color(0.6f, 0.6f, 0.6f)))
            println("Grey fog round the figure, density $density, radius $radius")
        }
        return SceneDefinition(
            name = "studio-model",
            world = world,
            camera = Camera().apply {
                aspectRatio = 9.0f / 16.0f
                imageWidth = 1080
                samplesPerPixel = 64
                maxReflectionDepth = 8
                background = sky
                // -Drt.camera.from=x,y,z, -Drt.camera.at=x,y,z, -Drt.camera.vfov=<degrees> place the
                // camera elsewhere (focused on the point it looks at).
                fun point(property: String): Point? = System.getProperty(property)?.split(",")
                    ?.mapNotNull { it.trim().toFloatOrNull() }?.takeIf { it.size == 3 }?.let { Point(it[0], it[1], it[2]) }
                val from = point("rt.camera.from") ?: eye
                val at = point("rt.camera.at") ?: target
                vfov = System.getProperty("rt.camera.vfov")?.toFloatOrNull() ?: 45.0f
                lookFrom = from
                lookAt = at
                vUp = Vector(0f, 1f, 0f)
                // 0.4 degrees of aperture, focused on the figure's axis.
                defocusAngle = 0.4f
                focusDistance = if (from === eye && at === target) 3.3883f else (at - from).length()
                lights = HittableList(lamps)
                // StlViewer caps a sample at 16 before its exposure; here the exposure is in the
                // lamps already, so the same cap is 16 times as much.
                fireflyClamp = Camera.DEFAULT_FIREFLY_CLAMP * exposure
            },
            parts = listOf(ScenePart("Figura", figure))
        )
    }

    /**
     * Not from the books: a whole scene from one model file (-Drt.model), such as Amazon
     * Lumberyard Bistro exported to OBJ - at its authored size, in the colours its .mtl gives
     * (grey without them), under the sky gradient and a sun, the camera framing its box from
     * above one corner. A placed mesh, so OptiX builds its tree on the card and the CPU never
     * makes one. -Drt.model.zup stands a z-up file up; -Drt.sun=off leaves only the sky;
     * -Drt.sun=<x,y,z> points at the sun; -Drt.camera.from=<x,y,z>, -Drt.camera.at=<x,y,z> and
     * -Drt.camera.vfov=<degrees> place the camera instead (Bistro's own street-level camera:
     * from 24.82,3.16,-61.65 at 24.77,3.15,-61.5, 43 degrees).
     *
     * A `<model>.lights` file beside the model (what the Blender conversion writes from the FBX's
     * lights) adds them: `point x y z r g b watts` becomes a small glowing ball the renderer aims
     * samples at, `sun dx dy dz r g b strength` sets where the sun stands and its colour.
     * -Drt.lights.scale=<k> brightens or dims the point lights.
     *
     * -Drt.env=<file.hdr> lights it with an HDR map instead of the sky gradient (-Drt.env.intensity,
     * -Drt.env.turn=<degrees>), -Drt.sun.power=<k> makes the sun k times as bright. Its images are
     * tone mapped with ACES unless -Drt.tonemap says otherwise ([rayTraceTypescript.render.ToneMap]).
     */
    fun fileScene(): SceneDefinition {
        val path = System.getProperty("rt.model") ?: System.getenv("MODEL_FILE") ?: "model.stl"
        val loaded = MeshLoader.load(path)
        val mesh = if (System.getProperty("rt.model.zup") != null) loaded.zUpToYUp() else loaded
        println("Loaded ${mesh.triangleCount} triangles from $path${if (mesh.hasColours) ", ${mesh.palette.size} materials" else ""}")
        val block = rayTraceTypescript.gpu.MeshBlock(mesh)
        val palette = if (mesh.hasColours) mesh.palette.map { it.material() } else null
        val model = rayTraceTypescript.gpu.PlacedMesh(block, 1f, 0f, Vector(0f, 0f, 0f), Lambertian(Color(0.6f, 0.6f, 0.6f)), palette)

        val b = block.bounds
        val centre = Point((b.x.min + b.x.max) / 2, (b.y.min + b.y.max) / 2, (b.z.min + b.z.max) / 2)
        val size = Vector(b.x.max - b.x.min, b.y.max - b.y.min, b.z.max - b.z.min)
        val diagonal = size.length()

        val world = HittableList(mutableListOf<Hittable>(model))
        val lamps = mutableListOf<Hittable>()
        var fileSun: Pair<Vector, Color>? = null
        val sidecar = java.io.File(path.substringBeforeLast('.') + ".lights")
        if (sidecar.isFile) {
            val scale = System.getProperty("rt.lights.scale")?.toFloatOrNull() ?: 1f
            sidecar.readLines().map { it.trim().split(Regex("[ \\t]+")) }.forEach { p ->
                val n = p.drop(1).mapNotNull { it.toFloatOrNull() }
                if (n.size < 7) return@forEach
                when (p[0]) {
                    "point" -> {
                        // A ball LAMP_RADIUS across - or the eighth number's radius - as bright as a point
                        // light of that many watts: its intensity, watts / 4 pi, is its radiance times its
                        // disc, pi r^2. A larger ball softens shadows and caustics, and clears their noise.
                        val r = n.getOrNull(7)?.takeIf { it > 0f } ?: LAMP_RADIUS
                        val radiance = n[6] / (4f * Math.PI.toFloat()) * scale / (Math.PI.toFloat() * r * r)
                        lamps += Sphere(Point(n[0], n[1], n[2]), r, DiffuseLight(Color(n[3] * radiance, n[4] * radiance, n[5] * radiance)))
                    }
                    "sun" -> fileSun = Vector(n[0], n[1], n[2]) to Color(n[3], n[4], n[5])
                }
            }
            println("Lights from ${sidecar.name}: ${lamps.size} lamps${if (fileSun != null) " and the sun" else ""}")
        }
        lamps.forEach { world.add(it) }
        val sunSetting = System.getProperty("rt.sun")
        val sun = if (sunSetting == "off") null else {
            val toward = sunSetting?.split(",")?.mapNotNull { it.trim().toFloatOrNull() }?.takeIf { it.size == 3 }
                ?.let { Vector(it[0], it[1], it[2]) } ?: fileSun?.first ?: Vector(0.45f, 1.0f, 0.3f)
            val unit = toward * (1f / toward.length())
            // A disc 0.05 rad across at 4 diagonals: about three times the sky's light on a lit face,
            // in the file's colour when it gives one.
            val distance = diagonal * 4f
            val at = centre + unit * distance
            val tint = fileSun?.second?.let { c -> val top = maxOf(c.r, c.g, c.b, 1e-6f); Color(c.r / top, c.g / top, c.b / top) } ?: Color(1f, 0.92f, 0.82f)
            // -Drt.sun.size=<radians> narrows the disc for crisper shadows; its radiance rises with
            // the square of the shrink, so the light it casts stays the same.
            val size = System.getProperty("rt.sun.size")?.toFloatOrNull()?.takeIf { it > 0f } ?: 0.05f
            val power = 380f * (System.getProperty("rt.sun.power")?.toFloatOrNull() ?: 1f) * (0.05f / size) * (0.05f / size)
            Sphere(Point(at.x, at.y, at.z), distance * size, DiffuseLight(Color(power * tint.r, power * tint.g, power * tint.b)))
        }
        sun?.let { world.add(it) }
        val aimed = (listOfNotNull(sun) + lamps).takeIf { it.isNotEmpty() }
        val environment = System.getProperty("rt.env")?.let { file ->
            rayTraceTypescript.textures.Environment.load(java.nio.file.Paths.get(file)).with(
                intensity = System.getProperty("rt.env.intensity")?.toFloatOrNull() ?: 1f,
                turnDegrees = System.getProperty("rt.env.turn")?.toFloatOrNull() ?: 0f
            ).also { println("Environment ${java.io.File(file).name}: ${it.width}x${it.height}") }
        }
        // A whole scene from a game looks right through a filmic curve, as the game showed it.
        if (System.getProperty("rt.tonemap") == null) System.setProperty("rt.tonemap", "aces")

        val view = Vector(1.0f, 0.55f, 1.0f).let { it * (1f / it.length()) }
        fun point(property: String): Point? = System.getProperty(property)?.split(",")?.mapNotNull { it.trim().toFloatOrNull() }
            ?.takeIf { it.size == 3 }?.let { Point(it[0], it[1], it[2]) }
        val from = point("rt.camera.from") ?: (centre + view * (diagonal * 0.75f)).let { Point(it.x, it.y, it.z) }
        val at = point("rt.camera.at") ?: centre
        return SceneDefinition(
            name = "file",
            world = world,
            camera = Camera().apply {
                aspectRatio = 16.0f / 9.0f
                imageWidth = 1280
                samplesPerPixel = 64
                maxReflectionDepth = 8
                vfov = System.getProperty("rt.camera.vfov")?.toFloatOrNull() ?: 50.0f
                lookAt = at
                lookFrom = from
                vUp = Vector(0f, 1f, 0f)
                defocusAngle = 0f
                lights = aimed?.let { HittableList(it.toMutableList()) }
                this.environment = environment
            }
        )
    }

    /** How big a point light of a scene file is drawn, in its units (metres for Bistro and Sun Temple). */
    private const val LAMP_RADIUS = 0.1f

    /** A fog ball this big takes in the studio: the floor's half width is 48.7, the wall 50 away. */
    private const val STUDIO_ROOM_RADIUS = 60f

    /** Height of the studio figure, which fixes the scale everything else is laid out in. */
    private const val STUDIO_HEIGHT = 1.6226f
}
