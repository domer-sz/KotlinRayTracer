package rayTraceTypescript.scenes

import rayTraceTypescript.Camera
import rayTraceTypescript.Color
import rayTraceTypescript.Point
import rayTraceTypescript.Vector
import rayTraceTypescript.materials.Dielectric
import rayTraceTypescript.materials.DiffuseLight
import rayTraceTypescript.materials.Lambertian
import rayTraceTypescript.materials.Material
import rayTraceTypescript.objects.Hittable
import rayTraceTypescript.objects.HittableList
import rayTraceTypescript.objects.Quad
import rayTraceTypescript.objects.Sphere
import rayTraceTypescript.objects.Triangle
import rayTraceTypescript.objects.ConstantMedium
import rayTraceTypescript.optics.Spectrum
import rayTraceTypescript.optics.GlassCatalog
import rayTraceTypescript.objects.Lens
import rayTraceTypescript.objects.Prism
import rayTraceTypescript.objects.box
import rayTraceTypescript.optics.LensCatalog
import rayTraceTypescript.optics.ZemaxLens
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.tan

/**
 * Scenes for real lenses and glass that splits colours, laid out in millimetres (see
 * [rayTraceTypescript.objects.Lens]: rays start 0.001 units off a surface, so a lens has to be
 * many of them thick).
 */
object OpticsScenes {

    /**
     * An optical bench: sunlight, parallel along +x, through a black stop that leaves only the lens
     * open, onto a white screen. `-Drt.lens=<name>` picks the lens from the catalog (AC254-100-A
     * by default; `-Drt.lens.list` prints them all) or reads a Zemax file (`-Drt.lens=path.zmx`, an
     * aspheric collimator from its maker, say), `-Drt.lens.screen=<mm>` sets the screen's distance
     * behind the lens's last vertex - its catalog back focal length by default, or the file's image
     * surface, where the sun comes to a point. The spot is a caustic: `-Drt.caustics=on` gathers it.
     */
    fun bench(): SceneDefinition {
        if (System.getProperty("rt.lens.list") != null) {
            LensCatalog.all.values.sortedBy { it.name }.forEach { println(it) }
        }
        val chosen = System.getProperty("rt.lens") ?: "AC254-100-A"
        val axis = Vector(1f, 0f, 0f)
        val lenses: List<Lens>
        val focus: Float
        val title: String
        if (chosen.endsWith(".zmx", ignoreCase = true)) {
            val file = ZemaxLens.read(java.nio.file.Paths.get(chosen))
            lenses = file.build(Point(0f, 0f, 0f), axis)
            focus = file.imageAt
            title = file.toString()
        } else {
            val spec = LensCatalog.lens(chosen)
            lenses = listOf(spec.build(Point(0f, 0f, 0f), axis))
            focus = spec.length + spec.backFocalLength
            title = spec.toString()
        }
        val length = lenses.maxOf { it.vertex.x + it.vertices.last() }
        val behind = System.getProperty("rt.lens.screen")?.toFloat() ?: (focus - length)
        val screenX = length + behind
        val radius = lenses.first().semiDiameter
        val side = maxOf(4f * radius, 0.5f * behind)
        println("Bench: $title, screen ${"%.3f".format(behind)} mm behind it")

        // The sun: 0.27 degrees across, as the real one is, a few metres off.
        val sunDistance = 4000f
        val sun = Sphere(Point(-sunDistance, 0f, 0f), sunDistance * 0.00465f, DiffuseLight(Color(4000f, 3900f, 3700f)))
        val black = Lambertian(Color(0.02f, 0.02f, 0.02f))
        val world = HittableList(
            mutableListOf(
                sun,
                HittableList(lenses.toMutableList<rayTraceTypescript.objects.Hittable>()),
                // The stop in front of the lens, over its edge as a mount is, and the screen.
                annulus(Point(-minOf(2f, radius), 0f, 0f), axis, radius * 0.96f, side, black),
                Quad(Point(screenX, -side / 2f, side / 2f), Vector(0f, 0f, -side), Vector(0f, side, 0f), Lambertian(Color(0.8f, 0.8f, 0.8f))),
                // A dim floor under it all, so the bench is not floating in nothing.
                Quad(Point(-side, -side / 2f - 1f, -side), Vector(screenX + 2f * side, 0f, 0f), Vector(0f, 0f, 2f * side), Lambertian(Color(0.08f, 0.08f, 0.09f)))
            )
        )
        // -Drt.lens.closeup=on looks at the spot itself, 5 mm of the screen across.
        val closeup = System.getProperty("rt.lens.closeup")?.lowercase() in setOf("on", "true", "1", "")
        return SceneDefinition(
            name = "optical-bench",
            world = world,
            camera = Camera().apply {
                aspectRatio = 16f / 9f
                imageWidth = 1280
                samplesPerPixel = 1024
                maxReflectionDepth = 16
                if (closeup) {
                    vfov = 4.5f
                    lookFrom = Point(screenX - 50f, 12f, 25f)
                    lookAt = Point(screenX, 0f, 0f)
                } else {
                    vfov = 32f
                    lookFrom = Point(screenX * 0.35f, side * 0.45f, side * 1.35f)
                    lookAt = Point(screenX * 0.7f, 0f, 0f)
                }
                vUp = Vector(0f, 1f, 0f)
                background = Color(0f, 0f, 0f)
                lights = HittableList(mutableListOf(sun))
            },
            parts = lenses.mapIndexed { i, lens -> ScenePart(if (lenses.size == 1) "Soczewka" else "Soczewka ${i + 1}", lens) } +
                ScenePart("Słońce", sun, lightProxy = sun, isLight = true)
        )
    }

    /**
     * Newton's experiment: a narrow sheet of sunlight through a slit, onto a 60-degree prism of dense
     * flint (N-SF11, Abbe number 25.7; `-Drt.prism.glass` takes another) standing on a black block,
     * which spreads it into its colours across a white floor. The beam meets the prism at the angle of
     * least deviation for the glass's index at the d line ([Prism.minimumDeviationIncidence]) - the
     * sharpest spectrum - and leaves it falling as steeply as it came. The colours are a caustic:
     * give it `-Drt.caustics=on`.
     */
    fun prism(): SceneDefinition {
        val glassMaterial = Dielectric(GlassCatalog.glass(System.getProperty("rt.prism.glass") ?: "N-SF11"))
        val apex = Math.toRadians(60.0)
        val side = 30f
        val depth = 20f
        val baseY = 30f
        val prism = Prism.equilateral(Point(0f, baseY, 0f), side, depth, glassMaterial)
        // The left face's middle, where the beam is aimed, and the direction it comes in along: the
        // face's inward normal (at -apex/2 from +x) turned up by the incidence of least deviation.
        val faceMiddle = Point((-side * sin(apex / 2) / 2).toFloat(), baseY + (side * cos(apex / 2) / 2).toFloat(), 0f)
        val incoming = -apex / 2 + Prism.minimumDeviationIncidence(apex, glassMaterial.ri.toDouble())
        val beam = Vector(cos(incoming).toFloat(), sin(incoming).toFloat(), 0f)

        // The sun down along the beam, 4 m off; the slit in a black wall 35 mm before the prism, 2 mm
        // high and as long as the prism is deep, so the sheet of light falls on its face alone.
        // Ten times the bench's sun: through a 2 mm slit and spread over a spectrum, its light is thin.
        val sun = Sphere((faceMiddle - beam * 4000f).toPoint(), 4000f * 0.00465f, DiffuseLight(Color(40000f, 39000f, 37000f)))
        val wallX = -35f
        val slitY = faceMiddle.y - (faceMiddle.x - wallX) * beam.y / beam.x
        val slit = 2f
        val reach = depth / 2f - 1f
        val black = Lambertian(Color(0.02f, 0.02f, 0.02f))
        fun wallPiece(y0: Float, y1: Float, z0: Float, z1: Float) =
            Quad(Point(wallX, y0, z0), Vector(0f, 0f, z1 - z0), Vector(0f, y1 - y0, 0f), black)
        val wall = HittableList(
            mutableListOf(
                wallPiece(-150f, slitY - slit / 2f, -80f, 80f),
                wallPiece(slitY + slit / 2f, 150f, -80f, 80f),
                wallPiece(slitY - slit / 2f, slitY + slit / 2f, -80f, -reach),
                wallPiece(slitY - slit / 2f, slitY + slit / 2f, reach, 80f)
            )
        )
        val block = box(Point(-18f, 0f, -14f), Point(18f, baseY, 14f), black)
        val floor = Quad(Point(wallX, 0f, -80f), Vector(220f, 0f, 0f), Vector(0f, 0f, 160f), Lambertian(Color(0.8f, 0.8f, 0.8f)))
        return SceneDefinition(
            name = "prism",
            world = HittableList(mutableListOf(sun, prism, block, wall, floor)),
            camera = Camera().apply {
                aspectRatio = 16f / 9f
                imageWidth = 1280
                samplesPerPixel = 1024
                maxReflectionDepth = 16
                vfov = 40f
                lookFrom = Point(115f, 75f, 110f)
                lookAt = Point(32f, 22f, 0f)
                vUp = Vector(0f, 1f, 0f)
                // A faint sky, so the glass and the room show beside the sunbeam.
                background = Color(0.008f, 0.009f, 0.011f)
                lights = HittableList(mutableListOf(sun))
            },
            parts = listOf(ScenePart("Pryzmat", prism), ScenePart("Słońce", sun, lightProxy = sun, isLight = true))
        )
    }

    /**
     * Every kind of lens the renderer has, side by side on a white table in low sunlight: plano-convex
     * singlets (LA1131-A, f = 50; LA1608-A, f = 75), achromats (AC254-075-A, AC254-100-A, the
     * two-inch AC508-100-A), a diverging achromat (ACN254-100-A), Thorlabs' molded aspheric collimator
     * 354710-C - enlarged 8 times, at its true 1.5 mm it would not show - and a 60-degree prism of
     * N-SF11. Each lens is tilted along the sunlight and set at the height that brings its focus down
     * onto the table (its axis meets the table at its catalog back focal length), so each throws its
     * own caustic: the singlets' with a violet rim, the achromats' clean, the diverging lens a dark
     * shadow ringed with light, the prism a spectrum. Give it `-Drt.caustics=on`.
     */
    fun gallery(): SceneDefinition {
        val elevation = Math.toRadians(20.0)
        val sunward = Vector(cos(elevation).toFloat(), (-sin(elevation)).toFloat(), 0f)
        val black = Lambertian(Color(0.03f, 0.03f, 0.03f))
        val things = mutableListOf<rayTraceTypescript.objects.Hittable>()
        val parts = mutableListOf<ScenePart>()
        var z = 0f
        fun post(height: Float, atZ: Float) {
            if (height > 0.5f) things += box(Point(-1.5f, 0f, atZ - 1.5f), Point(1.5f, height, atZ + 1.5f), black)
        }
        for (name in listOf("LA1131-A", "LA1608-A", "AC254-075-A", "AC254-100-A", "ACN254-100-A", "AC508-100-A")) {
            val spec = LensCatalog.lens(name)
            val radius = spec.diameter / 2f
            z += radius + 14f
            // The centre high enough that the focus, length / 2 + fb along the axis, is on the table;
            // the diverging lens, which has none, at a height of its own.
            val toFocus = spec.length / 2f + spec.backFocalLength
            val height = if (toFocus > 0f) toFocus * sin(elevation).toFloat() else 30f
            val centre = Point(0f, height, z)
            val lens = spec.build((centre - sunward * (spec.length / 2f)).toPoint(), sunward)
            things += lens
            parts += ScenePart(name, lens)
            post(height - radius * cos(elevation).toFloat() - 1f, z)
            z += radius + 14f
        }
        // The collimator, 8 times its size.
        run {
            val file = ZemaxLens.read(collimatorFile())
            val scale = 8f
            val asphere = file.build(Point(0f, 0f, 0f), sunward, scale).first()
            val radius = asphere.semiDiameter
            z += radius + 14f
            val height = radius + 6f
            val centre = Point(0f, height, z)
            val lens = asphere.placed((centre - sunward * (asphere.vertices.last() / 2f)).toPoint(), sunward)
            things += lens
            parts += ScenePart("354710-C (x8)", lens)
            post(height - radius * cos(elevation).toFloat() - 1f, z)
            z += radius + 14f
        }
        // The prism stands on its end, ridge upright, turned so the sunlight crossing it horizontally
        // meets its first face near the angle of least deviation.
        run {
            val glass = Dielectric(GlassCatalog.glass("N-SF11"))
            val side = 30f
            val depth = 30f
            val apex = Math.toRadians(60.0)
            val incidence = Prism.minimumDeviationIncidence(apex, glass.ri.toDouble())
            // The apex-up cross-section of [Prism.standing], turned so the beam comes along +x.
            val turn = -(-apex / 2 + incidence)
            val h = side * cos(apex / 2)
            val r = side * sin(apex / 2)
            z += r.toFloat() + 20f
            val centreZ = z
            fun placed(x: Double, y: Double) = Point(
                (x * cos(turn) - y * sin(turn)).toFloat(), depth / 2f, centreZ + (x * sin(turn) + y * cos(turn)).toFloat()
            )
            val prism = Prism.of(placed(0.0, h * 2 / 3), placed(-r, -h / 3), placed(r, -h / 3), Vector(0f, -1f, 0f), depth, glass)
            things += prism
            parts += ScenePart("Pryzmat N-SF11", prism)
            z += r.toFloat()
        }
        val middle = z / 2f
        // Five times the bench's sun: at 20 degrees it lights the table at a third of its strength.
        val sun = Sphere((Point(0f, 0f, middle) - sunward * 4000f).toPoint(), 4000f * 0.00465f, DiffuseLight(Color(20000f, 19500f, 18500f)))
        val table = Quad(Point(-200f, 0f, -150f), Vector(600f, 0f, 0f), Vector(0f, 0f, z + 300f), Lambertian(Color(0.75f, 0.75f, 0.75f)))
        parts += ScenePart("Słońce", sun, lightProxy = sun, isLight = true)
        return SceneDefinition(
            name = "lens-gallery",
            world = HittableList((listOf(sun, table) + things).toMutableList()),
            camera = Camera().apply {
                aspectRatio = 16f / 9f
                imageWidth = 1600
                samplesPerPixel = 2048
                maxReflectionDepth = 16
                vfov = 42f
                lookFrom = Point(340f, 230f, middle - 210f)
                lookAt = Point(55f, 0f, middle - 5f)
                vUp = Vector(0f, 1f, 0f)
                background = Color(0.02f, 0.022f, 0.026f)
                lights = HittableList(mutableListOf(sun))
            },
            parts = parts
        )
    }

    /**
     * A laser, as a laser diode module is built: a tiny glowing source at the focus of Thorlabs'
     * collimating asphere 354710-C (4 times its size, f = 6 mm), in a black tube, sending a narrow,
     * nearly parallel beam across a dark, slightly smoky room onto a white wall. The beam's
     * divergence is the source's size over the focal length - 0.04 mm over 6 mm, 7 mrad - which is
     * the whole trick: a lens collimates a point, not a lamp. The light is one wavelength,
     * `-Drt.laser.nm` (650 by default; 532 for green), its colour the eye's for that wavelength
     * ([Spectrum]), and the glass is taken at its index there, so the beam does not split.
     *
     * The beam in the smoke and its spot are caustics - light from a small source through glass -
     * so it needs `-Drt.caustics=on`: photons stopped in the smoke past the lens are gathered there
     * as on surfaces. `-Drt.laser.fog=<per mm>` sets the haze (0.001), `-Drt.laser.clouds` the clouds in it.
     */
    fun laser(): SceneDefinition {
        val nm = System.getProperty("rt.laser.nm")?.toFloat() ?: 650f
        val microns = nm / 1000f
        val scale = 4f
        val beam = Vector(1f, 0f, 0f)
        val file = ZemaxLens.read(collimatorFile())
        // The file brings light coming in along its axis to a focus at its image surface: laid
        // along -x, the source sits at the focus and the beam leaves along +x. Each glass is fixed
        // at its index for the laser's wavelength.
        val lenses = file.build(Point(0f, 0f, 0f), -beam, scale).map { lens ->
            Lens(
                lens.vertex, lens.axis, lens.semiDiameter, lens.vertices, lens.curvatures,
                lens.media.map { m -> m?.let { Dielectric(it.glass?.n(microns) ?: it.ri) } },
                lens.conics, lens.aspherics
            )
        }
        val sourceX = -focusBehind(lenses, -beam, 0.5f * lenses.first().semiDiameter)
        val sourceRadius = System.getProperty("rt.laser.source")?.toFloat() ?: 0.02f
        val colour = Spectrum.weight(microns).let { c ->
            val m = maxOf(c.r, c.g, c.b)
            Color(maxOf(c.r, 0f) / m, maxOf(c.g, 0f) / m, maxOf(c.b, 0f) / m)
        }
        val targetKind = (System.getProperty("rt.laser.target") ?: "diamond").lowercase()
        val aimed = targetKind != "none"
        // Five times brighter with something in the beam, which splits it into weaker ones.
        val radiance = 1.0e7f * (System.getProperty("rt.laser.power")?.toFloat() ?: if (aimed) 5f else 1f)
        val source = Sphere(Point(sourceX, 0f, 0f), sourceRadius, DiffuseLight(Color(colour.r * radiance, colour.g * radiance, colour.b * radiance)))
        println("Laser: $nm nm, source ${"%.3f".format(-sourceX)} mm behind the lens, beam ${"%.1f".format(2f * lenses.first().semiDiameter)} mm, divergence ${"%.1f".format(2000f * sourceRadius / -sourceX)} mrad")

        // The module: a tube from behind the source to just past the lens, closed at the back, a
        // ring round the lens at the front. Inside it takes every bit of light: lit by a source
        // ten million times as bright as the room, even 3 % back off its walls made them a lamp
        // of their own that filled the opening, a wide cone round the beam.
        val absorbing = Lambertian(Color(0f, 0f, 0f))
        val black = Lambertian(Color(0.03f, 0.03f, 0.035f))
        val tubeRadius = lenses.first().semiDiameter + 3f
        // The body round it, as a module's: a grey metal barrel 60 mm long.
        val shell = tubeRadius + 4f
        val back = sourceX - 6f
        val front = 1f
        val bodyBack = back - 50f
        val grey = Lambertian(Color(0.18f, 0.18f, 0.2f))
        val module = HittableList(
            mutableListOf(
                tube(Point(back, 0f, 0f), beam, tubeRadius, front - back, absorbing),
                annulus(Point(back, 0f, 0f), beam, 0f, tubeRadius, absorbing),
                tube(Point(bodyBack, 0f, 0f), beam, shell, front - bodyBack, grey),
                annulus(Point(bodyBack, 0f, 0f), beam, 0f, shell, grey),
                annulus(Point(front, 0f, 0f), beam, lenses.first().semiDiameter, shell, black)
            )
        )
        val tableY = -shell - 1f
        val wallX = 700f
        val fogDensity = System.getProperty("rt.laser.fog")?.toFloat() ?: 0.001f
        val smoke = ConstantMedium(
            // The whole room, up to the ceiling: an edge of the smoke in sight showed on the wall.
            box(Point(bodyBack - 99f, tableY + 0.5f, -249f), Point(wallX - 0.5f, 415f, 249f), black),
            fogDensity, Color(0.9f, 0.9f, 0.9f)
        )
        // Clouds of thicker smoke along the beam, as a fog machine leaves it: balls of fog of their own
        // (`-Drt.laser.clouds=<n>`, 40; `-Drt.laser.seed`), where the beam lights up brighter.
        val cloudCount = System.getProperty("rt.laser.clouds")?.toInt() ?: 40
        val random = kotlin.random.Random(System.getProperty("rt.laser.seed")?.toInt() ?: 7)
        val clouds = List(cloudCount) {
            val radius = 15f + 40f * random.nextFloat()
            val centre = Point(
                front + radius + 10f + (wallX - 2f * radius - front - 20f) * random.nextFloat(),
                -15f + 40f * random.nextFloat(),
                -40f + 80f * random.nextFloat()
            )
            ConstantMedium(Sphere(centre, radius, black), fogDensity * (1.5f + 2.5f * random.nextFloat()), Color(0.9f, 0.9f, 0.9f))
        }
        val wall = Quad(Point(wallX, tableY, 250f), Vector(0f, 0f, -500f), Vector(0f, 420f - tableY, 0f), Lambertian(Color(0.8f, 0.8f, 0.8f)))
        // A wall along the far side too, to catch the beams a crystal throws.
        val sideWall = Quad(Point(bodyBack - 100f, tableY, -250f), Vector(wallX - bodyBack + 100f, 0f, 0f), Vector(0f, 420f - tableY, 0f), Lambertian(Color(0.7f, 0.7f, 0.7f)))
        val target = target(targetKind, tableY, black)
        val table = Quad(Point(bodyBack - 100f, tableY, -250f), Vector(wallX - bodyBack + 100f, 0f, 0f), Vector(0f, 0f, 500f), Lambertian(Color(0.12f, 0.12f, 0.13f)))
        // A dim panel in the ceiling, so the table, the module and the smoke show round the beam
        // (`-Drt.laser.room=<k>` scales it; 0 leaves the room dark).
        val room = System.getProperty("rt.laser.room")?.toFloat() ?: 0.3f
        val ceiling = Quad(Point(0f, 420f, -200f), Vector(wallX, 0f, 0f), Vector(0f, 0f, 400f), DiffuseLight(Color(0.4f * room, 0.4f * room, 0.45f * room)))
        return SceneDefinition(
            name = "laser",
            world = HittableList(
                (mutableListOf(source, HittableList(lenses.toMutableList<Hittable>()), module, smoke, wall, sideWall, table) + clouds + target +
                    (if (room > 0f) listOf(ceiling) else emptyList())).toMutableList()
            ),
            camera = Camera().apply {
                aspectRatio = 16f / 9f
                imageWidth = 1600
                samplesPerPixel = 1024
                maxReflectionDepth = 16
                // -Drt.camera.from=x,y,z, -Drt.camera.at=x,y,z, -Drt.camera.vfov=<degrees>, as in studio-model.
                fun point(property: String): Point? = System.getProperty(property)?.split(",")
                    ?.mapNotNull { it.trim().toFloatOrNull() }?.takeIf { it.size == 3 }?.let { Point(it[0], it[1], it[2]) }
                // The whole beam, or close by what it shines into, the far walls behind.
                vfov = System.getProperty("rt.camera.vfov")?.toFloatOrNull() ?: if (aimed) 50f else 40f
                lookFrom = point("rt.camera.from") ?: if (aimed) Point(90f, 110f, 230f) else Point(-170f, 75f, 210f)
                lookAt = point("rt.camera.at") ?: if (aimed) Point(330f, 10f, -60f) else Point(260f, 0f, -20f)
                vUp = Vector(0f, 1f, 0f)
                // A faint room light, so the module and the table show beside the beam.
                background = Color(0.004f, 0.004f, 0.005f)
                lights = HittableList((mutableListOf<Hittable>(source) + (if (room > 0f) listOf(ceiling) else emptyList())).toMutableList())
            },
            parts = lenses.mapIndexed { i, lens -> ScenePart("Kolimator ${i + 1}", lens) } +
                ScenePart("Źródło", source, lightProxy = source, isLight = true)
        )
    }

    /**
     * What the laser shines into, 250 mm out, at the beam's height (`-Drt.laser.target`): `diamond`
     * - a round brilliant 30 mm across ([MeshFill.BRILLIANT], index 2.42), its table turned 50
     * degrees towards the laser, which splits the beam into many through the smoke and throws
     * their spots on the walls; `ball` - a glass ball 30 mm across, which brings the beam to a
     * focus just behind it and spreads it into a cone (a ring of light on the wall); `prism` - the
     * 60-degree N-SF11 prism standing on end, bending the beam, which, being one wavelength, does
     * not split; `none` - the beam alone. Each on a thin black post.
     */
    private fun target(kind: String, tableY: Float, black: Material): List<Hittable> {
        val at = Point(250f, 0f, 0f)
        fun post(top: Float) = box(Point(at.x - 1.5f, tableY, at.z - 1.5f), Point(at.x + 1.5f, top, at.z + 1.5f), black)
        return when (kind.lowercase()) {
            "none" -> emptyList()
            "ball" -> listOf(Sphere(at, 15f, Dielectric(1.5f)), post(-15f))
            "prism" -> {
                val side = 30f
                val glass = Dielectric(GlassCatalog.glass("N-SF11").n(0.65f))
                val apex = Math.toRadians(60.0)
                val incidence = Prism.minimumDeviationIncidence(apex, glass.ri.toDouble())
                val turn = -(-apex / 2 + incidence)
                val h = side * cos(apex / 2)
                val r = side * sin(apex / 2)
                fun placed(x: Double, y: Double) = Point(
                    at.x + (x * cos(turn) - y * sin(turn)).toFloat(), tableY + 15f, at.z + (x * sin(turn) + y * cos(turn)).toFloat()
                )
                listOf(Prism.of(placed(0.0, h * 2 / 3), placed(-r, -h / 3), placed(r, -h / 3), Vector(0f, -1f, 0f), 30f, glass))
            }
            "diamond" -> {
                val gem = rayTraceTypescript.objects.MeshFill.BRILLIANT
                val size = 15f
                // The table (up, +y) turned towards the laser about z, then a little about y, so the
                // beam meets crown facets of several slopes.
                val tilt = Math.toRadians(50.0)
                val yaw = Math.toRadians(20.0)
                fun corner(i: Int): Point {
                    val c = gem.corners[i]
                    val x1 = c[0] * cos(tilt) - c[1] * sin(tilt)
                    val y1 = c[0] * sin(tilt) + c[1] * cos(tilt)
                    val z1 = c[2].toDouble()
                    val x2 = x1 * cos(yaw) + z1 * sin(yaw)
                    val z2 = -x1 * sin(yaw) + z1 * cos(yaw)
                    return Point(at.x + size * x2.toFloat(), at.y + size * y1.toFloat(), at.z + size * z2.toFloat())
                }
                val diamond = Dielectric(2.42f)
                val faces = ArrayList<Hittable>()
                for (f in gem.faces.indices step 3) faces += Triangle(corner(gem.faces[f]), corner(gem.faces[f + 1]), corner(gem.faces[f + 2]), diamond)
                listOf(HittableList(faces), post(-size * 0.8f))
            }
            else -> throw IllegalArgumentException("-Drt.laser.target: diamond, ball, prism or none, not $kind")
        }
    }

    /**
     * How far behind [lenses]' first vertex (along [axis], the way light goes through them) a ray
     * coming in parallel to the axis at [height] crosses it, refracted surface by surface at each
     * glass's index - where a source sends light back out parallel.
     */
    private fun focusBehind(lenses: List<Lens>, axis: Vector, height: Float): Float {
        val a = axis.unit()
        val side = if (kotlin.math.abs(a.y) < 0.9f) Vector.cross(a, Vector(0f, 1f, 0f)).unit() else Vector(1f, 0f, 0f)
        val start = lenses.first().vertex
        var ray = rayTraceTypescript.Ray((start + side * height - a * 5f).toPoint(), a)
        while (true) {
            val hits = lenses.mapNotNull { lens ->
                lens.hit(ray, rayTraceTypescript.Interval(0.0001f, Float.POSITIVE_INFINITY))?.let { lens to it }
            }
            val (lens, hit) = hits.minByOrNull { it.second.t } ?: break
            val (before, after) = lens.sides(hit.u.toInt())
            fun n(m: Dielectric?) = m?.ri ?: 1f
            val ratio = if (hit.frontFace) n(before) / n(after) else n(after) / n(before)
            ray = rayTraceTypescript.Ray(hit.point, Vector.refract(ray.direction.unit(), hit.normal, ratio))
        }
        // Where it meets the axis: its distance from the axis falls to zero.
        val o = ray.origin - start
        val d = ray.direction.unit()
        val off = Vector.dotProduct(o, side)
        val t = -off / Vector.dotProduct(d, side)
        return Vector.dotProduct(o + d * t, a)
    }

    /** An open cylinder of triangles about [axis] from [start], [length] long. */
    fun tube(start: Point, axis: Vector, radius: Float, length: Float, material: Material, segments: Int = 64): Hittable {
        val w = axis.unit()
        val helper = if (kotlin.math.abs(w.x) > 0.9f) Vector(0f, 1f, 0f) else Vector(1f, 0f, 0f)
        val u = Vector.cross(w, helper).unit()
        val v = Vector.cross(w, u)
        fun at(along: Float, k: Int): Point {
            val angle = 2.0 * Math.PI * k / segments
            return (start + w * along + u * (radius * cos(angle).toFloat()) + v * (radius * sin(angle).toFloat())).toPoint()
        }
        val faces = ArrayList<Hittable>()
        for (k in 0 until segments) {
            val a = at(0f, k); val b = at(0f, k + 1); val c = at(length, k + 1); val d = at(length, k)
            faces += Triangle(a, b, c, material)
            faces += Triangle(a, c, d, material)
        }
        return HittableList(faces)
    }

    /** Thorlabs' 354710-C, as the tests keep it, or -Drt.collimator=<file.zmx>. */
    private fun collimatorFile(): java.nio.file.Path {
        System.getProperty("rt.collimator")?.let { return java.nio.file.Paths.get(it) }
        val local = java.nio.file.Paths.get("src/test/resources/optics/354710-C.zmx")
        require(java.nio.file.Files.exists(local)) { "354710-C.zmx not found: run from the repository root or give -Drt.collimator=<file.zmx>" }
        return local
    }

    /** A flat ring of triangles about [axis] at [centre], between [inner] and [outer] radii. */
    fun annulus(centre: Point, axis: Vector, inner: Float, outer: Float, material: Material, segments: Int = 96): Hittable {
        val w = axis.unit()
        val helper = if (kotlin.math.abs(w.x) > 0.9f) Vector(0f, 1f, 0f) else Vector(1f, 0f, 0f)
        val u = Vector.cross(w, helper).unit()
        val v = Vector.cross(w, u)
        fun at(r: Float, k: Int): Point {
            val angle = 2.0 * Math.PI * k / segments
            return (centre + u * (r * cos(angle).toFloat()) + v * (r * sin(angle).toFloat())).toPoint()
        }
        val faces = ArrayList<Hittable>()
        for (k in 0 until segments) {
            val a = at(inner, k); val b = at(outer, k); val c = at(outer, k + 1); val d = at(inner, k + 1)
            faces += Triangle(a, b, c, material)
            faces += Triangle(a, c, d, material)
        }
        return HittableList(faces)
    }
}
