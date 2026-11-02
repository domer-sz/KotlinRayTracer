package rayTraceTypescript.gui

import java.awt.Component
import java.awt.Dimension
import java.awt.GridBagConstraints
import java.awt.GridBagLayout
import java.awt.Insets
import java.io.File
import javax.swing.BorderFactory
import javax.swing.ButtonGroup
import javax.swing.DefaultListModel
import javax.swing.JButton
import javax.swing.JCheckBox
import javax.swing.JColorChooser
import javax.swing.JComboBox
import javax.swing.JFileChooser
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.JList
import javax.swing.JOptionPane
import javax.swing.JPanel
import javax.swing.JProgressBar
import javax.swing.JSeparator
import javax.swing.JToggleButton
import javax.swing.JScrollPane
import javax.swing.JSpinner
import javax.swing.ListSelectionModel
import javax.swing.SpinnerNumberModel
import javax.swing.filechooser.FileNameExtensionFilter

/**
 * The viewer's side panel: output size and sample target, denoising and saving, and the models
 * added to the scene - loading them, choosing one, and setting where it stands, how it is
 * turned, how tall it is and what it is made of.
 */
class ControlPanel(private val viewer: Viewer, private val frame: JFrame) : JPanel(GridBagLayout()) {
    private var row = 0
    /** Set while the panel fills its own fields, so that does not count as an edit. */
    private var syncing = false

    private val presets = listOf(
        "Jak scena" to viewer.sceneResolution,
        "640 × 360" to Viewer.Resolution(640, 360),
        "800 × 800" to Viewer.Resolution(800, 800),
        "1280 × 720" to Viewer.Resolution(1280, 720),
        "1920 × 1080" to Viewer.Resolution(1920, 1080),
        "2000 × 2000" to Viewer.Resolution(2000, 2000),
        "3840 × 2160" to Viewer.Resolution(3840, 2160)
    )
    private val preset = JComboBox(presets.map { it.first }.toTypedArray())
    private val width = JSpinner(SpinnerNumberModel(viewer.sceneResolution.width, 64, 8192, 16))
    private val height = JSpinner(SpinnerNumberModel(viewer.sceneResolution.height, 64, 8192, 16))
    private val applySize = JButton("Zastosuj rozdzielczość")
    private val target = JSpinner(SpinnerNumberModel(viewer.targetSamples, 1, 1_000_000, 64))
    private val depth = JSpinner(SpinnerNumberModel(viewer.maxDepth, 1, 500, 1)).apply {
        toolTipText = "Ile razy promień może się odbić lub załamać, zanim ścieżka się urwie. " +
            "Mniej: szybciej, ciemniej w zakamarkach i w szkle; więcej: pełniejsze światło odbite."
    }
    private val progress = JProgressBar(0, 1000).apply { isStringPainted = true }
    private val saveSceneButton = JButton("Zapisz…").apply { toolTipText = "Zapisz scenę do pliku .${SceneFile.EXTENSION} (Ctrl+S)" }
    private val loadSceneButton = JButton("Wczytaj…").apply { toolTipText = "Wczytaj scenę z pliku .${SceneFile.EXTENSION} (Ctrl+O)" }
    private var lastSceneFolder: File = File(".").absoluteFile
    private val quick = JToggleButton("Szybki podgląd", !viewer.raytrace)
    private val raytrace = JToggleButton("Ray tracing", viewer.raytrace)
    private val denoise = JCheckBox("Odszumianie (D)", viewer.denoiseOn)
    private val save = JButton("Zapisz obraz (S)").apply {
        // The one action most sessions end with: drawn in the accent colour.
        putClientProperty(
            "FlatLaf.style",
            "background: \$Button.default.background; foreground: \$Button.default.foreground; " +
                "borderColor: \$Button.default.borderColor; font: bold"
        )
    }

    private val lightSwitches = viewer.lights.map { light ->
        JCheckBox(light.part.name, light.enabled).apply {
            addActionListener { viewer.setLight(light, isSelected) }
        }
    }
    /** Each light's power, as a multiple of the strength the scene gave it. */
    private val lightPowers = viewer.lights.map { light ->
        JSpinner(SpinnerNumberModel(light.power, 0.05, 50.0, 0.1)).apply {
            editor = JSpinner.NumberEditor(this, "0.0#'×'")
            toolTipText = "Moc światła: wielokrotność mocy ze sceny (1× jak w scenie)"
            preferredSize = Dimension(84, preferredSize.height)
            addChangeListener { if (!syncing) viewer.setLightPower(light, value as Double) }
        }
    }
    private val restore = JButton("Przywróć obiekty sceny")

    private val listModel = DefaultListModel<SceneEntry>()
    private val list = JList(listModel).apply {
        selectionMode = ListSelectionModel.SINGLE_SELECTION
        visibleRowCount = 6
        fixedCellHeight = 24
        // Each row: the name, and what kind of thing it is in a quieter colour.
        cellRenderer = object : javax.swing.DefaultListCellRenderer() {
            override fun getListCellRendererComponent(
                list: JList<*>?, value: Any?, index: Int, isSelected: Boolean, cellHasFocus: Boolean
            ): Component {
                val kind = when (value) {
                    is WallObject -> "ściana"
                    is BuiltInObject -> "obiekt sceny"
                    is PlacedObject -> when (value.shape) {
                        is Shape.Model -> "model"
                        Shape.Ball -> "kula"
                        Shape.Block -> "graniastosłup"
                    }
                    else -> ""
                }
                val tint = Theme.hex(if (isSelected) javax.swing.UIManager.getColor("List.selectionForeground") ?: Theme.text else Theme.muted)
                val text = "<html>${escapeHtml(value.toString())}&nbsp;&nbsp;<span style='color:$tint; font-size:90%'>$kind</span></html>"
                return super.getListCellRendererComponent(list, text, index, isSelected, cellHasFocus).apply {
                    (this as javax.swing.JLabel).border = BorderFactory.createEmptyBorder(0, 8, 0, 8)
                }
            }
        }
    }

    private fun escapeHtml(s: String) = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    private val add = JButton("Dodaj model…")
    private val remove = JButton("Usuń")
    private val centre = JButton("Na środek widoku")
    private val step = viewer.defaultObjectHeight / 20
    private val posX = numberSpinner(step)
    private val posY = numberSpinner(step)
    private val posZ = numberSpinner(step)
    private val turn = JSpinner(SpinnerNumberModel(0.0, -360.0, 360.0, 15.0))
    private val size = JSpinner(SpinnerNumberModel(viewer.defaultObjectHeight, 1e-3, 1e6, step))
    private val addBall = JButton("Dodaj kulę")
    private val addBlock = JButton("Dodaj graniastosłup")
    private val blockWidth = JSpinner(SpinnerNumberModel(viewer.defaultObjectHeight / 2, 1e-3, 1e6, step))
    private val blockDepth = JSpinner(SpinnerNumberModel(viewer.defaultObjectHeight / 2, 1e-3, 1e6, step))
    private val materialPreset = JComboBox<Any>((listOf<Any>(CUSTOM, FILE_COLOURS) + MaterialPreset.values()).toTypedArray())
    private val materialKind = JComboBox(MaterialKind.values())
    private val colour = JButton("Wybierz kolor…")
    private val parameterLabel = JLabel("Parametr")
    private val parameter = JSpinner(SpinnerNumberModel(0.0, 0.0, 100.0, 0.05))
    private val zUp = JCheckBox("Oś Z w górę (pliki do druku 3D)")
    private val info = JLabel(" ").apply { foreground = Theme.muted }
    /** Fields for anything that can be moved: an added object or one of the scene's own. */
    private val placeFields: List<Component> = listOf(centre, posX, posY, posZ, turn)
    /** Fields only an added object has: its size and what it is made of. */
    private val modelFields: List<Component> = listOf(size, zUp)
    /** Fields for anything with a material: an added object or a wall. */
    private val materialFields: List<Component> = listOf(materialPreset, materialKind, colour, parameter)
    private val wallReset = JButton("Materiał jak w scenie").apply { isVisible = false }

    private var lastFolder: File? = File("../stls").takeIf { it.isDirectory } ?: File(".")

    init {
        border = BorderFactory.createEmptyBorder(12, 14, 14, 14)

        section("Scena")
        full(JPanel(GridBagLayout()).apply {
            add(saveSceneButton, GridBagConstraints().apply { weightx = 1.0; fill = GridBagConstraints.HORIZONTAL; insets = Insets(0, 0, 0, 4) })
            add(loadSceneButton, GridBagConstraints().apply { weightx = 1.0; fill = GridBagConstraints.HORIZONTAL })
        })

        section("Tryb (T)")
        ButtonGroup().apply { add(quick); add(raytrace) }
        // Two halves of one switch; the chosen mode lights up in the accent colour.
        for (b in listOf(quick, raytrace)) {
            b.putClientProperty(
                "FlatLaf.style",
                "selectedBackground: \$Button.default.background; selectedForeground: \$Button.default.foreground"
            )
        }
        full(JPanel(GridBagLayout()).apply {
            add(quick, GridBagConstraints().apply { weightx = 1.0; fill = GridBagConstraints.HORIZONTAL; insets = Insets(0, 0, 0, 4) })
            add(raytrace, GridBagConstraints().apply { weightx = 1.0; fill = GridBagConstraints.HORIZONTAL })
        })

        section("Obraz")
        field("Gotowe", preset)
        field("Szerokość", width)
        field("Wysokość", height)
        full(applySize)
        field("Cel spp", target)
        field("Odbicia (maks.)", depth)
        full(progress)
        full(denoise)
        full(save)
        full(JButton("Film obrotowy…").apply {
            toolTipText = "Film MP4: obiekty obracają się w miejscu, kamera i światła stoją"
            addActionListener { TurntableDialog.show(viewer, frame) }
        })
        full(JButton("Pomoc - skróty (F1)").apply { addActionListener { HelpDialog.show(frame) } })

        if (lightSwitches.isNotEmpty()) {
            section("Światła")
            // One row a light: its switch, then its power.
            lightSwitches.forEachIndexed { i, box ->
                full(JPanel(GridBagLayout()).apply {
                    isOpaque = false
                    add(box, GridBagConstraints().apply { weightx = 1.0; anchor = GridBagConstraints.WEST })
                    add(JLabel("Moc").apply { foreground = Theme.muted }, GridBagConstraints().apply { insets = Insets(0, 8, 0, 6) })
                    add(lightPowers[i], GridBagConstraints())
                })
            }
        }

        section("Obiekty na scenie")
        full(add)
        full(JPanel(GridBagLayout()).apply {
            add(addBall, GridBagConstraints().apply { weightx = 1.0; fill = GridBagConstraints.HORIZONTAL; insets = Insets(0, 0, 0, 4) })
            add(addBlock, GridBagConstraints().apply { weightx = 1.0; fill = GridBagConstraints.HORIZONTAL })
        })
        full(JScrollPane(list).apply { preferredSize = Dimension(250, 150); minimumSize = Dimension(250, 150) })
        full(JPanel(GridBagLayout()).apply {
            add(remove, GridBagConstraints().apply { weightx = 1.0; fill = GridBagConstraints.HORIZONTAL; insets = Insets(0, 0, 0, 4) })
            add(centre, GridBagConstraints().apply { weightx = 1.0; fill = GridBagConstraints.HORIZONTAL })
        })
        field("X", posX)
        field("Y (podstawa)", posY)
        field("Z", posZ)
        field("Obrót [°]", turn)
        field("Wysokość", size)
        field("Szerokość", blockWidth)
        field("Głębokość", blockDepth)
        section("Materiał")
        field("Gotowy", materialPreset)
        field("Rodzaj", materialKind)
        field("Kolor", colour)
        field(parameterLabel, parameter)
        full(wallReset)
        full(zUp)
        full(info)
        if (viewer.builtIns.any { !it.part.isLight }) full(restore)
        full(JLabel("<html>Ctrl + przeciąganie przestawia zaznaczony obiekt, strzałki go przesuwają, Q / E obracają.</html>").apply {
            font = Theme.font(11.5f)
            foreground = Theme.muted
        })
        add(JPanel(), GridBagConstraints().apply { gridy = row++; weighty = 1.0 })

        preset.addActionListener {
            val r = presets[preset.selectedIndex].second
            syncing = true
            width.value = r.width
            height.value = r.height
            syncing = false
        }
        applySize.addActionListener {
            viewer.resolution = Viewer.Resolution(width.value as Int, height.value as Int)
        }
        target.addChangeListener { viewer.targetSamples = target.value as Int }
        depth.addChangeListener { viewer.maxDepth = depth.value as Int }
        denoise.addActionListener { viewer.denoiseOn = denoise.isSelected }
        saveSceneButton.addActionListener { chooseSceneToSave() }
        loadSceneButton.addActionListener { chooseSceneToLoad() }
        quick.addActionListener { viewer.raytrace = false; syncMode() }
        raytrace.addActionListener { viewer.raytrace = true; syncMode() }
        save.addActionListener { viewer.chooseSave(frame) }

        add.addActionListener { chooseModel() }
        remove.addActionListener { viewer.selected?.let { viewer.removeObject(it); refreshObjects() } }
        centre.addActionListener { viewer.selectedMovable?.let { viewer.centreInView(it); refreshSelection() } }
        restore.addActionListener { viewer.restoreBuiltIns(); refreshObjects() }
        list.addListSelectionListener {
            if (!it.valueIsAdjusting && !syncing) {
                viewer.selected = list.selectedValue
                refreshSelection()
            }
        }
        posX.addChangeListener { move { o -> o.x = posX.value as Double } }
        posY.addChangeListener { move { o -> o.y = posY.value as Double } }
        posZ.addChangeListener { move { o -> o.z = posZ.value as Double } }
        turn.addChangeListener { move { o -> o.rotation = turn.value as Double } }
        size.addChangeListener { edit { o -> o.height = size.value as Double } }
        materialPreset.addActionListener {
            if (syncing) return@addActionListener
            if (materialPreset.selectedItem == FILE_COLOURS) {
                // Only a model whose file coloured it can go back to those colours.
                edit { o -> if (o.hasFileColours) o.fileColours = true }
                refreshSelection()
                return@addActionListener
            }
            val chosen = materialPreset.selectedItem as? MaterialPreset ?: return@addActionListener
            editMaterial { o -> o.material = chosen.spec }
            refreshSelection()
        }
        materialKind.addActionListener {
            if (syncing) return@addActionListener
            val kind = materialKind.selectedItem as MaterialKind
            editMaterial { o ->
                // A new kind starts from its usual value: slightly rough metal, window glass, a bright light.
                if (o.material.kind != kind) o.material = o.material.copy(kind = kind, parameter = defaultParameter(kind))
            }
            refreshSelection()
        }
        colour.addActionListener {
            val o = viewer.selectedMaterial ?: return@addActionListener
            val picked = JColorChooser.showDialog(frame, "Kolor materiału - $o", o.material.screenColor())
                ?: return@addActionListener
            editMaterial { it.material = it.material.withScreenColor(picked) }
            refreshSelection()
        }
        parameter.addChangeListener { editMaterial { o -> o.material = o.material.copy(parameter = parameter.value as Double) } }
        wallReset.addActionListener {
            val own = viewer.selected as? SceneMaterial ?: return@addActionListener
            editMaterial { own.material = own.original }
            refreshSelection()
        }
        zUp.addActionListener { edit { o -> o.zUp = zUp.isSelected } }
        blockWidth.addChangeListener { edit { o -> o.width = blockWidth.value as Double } }
        blockDepth.addChangeListener { edit { o -> o.depth = blockDepth.value as Double } }
        addBall.addActionListener { viewer.addShape(Shape.Ball); refreshObjects() }
        addBlock.addActionListener { viewer.addShape(Shape.Block); refreshObjects() }
        refreshObjects()
        syncMode()
    }

    /** Moves or turns the selected object - added or the scene's own - unless the panel is filling its fields. */
    private fun move(change: (Movable) -> Unit) {
        if (syncing) return
        val o = viewer.selectedMovable ?: return
        change(o)
        viewer.sceneChanged()
        restore.isEnabled = viewer.builtIns.any { !it.part.isLight && (!it.enabled || it.moved || it.changed) }
    }

    /** Changes the material of the selection - an added object or a wall - unless the panel is filling its fields. */
    private fun editMaterial(change: (HasMaterial) -> Unit) {
        if (syncing) return
        val o = viewer.selectedMaterial ?: return
        change(o)
        // A material set by hand replaces the file's own colours.
        (o as? PlacedObject)?.fileColours = false
        viewer.sceneChanged()
    }

    /** Applies a field change to the selected added object, unless the panel is filling its fields. */
    private fun edit(change: (PlacedObject) -> Unit) {
        if (syncing) return
        val o = viewer.selectedModel ?: return
        change(o)
        viewer.sceneChanged()
    }

    private fun chooseModel() {
        val chooser = JFileChooser(lastFolder).apply {
            fileFilter = FileNameExtensionFilter("Modele 3D (STL, OBJ, PLY, OFF, glTF, GLB, 3MF)", "stl", "obj", "ply", "off", "gltf", "glb", "3mf")
        }
        if (chooser.showOpenDialog(frame) != JFileChooser.APPROVE_OPTION) return
        val file = chooser.selectedFile
        lastFolder = file.parentFile
        add.isEnabled = false
        add.text = "Wczytuję ${file.name}…"
        viewer.addModel(file) { placed, error ->
            add.isEnabled = true
            add.text = "Dodaj model…"
            if (error != null) {
                JOptionPane.showMessageDialog(frame, "Nie udało się wczytać ${file.name}:\n$error", "Błąd wczytywania", JOptionPane.ERROR_MESSAGE)
            }
            if (placed != null) refreshObjects()
        }
    }

    private fun sceneChooser() = JFileChooser(lastSceneFolder).apply {
        fileFilter = FileNameExtensionFilter("Scena ray tracera (.${SceneFile.EXTENSION})", SceneFile.EXTENSION)
    }

    /** Asks for a file name and saves the scene there, adding the extension when it is missing. */
    fun chooseSceneToSave() {
        val chooser = sceneChooser().apply {
            dialogTitle = "Zapisz scenę"
            selectedFile = File(lastSceneFolder, "scena.${SceneFile.EXTENSION}")
        }
        while (true) {
            if (chooser.showSaveDialog(frame) != JFileChooser.APPROVE_OPTION) return
            var file = chooser.selectedFile
            if (!file.name.endsWith(".${SceneFile.EXTENSION}", ignoreCase = true)) file = File(file.path + ".${SceneFile.EXTENSION}")
            if (file.exists() && JOptionPane.showConfirmDialog(
                    frame, "Plik ${file.name} już istnieje. Zastąpić go?", "Zapisz scenę", JOptionPane.YES_NO_OPTION
                ) != JOptionPane.YES_OPTION
            ) continue
            lastSceneFolder = file.absoluteFile.parentFile
            try {
                viewer.saveScene(file)
            } catch (e: Exception) {
                JOptionPane.showMessageDialog(frame, "Nie udało się zapisać sceny:\n${e.message}", "Zapisz scenę", JOptionPane.ERROR_MESSAGE)
            }
            return
        }
    }

    fun chooseSceneToLoad() {
        val chooser = sceneChooser().apply { dialogTitle = "Wczytaj scenę" }
        if (chooser.showOpenDialog(frame) != JFileChooser.APPROVE_OPTION) return
        loadScene(chooser.selectedFile)
    }

    /** Loads [file] into the window, then refreshes every field and reports what went wrong. */
    fun loadScene(file: File) {
        lastSceneFolder = file.absoluteFile.parentFile
        loadSceneButton.isEnabled = false
        viewer.loadScene(file) { error, warnings ->
            loadSceneButton.isEnabled = true
            syncAll()
            if (error != null) {
                JOptionPane.showMessageDialog(frame, "Nie udało się wczytać sceny:\n$error", "Wczytaj scenę", JOptionPane.ERROR_MESSAGE)
            } else if (warnings.isNotEmpty()) {
                JOptionPane.showMessageDialog(frame, warnings.joinToString("\n"), "Scena wczytana z uwagami", JOptionPane.WARNING_MESSAGE)
            }
        }
    }

    /** Fills every field from the viewer, after a scene was loaded into it. */
    fun syncAll() {
        syncing = true
        width.value = viewer.resolution.width
        height.value = viewer.resolution.height
        target.value = viewer.targetSamples
        depth.value = viewer.maxDepth
        lightSwitches.forEachIndexed { i, box -> box.isSelected = viewer.lights[i].enabled }
        lightPowers.forEachIndexed { i, spinner -> spinner.value = viewer.lights[i].power }
        syncing = false
        syncDenoise()
        syncMode()
        refreshObjects()
    }

    /** Rebuilds the list from the viewer's objects and shows the selected one's fields. */
    fun refreshObjects() {
        syncing = true
        listModel.clear()
        viewer.entries().forEach { listModel.addElement(it) }
        list.setSelectedValue(viewer.selected, true)
        syncing = false
        refreshSelection()
    }

    /**
     * Fills the fields from the selection: where it stands for anything movable, and size and
     * material for an added object; greys out what does not apply.
     */
    fun refreshSelection() {
        val o = viewer.selectedModel
        val m = viewer.selectedMovable
        val mat = viewer.selectedMaterial
        val wall = viewer.selected as? WallObject
        val own = viewer.selected as? SceneMaterial
        syncing = true
        remove.isEnabled = viewer.selected != null && wall == null
        placeFields.forEach { it.isEnabled = m != null }
        modelFields.forEach { it.isEnabled = o != null }
        materialFields.forEach { it.isEnabled = mat != null }
        wallReset.isVisible = own != null
        wallReset.isEnabled = own?.changed == true
        if (mat != null) showMaterial(mat.material, (mat as? PlacedObject)?.fileColours == true)
        // Only a loaded model can be stood up from z; only a block has its own width and depth.
        zUp.isEnabled = o?.isModel == true
        blockWidth.isEnabled = o?.shape == Shape.Block
        blockDepth.isEnabled = o?.shape == Shape.Block
        restore.isEnabled = viewer.builtIns.any { !it.part.isLight && (!it.enabled || it.moved || it.changed) }
        if (m != null) {
            posX.value = m.x
            posY.value = m.y
            posZ.value = m.z
            turn.value = m.rotation
        }
        if (viewer.selected is BuiltInObject) {
            info.text = "Obiekt sceny - można go przesuwać, obracać, zmienić materiał i usunąć"
        } else if (wall != null) {
            info.text = "Ściana - można zmienić jej materiał"
        } else if (o != null) {
            size.value = o.height
            zUp.isSelected = o.zUp
            blockWidth.value = o.width
            blockDepth.value = o.depth
            info.text = o.description
        } else {
            info.text = if (viewer.entries().isEmpty()) "Scena jest pusta" else "Zaznacz obiekt na liście"
        }
        syncing = false
    }

    /**
     * Shows [spec] in the material fields: the matching ready-made material (or "Własny"), the
     * kind, a colour swatch, and the kind's number with its own name and range.
     */
    private fun showMaterial(spec: MaterialSpec, fromFile: Boolean = false) {
        materialPreset.selectedItem = if (fromFile) FILE_COLOURS else MaterialPreset.values().firstOrNull { it.spec == spec } ?: CUSTOM
        materialKind.selectedItem = spec.kind
        colour.icon = Swatch(spec.screenColor())
        colour.isEnabled = spec.kind != MaterialKind.GLASS
        colour.toolTipText = if (spec.kind == MaterialKind.GLASS) "Szkło w tym rendererze jest bezbarwne" else null
        val kind = spec.kind
        parameterLabel.text = kind.parameterLabel ?: "Parametr"
        parameter.isEnabled = kind.parameterLabel != null
        parameter.model = SpinnerNumberModel(spec.parameter.coerceIn(kind.min, kind.max), kind.min, kind.max, kind.step)
    }

    private fun defaultParameter(kind: MaterialKind): Double = when (kind) {
        MaterialKind.MATTE -> 0.0
        MaterialKind.METAL -> 0.1
        MaterialKind.GLASS -> 1.5
        MaterialKind.LIGHT -> 4.0
    }

    /** A small filled square showing a material's colour on its button. */
    private class Swatch(private val colour: java.awt.Color) : javax.swing.Icon {
        override fun getIconWidth() = 14
        override fun getIconHeight() = 14
        override fun paintIcon(c: Component?, g: java.awt.Graphics, x: Int, y: Int) {
            g.color = colour
            g.fillRect(x, y, 14, 14)
            g.color = java.awt.Color.DARK_GRAY
            g.drawRect(x, y, 13, 13)
        }
    }

    /** A fixed width; the height is whatever the fields need (the window scrolls the panel). */
    override fun getPreferredSize(): Dimension = Dimension(PANEL_WIDTH, super.getPreferredSize().height)

    fun showProgress(samples: Int, target: Int) {
        progress.value = (samples.toLong() * 1000 / target.coerceAtLeast(1)).toInt()
        progress.string = "$samples / $target spp"
    }

    fun syncDenoise() {
        denoise.isSelected = viewer.denoiseOn
    }

    /** Shows the viewer's mode; the sample target and denoising only matter when ray tracing. */
    fun syncMode() {
        quick.isSelected = !viewer.raytrace
        raytrace.isSelected = viewer.raytrace
        target.isEnabled = viewer.raytrace
        depth.isEnabled = viewer.raytrace
        denoise.isEnabled = viewer.raytrace
    }

    private fun numberSpinner(step: Double) = JSpinner(SpinnerNumberModel(0.0, -1e6, 1e6, step)).apply {
        editor = JSpinner.NumberEditor(this, "0.##")
    }

    /** A section heading: its title in small capitals, then a rule to the panel's edge. */
    private fun section(title: String) {
        val heading = JPanel(GridBagLayout()).apply {
            isOpaque = false
            add(JLabel(title.uppercase()).apply {
                font = Theme.font(11f, bold = true)
                foreground = Theme.muted
            }, GridBagConstraints().apply { anchor = GridBagConstraints.WEST; insets = Insets(0, 0, 0, 8) })
            add(JSeparator(), GridBagConstraints().apply { weightx = 1.0; fill = GridBagConstraints.HORIZONTAL })
        }
        add(heading, GridBagConstraints().apply {
            gridx = 0; gridy = row++; gridwidth = 2; weightx = 1.0; fill = GridBagConstraints.HORIZONTAL
            insets = Insets(if (row > 1) 18 else 2, 0, 8, 0)
        })
    }

    private fun field(label: String, component: Component) = field(JLabel(label), component)

    /** A labelled row; the label is passed in when it changes later (the material parameter). */
    private fun field(label: JLabel, component: Component) {
        add(label, GridBagConstraints().apply {
            gridx = 0; gridy = row; anchor = GridBagConstraints.WEST; insets = Insets(2, 0, 2, 8)
        })
        add(component, GridBagConstraints().apply {
            gridx = 1; gridy = row++; weightx = 1.0; fill = GridBagConstraints.HORIZONTAL; insets = Insets(2, 0, 2, 0)
        })
    }

    private fun full(component: Component) {
        add(component, GridBagConstraints().apply {
            gridx = 0; gridy = row++; gridwidth = 2; weightx = 1.0; fill = GridBagConstraints.HORIZONTAL; insets = Insets(3, 0, 3, 0)
        })
    }

    companion object {
        private const val PANEL_WIDTH = 330
        /** The preset list's entry for a material changed by hand. */
        private const val CUSTOM = "Własny"
        /** The preset list's entry for a model made of the materials its file names. */
        private const val FILE_COLOURS = "Kolory z pliku"
    }
}
