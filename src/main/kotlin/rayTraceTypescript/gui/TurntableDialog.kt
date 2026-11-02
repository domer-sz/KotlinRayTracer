package rayTraceTypescript.gui

import rayTraceTypescript.gpu.TurntableSettings
import rayTraceTypescript.render.VideoEncoder
import java.awt.BorderLayout
import java.awt.Component
import java.awt.FlowLayout
import java.awt.GridBagConstraints
import java.awt.GridBagLayout
import java.awt.Insets
import java.awt.event.KeyEvent
import java.awt.event.WindowAdapter
import java.awt.event.WindowEvent
import java.io.File
import javax.swing.BorderFactory
import javax.swing.JButton
import javax.swing.JCheckBox
import javax.swing.JComboBox
import javax.swing.JComponent
import javax.swing.JDialog
import javax.swing.JFileChooser
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.JOptionPane
import javax.swing.JPanel
import javax.swing.JProgressBar
import javax.swing.JSpinner
import javax.swing.KeyStroke
import javax.swing.SpinnerNumberModel
import javax.swing.SwingUtilities
import javax.swing.WindowConstants
import javax.swing.filechooser.FileNameExtensionFilter

/**
 * "Film obrotowy": a film of the scene's objects turning in place while the camera and the
 * lights stand still, after StlViewer's dialog of the same name. The options are locked while it
 * renders; Przerwij (or Escape, or closing the dialog) stops after the current frame and saves
 * nothing. The film is denoised frame by frame when the view is (D).
 */
class TurntableDialog private constructor(private val viewer: Viewer, owner: JFrame) :
    JDialog(owner, "Film obrotowy", true) {

    private class Format(val label: String, val width: Int, val height: Int) {
        override fun toString() = label
    }

    private val panelSize = viewer.resolution
    private val formats = listOf(
        Format("Jak w panelu (${even(panelSize.width)} × ${even(panelSize.height)})", even(panelSize.width), even(panelSize.height)),
        Format("Pion 1080 × 1920", 1080, 1920),
        Format("Kwadrat 1080 × 1080", 1080, 1080),
        Format("Poziom 1920 × 1080", 1920, 1080)
    )
    private val format = JComboBox(formats.toTypedArray()).apply { selectedIndex = remembered.format.coerceIn(0, formats.size - 1) }
    private val seconds = JSpinner(SpinnerNumberModel(remembered.settings.seconds, 1.0, 60.0, 1.0))
    private val turns = JSpinner(SpinnerNumberModel(remembered.settings.turns, 1, 10, 1))
    private val fps = JComboBox(arrayOf(24, 30, 60)).apply { selectedItem = remembered.settings.fps }
    private val samples = JSpinner(SpinnerNumberModel(remembered.settings.samples, 1, 4096, 8))
    private val reverse = JCheckBox("Obrót w drugą stronę", remembered.settings.reverse)
    private val compact = JCheckBox("Mniejszy plik", remembered.settings.compact)
    private val turning = JLabel()
    private val info = JLabel()
    private val progress = JProgressBar(0, 1000).apply { isStringPainted = true; string = "" }
    private val state = JLabel(" ")
    private val start = JButton("Renderuj film…")
    private val close = JButton("Zamknij")
    private val options: List<JComponent> = listOf(format, seconds, turns, fps, samples, reverse, compact, start)

    @Volatile private var job: Viewer.FilmJob? = null

    init {
        defaultCloseOperation = WindowConstants.DO_NOTHING_ON_CLOSE
        val form = JPanel(GridBagLayout()).apply { border = BorderFactory.createEmptyBorder(14, 16, 8, 16) }
        var row = 0
        fun field(label: String, component: Component) {
            form.add(JLabel(label), GridBagConstraints().apply {
                gridx = 0; gridy = row; anchor = GridBagConstraints.WEST; insets = Insets(3, 0, 3, 12)
            })
            form.add(component, GridBagConstraints().apply {
                gridx = 1; gridy = row++; weightx = 1.0; fill = GridBagConstraints.HORIZONTAL; insets = Insets(3, 0, 3, 0)
            })
        }
        fun full(component: Component) {
            form.add(component, GridBagConstraints().apply {
                gridx = 0; gridy = row++; gridwidth = 2; weightx = 1.0; fill = GridBagConstraints.HORIZONTAL; insets = Insets(4, 0, 4, 0)
            })
        }
        field("Format", format)
        field("Czas jednego obrotu (s)", seconds)
        field("Obrotów w filmie", turns)
        field("Klatek na sekundę", fps)
        field("Próbek na klatkę", samples)
        full(reverse)
        full(compact)
        full(turning.apply { foreground = Theme.muted })
        full(info.apply { foreground = Theme.muted })
        full(progress)
        full(state)

        val buttons = JPanel(FlowLayout(FlowLayout.RIGHT, 8, 0)).apply {
            border = BorderFactory.createEmptyBorder(4, 16, 14, 16)
            add(close)
            add(start)
        }
        start.putClientProperty(
            "FlatLaf.style",
            "background: \$Button.default.background; foreground: \$Button.default.foreground; " +
                "borderColor: \$Button.default.borderColor; font: bold"
        )
        contentPane.add(form, BorderLayout.CENTER)
        contentPane.add(buttons, BorderLayout.SOUTH)

        listOf(format, fps).forEach { it.addActionListener { describe() } }
        listOf(seconds, turns, samples).forEach { it.addChangeListener { describe() } }
        listOf(reverse, compact).forEach { it.addActionListener { describe() } }
        start.addActionListener { begin() }
        close.addActionListener { closeOrCancel() }
        addWindowListener(object : WindowAdapter() {
            override fun windowClosing(e: WindowEvent) = closeOrCancel()
        })
        rootPane.registerKeyboardAction(
            { closeOrCancel() }, KeyStroke.getKeyStroke(KeyEvent.VK_ESCAPE, 0), JComponent.WHEN_IN_FOCUSED_WINDOW
        )
        rootPane.defaultButton = start

        val names = viewer.filmTurning().map { it.toString() }
        turning.text = when {
            names.isEmpty() -> "Na scenie nie ma nic do obracania."
            viewer.selectedMovable != null -> "Obraca się zaznaczony obiekt: ${names.first()}"
            else -> "Obraca się: ${names.joinToString()} (zaznacz obiekt, by obracać tylko jego)"
        }
        start.isEnabled = names.isNotEmpty()
        describe()
        pack()
        setLocationRelativeTo(owner)
    }

    private fun settings() = TurntableSettings(
        seconds = (seconds.value as Number).toDouble(),
        turns = (turns.value as Number).toInt(),
        fps = fps.selectedItem as Int,
        samples = (samples.value as Number).toInt(),
        reverse = reverse.isSelected,
        compact = compact.isSelected
    )

    private fun chosenFormat() = format.selectedItem as Format

    /** The line under the options: size, frames, length, file size, denoising. */
    private fun describe() {
        val s = settings()
        val f = chosenFormat()
        val length = s.framesPerTurn.toDouble() / s.fps * s.turns
        val megabytes = VideoEncoder.bitrate(f.width, f.height, s.fps, s.compact) * length / 8 / 1e6
        val turnsText = if (s.turns == 1) "jeden pełny obrót" else "${s.turns} obroty (liczony jeden, powtarzany)"
        info.text = "<html>${f.width} × ${f.height}, ${s.framesPerTurn} klatek (%.0f s przy ${s.fps} kl./s), $turnsText.<br>".format(s.seconds) +
            "Plik ok. %.0f MB; odszumianie każdej klatki: ${if (viewer.filmDenoised()) "tak" else "nie"}.</html>".format(megabytes)
    }

    private fun begin() {
        val output = chooseFile() ?: return
        val s = settings()
        val f = chosenFormat()
        remembered = remembered.copy(settings = s, format = format.selectedIndex)
        options.forEach { it.isEnabled = false }
        close.text = "Przerwij"
        progress.value = 0
        progress.string = "0 / ${s.framesPerTurn}"
        state.text = "Pierwsza klatka…"
        val started = Viewer.FilmJob(s, Viewer.Resolution(f.width, f.height), output, object : Viewer.FilmListener {
            override fun progress(done: Int, total: Int, elapsedNanos: Long) = SwingUtilities.invokeLater {
                progress.value = (done.toLong() * 1000 / total).toInt()
                progress.string = "$done / $total"
                // From the measured pace, not a guess made before the start.
                val perFrame = elapsedNanos / 1e9 / done
                state.text = "Klatka $done z $total · %.1f s na klatkę · zostało ok. ${duration(perFrame * (total - done))}".format(perFrame)
            }

            override fun finished(saved: Boolean, error: String?) = SwingUtilities.invokeLater {
                job = null
                when {
                    saved -> {
                        dispose()
                        JOptionPane.showMessageDialog(
                            owner, "Zapisano ${output.name} (%.1f MB)".format(output.length() / 1e6), "Film obrotowy",
                            JOptionPane.INFORMATION_MESSAGE
                        )
                    }
                    error != null -> {
                        reset("Nie udało się zapisać filmu.")
                        JOptionPane.showMessageDialog(this@TurntableDialog, "Nie udało się zapisać filmu: $error", "Film obrotowy", JOptionPane.ERROR_MESSAGE)
                    }
                    else -> reset("Przerwano - film nie został zapisany.")
                }
            }
        })
        job = started
        viewer.startFilm(started)
    }

    private fun reset(message: String) {
        options.forEach { it.isEnabled = true }
        close.text = "Zamknij"
        close.isEnabled = true
        progress.value = 0
        progress.string = ""
        state.text = message
    }

    private fun closeOrCancel() {
        val running = job
        if (running != null) {
            running.cancelled = true
            close.isEnabled = false
            state.text = "Przerywam po bieżącej klatce…"
        } else {
            dispose()
        }
    }

    private fun chooseFile(): File? {
        val chooser = JFileChooser(remembered.folder ?: viewer.filmFolder).apply {
            dialogTitle = "Zapisz film"
            fileFilter = FileNameExtensionFilter("Film MP4", "mp4")
            selectedFile = File(currentDirectory, "${viewer.sceneName}-turntable.mp4")
        }
        while (true) {
            if (chooser.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) return null
            val picked = chooser.selectedFile
            val file = if (picked.name.lowercase().endsWith(".mp4")) picked else File(picked.parentFile, picked.name + ".mp4")
            if (file.exists() && JOptionPane.showConfirmDialog(
                    this, "Plik ${file.name} już istnieje. Zastąpić go?", "Zapisz film", JOptionPane.YES_NO_OPTION
                ) != JOptionPane.YES_OPTION
            ) continue
            remembered = remembered.copy(folder = file.parentFile)
            return file
        }
    }

    /** What the dialog shows the next time it opens in this window. */
    private data class Remembered(
        val settings: TurntableSettings = TurntableSettings(),
        val format: Int = 0,
        val folder: File? = null
    )

    companion object {
        private var remembered = Remembered()

        fun show(viewer: Viewer, owner: JFrame) {
            TurntableDialog(viewer, owner).isVisible = true
        }

        /** Rounded down to even: H.264 in 4:2:0 wants both sides even. */
        private fun even(n: Int) = maxOf(2, n - n % 2)

        private fun duration(seconds: Double): String {
            val s = seconds.toLong().coerceAtLeast(0)
            return if (s >= 60) "${s / 60} min ${s % 60} s" else "$s s"
        }
    }
}
