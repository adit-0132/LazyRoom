import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs

PmDialog {
    id: dlg
    title: "Export image"
    modal: true
    anchors.centerIn: parent
    width: 460
    standardButtons: Dialog.Cancel
    onAboutToShow: { path.text = controller.suggestedExportName(); fmt.currentIndex = 0 }
    ColumnLayout {
        anchors.fill: parent
        spacing: 12
        RowLayout {
            TextField { id: path; Layout.fillWidth: true; placeholderText: "Output file"; selectByMouse: true }
            IconButton { iconName: "open"; tip: "Browse…"; onClicked: saveDlg2.open() }
        }
        RowLayout {
            Label { text: "Format"; Layout.preferredWidth: 90; color: Theme.textMuted }
            ComboBox { id: fmt; Layout.fillWidth: true; model: controller.exportFormats
                onActivated: function(i) { path.text = path.text.replace(/\.[A-Za-z0-9]+$/, "") + "." + model[i] } }
        }
        RowLayout {
            visible: ["jpg", "webp", "avif", "heic", "jp2"].indexOf(fmt.currentText) >= 0
            Label { text: "Quality"; Layout.preferredWidth: 90; color: Theme.textMuted }
            Slider { id: q; Layout.fillWidth: true; from: 1; to: 100; stepSize: 1; value: 92 }
            Label { text: Math.round(q.value); Layout.preferredWidth: 30; color: Theme.text }
        }
        RowLayout {
            Label { text: "Metadata"; Layout.preferredWidth: 90; color: Theme.textMuted }
            ComboBox { id: meta; Layout.fillWidth: true; textRole: "t"; valueRole: "v"
                model: [{t: "Keep (EXIF, XMP)", v: "keep"}, {t: "Keep, remove GPS location", v: "keep_no_gps"}, {t: "Strip all", v: "strip"}] }
        }
        CheckBox { id: over; text: "Overwrite if the file exists (originals are always protected)" }
        Button {
            text: "Export"; highlighted: true; Layout.alignment: Qt.AlignRight
            enabled: path.text.length > 0
            onClicked: { controller.exportImage(path.text, fmt.currentText, Math.round(q.value), meta.currentValue, over.checked); dlg.close() }
        }
    }
    FileDialog {
        id: saveDlg2
        fileMode: FileDialog.SaveFile
        onAccepted: path.text = controller.urlToPath(selectedFile)
    }
}
