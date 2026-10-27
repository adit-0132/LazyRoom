import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

// Apply the current edit (or a look) to a whole folder in the background.
PmDialog {
    id: dlg
    title: "Batch process"
    modal: true
    anchors.centerIn: parent
    width: 560
    standardButtons: Dialog.Close
    property var looks: []
    onAboutToShow: looks = [{id: "__project__", name: "Current edit (this project's steps)"}].concat(controller.presetChoices())
    ColumnLayout {
        anchors.fill: parent; spacing: 10
        RowLayout { Label { text: "Input folder"; Layout.preferredWidth: 110; color: Theme.textMuted }
            TextField { id: inDir; Layout.fillWidth: true } IconButton { iconName: "open"; onClicked: inDlg.open() } }
        RowLayout { Label { text: "Output folder"; Layout.preferredWidth: 110; color: Theme.textMuted }
            TextField { id: outDir; Layout.fillWidth: true } IconButton { iconName: "open"; onClicked: outDlg.open() } }
        RowLayout { Label { text: "Apply"; Layout.preferredWidth: 110; color: Theme.textMuted }
            ComboBox { id: look; Layout.fillWidth: true; model: dlg.looks; textRole: "name"; valueRole: "id" } }
        RowLayout { Label { text: "Consistency"; Layout.preferredWidth: 110; color: Theme.textMuted }
            ComboBox { id: norm; Layout.fillWidth: true; textRole: "t"; valueRole: "v"
                model: [{t: "None", v: "none"}, {t: "Even out exposure", v: "exposure"}, {t: "Neutralise white balance", v: "white_balance"}, {t: "Both", v: "both"}] } }
        RowLayout { Label { text: "File names"; Layout.preferredWidth: 110; color: Theme.textMuted }
            TextField { id: naming; Layout.fillWidth: true; text: "{stem}-edit{ext}" } }
        RowLayout { Label { text: "Format"; Layout.preferredWidth: 110; color: Theme.textMuted }
            ComboBox { id: fmt; Layout.fillWidth: true; model: ["(same as input)"].concat(controller.exportFormats) } }
        CheckBox { id: rec; text: "Include subfolders" }
        ProgressBar { Layout.fillWidth: true; value: controller.batchProgress; visible: controller.batchRunning || controller.batchProgress > 0 }
        Label { text: controller.batchStatus; color: Theme.textMuted; wrapMode: Text.Wrap; Layout.fillWidth: true }
        RowLayout {
            FilledButton { text: "Start"; enabled: !controller.batchRunning && inDir.text !== "" && outDir.text !== ""
                onClicked: controller.startBatch(inDir.text, outDir.text, look.currentValue, naming.text,
                                                 fmt.currentIndex === 0 ? "" : fmt.currentText, norm.currentValue, rec.checked) }
            Button { text: "Cancel batch"; visible: controller.batchRunning; onClicked: controller.cancelBatch() }
        }
    }
    FolderDialog { id: inDlg; onAccepted: inDir.text = controller.urlToPath(selectedFolder) }
    FolderDialog { id: outDlg; onAccepted: outDir.text = controller.urlToPath(selectedFolder) }
}
