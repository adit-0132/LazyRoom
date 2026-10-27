import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs

// Run any utility tool (palette, collage, OCR, QR, stack, HDR, panorama, textures, fractals, PDF...) from a
// form generated from its parameter schema — the same tools agents call via run_tool.
PmDialog {
    id: dlg
    title: "Tools"
    modal: true
    anchors.centerIn: parent
    width: Math.min(1100, parent.width - 80); height: Math.min(760, parent.height - 80)
    standardButtons: Dialog.Close
    property string current: ""
    property var values: ({})
    onAboutToShow: controller.refreshTools()

    RowLayout {
        anchors.fill: parent
        spacing: 16
        ColumnLayout {
            Layout.preferredWidth: 280; Layout.fillHeight: true
            TextField { id: tq; Layout.fillWidth: true; placeholderText: "Filter tools" }
            ListView {
                id: tl
                Layout.fillWidth: true; Layout.fillHeight: true; clip: true
                model: controller.toolsModel
                ScrollBar.vertical: ScrollBar {}
                delegate: ItemDelegate {
                    width: tl.width - 8
                    visible: tq.text === "" || (model.title + " " + model.name + " " + model.category).toLowerCase().indexOf(tq.text.toLowerCase()) >= 0
                    height: visible ? 46 : 0
                    enabled: model.available
                    highlighted: dlg.current === model.name
                    onClicked: { dlg.current = model.name; dlg.values = ({}); controller.selectTool(model.name) }
                    contentItem: ColumnLayout {
                        spacing: 0
                        Label { text: model.title; color: Theme.text; font.pixelSize: 13; elide: Text.ElideRight; Layout.fillWidth: true }
                        Label { text: model.category + (model.available ? "" : " · unavailable"); color: Theme.textMuted; font.pixelSize: 10 }
                    }
                }
            }
        }
        ColumnLayout {
            Layout.fillWidth: true; Layout.fillHeight: true
            ScrollView {
                Layout.fillWidth: true; Layout.preferredHeight: dlg.height * 0.42
                contentWidth: availableWidth; clip: true
                ColumnLayout {
                    width: parent.width
                    Repeater {
                        model: controller.toolParamsModel
                        delegate: RowLayout {
                            Layout.fillWidth: true
                            Label { text: model.title; color: Theme.textMuted; Layout.preferredWidth: 170; elide: Text.ElideRight
                                ToolTip.visible: hh.hovered && model.description !== ""; ToolTip.text: model.description; HoverHandler { id: hh } }
                            Loader {
                                Layout.fillWidth: true
                                property var row: model
                                sourceComponent: model.type === "bool" ? boolC : (model.type === "choice" ? choiceC : textC)
                            }
                            IconButton { visible: model.type === "file" || model.name === "output" || model.name.indexOf("dir") >= 0 || model.type === "string_list"
                                iconName: "open"; tip: "Browse…"
                                onClicked: { pick.target = model.name; pick.folder = model.name.indexOf("dir") >= 0; model.name.indexOf("dir") >= 0 ? folderDlg.open() : (model.name === "output" ? saveDlg3.open() : openDlg3.open()) } }
                        }
                    }
                }
            }
            RowLayout {
                FilledButton { text: controller.toolBusy ? "Running…" : "Run"; enabled: dlg.current !== "" && !controller.toolBusy
                    onClicked: controller.runTool(dlg.current, JSON.stringify(dlg.values)) }
                BusyIndicator { running: controller.toolBusy; visible: running; Layout.preferredHeight: 32; Layout.preferredWidth: 32 }
            }
            RowLayout {
                Layout.fillWidth: true; Layout.fillHeight: true
                ScrollView {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    TextArea { text: controller.toolResult; readOnly: true; font.family: "monospace"; font.pixelSize: 11; wrapMode: TextEdit.WrapAnywhere
                        background: Rectangle { radius: 8; color: Theme.surfaceHigh } }
                }
                Image { source: controller.toolOutput; visible: source.toString() !== ""; cache: false; asynchronous: true
                    fillMode: Image.PreserveAspectFit; Layout.preferredWidth: 280; Layout.fillHeight: true }
            }
        }
    }
    Component { id: boolC; Switch { checked: row.value === true; onToggled: { var v = dlg.values; v[row.name] = checked; dlg.values = v } } }
    Component { id: choiceC; ComboBox { model: row.choices; currentIndex: Math.max(0, row.choices.indexOf(row.value))
        onActivated: function(i) { var v = dlg.values; v[row.name] = row.choices[i]; dlg.values = v } } }
    Component { id: textC; TextField {
        text: dlg.values[row.name] !== undefined ? dlg.values[row.name] : (row.value === null || row.value === undefined ? "" : String(row.value))
        placeholderText: row.type === "file" ? "path…" : ""
        onEditingFinished: { var v = dlg.values; var t = text;
            if (row.type === "int" || row.type === "float") t = Number(text);
            v[row.name] = t; dlg.values = v } } }
    QtObject { id: pick; property string target: ""; property bool folder: false }
    function setPicked(path) { var v = dlg.values; v[pick.target] = path; dlg.values = v; controller.toolParamsModel; }
    FileDialog { id: openDlg3; onAccepted: dlg.setPicked(controller.urlToPath(selectedFile)) }
    FileDialog { id: saveDlg3; fileMode: FileDialog.SaveFile; onAccepted: dlg.setPicked(controller.urlToPath(selectedFile)) }
    FolderDialog { id: folderDlg; onAccepted: dlg.setPicked(controller.urlToPath(selectedFolder)) }
}
