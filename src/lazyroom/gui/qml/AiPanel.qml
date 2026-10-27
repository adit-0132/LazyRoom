import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs

// Natural-language editing + reference matching. Results are ordinary editable steps.
ScrollView {
    id: sv
    contentWidth: availableWidth
    clip: true
    ColumnLayout {
        width: sv.availableWidth
        spacing: 12
        Item { height: 4 }
        RowLayout {
            Layout.leftMargin: 16; Layout.rightMargin: 16
            IconImageTint { iconName: "sparkle"; tint: Theme.primary; size: 22 }
            Label { text: "AI assistant"; font.pixelSize: 17; font.weight: Font.DemiBold; color: Theme.text }
        }
        Label {
            Layout.leftMargin: 16; Layout.rightMargin: 16; Layout.fillWidth: true
            text: "Describe the look you want. LazyRoom turns it into ordinary steps you can inspect, tweak, reorder or disable."
            wrapMode: Text.Wrap; color: Theme.textMuted; font.pixelSize: 12
        }
        TextArea {
            id: req
            Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16
            Layout.preferredHeight: 110
            wrapMode: TextEdit.Wrap
            placeholderText: "e.g. Warm, faded 1970s film look with muted greens and subtle grain"
            background: Rectangle { radius: 16; color: Theme.surfaceHigh; border.color: req.activeFocus ? Theme.primary : "transparent" }
            Keys.onPressed: function(e) { if ((e.key === Qt.Key_Return || e.key === Qt.Key_Enter) && (e.modifiers & Qt.ControlModifier)) { controller.runIntent(req.text); e.accepted = true } }
        }
        RowLayout {
            Layout.leftMargin: 16; Layout.rightMargin: 16
            FilledButton {
                text: "Apply"
                icon.source: "icons/send.svg"; icon.width: 16; icon.height: 16
                enabled: controller.projectOpen && req.text.trim().length > 0
                onClicked: controller.runIntent(req.text)
            }
            Label { text: "Ctrl+Enter"; color: Theme.outline; font.pixelSize: 11 }
        }
        Flow {
            Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16
            spacing: 6
            Repeater {
                model: ["Faded 1990s magazine photo", "Cinematic teal and orange", "Vintage Polaroid",
                        "Make the shadows warmer without changing skin tones", "Cool, desaturated urban look",
                        "Bright and airy", "Moody, dark, muted greens"]
                delegate: Rectangle {
                    height: 30; radius: 8; width: sl.implicitWidth + 20
                    color: "transparent"; border.color: Theme.outlineVariant
                    Label { id: sl; anchors.centerIn: parent; text: modelData; font.pixelSize: 12; color: Theme.textMuted }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: req.text = modelData }
                }
            }
        }
        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.outlineVariant; opacity: 0.6; Layout.topMargin: 4 }
        ColumnLayout {
            Layout.leftMargin: 16; Layout.rightMargin: 16; spacing: 6
            visible: controller.aiModel.count > 0 || controller.aiText !== ""
            Label { text: "Interpretation"; font.pixelSize: 13; font.weight: Font.DemiBold; color: Theme.text }
            Repeater {
                model: controller.aiModel
                delegate: RowLayout {
                    Layout.fillWidth: true
                    Rectangle { width: 6; height: 6; radius: 3; color: Theme.primary }
                    Label { text: "<b>" + model.phrase + "</b> → " + model.effect; textFormat: Text.StyledText
                        color: Theme.textMuted; font.pixelSize: 12; wrapMode: Text.Wrap; Layout.fillWidth: true }
                }
            }
            Label { text: controller.aiText; visible: text !== ""; color: Theme.outline; font.pixelSize: 11; wrapMode: Text.Wrap; Layout.fillWidth: true }
        }
        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.outlineVariant; opacity: 0.6 }
        ColumnLayout {
            Layout.leftMargin: 16; Layout.rightMargin: 16; spacing: 6
            Label { text: "Match a reference photo"; font.pixelSize: 13; font.weight: Font.DemiBold; color: Theme.text }
            Label { text: "Fits white balance, tone curve, colour grade, saturation, grain and vignette to the reference — as editable steps."
                wrapMode: Text.Wrap; color: Theme.textMuted; font.pixelSize: 12; Layout.fillWidth: true }
            Button { text: "Choose reference…"; enabled: controller.projectOpen; onClicked: refDlg.open()
                icon.source: "icons/image.svg"; icon.width: 16; icon.height: 16 }
        }
        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.outlineVariant; opacity: 0.6 }
        Label {
            Layout.leftMargin: 16; Layout.rightMargin: 16; Layout.fillWidth: true
            text: "Agents (Claude, Cursor, Copilot, …) can drive LazyRoom directly over MCP. Run <b>lazyroom agent-setup</b> for the one-line registration. Projects saved here update live when an agent edits them."
            textFormat: Text.StyledText; wrapMode: Text.Wrap; color: Theme.outline; font.pixelSize: 11
        }
        Item { height: 12 }
    }
    FileDialog { id: refDlg; title: "Reference photo"; onAccepted: controller.matchReference(selectedFile) }
}
