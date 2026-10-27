import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// Steps (non-destructive adjustment layers) + parameters of the selected step.
ColumnLayout {
    id: panel
    spacing: 0

    RowLayout {
        Layout.fillWidth: true; Layout.margins: 12; Layout.bottomMargin: 4
        Label { text: "Steps"; font.pixelSize: 15; font.weight: Font.DemiBold; color: Theme.text }
        Label { text: controller.stepsModel.count; color: Theme.textMuted; font.pixelSize: 13 }
        Item { Layout.fillWidth: true }
        IconButton { iconName: "delete"; tip: "Clear all steps"; enabled: controller.stepsModel.count > 0; onClicked: controller.clearSteps() }
    }
    ListView {
        id: steps
        Layout.fillWidth: true
        Layout.preferredHeight: Math.min(contentHeight + 8, panel.height * 0.38)
        Layout.minimumHeight: 56
        clip: true
        model: controller.stepsModel
        spacing: 2
        ScrollBar.vertical: ScrollBar {}
        Label {
            anchors.centerIn: parent
            visible: steps.count === 0
            text: controller.projectOpen ? "No edits yet — pick a tool on the left,\na look, or ask the AI assistant." : ""
            horizontalAlignment: Text.AlignHCenter
            color: Theme.outline; font.pixelSize: 12
        }
        delegate: ItemDelegate {
            id: sd
            width: steps.width - 16
            x: 8
            height: 48
            z: dragH.active ? 10 : 0
            opacity: dragH.active ? 0.85 : 1
            property bool sel: controller.selectedStepId === model.id
            onClicked: controller.selectStep(model.id)
            DragHandler {  // drag a step up/down to reorder it (within its level)
                id: dragH
                target: sd
                xAxis.enabled: false
                property real startY: 0
                onActiveChanged: {
                    if (active) { startY = sd.y; return }
                    var row = Math.round((sd.y + sd.height / 2 - 24) / (sd.height + steps.spacing))
                    var id = model.id
                    sd.y = startY
                    if (row !== index) controller.moveStepToRow(id, row)
                }
            }
            background: Rectangle {
                radius: 12
                color: sd.sel ? Theme.secondaryContainer : (sd.hovered ? Theme.surfaceHigh : "transparent")
                Behavior on color { ColorAnimation { duration: 120 } }
            }
            contentItem: RowLayout {
                spacing: 4
                Item { Layout.preferredWidth: model.depth * 16; Layout.minimumWidth: model.depth * 16; Layout.maximumWidth: model.depth * 16 }
                IconButton {
                    iconName: model.enabled ? "eye" : "eye_off"
                    tip: model.enabled ? "Disable" : "Enable"
                    implicitWidth: 32; implicitHeight: 32; icon.width: 17; icon.height: 17
                    onClicked: controller.toggleStep(model.id)
                }
                ColumnLayout {
                    spacing: 0
                    Layout.fillWidth: true
                    RowLayout {
                        spacing: 6
                        Layout.fillWidth: true
                        Label {
                            text: model.title
                            color: model.enabled ? (sd.sel ? Theme.secondaryContainerText : Theme.text) : Theme.outline
                            font.pixelSize: 13; font.weight: model.isGroup ? Font.DemiBold : Font.Normal
                            elide: Text.ElideRight; Layout.fillWidth: true
                        }
                        Rectangle {
                            visible: model.origin !== "user"
                            radius: 4; height: 16; width: ob.implicitWidth + 8
                            color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.16)
                            Label { id: ob; anchors.centerIn: parent; font.pixelSize: 9; color: Theme.primary
                                text: model.origin.indexOf("preset") === 0 ? "PRESET" : (model.origin === "style_match" ? "MATCH" : (model.origin === "look" ? "LOOK" : "AI")) }
                        }
                        IconImageTint { visible: model.hasMask; iconName: "mask"; size: 14; tint: Theme.tertiary }
                    }
                    Label { text: model.summary; visible: text !== ""; color: Theme.textMuted; font.pixelSize: 10; elide: Text.ElideRight; Layout.fillWidth: true }
                }
                IconButton { visible: sd.hovered || sd.sel; iconName: "up"; tip: "Move up"; implicitWidth: 28; implicitHeight: 28; icon.width: 16; icon.height: 16; onClicked: controller.moveStep(model.id, -1) }
                IconButton { visible: sd.hovered || sd.sel; iconName: "down"; tip: "Move down"; implicitWidth: 28; implicitHeight: 28; icon.width: 16; icon.height: 16; onClicked: controller.moveStep(model.id, 1) }
                IconButton { visible: sd.hovered || sd.sel; iconName: "delete"; tip: "Remove (Del)"; implicitWidth: 28; implicitHeight: 28; icon.width: 16; icon.height: 16; onClicked: controller.removeStep(model.id) }
            }
        }
    }
    Rectangle { Layout.fillWidth: true; height: 1; color: Theme.outlineVariant; opacity: 0.6; Layout.topMargin: 6 }

    ScrollView {
        id: sv
        Layout.fillWidth: true; Layout.fillHeight: true
        contentWidth: availableWidth
        clip: true
        ColumnLayout {
            width: sv.availableWidth
            spacing: 14
            Item { height: 4 }
            ColumnLayout {
                Layout.leftMargin: 16; Layout.rightMargin: 16
                visible: controller.selectedStepId !== ""
                spacing: 2
                Label { text: controller.selectedTitle; font.pixelSize: 17; font.weight: Font.DemiBold; color: Theme.text; wrapMode: Text.Wrap; Layout.fillWidth: true }
                Label { text: controller.selectedDescription; color: Theme.textMuted; font.pixelSize: 12; wrapMode: Text.Wrap; Layout.fillWidth: true; visible: text !== "" }
            }
            Label {
                visible: controller.selectedStepId === "" && controller.stepsModel.count > 0
                text: "Select a step to adjust it."
                color: Theme.outline; Layout.leftMargin: 16
            }
            RowLayout {  // white-balance eyedropper
                visible: controller.canPickNeutral
                Layout.leftMargin: 16; Layout.rightMargin: 16
                Button {
                    text: controller.overlayMode === "pick" ? "Click a neutral grey/white area…" : "Pick neutral point"
                    checkable: true; checked: controller.overlayMode === "pick"; flat: true
                    icon.source: "icons/colorize.svg"; icon.width: 18; icon.height: 18
                    onClicked: controller.setPickNeutral(controller.overlayMode !== "pick")
                }
            }
            Repeater {
                model: controller.paramsModel
                delegate: ParamEditor {
                    Layout.fillWidth: true
                    Layout.leftMargin: 16; Layout.rightMargin: 16
                    pname: model.name; ptitle: model.title; ptype: model.type; pmin: model.min; pmax: model.max
                    pstep: model.step; pvalue: model.value; pchoices: model.choices || []; punit: model.unit
                    pdesc: model.description; pfileKind: model.fileKind
                }
            }
            // Layer / blending / mask ---------------------------------------
            ColumnLayout {
                visible: controller.selectedStepId !== "" && controller.selectedMaskable
                Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16; Layout.topMargin: 8
                spacing: 6
                Rectangle { Layout.fillWidth: true; height: 1; color: Theme.outlineVariant; opacity: 0.6 }
                Label { text: "Layer"; font.pixelSize: 13; font.weight: Font.DemiBold; color: Theme.text; Layout.topMargin: 6 }
                RowLayout {
                    Label { text: "Opacity"; color: Theme.textMuted; font.pixelSize: 13; Layout.preferredWidth: 70 }
                    Slider { Layout.fillWidth: true; from: 0; to: 1; value: controller.selectedOpacity
                        onMoved: controller.setOpacityLive(value); onPressedChanged: if (!pressed) controller.commitOpacity() }
                    Label { text: Math.round(controller.selectedOpacity * 100) + "%"; color: Theme.textMuted; font.pixelSize: 12; Layout.preferredWidth: 38 }
                }
                RowLayout {
                    Label { text: "Blend"; color: Theme.textMuted; font.pixelSize: 13; Layout.preferredWidth: 70 }
                    ComboBox { Layout.fillWidth: true; model: controller.blendModes
                        currentIndex: controller.blendModes.indexOf(controller.selectedBlend)
                        onActivated: function(i) { controller.setBlend(controller.blendModes[i]) } }
                }
                RowLayout {
                    Label { text: "Mask"; color: Theme.textMuted; font.pixelSize: 13; Layout.preferredWidth: 70 }
                    ComboBox {
                        id: maskBox
                        Layout.fillWidth: true
                        textRole: "t"; valueRole: "v"
                        model: [{t: controller.selectedMask ? "Custom…" : "None", v: "keep"}, {t: "None", v: "none"},
                                {t: "Radial (centre)", v: "radial"}, {t: "Radial inverted (edges)", v: "radial_inverted"},
                                {t: "Graduated (top)", v: "top_gradient"}, {t: "Graduated (bottom)", v: "bottom_gradient"},
                                {t: "Shadows", v: "shadows"}, {t: "Highlights", v: "highlights"},
                                {t: "Skin tones", v: "skin"}, {t: "Everything except skin", v: "not_skin"}]
                        onActivated: function(i) { var v = model[i].v; if (v !== "keep") controller.setMaskPreset(v) }
                    }
                }
                RowLayout {
                    Label { text: "Brush"; color: Theme.textMuted; font.pixelSize: 13; Layout.preferredWidth: 70 }
                    Button { text: "Paint"; checkable: true; checked: controller.paintMode === "paint"; flat: true
                        onClicked: controller.setPaintMode(controller.paintMode === "paint" ? "" : "paint") }
                    Button { text: "Erase"; checkable: true; checked: controller.paintMode === "erase"; flat: true
                        onClicked: controller.setPaintMode(controller.paintMode === "erase" ? "" : "erase") }
                    CheckBox { text: "Show"; checked: controller.showMask; onToggled: controller.setShowMask(checked) }
                }
                RowLayout {
                    visible: controller.paintMode !== ""
                    Label { text: "Size"; color: Theme.textMuted; font.pixelSize: 13; Layout.preferredWidth: 70 }
                    Slider { Layout.fillWidth: true; from: 0.005; to: 0.3; value: controller.brushSize; onMoved: controller.setBrushSize(value) }
                }
                TextArea {
                    Layout.fillWidth: true
                    visible: controller.selectedMask !== ""
                    text: controller.selectedMask
                    font.family: "monospace"; font.pixelSize: 11
                    wrapMode: TextEdit.WrapAnywhere
                    placeholderText: "Mask JSON"
                    onEditingFinished: controller.setMaskJson(text)
                    background: Rectangle { radius: 8; color: Theme.surfaceHigh }
                }
            }
            Item { height: 16 }
        }
    }
}
