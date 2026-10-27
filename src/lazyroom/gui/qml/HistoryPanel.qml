import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    spacing: 0
    Label { text: "History"; font.pixelSize: 15; font.weight: Font.DemiBold; color: Theme.text; Layout.margins: 12 }
    Label { text: "Click an entry to restore it (restoring is itself undoable).\nEdits by agents and humans share one history."
        color: Theme.textMuted; font.pixelSize: 11; Layout.leftMargin: 12; Layout.bottomMargin: 6 }
    ListView {
        id: hl
        Layout.fillWidth: true; Layout.fillHeight: true
        clip: true
        model: controller.historyModel
        ScrollBar.vertical: ScrollBar {}
        delegate: ItemDelegate {
            width: hl.width - 16; x: 8; height: 46
            onClicked: if (!model.current) controller.revertTo(model.id)
            background: Rectangle { radius: 12; color: model.current ? Theme.secondaryContainer : (parent.hovered ? Theme.surfaceHigh : "transparent") }
            contentItem: RowLayout {
                IconImageTint { iconName: model.actor === "agent" ? "sparkle" : (model.actor === "system" ? "image" : "draw"); size: 16
                    tint: model.actor === "agent" ? Theme.primary : Theme.textMuted }
                ColumnLayout {
                    spacing: 0; Layout.fillWidth: true
                    Label { text: model.description; color: Theme.text; font.pixelSize: 12; elide: Text.ElideRight; Layout.fillWidth: true }
                    Label { text: model.time + " · " + model.actor; color: Theme.textMuted; font.pixelSize: 10 }
                }
                IconImageTint { visible: model.current; iconName: "check"; size: 16; tint: Theme.primary }
            }
        }
    }
}
