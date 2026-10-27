import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material

// Compact Material 3 icon button with tooltip.
ToolButton {
    id: control
    property string iconName: ""
    property string tip: ""
    property bool active: false
    property color tint: active ? Theme.secondaryContainerText : Theme.textMuted
    icon.source: iconName ? Qt.resolvedUrl("icons/" + iconName + ".svg") : ""
    icon.color: enabled ? tint : Qt.rgba(tint.r, tint.g, tint.b, 0.38)
    icon.width: 20
    icon.height: 20
    implicitWidth: 40
    implicitHeight: 40
    padding: 0
    ToolTip.visible: hovered && tip.length > 0
    ToolTip.delay: 600
    ToolTip.text: tip
    background: Rectangle {
        radius: height / 2
        color: control.active ? Theme.secondaryContainer
              : control.down ? Qt.rgba(Theme.text.r, Theme.text.g, Theme.text.b, 0.12)
              : control.hovered ? Qt.rgba(Theme.text.r, Theme.text.g, Theme.text.b, 0.08) : "transparent"
        Behavior on color { ColorAnimation { duration: 120 } }
    }
}
