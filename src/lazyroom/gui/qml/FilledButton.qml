import QtQuick
import QtQuick.Controls
import QtQuick.Controls.impl

// Material 3 filled (primary) button with a pill background.
Button {
    id: b
    implicitHeight: 40
    leftPadding: icon.source.toString() !== "" ? 16 : 24
    rightPadding: 24
    font.weight: Font.Medium
    icon.color: enabled ? Theme.primaryText : Qt.rgba(Theme.text.r, Theme.text.g, Theme.text.b, 0.38)
    contentItem: IconLabel {
        spacing: 8
        display: b.display
        icon: b.icon
        text: b.text
        font: b.font
        color: b.enabled ? Theme.primaryText : Qt.rgba(Theme.text.r, Theme.text.g, Theme.text.b, 0.38)
    }
    background: Rectangle {
        implicitHeight: 40
        radius: height / 2
        color: !b.enabled ? Qt.rgba(Theme.text.r, Theme.text.g, Theme.text.b, 0.12)
             : b.down ? Qt.darker(Theme.primary, 1.12) : b.hovered ? Qt.lighter(Theme.primary, 1.05) : Theme.primary
        Behavior on color { ColorAnimation { duration: 120 } }
    }
}
