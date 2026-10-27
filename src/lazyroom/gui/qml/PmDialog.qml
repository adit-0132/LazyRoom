import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material

// Material 3 dialog surface with explicit colours (consistent in light and dark themes).
Dialog {
    id: d
    Material.theme: Theme.dark ? Material.Dark : Material.Light
    Material.background: Theme.surfaceContainer
    Material.accent: Theme.primary
    padding: 24
    topPadding: 8
    background: Rectangle { radius: Theme.radiusXL; color: Theme.surfaceContainer; border.color: Theme.outlineVariant; border.width: Theme.dark ? 1 : 0 }
    header: Label {
        text: d.title
        padding: 24; bottomPadding: 8
        font.pixelSize: 22
        color: Theme.text
        background: null
    }
    footer: DialogButtonBox {
        visible: count > 0
        padding: 16
        background: null
        Material.background: "transparent"
        standardButtons: d.standardButtons
    }
    Overlay.modal: Rectangle { color: Qt.rgba(0, 0, 0, Theme.dark ? 0.55 : 0.32) }
}
