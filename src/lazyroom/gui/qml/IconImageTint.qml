import QtQuick
import QtQuick.Controls.impl

// Monochrome SVG icon tinted to a theme colour.
IconImage {
    property string iconName
    property color tint: Theme.textMuted
    property int size: 22
    source: iconName ? Qt.resolvedUrl("icons/" + iconName + ".svg") : ""
    color: tint
    sourceSize: Qt.size(size, size)
    width: size; height: size
}
