import QtQuick
import QtQuick.Controls

Popup {
    id: toast
    property alias text: label.text
    x: (parent.width - width) / 2
    y: parent.height - height - 56
    padding: 0
    closePolicy: Popup.NoAutoClose
    background: Rectangle { radius: 10; color: Theme.dark ? "#E6E0E9" : "#322F35" }
    contentItem: Label {
        id: label
        leftPadding: 18; rightPadding: 18; topPadding: 12; bottomPadding: 12
        color: Theme.dark ? "#322F35" : "#F5EFF7"
        wrapMode: Text.Wrap
        width: Math.min(implicitWidth, 560)
    }
    enter: Transition { NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 150 } }
    exit: Transition { NumberAnimation { property: "opacity"; from: 1; to: 0; duration: 200 } }
    Timer { id: hide; interval: 3200; onTriggered: toast.close() }
    function show(msg) { text = msg; open(); hide.restart() }
}
