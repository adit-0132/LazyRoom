import QtQuick

// Interactive crop rectangle over the (uncropped) preview. Coordinates are normalised 0..1.
Item {
    id: ov
    property var rect: [0, 0, 1, 1]
    property real rx: rect[0]
    property real ry: rect[1]
    property real rw: rect[2]
    property real rh: rect[3]
    onRectChanged: if (!dragging) { rx = rect[0]; ry = rect[1]; rw = rect[2]; rh = rect[3] }
    property bool dragging: false
    readonly property real minSize: 0.02

    function push() { controller.setCropLive(rx, ry, rw, rh) }
    function done() { dragging = false; controller.commitParams() }

    // dim outside
    Rectangle { color: "#99000000"; x: 0; y: 0; width: parent.width; height: ov.ry * parent.height }
    Rectangle { color: "#99000000"; x: 0; y: (ov.ry + ov.rh) * parent.height; width: parent.width; height: parent.height - y }
    Rectangle { color: "#99000000"; x: 0; y: ov.ry * parent.height; width: ov.rx * parent.width; height: ov.rh * parent.height }
    Rectangle { color: "#99000000"; x: (ov.rx + ov.rw) * parent.width; y: ov.ry * parent.height; width: parent.width - x; height: ov.rh * parent.height }

    Rectangle {
        id: frame
        x: ov.rx * ov.width; y: ov.ry * ov.height; width: ov.rw * ov.width; height: ov.rh * ov.height
        color: "transparent"; border.color: "white"; border.width: 1.5
        Repeater {  // rule of thirds
            model: 2
            Rectangle { x: frame.width * (index + 1) / 3; width: 1; height: frame.height; color: "#88FFFFFF" }
        }
        Repeater {
            model: 2
            Rectangle { y: frame.height * (index + 1) / 3; height: 1; width: frame.width; color: "#88FFFFFF" }
        }
        MouseArea {  // move
            anchors.fill: parent; anchors.margins: 14
            cursorShape: Qt.SizeAllCursor
            property point p0
            property var r0
            onPressed: function(e) { ov.dragging = true; p0 = mapToItem(ov, e.x, e.y); r0 = [ov.rx, ov.ry] }
            onPositionChanged: function(e) {
                var p = mapToItem(ov, e.x, e.y)
                ov.rx = Math.max(0, Math.min(1 - ov.rw, r0[0] + (p.x - p0.x) / ov.width))
                ov.ry = Math.max(0, Math.min(1 - ov.rh, r0[1] + (p.y - p0.y) / ov.height))
                ov.push()
            }
            onReleased: ov.done()
        }
    }
    Repeater {  // 8 handles: corners and edges
        model: [[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 1], [0, 1], [0, 0.5]]
        Rectangle {
            property var h: modelData
            width: 16; height: 16; radius: 8; color: "white"; border.color: "#33000000"
            x: (ov.rx + ov.rw * h[0]) * ov.width - 8; y: (ov.ry + ov.rh * h[1]) * ov.height - 8
            MouseArea {
                anchors.fill: parent; anchors.margins: -8
                cursorShape: (h[0] === 0.5) ? Qt.SizeVerCursor : (h[1] === 0.5 ? Qt.SizeHorCursor
                             : ((h[0] === h[1]) ? Qt.SizeFDiagCursor : Qt.SizeBDiagCursor))
                onPressed: ov.dragging = true
                onPositionChanged: function(e) {
                    var p = mapToItem(ov, e.x, e.y)
                    var nx = Math.max(0, Math.min(1, p.x / ov.width)), ny = Math.max(0, Math.min(1, p.y / ov.height))
                    var x0 = ov.rx, y0 = ov.ry, x1 = ov.rx + ov.rw, y1 = ov.ry + ov.rh
                    if (h[0] === 0) x0 = Math.min(nx, x1 - ov.minSize)
                    if (h[0] === 1) x1 = Math.max(nx, x0 + ov.minSize)
                    if (h[1] === 0) y0 = Math.min(ny, y1 - ov.minSize)
                    if (h[1] === 1) y1 = Math.max(ny, y0 + ov.minSize)
                    ov.rx = x0; ov.ry = y0; ov.rw = x1 - x0; ov.rh = y1 - y0
                    ov.push()
                }
                onReleased: ov.done()
            }
        }
    }
}
