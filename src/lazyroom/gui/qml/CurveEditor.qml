import QtQuick
import QtQuick.Controls

// Interactive monotone tone-curve editor: drag points, double-click to add, right-click to remove.
Item {
    id: root
    property var points: [[0, 0], [1, 1]]
    signal moved(var pts)
    signal released(var pts)
    implicitHeight: width
    property int dragIndex: -1

    function toPx(p) { return Qt.point(p[0] * width, (1 - p[1]) * height) }
    function clamp(v) { return Math.max(0, Math.min(1, v)) }
    function sorted(pts) { return pts.slice().sort(function(a, b) { return a[0] - b[0] }) }

    Rectangle { anchors.fill: parent; radius: 6; color: Theme.surfaceHigh; border.color: Theme.outlineVariant }
    Canvas {
        id: cv
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            ctx.strokeStyle = Qt.rgba(Theme.textMuted.r, Theme.textMuted.g, Theme.textMuted.b, 0.18)
            ctx.lineWidth = 1
            for (var i = 1; i < 4; i++) {
                ctx.beginPath(); ctx.moveTo(i * width / 4, 0); ctx.lineTo(i * width / 4, height); ctx.stroke()
                ctx.beginPath(); ctx.moveTo(0, i * height / 4); ctx.lineTo(width, i * height / 4); ctx.stroke()
            }
            ctx.beginPath(); ctx.moveTo(0, height); ctx.lineTo(width, 0); ctx.stroke()
            var pts = root.sorted(root.points)
            ctx.strokeStyle = Theme.primary
            ctx.lineWidth = 2
            ctx.beginPath()
            // Catmull-Rom style smooth path through the points (display only; engine uses monotone cubic)
            for (var k = 0; k < pts.length; k++) {
                var p = root.toPx(pts[k])
                if (k === 0) ctx.moveTo(0, p.y)
                if (k === 0) ctx.lineTo(p.x, p.y)
                else {
                    var p0 = root.toPx(pts[Math.max(0, k - 2)]), p1 = root.toPx(pts[k - 1])
                    var p3 = root.toPx(pts[Math.min(pts.length - 1, k + 1)])
                    ctx.bezierCurveTo(p1.x + (p.x - p0.x) / 6, p1.y + (p.y - p0.y) / 6,
                                      p.x - (p3.x - p1.x) / 6, p.y - (p3.y - p1.y) / 6, p.x, p.y)
                }
            }
            var last = root.toPx(pts[pts.length - 1]); ctx.lineTo(width, last.y)
            ctx.stroke()
        }
    }
    Repeater {
        model: root.points.length
        Rectangle {
            property var pt: root.points[index] || [0, 0]
            x: pt[0] * root.width - width / 2; y: (1 - pt[1]) * root.height - height / 2
            width: 12; height: 12; radius: 6
            color: root.dragIndex === index ? Theme.primary : Theme.surface
            border.color: Theme.primary; border.width: 2
        }
    }
    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        function nearest(mx, my) {
            var best = -1, bd = 1e9
            for (var i = 0; i < root.points.length; i++) {
                var p = root.toPx(root.points[i]); var d = Math.hypot(p.x - mx, p.y - my)
                if (d < bd) { bd = d; best = i }
            }
            return bd < 14 ? best : -1
        }
        onPressed: function(e) {
            var i = nearest(e.x, e.y)
            if (e.button === Qt.RightButton) {
                if (i > 0 && root.points.length > 2) {
                    var pts = root.points.slice(); pts.splice(i, 1); root.points = pts; cv.requestPaint(); root.released(pts)
                }
                return
            }
            root.dragIndex = i
        }
        onDoubleClicked: function(e) {
            var pts = root.points.slice()
            pts.push([root.clamp(e.x / width), root.clamp(1 - e.y / height)])
            root.points = root.sorted(pts); cv.requestPaint(); root.released(root.points)
        }
        onPositionChanged: function(e) {
            if (root.dragIndex < 0) return
            var pts = root.points.slice()
            pts[root.dragIndex] = [root.clamp(e.x / width), root.clamp(1 - e.y / height)]
            root.points = pts; cv.requestPaint(); root.moved(root.sorted(pts))
        }
        onReleased: { if (root.dragIndex >= 0) { root.dragIndex = -1; root.released(root.sorted(root.points)) } }
    }
    onPointsChanged: cv.requestPaint()
    Connections { target: Theme; function onDarkChanged() { cv.requestPaint() } }
}
