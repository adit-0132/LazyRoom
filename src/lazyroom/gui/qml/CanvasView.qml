import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Material
import QtQuick.Dialogs

// Image canvas: fit/zoom/pan, before-after split, drag & drop import.
Rectangle {
    id: root
    color: Theme.canvas
    clip: true
    property real zoom: 1.0
    property real panX: 0
    property real panY: 0
    property real split: 0.5
    readonly property real iw: img.status === Image.Ready ? img.implicitWidth : 0
    readonly property real ih: img.status === Image.Ready ? img.implicitHeight : 0
    readonly property real fitZoom: iw > 0 ? Math.min((width - 64) / iw, (height - 64) / ih) : 1
    readonly property real effectiveZoom: win.fitMode ? fitZoom : zoom
    property real previewScale: 1

    function zoomBy(f, cx, cy) {
        if (iw <= 0) return
        if (cx === undefined) { cx = width / 2; cy = height / 2 }
        var old = effectiveZoom
        if (win.fitMode) { panX = 0; panY = 0; win.fitMode = false }
        zoom = Math.max(0.05, Math.min(32, old * f))
        var k = zoom / old
        // keep the point under the cursor fixed
        panX = (panX - (cx - width / 2)) * k + (cx - width / 2)
        panY = (panY - (cy - height / 2)) * k + (cy - height / 2)
    }
    onWidthChanged: sizeTimer.restart()
    onHeightChanged: sizeTimer.restart()
    Timer { id: sizeTimer; interval: 250; onTriggered: controller.setPreviewSize(Math.max(root.width, root.height) * Screen.devicePixelRatio) }
    Component.onCompleted: sizeTimer.start()

    // subtle checker for transparency is drawn by the backend; here a soft vignette frame
    Image {
        id: img
        source: controller.previewSource
        cache: false
        asynchronous: false
        smooth: true
        mipmap: true
        visible: controller.projectOpen
        width: root.iw * root.effectiveZoom
        height: root.ih * root.effectiveZoom
        x: (root.width - width) / 2 + (win.fitMode ? 0 : root.panX)
        y: (root.height - height) / 2 + (win.fitMode ? 0 : root.panY)
    }
    Image {  // local-adjustment mask overlay (red) for the selected step
        visible: controller.showMask && controller.projectOpen && controller.overlayMode === ""
        source: controller.maskSource
        cache: false
        smooth: true
        x: img.x; y: img.y; width: img.width; height: img.height
        z: 4
    }
    MouseArea {  // brush painting of masks
        id: painter
        visible: controller.paintMode !== "" && controller.projectOpen
        x: img.x; y: img.y; width: img.width; height: img.height
        z: 6
        hoverEnabled: true
        cursorShape: Qt.BlankCursor
        property point pos: Qt.point(-100, -100)
        function nx(px) { return Math.max(0, Math.min(1, px / width)) }
        function ny(py) { return Math.max(0, Math.min(1, py / height)) }
        onPressed: function(e) { controller.strokeBegin(nx(e.x), ny(e.y)) }
        onPositionChanged: function(e) { pos = Qt.point(e.x, e.y); if (pressed) controller.strokeTo(nx(e.x), ny(e.y)) }
        onReleased: controller.strokeEnd()
        Rectangle {  // brush cursor
            property real r: controller.brushSize * Math.min(painter.width, painter.height)
            x: painter.pos.x - r; y: painter.pos.y - r; width: 2 * r; height: 2 * r; radius: r
            color: "transparent"; border.color: controller.paintMode === "erase" ? "#FF6B6B" : "white"; border.width: 1.5
        }
    }
    MouseArea {  // eyedropper (white balance neutral point)
        visible: controller.overlayMode === "pick" && controller.projectOpen
        x: img.x; y: img.y; width: img.width; height: img.height
        z: 6
        cursorShape: Qt.CrossCursor
        onClicked: function(e) { controller.pickNeutral(e.x / width, e.y / height) }
    }
    CropOverlay {
        visible: controller.overlayMode === "crop" && controller.projectOpen
        x: img.x; y: img.y; width: img.width; height: img.height
        rect: controller.cropRect
        z: 5
    }
    Rectangle {  // drop shadow behind the photo
        z: -1
        x: img.x - 1; y: img.y - 1; width: img.width + 2; height: img.height + 2
        color: "transparent"; border.color: Qt.rgba(0, 0, 0, Theme.dark ? 0.6 : 0.15); visible: img.visible
    }
    Item {  // original image, clipped to the left of the split line
        visible: win.compare && controller.projectOpen
        x: img.x; y: img.y
        width: Math.max(0, Math.min(img.width, root.split * root.width - img.x))
        height: img.height
        clip: true
        Image {
            source: controller.originalSource
            cache: false
            smooth: true; mipmap: true
            width: img.width; height: img.height
            fillMode: Image.PreserveAspectFit  // edits may change the geometry (crop, borders)
        }
    }
    Rectangle {
        visible: win.compare && controller.projectOpen
        x: root.split * root.width - 1; y: 0; width: 2; height: root.height
        color: "white"; opacity: 0.85
        Rectangle {
            anchors.centerIn: parent; width: 34; height: 34; radius: 17
            color: Theme.surfaceHigh; border.color: "white"; border.width: 2
            Label { anchors.centerIn: parent; text: "⟷"; color: Theme.text; font.pixelSize: 16 }
        }
        DragHandler {
            target: null
            xAxis.enabled: true; yAxis.enabled: false
            onCentroidChanged: if (active) root.split = Math.max(0.02, Math.min(0.98, (centroid.scenePosition.x - root.mapToItem(null, 0, 0).x) / root.width))
        }
    }
    Label {
        visible: win.compare && controller.projectOpen
        text: "BEFORE"; color: "white"; font.pixelSize: 11; font.letterSpacing: 1.5
        x: 14; y: 12; opacity: 0.8
        padding: 4
        background: Rectangle { color: Qt.rgba(0, 0, 0, 0.45); radius: 4 }
    }
    Label {
        visible: win.compare && controller.projectOpen
        text: "AFTER"; color: "white"; font.pixelSize: 11; font.letterSpacing: 1.5
        anchors.right: parent.right; anchors.rightMargin: 14; y: 12; opacity: 0.8
        padding: 4
        background: Rectangle { color: Qt.rgba(0, 0, 0, 0.45); radius: 4 }
    }

    WheelHandler {
        enabled: controller.projectOpen
        onWheel: function(ev) { root.zoomBy(ev.angleDelta.y > 0 ? 1.12 : 1 / 1.12, ev.x, ev.y) }
    }
    DragHandler {
        enabled: controller.projectOpen && !win.fitMode
        target: null
        property real sx: 0
        property real sy: 0
        onActiveChanged: if (active) { sx = root.panX; sy = root.panY }
        onTranslationChanged: { root.panX = sx + translation.x; root.panY = sy + translation.y }
    }
    TapHandler {
        enabled: controller.projectOpen
        onDoubleTapped: function(ev) { if (win.fitMode) root.zoomBy(2, ev.position.x, ev.position.y); else win.fitMode = true }
    }

    // Empty state ---------------------------------------------------------------
    Rectangle {
        visible: !controller.projectOpen
        anchors.centerIn: parent
        width: Math.min(520, parent.width - 60); height: 300
        radius: Theme.radiusXL
        color: drop.containsDrag ? Theme.primaryContainer : Theme.surfaceContainer
        border.color: drop.containsDrag ? Theme.primary : Theme.outlineVariant
        border.width: drop.containsDrag ? 2 : 1
        Behavior on color { ColorAnimation { duration: 150 } }
        ColumnLayout {
            anchors.centerIn: parent
            spacing: 14
            IconImageTint { iconName: "folder_image"; size: 52; tint: Theme.primary; Layout.alignment: Qt.AlignHCenter }
            Label { text: "Drop a photo here"; font.pixelSize: 22; font.weight: Font.DemiBold; color: Theme.text; Layout.alignment: Qt.AlignHCenter }
            Label { text: "JPEG, PNG, WebP, AVIF, HEIC, TIFF, RAW or a .lazyroom project"; color: Theme.textMuted; Layout.alignment: Qt.AlignHCenter }
            FilledButton { text: "Open image…"; Layout.alignment: Qt.AlignHCenter; onClicked: actOpen.trigger() }
            Label { text: "Prefer an AI agent? Run  lazyroom agent-setup  to connect via MCP."
                color: Theme.outline; font.pixelSize: 12; Layout.alignment: Qt.AlignHCenter }
        }
    }
    DropArea {
        id: drop
        anchors.fill: parent
        onDropped: function(d) { if (d.hasUrls) controller.openFile(d.urls[0]) }
    }

    // Zoom chip -------------------------------------------------------------------
    Rectangle {
        visible: controller.projectOpen
        anchors.left: parent.left; anchors.bottom: parent.bottom; anchors.margins: 14
        height: 40; radius: 20; width: zr.implicitWidth + 16
        color: Qt.rgba(Theme.surfaceHigh.r, Theme.surfaceHigh.g, Theme.surfaceHigh.b, 0.92)
        RowLayout {
            id: zr; anchors.centerIn: parent; spacing: 0
            IconButton { iconName: "zoom_out"; tip: "Zoom out"; onClicked: root.zoomBy(0.8) }
            Label { text: Math.round(root.effectiveZoom * 100) + "%"; color: Theme.text; font.pixelSize: 12; Layout.preferredWidth: 46; horizontalAlignment: Text.AlignHCenter }
            IconButton { iconName: "zoom_in"; tip: "Zoom in"; onClicked: root.zoomBy(1.25) }
            IconButton { iconName: "fit"; tip: "Fit"; onClicked: win.fitMode = true }
        }
    }
}
