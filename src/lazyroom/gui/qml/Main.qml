import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    id: win
    width: 1480; height: 920
    minimumWidth: 1040; minimumHeight: 640
    visible: true
    title: controller.projectOpen ? controller.projectName + " — LazyRoom" : "LazyRoom"
    color: Theme.background

    Material.theme: Theme.dark ? Material.Dark : Material.Light
    Material.accent: Theme.primary
    Material.primary: Theme.primary
    Material.background: Theme.surface
    Material.foreground: Theme.text
    Material.roundedScale: Material.MediumScale

    Binding { target: Theme; property: "dark"; value: controller.darkTheme }

    property real zoom: 1.0
    property bool fitMode: true
    property bool compare: false
    property bool showAi: false

    // ------------------------------------------------------------------ actions
    Action { id: actOpen; text: "Open…"; shortcut: StandardKey.Open; onTriggered: openDlg.open() }
    Action { id: actSave; text: "Save project"; shortcut: StandardKey.Save; enabled: controller.projectOpen
        onTriggered: controller.projectPath ? controller.saveProject("") : saveDlg.open() }
    Action { id: actSaveAs; text: "Save project as…"; shortcut: "Ctrl+Shift+S"; enabled: controller.projectOpen; onTriggered: saveDlg.open() }
    Action { id: actExport; text: "Export…"; shortcut: "Ctrl+E"; enabled: controller.projectOpen; onTriggered: exportDlg.open() }
    Action { id: actUndo; text: "Undo"; shortcut: StandardKey.Undo; enabled: controller.canUndo; onTriggered: controller.undo() }
    Action { id: actRedo; text: "Redo"; shortcut: StandardKey.Redo; enabled: controller.canRedo; onTriggered: controller.redo() }
    Action { id: actCompare; text: "Before / after"; shortcut: "\\"; checkable: true; checked: win.compare; onTriggered: win.compare = !win.compare }
    Action { id: actFit; text: "Fit"; shortcut: "Ctrl+0"; onTriggered: { win.fitMode = true } }
    Action { id: actOne; text: "100%"; shortcut: "Ctrl+1"; onTriggered: { win.fitMode = false; win.zoom = 1.0 / Screen.devicePixelRatio * canvas.previewScale } }
    Action { id: actZoomIn; shortcut: StandardKey.ZoomIn; onTriggered: canvas.zoomBy(1.25) }
    Action { id: actZoomOut; shortcut: StandardKey.ZoomOut; onTriggered: canvas.zoomBy(0.8) }
    Action { id: actSearch; shortcut: StandardKey.Find; onTriggered: search.forceActiveFocus() }
    Action { id: actAi; text: "AI assistant"; shortcut: "Ctrl+I"; onTriggered: win.showAi = !win.showAi }
    Action { id: actDelete; shortcut: StandardKey.Delete; enabled: controller.selectedStepId !== ""
        onTriggered: controller.removeStep(controller.selectedStepId) }
    Action { id: actDup; shortcut: "Ctrl+D"; enabled: controller.selectedStepId !== ""; onTriggered: controller.duplicateStep(controller.selectedStepId) }
    Action { id: actTheme; shortcut: "Ctrl+T"; onTriggered: controller.toggleTheme() }
    Action { id: actFull; shortcut: StandardKey.FullScreen; onTriggered: win.visibility = win.visibility === Window.FullScreen ? Window.Windowed : Window.FullScreen }

    // ------------------------------------------------------------------ header
    header: ToolBar {
        Material.elevation: 0
        height: 60
        background: Rectangle { color: Theme.surface; Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.outlineVariant; opacity: 0.6 } }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 14; anchors.rightMargin: 10
            spacing: 4
            IconImageTint { iconName: "logo"; tint: Theme.primary; size: 30; Layout.rightMargin: 6 }
            ColumnLayout {
                spacing: 0
                Label { text: "LazyRoom"; font.pixelSize: 16; font.weight: Font.DemiBold; color: Theme.text }
                Label { text: controller.projectOpen ? controller.imageInfo : "AI-native image toolbox"; font.pixelSize: 11; color: Theme.textMuted
                    elide: Text.ElideMiddle; Layout.maximumWidth: 380 }
            }
            Item { Layout.preferredWidth: 18 }
            IconButton { iconName: "open"; tip: "Open image or project (Ctrl+O)"; action: actOpen; text: "" }
            IconButton { iconName: "save"; tip: "Save project (Ctrl+S)"; onClicked: actSave.trigger(); enabled: actSave.enabled }
            Rectangle { width: 1; height: 24; color: Theme.outlineVariant; Layout.leftMargin: 6; Layout.rightMargin: 6 }
            IconButton { iconName: "undo"; tip: "Undo (Ctrl+Z)"; enabled: controller.canUndo; onClicked: controller.undo() }
            IconButton { iconName: "redo"; tip: "Redo (Ctrl+Shift+Z)"; enabled: controller.canRedo; onClicked: controller.redo() }
            Rectangle { width: 1; height: 24; color: Theme.outlineVariant; Layout.leftMargin: 6; Layout.rightMargin: 6 }
            IconButton { iconName: "compare"; tip: "Before / after (\\)"; active: win.compare; enabled: controller.projectOpen; onClicked: win.compare = !win.compare }
            IconButton { iconName: "fit"; tip: "Fit to window (Ctrl+0)"; onClicked: win.fitMode = true; enabled: controller.projectOpen }
            Item { Layout.fillWidth: true }
            BusyIndicator { running: controller.rendering; visible: running; Layout.preferredWidth: 28; Layout.preferredHeight: 28 }
            IconButton { iconName: "build"; tip: "Tools (palette, collage, OCR, QR, HDR, panorama, textures…)"; onClicked: toolsDlg.open() }
            IconButton { iconName: "batch"; tip: "Batch process a folder"; onClicked: batchDlg.open() }
            IconButton { iconName: "sparkle"; tip: "AI assistant (Ctrl+I)"; active: win.showAi; onClicked: win.showAi = !win.showAi }
            IconButton { iconName: Theme.dark ? "sun" : "moon"; tip: "Toggle theme (Ctrl+T)"; onClicked: controller.toggleTheme() }
            IconButton { iconName: "settings"; tip: "Settings"; onClicked: settingsDlg.open() }
            FilledButton {
                text: "Export"
                icon.source: "icons/export.svg"; icon.width: 18; icon.height: 18
                enabled: controller.projectOpen
                Layout.leftMargin: 8
                onClicked: exportDlg.open()
            }
        }
    }

    footer: Rectangle {
        height: 28; color: Theme.surfaceLow
        RowLayout {
            anchors.fill: parent; anchors.leftMargin: 14; anchors.rightMargin: 14
            Label { text: controller.status; color: Theme.textMuted; font.pixelSize: 12; elide: Text.ElideRight; Layout.fillWidth: true }
            Label { text: controller.projectOpen ? Math.round(canvas.effectiveZoom * 100) + "%" : ""; color: Theme.textMuted; font.pixelSize: 12 }
            Label { text: "MCP: lazyroom mcp"; color: Theme.outline; font.pixelSize: 12; Layout.leftMargin: 16 }
        }
    }

    // ------------------------------------------------------------------ layout
    RowLayout {
        anchors.fill: parent
        spacing: 0

        // Navigation rail -----------------------------------------------------
        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: 84
            color: Theme.surface
            ListView {
                id: rail
                anchors.fill: parent; anchors.topMargin: 10
                model: controller.categoriesModel
                spacing: 4
                interactive: false
                delegate: ItemDelegate {
                    width: rail.width; height: 62
                    property bool sel: controller.category === model.id
                    onClicked: { controller.setCategory(model.id); search.text = "" }
                    background: null
                    contentItem: ColumnLayout {
                        spacing: 3
                        Rectangle {
                            Layout.alignment: Qt.AlignHCenter
                            width: 56; height: 32; radius: 16
                            color: sel ? Theme.secondaryContainer : (parent.parent.hovered ? Qt.rgba(Theme.text.r, Theme.text.g, Theme.text.b, 0.08) : "transparent")
                            Behavior on color { ColorAnimation { duration: 150 } }
                            IconImageTint { anchors.centerIn: parent; iconName: model.icon; tint: sel ? Theme.secondaryContainerText : Theme.textMuted }
                        }
                        Label { text: model.title; font.pixelSize: 11; font.weight: sel ? Font.DemiBold : Font.Normal
                            color: sel ? Theme.text : Theme.textMuted; Layout.alignment: Qt.AlignHCenter }
                    }
                }
            }
        }

        // Operation browser ----------------------------------------------------
        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: 270
            color: Theme.surfaceLow
            radius: 0
            ColumnLayout {
                anchors.fill: parent; anchors.margins: 12
                spacing: 10
                TextField {
                    id: search
                    Layout.fillWidth: true
                    placeholderText: "Search 600+ tools (Ctrl+F)"
                    leftPadding: 38
                    onTextChanged: controller.setSearch(text)
                    background: Rectangle { radius: 24; color: Theme.surfaceHigh; implicitHeight: 44
                        border.color: search.activeFocus ? Theme.primary : "transparent" }
                    IconImageTint { iconName: "search"; tint: Theme.textMuted; x: 12; anchors.verticalCenter: parent.verticalCenter }
                }
                Flow {
                    Layout.fillWidth: true
                    spacing: 6
                    visible: controller.groupsModel.count > 0 && search.text === ""
                    Repeater {
                        model: controller.groupsModel
                        delegate: Rectangle {
                            property bool sel: groupSel === model.id
                            height: 28; radius: 8
                            width: gl.implicitWidth + 20
                            color: sel ? Theme.secondaryContainer : "transparent"
                            border.color: sel ? "transparent" : Theme.outlineVariant
                            Label { id: gl; anchors.centerIn: parent; text: model.title; font.pixelSize: 12; color: sel ? Theme.secondaryContainerText : Theme.textMuted }
                            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: { groupSel = model.id; controller.setGroup(model.id) } }
                        }
                    }
                }
                Label {
                    text: controller.opsModel.count + (controller.category === "looks" && search.text === "" ? " looks" : " tools")
                    color: Theme.outline; font.pixelSize: 11
                }
                ListView {
                    id: opsList
                    Layout.fillWidth: true; Layout.fillHeight: true
                    clip: true
                    model: controller.opsModel
                    spacing: 2
                    ScrollBar.vertical: ScrollBar {}
                    delegate: ItemDelegate {
                        id: opd
                        width: opsList.width - 6
                        height: 44
                        enabled: model.available
                        onClicked: model.kind === "preset" ? controller.applyPreset(model.id, 1.0) : controller.addOperation(model.id)
                        ToolTip.visible: hovered && !model.available
                        ToolTip.text: "Requires an optional backend (see Settings › Backends)"
                        background: Rectangle { radius: 12; color: opd.hovered ? Theme.surfaceHigh : "transparent" }
                        contentItem: RowLayout {
                            spacing: 8
                            ColumnLayout {
                                spacing: 0; Layout.fillWidth: true
                                Label { text: model.title; color: model.available ? Theme.text : Theme.outline; font.pixelSize: 13; elide: Text.ElideRight; Layout.fillWidth: true }
                                Label { text: model.group; color: Theme.textMuted; font.pixelSize: 10; visible: text !== "" && (search.text !== "" || controller.category === "favorites" || controller.category === "recent") }
                            }
                            IconButton {
                                visible: model.kind === "op" && (opd.hovered || model.favorite)
                                iconName: model.favorite ? "star_filled" : "star"
                                tint: model.favorite ? Theme.primary : Theme.textMuted
                                implicitWidth: 30; implicitHeight: 30; icon.width: 16; icon.height: 16
                                onClicked: controller.toggleFavorite(model.id)
                            }
                        }
                    }
                }
            }
        }

        // Canvas ---------------------------------------------------------------
        CanvasView {
            id: canvas
            Layout.fillWidth: true; Layout.fillHeight: true
        }

        // Right panel ----------------------------------------------------------
        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: 360
            color: Theme.surfaceLow
            ColumnLayout {
                anchors.fill: parent
                spacing: 0
                TabBar {
                    id: tabs
                    Layout.fillWidth: true
                    background: Rectangle { color: Theme.surfaceLow }
                    TabButton { text: "Edit" }
                    TabButton { text: "History" }
                    TabButton { text: "AI" }
                }
                StackLayout {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    currentIndex: win.showAi ? 2 : tabs.currentIndex
                    EditPanel {}
                    HistoryPanel {}
                    AiPanel {}
                }
            }
        }
    }

    property string groupSel: ""

    // ------------------------------------------------------------------ dialogs
    FileDialog {
        id: openDlg
        title: "Open image or project"
        nameFilters: ["Images and projects (*.jpg *.jpeg *.png *.webp *.avif *.heic *.heif *.tif *.tiff *.bmp *.gif *.jp2 *.qoi *.tga *.dng *.cr2 *.cr3 *.nef *.arw *.raf *.orf *.rw2 *.lazyroom *.pmproj)", "All files (*)"]
        onAccepted: controller.openFile(selectedFile)
    }
    FileDialog {
        id: saveDlg
        title: "Save project"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "lazyroom"
        nameFilters: ["LazyRoom project (*.lazyroom)"]
        onAccepted: controller.saveProject(selectedFile)
    }
    ExportDialog { id: exportDlg }
    ToolsDialog { id: toolsDlg; objectName: "toolsDlg" }
    BatchDialog { id: batchDlg; objectName: "batchDlg" }
    SettingsDialog { id: settingsDlg; objectName: "settingsDlg" }
    Toast { id: toast }
    Connections { target: controller; function onToast(msg) { toast.show(msg) } }
    Connections { target: tabs; function onCurrentIndexChanged() { if (tabs.currentIndex !== 2) win.showAi = false } }
    onShowAiChanged: if (showAi) tabs.currentIndex = 2
}
