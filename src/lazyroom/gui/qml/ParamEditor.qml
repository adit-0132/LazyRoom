import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs

// One parameter control, generated from the operation's parameter schema.
ColumnLayout {
    id: root
    property string pname
    property string ptitle
    property string ptype
    property var pmin
    property var pmax
    property var pstep
    property var pvalue
    property var pchoices: []
    property string punit
    property string pdesc
    property string pfileKind
    spacing: 2

    function fmt(v) {
        if (v === undefined || v === null) return ""
        if (ptype === "int") return Math.round(v).toString()
        var r = Math.abs(pmax - pmin) <= 2 ? 2 : (Math.abs(pmax - pmin) <= 20 ? 1 : 0)
        return Number(v).toFixed(r)
    }

    RowLayout {
        Layout.fillWidth: true
        visible: root.ptype !== "bool"
        Label {
            text: root.ptitle + (root.punit && root.punit !== "px" ? "  (" + root.punit + ")" : "")
            color: Theme.textMuted
            font.pixelSize: 13
            elide: Text.ElideRight
            Layout.fillWidth: true
            ToolTip.visible: hint.hovered && root.pdesc.length > 0
            ToolTip.text: root.pdesc
            HoverHandler { id: hint }
        }
        TextField {
            id: numField
            visible: root.ptype === "float" || root.ptype === "int"
            text: root.fmt(root.pvalue)
            horizontalAlignment: Text.AlignRight
            font.pixelSize: 13
            Layout.preferredWidth: 64
            Layout.preferredHeight: 30
            topPadding: 4; bottomPadding: 4
            selectByMouse: true
            validator: DoubleValidator {}
            background: Rectangle { radius: 6; color: Theme.surfaceHigh; border.color: numField.activeFocus ? Theme.primary : "transparent" }
            onEditingFinished: {
                var v = Number(text)
                if (isNaN(v)) return
                if (root.pmin !== null && root.pmin !== undefined) v = Math.max(root.pmin, v)
                if (root.pmax !== null && root.pmax !== undefined) v = Math.min(root.pmax, v)
                controller.setParam(root.pname, v)
            }
        }
    }

    Loader {
        Layout.fillWidth: true
        sourceComponent: {
            switch (root.ptype) {
            case "float": case "int":
                // unbounded numbers (e.g. pixel crop values) get only the text field
                return (root.pmin === null || root.pmin === undefined || root.pmax === null || root.pmax === undefined) ? null : sliderC
            case "bool": return boolC
            case "choice": return choiceC
            case "color": return colorC
            case "points": return curveC
            case "point": return pointC
            default: return textC
            }
        }
    }

    Component {
        id: sliderC
        Slider {
            id: s
            from: root.pmin !== null && root.pmin !== undefined ? root.pmin : 0
            to: root.pmax !== null && root.pmax !== undefined ? root.pmax : 100
            stepSize: root.ptype === "int" ? 1 : (root.pstep ? root.pstep : 0)
            value: Number(root.pvalue)
            live: true
            topPadding: 0; bottomPadding: 0
            onMoved: controller.setParamLive(root.pname, root.ptype === "int" ? Math.round(value) : value)
            onPressedChanged: if (!pressed) controller.commitParams()
            // double-click resets to default (0 when in range, else min)
            TapHandler { onDoubleTapped: controller.setParam(root.pname, (s.from <= 0 && s.to >= 0) ? 0 : s.from) }
        }
    }
    Component {
        id: boolC
        Switch {
            text: root.ptitle
            checked: root.pvalue === true
            onToggled: controller.setParam(root.pname, checked)
        }
    }
    Component {
        id: choiceC
        ComboBox {
            model: root.pchoices
            currentIndex: Math.max(0, root.pchoices.indexOf(root.pvalue))
            onActivated: function(i) { controller.setParam(root.pname, root.pchoices[i]) }
            Layout.preferredHeight: 40
        }
    }
    Component {
        id: colorC
        RowLayout {
            Rectangle {
                width: 36; height: 24; radius: 6
                color: root.pvalue || "#000000"
                border.color: Theme.outlineVariant
                MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: cdlg.open() }
            }
            Label { text: root.pvalue; color: Theme.textMuted; font.family: "monospace"; font.pixelSize: 12 }
            ColorDialog {
                id: cdlg
                selectedColor: root.pvalue || "#000000"
                options: ColorDialog.ShowAlphaChannel
                onAccepted: {
                    var c = selectedColor
                    function h(x) { var s = Math.round(x * 255).toString(16); return s.length < 2 ? "0" + s : s }
                    controller.setParam(root.pname, "#" + h(c.r) + h(c.g) + h(c.b) + (c.a < 1 ? h(c.a) : ""))
                }
            }
        }
    }
    Component {
        id: curveC
        CurveEditor {
            width: parent ? parent.width : 200
            height: width * 0.8
            points: root.pvalue || [[0, 0], [1, 1]]
            onMoved: function(p) { controller.setParamLive(root.pname, p) }
            onReleased: function(p) { controller.setParamLive(root.pname, p); controller.commitParams() }
        }
    }
    Component {
        id: pointC
        RowLayout {
            Label { text: "x"; color: Theme.textMuted }
            Slider { Layout.fillWidth: true; from: 0; to: 1; value: root.pvalue ? root.pvalue[0] : 0.5
                onMoved: controller.setParamLive(root.pname, [value, root.pvalue[1]])
                onPressedChanged: if (!pressed) controller.commitParams() }
            Label { text: "y"; color: Theme.textMuted }
            Slider { Layout.fillWidth: true; from: 0; to: 1; value: root.pvalue ? root.pvalue[1] : 0.5
                onMoved: controller.setParamLive(root.pname, [root.pvalue[0], value])
                onPressedChanged: if (!pressed) controller.commitParams() }
        }
    }
    Component {
        id: textC
        RowLayout {
            TextField {
                id: tf
                Layout.fillWidth: true
                text: (typeof root.pvalue === "string") ? root.pvalue : JSON.stringify(root.pvalue)
                font.pixelSize: 13
                selectByMouse: true
                placeholderText: root.ptype === "file" ? "path to " + root.pfileKind + " file" : ""
                onEditingFinished: {
                    var v = text
                    if (root.ptype !== "text" && root.ptype !== "file") { try { v = JSON.parse(text) } catch (e) { v = text } }
                    controller.setParam(root.pname, v)
                }
            }
            IconButton {
                visible: root.ptype === "file"
                iconName: "open"; tip: "Browse…"
                onClicked: fdlg.open()
            }
            FileDialog {
                id: fdlg
                onAccepted: controller.setParam(root.pname, controller.urlToPath(selectedFile))
            }
        }
    }
}
