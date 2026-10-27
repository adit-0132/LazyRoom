pragma Singleton
import QtQuick

// Material 3 colour roles (baseline violet scheme) for dark and light themes.
QtObject {
    property bool dark: true
    readonly property color background: dark ? "#141218" : "#FEF7FF"
    readonly property color surface: dark ? "#141218" : "#FEF7FF"
    readonly property color surfaceLow: dark ? "#1D1B20" : "#F7F2FA"
    readonly property color surfaceContainer: dark ? "#211F26" : "#F3EDF7"
    readonly property color surfaceHigh: dark ? "#2B2930" : "#ECE6F0"
    readonly property color surfaceHighest: dark ? "#36343B" : "#E6E0E9"
    readonly property color canvas: dark ? "#0E0D11" : "#DDD7E2"
    readonly property color text: dark ? "#E6E0E9" : "#1D1B20"
    readonly property color textMuted: dark ? "#CAC4D0" : "#49454F"
    readonly property color outline: dark ? "#938F99" : "#79747E"
    readonly property color outlineVariant: dark ? "#49454F" : "#CAC4D0"
    readonly property color primary: dark ? "#D0BCFF" : "#6750A4"
    readonly property color primaryText: dark ? "#381E72" : "#FFFFFF"
    readonly property color primaryContainer: dark ? "#4F378B" : "#EADDFF"
    readonly property color primaryContainerText: dark ? "#EADDFF" : "#21005D"
    readonly property color secondaryContainer: dark ? "#4A4458" : "#E8DEF8"
    readonly property color secondaryContainerText: dark ? "#E8DEF8" : "#1D192B"
    readonly property color tertiary: dark ? "#EFB8C8" : "#7D5260"
    readonly property color error: dark ? "#F2B8B5" : "#B3261E"
    readonly property int radiusS: 8
    readonly property int radiusM: 12
    readonly property int radiusL: 16
    readonly property int radiusXL: 28
    readonly property string fontFamily: Qt.application.font.family
}
