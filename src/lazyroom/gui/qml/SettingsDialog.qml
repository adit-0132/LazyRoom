import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

PmDialog {
    id: dlg
    title: "Settings"
    modal: true
    anchors.centerIn: parent
    width: 620
    standardButtons: Dialog.Cancel | Dialog.Save
    property var s: ({})
    onAboutToShow: {
        s = JSON.parse(controller.settingsJson())
        prov.currentIndex = Math.max(0, ["none", "openai_compatible", "anthropic"].indexOf(s.llm_provider))
        base.text = s.llm_base_url; model.text = s.llm_model; keyenv.text = s.llm_api_key_env; sendimg.checked = s.llm_send_image
        reads.text = (s.read_roots || []).join(";"); writes.text = (s.write_roots || []).join(";")
        approval.currentIndex = s.approval_policy === "never" ? 1 : 0
        var b = s.backends; var parts = []
        for (var k in b) parts.push(k + ": " + (b[k].available ? "available" : "missing"))
        backends.text = parts.join("   ·   ")
    }
    onAccepted: controller.saveSettings(JSON.stringify({
        llm_provider: prov.currentText, llm_base_url: base.text, llm_model: model.text, llm_api_key_env: keyenv.text,
        llm_send_image: sendimg.checked, read_roots: reads.text, write_roots: writes.text,
        approval_policy: approval.currentIndex === 1 ? "never" : "explicit" }))
    ColumnLayout {
        anchors.fill: parent; spacing: 8
        Label { text: "AI assistant backend (optional)"; font.weight: Font.DemiBold; color: Theme.text }
        Label { text: "Without a backend the built-in rule engine interprets requests offline. Images are only sent if you enable it below."
            wrapMode: Text.Wrap; Layout.fillWidth: true; color: Theme.textMuted; font.pixelSize: 12 }
        RowLayout { Label { text: "Provider"; Layout.preferredWidth: 150; color: Theme.textMuted }
            ComboBox { id: prov; Layout.fillWidth: true; model: ["none", "openai_compatible", "anthropic"] } }
        RowLayout { Label { text: "Base URL"; Layout.preferredWidth: 150; color: Theme.textMuted }
            TextField { id: base; Layout.fillWidth: true; placeholderText: "http://localhost:11434/v1 (Ollama)" } }
        RowLayout { Label { text: "Model"; Layout.preferredWidth: 150; color: Theme.textMuted }
            TextField { id: model; Layout.fillWidth: true; placeholderText: "e.g. llama3.1 / claude-opus-5-5" } }
        RowLayout { Label { text: "API key env var"; Layout.preferredWidth: 150; color: Theme.textMuted }
            TextField { id: keyenv; Layout.fillWidth: true; placeholderText: "e.g. ANTHROPIC_API_KEY (the key itself is never stored)" } }
        CheckBox { id: sendimg; text: "Allow sending a preview image to the backend" }
        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.outlineVariant }
        Label { text: "Safety"; font.weight: Font.DemiBold; color: Theme.text }
        RowLayout { Label { text: "Readable folders"; Layout.preferredWidth: 150; color: Theme.textMuted }
            TextField { id: reads; Layout.fillWidth: true; placeholderText: "empty = home, temp, current folder (separate with ;)" } }
        RowLayout { Label { text: "Writable folders"; Layout.preferredWidth: 150; color: Theme.textMuted }
            TextField { id: writes; Layout.fillWidth: true; placeholderText: "empty = home, temp, current folder" } }
        RowLayout { Label { text: "Destructive actions"; Layout.preferredWidth: 150; color: Theme.textMuted }
            ComboBox { id: approval; Layout.fillWidth: true; model: ["Require explicit confirmation", "Never allow"] } }
        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.outlineVariant }
        Label { text: "Optional backends"; font.weight: Font.DemiBold; color: Theme.text }
        Label { id: backends; wrapMode: Text.Wrap; Layout.fillWidth: true; color: Theme.textMuted; font.pixelSize: 12 }
    }
}
