"""Offscreen GUI test: the QML loads, a photo opens, edits render, and the GUI uses the same project model."""
import os
import time

import pytest

pytest.importorskip("PySide6")
pytestmark = [pytest.mark.gui, pytest.mark.timeout(180)]


def test_gui_loads_and_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    from pathlib import Path

    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle

    from lazyroom.api import LazyRoom
    from lazyroom.core.io import save_image
    from lazyroom.gui.controller import Controller, PreviewProvider
    from tests.conftest import synthetic_image

    app = QGuiApplication.instance() or QGuiApplication([])
    QQuickStyle.setStyle("Material")
    api = LazyRoom()
    provider = PreviewProvider()
    ctl = Controller(api, provider)
    eng = QQmlApplicationEngine()
    eng.addImageProvider("pm", provider)
    eng.rootContext().setContextProperty("controller", ctl)
    qml = Path(__file__).resolve().parents[2] / "src/lazyroom/gui/qml/Main.qml"
    errors = []
    eng.warnings.connect(lambda ws: errors.extend(w.toString() for w in ws))
    eng.load(QUrl.fromLocalFile(str(qml)))
    assert eng.rootObjects(), errors

    def pump(cond, timeout=60):
        t0 = time.time()
        while not cond() and time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.01)
        return cond()

    src = tmp_path / "p.png"
    save_image(synthetic_image(300, 200), src)
    ctl.openFile(str(src))
    assert ctl.projectOpen
    assert pump(lambda: "current" in provider.images and not ctl.rendering)
    ctl.addOperation("adjust.exposure")
    assert ctl.stepsModel.count == 1 and ctl.paramsModel.count == 3
    v0 = ctl._preview_version
    ctl.setParamLive("ev", 1.0)
    ctl.commitParams()
    assert pump(lambda: ctl._preview_version > v0 and not ctl.rendering)
    state = api.project_state(ctl.pid)
    assert state["steps"][0]["params"]["ev"] == 1.0
    assert ctl.canUndo
    ctl.undo()
    assert api.project_state(ctl.pid)["steps"][0]["params"]["ev"] == 0.0
    ctl.applyPreset("teal-orange", 1.0)
    assert any(r["isGroup"] for r in ctl.stepsModel.items())
    ctl.runIntent("slightly warmer with a vignette")
    assert ctl.aiModel.count > 0
    img = provider.images["current"]
    assert img.width() > 0
    bad = [e for e in errors if "TypeError" in e or "ReferenceError" in e]
    assert not bad, bad
