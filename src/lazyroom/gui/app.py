"""Desktop application entry point (PySide6 / Qt Quick, Material 3 style).

``LAZYROOM_GUI_SCREENSHOT=out.png`` renders the window, saves a screenshot
and exits (used for automated UI checks; works with QT_QPA_PLATFORM=offscreen).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        from PySide6.QtCore import QTimer, QUrl
        from PySide6.QtGui import QGuiApplication, QIcon
        from PySide6.QtQml import QQmlApplicationEngine
        from PySide6.QtQuickControls2 import QQuickStyle
    except ImportError:
        print("The desktop GUI needs PySide6: pip install 'lazyroom[gui]'", file=sys.stderr)
        return 6

    from ..api import LazyRoom
    from .controller import Controller, PreviewProvider

    os.environ.setdefault("QT_QUICK_CONTROLS_MATERIAL_VARIANT", "Dense" if os.environ.get("LAZYROOM_DENSE") else "Normal")
    QGuiApplication.setApplicationName("LazyRoom")
    QGuiApplication.setOrganizationName("LazyRoom")
    QGuiApplication.setDesktopFileName("lazyroom")
    app = QGuiApplication([sys.argv[0]] + [a for a in argv if a.startswith("-")])
    qml_dir = Path(__file__).resolve().parent / "qml"
    app.setWindowIcon(QIcon(str(qml_dir / "icons" / "logo.svg")))
    QQuickStyle.setStyle("Material")

    api = LazyRoom()
    provider = PreviewProvider()
    controller = Controller(api, provider)
    engine = QQmlApplicationEngine()
    engine.addImageProvider("pm", provider)
    engine.rootContext().setContextProperty("controller", controller)
    engine.addImportPath(str(qml_dir))
    engine.load(QUrl.fromLocalFile(str(qml_dir / "Main.qml")))
    if not engine.rootObjects():
        print("Failed to load the user interface (QML errors above)", file=sys.stderr)
        return 70
    files = [a for a in argv if not a.startswith("-")]
    if files:
        QTimer.singleShot(0, lambda: controller.openFile(str(Path(files[0]).resolve())))
    shot = os.environ.get("LAZYROOM_GUI_SCREENSHOT")
    if shot:
        win = engine.rootObjects()[0]
        for key, val in (("compare", "LAZYROOM_GUI_COMPARE"), ("showAi", "LAZYROOM_GUI_AI")):
            if os.environ.get(val):
                win.setProperty(key, True)

        import json as _json

        actions = _json.loads(os.environ.get("LAZYROOM_GUI_ACTIONS", "[]"))

        def run_actions():
            for act in actions:
                name, *args = act
                if name in ("selectStep", "moveStepToRow", "removeStep", "toggleStep") and args and args[0] in ("first", "last"):
                    rows = controller.stepsModel.items()
                    if rows:
                        args = [rows[0 if args[0] == "first" else -1]["id"], *args[1:]]
                if name == "setProperty":
                    win.setProperty(args[0], args[1])
                    continue
                if name == "openDialog":
                    from PySide6.QtCore import QMetaObject, QObject

                    dlg = win.findChild(QObject, args[0])
                    if dlg is not None:
                        QMetaObject.invokeMethod(dlg, "open")
                    continue
                getattr(controller, name)(*args)

        if actions:
            QTimer.singleShot(800, run_actions)

        def capture():
            if controller.rendering:
                QTimer.singleShot(300, capture)
                return
            win.grabWindow().save(shot)
            app.quit()

        QTimer.singleShot(int(os.environ.get("LAZYROOM_GUI_DELAY", "2500")), capture)
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
