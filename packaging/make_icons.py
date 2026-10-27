"""Render the app icon (SVG) to PNG sizes and a Windows .ico using Qt's SVG renderer."""
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "src/lazyroom/gui/qml/icons/logo.svg"


def render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(str(SVG)).render(p)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(img.rect(), QColor("#6750A4"))  # Material 3 primary
    p.end()
    return img


def main() -> None:
    app = QGuiApplication(sys.argv[:1])  # noqa: F841
    out = ROOT / "packaging"
    for s in (16, 32, 48, 64, 128, 256, 512):
        render(s).save(str(out / f"icon-{s}.png"))
    from PIL import Image

    Image.open(out / "icon-256.png").save(out / "lazyroom.ico", sizes=[(s, s) for s in (16, 32, 48, 64, 128, 256)])
    print("icons written")


if __name__ == "__main__":
    main()
