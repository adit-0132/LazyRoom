"""Palette contact-sheet PDF (ImageToolbox feature/palette-pdf ``PalettePdfRenderer``, Apache-2.0).

A small, dependency-free vector PDF writer reproducing the Android ``PdfDocument`` layout: A4 pages of
595 x 842 pt, page background rgb(247, 247, 249), optional header (source filename as a 22 pt bold
title, palette name below in 12 pt), optional source image (fitted into the content width, at most
220 pt high, rounded 8 pt card with a 1 pt argb(28, 0, 0, 0) border) on the first page, then a grid of
colour cards (``columns`` 1..6, ``margin`` 0..198, ``spacing`` 0..36, card height
``clamp(cellWidth * 0.62 + caption, 64, 160)``, caption 38 / 26 / 0 pt), names in 10 pt bold, HEX in
9 pt monospace, page numbers bottom-right in 8 pt.  Text uses the standard-14 Helvetica /
Helvetica-Bold / Courier fonts (metrics from pypdf's bundled AFM tables) instead of Android's
Roboto / Droid Sans Mono, so ellipsizing widths differ slightly.
"""
from __future__ import annotations

import io
import math
import zlib
from dataclasses import dataclass

import numpy as np

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
FOOTER_HEIGHT = 12.0
CORNER_RADIUS = 8.0
K = 0.5522847498  # Bezier circle constant


@dataclass
class PdfColor:
    argb: int
    name: str
    hex: str


@dataclass
class PdfParams:
    columns: int = 3
    margin: float = 24.0
    spacing: float = 8.0
    include_source_image: bool = True
    show_source_filename: bool = False
    show_palette_name: bool = True
    show_color_names: bool = True
    show_hex_values: bool = True


def argb_hex(argb: int) -> str:
    """``Int.toHex()`` of the exporter: #RRGGBB when opaque, else #AARRGGBB."""
    return f"#{argb & 0xFFFFFF:06X}" if (argb >> 24) & 0xFF == 0xFF else f"#{argb & 0xFFFFFFFF:08X}"


class _Font:
    def __init__(self, base: str, res: str):
        from pypdf._codecs.core_font_metrics import CORE_FONT_METRICS

        m = CORE_FONT_METRICS[base]
        self.base, self.res = base, res
        self.widths = {k: float(v) for k, v in m.character_widths.items() if k != "default"}
        self.default = float(m.character_widths.get("default", 500))

    def encode(self, text: str) -> bytes:
        return text.encode("cp1252", errors="replace")

    def clean(self, text: str) -> str:
        return self.encode(text).decode("cp1252")

    def measure(self, text: str, size: float) -> float:
        return sum(self.widths.get(c, self.default) for c in self.clean(text)) * size / 1000.0


def _esc(b: bytes) -> bytes:
    return b.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)").replace(b"\r", b"\\r")


def _n(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


class _Page:
    def __init__(self):
        self.ops: list[str] = []
        self.gstates: dict[str, tuple[float, float]] = {}
        self.images: dict[str, int] = {}

    # Android canvas coordinates (y down) -> PDF (y up)
    @staticmethod
    def y(v: float) -> float:
        return PAGE_HEIGHT - v

    def _alpha(self, fill_a: float = 1.0, stroke_a: float = 1.0) -> None:
        if fill_a >= 1.0 and stroke_a >= 1.0:
            return
        key = f"GS{len(self.gstates)}"
        for k, v in self.gstates.items():
            if v == (fill_a, stroke_a):
                key = k
                break
        self.gstates[key] = (fill_a, stroke_a)
        self.ops.append(f"/{key} gs")

    def rect_path(self, l: float, t: float, r: float, b: float) -> str:
        return f"{_n(l)} {_n(self.y(b))} {_n(r - l)} {_n(b - t)} re"

    def rrect_path(self, l: float, t: float, r: float, b: float, rad: float) -> str:
        rad = max(0.0, min(rad, (r - l) / 2, (b - t) / 2))
        Y = self.y
        c = rad * K
        p = [f"{_n(l + rad)} {_n(Y(t))} m",
             f"{_n(r - rad)} {_n(Y(t))} l",
             f"{_n(r - rad + c)} {_n(Y(t))} {_n(r)} {_n(Y(t + rad - c))} {_n(r)} {_n(Y(t + rad))} c",
             f"{_n(r)} {_n(Y(b - rad))} l",
             f"{_n(r)} {_n(Y(b - rad + c))} {_n(r - rad + c)} {_n(Y(b))} {_n(r - rad)} {_n(Y(b))} c",
             f"{_n(l + rad)} {_n(Y(b))} l",
             f"{_n(l + rad - c)} {_n(Y(b))} {_n(l)} {_n(Y(b - rad + c))} {_n(l)} {_n(Y(b - rad))} c",
             f"{_n(l)} {_n(Y(t + rad))} l",
             f"{_n(l)} {_n(Y(t + rad - c))} {_n(l + rad - c)} {_n(Y(t))} {_n(l + rad)} {_n(Y(t))} c", "h"]
        return " ".join(p)

    def fill(self, path: str, rgb, alpha: float = 1.0) -> None:
        self.ops.append("q")
        self._alpha(fill_a=alpha)
        self.ops.append(f"{_n(rgb[0] / 255)} {_n(rgb[1] / 255)} {_n(rgb[2] / 255)} rg {path} f Q")

    def stroke(self, path: str, rgb, alpha: float = 1.0, width: float = 1.0) -> None:
        self.ops.append("q")
        self._alpha(stroke_a=alpha)
        self.ops.append(f"{_n(width)} w {_n(rgb[0] / 255)} {_n(rgb[1] / 255)} {_n(rgb[2] / 255)} RG {path} S Q")

    def text(self, s: str, x: float, baseline: float, font: _Font, size: float, rgb, align_right: bool = False) -> None:
        if align_right:
            x -= font.measure(s, size)
        data = _esc(font.encode(s)).decode("latin-1")
        self.ops.append(f"BT /{font.res} {_n(size)} Tf {_n(rgb[0] / 255)} {_n(rgb[1] / 255)} {_n(rgb[2] / 255)} rg "
                        f"{_n(x)} {_n(self.y(baseline))} Td ({data}) Tj ET")

    def image(self, name: str, obj: int, l: float, t: float, r: float, b: float, clip: str) -> None:
        self.images[name] = obj
        self.ops.append(f"q {clip} W n {_n(r - l)} 0 0 {_n(b - t)} {_n(l)} {_n(self.y(b))} cm /{name} Do Q")


class PalettePdf:
    def __init__(self, image: np.ndarray | None, colors: list[PdfColor], params: PdfParams,
                 source_filename: str, palette_name: str | None, image_dpi: int = 300):
        self.image = image
        self.colors = colors
        self.columns = min(6, max(1, params.columns))
        self.margin = min(PAGE_WIDTH / 3.0, max(0.0, params.margin))
        self.spacing = min(36.0, max(0.0, params.spacing))
        self.include_image = params.include_source_image and image is not None
        self.p = params
        self.filename = source_filename
        self.palette_name = palette_name
        self.dpi = image_dpi
        self.bold = _Font("Helvetica-Bold", "F1")
        self.regular = _Font("Helvetica", "F2")
        self.mono = _Font("Courier", "F3")
        self.pages: list[_Page] = []
        self.objects: list[bytes | None] = []
        self.layout: dict = {}

    # -- object helpers -------------------------------------------------
    def _obj(self, data: bytes | None = None) -> int:
        self.objects.append(data)
        return len(self.objects)

    def _image_object(self, draw_w: float, draw_h: float) -> int:
        import cv2
        from PIL import Image

        img = np.clip(self.image, 0, 1)
        if img.shape[2] == 4:  # composited over the white card
            img = img[..., :3] * img[..., 3:4] + (1 - img[..., 3:4])
        target_w = max(1, int(round(draw_w / 72.0 * self.dpi)))
        if img.shape[1] > target_w:
            th = max(1, int(round(img.shape[0] * target_w / img.shape[1])))
            img = cv2.resize(img.astype(np.float32), (target_w, th), interpolation=cv2.INTER_AREA)
        u8 = (img * 255 + 0.5).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(u8, "RGB").save(buf, "JPEG", quality=92, subsampling=0)
        data = buf.getvalue()
        head = (f"<< /Type /XObject /Subtype /Image /Width {u8.shape[1]} /Height {u8.shape[0]} "
                f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(data)} >>").encode()
        self.layout["image_pixels"] = [int(u8.shape[1]), int(u8.shape[0])]
        return self._obj(head + b"\nstream\n" + data + b"\nendstream")

    @staticmethod
    def _ellipsize(text: str, font: _Font, size: float, max_w: float) -> str:
        text = font.clean(text)
        if font.measure(text, size) <= max_w:
            return text
        end = len(text)
        while end > 0 and font.measure(text[:end] + "...", size) > max_w:
            end -= 1
        return text[:end] + "..."

    # -- PalettePdfRenderer.render ---------------------------------------
    def render(self) -> bytes:
        if not self.colors:
            raise ValueError("Palette must contain at least one color")
        m, sp, cols = self.margin, self.spacing, self.columns
        cell_w = (PAGE_WIDTH - m * 2 - sp * (cols - 1)) / cols
        if cell_w <= 0:
            raise ValueError("Invalid palette PDF layout")
        names, hexes = self.p.show_color_names, self.p.show_hex_values
        caption = 38.0 if (names and hexes) else 26.0 if (names or hexes) else 0.0
        cell_h = min(160.0, max(64.0, cell_w * 0.62 + caption))
        self.layout.update(cell_width=round(cell_w, 3), cell_height=round(cell_h, 3), caption_height=caption,
                           columns=cols, margin=m, spacing=sp)
        img_obj = None
        idx, page_no = 0, 1
        per_page = []
        while True:
            pg = _Page()
            pg.fill(pg.rect_path(0, 0, PAGE_WIDTH, PAGE_HEIGHT), (247, 247, 249))
            top = self._header(pg)
            if self.include_image and page_no == 1:
                width = PAGE_WIDTH - m * 2
                scale = min(width / self.image.shape[1], 220.0 / self.image.shape[0])
                dw, dh = self.image.shape[1] * scale, self.image.shape[0] * scale
                left = m + (width - dw) / 2
                if img_obj is None:
                    img_obj = self._image_object(dw, dh)
                path = pg.rrect_path(left, top, left + dw, top + dh, CORNER_RADIUS)
                pg.fill(path, (255, 255, 255))
                pg.image("Im1", img_obj, left, top, left + dw, top + dh, path)
                pg.stroke(path, (0, 0, 0), alpha=28 / 255)
                self.layout["image_rect"] = [round(left, 3), round(top, 3), round(dw, 3), round(dh, 3)]
                grid_top = top + dh + sp
            else:
                grid_top = top
            grid_bottom = PAGE_HEIGHT - m - FOOTER_HEIGHT
            rows = max(1, int(math.floor((grid_bottom - grid_top + sp) / (cell_h + sp))))
            cap = rows * cols
            chunk = self.colors[idx:idx + cap]
            for i, c in enumerate(chunk):
                col, row = i % cols, i // cols
                left = m + col * (cell_w + sp)
                t = grid_top + row * (cell_h + sp)
                self._card(pg, c, left, t, left + cell_w, t + cell_h, caption)
            per_page.append(len(chunk))
            idx += cap
            pg.text(str(page_no), PAGE_WIDTH - m, PAGE_HEIGHT - m / 2, self.regular, 8, (115, 115, 122),
                    align_right=True)
            self.pages.append(pg)
            page_no += 1
            if idx >= len(self.colors):
                break
        self.layout["colors_per_page"] = per_page
        return self._serialize()

    def _header(self, pg: _Page) -> float:
        m = self.margin
        lines = []
        if self.p.show_source_filename and self.filename.strip():
            lines.append(self.filename)
        if self.p.show_palette_name and self.palette_name and self.palette_name.strip():
            lines.append(self.palette_name)
        if not lines:
            return m
        maxw = PAGE_WIDTH - m * 2
        pg.text(self._ellipsize(lines[0], self.bold, 22, maxw), m, m + 22, self.bold, 22, (30, 30, 34))
        if len(lines) > 1:
            pg.text(self._ellipsize(" \u00b7 ".join(lines[1:]), self.regular, 12, maxw), m, m + 22 + 12 + 6,
                    self.regular, 12, (95, 95, 102))
        return m + 22 + (30.0 if len(lines) > 1 else 18.0)

    def _card(self, pg: _Page, c: PdfColor, l: float, t: float, r: float, b: float, caption: float) -> None:
        path = pg.rrect_path(l, t, r, b, CORNER_RADIUS)
        pg.fill(path, (255, 255, 255))
        a = ((c.argb >> 24) & 255) / 255.0
        rgb = ((c.argb >> 16) & 255, (c.argb >> 8) & 255, c.argb & 255)
        pg.ops.append(f"q {path} W n")
        pg.fill(pg.rect_path(l, t, r, b - caption), rgb, alpha=a)
        pg.ops.append("Q")
        if caption > 0:
            x, maxw = l + 10, (r - l) - 20
            sb = b - caption
            if self.p.show_color_names and self.p.show_hex_values:
                pg.text(self._ellipsize(c.name, self.bold, 10, maxw), x, sb + 15, self.bold, 10, (35, 35, 39))
                pg.text(self._ellipsize(c.hex.upper(), self.mono, 9, maxw), x, sb + 29, self.mono, 9, (95, 95, 102))
            elif self.p.show_color_names:
                pg.text(self._ellipsize(c.name, self.bold, 10, maxw), x, sb + 17, self.bold, 10, (35, 35, 39))
            else:
                pg.text(self._ellipsize(c.hex.upper(), self.mono, 9, maxw), x, sb + 17, self.mono, 9, (95, 95, 102))
        pg.stroke(path, (0, 0, 0), alpha=28 / 255)

    def _serialize(self) -> bytes:
        fonts = {f.res: self._obj(f"<< /Type /Font /Subtype /Type1 /BaseFont /{f.base} "
                                  f"/Encoding /WinAnsiEncoding >>".encode())
                 for f in (self.bold, self.regular, self.mono)}
        pages_id = self._obj(None)
        kids = []
        for pg in self.pages:
            content = "\n".join(pg.ops).encode("latin-1")
            comp = zlib.compress(content, 9)
            cid = self._obj(f"<< /Length {len(comp)} /Filter /FlateDecode >>".encode() + b"\nstream\n" + comp
                            + b"\nendstream")
            gs = " ".join(f"/{k} << /Type /ExtGState /ca {_n(v[0])} /CA {_n(v[1])} >>" for k, v in pg.gstates.items())
            xo = " ".join(f"/{k} {v} 0 R" for k, v in pg.images.items())
            res = f"/Font << {' '.join(f'/{k} {v} 0 R' for k, v in fonts.items())} >>"
            if gs:
                res += f" /ExtGState << {gs} >>"
            if xo:
                res += f" /XObject << {xo} >>"
            kids.append(self._obj(f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                                  f"/Resources << {res} >> /Contents {cid} 0 R >>".encode()))
        self.objects[pages_id - 1] = (f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] "
                                      f"/Count {len(kids)} >>").encode()
        info = self._obj(b"<< /Producer (LazyRoom palette_pdf) /Creator (LazyRoom) >>")
        catalog = self._obj(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, data in enumerate(self.objects, 1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + (data or b"null") + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(self.objects) + 1}\n0000000000 65535 f \n".encode()
        for o in offsets:
            out += f"{o:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(self.objects) + 1} /Root {catalog} 0 R /Info {info} 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode()
        return bytes(out)
