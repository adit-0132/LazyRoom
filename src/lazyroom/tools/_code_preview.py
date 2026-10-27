"""Code preview renderer (port of ImageToolbox ``feature/code-preview``).

Geometry follows ``CodePreviewBitmapRenderer`` exactly: everything is laid out at
``RenderScale = 3`` (560 px minimum card width x 3), monospace text with a 1.55 line
spacing multiplier (Android ``StaticLayout``: no extra spacing after the last line,
tab stops every 20 px), right-aligned line numbers at 32 % opacity shown only on the
first visual line of each logical line, a 42 px header with macOS window controls,
title (bold, ellipsized) and language label, a 1 px divider, a rounded card with a
Skia-style shadow layer, rotation about the centre, and a rounded linear-gradient
canvas.  Output is limited to 65,545,216 pixels like ImageToolbox.

Highlighting: ImageToolbox runs highlight.js (compose-highlight) with highlight.js CSS
themes.  Here the code is tokenised with Pygments (BSD, optional) and Pygments token
types are mapped to highlight.js scopes, which are coloured from the same 256 themes
(``_code_themes``).  Languages without a Pygments lexer render in the default colour.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ._code_themes import SCOPES, THEMES

RENDER_SCALE = 3.0
BASE_CARD_WIDTH = 560.0
MAX_BITMAP_PIXELS = 65_545_216.0
LINE_SPACING = 1.55
TAB_INCREMENT = 20.0  # android.text.Layout.TAB_INCREMENT (pixels)

DEFAULT_CODE = """fun main() {
    val toolbox = ImageToolbox(
        fast = true,
        private = true
    )

    println(toolbox.create())
}"""

# GradientBackgroundPreset (start, end)
BACKGROUND_PRESETS: dict[str, tuple[str, str]] = {
    "aurora": ("#6D5DFB", "#24C6DC"), "sunset": ("#FF6B6B", "#FFC371"), "candy": ("#B24592", "#F15F79"),
    "ocean": ("#005AA7", "#FFFDE4"), "lime": ("#11998E", "#38EF7D"), "ember": ("#8E2DE2", "#4A00E0"),
    "graphite": ("#232526", "#414345"), "paper": ("#F8FAFC", "#E2E8F0"), "electric": ("#7F00FF", "#E100FF"),
    "midnight": ("#0F2027", "#2C5364"), "synthwave": ("#FF00CC", "#333399"), "peach": ("#ED4264", "#FFB88C"),
    "mango": ("#F09819", "#EDDE5D"), "forest": ("#134E5E", "#71B280"), "arctic": ("#74EBD5", "#ACB6E5"),
    "rose": ("#F4C4F3", "#FC67FA"), "neon_sky": ("#00C6FF", "#0072FF"), "purple_bliss": ("#360033", "#0B8793"),
    "deep_space": ("#000428", "#004E92"), "firewatch": ("#CB2D3E", "#EF473A"), "mojito": ("#1D976C", "#93F9B9"),
    "cotton_candy": ("#D9A7C7", "#FFFCDC"), "royal_night": ("#141E30", "#243B55"),
    "black_rose": ("#2B0A3D", "#8B1E3F"), "desert_sun": ("#C79081", "#DFA579"), "mint": ("#00B09B", "#96C93D"),
    "ice": ("#83A4D4", "#B6FBFF"), "lavender": ("#DA22FF", "#9733EE"), "solar_flare": ("#F12711", "#F5AF19"),
    "northern_lights": ("#43C6AC", "#191654"),
}

# ---------------------------------------------------------------------------
# Languages (CodeLanguage.kt): order = Kotlin, popular, bundled highlight.js keys
# ---------------------------------------------------------------------------
_POPULAR = ("kotlin java javascript typescript python rust go swift dart c cpp csharp json xml css bash sql yaml "
            "markdown").split()
_BUNDLED = (
    "1c abnf accesslog actionscript ada angelscript apache applescript arcade arduino armasm xml asciidoc aspectj "
    "autohotkey autoit avrasm awk axapta bash basic bnf brainfuck c cal capnproto ceylon clean clojure clojure-repl "
    "cmake coffeescript coq cos cpp crmsh crystal csharp csp css d markdown dart delphi diff django dns dockerfile "
    "dos dsconfig dts dust ebnf elixir elm ruby erb erlang-repl erlang excel fix flix fortran fsharp gams gauss "
    "gcode gherkin glsl gml go golo gradle graphql groovy haml handlebars haskell haxe hsp http hy inform7 ini "
    "irpf90 isbl java javascript jboss-cli json julia julia-repl kotlin lasso latex ldif leaf less lisp "
    "livecodeserver livescript llvm lsl lua makefile mathematica matlab maxima mel mercury mipsasm mizar perl "
    "mojolicious monkey moonscript n1ql nestedtext nginx nim nix node-repl nsis objectivec ocaml openscad oxygene "
    "parser3 pf pgsql php php-template plaintext pony powershell processing profile prolog properties protobuf "
    "puppet purebasic python python-repl q qml r reasonml rib roboconf routeros rsl ruleslanguage rust sas scala "
    "scheme scilab scss shell smali smalltalk sml sqf sql stan stata step21 stylus subunit swift taggerscript yaml "
    "tap tcl thrift tp twig typescript vala vbnet vbscript vbscript-html verilog vhdl vim wasm wren x86asm xl "
    "xquery zephir").split()
LANGUAGE_KEYS: list[str] = ["kotlin"] + [k for k in dict.fromkeys(_POPULAR[1:] + _BUNDLED) if k != "kotlin"]

_DISPLAY = {
    "1c": "1C", "abnf": "ABNF", "accesslog": "Access Log", "actionscript": "ActionScript", "angelscript": "AngelScript",
    "applescript": "AppleScript", "armasm": "ARM Assembly", "xml": "HTML / XML", "asciidoc": "AsciiDoc",
    "aspectj": "AspectJ", "autohotkey": "AutoHotkey", "autoit": "AutoIt", "avrasm": "AVR Assembly", "awk": "AWK",
    "bnf": "BNF", "capnproto": "Cap\u2019n Proto", "clojure-repl": "Clojure REPL", "cmake": "CMake",
    "coffeescript": "CoffeeScript", "cpp": "C++", "csharp": "C#", "csp": "CSP", "css": "CSS", "d": "D", "dns": "DNS",
    "dos": "DOS", "ebnf": "EBNF", "erb": "ERB", "erlang-repl": "Erlang REPL", "fix": "FIX", "fsharp": "F#",
    "gams": "GAMS", "gcode": "G-code", "glsl": "GLSL", "gml": "GML", "graphql": "GraphQL", "haml": "HAML", "hsp": "HSP",
    "http": "HTTP", "ini": "INI", "irpf90": "IRPF90", "isbl": "ISBL", "javascript": "JavaScript",
    "jboss-cli": "JBoss CLI", "json": "JSON", "julia-repl": "Julia REPL", "latex": "LaTeX", "ldif": "LDIF",
    "less": "LESS", "livecodeserver": "LiveCode Server", "livescript": "LiveScript", "llvm": "LLVM", "lsl": "LSL",
    "matlab": "MATLAB", "mel": "MEL", "mipsasm": "MIPS Assembly", "n1ql": "N1QL", "node-repl": "Node.js REPL",
    "nsis": "NSIS", "objectivec": "Objective-C", "openscad": "OpenSCAD", "pgsql": "PostgreSQL", "php": "PHP",
    "php-template": "PHP Template", "plaintext": "Plain Text", "python-repl": "Python REPL", "qml": "QML",
    "reasonml": "ReasonML", "r": "R", "routeros": "RouterOS", "sas": "SAS", "scss": "SCSS", "sml": "SML", "sql": "SQL",
    "step21": "STEP Part 21", "taggerscript": "TaggerScript", "typescript": "TypeScript", "vbnet": "VB.NET",
    "vbscript": "VBScript", "vbscript-html": "VBScript in HTML", "vhdl": "VHDL", "wasm": "WebAssembly",
    "x86asm": "x86 Assembly", "yaml": "YAML",
}
_EXT = {
    "1c": "bsl", "accesslog": "log", "actionscript": "as", "angelscript": "as", "apache": "conf", "arduino": "ino",
    "armasm": "s", "xml": "html", "asciidoc": "adoc", "autohotkey": "ahk", "autoit": "au3", "avrasm": "asm",
    "axapta": "x++", "bash": "sh", "basic": "bas", "brainfuck": "bf", "capnproto": "capnp", "clean": "dcl",
    "clojure": "clj", "clojure-repl": "clj", "coffeescript": "coffee", "coq": "v", "crystal": "cr", "csharp": "cs",
    "markdown": "md", "delphi": "pas", "django": "html", "dns": "zone", "dockerfile": "Dockerfile", "dos": "bat",
    "dsconfig": "conf", "elixir": "ex", "ruby": "rb", "erlang": "erl", "erlang-repl": "erl", "fortran": "f90",
    "fsharp": "fs", "gams": "gms", "gherkin": "feature", "handlebars": "hbs", "haskell": "hs", "haxe": "hx",
    "inform7": "ni", "javascript": "js", "julia": "jl", "julia-repl": "jl", "kotlin": "kt", "latex": "tex",
    "livecodeserver": "lc", "livescript": "ls", "llvm": "ll", "makefile": "Makefile", "mathematica": "wl",
    "matlab": "m", "maxima": "mac", "mercury": "m", "mipsasm": "asm", "objectivec": "m", "ocaml": "ml",
    "openscad": "scad", "parser3": "p", "pf": "conf", "pgsql": "sql", "plaintext": "txt", "powershell": "ps1",
    "processing": "pde", "prolog": "pl", "protobuf": "proto", "puppet": "pp", "purebasic": "pb", "python": "py",
    "python-repl": "py", "reasonml": "re", "roboconf": "graph", "routeros": "rsc", "rust": "rs", "scheme": "scm",
    "scilab": "sci", "shell": "sh", "smalltalk": "st", "stata": "do", "step21": "stp", "stylus": "styl",
    "taggerscript": "tagscript", "typescript": "ts", "vbnet": "vb", "vbscript": "vbs", "vbscript-html": "html",
    "verilog": "v", "vhdl": "vhd", "wasm": "wat", "x86asm": "asm", "xquery": "xq", "zephir": "zep",
}
# highlight.js key -> Pygments alias where the names differ (approximations noted in the port report)
_PYGMENTS_ALIAS = {
    "xml": "html", "clojure-repl": "clojure", "dos": "batch", "erlang-repl": "erl", "gradle": "groovy",
    "mipsasm": "mips", "node-repl": "nodejsrepl", "pgsql": "postgresql", "php-template": "html+php",
    "plaintext": "text", "processing": "java", "python-repl": "pycon", "wasm": "wast", "x86asm": "nasm",
    "dns": "zone", "capnproto": "capnp", "n1ql": "sql", "irpf90": "fortran", "oxygene": "delphi",
    "vbscript-html": "html", "shell": "console",
}


def language_title(key: str) -> str:
    return _DISPLAY.get(key) or " ".join(p[:1].upper() + p[1:] for p in key.split("-"))


def language_extension(key: str) -> str:
    return _EXT.get(key, key)


def theme_title(asset: str) -> str:
    words = {"1c": "1C", "a11y": "A11y", "androidstudio": "Android Studio", "codepen": "CodePen", "github": "GitHub",
             "gml": "GML", "googlecode": "Google Code", "ia": "iA", "ir": "IR", "isbl": "ISBL", "nnfx": "NNFX",
             "oceanicnext": "Oceanic Next", "papercolor": "PaperColor", "phd": "PhD", "qtcreator": "Qt Creator",
             "ros": "Ros\u00e9", "rose": "Ros\u00e9", "stackoverflow": "Stack Overflow", "ui": "UI", "vs": "VS",
             "vs2015": "VS 2015", "xcode": "Xcode", "xt256": "XT 256"}
    name = asset.split("/", 1)[1] if "/" in asset else asset
    return " ".join(words.get(w) or (w[:1].upper() + w[1:]) for w in name.split("-"))


# ---------------------------------------------------------------------------
# Themes and highlighting
# ---------------------------------------------------------------------------
_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
RGBA = tuple[int, int, int, int]


def _hex(c: str) -> RGBA:
    v = [int(c[i:i + 2], 16) for i in range(0, len(c), 2)]
    return (v[0], v[1], v[2], v[3] if len(v) > 3 else 255)


@dataclass(frozen=True)
class Theme:
    name: str
    background: RGBA
    text: RGBA
    scopes: dict[str, RGBA]


@lru_cache(maxsize=64)
def get_theme(name: str) -> Theme:
    raw = THEMES[name]
    pal_s, idx = raw.split("|")
    pal = [_hex(p) for p in pal_s.split()]
    scopes = {s: pal[_DIGITS.index(ch)] for s, ch in zip(SCOPES, idx)}
    return Theme(name, pal[0], pal[1], scopes)


def _token_map() -> dict[Any, str]:
    from pygments.token import Comment, Generic, Keyword, Literal, Name, Number, Operator, Punctuation, String

    return {
        Keyword: "keyword", Keyword.Constant: "literal", Keyword.Type: "type",
        Name.Builtin: "built_in", Name.Builtin.Pseudo: "variable.language_", Name.Class: "title.class_",
        Name.Exception: "title.class_", Name.Function: "title.function_", Name.Decorator: "meta",
        Name.Tag: "tag name", Name.Attribute: "attr", Name.Variable: "variable", Name.Constant: "variable.constant_",
        Name.Label: "symbol", Name.Entity: "symbol", Name.Property: "property",
        Literal: "literal", Literal.Date: "number", String: "string", String.Escape: "char.escape_",
        String.Interpol: "subst", String.Regex: "regexp", String.Symbol: "symbol", Number: "number",
        Operator: "operator", Operator.Word: "keyword", Punctuation: "punctuation",
        Comment: "comment", Comment.Preproc: "meta", Comment.PreprocFile: "meta string", Comment.Special: "doctag",
        Generic.Heading: "section", Generic.Subheading: "section", Generic.Inserted: "addition",
        Generic.Deleted: "deletion", Generic.Emph: "emphasis", Generic.Strong: "strong", Generic.Prompt: "meta",
    }


def _lexer(language: str) -> Any | None:
    try:
        from pygments.lexers import get_lexer_by_name
        from pygments.util import ClassNotFound
    except ImportError:
        return None
    try:
        return get_lexer_by_name(_PYGMENTS_ALIAS.get(language, language), stripnl=False, ensurenl=False,
                                 stripall=False, tabsize=0)
    except ClassNotFound:
        return None


def highlight(code: str, language: str, theme: Theme) -> tuple[list[tuple[int, int, RGBA]], str]:
    """Colour spans (start, end, rgba) over ``code`` and the engine used ('pygments' | 'none')."""
    lexer = _lexer(language)
    if lexer is None or not code:
        return [], "none"
    tmap = _token_map()
    spans: list[tuple[int, int, RGBA]] = []
    pos = 0
    for ttype, value in lexer.get_tokens(code):
        t, scope = ttype, None
        while t is not None:
            if t in tmap:
                scope = tmap[t]
                break
            t = t.parent
        if scope and value:
            spans.append((pos, pos + len(value), theme.scopes.get(scope, theme.text)))
        pos += len(value)
    if pos != len(code):  # the lexer altered the text: fall back to plain text like ImageToolbox does
        return [], "none"
    return spans, "pygments"


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
_MONO_FALLBACKS = ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "C:/Windows/Fonts/consola.ttf")


@lru_cache(maxsize=4)
def _mono_path(bold: bool) -> str | None:
    import os

    from ..ops.draw import find_font

    p = find_font("monospace", bold=bold)
    if p:
        return p
    for cand in _MONO_FALLBACKS:
        if os.path.exists(cand):
            return cand
    return None


@lru_cache(maxsize=64)
def mono_font(size: float, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _mono_path(bold)
    if path:
        try:
            f = ImageFont.truetype(path, max(1, int(round(size))))
            if bold:
                try:
                    names = [n.decode() if isinstance(n, bytes) else n for n in f.get_variation_names()]
                    if "Bold" in names:
                        f.set_variation_by_name("Bold")
                except Exception:
                    pass
            return f
        except OSError:
            pass
    return ImageFont.load_default(size=max(1, int(round(size))))


def _metrics(font: Any) -> tuple[int, int]:
    asc, desc = font.getmetrics()
    return int(asc), int(desc)


# ---------------------------------------------------------------------------
# Text layout (android.text.StaticLayout subset)
# ---------------------------------------------------------------------------

def _next_tab(x: float) -> float:
    return int((x + TAB_INCREMENT) / TAB_INCREMENT) * TAB_INCREMENT


def _advance(font: Any, text: str, x0: float = 0.0) -> float:
    """Width of ``text`` starting at line offset x0, honouring Android's 20 px default tab stops."""
    x = x0
    parts = text.split("\t")
    for i, part in enumerate(parts):
        if part:
            x += font.getlength(part)
        if i < len(parts) - 1:
            x = _next_tab(x)
    return x - x0


_NEWLINE_RE = re.compile(r"\r\n|\n|\r")
_BREAK_RE = re.compile(r"\S*\s+|\S+")


def logical_lines(code: str) -> list[tuple[int, int]]:
    starts = [0] + [m.end() for m in _NEWLINE_RE.finditer(code)]
    ends = [m.start() for m in _NEWLINE_RE.finditer(code)] + [len(code)]
    return list(zip(starts, ends))


def layout_lines(code: str, font: Any, width: float | None) -> list[tuple[int, int]]:
    """Visual lines as (start, end) offsets: greedy breaks after whitespace (trailing spaces hang),
    characters split only when a word is wider than the line.  ``width`` None disables wrapping."""
    lines: list[tuple[int, int]] = []
    for s, e in logical_lines(code):
        text = code[s:e]
        if width is None or not text:
            lines.append((s, e))
            continue
        line_start, x = 0, 0.0
        for m in _BREAK_RE.finditer(text):
            a, b = m.span()
            body_end = a + len(m.group(0).rstrip())
            body_w = _advance(font, text[a:body_end], x)
            if a > line_start and x + body_w > width:
                lines.append((s + line_start, s + a))
                line_start, x = a, 0.0
                body_w = _advance(font, text[a:body_end], 0.0)
            if body_w > width:
                for j in range(a, body_end):
                    cw = _advance(font, text[j], x)
                    if j > line_start and x + cw > width:
                        lines.append((s + line_start, s + j))
                        line_start, x = j, 0.0
                        cw = _advance(font, text[j], 0.0)
                    x += cw
                x += _advance(font, text[body_end:b], x)
            else:
                x += _advance(font, text[a:b], x)
        lines.append((s + line_start, e))
    return lines


# ---------------------------------------------------------------------------
# Raster helpers (uint8 buffers, processed in strips to bound memory)
# ---------------------------------------------------------------------------
_STRIP = 512


def rrect_mask(h: int, w: int, x0: float, y0: float, x1: float, y1: float, r: float) -> np.ndarray:
    """Anti-aliased coverage (uint8) of a rounded rectangle, sampled at pixel centres."""
    out = np.zeros((h, w), np.uint8)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = max((x1 - x0) / 2, 0.0), max((y1 - y0) / 2, 0.0)
    r = max(0.0, min(r, hx, hy))
    c0, c1 = max(0, int(math.floor(x0)) - 1), min(w, int(math.ceil(x1)) + 1)
    r0, r1 = max(0, int(math.floor(y0)) - 1), min(h, int(math.ceil(y1)) + 1)
    if c0 >= c1 or r0 >= r1:
        return out
    qx = np.abs(np.arange(c0, c1, dtype=np.float32) + 0.5 - cx) - (hx - r)

    def rows(ya: int, yb: int) -> np.ndarray:
        qy = (np.abs(np.arange(ya, yb, dtype=np.float32) + 0.5 - cy) - (hy - r))[:, None]
        d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
        return (np.clip(0.5 - d, 0, 1) * 255 + 0.5).astype(np.uint8)

    # rows away from the corners all share one horizontal profile
    m0 = max(r0, int(math.ceil(y0 + r + 1)))
    m1 = min(r1, int(math.floor(y1 - r - 1)))
    if m1 - m0 > 2:
        out[m0:m1, c0:c1] = rows(int(cy), int(cy) + 1)[0]
        bands = [(r0, m0), (m1, r1)]
    else:
        bands = [(r0, r1)]
    for a, b in bands:
        for ya in range(a, b, _STRIP):
            out[ya:min(b, ya + _STRIP), c0:c1] = rows(ya, min(b, ya + _STRIP))
    return out


def circle_patch(cx: float, cy: float, r: float) -> tuple[int, int, np.ndarray]:
    x0, y0 = int(math.floor(cx - r - 1)), int(math.floor(cy - r - 1))
    n = int(math.ceil(2 * r + 3))
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.hypot(xx + x0 + 0.5 - cx, yy + y0 + 0.5 - cy) - r
    return x0, y0, np.clip(0.5 - d, 0, 1).astype(np.float32)


def fill_gradient(out: np.ndarray, colors: list[tuple[float, float, float, float]]) -> None:
    """``LinearGradient(0, 0, w, h, colors, null, CLAMP)`` (evenly spaced stops) into uint8 RGBA ``out``."""
    h, w = out.shape[:2]
    cols = np.asarray(colors, np.float32).reshape(-1, 4)
    if len(cols) == 1:
        out[:] = (cols[0] * 255 + 0.5).astype(np.uint8)
        return
    n = 4096  # colour lookup table over t in [0, 1]
    t_lut = np.linspace(0, 1, n, dtype=np.float32)
    pos = np.linspace(0, 1, len(cols), dtype=np.float32)
    lut = np.stack([(np.interp(t_lut, pos, cols[:, c]) * 255 + 0.5) for c in range(4)], axis=1).astype(np.uint8)
    denom = float(w * w + h * h) or 1.0
    xs = (np.arange(w, dtype=np.float32) + 0.5) * np.float32(w * (n - 1) / denom)
    for ya in range(0, h, _STRIP):
        yb = min(h, ya + _STRIP)
        ys = (np.arange(ya, yb, dtype=np.float32) + 0.5) * np.float32(h * (n - 1) / denom)
        idx = np.clip(xs[None, :] + ys[:, None] + 0.5, 0, n - 1).astype(np.int32)
        out[ya:yb] = lut[idx]


def skia_sigma(radius: float) -> float:
    """SkBlurMask::ConvertRadiusToSigma (shadow layer / blur mask radius -> Gaussian sigma)."""
    return 0.57735 * radius + 0.5 if radius > 0 else 0.0


def blur_u8(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    return cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)


def _over_block(d: np.ndarray, s: np.ndarray, premultiplied: bool) -> None:
    """Source-over of uint8 RGBA block ``s`` onto the uint8 RGBA view ``d`` (in place, straight alpha out)."""
    sa8 = s[..., 3]
    nz = sa8.any(axis=0)
    if not nz.any():
        return
    cols = np.flatnonzero(nz)
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    s, d, sa8 = s[:, c0:c1], d[:, c0:c1], sa8[:, c0:c1]
    sa = sa8.astype(np.float32)[..., None] * np.float32(1 / 255)
    s_rgb = s[..., :3].astype(np.float32)
    if not premultiplied:
        s_rgb *= sa
    if not d[..., 3].any():  # empty destination: result is the source
        if premultiplied:
            s_rgb /= np.maximum(sa, np.float32(1 / 255))
        d[..., :3] = np.clip(s_rgb + 0.5, 0, 255)
        d[..., 3] = sa8
        return
    df = d.astype(np.float32)
    da = df[..., 3:4] * np.float32(1 / 255)
    inv = 1 - sa
    oa = sa + da * inv
    rgb = (s_rgb + df[..., :3] * (da * inv)) / np.maximum(oa, np.float32(1e-6))
    d[..., :3] = np.clip(rgb + 0.5, 0, 255)
    d[..., 3] = np.clip(oa[..., 0] * 255 + 0.5, 0, 255)


def over_u8(dst: np.ndarray, src: np.ndarray, x: int = 0, y: int = 0, premultiplied: bool = False) -> None:
    """In-place source-over of uint8 RGBA ``src`` onto uint8 RGBA ``dst`` at (x, y)."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    xa, ya, xb, yb = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    for r in range(ya, yb, _STRIP):
        r2 = min(yb, r + _STRIP)
        _over_block(dst[r:r2, xa:xb], src[r - y:r2 - y, xa - x:xb - x], premultiplied)


def over_affine(dst: np.ndarray, src: np.ndarray, M: np.ndarray) -> None:
    """Warp premultiplied uint8 RGBA ``src`` by the 2x3 matrix ``M`` (bilinear) and composite it onto ``dst``,
    one strip at a time so no full-size intermediate is allocated."""
    H, W = dst.shape[:2]
    for r in range(0, H, _STRIP):
        r2 = min(H, r + _STRIP)
        Ms = M.copy()
        Ms[1, 2] -= r
        strip = cv2.warpAffine(src, Ms, (W, r2 - r), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))
        _over_block(dst[r:r2], strip, True)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

@dataclass
class CodePreviewParams:
    code: str = DEFAULT_CODE
    language: str = "kotlin"
    theme: str = "base16/dracula"
    background_colors: tuple = ((0x6D / 255, 0x5D / 255, 0xFB / 255, 1.0), (0x24 / 255, 0xC6 / 255, 0xDC / 255, 1.0))
    title: str = "Main.kt"
    font_size: int = 15
    outer_padding: int = 28
    inner_padding: int = 20
    corner_radius: int = 18
    canvas_corner_radius: int = 24
    rotation: float = 0.0
    show_card_shadow: bool = True
    card_shadow_color: tuple = (0.0, 0.0, 0.0, 0x66 / 255)
    card_shadow_blur_radius: int = 14
    card_shadow_offset_x: int = 0
    card_shadow_offset_y: int = 5
    show_canvas_background: bool = True
    show_window_controls: bool = True
    show_line_numbers: bool = True
    show_title: bool = True
    wrap_long_lines: bool = True


def _with_alpha(c: RGBA, a: float) -> RGBA:
    return (c[0], c[1], c[2], int(round(a * 255)))


def render(p: CodePreviewParams, max_pixels: float = MAX_BITMAP_PIXELS) -> tuple[np.ndarray, dict[str, Any]]:
    """Render to uint8 RGBA.  Returns (image, info)."""
    s = RENDER_SCALE
    theme = get_theme(p.theme)
    font = mono_font(p.font_size * s)
    inner = p.inner_padding * s
    code = p.code
    spans, engine = highlight(code, p.language, theme)
    logical = logical_lines(code)
    line_count = max(1, len(logical))
    gap = 14 * s if p.show_line_numbers else 0.0
    ln_w = font.getlength(str(line_count)) if p.show_line_numbers else 0.0
    desired = 0.0 if p.wrap_long_lines else max(_advance(font, code[a:b]) for a, b in logical) + s
    card_w = int(math.ceil(max(BASE_CARD_WIDTH * s, desired + ln_w + gap + inner * 2)))
    code_w = max(1, int(card_w - inner * 2 - ln_w - gap))
    lines = layout_lines(code, font, float(code_w))
    asc, desc = _metrics(font)
    line_h = asc + desc
    extra = int(line_h * (LINE_SPACING - 1) + 0.5)
    content_h = len(lines) * line_h + (len(lines) - 1) * extra
    header_h = 42 * s if (p.show_window_controls or p.show_title) else 0.0
    divider_h = max(s, 1.0) if header_h > 0 else 0.0
    card_h = int(math.ceil(header_h + divider_h + inner * 2 + content_h))
    shadow_pad = (int(math.ceil((p.card_shadow_blur_radius + max(abs(p.card_shadow_offset_x),
                                                                  abs(p.card_shadow_offset_y))) * s))
                  if p.show_card_shadow and p.card_shadow_blur_radius > 0 else 0)
    layer_w, layer_h = card_w + shadow_pad * 2, card_h + shadow_pad * 2
    rad = math.radians(abs(p.rotation))
    rot_w = int(math.ceil(layer_w * math.cos(rad) + layer_h * math.sin(rad)))
    rot_h = int(math.ceil(layer_w * math.sin(rad) + layer_h * math.cos(rad)))
    outer = int(math.ceil(p.outer_padding * s))
    canvas_w, canvas_h = rot_w + outer * 2, rot_h + outer * 2
    k = min(1.0, math.sqrt(max(1.0, max_pixels) / (canvas_w * canvas_h)))
    if k < 1:
        k *= 0.99
    W, H = int(math.ceil(canvas_w * k)), int(math.ceil(canvas_h * k))

    out = np.zeros((H, W, 4), np.uint8)
    if p.show_canvas_background:
        fill_gradient(out, list(p.background_colors))
        out[..., 3] = (out[..., 3].astype(np.uint16) * rrect_mask(H, W, 0, 0, W, H, p.canvas_corner_radius * s * k)
                       // 255).astype(np.uint8)

    # --- card layer, drawn directly at output scale k
    lw, lh = max(1, int(math.ceil(layer_w * k))), max(1, int(math.ceil(layer_h * k)))
    x0 = y0 = shadow_pad * k
    x1, y1 = (shadow_pad + card_w) * k, (shadow_pad + card_h) * k
    card_r = p.corner_radius * s * k
    bg = theme.background
    content = Image.new("RGB", (lw, lh), bg[:3])
    draw = ImageDraw.Draw(content, "RGBA")
    fnt = mono_font(p.font_size * s * k)

    def X(v: float) -> float:
        return x0 + v * k

    def Y(v: float) -> float:
        return y0 + v * k

    if header_h > 0:
        hpad = 14 * s
        title_start = hpad
        cy = header_h / 2
        if p.show_window_controls:
            r = 5 * s
            for i, col in enumerate(((0xFF, 0x5F, 0x57), (0xFE, 0xBC, 0x2E), (0x28, 0xC8, 0x40))):
                px, py, a = circle_patch(X(hpad + r + i * 17 * s), Y(cy), r * k)
                _blend_patch(content, px, py, a, col)
            title_start += 63 * s
        if p.show_title:
            lang_label = language_title(p.language)
            lang_layout = mono_font(10 * s)
            lang_x = card_w - hpad - lang_layout.getlength(lang_label)
            title_layout = mono_font(12 * s, bold=True)
            title = _ellipsize(p.title, title_layout, max(0.0, lang_x - title_start - 8 * s))
            ta, td = _metrics(title_layout)
            la, ld = _metrics(lang_layout)
            if title:
                _text(content, draw, (X(title_start), Y(cy + (ta - td) / 2)), title,
                      font=mono_font(12 * s * k, bold=True), fill=theme.text)
            _text(content, draw, (X(lang_x), Y(cy + (la - ld) / 2)), lang_label, font=mono_font(10 * s * k),
                  fill=_with_alpha(theme.text, 0.52))
        # non-anti-aliased Paint: rows whose centres fall inside the divider
        top_row, end_row = int(round(Y(header_h))), int(round(Y(header_h + divider_h)))
        if end_row > top_row:
            draw.rectangle([int(round(X(0))), top_row, int(round(X(card_w))) - 1, end_row - 1],
                           fill=_with_alpha(theme.text, 0.1))

    content_top = header_h + divider_h + inner
    first_lines = {a: i + 1 for i, (a, _) in enumerate(logical)}
    ln_color = _with_alpha(theme.text, 0.32)
    code_x = inner + ln_w + gap
    ln_box = math.ceil(ln_w)
    si = 0
    for li, (a, b) in enumerate(lines):
        base = content_top + li * (line_h + extra) + asc
        if p.show_line_numbers and a in first_lines:
            label = str(first_lines[a])
            _text(content, draw, (X(inner + ln_box - font.getlength(label)), Y(base)), label, font=fnt, fill=ln_color)
        while si < len(spans) and spans[si][1] <= a:
            si += 1
        x, pos, j = 0.0, a, si
        while pos < b:
            col, nxt = theme.text, b
            if j < len(spans) and spans[j][0] <= pos:
                col, nxt = spans[j][2], min(b, spans[j][1])
            elif j < len(spans):
                nxt = min(b, spans[j][0])
            parts = code[pos:nxt].split("\t")
            for pi, part in enumerate(parts):
                if part:
                    if part.strip():
                        _text(content, draw, (X(code_x + x), Y(base)), part, font=fnt, fill=col)
                    x += font.getlength(part)
                if pi < len(parts) - 1:
                    x = _next_tab(x)
            pos = nxt
            if j < len(spans) and pos >= spans[j][1]:
                j += 1
    del draw

    layer = np.zeros((lh, lw, 4), np.uint8)
    if shadow_pad:
        sx, sy = p.card_shadow_offset_x * s * k, p.card_shadow_offset_y * s * k
        sh = blur_u8(rrect_mask(lh, lw, x0 + sx, y0 + sy, x1 + sx, y1 + sy, card_r),
                     skia_sigma(p.card_shadow_blur_radius * s * k))
        sc = p.card_shadow_color
        layer[..., :3] = (np.asarray(sc[:3]) * 255 + 0.5).astype(np.uint8)
        layer[..., 3] = (sh.astype(np.float32) * sc[3] + 0.5).astype(np.uint8)
    card = np.dstack([np.asarray(content), rrect_mask(lh, lw, x0, y0, x1, y1, card_r)])
    del content
    over_u8(layer, card)
    del card

    if p.rotation == 0:
        over_u8(out, layer, int(round(W / 2 - lw / 2)), int(round(H / 2 - lh / 2)))
    else:  # canvas.rotate about the centre (positive = clockwise on screen)
        M = cv2.getRotationMatrix2D((lw / 2.0, lh / 2.0), -p.rotation, 1.0)
        M[0, 2] += W / 2.0 - lw / 2.0
        M[1, 2] += H / 2.0 - lh / 2.0
        a = layer[..., 3:4].astype(np.uint16)
        layer[..., :3] = ((layer[..., :3].astype(np.uint16) * a + 127) // 255).astype(np.uint8)
        over_affine(out, layer, M)
    info = {"width": W, "height": H, "lines": line_count, "visual_lines": len(lines), "highlighter": engine,
            "theme": p.theme, "language": p.language, "output_scale": round(k, 6),
            "font": getattr(font, "path", None)}
    return out, info


def _text(img: Image.Image, draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: Any,
          fill: RGBA) -> None:
    """Draw text at a left-baseline anchor; Pillow ignores the ink alpha for text, so blend it via a mask."""
    if fill[3] >= 255:
        draw.text(xy, text, font=font, fill=fill[:3], anchor="ls")
        return
    if fill[3] <= 0:
        return
    left, t, r, b = draw.textbbox(xy, text, font=font, anchor="ls")
    x0, y0 = int(math.floor(left)), int(math.floor(t))
    w, h = int(math.ceil(r)) - x0 + 1, int(math.ceil(b)) - y0 + 1
    if w <= 0 or h <= 0:
        return
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text((xy[0] - x0, xy[1] - y0), text, font=font, fill=255, anchor="ls")
    mask = mask.point(lambda v: (v * fill[3] + 127) // 255)
    img.paste(fill[:3], (x0, y0, x0 + w, y0 + h), mask)


def _blend_patch(img: Image.Image, x: int, y: int, alpha: np.ndarray, rgb: tuple[int, int, int]) -> None:
    h, w = alpha.shape
    box = (x, y, x + w, y + h)
    patch = np.asarray(img.crop(box), np.float32)
    a = alpha[..., None]
    patch = patch * (1 - a) + np.asarray(rgb, np.float32) * a
    img.paste(Image.fromarray((np.clip(patch, 0, 255) + 0.5).astype(np.uint8), "RGB"), box)


def _ellipsize(text: str, font: Any, avail: float) -> str:
    """``TextUtils.ellipsize(..., END)``."""
    if font.getlength(text) <= avail:
        return text
    ell = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.getlength(text[:mid] + ell) <= avail:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ell if lo > 0 or font.getlength(ell) <= avail else ""
