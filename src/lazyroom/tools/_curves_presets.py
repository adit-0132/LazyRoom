"""ImageToolbox ``CurvesPresetCodec`` (lib/curves) port + curves preset tools.

Formats (same detection order as the Kotlin ``decode(ByteArray)``):

* ZIP archive (``PK`` signature): the entry with the best extension priority
  ``itcurve > xmp > lrtemplate > acv`` (max 128 entries, 2 MiB per preset, macOS junk skipped).
* Text: ImageToolbox ``.itcurve`` JSON (``"format": "ImageToolbox Curves"``, ``version`` 1, 17
  ``curves`` + 7 ``colorWheels`` numbers), otherwise Lightroom/Camera Raw ``ToneCurvePV2012``
  (+Red/Green/Blue; legacy ``ToneCurve*``) as XMP ``rdf:li`` lists or ``.lrtemplate`` tables, 0..255.
* Binary: Photoshop ``.acv`` (version 1 or 4, big-endian output/input pairs, 0..255 or 0..65535).

``encode`` writes the native ``.itcurve`` JSON byte-for-byte like Kotlin (Java ``Float.toString``
number formatting); ``encode_xmp`` writes the Lightroom-compatible XMP with the RGB curves.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Any

import numpy as np

from ..core.errors import ErrorCode, LazyRoomError, invalid
from ..core.io import atomic_write
from ..core.params import Bool, Choice, File, Float, Point, Text
from ..ops import curves_ext as ce
from .registry import out_path, tool

ITB = "ImageToolbox"
CURVE_COUNT = 17
RGB_CURVE_COUNT = 4
COLOR_WHEEL_VALUE_COUNT = 7
MAX_ACV_CURVE_COUNT = 19
MAX_POINT_COUNT = 256
MAX_ARCHIVE_ENTRY_COUNT = 128
MAX_ARCHIVE_PRESET_SIZE = 2 * 1024 * 1024
MAX_PRESET_SIZE_BYTES = 20 * 1024 * 1024       # CurvesComponent.MaxPresetSizeBytes
LIGHTROOM_POINT_SCALE = 255.0
ARCHIVE_EXTENSIONS = ["itcurve", "xmp", "lrtemplate", "acv"]
LIGHTROOM_CURVE_NAMES = ["ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue"]
_LEGACY = {"ToneCurvePV2012": "ToneCurve", "ToneCurvePV2012Red": "ToneCurveRed",
           "ToneCurvePV2012Green": "ToneCurveGreen", "ToneCurvePV2012Blue": "ToneCurveBlue"}
FILE_EXTENSION = ".itcurve"

_RX = re.ASCII | re.IGNORECASE
_FORMAT_RE = re.compile(r'"format"\s*:\s*"ImageToolbox Curves"', _RX)
_XML_LI_RE = re.compile(r"<(?:[\w.-]+:)?li\b[^>]*>([\s\S]*?)</(?:[\w.-]+:)?li\s*>", _RX)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", re.ASCII)


class CurvesPresetError(ValueError):
    """``require``/``error`` failures of the Kotlin codec."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise CurvesPresetError(msg)


def _f32(v: float) -> float:
    return float(np.float32(v))


# -----------------------------------------------------------------------------
# Number formatting / parsing
# -----------------------------------------------------------------------------

def java_float_str(v: float) -> str:
    """``Float.toString``: shortest round-trip digits; plain for 1e-3 <= |v| < 1e7, else ``d.dE-n``."""
    f = np.float32(v)
    if f == 0:
        return "-0.0" if np.signbit(f) else "0.0"
    a = abs(float(f))
    if 1e-3 <= a < 1e7:
        s = np.format_float_positional(f, unique=True, trim="0")
        return s if "." in s else s + ".0"
    s = np.format_float_scientific(f, unique=True, trim="0")
    mant, exp = s.split("e")
    if "." not in mant:
        mant += ".0"
    return f"{mant}E{int(exp)}"


def _parse_numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER_RE.finditer(text):
        try:
            out.append(_f32(float(m.group(0))))
        except ValueError:  # toFloatOrNull
            continue
    return out


# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

def _point_array(values: list[float]) -> str:
    parts = []
    for i in range(0, len(values), 2):
        chunk = values[i:i + 2]
        if len(chunk) == 2:
            parts.append(f"[{java_float_str(chunk[0])}, {java_float_str(chunk[1])}]")
        elif len(chunk) == 1:
            parts.append(f"[{java_float_str(chunk[0])}]")
    return "[" + ", ".join(parts) + "]"


def encode(state: ce.CurvesState) -> str:
    """The native, human-readable ImageToolbox ``.itcurve`` format."""
    cps = state.control_points()
    lines = ["{\n", '  "format": "ImageToolbox Curves",\n', '  "version": 1,\n', '  "curves": [\n']
    for i, pts in enumerate(cps[:CURVE_COUNT]):
        lines.append("    " + _point_array(pts) + ("," if i < CURVE_COUNT - 1 else "") + "\n")
    lines.append("  ],\n")
    lines.append('  "colorWheels": ' + _point_array([v for p in cps[CURVE_COUNT:] for v in p]) + "\n}\n")
    return "".join(lines)


def encode_xmp(state: ce.CurvesState) -> str:
    """Lightroom-compatible XMP preset with the master and R/G/B point curves."""
    cps = state.control_points()
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n'
           '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
           '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
           '    <rdf:Description rdf:about=""\n'
           '      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"\n'
           '      crs:PresetType="Normal"\n'
           '      crs:Cluster="ImageToolbox"\n'
           '      crs:SupportsAmount="False"\n'
           '      crs:ProcessVersion="11.0"\n'
           '      crs:HasSettings="True"\n'
           '      crs:ToneCurveName2012="Custom">\n']
    scale = np.float32(LIGHTROOM_POINT_SCALE)
    for idx, name in enumerate(LIGHTROOM_CURVE_NAMES):
        out.append(f"    <crs:{name}>\n      <rdf:Seq>\n")
        pts = cps[idx]
        for i in range(0, len(pts) - 1, 2):
            x = int(np.floor(np.float32(pts[i]) * scale + np.float32(0.5)))   # roundToInt
            y = int(np.floor(np.float32(pts[i + 1]) * scale + np.float32(0.5)))
            out.append(f"        <rdf:li>{x}, {y}</rdf:li>\n")
        out.append(f"      </rdf:Seq>\n    </crs:{name}>\n")
    out.append("    </rdf:Description>\n  </rdf:RDF>\n</x:xmpmeta>\n")
    return "".join(out)


# -----------------------------------------------------------------------------
# Decoding
# -----------------------------------------------------------------------------

def _looks_like_zip(b: bytes) -> bool:
    return len(b) >= 4 and b[0] == ord("P") and b[1] == ord("K") and b[2] in (3, 5, 7) and b[3] in (4, 6, 8)


def _looks_like_text(b: bytes) -> bool:
    return all(v in (0x09, 0x0A, 0x0D) or 0x20 <= v <= 0x7E or v >= 0x80 for v in b[:64])


def _state_with_rgb_curves(curves: list[list[float] | None]) -> ce.CurvesState:
    points = ce.CurvesState().control_points()
    for i, c in enumerate(curves):
        if c is not None and i < RGB_CURVE_COUNT:
            points[i] = c
    return ce.CurvesState.from_control_points(points)


def _extract_json_array(text: str, key: str) -> str | None:
    m = re.search('"' + re.escape(key) + r'"\s*:', text)
    if not m:
        return None
    start = text.find("[", m.end())
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None


def _top_level_arrays(s: str) -> list[str]:
    arrays, depth, start = [], 0, -1
    for i, ch in enumerate(s):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                arrays.append(s[start + 1:i])
    return arrays


def _decode_itb_json(text: str) -> ce.CurvesState:
    _require(bool(_FORMAT_RE.search(text)), "Unsupported curve preset format")
    body = _extract_json_array(text, "curves")
    curves = [_parse_numbers(a) for a in _top_level_arrays(body)] if body is not None else []
    _require(len(curves) == CURVE_COUNT and all(len(c) >= 4 and len(c) % 2 == 0 for c in curves),
             "The ImageToolbox curve preset is incomplete")
    wbody = _extract_json_array(text, "colorWheels")
    wheels = _parse_numbers(wbody) if wbody is not None else []
    _require(len(wheels) == COLOR_WHEEL_VALUE_COUNT, "The ImageToolbox color wheels are invalid")
    return ce.CurvesState.from_control_points(curves + [wheels[0:2], wheels[2:4], wheels[4:6], wheels[6:7]])


def _extract_xml_sequence(text: str, name: str, scale: float) -> list[float] | None:
    m = re.search(r"<(?:[\w.-]+:)?" + name + r"\b[^>]*>([\s\S]*?)</(?:[\w.-]+:)?" + name + r"\s*>", text, _RX)
    if not m:
        return None
    pts: list[float] = []
    for li in _XML_LI_RE.finditer(m.group(1)):
        nums = _parse_numbers(li.group(1))
        if len(nums) >= 2:
            pts += [_f32(np.float32(v) / np.float32(scale)) for v in nums[:2]]
    return pts if len(pts) >= 4 and len(pts) % 2 == 0 else None


def _extract_lrtemplate_sequence(text: str, name: str, scale: float) -> list[float] | None:
    m = re.search(r"\b" + name + r"\s*=\s*\{([\s\S]*?)\}", text, _RX)
    if not m:
        return None
    pts = [_f32(np.float32(v) / np.float32(scale)) for v in _parse_numbers(m.group(1))]
    return pts if len(pts) >= 4 and len(pts) % 2 == 0 else None


def _decode_text(text: str) -> tuple[ce.CurvesState, str]:
    _require(bool(text.strip()), "The curve preset is empty")
    if text.lstrip("﻿ \t\r\n").startswith("{"):
        return _decode_itb_json(text), "itcurve"
    curves: list[list[float] | None] = []
    kinds = set()
    for name in LIGHTROOM_CURVE_NAMES:
        found = None
        for n in (name, _LEGACY[name]):
            found = _extract_xml_sequence(text, n, LIGHTROOM_POINT_SCALE)
            if found is not None:
                kinds.add("xmp")
                break
            found = _extract_lrtemplate_sequence(text, n, LIGHTROOM_POINT_SCALE)
            if found is not None:
                kinds.add("lrtemplate")
                break
        curves.append(found)
    _require(any(c is not None for c in curves), "No supported tone curves were found in the preset")
    return _state_with_rgb_curves(curves), ("xmp" if "xmp" in kinds else "lrtemplate")


def _decode_acv(b: bytes) -> ce.CurvesState:
    off = 0

    def u16() -> int:
        nonlocal off
        _require(off + 2 <= len(b), "The ACV file is truncated")
        v = (b[off] << 8) | b[off + 1]
        off += 2
        return v

    version = u16()
    _require(version in (1, 4), f"Unsupported ACV version: {version}")
    count = u16()
    _require(1 <= count <= MAX_ACV_CURVE_COUNT, "Invalid ACV curve count")
    raw = []
    for _ in range(count):
        n = u16()
        _require(2 <= n <= MAX_POINT_COUNT, "Invalid ACV point count")
        pts = []
        for _ in range(n):
            output = u16()
            inp = u16()
            pts.append((inp, output))
        raw.append(pts)
    scale = 65535.0 if any(x > 255 or y > 255 for c in raw for x, y in c) else LIGHTROOM_POINT_SCALE
    curves = [[_f32(np.float32(v) / np.float32(scale)) for x, y in c for v in (x, y)] for c in raw[:RGB_CURVE_COUNT]]
    return _state_with_rgb_curves(curves)


def _decode_zip(b: bytes) -> tuple[ce.CurvesState, str]:
    candidates: list[tuple[int, bytes, str]] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(b))
    except zipfile.BadZipFile as e:
        raise CurvesPresetError(f"Invalid preset archive: {e}") from None
    with zf:
        for count, info in enumerate(zf.infolist(), 1):
            _require(count <= MAX_ARCHIVE_ENTRY_COUNT, "The preset archive has too many files")
            name = info.filename.replace("\\", "/")
            fname = name.rsplit("/", 1)[-1]
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            prio = ARCHIVE_EXTENSIONS.index(ext) if ext in ARCHIVE_EXTENSIONS else -1
            if info.is_dir() or name.startswith("__MACOSX/") or fname.startswith("._") or prio < 0:
                continue
            buf = bytearray()
            with zf.open(info) as fh:
                while True:
                    chunk = fh.read(8192)
                    if not chunk:
                        break
                    buf += chunk
                    _require(len(buf) <= MAX_ARCHIVE_PRESET_SIZE, "A preset in the archive is too large")
            candidates.append((prio, bytes(buf), fname))
    if not candidates:
        raise CurvesPresetError("No supported curve preset was found in the archive")
    best = min(candidates, key=lambda c: c[0])  # minByOrNull: first of the equal minima
    st, fmt = decode_with_format(best[1])
    return st, f"zip:{fmt}"


def decode_with_format(data: bytes | str) -> tuple[ce.CurvesState, str]:
    """Decode any supported preset; returns (state, detected format)."""
    if isinstance(data, str):
        return _decode_text(data)
    _require(len(data) > 0, "The curve preset is empty")
    if _looks_like_zip(data):
        return _decode_zip(data)
    if _looks_like_text(data):
        return _decode_text(data.decode("utf-8", errors="replace"))
    return _decode_acv(data), "acv"


def decode(data: bytes | str) -> ce.CurvesState:
    return decode_with_format(data)[0]


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def summarize(st: ce.CurvesState) -> dict[str, Any]:
    active_types = [t for t, names in ce.EDITOR_TYPES.items()
                    if names and any(not st.curves[n].is_default for n in names)]
    if not all(x * x + y * y < 0.000001 for x, y in st.wheels.values()):
        active_types.append("ColorWheels")
    return {"default": st.is_default(), "active_curves": [n for n in ce.CURVE_NAMES if not st.curves[n].is_default],
            "active_editor_types": active_types,
            "point_counts": {n: len(st.curves[n].points) for n in ce.CURVE_NAMES}}


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------

def _state_params() -> list:
    return [*ce._curve_params(),
            Point("wheel_shadows", (0.0, 0.0), min=-1, max=1, title="Shadows wheel"),
            Point("wheel_midtones", (0.0, 0.0), min=-1, max=1, title="Midtones wheel"),
            Point("wheel_highlights", (0.0, 0.0), min=-1, max=1, title="Highlights wheel"),
            Float("wheel_edges", ce.DEFAULT_EDGES, min=0, max=1, step=1 / 15, title="Wheel edges")]


def _read_preset(api, path: str) -> tuple[ce.CurvesState, str]:
    p = api.ws.policy.check_read(path)
    data = p.read_bytes()
    if len(data) > MAX_PRESET_SIZE_BYTES:
        raise invalid("The curve preset is too large", param="input")
    try:
        return decode_with_format(data)
    except CurvesPresetError as e:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Could not read curves preset {p.name}: {e}",
                              details={"param": "input"}) from None


@tool("curves_preset_export", "Export curves preset (.itcurve / .xmp)", category="color", params=[
    *_state_params(),
    File("source", "", file_kind="any",
         description="Optional preset to start from (.itcurve/.xmp/.lrtemplate/.acv/.zip); non-default curve "
                     "params override it (use this to convert e.g. Photoshop .acv to .itcurve)"),
    Choice("format", "itcurve", choices=["itcurve", "xmp"],
           description="itcurve: ImageToolbox JSON (all 17 curves + colour wheels); xmp: Lightroom XMP "
                       "(RGB master + R/G/B point curves only)"),
    Text("output", "", description="Output file (.itcurve or .xmp); empty = only return the text"),
    Bool("overwrite", False),
], source={"project": ITB, "feature": "Curves: CurvesPresetCodec.encode / encodeXmp"})
def curves_preset_export(api, source, format, output, overwrite, **values):
    """Encode curves (the same parameters as adjust.color_curves) as an ImageToolbox .itcurve preset or a
    Lightroom-compatible XMP; optionally write it to a file."""
    base = _read_preset(api, source)[0] if source else None
    st = ce.state_from_params(values, base)
    text = encode(st) if format == "itcurve" else encode_xmp(st)
    res: dict[str, Any] = {"format": format, "text": text, "params": st.to_params(), **summarize(st)}
    if format == "xmp":
        dropped = [n for n in ce.CURVE_NAMES[4:] if not st.curves[n].is_default]
        if dropped or not all(x * x + y * y < 0.000001 for x, y in st.wheels.values()):
            res["warning"] = "XMP stores only the RGB master and R/G/B curves; other curves/wheels were not exported"
    if output:
        p = out_path(api, output, overwrite=overwrite)
        data = text.encode("utf-8")
        atomic_write(p, data, overwrite=overwrite)
        res.update(output=str(p), bytes=len(data))
    return res


@tool("curves_preset_import", "Import curves preset", category="color", params=[
    File("input", "", file_kind="any", required=True,
         description="ImageToolbox .itcurve, Lightroom .xmp/.lrtemplate, Photoshop .acv, or a .zip containing one"),
], source={"project": ITB, "feature": "Curves: CurvesPresetCodec.decode"})
def curves_preset_import(api, input):
    """Read a curves preset and return parameters that can be passed directly to adjust.color_curves."""
    st, fmt = _read_preset(api, input)
    return {"input": input, "detected_format": fmt, "params": st.to_params(),
            "control_points": st.control_points(), **summarize(st)}
