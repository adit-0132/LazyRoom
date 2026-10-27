"""ImageToolbox filter-template codec (the string shared as text, file or QR code).

Port of ImageToolbox ``FilterSerializationUtils.kt`` / ``Mappings.kt`` /
``AndroidFilterParamsInteractor`` (Apache-2.0, T8RIN).  A template is::

    https://github.com/T8RIN/ImageToolbox?<name>+<filter>,<filter>,...

and every ``<filter>`` is ``<class>:<value type>:<value>`` where ``<class>`` is
the fully qualified ``Ui*Filter`` class name with ``com.t8rin.imagetoolbox``
replaced by ``^^``, ``<value type>`` the runtime simple class name of the filter
value (``Float``, ``Pair{Float$Boolean}``, ``ClaheParams``, ...) and
``<value>`` its ``$``-separated serialisation (floats printed like Java's
``Float.toString``, strings Base64-encoded).  The per-filter mapping between
ImageToolbox values and LazyRoom ``filter.*`` params lives in
``_itb_template_table`` (generated from the Kotlin sources and verified against
every filter's defaults).
"""
from __future__ import annotations

import base64
from typing import Any

import numpy as np

from ..core.errors import invalid
from ..core.params import color_to_hex, parse_color
from ._itb_template_table import FILTERS, FONTS, PALETTES

LINK_HEADER = "https://github.com/T8RIN/ImageToolbox?"
REAL_PACKAGE = "com.t8rin.imagetoolbox"
PACKAGE_ALIAS = "^^"
UI_PACKAGE = PACKAGE_ALIAS + ".core.filters.presentation.model."
FILTERS_SEP, TEMPLATES_SEP, CONTENT_SEP, VALUE_SEP = ",", "\\", "+", ":"
PROP_SEP, ADD_PROP_SEP = "$", "*"

# op id -> ImageToolbox filter names (several colour-map filters share filter.color_map)
BY_OP: dict[str, list[str]] = {}
for _n, _e in FILTERS.items():
    BY_OP.setdefault(_e["op"], []).append(_n)

# ToneCurves: ImageToolbox stores 17 curves (luma, R, G, B, C, M, Y, K, L*, a*, b*, 6 hue/luma/sat curves) + 4
# colour-wheel entries; the LazyRoom port implements the first four (GPUImage tone curve).
_CURVE_PARAMS = ["rgb", "red", "green", "blue"]
_DEFAULT_CURVES = [[0.0, 0.0, 1.0, 1.0]] * 11 + [[0.0, 0.5, 1.0, 0.5]] * 6
_DEFAULT_WHEELS = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5]]


# ---------------------------------------------------------------------------
# Kotlin/Java primitive formatting & parsing
# ---------------------------------------------------------------------------

def jfloat(x: float) -> str:
    """Java ``Float.toString`` (shortest float32 digits; plain notation for 1e-3 <= |x| < 1e7)."""
    f = np.float32(x)
    if np.isnan(f):
        return "NaN"
    if np.isinf(f):
        return "Infinity" if f > 0 else "-Infinity"
    if f == 0:
        return "-0.0" if np.signbit(f) else "0.0"
    mant, exp_s = np.format_float_scientific(f, unique=True, trim="-").split("e")
    exp = int(exp_s)
    neg = mant.startswith("-")
    digits = mant.lstrip("-").replace(".", "")
    a = abs(float(f))
    if 1e-3 <= a < 1e7:
        point = exp + 1
        if point <= 0:
            s = "0." + "0" * (-point) + digits
        elif point >= len(digits):
            s = digits + "0" * (point - len(digits)) + ".0"
        else:
            s = digits[:point] + "." + digits[point:]
    else:
        s = digits[0] + "." + (digits[1:] or "0") + "E" + str(exp)
    return ("-" if neg else "") + s


def kfloat(s: str) -> float:
    """Kotlin ``String.toFloat()`` (Java ``Float.parseFloat``): the float32 value, as its shortest decimal."""
    t = s.strip()
    if t[-1:] in ("f", "F", "d", "D"):
        t = t[:-1]
    if not t or t.lower().startswith(("0x", "-0x", "+0x")):
        raise ValueError(f"not a float: {s!r}")
    f = np.float32(float(t))
    return float(np.format_float_positional(f, unique=True, trim="-")) if np.isfinite(f) else float(f)


def kint(s: str) -> int:
    t = s.strip()
    if not t or not (t.lstrip("+-").isdigit()):
        raise ValueError(f"not an int: {s!r}")
    v = int(t)
    if not -(1 << 31) <= v < (1 << 31):
        raise ValueError(f"int overflow: {s!r}")
    return v


def kbool(s: str) -> bool:  # Kotlin String.toBoolean(): equals("true", ignoreCase = true)
    return s.strip().lower() == "true"


def b64e(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def b64d(text: str) -> str:
    t = text.strip()
    return base64.b64decode(t + "=" * (-len(t) % 4), validate=True).decode("utf-8")


def sint32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def argb_to_hex(c: int) -> str:
    c &= 0xFFFFFFFF
    a, r, g, b = c >> 24, (c >> 16) & 255, (c >> 8) & 255, c & 255
    return f"#{r:02X}{g:02X}{b:02X}" + ("" if a == 255 else f"{a:02X}")


def hex_to_argb(value: Any) -> int:
    r, g, b, a = (int(round(c * 255)) for c in parse_color(value))
    return sint32((a << 24) | (r << 16) | (g << 8) | b)


# ---------------------------------------------------------------------------
# Template envelope
# ---------------------------------------------------------------------------

def is_valid_template(s: str) -> bool:
    """``AndroidFilterParamsInteractor.isValidTemplateFilter``."""
    return (REAL_PACKAGE in s or PACKAGE_ALIAS in s) and "Filter" in s and LINK_HEADER in s


def split_template(content: str) -> tuple[str, list[tuple[str, str | None, str | None]]]:
    """-> (template name, [(class name, value type, value string)]) like ``toTemplateFiltersList``."""
    if not is_valid_template(content):
        raise invalid("Not an ImageToolbox filter template",
                      hint=f"Templates start with {LINK_HEADER} and name '^^...Filter' classes.")
    body = content.strip()
    if body.startswith(LINK_HEADER):
        body = body[len(LINK_HEADER):]
    first = body.split(TEMPLATES_SEP)[0]
    parts = first.split(CONTENT_SEP)
    if len(parts) < 2:
        raise invalid("Malformed ImageToolbox template: missing '+' between the name and the filters")
    name = parts[0]
    # ImageToolbox keeps only splitData[1]; a Base64 '+' inside a value would truncate it there.  Rejoining is a
    # lenient superset that still reads every template ImageToolbox reads.
    filters = CONTENT_SEP.join(parts[1:])
    out = []
    for line in filters.split(FILTERS_SEP):
        if not line.strip():
            continue
        seg = line.split(VALUE_SEP)
        if len(seg) >= 3:
            out.append((seg[0].strip(), seg[1], seg[2]))
        else:
            out.append((line.strip(), None, None))
    return name, out


def filter_name(cls: str) -> str | None:
    """'^^.core.filters.presentation.model.UiBrightnessFilter' -> 'Brightness' (also data-model names)."""
    simple = cls.replace(PACKAGE_ALIAS, REAL_PACKAGE).rsplit(".", 1)[-1]
    if simple.startswith("Ui") and simple[2:3].isupper():
        simple = simple[2:]
    if not simple.endswith("Filter"):
        return None
    simple = simple[: -len("Filter")]
    return simple if simple in FILTERS else ({"SimpleWeavePixelation": "SimpleWeavePixelization"}.get(simple))


# ---------------------------------------------------------------------------
# Value parsing (fromPair / fromPart) into flat component lists
# ---------------------------------------------------------------------------

def _generic_types(t: str) -> list[str]:
    return t[t.index("{") + 1:-1].split(PROP_SEP)


def _part(kind: str, s: str) -> Any:
    if kind in ("F", "PF"):
        return kfloat(s)
    if kind in ("I", "PI"):
        return kint(s)
    if kind == "IF":  # SmearParams distance/shape: parsed with toFloat() then toInt()
        return int(kfloat(s))
    if kind == "CF":  # RubberStampParams colours: toFloat() then toInt() (float32 precision, like ImageToolbox)
        return sint32(int(kfloat(s)))
    if kind in ("C", "PC"):
        return kint(s)
    if kind == "B":
        return kbool(s)
    if kind.startswith("E:"):
        return s.strip()
    if kind == "S":
        return b64d(s)
    if kind == "FONT":
        return s.strip()
    if kind == "FP":
        x, y = s.split(ADD_PROP_SEP)[:2]
        return (kfloat(x), kfloat(y))
    if kind == "FILE":
        return None
    raise ValueError(f"unknown component kind {kind}")


def parse_components(entry: dict[str, Any], vtype: str, value: str) -> list[Any]:
    """Parse a serialised value into the flat component list described by ``entry['m']``."""
    t = entry["t"]
    leaves = entry.get("m") or []
    if t == "Unit":
        return []
    if t in ("Float", "Int", "Boolean", "String"):
        conv = {"Float": kfloat, "Int": kint, "Boolean": kbool, "String": b64d}[t]
        return [conv(value)]
    if t == "FloatArray":
        return [[kfloat(v) for v in value.split(PROP_SEP)]]
    if t == "FilterValueWrapper{ColorModel}":
        return [kint(value)]
    if t in ("PolarCoordinatesType",):
        return [value.strip()]
    if t == "GradientPalette":
        return [value.strip()]
    if t == "GmicFilterParams":
        return [b64d(v) for v in value.split(PROP_SEP)]
    if t == "ToneCurvesParams":
        groups = []
        for g in value.split(PROP_SEP):
            vals = []
            for x in g.split(ADD_PROP_SEP):
                try:
                    vals.append(kfloat(x))
                except (ValueError, IndexError):
                    vals.append(0.0)  # toFloatOrNull() ?: 0f
            groups.append(vals)
        return [groups]
    if t == "ProceduralParams":
        decoded = b64d(value)
        vals: dict[str, float] = {}
        cols: dict[str, int] = {}
        for item in decoded.split(";"):
            if not item:
                continue
            key, num = item.split("=", 1)
            if key.startswith("color:"):
                cols[key[len("color:"):]] = kint(num)
            elif ":" not in key:
                vals[key] = kfloat(num)
        return [cols.get(lf[2]) if lf[0] == "PC" else vals.get(lf[2]) for lf in leaves]
    if t == "NtscParams":  # NtscParams.fromPair: floatAt/intAt/booleanAt with per-field fallbacks
        parts = value.split(PROP_SEP)
        out: list[Any] = []
        for i, lf in enumerate(leaves):
            raw = parts[i] if i < len(parts) else None
            if raw is None:
                out.append(None)
            elif lf[0] == "B":
                out.append(True if raw == "true" else False if raw == "false" else None)
            else:
                try:
                    out.append(_part(lf[0], raw))
                except ValueError:
                    out.append(None)
        return out
    if t.startswith(("Pair{", "Triple{", "Quad{")) or leaves:
        parts = value.split(PROP_SEP)
        if t == "DistortPerspectiveParams":
            nums = [kfloat(p) for p in parts[:8]]
            return [(nums[0], nums[1]), (nums[2], nums[3]), (nums[4], nums[5]), (nums[6], nums[7]),
                    kbool(parts[8])]
        if len(parts) < len(leaves):
            raise ValueError(f"{t}: expected {len(leaves)} components, got {len(parts)}")
        return [_part(lf[0], parts[i]) for i, lf in enumerate(leaves)]
    raise ValueError(f"unsupported value type {t}")


# ---------------------------------------------------------------------------
# ImageToolbox filter -> LazyRoom step
# ---------------------------------------------------------------------------

def _op_params(op_id: str) -> dict[str, Any]:
    from ..core.registry import get_op

    return {p.name: p for p in get_op(op_id).params}


def _clamp(p: Any, v: float, warn: list[str], where: str) -> float:
    lo, hi = getattr(p, "min", None), getattr(p, "max", None)
    if lo is not None and v < lo:
        warn.append(f"{where}: {v} below the LazyRoom range, clamped to {lo}")
        v = lo
    if hi is not None and v > hi:
        warn.append(f"{where}: {v} above the LazyRoom range, clamped to {hi}")
        v = hi
    return v


def _palette_stops(colors: list[int]) -> list[list[Any]]:
    n = len(colors)
    return [[i / (n - 1), argb_to_hex(c)] for i, c in enumerate(colors)]


def _curve_points(vals: list[float]) -> list[list[float]]:
    if len(vals) == 5:  # legacy Y-only curve
        pts = [[i / 4, y] for i, y in enumerate(vals)]
    else:
        pts = [[vals[i], vals[i + 1]] for i in range(0, len(vals) - 1, 2)]
    pts = sorted(([min(1.0, max(0.0, x)), min(1.0, max(0.0, y))] for x, y in pts), key=lambda p: p[0])
    dedup: list[list[float]] = []
    for p in pts:  # distinctBy { it.x }
        if not dedup or dedup[-1][0] != p[0]:
            dedup.append(p)
    return dedup


def _convert_in(kind: str, extra: Any, p: Any, v: Any, warn: list[str], where: str) -> Any:
    """ImageToolbox component -> LazyRoom param JSON value (None = keep the op default)."""
    if v is None:
        return None
    pk = p.kind
    if kind in ("C", "CF", "PC", "G:color"):
        return argb_to_hex(int(v) if not isinstance(v, str) else kint(v))
    if isinstance(extra, dict) and "index" in extra:
        idx = int(round(float(v)))
        choices = extra["index"]
        if not 0 <= idx < len(choices):
            warn.append(f"{where}: index {idx} out of range, default kept")
            return None
        return choices[idx]
    if kind.startswith("E:") or kind == "G:sel":
        if isinstance(extra, dict) and v in extra:
            return extra[v]
        warn.append(f"{where}: unknown value {v!r}, default kept")
        return None
    if kind == "G:toggle":
        return kbool(v)
    if kind in ("G:num_i", "G:num_f"):
        v = kfloat(v)
    if kind == "S" or kind == "FONT":
        if kind == "FONT":
            try:
                i = int(v)
            except ValueError:
                i = 0
            return "sans-serif" if i <= 0 or i >= len(FONTS) else FONTS[i]
        return str(v)
    if kind == "B":
        return bool(v)
    if kind in ("FP", "FP2"):
        return [float(v[0]), float(v[1])]
    if kind == "FA":
        vals = [float(x) for x in v]
        length = getattr(p, "length", None)
        if length is not None and len(vals) != length:
            warn.append(f"{where}: expected {length} values, got {len(vals)}; default kept")
            return None
        return vals
    if kind == "GP":
        if v in PALETTES:
            return _palette_stops(PALETTES[v])
        if isinstance(v, str) and v.startswith("custom;"):
            cols = [sint32(int(h, 16)) for h in v[len("custom;"):].split(";") if len(h) == 8]
            if len(cols) >= 2:
                return _palette_stops(cols)
        warn.append(f"{where}: unknown gradient palette {v!r}, default kept")
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if pk == "int":
            v = int(round(float(v)))
        elif pk == "float":
            v = float(v)
        if pk in ("int", "float"):
            v = _clamp(p, v, warn, where)
            return int(v) if pk == "int" else float(v)
    return v


def filter_to_step(cls: str, vtype: str | None, value: str | None) -> tuple[dict[str, Any] | None, list[str]]:
    """One ImageToolbox template filter -> LazyRoom step dict (or None when it must be skipped)."""
    warn: list[str] = []
    name = filter_name(cls)
    if name is None:
        return None, [f"{cls}: unknown ImageToolbox filter class, skipped"]
    entry = FILTERS[name]
    from ..core.registry import get_op

    label = get_op(entry["op"]).title
    if vtype is None or value is None:
        # ImageToolbox writes '<class>:null' for values it cannot serialise and then drops the filter on import.
        return None, [f"{name}: value not serialised ('{(vtype or 'null')}'); ImageToolbox drops this filter too, skipped"]
    if entry.get("unsupported"):
        return None, [f"{name}: {entry['unsupported']}; skipped"]
    step: dict[str, Any] = {"op": entry["op"], "params": dict(entry.get("fixed", {})), "label": label,
                            "origin": "import"}
    if vtype.strip() != entry["t"]:
        warn.append(f"{name}: value type '{vtype}' differs from '{entry['t']}'; ImageToolbox falls back to the "
                    "filter defaults, so defaults are used")
        return step, warn
    try:
        comps = parse_components(entry, vtype, value)
    except (ValueError, IndexError, UnicodeDecodeError, base64.binascii.Error) as exc:
        return None, [f"{name}: cannot parse value {value!r} ({exc}); ImageToolbox drops it too, skipped"]
    params = _op_params(entry["op"])
    leaves = entry.get("m") or []
    if entry["t"] == "ToneCurvesParams":
        groups = comps[0]
        for i, pname in enumerate(_CURVE_PARAMS):
            if i < len(groups) and len(groups[i]) >= 2:
                pts = _curve_points(groups[i])
                if len(pts) >= 2:
                    step["params"][pname] = pts
        ignored = [i for i, g in enumerate(groups[4:17], start=4) if g and _curve_points(g) != _curve_points(
            _DEFAULT_CURVES[i])]
        wheels = groups[17:21]
        if ignored or any(g and g != d for g, d in zip(wheels, _DEFAULT_WHEELS, strict=False)):
            warn.append(f"{name}: CMYK/LAB/hue curves or colour wheels are not supported by filter.tone_curves "
                        "and were ignored")
        return step, warn
    file_leaves = [lf for lf in leaves if lf[0] == "FILE"]
    if file_leaves:
        req = [lf[1] for lf in file_leaves if getattr(params.get(lf[1]), "required", False)]
        if req:
            return None, [f"{name}: needs a file ({', '.join(req)}) that ImageToolbox templates do not carry; skipped"]
        warn.append(f"{name}: ImageToolbox templates do not carry the image/file argument "
                    f"({', '.join(lf[1] for lf in file_leaves)}); LazyRoom's default is used")
    for lf, v in zip(leaves, comps, strict=False):
        kind, pname = lf[0], lf[1]
        extra = lf[2] if len(lf) > 2 else None
        if kind == "FILE" or pname not in params:
            continue
        if entry["t"] == "ProceduralParams" and v is None:
            continue
        out = _convert_in(kind, extra, params[pname], v, warn, f"{name}.{pname}")
        if out is not None:
            step["params"][pname] = out
    return step, warn


def import_template(content: str) -> dict[str, Any]:
    name, raw = split_template(content)
    steps, warnings, skipped = [], [], []
    for cls, vtype, value in raw:
        step, warn = filter_to_step(cls, vtype, value)
        warnings.extend(warn)
        if step is None:
            skipped.append(cls.rsplit(".", 1)[-1])
        else:
            steps.append(step)
    return {"name": name, "steps": steps, "warnings": warnings, "skipped": skipped, "filters_in_template": len(raw)}


# ---------------------------------------------------------------------------
# LazyRoom step -> ImageToolbox filter
# ---------------------------------------------------------------------------

def _rev_enum(extra: dict[str, str], v: str, where: str) -> str:
    for k, c in extra.items():
        if c == v:
            return k
    raise invalid(f"{where}: value {v!r} has no ImageToolbox equivalent")


def _ser_part(kind: str, extra: Any, v: Any, where: str) -> str:
    if isinstance(extra, dict) and "index" in extra:
        return str(extra["index"].index(v))
    if kind.startswith("E:"):
        return _rev_enum(extra, v, where)
    if kind in ("F", "PF"):
        return jfloat(float(v))
    if kind in ("I", "IF", "PI"):
        return str(int(round(float(v))))
    if kind in ("C", "CF", "PC"):
        return str(hex_to_argb(v))
    if kind == "B":
        return "true" if v else "false"
    if kind == "S":
        return b64e(str(v))
    if kind == "FP":
        return jfloat(v[0]) + ADD_PROP_SEP + jfloat(v[1])
    if kind == "FP2":
        return jfloat(v[0]) + PROP_SEP + jfloat(v[1])
    if kind == "FONT":
        s = str(v).strip().lower().replace(" ", "")
        for i, f in enumerate(FONTS):
            if i and f.lower() == s:
                return str(i)
        return "0"  # System (ImageToolbox also writes System for custom fonts)
    if kind == "FILE":
        return ""
    raise invalid(f"{where}: cannot serialise component kind {kind}")


def _gradient_to_palette(stops: list[Any], warn: list[str], where: str) -> str:
    pos = [float(s[0]) for s in stops]
    cols = [hex_to_argb(s[1]) for s in stops]
    n = len(stops)
    even = n >= 2 and all(abs(p - i / (n - 1)) < 1e-4 for i, p in enumerate(pos))
    if not even:  # ImageToolbox palettes are evenly spaced: resample the gradient at 33 positions
        from ..core.params import parse_color as pc

        xs = np.linspace(0, 1, 33)
        rgba = np.array([pc(s[1]) for s in stops], dtype=np.float64)
        samp = np.stack([np.interp(xs, pos, rgba[:, c]) for c in range(4)], -1)
        cols = [hex_to_argb(color_to_hex(c)) for c in samp]
        warn.append(f"{where}: unevenly spaced gradient stops resampled to 33 evenly spaced colours")
    for pname, pcols in PALETTES.items():
        if pcols == cols:
            return pname
    return "custom;" + ";".join(f"{c & 0xFFFFFFFF:08x}" for c in cols)


def step_to_filter(step: dict[str, Any], name: str | None = None) -> tuple[str, list[str]]:
    """LazyRoom step (validated, JSON params) -> ImageToolbox '<class>:<type>:<value>' string."""
    from ..core.registry import get_op

    warn: list[str] = []
    spec = get_op(step["op"])
    params = spec.params_json(spec.validate(step.get("params") or {}))
    names = BY_OP.get(spec.id, [])
    if not names:
        raise ValueError("not an ImageToolbox filter")
    if name is None:
        name = names[0]
        if len(names) > 1:  # colour maps: pick the class whose fixed choice matches
            for n in names:
                fixed = FILTERS[n].get("fixed") or {}
                if all(params.get(k) == v for k, v in fixed.items()):
                    name = n
                    break
    entry = FILTERS[name]
    if entry.get("unsupported"):
        raise ValueError(entry["unsupported"])
    t = entry["t"]
    cls = UI_PACKAGE + entry["cls"]
    where = spec.id
    leaves = entry.get("m") or []
    mapped = {lf[1] for lf in leaves} | set(entry.get("fixed", {}))
    defaults = spec.defaults()
    dropped = [k for k, v in params.items() if k not in mapped and v != defaults.get(k)]
    if t == "ToneCurvesParams":
        dropped = []
    if dropped:
        warn.append(f"{where}: {', '.join(dropped)} cannot be represented in ImageToolbox templates (dropped)")
    if t == "Unit":
        return f"{cls}:Unit:Unit", warn
    if t in ("Float", "Int", "Boolean", "String"):
        kind, pname = leaves[0][0], leaves[0][1]
        v = params[pname]
        val = {"Float": lambda x: jfloat(float(x)), "Int": lambda x: str(int(x)),
               "Boolean": lambda x: "true" if x else "false", "String": lambda x: b64e(str(x))}[t](v)
        if t == "Boolean":
            warn.append(f"{where}: ImageToolbox itself saves Boolean-valued filters as ':null' (dropped on its own "
                        "export) but reads 'Boolean:<value>' correctly; written in the readable form")
        return f"{cls}:{t}:{val}", warn
    if t == "FloatArray":
        return f"{cls}:FloatArray:" + PROP_SEP.join(jfloat(x) for x in params[leaves[0][1]]), warn
    if t == "FilterValueWrapper{ColorModel}":
        return f"{cls}:{t}:{hex_to_argb(params[leaves[0][1]])}", warn
    if t == "PolarCoordinatesType":
        lf = leaves[0]
        return f"{cls}:{t}:{_rev_enum(lf[2], params[lf[1]], where)}", warn
    if t == "GradientPalette":
        return f"{cls}:{t}:{_gradient_to_palette(params[leaves[0][1]], warn, where)}", warn
    if t == "GmicFilterParams":
        out = []
        for lf in leaves:
            kind, pname = lf[0], lf[1]
            v = params[pname]
            if kind == "G:num_i":
                s = str(int(round(float(v))))
            elif kind == "G:num_f":
                s = jfloat(float(v))
            elif kind == "G:toggle":
                s = "true" if v else "false"
            elif kind == "G:color":
                s = str(hex_to_argb(v))
            else:
                s = _rev_enum(lf[2], v, f"{where}.{pname}")
            out.append(b64e(s))
        return f"{cls}:{t}:" + PROP_SEP.join(out), warn
    if t == "ProceduralParams":
        items, colors = [], []
        for lf in leaves:
            kind, pname, key = lf[0], lf[1], lf[2]
            v = params[pname]
            if kind == "PC":
                colors.append(f"color:{key}={hex_to_argb(v)}")
            else:
                items.append(f"{key}={jfloat(float(v))}")
        return f"{cls}:{t}:" + b64e(";".join(items + colors)), warn
    if t == "ToneCurvesParams":
        groups = [list(g) for g in _DEFAULT_CURVES] + [list(w) for w in _DEFAULT_WHEELS]
        for i, pname in enumerate(_CURVE_PARAMS):
            groups[i] = [c for pt in params[pname] for c in pt]
        return f"{cls}:{t}:" + PROP_SEP.join(ADD_PROP_SEP.join(jfloat(x) for x in g) for g in groups), warn
    # Pair/Triple/Quad and the *Params data classes: '$'-joined components in serialisation order
    out = []
    for lf in leaves:
        kind, pname = lf[0], lf[1]
        extra = lf[2] if len(lf) > 2 else None
        if kind == "FILE":
            warn.append(f"{where}: the {pname} file cannot travel in an ImageToolbox template (written empty, "
                        "as ImageToolbox does)")
            out.append("")
            continue
        out.append(_ser_part(kind, extra, params[pname], f"{where}.{pname}"))
    return f"{cls}:{t}:" + PROP_SEP.join(out), warn


def flatten_steps(steps: list[dict[str, Any]], warn: list[str]) -> list[dict[str, Any]]:
    out = []
    for s in steps:
        if s.get("op") == "group":
            if s.get("opacity", 1.0) != 1.0 or s.get("mask") or s.get("blend", "normal") != "normal":
                warn.append(f"group '{s.get('label') or s.get('id')}': opacity/blend/mask cannot be represented and "
                            "were dropped")
            if not s.get("enabled", True):
                warn.append(f"group '{s.get('label') or s.get('id')}' is disabled and was skipped")
                continue
            out.extend(flatten_steps(s.get("steps") or [], warn))
            continue
        if not s.get("enabled", True):
            warn.append(f"{s.get('op')}: disabled step skipped")
            continue
        if s.get("opacity", 1.0) != 1.0 or s.get("mask") or s.get("blend", "normal") != "normal":
            warn.append(f"{s.get('op')}: step opacity/blend/mask cannot be represented and were dropped")
        out.append(s)
    return out


def export_template(name: str, steps: list[dict[str, Any]], *, skip_unsupported: bool = False) -> dict[str, Any]:
    from ..core.registry import get_op

    warn: list[str] = []
    flat = flatten_steps(steps, warn)
    bad: list[dict[str, str]] = []
    parts: list[str] = []
    for s in flat:
        op_id = get_op(s["op"]).id
        if op_id not in BY_OP:
            bad.append({"op": op_id, "reason": "not an ImageToolbox filter (no ImageToolbox equivalent)"})
            continue
        try:
            part, w = step_to_filter(s)
        except ValueError as exc:
            bad.append({"op": op_id, "reason": str(exc)})
            continue
        parts.append(part)
        warn.extend(w)
    if bad and not skip_unsupported:
        raise invalid(f"{len(bad)} step(s) cannot be exported to an ImageToolbox template: "
                      + ", ".join(sorted({b['op'] for b in bad})), unsupported=bad,
                      hint="Remove them, or pass skip_unsupported=true to export the remaining filters.")
    if bad:
        warn.extend(f"{b['op']}: skipped ({b['reason']})" for b in bad)
    if not parts:
        raise invalid("Nothing to export: no ImageToolbox filters in the steps")
    clean = name.strip()
    for ch in (CONTENT_SEP, TEMPLATES_SEP):
        if ch in clean:
            warn.append(f"template name: '{ch}' is a template separator in ImageToolbox and was replaced by '_'")
            clean = clean.replace(ch, "_")
    body = FILTERS_SEP.join(parts)
    if CONTENT_SEP in body:
        warn.append("a Base64 value contains '+', which ImageToolbox's template parser treats as a separator: "
                    "ImageToolbox will read only the filters before it")
    return {"template": LINK_HEADER + clean + CONTENT_SEP + body, "name": clean, "filters": len(parts),
            "warnings": warn, "unsupported": bad}


__all__ = ["LINK_HEADER", "is_valid_template", "import_template", "export_template", "jfloat", "kfloat",
           "split_template", "filter_to_step", "step_to_filter"]
