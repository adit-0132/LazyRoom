"""Typed, self-describing operation parameters.

A single parameter declaration drives:
  * strict validation / coercion of agent- or user-supplied values,
  * JSON Schema generation (MCP tool schemas, CLI ``describe``),
  * automatic GUI control generation (slider, switch, colour picker, ...),
  * human/agent documentation (range, unit, default, description).

Units: ``unit="px"`` marks a value measured in pixels of the *full-resolution*
image.  The render context rescales such values for preview proxies so that a
preview looks like the final export.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .errors import invalid

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{8}|[0-9a-fA-F]{3})$")

NAMED_COLORS = {
    "black": "#000000", "white": "#FFFFFF", "red": "#FF0000", "green": "#00FF00",
    "blue": "#0000FF", "yellow": "#FFFF00", "cyan": "#00FFFF", "magenta": "#FF00FF",
    "gray": "#808080", "grey": "#808080", "orange": "#FFA500", "transparent": "#00000000",
}


@dataclass
class Param:
    name: str
    default: Any = None
    title: str | None = None
    description: str = ""
    unit: str | None = None
    advanced: bool = False

    kind: str = field(default="any", init=False)

    @property
    def label(self) -> str:
        return self.title or self.name.replace("_", " ").capitalize()

    # -- validation -------------------------------------------------------
    def coerce(self, value: Any) -> Any:  # pragma: no cover - overridden
        return value

    # -- serialisation ----------------------------------------------------
    def to_json(self, value: Any) -> Any:
        return value

    def json_schema(self) -> dict[str, Any]:  # pragma: no cover - overridden
        return {}

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.kind, "default": self.to_json(self.default)}
        if self.title:
            d["title"] = self.title
        if self.description:
            d["description"] = self.description
        if self.unit:
            d["unit"] = self.unit
        if self.advanced:
            d["advanced"] = True
        return d

    def _schema_base(self) -> dict[str, Any]:
        s: dict[str, Any] = {"title": self.label}
        desc = self.description
        if self.unit:
            desc = f"{desc} (unit: {self.unit})".strip()
        if desc:
            s["description"] = desc
        s["default"] = self.to_json(self.default)
        return s


def _num(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise invalid(f"Parameter '{name}' must be a number, got boolean", param=name)
    if isinstance(value, (int, float)):
        v = float(value)
    elif isinstance(value, str):
        try:
            v = float(value.strip())
        except ValueError:
            raise invalid(f"Parameter '{name}' must be a number, got {value!r}", param=name) from None
    else:
        raise invalid(f"Parameter '{name}' must be a number, got {type(value).__name__}", param=name)
    if not math.isfinite(v):
        raise invalid(f"Parameter '{name}' must be finite", param=name)
    return v


@dataclass
class Float(Param):
    min: float | None = None
    max: float | None = None
    step: float | None = None

    def __post_init__(self) -> None:
        self.kind = "float"
        if self.default is None:
            self.default = self.min if self.min is not None else 0.0

    def coerce(self, value: Any) -> float:
        v = _num(self.name, value)
        if (self.min is not None and v < self.min - 1e-9) or (self.max is not None and v > self.max + 1e-9):
            raise invalid(
                f"Parameter '{self.name}'={v} is outside the allowed range [{self.min}, {self.max}]",
                param=self.name, value=v, min=self.min, max=self.max,
                hint=f"Use a value between {self.min} and {self.max}.",
            )
        return v

    def json_schema(self) -> dict[str, Any]:
        s = {"type": "number", **self._schema_base()}
        if self.min is not None:
            s["minimum"] = self.min
        if self.max is not None:
            s["maximum"] = self.max
        return s

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["min"], d["max"] = self.min, self.max
        if self.step:
            d["step"] = self.step
        return d


@dataclass
class Int(Param):
    min: int | None = None
    max: int | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        self.kind = "int"
        if self.default is None:
            self.default = self.min if self.min is not None else 0

    def coerce(self, value: Any) -> int:
        v = _num(self.name, value)
        if abs(v - round(v)) > 1e-6:
            raise invalid(f"Parameter '{self.name}' must be an integer, got {v}", param=self.name)
        v = int(round(v))
        if (self.min is not None and v < self.min) or (self.max is not None and v > self.max):
            raise invalid(
                f"Parameter '{self.name}'={v} is outside the allowed range [{self.min}, {self.max}]",
                param=self.name, value=v, min=self.min, max=self.max,
                hint=f"Use an integer between {self.min} and {self.max}.",
            )
        return v

    def json_schema(self) -> dict[str, Any]:
        s = {"type": "integer", **self._schema_base()}
        if self.min is not None:
            s["minimum"] = self.min
        if self.max is not None:
            s["maximum"] = self.max
        return s

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["min"], d["max"] = self.min, self.max
        return d


@dataclass
class Bool(Param):
    def __post_init__(self) -> None:
        self.kind = "bool"
        if self.default is None:
            self.default = False

    def coerce(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            return value.strip().lower() in ("true", "1", "yes", "on")
        raise invalid(f"Parameter '{self.name}' must be a boolean, got {value!r}", param=self.name)

    def json_schema(self) -> dict[str, Any]:
        return {"type": "boolean", **self._schema_base()}


@dataclass
class Choice(Param):
    choices: Sequence[str] = ()
    labels: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self.kind = "choice"
        if self.default is None and self.choices:
            self.default = self.choices[0]

    def coerce(self, value: Any) -> str:
        if not isinstance(value, str):
            value = str(value)
        if value in self.choices:
            return value
        low = {c.lower(): c for c in self.choices}
        norm = value.strip().lower().replace(" ", "_").replace("-", "_")
        for cand in (value.lower(), norm):
            if cand in low:
                return low[cand]
        raise invalid(
            f"Parameter '{self.name}' must be one of {list(self.choices)}, got {value!r}",
            param=self.name, choices=list(self.choices),
        )

    def json_schema(self) -> dict[str, Any]:
        return {"type": "string", "enum": list(self.choices), **self._schema_base()}

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["choices"] = list(self.choices)
        if self.labels:
            d["labels"] = self.labels
        return d


def parse_color(value: Any, name: str = "color") -> tuple[float, float, float, float]:
    """Parse '#RRGGBB', '#RRGGBBAA', '#RGB', a named colour, or a 3/4 float list (0..1)."""
    if isinstance(value, str):
        s = NAMED_COLORS.get(value.strip().lower(), value.strip())
        m = _HEX_RE.match(s)
        if not m:
            raise invalid(f"Parameter '{name}' is not a valid colour: {value!r}", param=name,
                          hint="Use '#RRGGBB', '#RRGGBBAA' or [r, g, b(, a)] with components in 0..1.")
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        comps = [int(h[i:i + 2], 16) / 255.0 for i in range(0, len(h), 2)]
        if len(comps) == 3:
            comps.append(1.0)
        return tuple(comps)  # type: ignore[return-value]
    if isinstance(value, (list, tuple)) and len(value) in (3, 4):
        comps = [_num(name, c) for c in value]
        if any(c > 1.0 for c in comps[:3]) and all(c <= 255 for c in comps):
            comps = [c / 255.0 for c in comps[:3]] + comps[3:]
        if len(comps) == 3:
            comps.append(1.0)
        return tuple(max(0.0, min(1.0, c)) for c in comps)  # type: ignore[return-value]
    raise invalid(f"Parameter '{name}' is not a valid colour: {value!r}", param=name)


def color_to_hex(c: Sequence[float]) -> str:
    vals = [max(0, min(255, int(round(x * 255)))) for x in c]
    if len(vals) == 4 and vals[3] == 255:
        vals = vals[:3]
    return "#" + "".join(f"{v:02X}" for v in vals)


@dataclass
class Color(Param):
    """RGBA colour. JSON form: '#RRGGBB' or '#RRGGBBAA'. Python value: (r, g, b, a) floats."""

    def __post_init__(self) -> None:
        self.kind = "color"
        if self.default is None:
            self.default = "#000000"
        self.default = parse_color(self.default, self.name)

    def coerce(self, value: Any) -> tuple[float, float, float, float]:
        return parse_color(value, self.name)

    def to_json(self, value: Any) -> str:
        return color_to_hex(value)

    def json_schema(self) -> dict[str, Any]:
        return {"type": "string", "pattern": "^#([0-9a-fA-F]{6}|[0-9a-fA-F]{8})$", **self._schema_base()}


@dataclass
class Text(Param):
    max_length: int = 10_000
    multiline: bool = False

    def __post_init__(self) -> None:
        self.kind = "text"
        if self.default is None:
            self.default = ""

    def coerce(self, value: Any) -> str:
        if not isinstance(value, str):
            value = str(value)
        if len(value) > self.max_length:
            raise invalid(f"Parameter '{self.name}' exceeds {self.max_length} characters", param=self.name)
        return value

    def json_schema(self) -> dict[str, Any]:
        return {"type": "string", "maxLength": self.max_length, **self._schema_base()}


@dataclass
class Point(Param):
    """2D point. Normalised image coordinates (0..1) unless ``unit='px'``."""
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        self.kind = "point"
        if self.default is None:
            self.default = (0.5, 0.5)
        self.default = tuple(self.default)

    def coerce(self, value: Any) -> tuple[float, float]:
        if isinstance(value, dict) and "x" in value and "y" in value:
            value = (value["x"], value["y"])
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise invalid(f"Parameter '{self.name}' must be [x, y]", param=self.name)
        x, y = (_num(self.name, v) for v in value)
        for v in (x, y):
            if (self.min is not None and v < self.min) or (self.max is not None and v > self.max):
                raise invalid(f"Parameter '{self.name}' coordinate {v} outside [{self.min}, {self.max}]",
                              param=self.name)
        return (x, y)

    def to_json(self, value: Any) -> list[float]:
        return [float(value[0]), float(value[1])]

    def json_schema(self) -> dict[str, Any]:
        item: dict[str, Any] = {"type": "number"}
        if self.min is not None:
            item["minimum"] = self.min
        if self.max is not None:
            item["maximum"] = self.max
        return {"type": "array", "items": item, "minItems": 2, "maxItems": 2, **self._schema_base()}


@dataclass
class Points(Param):
    """List of 2D points, e.g. curve control points [[x, y], ...]."""
    min_items: int = 0
    max_items: int = 4096
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        self.kind = "points"
        if self.default is None:
            self.default = []
        self.default = [tuple(p) for p in self.default]

    def coerce(self, value: Any) -> list[tuple[float, float]]:
        if not isinstance(value, (list, tuple)):
            raise invalid(f"Parameter '{self.name}' must be a list of [x, y] points", param=self.name)
        if not (self.min_items <= len(value) <= self.max_items):
            raise invalid(f"Parameter '{self.name}' needs {self.min_items}..{self.max_items} points, got {len(value)}",
                          param=self.name)
        pt = Point(self.name, min=self.min, max=self.max)
        return [pt.coerce(p) for p in value]

    def to_json(self, value: Any) -> list[list[float]]:
        return [[float(p[0]), float(p[1])] for p in value]

    def json_schema(self) -> dict[str, Any]:
        item: dict[str, Any] = {"type": "number"}
        if self.min is not None:
            item["minimum"] = self.min
        if self.max is not None:
            item["maximum"] = self.max
        return {"type": "array", "minItems": self.min_items, "maxItems": self.max_items,
                "items": {"type": "array", "items": item, "minItems": 2, "maxItems": 2},
                **self._schema_base()}


@dataclass
class FloatList(Param):
    length: int | None = None
    min_length: int = 0
    max_length: int = 4096
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        self.kind = "float_list"
        if self.default is None:
            self.default = [0.0] * (self.length or 0)
        self.default = [float(v) for v in self.default]

    def coerce(self, value: Any) -> list[float]:
        if isinstance(value, str):
            value = [v for v in re.split(r"[,\s]+", value.strip()) if v]
        if not isinstance(value, (list, tuple)):
            raise invalid(f"Parameter '{self.name}' must be a list of numbers", param=self.name)
        vals = [_num(self.name, v) for v in value]
        if self.length is not None and len(vals) != self.length:
            raise invalid(f"Parameter '{self.name}' must have exactly {self.length} numbers, got {len(vals)}",
                          param=self.name)
        if not (self.min_length <= len(vals) <= self.max_length):
            raise invalid(f"Parameter '{self.name}' length must be in {self.min_length}..{self.max_length}",
                          param=self.name)
        for v in vals:
            if (self.min is not None and v < self.min) or (self.max is not None and v > self.max):
                raise invalid(f"Parameter '{self.name}' value {v} outside [{self.min}, {self.max}]", param=self.name)
        return vals

    def json_schema(self) -> dict[str, Any]:
        item: dict[str, Any] = {"type": "number"}
        if self.min is not None:
            item["minimum"] = self.min
        if self.max is not None:
            item["maximum"] = self.max
        s: dict[str, Any] = {"type": "array", "items": item, **self._schema_base()}
        if self.length is not None:
            s["minItems"] = s["maxItems"] = self.length
        return s

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        if self.length is not None:
            d["length"] = self.length
        return d


@dataclass
class File(Param):
    """Path to an auxiliary file (image, LUT, font, mask...).

    ``kind`` hints at the expected type: image | lut | font | any.  Paths are
    resolved and checked against the filesystem policy at render time.
    """
    file_kind: str = "image"
    required: bool = False

    def __post_init__(self) -> None:
        self.kind = "file"
        if self.default is None:
            self.default = ""

    def coerce(self, value: Any) -> str:
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise invalid(f"Parameter '{self.name}' must be a file path string", param=self.name)
        if self.required and not value:
            raise invalid(f"Parameter '{self.name}' requires a {self.file_kind} file path", param=self.name)
        return value

    def json_schema(self) -> dict[str, Any]:
        return {"type": "string", "x-file-kind": self.file_kind, **self._schema_base()}

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["file_kind"] = self.file_kind
        d["required"] = self.required
        return d


@dataclass
class StringList(Param):
    """List of strings (e.g. input file paths, glob patterns, folders). A single string is accepted."""
    min_items: int = 0
    max_items: int = 100_000
    file_kind: str | None = None

    def __post_init__(self) -> None:
        self.kind = "string_list"
        if self.default is None:
            self.default = []

    def coerce(self, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise invalid(f"Parameter '{self.name}' must be a list of strings", param=self.name)
        if not (self.min_items <= len(value) <= self.max_items):
            raise invalid(f"Parameter '{self.name}' needs {self.min_items}..{self.max_items} items, got {len(value)}",
                          param=self.name)
        return list(value)

    def json_schema(self) -> dict[str, Any]:
        s = {"type": "array", "items": {"type": "string"}, "minItems": self.min_items, **self._schema_base()}
        if self.file_kind:
            s["x-file-kind"] = self.file_kind
        return s


@dataclass
class Gradient(Param):
    """Colour gradient: list of [position 0..1, '#RRGGBB(AA)'] stops."""

    def __post_init__(self) -> None:
        self.kind = "gradient"
        if self.default is None:
            self.default = [[0.0, "#000000"], [1.0, "#FFFFFF"]]
        self.default = self.coerce(self.default)

    def coerce(self, value: Any) -> list[tuple[float, tuple[float, float, float, float]]]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise invalid(f"Parameter '{self.name}' must be a list of at least two [position, colour] stops",
                          param=self.name)
        stops = []
        for s in value:
            if isinstance(s, dict):
                s = (s.get("pos", s.get("position")), s.get("color"))
            if not isinstance(s, (list, tuple)) or len(s) != 2:
                raise invalid(f"Gradient stop must be [position, colour], got {s!r}", param=self.name)
            pos = _num(self.name, s[0])
            if not 0.0 <= pos <= 1.0:
                raise invalid(f"Gradient stop position {pos} outside 0..1", param=self.name)
            stops.append((pos, parse_color(s[1], self.name)))
        stops.sort(key=lambda t: t[0])
        return stops

    def to_json(self, value: Any) -> list[list[Any]]:
        return [[float(p), color_to_hex(c)] for p, c in value]

    def json_schema(self) -> dict[str, Any]:
        return {"type": "array", "minItems": 2,
                "items": {"type": "array", "prefixItems": [{"type": "number", "minimum": 0, "maximum": 1},
                                                           {"type": "string"}], "minItems": 2, "maxItems": 2},
                **self._schema_base()}


def validate_params(params: Sequence[Param], values: dict[str, Any] | None, *, op_id: str = "") -> dict[str, Any]:
    """Validate a parameter dict against declarations, filling defaults.

    Unknown parameter names are rejected (with suggestions) so agents get a
    clear signal instead of silently ignored typos.
    """
    values = dict(values or {})
    by_name = {p.name: p for p in params}
    unknown = [k for k in values if k not in by_name]
    if unknown:
        import difflib

        sugg = {k: difflib.get_close_matches(k, list(by_name), n=2) for k in unknown}
        raise invalid(
            f"Unknown parameter(s) {unknown} for operation '{op_id}'",
            unknown=unknown, valid=list(by_name), suggestions=sugg,
            hint=f"Valid parameters: {', '.join(by_name) or '(none)'}",
        )
    out: dict[str, Any] = {}
    for p in params:
        if p.name in values and values[p.name] is not None:
            out[p.name] = p.coerce(values[p.name])
        else:
            out[p.name] = p.coerce(p.to_json(p.default)) if p.kind in ("color", "gradient") else p.default
    return out


def params_to_json(params: Sequence[Param], values: dict[str, Any]) -> dict[str, Any]:
    by_name = {p.name: p for p in params}
    return {k: (by_name[k].to_json(v) if k in by_name else v) for k, v in values.items()}


def params_schema(params: Sequence[Param]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {p.name: p.json_schema() for p in params},
        "additionalProperties": False,
    }
