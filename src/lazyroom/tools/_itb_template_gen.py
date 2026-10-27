"""DEV-TIME generator of ``_itb_template_table.py`` (not imported at runtime).

Reads the ImageToolbox Kotlin sources (``Ui*Filter`` value types and defaults, ``*Params`` data classes, enums,
G'MIC parameter infos, ``ProceduralEffect``, ``GradientPalette``) and the live LazyRoom registry, maps every
ImageToolbox filter value component to its ``filter.*`` op parameter (positional, as the ports mirror
``paramsInfo``; by name for procedural/NTSC params; explicit overrides below) and verifies the mapping against
every ImageToolbox default.  Usage::

    LAZYROOM_ITB_ONLY= python -m lazyroom.tools._itb_template_gen /path/to/ImageToolbox

It rewrites ``_itb_template_table.py`` next to this file and prints the mappings that need a human look.
ImageToolbox (https://github.com/T8RIN/ImageToolbox) by T8RIN is Apache-2.0.
"""
# flake8: noqa
from __future__ import annotations

import json
import pathlib
import re
import sys

ITB = pathlib.Path("/tmp/ref/ImageToolbox")
ROOT = MODEL = ENUMS = PALETTE_KT = pathlib.Path()
ENUM: dict = {}
PROC: dict = {}


def configure(itb_root) -> None:
    global ITB, ROOT, MODEL, ENUMS, PALETTE_KT
    ITB = pathlib.Path(itb_root)
    ROOT = ITB / "core/filters/src/main/java/com/t8rin/imagetoolbox/core/filters"
    MODEL = ROOT / "presentation/model"
    ENUMS = ROOT / "domain/model/enums"
    PALETTE_KT = ITB / "core/domain/src/main/kotlin/com/t8rin/imagetoolbox/core/domain/model/GradientPalette.kt"


# ---------------------------------------------------------------- Ui*Filter extraction
def extract_balanced(s, i):
    """from index i read an expression until top-level ',' or unmatched ')'."""
    depth = 0; j = i; instr = False
    while j < len(s):
        c = s[j]
        if instr:
            if c == '\\': j += 2; continue
            if c == '"': instr = False
        elif c == '"': instr = True
        elif c in '([{': depth += 1
        elif c in ')]}':
            if depth == 0: return s[i:j].strip(), j
            depth -= 1
        elif c == ',' and depth == 0: return s[i:j].strip(), j
        j += 1
    return s[i:].strip(), j

def extract_type(s, i):
    depth = 0; j = i
    while j < len(s):
        c = s[j]
        if c == '<': depth += 1
        elif c == '>': depth -= 1
        elif depth == 0 and c in '=,)': return s[i:j].strip(), j
        j += 1



def extract_ui() -> dict:
    out = {}
    for f in sorted(MODEL.glob("Ui*Filter.kt")):
        src = f.read_text()
        m = re.search(r"class (Ui\w+Filter)\s*(?:<[^>]*>)?\s*\(", src)
        if not m:
            continue
        rec = {"file": f.name}
        vm = re.search(r"override val value:\s*", src[m.end():])
        if vm:
            k = m.end() + vm.end()
            typ, j = extract_type(src, k)
            rec["type"] = typ
            if src[j] == "=":
                expr, _ = extract_balanced(src, j + 1)
                rec["default"] = re.sub(r"\s+", " ", expr)
        out[m.group(1)] = rec
    for f in sorted(MODEL.glob("Ui*Filter.kt")):  # classes without a constructor (value = Unit)
        src = f.read_text()
        for m in re.finditer(r"class (Ui\w+Filter)\s*:\s*(\w+)<([^>]*)>\s*\(", src):
            if m.group(1) in out:
                continue
            rec = {"file": f.name, "type": m.group(3).strip(), "noctor": True, "default": "Unit"}
            if f.name == "UiColorMapFilter.kt":
                rec["colormap"] = True
            out[m.group(1)] = rec
    return out


def load_enums() -> None:
    for f in ENUMS.glob("*.kt"):
        s = f.read_text()
        m = re.search(r"enum class (\w+)\s*(\([^)]*\))?\s*\{(.*?)(;|\n\})", s, re.S)
        if not m:
            continue
        body = re.sub(r"/\*.*?\*/", "", m.group(3), flags=re.S)
        body = re.sub(r"//.*", "", body)
        names, depth, cur = [], 0, ""
        for ch in body:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                names.append(cur)
                cur = ""
            else:
                cur += ch
        names.append(cur)
        ENUM[m.group(1)] = [re.match(r"\s*(\w+)", n).group(1) for n in names if n.strip()]


def load_procedural() -> None:
    body = (ENUMS / "ProceduralEffect.kt").read_text().split("enum class ProceduralEffect")[1]
    for m in re.finditer(r"\n    (\w+)\(\s*\n(.*?)\n    \)[,;]", body, re.S):
        name, inner = m.group(1), m.group(2)
        params = [(pm.group(1), float(pm.group(2)), "integer = true" in pm.group(0))
                  for pm in re.finditer(r"ProceduralParameter\(\"(\w+)\",\s*(-?[\d.]+)f,[^)]*\)", inner)]
        colors = [(cm.group(1), int(cm.group(2))) for cm in
                  re.finditer(r"ProceduralColorParameter\(\"(\w+)\",\s*(-?\d+)\)", inner)]
        if "hasTransform = false" not in inner:
            params += [("offsetX", 0.0, False), ("offsetY", 0.0, False), ("scale", 1.0, False),
                       ("rotation", 0.0, False)]
        PROC[name] = {"params": params, "colors": colors}


def dump_ops() -> dict:
    from lazyroom.core.registry import REGISTRY, ensure_loaded

    ensure_loaded()
    out = {}
    for s in REGISTRY.values():
        src = s.source or {}
        if src.get("project") == "ImageToolbox" and "filter" in src:
            ps = []
            for p in s.params:
                d = p.describe()
                d["kind"] = p.kind
                ps.append(json.loads(json.dumps(d, default=str)))
            out[src["filter"]] = {"id": s.id, "params": ps}
    return out


# ---------------------------------------------------------------- params classes: serialisation order (Mappings.kt toPair)
PARAMS = {
    "GlitchParams": [("channelsShiftX", "F", -0.08), ("channelsShiftY", "F", -0.08), ("corruptionSize", "F", 0.01),
                     ("corruptionCount", "I", 60), ("corruptionShiftX", "F", -0.05), ("corruptionShiftY", "F", 0.0)],
    "LinearTiltShiftParams": [("blurRadius", "F", 25), ("sigma", "F", 10), ("anchorX", "F", 0.5), ("anchorY", "F", 0.5),
                              ("holeRadius", "F", 0.2), ("angle", "F", 45)],
    "RadialTiltShiftParams": [("blurRadius", "F", 25), ("sigma", "F", 10), ("anchorX", "F", 0.5), ("anchorY", "F", 0.5),
                              ("holeRadius", "F", 0.2)],
    "EnhancedZoomBlurParams": [("radius", "I", 25), ("sigma", "F", 5), ("centerX", "F", 0.5), ("centerY", "F", 0.5),
                               ("strength", "F", 1), ("angle", "F", 135)],
    "SideFadeParams": [("side", "E:FadeSide", "End"), ("scale", "F", 0.5)],
    "WaterParams": [("fractionSize", "F", 0.2), ("frequencyX", "F", 2), ("frequencyY", "F", 2), ("amplitudeX", "F", 0.5),
                    ("amplitudeY", "F", 0.5)],
    "FlareParams": [("radius", "F", 0.3), ("baseAmount", "F", 1), ("ringAmount", "F", 0.2), ("rayAmount", "F", 0.1),
                    ("ringWidth", "F", 1.6), ("centreX", "F", 0.5), ("centreY", "F", 0.5), ("color", "C", -1)],
    "DistortPerspectiveParams": [("topLeft", "FP2", (0.2, 0.0)), ("topRight", "FP2", (0.8, 0.0)),
                                 ("bottomLeft", "FP2", (0.0, 1.0)), ("bottomRight", "FP2", (1.0, 1.0)),
                                 ("clip", "B", False)],
    "ShearParams": [("xAngle", "F", 15), ("yAngle", "F", 0), ("resize", "B", True)],
    "WaterDropParams": [("wavelength", "F", 0.1), ("amplitude", "F", 0.05), ("phase", "F", 0), ("centreX", "F", 0.5),
                        ("centreY", "F", 0.5), ("radius", "F", 0.5)],
    "ClaheParams": [("threshold", "F", 0.5), ("gridSizeHorizontal", "I", 8), ("gridSizeVertical", "I", 8),
                    ("binsCount", "I", 128)],
    "LinearGaussianParams": [("kernelSize", "I", 25), ("sigma", "F", 10), ("edgeMode", "E:BlurEdgeMode", "Reflect101"),
                             ("transferFunction", "E:TransferFunc", "SRGB")],
    "BilaterialBlurParams": [("radius", "I", 12), ("spatialSigma", "F", 10), ("rangeSigma", "F", 3),
                             ("edgeMode", "E:BlurEdgeMode", "Reflect101")],
    "KaleidoscopeParams": [("angle", "F", 0), ("angle2", "F", 0), ("centreX", "F", 0.5), ("centreY", "F", 0.5),
                           ("sides", "I", 5), ("radius", "F", 0)],
    "ChannelMixParams": [("blueGreen", "I", 255), ("redBlue", "I", 255), ("greenRed", "I", 255), ("intoR", "I", 255),
                         ("intoG", "I", 255), ("intoB", "I", 127)],
    "VoronoiCrystallizeParams": [("borderThickness", "F", 0.4), ("scale", "F", 32), ("randomness", "F", 5),
                                 ("shape", "I", 0), ("turbulence", "F", 1), ("angle", "F", 0), ("stretch", "F", 1),
                                 ("amount", "F", 1), ("color", "C", -16777216)],
    "PinchParams": [("angle", "F", 15), ("centreX", "F", 0.5), ("centreY", "F", 0.5), ("radius", "F", 1), ("amount", "F", 0.5)],
    "RubberStampParams": [("threshold", "F", 0.3), ("softness", "F", 0.1), ("radius", "F", 0.05),
                          ("firstColor", "CF", 0xff3FA08F - (1 << 32)), ("secondColor", "CF", 0xff003027 - (1 << 32))],
    "SmearParams": [("angle", "F", 45), ("density", "F", 1), ("mix", "F", 1), ("distance", "IF", 100), ("shape", "IF", 0)],
    "ArcParams": [("radius", "F", 0.2), ("height", "F", 0.3), ("angle", "F", 0), ("spreadAngle", "F", 360),
                  ("centreX", "F", 0.5), ("centreY", "F", 0.5)],
    "SparkleParams": [("amount", "I", 50), ("rays", "I", 50), ("radius", "F", 0.3), ("randomness", "I", 25),
                      ("centreX", "F", 0.5), ("centreY", "F", 0.5), ("color", "C", -1)],
    "AsciiParams": [("gradient", "S", None), ("fontSize", "F", 10), ("backgroundColor", "C", -16777216),
                    ("isGrayscale", "B", False), ("font", "FONT", None)],
    "BloomParams": [("threshold", "F", 0.6), ("intensity", "F", 1.5), ("radius", "I", 25), ("softKnee", "F", 0.5),
                    ("exposure", "F", 0.02), ("gamma", "F", 1.1)],
    "DropShadowParams": [("color", "C", -16777216), ("offsetX", "F", 0), ("offsetY", "F", 8), ("blurRadius", "F", 18)],
    "TornEdgeParams": [("toothHeight", "I", 16), ("horizontalToothRange", "I", 22), ("verticalToothRange", "I", 22),
                       ("top", "B", True), ("right", "B", True), ("bottom", "B", True), ("left", "B", True)],
    "CropOrPerspectiveParams": [("topLeft", "FP", (0.1, 0.0)), ("topRight", "FP", (1.0, 0.0)),
                                ("bottomLeft", "FP", (0.0, 1.0)), ("bottomRight", "FP", (0.9, 1.0)),
                                ("isAbsolute", "B", False)],
}
NTSC_FIELDS = """amount F 1|chromaBleed F 1|tapeWear F 1|noise F 1|tracking F 1|seed I 0|lumaSmear F 0.5|compositeSharpening F 1|
ringing F 4|snow F 1|useField I 4|filterType I 1|inputLumaFilter I 2|chromaLowpassIn I 2|chromaDemodulation I 1|
videoScanlinePhaseShift I 2|videoScanlinePhaseShiftOffset I 0|headSwitchingHeight I 8|headSwitchingOffset I 3|
headSwitchingHorizontalShift F 72|headSwitchingMidLinePosition F 0.95|headSwitchingMidLineJitter F 0.03|
trackingNoiseHeight I 12|trackingNoiseWaveIntensity F 15|trackingNoiseSnowIntensity F 0.025|
trackingNoiseSnowAnisotropy F 0.25|trackingNoiseNoiseIntensity F 0.25|compositeNoiseFrequency F 0.5|
compositeNoiseIntensity F 0.05|compositeNoiseDetail I 1|ringingFrequency F 0.45|ringingPower F 4|
lumaNoiseFrequency F 0.5|lumaNoiseIntensity F 0.01|lumaNoiseDetail I 1|chromaNoiseFrequency F 0.05|
chromaNoiseIntensity F 0.1|chromaNoiseDetail I 2|snowIntensity F 0.00025|snowAnisotropy F 0.5|
chromaPhaseNoiseIntensity F 0.001|chromaPhaseError F 0|chromaDelayHorizontal F 0|chromaDelayVertical I 0|
vhsTapeSpeed I 2|vhsChromaLoss F 0.000025|vhsSharpenIntensity F 0.25|vhsSharpenFrequency F 1|
vhsEdgeWaveIntensity F 0.5|vhsEdgeWaveSpeed F 4|vhsEdgeWaveFrequency F 0.05|vhsEdgeWaveDetail I 2|
chromaLowpassOut I 2|scaleHorizontal F 1|scaleVertical F 1|scaleFactorX F 1|scaleFactorY F 1|
headSwitchingEnabled B true|trackingNoiseEnabled B true|compositeNoiseEnabled B true|ringingEnabled B true|
lumaNoiseEnabled B true|chromaNoiseEnabled B true|vhsEnabled B true|scaleEnabled B true"""
PARAMS["NtscParams"] = []
for item in NTSC_FIELDS.replace("\n", "").split("|"):
    n, k, v = item.split()
    PARAMS["NtscParams"].append((n, k, {"F": float, "I": int, "B": lambda x: x == "true"}[k](v)))
assert len(PARAMS["NtscParams"]) == 65

NOT_SERIALISABLE = {"DisplacementParams", "SeamCarvingParams", "ShaderParams"}

COLORS = {"Black": 0xFF000000, "White": 0xFFFFFFFF, "Red": 0xFFFF0000, "Green": 0xFF00FF00, "Blue": 0xFF0000FF,
          "Yellow": 0xFFFFFF00, "Cyan": 0xFF00FFFF, "Magenta": 0xFFFF00FF, "Transparent": 0x00000000,
          "Gray": 0xFF888888, "DarkGray": 0xFF444444, "LightGray": 0xFFCCCCCC}


def sint(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


# ---------------------------------------------------------------- tiny Kotlin expression evaluator
TOK = re.compile(r"\s*(0x[0-9A-Fa-f]+|-?\d+\.\d*f?|-?\d+f|-?\d+|\"(?:[^\"\\]|\\.)*\"|[A-Za-z_][\w.]*|[(),=])")


def tokens(s):
    out, i = [], 0
    s = s.strip()
    while i < len(s):
        m = TOK.match(s, i)
        if not m:
            raise ValueError(f"tok at {s[i:]!r}")
        out.append(m.group(1))
        i = m.end()
        while i < len(s) and s[i].isspace():
            i += 1
    return out


class P:
    def __init__(self, s):
        self.t = tokens(s)
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self, x=None):
        v = self.t[self.i]
        if x is not None and v != x:
            raise ValueError(f"expected {x} got {v}")
        self.i += 1
        return v

    def expr(self):
        v = self.term()
        while self.peek() in ("to", "qto"):
            op = self.take()
            w = self.term()
            v = ("pair", v, w) if op == "to" else ("quad", v[1], v[2], w[1], w[2])
        return v

    def args(self):
        self.take("(")
        out = []
        while self.peek() != ")":
            if self.i + 1 < len(self.t) and self.t[self.i + 1] == "=" and re.match(r"[a-z]\w*$", self.peek()):
                name = self.take()
                self.take("=")
                out.append((name, self.expr()))
            else:
                out.append((None, self.expr()))
            if self.peek() == ",":
                self.take()
        self.take(")")
        return out

    def term(self):
        t = self.take()
        if t == "(":
            v = self.expr()
            self.take(")")
            return v
        if re.match(r"0x", t):
            v = ("int", int(t, 16))
        elif re.match(r"-?\d", t):
            v = ("float", float(t.rstrip("f"))) if ("." in t or t.endswith("f")) else ("int", int(t))
        elif t.startswith('"'):
            v = ("str", json.loads(t))
        elif t in ("true", "false"):
            v = ("bool", t == "true")
        else:
            name = t
            if self.peek() == "(":
                v = ("call", name, self.args())
            else:
                v = ("ref", name)
        # postfix .member / .member(args) handled by dotted identifiers in tokens ("x.toModel")
        return v


def color_of(v):
    """evaluate colour expressions -> signed ARGB int"""
    k = v[0]
    if k == "ref":
        parts = v[1].split(".")
        if parts[0] == "Color":
            return sint(COLORS[parts[1]])
    if k == "call":
        name, args = v[1], v[2]
        parts = name.split(".")
        if parts[0] == "Color" and len(parts) == 1:
            if len(args) == 1 and args[0][1][0] == "int":
                return sint(args[0][1][1])
            d = {n: a[1] for n, a in args}
            r, g, b = d["red"], d["green"], d["blue"]
            al = d.get("alpha", 1.0)
            c = [int(x * 255 + 0.5) for x in (al, r, g, b)]
            return sint((c[0] << 24) | (c[1] << 16) | (c[2] << 8) | c[3])
        if parts[0] == "Color" and parts[1:2] == ["copy"]:
            pass
    raise ValueError(f"colour {v}")


def ev(v):
    """Evaluate to python: ('F',x) etc. Returns a nested structure."""
    k = v[0]
    if k in ("float",):
        return v[1]
    if k == "int":
        return v[1]
    if k == "bool":
        return v[1]
    if k == "str":
        return v[1]
    if k == "pair":
        return ("Pair", ev(v[1]), ev(v[2]))
    if k == "quad":
        return ("Quad", ev(v[1]), ev(v[2]), ev(v[3]), ev(v[4]))
    if k == "ref":
        name = v[1]
        parts = name.split(".")
        if parts[0] == "Color":
            col = sint(COLORS[parts[1]])
            return ("Color", col)
        if parts[0] in ENUM:
            return ("Enum", parts[0], parts[1])
        if parts[0] == "GradientPalette":
            return ("GP", parts[1])
        if name.endswith(".Default"):
            return ("Params", parts[0])
        if name == "Unit":
            return ("Unit",)
        raise ValueError(f"ref {name}")
    if k == "call":
        name, args = v[1], v[2]
        if name in ("Pair",):
            return ("Pair", ev(args[0][1]), ev(args[1][1]))
        if name == "Triple":
            return ("Triple",) + tuple(ev(a[1]) for a in args)
        if name == "Quad":
            return ("Quad",) + tuple(ev(a[1]) for a in args)
        if name == "floatArrayOf":
            return ("FA", [float(ev(a[1])) for a in args])
        if name == "listOf":
            return ("List", [ev(a[1]) for a in args])
        if name == "GmicFilterParams":
            return ("Gmic", ev(args[0][1])[1])
        if name in ("FileModel", "ImageModel"):
            return ("File",)
        if name.endswith("Params") or name.endswith("Params.Relative"):
            return ("Params", name.split(".")[0])
        if name.startswith("SideFadeParams.Relative"):
            return ("Params", "SideFadeParams")
        # colour calls with postfix .toModel() etc. come as dotted names: "Color.Yellow.copy"(0.3f).toModel().wrap()
        if name.startswith("Color"):
            base = name.split(".")
            if base[0] == "Color" and len(base) == 1:
                return ("Color", color_of(v))
            if base[:1] == ["Color"] and len(base) >= 3 and base[2] == "copy":
                col = COLORS[base[1]] & 0x00FFFFFF
                a = args[0][1][1]
                return ("Color", sint(col | (int(a * 255 + 0.5) << 24)))
        raise ValueError(f"call {name}")
    raise ValueError(f"?? {v}")


def ev_str(s):
    # normalise ".toModel()" ".wrap()" ".toArgb()" ".toColorModel()" postfixes
    s2 = re.sub(r"\)\s*\.(toModel|wrap|toArgb|toColorModel)\(\)", ")", s)
    s2 = re.sub(r"\.(toModel|wrap|toArgb|toColorModel)\(\)", "", s2)
    p = P(s2)
    v = p.expr()
    if p.i != len(p.t):
        # trailing tokens like (0.3f).toModel... -> handle "Color.Yellow.copy(0.3f)"
        raise ValueError(f"trailing {p.t[p.i:]} in {s}")
    return ev(v)


# ---------------------------------------------------------------- gmic param infos
def gmic_infos(src):
    m = re.search(r"gmicParamsInfo\s*=\s*listOf\((.*)\)\s*\)\s*,\s*Filter\.", src, re.S)
    if not m:
        m = re.search(r"gmicParamsInfo\s*=\s*listOf\((.*?)\)\s*\)\s*(,|\{|$)", src, re.S)
    body = m.group(1)
    infos = []
    for im in re.finditer(r"GmicParameterInfo\.(Number|Toggle|Selection|Color)\((.*?)\)(?=\s*,\s*GmicParameterInfo|\s*$)",
                          body, re.S):
        kind, inner = im.group(1), im.group(2)
        if kind == "Number":
            rng = re.search(r"(-?[\d.]+)f\s*\.\.\s*(-?[\d.]+)f", inner)
            isint = "isInteger = true" in inner
            infos.append({"k": "num_i" if isint else "num_f", "min": float(rng.group(1)), "max": float(rng.group(2))})
        elif kind == "Toggle":
            infos.append({"k": "toggle"})
        elif kind == "Color":
            infos.append({"k": "color"})
        else:
            ents = re.findall(r"\"((?:[^\"\\]|\\.)*)\"", inner)
            infos.append({"k": "sel", "entries": ents})
    return infos


def snake(n):
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", n)
    return s.lower()


# ---------------------------------------------------------------- leaves
def split_generic(t):
    inner = t[t.index("<") + 1:-1]
    parts, depth, cur = [], 0, ""
    for ch in inner:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return t[:t.index("<")], parts


def kind_of(t):
    return {"Float": "F", "Int": "I", "Boolean": "B", "ColorModel": "C", "String": "S"}.get(
        t, f"E:{t}" if t in ENUM else ("FILE" if t in ("ImageModel", "FileModel") else f"?{t}"))


def to_default(kind, v):
    if kind == "C":
        return v[1] if isinstance(v, tuple) else v
    if kind.startswith("E:"):
        return v[2] if isinstance(v, tuple) else v
    if kind == "FILE":
        return None
    return v


def leaves_for(cls, rec):
    """-> (typename, [ (leafname, kind, default) ], extra)"""
    t = rec["type"]
    d = rec.get("default")
    if t == "Unit":
        return "Unit", [], {}
    if t in ("Float", "Int", "Boolean", "String"):
        return t, [("value", kind_of(t), ev_str(d))], {}
    if t == "FloatArray":
        return "FloatArray", [("values", "FA", ev_str(d)[1])], {}
    if t == "FilterValueWrapper<ColorModel>":
        return "FilterValueWrapper{ColorModel}", [("color", "C", ev_str(d)[1])], {}
    if t == "GradientPalette":
        return "GradientPalette", [("palette", "GP", ev_str(d)[1])], {}
    if t == "PolarCoordinatesType":
        return "PolarCoordinatesType", [("type", "E:PolarCoordinatesType", ev_str(d)[2])], {}
    if t.startswith(("Pair<", "Triple<", "Quad<")):
        base, parts = split_generic(t)
        val = ev_str(d)
        vals = list(val[1:])
        names = ["first", "second", "third", "fourth"]
        lv = [(names[i], kind_of(p), to_default(kind_of(p), vals[i])) for i, p in enumerate(parts)]
        tn = f"{base}{{{'$'.join(parts)}}}"
        return tn, lv, {}
    if t in PARAMS:
        return t, [(n, k, dv) for n, k, dv in PARAMS[t]], {}
    if t == "ToneCurvesParams":
        return t, [("controlPoints", "TC", None)], {}
    if t == "GmicFilterParams":
        src = (MODEL / rec["file"]).read_text()
        vals = ev_str(d)[1]
        infos = gmic_infos(src)
        assert len(infos) == len(vals), (cls, len(infos), len(vals))
        return t, [(f"g{i}", "G:" + inf["k"], v) for i, (inf, v) in enumerate(zip(infos, vals))], {"gmic": infos}
    if t == "ProceduralParams":
        src = (MODEL / rec["file"]).read_text()
        eff = re.search(r"ProceduralEffect\.(\w+)", src).group(1)
        pr = PROC[eff]
        lv = [(n, "PI" if integer else "PF", dv) for n, dv, integer in pr["params"]]
        lv += [(n, "PC", dv) for n, dv in pr["colors"]]
        return t, lv, {"effect": eff}
    if t in NOT_SERIALISABLE:
        return t, None, {}
    raise ValueError(f"unhandled {cls} {t}")


# ---------------------------------------------------------------- matching
OPKIND_OK = {
    "F": ("float", "int"), "I": ("int", "float"), "IF": ("int", "float"), "B": ("bool",), "C": ("color",),
    "CF": ("color",), "S": ("text",), "FA": ("float_list",), "GP": ("choice", "gradient", "text"),
    "PF": ("float", "int"), "PI": ("int", "float"), "PC": ("color",), "FP": ("point",), "FP2": ("point",), "TC": ("points",),
    "FONT": ("text", "choice"), "FILE": ("file",),
}


def compatible(kind, opkind):
    if kind.startswith("E:"):
        return opkind in ("choice", "int")
    if kind.startswith("G:"):
        k = kind[2:]
        return {"num_i": ("int", "float"), "num_f": ("float", "int"), "toggle": ("bool",), "color": ("color",),
                "sel": ("choice",)}[k].__contains__(opkind)
    return opkind in OPKIND_OK.get(kind, ())


def norm_name(n):
    return re.sub(r"[^a-z0-9]", "", n.lower())


def argb_to_hex(c):
    c &= 0xFFFFFFFF
    a, r, g, b = c >> 24, (c >> 16) & 255, (c >> 8) & 255, c & 255
    return f"#{r:02X}{g:02X}{b:02X}" + ("" if a == 255 else f"{a:02X}")


def eq_default(kind, itb, opd, op):
    try:
        if kind in ("C", "CF", "PC") or kind == "G:color":
            return argb_to_hex(int(itb)).upper() == str(opd).upper()
        if kind.startswith("E:") or kind == "G:sel":
            return True  # checked via enum mapping
        if kind in ("B", "G:toggle"):
            return (itb in (True, "true")) == bool(opd)
        if kind == "FA":
            return [float(x) for x in itb] == [float(x) for x in opd]
        if kind in ("F", "I", "IF", "PF", "PI") or kind.startswith("G:num"):
            return abs(float(itb) - float(opd)) < 1e-6
        if kind in ("FP", "FP2"):
            return abs(itb[0] - opd[0]) < 1e-6 and abs(itb[1] - opd[1]) < 1e-6
    except Exception:
        return False
    return True


def enum_map(enum_entries, choices):
    """ITB enum entry names -> op choice strings (by normalised name, else by order)."""
    res = {}
    nchoices = {norm_name(c): c for c in choices}
    for e in enum_entries:
        c = nchoices.get(norm_name(e))
        if c is None:
            return None
        res[e] = c
    return res


OVERRIDES = {
    "Ascii": {"gradient": "gradient", "fontSize": "font_size", "backgroundColor": "background_color",
              "isGrayscale": "gray_scale", "font": "font"},
    "DropShadow": {"color": "shadow_color", "offsetX": "offset_x", "offsetY": "offset_y", "blurRadius": "blur_radius"},
    "SideFade": {"side": "side", "scale": "strength"},
    "ToneCurves": {"controlPoints": "rgb"}, "HueShiftBands": {"hueOffset": "hue"}, "HueShiftFan": {"hueOffset": "hue"},
    "HueShiftTunnel": {"hueOffset": "hue"},
}


def match(lv, params, named, ovr, extra):
    mapping, notes = [], []
    used = set()
    byname = {p["name"]: p for p in params}
    for i, (ln, kind, dv) in enumerate(lv):
        target = None
        if ln in ovr:
            target = byname[ovr[ln]]
        elif named:
            nl = norm_name(ln.replace("centre", "center"))
            cands = [p for p in params if norm_name(p["name"]) == nl and p["name"] not in used]
            if not cands and kind == "PC":
                cands = [p for p in params if p["kind"] == "color" and p["name"] not in used]
            target = cands[0] if cands else None
        elif i < len(params):
            target = params[i]
        if target is None:
            mapping.append(None)
            notes.append(f"leaf {ln}({kind}) unmapped")
            continue
        used.add(target["name"])
        m = {"leaf": ln, "kind": kind, "param": target["name"], "pkind": target["kind"]}
        ok = compatible(kind, target["kind"])
        if kind.startswith("E:"):
            en = kind[2:]
            if target["kind"] == "choice":
                em = enum_map(ENUM[en], target.get("choices", []))
                if em is None and len(ENUM[en]) == len(target.get("choices", [])):
                    em = dict(zip(ENUM[en], target["choices"]))
                    notes.append(f"enum {en} mapped by order {ENUM[en]} -> {target['choices']}")
                if em is None:
                    notes.append(f"enum {en}->{target['name']} needs manual map {ENUM[en]} vs {target.get('choices')}")
                m["enum"] = em
            else:
                m["enum"] = {e: i for i, e in enumerate(ENUM[en])}
        elif kind in ("I", "F", "PI") and target["kind"] == "choice":
            ok = True
            m["index_choice"] = list(target["choices"])
            if not (isinstance(dv, (int, float)) and 0 <= int(dv) < len(target["choices"])
                    and target["choices"][int(dv)] == target["default"]):
                notes.append(f"index-choice default {ln}={dv} vs {target['default']}")
        if kind == "G:sel":
            ents = extra["gmic"][i]["entries"]
            if target["kind"] == "choice":
                em = enum_map(ents, target.get("choices", []))
                if em is None and len(ents) == len(target.get("choices", [])):
                    em = dict(zip(ents, target["choices"]))
                    notes.append(f"gmic sel {ents} mapped by order to {target['choices']}")
                if em is None:
                    notes.append(f"gmic sel {ents} vs {target.get('choices')}")
                m["enum"] = em
        if not ok:
            notes.append(f"kind {ln}:{kind} -> {target['name']}:{target['kind']}")
        if "index_choice" not in m and not eq_default(kind, dv, target["default"], target):
            notes.append(f"default {ln}={dv!r} vs {target['name']}={target['default']!r}")
        mapping.append(m)
    return mapping, notes




def build_table(ui: dict, ops: dict) -> tuple[dict, list[str]]:
    table, report = {}, []
    for cls, rec in sorted(ui.items()):
        if cls == "UiProceduralFilter":
            continue
        name = cls[2:-6]
        if rec.get("colormap"):
            choice = {"CoolVariant": "cool"}.get(name, snake(name))
            table[name] = {"cls": cls, "op": "filter.color_map", "type": "Unit", "fixed": {"color_map": choice}}
            continue
        op = ops.get({"SimpleWeavePixelization": "SimpleWeavePixelation"}.get(name, name))
        if op is None:
            report.append(f"NOOP {name}")
            continue
        try:
            tn, lv, extra = leaves_for(cls, rec)
        except Exception as e:  # noqa: BLE001
            report.append(f"ERR {name}: {e}")
            continue
        entry = {"cls": cls, "op": op["id"], "type": tn, **extra}
        if lv is None:
            entry["unsupported"] = f"{tn} is not serialisable by ImageToolbox"
            table[name] = entry
            continue
        params = op["params"]
        named = rec["type"] in ("ProceduralParams", "NtscParams")
        mapping, notes = match(lv, params, named, OVERRIDES.get(name, {}), extra)
        extra_params = [p["name"] for p in params if p["name"] not in {m["param"] for m in mapping if m}]
        if extra_params:
            notes.append(f"op params not covered: {extra_params}")
        entry["map"] = mapping
        if notes:
            report.append(f"{name} [{tn}] -> {op['id']}: " + "; ".join(notes))
        table[name] = entry
    return table, report


def palettes():
    src = PALETTE_KT.read_text()
    out = {}
    for m in re.finditer(r"data object (\w+)\s*:\s*GradientPalette\(\s*\"([^\"]+)\",(.*?)\n    \)", src, re.S):
        cols = [sint(int(h, 16)) for h in re.findall(r"0x([0-9A-Fa-f]{8})", m.group(3))]
        out[m.group(2)] = cols
    return out


def emit(table: dict) -> None:
    rows = {}
    for name, e in sorted(table.items()):
        row = {"cls": e["cls"], "op": e["op"], "t": e["type"]}
        if e.get("fixed"):
            row["fixed"] = e["fixed"]
        if e.get("unsupported"):
            row["unsupported"] = e["unsupported"]
        if e.get("effect"):
            row["effect"] = e["effect"]
        if "map" in e:
            leaves = []
            for m in e["map"]:
                if m is None:
                    leaves.append(None)
                    continue
                leaf = [m["kind"], m["param"]]
                if m["kind"].startswith(("P", )) and e["type"] == "ProceduralParams":
                    leaf.append(m["leaf"])
                elif m.get("enum") is not None:
                    leaf.append(m["enum"])
                elif m.get("index_choice") is not None:
                    leaf.append({"index": m["index_choice"]})
                elif e["type"] == "NtscParams":
                    pass
                leaves.append(leaf)
            row["m"] = leaves
        if e["type"] == "NtscParams":
            row["fields"] = [n for n, _, _ in PARAMS["NtscParams"]]
        if e["type"] == "GmicFilterParams":
            row["gmic"] = [inf["k"] for inf in e["gmic"]]
        rows[name] = row
    pal = palettes()
    fonts = ["System", "Montserrat", "Caveat", "Comfortaa", "Handjet", "YsabeauSC", "Jura", "Podkova", "Tektur",
             "DejaVu", "BadScript", "RuslanDisplay", "Catterdale", "FRM32", "TokeelyBrookings", "Nunito", "Nothing",
             "WOPRTweaked", "AlegreyaSans", "MinecraftGnu", "GraniteFixed", "NokiaPixel", "Ztivalia", "Axotrel",
             "LcdOctagon", "LcdMoving", "Unisource"]
    body = ['"""GENERATED (dev-time) from the ImageToolbox sources: Ui*Filter value schemas -> LazyRoom op params.',
            "",
            "ImageToolbox (https://github.com/T8RIN/ImageToolbox) by T8RIN, Apache-2.0.  Each entry maps one",
            "ImageToolbox filter class to its LazyRoom ``filter.*`` operation: ``t`` is the serialised value type",
            "name, ``m`` lists the value components (in ImageToolbox serialisation order) as [kind, op param, extra].",
            "Regenerate with ``python -m lazyroom.tools._itb_template_gen <ImageToolbox checkout>``; do not edit.",
            '"""',
            "# flake8: noqa",
            "# fmt: off",
            "FILTERS = {"]
    for name, row in rows.items():
        body.append(f"    {name!r}: {row!r},")
    body.append("}")
    body.append("")
    body.append(f"PALETTES = {pal!r}")
    body.append("")
    body.append(f"FONTS = {fonts!r}")
    body.append("")
    out = pathlib.Path(__file__).with_name("_itb_template_table.py")
    out.write_text("\n".join(body))
    print("wrote", out, len(rows), "filters,", len(pal), "palettes")


def main(argv: list[str]) -> int:
    configure(argv[1] if len(argv) > 1 else ITB)
    load_enums()
    load_procedural()
    table, report = build_table(extract_ui(), dump_ops())
    emit(table)
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
