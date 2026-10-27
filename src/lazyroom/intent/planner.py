"""Natural-language intent -> structured, editable edit actions.

Two engines, same output contract:

* ``rules`` (default, offline, deterministic): a transparent phrase -> parameter
  vocabulary with intensity modifiers ("slightly", "very"), base looks for
  named aesthetics ("1970s film", "polaroid", "teal and orange"), image-aware
  scaling (e.g. less brightening for an already bright photo) and skin
  protection ("without changing skin tones" adds an editable skin mask).
* ``llm`` (optional): a configured OpenAI-compatible or Anthropic endpoint is
  asked for JSON actions against the operation schemas; the result is
  validated and falls back to ``rules`` on any error.

External agents usually skip this module and author steps themselves — the
planner exists for the GUI's AI panel and as a starting point.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from ..core.errors import LazyRoomError
from ..core.pipeline import Step

SKIN_MASK = {"shapes": [{"type": "all"}, {"type": "skin", "mode": "subtract", "opacity": 0.85}]}

LOOK_RULES: list[tuple[str, str]] = [
    (r"\b(19)?70'?s\b|seventies|retro film|nostalgic", "warm-70s-film"),
    (r"polaroid|instant (film|camera)", "vintage-polaroid"),
    (r"teal (and|&) orange|blockbuster|cinematic", "teal-orange"),
    (r"(19)?90'?s|nineties", "faded-90s-magazine"),
    (r"faded (analog|film)|analog film|film (look|stock)|portra|kodak", "faded-analog"),
    (r"editorial|magazine portrait|flattering portrait", "warm-editorial-portrait"),
    (r"urban|street look|gritty|cool,? desaturated", "cool-urban"),
    (r"earthy|instagram|boho|beige", "earthy-muted"),
    (r"black (and|&) white|b&w|monochrome|grayscale|greyscale|noir", "bw-classic"),
    (r"moody|dramatic dark|low[- ]key", "moody-dark"),
    (r"bright (and|&) airy|airy|pastel", "bright-airy"),
    (r"cross[- ]?process", "cross-process"),
]

BANDS = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
BAND_ALIASES = {"reds": "red", "oranges": "orange", "yellows": "yellow", "greens": "green", "foliage": "green",
                "grass": "green", "leaves": "green", "aquas": "aqua", "cyan": "aqua", "cyans": "aqua", "blues": "blue",
                "sky": "blue", "skies": "blue", "water": "aqua", "purples": "purple", "magentas": "magenta",
                "skin": "orange", "skin tones": "orange", "sand": "yellow"}

# (pattern, [(op, param, delta)], description)
ATTR_RULES: list[tuple[str, list[tuple[str, str, float]], str]] = [
    (r"brown(ish)?\b(?! shadows)|sepia[- ]?ish|tobacco", [("adjust.white_balance", "temperature", 8),
                                                         ("adjust.color_grading", "shadows_saturation", 15)],
     "brownish tones"),
    (r"\bwarm(er|th)?\b|golden|sunset tones?", [("adjust.white_balance", "temperature", 18)], "warmer white balance"),
    (r"\b(cool(er)?|cold(er)?)\b(?! (shadows|highlights))", [("adjust.white_balance", "temperature", -18)], "cooler white balance"),
    (r"magenta tint|more magenta", [("adjust.white_balance", "tint", 15)], "magenta tint"),
    (r"green(ish)? tint|more green tint", [("adjust.white_balance", "tint", -15)], "green tint"),
    (r"bright(er|en)|lighter|lighten|more light|underexposed", [("adjust.exposure", "ev", 0.4)], "brighter exposure"),
    (r"dark(er|en)\b(?! (corners|edges))|overexposed|dimmer", [("adjust.exposure", "ev", -0.4)], "darker exposure"),
    (r"(more|high|strong|punchy|bold) contrast|contrasty|punchy|\bpop\b", [("adjust.contrast", "amount", 25)], "more contrast"),
    (r"(soft|low|less|gentle|reduced) contrast|flat(ter)?\b", [("adjust.contrast", "amount", -22)], "softer contrast"),
    (r"faded blacks?|lifted blacks?|matte|washed[- ]out|\bfade[d]?\b", [("adjust.fade", "amount", 32)], "faded / lifted blacks"),
    (r"soft highlights?|highlight roll[- ]?off|creamy highlights?", [("adjust.fade", "highlight_rolloff", 30),
                                                                     ("adjust.tone", "highlights", -20)], "soft highlights"),
    (r"recover (the )?highlights?|blown (out )?highlights?|tame highlights?", [("adjust.tone", "highlights", -50)],
     "recover highlights"),
    (r"(open|lift|brighten) (up )?(the )?shadows?|shadow detail", [("adjust.tone", "shadows", 40)], "lift shadows"),
    (r"deep(er)? (blacks|shadows)|crush(ed)? (the )?blacks", [("adjust.tone", "blacks", -30)], "deeper blacks"),
    (r"vibrant|vivid|colou?rful|saturated|more colou?r|richer colou?rs?", [("adjust.saturation", "vibrance", 25)],
     "more vibrant colours"),
    (r"muted( colou?rs)?\b(?! (green|red|blue|orange|yellow|aqua|purple|magenta))|desaturat\w*\b(?! (green|red|blue))|less saturat\w*|subdued",
     [("adjust.saturation", "saturation", -28)], "muted colours"),
    (r"film grain|\bgrain(y)?\b|\bnoise\b|analog texture", [("effect.grain", "amount", 20)], "film grain"),
    (r"vignett\w*|dark(er|en)? (corners|edges)", [("effect.vignette", "amount", -25)], "vignette"),
    (r"dream(y|like)|ethereal|soft glow|hazy glow|pro[- ]?mist|soft focus|diffus\w+", [("effect.diffusion", "amount", 25)],
     "dreamy diffusion glow"),
    (r"\bbloom\b|glowing highlights?", [("effect.bloom", "intensity", 40)], "bloom"),
    (r"halation|cinestill", [("effect.halation", "amount", 35)], "halation"),
    (r"sharp(er|en)|crisp(er)?|more detail", [("effect.sharpen", "amount", 60)], "sharpening"),
    (r"clarity|local contrast|structure", [("adjust.clarity", "clarity", 30)], "clarity"),
    (r"dehaze|remove (the )?haze|clear (the )?fog", [("adjust.dehaze", "amount", 35)], "dehaze"),
    (r"(add|more) haze|foggy|misty", [("adjust.dehaze", "amount", -25)], "add haze"),
    (r"light leak", [("effect.light_leak", "amount", 35)], "light leak"),
    (r"sepia", [("adjust.black_white", "tint_saturation", 35)], "sepia toning"),
    (r"denoise|noise reduction|reduce noise|clean up noise", [("effect.denoise", "luminance", 35)], "noise reduction"),
    (r"auto ?(fix|correct)|fix (the )?exposure|auto tone", [("adjust.auto_tone", "strength", 1.0)], "auto tone"),
    (r"fix (the )?white balance|neutrali[sz]e (the )?colou?r cast|remove (the )?cast", [("adjust.auto_white_balance", "strength", 1.0)],
     "auto white balance"),
]

ZONE_RULES: list[tuple[str, str, float, float, str]] = [
    # pattern, zone, hue, saturation, description
    (r"(warm(er)?|golden|amber) shadows|shadows (warmer|warm)", "shadows", 35, 25, "warm shadows"),
    (r"brown(ish)? shadows|shadows brown", "shadows", 28, 32, "brown shadows"),
    (r"(cool(er)?|blue|teal|cyan) shadows|shadows (cooler|cool|blue)", "shadows", 200, 28, "cool shadows"),
    (r"(warm(er)?|golden) highlights|highlights (warmer|warm)", "highlights", 45, 22, "warm highlights"),
    (r"(cool(er)?|blue) highlights|highlights (cooler|cool)", "highlights", 205, 18, "cool highlights"),
    (r"(magenta|pink|purple) shadows", "shadows", 300, 22, "magenta shadows"),
    (r"(green(ish)?) shadows", "shadows", 120, 18, "green shadows"),
]

INTENSITY = [(r"\b(very|really|extremely|heavy|heavily|strong(ly)?|much|lots of|a lot)\b", 1.6),
             (r"\b(slight(ly)?|subtle|subtly|a (little|bit|touch)|gentle|gently|mild(ly)?|hint of|touch of)\b", 0.5)]

CLAMPS = {("adjust.exposure", "ev"): (-2, 2), ("effect.grain", "amount"): (0, 100), ("effect.vignette", "amount"): (-100, 100),
          ("effect.diffusion", "amount"): (0, 100), ("effect.bloom", "intensity"): (0, 200),
          ("effect.sharpen", "amount"): (0, 500), ("adjust.fade", "amount"): (0, 100),
          ("adjust.fade", "highlight_rolloff"): (0, 100), ("adjust.auto_tone", "strength"): (0, 1),
          ("adjust.auto_white_balance", "strength"): (0, 1), ("adjust.black_white", "tint_saturation"): (0, 100),
          ("effect.denoise", "luminance"): (0, 100), ("effect.halation", "amount"): (0, 100),
          ("effect.light_leak", "amount"): (0, 100)}

COLOR_OPS = {"adjust.white_balance", "adjust.color_grading", "adjust.saturation", "adjust.hsl", "adjust.split_toning"}


def _intensity(text: str, start: int) -> float:
    window = re.split(r"[,;.]| and | with | but ", text[max(0, start - 28): start])[-1]
    f = 1.0
    for pat, mult in INTENSITY:
        if re.search(pat, window):
            f = mult
    return f


def plan_rules(request: str, analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    text = " " + request.lower().replace("’", "'") + " "
    interp: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    notes: list[str] = []
    look_id = None
    look_span = (0, 0)
    for pat, lid in LOOK_RULES:
        if re.search(pat, text):
            look_id = lid
            lm = re.search(pat, text)
            interp.append({"phrase": lm.group(0).strip(), "effect": f"base look '{lid}'"})
            look_span = lm.span()
            break
    params: dict[str, dict[str, float]] = {}

    def add(op: str, key: str, val: float) -> None:
        params.setdefault(op, {})
        params[op][key] = params[op].get(key, 0.0) + val

    consumed: list[tuple[int, int]] = [look_span] if look_id else []
    for pat, zone, hue, sat, desc in ZONE_RULES:
        for m in re.finditer(pat, text):
            f = _intensity(text, m.start())
            params.setdefault("adjust.color_grading", {})[f"{zone}_hue"] = float(hue)  # last phrase wins
            add("adjust.color_grading", f"{zone}_saturation", sat * f)
            interp.append({"phrase": m.group(0).strip(), "effect": desc, "intensity": f})
            consumed.append(m.span())
    for m in re.finditer(r"(?:muted|desaturated?|less saturated|dull(?:er)?|subdued|softer|duller|less) "
                         r"(?P<band>reds|oranges|yellows|greens|aquas|cyans?|blues|purples|magentas|red|orange|yellow|green|"
                         r"aqua|blue|purple|magenta|foliage|grass|leaves|sky|skies|water)", text):
        band = BAND_ALIASES.get(m.group("band"), m.group("band").rstrip("s"))
        if band in BANDS:
            f = _intensity(text, m.start())
            add("adjust.hsl", f"{band}_saturation", -40 * f)
            if band == "green":
                add("adjust.hsl", "green_hue", -12 * f)
            interp.append({"phrase": m.group(0).strip(), "effect": f"less saturated {band}s", "intensity": f})
            consumed.append(m.span())
    for m in re.finditer(r"(?:vivid|more saturated|richer|stronger|bolder|more|deeper|bluer|greener) "
                         r"(?P<band>reds|oranges|yellows|greens|blues|purples|magentas|sky|skies|foliage|water)", text):
        band = BAND_ALIASES.get(m.group("band"), m.group("band").rstrip("s"))
        if band in BANDS:
            f = _intensity(text, m.start())
            add("adjust.hsl", f"{band}_saturation", 30 * f)
            if band == "blue" and m.group("band") in ("sky", "skies"):
                add("adjust.hsl", "blue_luminance", -12 * f)
            interp.append({"phrase": m.group(0).strip(), "effect": f"more saturated {band}s", "intensity": f})
            consumed.append(m.span())

    def overlaps(span: tuple[int, int]) -> bool:
        return any(not (span[1] <= a or span[0] >= b) for a, b in consumed)

    for pat, deltas, desc in ATTR_RULES:
        for m in re.finditer(pat, text):
            if overlaps(m.span()):
                continue
            f = _intensity(text, m.start())
            for op, key, val in deltas:
                add(op, key, val * f if op not in ("adjust.auto_tone", "adjust.auto_white_balance") else val)
            if desc == "sepia toning":
                add("adjust.black_white", "tint_hue", 35)
            interp.append({"phrase": m.group(0).strip(), "effect": desc, "intensity": f})
            consumed.append(m.span())
            break  # count each attribute once

    cg = params.get("adjust.color_grading")
    if cg and "shadows_saturation" in cg and "shadows_hue" not in cg:
        cg["shadows_hue"] = 28.0

    # image-aware tempering ------------------------------------------------
    if analysis:
        med = analysis.get("tone", {}).get("median", 0.5)
        ev = params.get("adjust.exposure", {}).get("ev")
        if ev is not None:
            if ev > 0 and med > 0.7:
                params["adjust.exposure"]["ev"] = ev * 0.5
                notes.append("Image is already bright; brightening was reduced.")
            if ev < 0 and med < 0.35:
                params["adjust.exposure"]["ev"] = ev * 0.5
                notes.append("Image is already dark; darkening was reduced.")
        warm = analysis.get("color", {}).get("warmth", 0)
        t = params.get("adjust.white_balance", {}).get("temperature")
        if t is not None and t > 0 and warm > 0.03:
            params["adjust.white_balance"]["temperature"] = t * 0.6
            notes.append("Image already has a warm cast; warming was reduced.")
        if t is not None and t < 0 and warm < -0.03:
            params["adjust.white_balance"]["temperature"] = t * 0.6
            notes.append("Image already has a cool cast; cooling was reduced.")

    protect_skin = bool(re.search(r"(without|don'?t|not|keep|preserve|protect)\w* .{0,25}skin", text))
    if protect_skin:
        interp.append({"phrase": "skin tones", "effect": "colour changes masked away from skin (editable mask)"})

    # assemble ---------------------------------------------------------------
    if look_id:
        actions.append({"type": "add_group", "label": f"Look: {look_id}", "origin": "intent",
                        "steps": [], "_look": look_id})
    for op, vals in params.items():
        p = {}
        for k, v in vals.items():
            lo, hi = CLAMPS.get((op, k), (0, 360) if (op == "adjust.color_grading" and k.endswith("_hue"))
                                else (-100, 100))
            p[k] = round(max(lo, min(hi, v)), 3)
        if op == "adjust.black_white" and "tint_hue" not in p and "tint_saturation" in p:
            p["tint_hue"] = 35
        step: dict[str, Any] = {"type": "add", "op": op, "params": p, "origin": "intent",
                                "label": _label(op, p)}
        if protect_skin and op in COLOR_OPS:
            step["mask"] = deepcopy(SKIN_MASK)
        actions.append(step)
    unmatched = _unmatched(text, consumed, look_id)
    if not actions:
        notes.append("No known adjustment phrases were recognised; try words like warmer, faded, muted greens, "
                     "grain, vignette, soft contrast, or name a look (70s film, polaroid, teal and orange).")
    if re.search(r"\b(match|copy|same) .{0,40}(reference|grading|look|style) ", text) or "reference" in text:
        notes.append("To match a reference photo use the style_match command (project_id, reference=<path>).")
    if re.search(r"every (image|photo)|whole folder|all (the )?(images|photos)|batch", text):
        notes.append("To apply this to many images, save the result as a preset and use batch_start.")
    return {"engine": "rules", "actions": actions, "interpretation": interp, "notes": notes,
            "unrecognized": unmatched, "base_look": look_id}


def _label(op: str, p: dict[str, Any]) -> str:
    short = op.split(".", 1)[1].replace("_", " ")
    vals = ", ".join(f"{k.replace('_', ' ')} {v:+g}" if isinstance(v, (int, float)) else k for k, v in list(p.items())[:3])
    return f"{short.capitalize()}: {vals}"


def _unmatched(text: str, consumed: list[tuple[int, int]], look_id: str | None) -> list[str]:
    stop = set("a an the this that it its to of and or with make give look feel feeling my me please some more less "
               "i want like photo photograph image picture into in on for slightly very bit little but without keep "
               "apply add also so be is are as".split())
    masked = list(text)
    for a, b in consumed:
        for i in range(a, b):
            masked[i] = " "
    words = re.findall(r"[a-z][a-z'-]+", "".join(masked))
    return sorted({w for w in words if w not in stop and len(w) > 2})[:30]


def plan(request: str, *, analysis: dict[str, Any] | None = None, config: Any = None,
         use_llm: bool | None = None) -> dict[str, Any]:
    if not request or not request.strip():
        from ..core.errors import invalid

        raise invalid("Empty request")
    llm_cfg = getattr(config, "llm", None)
    want_llm = use_llm if use_llm is not None else bool(llm_cfg and llm_cfg.provider not in ("", "none"))
    if want_llm and llm_cfg and llm_cfg.provider not in ("", "none"):
        from .llm import plan_with_llm

        try:
            res = plan_with_llm(request, analysis=analysis, cfg=llm_cfg)
            _validate(res["actions"])
            return res
        except Exception as exc:  # fall back to rules, but say so
            res = _finalize(plan_rules(request, analysis))
            res["notes"].insert(0, f"LLM planning failed ({type(exc).__name__}: {exc}); used the rule engine instead.")
            return res
    return _finalize(plan_rules(request, analysis))


def _finalize(res: dict[str, Any]) -> dict[str, Any]:
    """Expand a base look into its group; fold attribute tweaks into matching look steps."""
    from ..core.registry import get_op

    actions = res["actions"]
    group = next((a for a in actions if a.get("_look")), None)
    if group is not None:
        from ..presets.store import PresetStore

        look = PresetStore().get(group["_look"])
        look_steps = [{**deepcopy(s), "origin": "intent"} for s in look["steps"]]
        rest = []
        merged = []
        for a in actions:
            if a is group:
                continue
            tgt = next((s for s in look_steps if s["op"] == a.get("op") and not a.get("mask")), None)
            if tgt is None:
                rest.append(a)
                continue
            spec = get_op(tgt["op"])
            ranges = {p.name: (getattr(p, "min", None), getattr(p, "max", None)) for p in spec.params}
            for k, v in a["params"].items():
                is_angle = k.endswith("_hue") and tgt["op"] in ("adjust.color_grading", "adjust.split_toning")
                if is_angle or not isinstance(v, (int, float)):
                    tgt.setdefault("params", {})[k] = v
                    continue
                lo, hi = ranges.get(k, (None, None))
                cur = tgt.setdefault("params", {}).get(k, 0)
                # the description usually restates what the look already does: reinforce, don't stack
                nv = (max(cur, v) if v > 0 else min(cur, v)) if cur * v > 0 else cur + v
                tgt["params"][k] = round(max(lo if lo is not None else nv, min(hi if hi is not None else nv, nv)), 3)
            merged.append(a["label"])
        g = {k: v for k, v in group.items() if k != "_look"}
        g["steps"] = look_steps
        g["label"] = f"Look: {look['name']}" + (" (customised)" if merged else "")
        res["actions"] = [g] + rest
        if merged:
            res["notes"].append("Folded into the base look: " + "; ".join(merged))
    _validate(res["actions"])
    return res


def _validate(actions: list[dict[str, Any]]) -> None:
    for a in actions:
        if a.get("type") == "add":
            Step.from_dict({k: v for k, v in a.items() if k in ("op", "params", "mask", "label", "origin")})
        elif a.get("type") == "add_group":
            for s in a.get("steps", []):
                Step.from_dict(s)
        else:
            raise LazyRoomError("INVALID_PARAMS", f"Planner produced unsupported action {a.get('type')}")
