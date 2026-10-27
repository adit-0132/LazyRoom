# Project format — `lazyroom.project/1`

A project (`*.lazyroom`) is UTF-8 JSON that any agent or script can read and write. Sources are referenced, never
copied or modified.

```jsonc
{
  "schema": "lazyroom.project/1",          // major version; readers reject unknown majors
  "id": "p3f9a2c1d7e",                        // stable within a file; MCP project ids are per server session
  "name": "beach",
  "created": 1792000000.0, "modified": 1792000123.4,   // UNIX seconds
  "seed": 1234567,                            // seeds every random operation (grain, noise...) → reproducible
  "source": {"path": "/home/me/Pictures/beach.jpg", "sha256": "…", "width": 6000, "height": 4000,
             "format": "JPEG", "file_size": 8123456},
  "steps": [                                  // rendered in order
    {"id": "s1a2b3c4", "op": "adjust.white_balance", "params": {"temperature": 18, "tint": 0},
     "enabled": true, "origin": "agent", "label": "Warm"},
    {"id": "s5e6f7a8", "op": "group", "label": "Look: Faded analog film", "opacity": 0.8, "enabled": true,
     "origin": "preset:faded-analog", "steps": [ { "...": "nested steps" } ]},
    {"id": "s9b0c1d2", "op": "adjust.exposure", "params": {"ev": 0.4}, "enabled": true, "origin": "user",
     "mask": {"shapes": [{"type": "radial", "center": [0.5, 0.45], "radius": [0.3, 0.35], "feather": 0.6}]},
     "blend": "normal"}
  ],
  "export": {"last_output": "/home/me/Pictures/beach-edit.jpg", "format": "jpg", "quality": 92,
             "bit_depth": 8, "metadata": "keep"},
  "notes": "", "meta": {},
  "history": {"cursor": 3, "entries": [
    {"id": "h…", "time": 1792000001.2, "action": "create", "description": "Opened beach.jpg", "actor": "system",
     "steps": []},
    {"id": "h…", "time": 1792000050.0, "action": "edit", "description": "Add adjust.white_balance",
     "actor": "agent", "steps": [ "…full pipeline snapshot…" ]}
  ]}
}
```

## Steps

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable step id (`s` + 8 hex) used by `update`/`remove`/`move`/`toggle` actions |
| `op` | string | Operation id (see docs/OPERATIONS.md) or `"group"` |
| `params` | object | Validated parameter values in JSON form (colours `#RRGGBB(AA)`, points `[x, y]`) |
| `steps` | array | Only for groups: nested steps rendered as one unit |
| `enabled` | bool | Disabled steps are skipped (kept in the project) |
| `opacity` | 0..1 | Mix of the step's result with its input |
| `blend` | string | One of 24 blend modes (`normal`, `multiply`, `screen`, `overlay`, `soft_light`, …) |
| `mask` | object | Local-adjustment mask (below); not allowed on geometry-changing operations |
| `label` | string | Human-readable description |
| `origin` | string | `user`, `agent`, `preset:<id>`, `look`, `style_match`, `intent`, `import`, `system` |

## Masks

```jsonc
{"shapes": [ {"type": "radial", "center": [0.5, 0.5], "radius": [0.3, 0.3], "feather": 0.5, "rotation": 0},
             {"type": "luminance", "min": 0.0, "max": 0.35, "softness": 0.1, "mode": "intersect"} ],
 "invert": false, "feather": 0.0, "density": 1.0}
```
Shape types: `all`, `rect` (x, y, w, h, rotation, roundness, feather), `ellipse`/`radial` (center, radius,
rotation, feather), `linear` (start → end), `polygon` (points, feather), `brush` (strokes: points, radius,
hardness, opacity, erase), `luminance` (min, max, softness), `color` (color, tolerance, softness), `hue` (hue,
width, softness, min_saturation), `skin` (softness), `image` (path, channel), `subject` (AI model). Each shape
may set `mode` (add | subtract | intersect | replace), `invert` and `opacity`. Coordinates are normalised to the
image (0..1, origin top-left); lengths are fractions of the shorter side, so masks are resolution independent.

## Colour pipeline

* Decoding converts embedded ICC profiles to sRGB (LittleCMS, perceptual intent); EXIF orientation is applied.
* Working buffer: float32, sRGB-encoded RGB or RGBA with **straight** alpha, values not clamped between steps
  (ImageToolbox `filter.*` ports clamp to 0..1 like the original 8-bit pipeline).
* Operations needing physical light (exposure, white balance, bloom, halation, vignette, "linear" blurs) convert
  to linear light internally; perceptual colour edits (saturation, HSL, grading, matching) use OKLab.
* 16-bit PNG/TIFF sources keep 16-bit precision; export can write 8 or 16 bits (PNG/TIFF).
* Export clamps to 0..1, embeds an sRGB ICC profile (where the format supports it), writes EXIF orientation 1,
  and keeps/strips metadata according to the policy (`keep`, `keep_no_gps`, `strip`).

## Compatibility

* Unknown fields in steps or actions are rejected (typos surface immediately).
* Operation ids are stable; renamed ids remain as aliases (old projects keep opening).
* Each operation has a `version`; parameter changes that alter meaning bump it.
* Presets (`lazyroom.preset/1`) are `{schema, id, name, description, tags, steps}`.
* Projects saved before the rename to LazyRoom (`*.pmproj`, schema `photomasta.project/1`) open unchanged;
  saving them writes a new `*.lazyroom` file next to the old one.
