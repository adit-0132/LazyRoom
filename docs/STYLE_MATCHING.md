# Reference style matching

`match_style(project_id, reference)` (MCP) · `lazyroom style match src.jpg ref.jpg` (CLI) ·
`pm.style_match(...)` (SDK) · AI panel › "Match a reference photo" (GUI).

## Goal

Give photo **A** the photographic *look* of reference **B** — warm/faded/brownish, teal-orange, muted film… —
**without copying B's scene content** (B's sky, skin or brightness) and **as editable steps**, never as an
opaque transform.

## Method

1. **Measure** (OKLab, ≤384 px proxies): encoded-luma quantiles; tint of *near-neutral* pixels globally and per
   tonal zone (shadows / midtones / highlights) — neutrals take on the colour grade, while saturated pixels mostly
   show scene colour; chroma quantiles; per-hue-band saturation; grain (Immerkær noise estimate); vignetting with a
   four-corner consistency check; palette.
2. **Estimate the grade** (`matching="look"`, default). A single reference cannot separate "graded" from
   "content" exactly, so the grade is estimated as the reference's deviation from typical natural-photo
   statistics and applied *relative to the source's own statistics*:
   * global cast: source cast + ½ of the reference cast (it mixes grade and scene colour);
   * split toning: zone tint minus global cast, transferred at 80 %;
   * fade / highlight roll-off: transferred only with strong evidence (lifted black point; compressed top-end
     plateau) — a dark scene is not mistaken for a roll-off;
   * saturation: damped ratio against a typical-photo prior; per-hue structure transferred at half strength;
   * scene key/brightness is **not** transferred (no exposure change).
   `matching="exact"` instead matches B's statistics directly (useful for series of the same scene).
3. **Fit an editable pipeline**: `adjust.white_balance` → `adjust.exposure` → `adjust.curves` (luminance) →
   `adjust.color_grading` (3-way) → `adjust.saturation` → `adjust.hsl` (+ `effect.vignette`, `effect.grain` when
   evidenced; `adjust.black_white` when B is monochrome). Initial values are analytic (white balance uses a
   numerically measured Jacobian); refinement is a damped fixed-point iteration on the measured residuals, keeping
   the best iterate and stopping on convergence or the render budget. Look mode caps edits (±40 white balance,
   ±50 saturation, moderate grade strengths).
4. **Skin protection**: when the source contains skin, the colour steps carry an editable mask that partially
   excludes skin tones.
5. **Report**: loss before/after, per-component distances (tone, zone tints, neutral tint, chroma), colour
   distribution distance before/after (sliced Wasserstein in OKLab), content preservation (SSIM of luminance),
   style descriptors of source/reference/result, `similarity` and `confidence`, `suggested_strength`, and
   human-readable limitations.
6. **Strength**: by default the fitted group is applied at `0.35 + 0.65 × similarity` — strong when the match is
   confident, gentle when the reference is ambiguous. Pass `strength` to override; every step stays editable.
7. Optional `mode="parametric+lut"` adds a residual 33³ LUT (Monge–Kantorovich transform in OKLab) at 50 %.

## Evaluation (reproducible: `python scripts/eval_style_match.py`)

Ground truth: reference = photo B graded with one of 8 built-in looks; ideal result = photo A graded with the same
look (grain/borders/light leaks excluded). 32 (look × photo-pair) cases on four CC0/public-domain photos. Metric:
mean CIE76 ΔE to the ground truth (lower is better).

| Method | mean ΔE |
|---|---|
| **LazyRoom `match_style` (default)** | **13.17** |
| Leave the photo unedited | 14.03 |
| Monge–Kantorovich linear transfer (MKL) | 33.88 |
| Reinhard mean/std transfer (OKLab) | 33.99 |
| Histogram matching | 35.22 |

Statistical colour transfer copies the reference's *content colours* and is far worse than doing nothing on
dissimilar scenes; LazyRoom's grade estimation is ~2.6× more accurate and also beats the no-op baseline. The
per-case table is in docs/eval/style_match.md. High-confidence matches (similarity ≥ 0.8) are accurate (e.g.
ΔE 5.7 vs 8.1 unedited); low-confidence ones are where errors concentrate — which is why confidence drives the
default strength and is reported to the agent.

## Limitations (stated honestly)

* Inferring a grade from one reference is ambiguous; strongly content-driven references (night scenes, single
  dominant colours) give low confidence.
* Local edits, relighting, retouching and compositing in the reference cannot be reproduced by global grading;
  lens softness/texture are only partially addressed (grain, vignette).
* The evaluation uses LazyRoom's own looks as ground truth and a small photo set; it measures grade recovery, not
  aesthetic preference.
