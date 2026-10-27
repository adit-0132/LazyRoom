"""Reference-based style matching that produces an *editable* grading pipeline.

Approach (see docs/STYLE_MATCHING.md):

1. Measure source and reference in OKLab: lightness quantiles, the tint of
   *near-neutral* pixels per tonal zone (shadows / midtones / highlights) —
   which reveals the colour grade independent of scene content — chroma
   quantiles, per-hue-band chroma, grain and vignetting.
2. Initialise a parametric pipeline analytically (white balance, exposure,
   luminance tone curve via quantile mapping, three-way colour grading,
   saturation, HSL band saturation, grain, vignette).
3. Refine block-wise by fixed-point iteration: render the pipeline on a
   proxy, measure residuals against the reference statistics, update each
   parameter block with damped corrections (WB uses a numerically estimated
   Jacobian).  Keep the best parameters; stop on convergence or budget.
4. Report the fitted pipeline, per-component distances, a similarity score,
   content preservation (SSIM of luminance structure) and limitations.

The result is ordinary editable steps — never an opaque black box.  An
optional residual 3D LUT (``parametric+lut``) can be added on top.
"""
from __future__ import annotations

import hashlib
import os
import math
from typing import Any

import cv2
import numpy as np

from ..core.color import LUMA_709, rgb_to_oklab, srgb_to_linear
from ..core.context import RenderContext
from ..core.image import fit_within
from ..core.pipeline import steps_from_list, render
from .analysis import BAND_HUES, _hsv_hue_of_oklab_angle, analyze, noise_sigma, vignette_strength, zone_masks

F32 = np.float32
LQ = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
CQ = np.array([0.25, 0.5, 0.75, 0.9])
KNOTS = np.array([0.0, 0.12, 0.3, 0.5, 0.7, 0.88, 1.0])
ZONES = ("shadows", "midtones", "highlights")
SKIN_MASK = {"shapes": [{"type": "all"}, {"type": "skin", "mode": "subtract", "opacity": 0.7}]}


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def measure(img: np.ndarray) -> dict[str, Any]:
    x = np.clip(img[..., :3], 0, 1).astype(F32)
    lab = rgb_to_oklab(x)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    y = (x @ LUMA_709).astype(F32)
    m: dict[str, Any] = {"Lq": np.quantile(L, LQ), "yq": np.quantile(y, LQ), "y": y,
                         "Cq": np.quantile(C, CQ), "C_med": float(np.median(C))}
    # tint of near-neutral pixels: low chroma relative to the zone
    zab = {}
    for name, w in zone_masks(L).items():
        sel = w > 0.5
        if sel.sum() < 30:
            zab[name] = None
            continue
        thr = np.quantile(C[sel], 0.4)
        nsel = sel & (C <= thr)
        if nsel.sum() < 15:
            nsel = sel
        zab[name] = np.array([a[nsel].mean(), b[nsel].mean()], np.float64)
    thr = np.quantile(C, 0.4)
    nsel = C <= thr
    m["neutral_ab"] = np.array([a[nsel].mean(), b[nsel].mean()], np.float64)
    m["zone_ab"] = zab
    m["zone_w"] = {k: float((v > 0.5).mean()) for k, v in zone_masks(L).items()}
    from ..core.color import rgb_to_hsv

    hue = rgb_to_hsv(x)[..., 0] * 360
    bands = {}
    for name, centre in BAND_HUES.items():
        d = np.abs((hue - centre + 180) % 360 - 180)
        sel = (d < 22) & (C > 0.03)
        bands[name] = (float(sel.mean()), float(C[sel].mean()) if sel.sum() > 20 else 0.0)
    m["bands"] = bands
    m["is_mono"] = bool(np.quantile(C, 0.95) < 0.02)
    return m


# Typical statistics of ungraded, well-exposed photographs (encoded luma quantiles at LQ; OKLab chroma at CQ).
PRIOR_YQ = np.array([0.02, 0.05, 0.09, 0.22, 0.34, 0.42, 0.50, 0.62, 0.78, 0.86, 0.96])
PRIOR_CQ = np.array([0.02, 0.045, 0.08, 0.12])
PRIOR_SPREAD = 0.55  # typical p10..p90 spread of encoded luma
CONTRAST_TRANSFER = os.environ.get("PM_MATCH_CONTRAST", "0") == "1"


def build_target(msrc: dict[str, Any], mref: dict[str, Any], matching: str = "look") -> dict[str, Any]:
    """Statistics the result should have.

    ``exact``: the reference's own statistics (for series of the same scene).
    ``look`` (default): estimate the *grade* as the reference's deviation from
    typical-photo statistics and apply that deviation to the source, so scene
    content (key, colours, saturation of the subject) is not copied.
    """
    if matching == "exact":
        return mref
    t: dict[str, Any] = {"y": msrc["y"], "is_mono": mref["is_mono"], "zone_w": mref["zone_w"]}
    sq, rq = msrc["yq"], mref["yq"]
    s_lo, s_hi = sq[1], sq[-2]
    # endpoints: faded blacks / rolled-off whites are absolute style features
    # (natural photos rarely have p1 > 0.07 or p99 < 0.9; beyond that it is likely a deliberate fade / roll-off)
    lo = max(s_lo, rq[1]) if rq[0] > 0.07 else s_lo
    # highlight roll-off only when the top end is compressed into a plateau (not merely a dark scene)
    plateau = (rq[-1] - rq[-3]) < 0.12 and rq[-1] < 0.9 and rq[-3] > 0.5
    hi = min(s_hi, rq[-2]) if plateau else s_hi
    # contrast character relative to a typical photo — strongly damped (scene contrast varies a lot)
    c = float(np.clip(((rq[8] - rq[2]) / PRIOR_SPREAD) ** 0.3, 0.88, 1.12)) if CONTRAST_TRANSFER else 1.0
    med = sq[5]
    adj = med + (sq - med) * c
    span = max(s_hi - s_lo, 1e-3)
    tq = lo + (adj - s_lo) * (hi - lo) / span
    t["yq"] = np.clip(np.maximum.accumulate(tq), 0, 1)
    t["keep_exposure"] = True
    # analytic tone curve (fixed during refinement): black lift, highlight roll-off, damped contrast
    lift = float(rq[0]) if rq[0] > 0.07 else 0.0
    top = float(rq[-1]) if plateau else 1.0
    x = KNOTS
    sx = np.clip(0.5 + (x - 0.5) * c, 0, 1)
    t["curve"] = lift + sx * (top - lift)
    scale = float(np.clip((mref["C_med"] / PRIOR_CQ[1]) ** 0.5, 0.35, 1.6))
    if mref["is_mono"]:
        scale = 0.0
    t["Cq"] = msrc["Cq"] * scale
    t["C_med"] = msrc["C_med"] * scale
    # global cast: half of the reference's (it mixes grade and scene colour), split-tone signature in full
    t["neutral_ab"] = msrc["neutral_ab"] + 0.5 * mref["neutral_ab"]
    t["zone_ab"] = {}
    for z in ZONES:
        a, r = msrc["zone_ab"].get(z), mref["zone_ab"].get(z)
        t["zone_ab"][z] = None if a is None or r is None else \
            a - msrc["neutral_ab"] + t["neutral_ab"] + 0.8 * (r - mref["neutral_ab"])
    t["limits"] = {"wb": 40.0, "zone": 0.048, "sat": 50.0}
    # relative per-hue saturation structure
    bands = {}
    for b, (sh_r, c_r) in mref["bands"].items():
        sh_s, c_s = msrc["bands"][b]
        if sh_r > 0.03 and sh_s > 0.03 and c_r > 0 and c_s > 0:
            rel = (c_r / max(mref["C_med"], 1e-4)) / (c_s / max(msrc["C_med"], 1e-4))
            bands[b] = (sh_s, c_s * scale * rel ** 0.5)
        else:
            bands[b] = (sh_s, c_s * scale)
    t["bands"] = bands
    return t


def loss(res: dict[str, Any], ref: dict[str, Any]) -> tuple[float, dict[str, float]]:
    tone = float(np.abs(res["yq"] - ref["yq"]).mean())
    zone = 0.0
    wsum = 0.0
    for z in ZONES:
        if res["zone_ab"].get(z) is None or ref["zone_ab"].get(z) is None:
            continue
        w = 0.3 + ref["zone_w"][z]
        zone += w * float(np.linalg.norm(res["zone_ab"][z] - ref["zone_ab"][z]))
        wsum += w
    zone = zone / wsum if wsum else 0.0
    neutral = float(np.linalg.norm(res["neutral_ab"] - ref["neutral_ab"]))
    chroma = float(np.abs(res["Cq"] - ref["Cq"]).mean())
    comps = {"tone": tone, "zone_tint": zone, "neutral_tint": neutral, "chroma": chroma}
    total = 1.0 * tone + 2.0 * zone + 1.5 * neutral + 1.0 * chroma
    return total, comps


# ---------------------------------------------------------------------------
# Parameters <-> steps
# ---------------------------------------------------------------------------

def _wheel(ab: np.ndarray) -> tuple[float, float]:
    mag = float(np.hypot(ab[0], ab[1]))
    sat = float(np.clip(mag / 0.08 * 100, 0, 100))
    hue = _hsv_hue_of_oklab_angle(math.atan2(ab[1], ab[0])) if mag > 1e-6 else 0.0
    return round(hue, 1), round(sat, 2)


class Params:
    def __init__(self) -> None:
        self.temperature = 0.0
        self.tint = 0.0
        self.ev = 0.0
        self.curve = KNOTS.copy()
        self.zones = {z: np.zeros(2) for z in ZONES}
        self.sat = 0.0
        self.mono = False
        self.mono_tint = np.zeros(2)
        self.hsl: dict[str, float] = {}
        self.grain: dict[str, float] | None = None
        self.vignette: float = 0.0
        self.skin_mask = False

    def copy(self) -> "Params":
        p = Params()
        p.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else
                               ({kk: (vv.copy() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                                if isinstance(v, dict) else v)) for k, v in self.__dict__.items()})
        return p

    def steps(self, upto_curve: bool = False, extras: bool = True) -> list[dict[str, Any]]:
        st: list[dict[str, Any]] = []
        o = "style_match"
        st.append({"op": "adjust.white_balance", "label": "Match: white balance", "origin": o,
                   "params": {"temperature": round(float(np.clip(self.temperature, -100, 100)), 2),
                              "tint": round(float(np.clip(self.tint, -100, 100)), 2)}})
        st.append({"op": "adjust.exposure", "label": "Match: exposure", "origin": o,
                   "params": {"ev": round(float(np.clip(self.ev, -5, 5)), 3)}})
        if upto_curve:
            return st
        pts = [[float(x), float(np.clip(y, 0, 1))] for x, y in zip(KNOTS, _monotone(self.curve))]
        st.append({"op": "adjust.curves", "label": "Match: tone curve", "origin": o,
                   "params": {"luminance": [[round(x, 4), round(y, 4)] for x, y in pts]}})
        mask = SKIN_MASK if self.skin_mask else None
        if self.mono:
            hue, sat = _wheel(self.mono_tint)
            st.append({"op": "adjust.black_white", "label": "Match: monochrome", "origin": o,
                       "params": {"tint_hue": hue, "tint_saturation": min(100.0, sat * 0.8)}})
        else:
            gp: dict[str, float] = {}
            for z in ZONES:
                hue, sat = _wheel(self.zones[z])
                gp[f"{z}_hue"], gp[f"{z}_saturation"] = hue, sat
            g = {"op": "adjust.color_grading", "label": "Match: colour grade", "origin": o, "params": gp}
            if mask:
                g["mask"] = mask
            st.append(g)
            s = {"op": "adjust.saturation", "label": "Match: saturation", "origin": o,
                 "params": {"saturation": round(float(np.clip(self.sat, -100, 100)), 2)}}
            if mask:
                s["mask"] = mask
            st.append(s)
            hsl = {k: round(float(np.clip(v, -100, 100)), 1) for k, v in self.hsl.items() if abs(v) >= 3}
            if hsl:
                h = {"op": "adjust.hsl", "label": "Match: colour mixer", "origin": o, "params": hsl}
                if mask:
                    h["mask"] = mask
                st.append(h)
        if extras:
            if abs(self.vignette) >= 3:
                st.append({"op": "effect.vignette", "label": "Match: vignette", "origin": o,
                           "params": {"amount": round(float(np.clip(self.vignette, -100, 100)), 1),
                                      "midpoint": 35, "feather": 70}})
            if self.grain:
                st.append({"op": "effect.grain", "label": "Match: grain", "origin": o, "params": self.grain})
        return st


def _monotone(y: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(y, float), 0, 1)
    return np.maximum.accumulate(y + np.arange(len(y)) * 1e-4)


def _render(img: np.ndarray, steps: list[dict[str, Any]]) -> np.ndarray:
    out, _ = render(img, steps_from_list(steps), RenderContext(seed=1))
    return out


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
_CAL: dict[str, float] = {}


def _grain_sigma_per_amount() -> float:
    if "grain" not in _CAL:
        flat = np.full((256, 256, 3), 0.5, F32)
        out = _render(flat, [{"op": "effect.grain", "params": {"amount": 50, "size": 1.0}}])
        _CAL["grain"] = noise_sigma((out @ LUMA_709).astype(F32)) / 50.0
    return _CAL["grain"]


def _vignette_per_amount() -> float:
    if "vig" not in _CAL:
        flat = np.full((192, 288, 3), 0.5, F32)
        out = _render(flat, [{"op": "effect.vignette", "params": {"amount": -50, "midpoint": 35, "feather": 70}}])
        L = rgb_to_oklab(out)[..., 0]
        _CAL["vig"] = vignette_strength(L) / -50.0
    return _CAL["vig"]


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def match_style(src: np.ndarray, ref: np.ndarray, *, strength: float = 1.0, preserve_skin: bool = True,
                include_grain: bool = True, include_vignette: bool = True, max_iterations: int = 60,
                mode: str = "parametric", reference_path: str | None = None, source_scale: float = 1.0,
                matching: str = "look") -> dict[str, Any]:
    """Fit an editable pipeline that gives ``src`` the photographic style of ``ref``.

    ``matching='look'`` transfers the grade (default); ``'exact'`` matches the reference's statistics.
    """
    work, _ = fit_within(src[..., :3], 384)
    ref_s, _ = fit_within(ref[..., :3], 384)
    work = np.ascontiguousarray(work)
    mref_raw = measure(ref_s)
    msrc = measure(work)
    mref = build_target(msrc, mref_raw, matching)
    budget = max(8, int(max_iterations))
    renders = 0
    p = Params()
    p.mono = mref_raw["is_mono"]
    # skin protection only when the source has a meaningful amount of skin
    if preserve_skin and not p.mono:
        from ..core.masks import render_mask

        skin = render_mask({"shapes": [{"type": "skin"}]}, work, RenderContext())
        p.skin_mask = bool(skin is not None and float(skin.mean()) > 0.03)

    # --- initial estimates -------------------------------------------------
    lin_ref = float(srgb_to_linear(np.array([mref["yq"][5]], F32))[0])
    lin_src = float(srgb_to_linear(np.array([msrc["yq"][5]], F32))[0])
    p.ev = 0.0 if mref.get("keep_exposure") else float(
        np.clip(0.6 * math.log2(max(lin_ref, 1e-4) / max(lin_src, 1e-4)), -2.0, 2.0))
    # white balance Jacobian (2 extra renders)
    base_img = _render(work, p.steps(upto_curve=True))
    renders += 1
    jac = np.zeros((2, 2))
    m0 = measure(base_img)["neutral_ab"]
    for j, key in enumerate(("temperature", "tint")):
        q = p.copy()
        setattr(q, key, 20.0)
        mj = measure(_render(work, q.steps(upto_curve=True)))["neutral_ab"]
        jac[:, j] = (mj - m0) / 20.0
        renders += 1
    try:
        jinv = np.linalg.inv(jac)
    except np.linalg.LinAlgError:
        jinv = np.zeros((2, 2))
    d = mref["neutral_ab"] - m0
    lim = mref.get("limits", {"wb": 100.0, "zone": 0.08, "sat": 100.0})
    tt = jinv @ d
    p.temperature = float(np.clip(tt[0], -lim["wb"], lim["wb"]))
    p.tint = float(np.clip(tt[1], -lim["wb"], lim["wb"]))
    if p.mono:
        p.mono_tint = mref_raw["neutral_ab"].copy()

    pre = measure(_render(work, p.steps(upto_curve=True)))
    renders += 1
    p.curve = mref["curve"].copy() if "curve" in mref else _curve_from_quantiles(pre["y"], mref["yq"])
    if not p.mono:
        for z in ZONES:
            if pre["zone_ab"].get(z) is not None and mref["zone_ab"].get(z) is not None:
                p.zones[z] = (mref["zone_ab"][z] - mref["neutral_ab"]) - (pre["zone_ab"][z] - pre["neutral_ab"])
        ratio = mref["C_med"] / max(pre["C_med"], 1e-4)
        p.sat = float(np.clip((ratio ** 0.85 - 1) * 100, -lim["sat"], lim["sat"]))
        for z in ZONES:
            mag = np.linalg.norm(p.zones[z])
            if mag > lim["zone"]:
                p.zones[z] *= lim["zone"] / mag

    # --- iterative refinement ---------------------------------------------
    history = []
    cur = _render(work, p.steps(extras=False))
    renders += 1
    mcur = measure(cur)
    l0, _ = loss(measure(work), mref)
    best_loss, best_comps = loss(mcur, mref)
    analytic_loss = best_loss
    best = p.copy()
    history.append(round(best_loss, 5))
    converged = False
    it = 0
    while renders + 2 <= budget:
        it += 1
        # white balance: correct the global neutral tint
        d = mref["neutral_ab"] - mcur["neutral_ab"]
        tt = jinv @ d * 0.7
        p.temperature = float(np.clip(p.temperature + tt[0], -lim["wb"], lim["wb"]))
        p.tint = float(np.clip(p.tint + tt[1], -lim["wb"], lim["wb"]))
        # tone curve: move knots by the quantile residual at matching levels
        pre = measure(_render(work, p.steps(upto_curve=True)))
        renders += 1
        if "curve" not in mref:
            p.curve = _monotone(p.curve + 0.7 * _curve_residual(pre["y"], mcur["yq"], mref["yq"]))
        if not p.mono:
            for z in ZONES:
                if mcur["zone_ab"].get(z) is not None and mref["zone_ab"].get(z) is not None:
                    rel_ref = mref["zone_ab"][z] - mref["neutral_ab"]
                    rel_cur = mcur["zone_ab"][z] - mcur["neutral_ab"]
                    p.zones[z] = p.zones[z] + 0.7 * (rel_ref - rel_cur)
                    mag = np.linalg.norm(p.zones[z])
                    if mag > lim["zone"]:
                        p.zones[z] *= lim["zone"] / mag
            ratio = float(mref["Cq"][1] / max(mcur["Cq"][1], 1e-4))
            f = (1 + p.sat / 100.0) * ratio ** 0.7
            p.sat = float(np.clip((f - 1) * 100, -lim["sat"], lim["sat"]))
        else:
            p.mono_tint = p.mono_tint + 0.7 * (mref["neutral_ab"] - mcur["neutral_ab"])
        cur = _render(work, p.steps(extras=False))
        renders += 1
        mcur = measure(cur)
        lval, comps = loss(mcur, mref)
        history.append(round(lval, 5))
        if lval < best_loss - 1e-6:
            improvement = (best_loss - lval) / max(best_loss, 1e-9)
            best_loss, best_comps, best = lval, comps, p.copy()
            if improvement < 0.005:
                converged = True
                break
        else:
            p = best.copy()  # step made things worse: restart from best with damping
            converged = it > 2
            if converged:
                break
    p = best

    # --- HSL band saturation (relative to the global change) ----------------
    final_base = _render(work, p.steps(extras=False))
    renders += 1
    mfin = measure(final_base)
    if not p.mono:
        for band, (share_r, c_r) in mref["bands"].items():
            share_s, c_s = mfin["bands"][band]
            if share_r > 0.03 and share_s > 0.03 and c_r > 0 and c_s > 0:
                rel = (c_r / c_s)
                val = (rel ** 0.7 - 1) * 100
                if abs(val) > 4:
                    p.hsl[f"{band}_saturation"] = float(np.clip(val, -60, 60))
    # --- grain / vignette -------------------------------------------------------
    gray_ref = (ref_s @ LUMA_709).astype(F32)
    gray_src = (final_base @ LUMA_709).astype(F32)
    g_ref, g_src = noise_sigma(gray_ref), noise_sigma(gray_src)
    if include_grain and g_ref > g_src + 0.003:
        amount = float(np.clip((g_ref - g_src) / max(_grain_sigma_per_amount(), 1e-6), 3, 100))
        p.grain = {"amount": round(amount, 1), "size": round(float(np.clip(1.2 * source_scale, 0.5, 12)), 2)}
    v_ref = vignette_strength(rgb_to_oklab(ref_s)[..., 0])
    v_src = vignette_strength(rgb_to_oklab(final_base)[..., 0])
    if include_vignette and v_ref < -0.12 and v_ref < v_src - 0.08 and _radial_falloff(rgb_to_oklab(ref_s)[..., 0]):
        p.vignette = float(np.clip((v_ref - v_src) / max(_vignette_per_amount(), 1e-6), -80, 0))

    steps = p.steps()
    lut_info = None
    if mode == "parametric+lut" and not p.mono:
        lut_info = _residual_lut(_render(work, steps), ref_s)
        steps.append({"op": "adjust.lut", "label": "Match: residual LUT", "origin": "style_match",
                      "params": {"path": lut_info["path"], "strength": 0.5}})
    result = _render(work, steps)
    renders += 1
    mres = measure(result)
    final_loss, final_comps = loss(mres, mref)
    from .metrics import ssim_gray, sliced_wasserstein

    ssim = ssim_gray((work @ LUMA_709).astype(F32), (np.clip(result, 0, 1) @ LUMA_709).astype(F32))
    swd_before = sliced_wasserstein(work, ref_s, n_proj=24, n_samples=8000)
    swd_after = sliced_wasserstein(result, ref_s, n_proj=24, n_samples=8000)
    similarity = float(np.clip(1 - final_loss / 0.25, 0, 1))
    an_ref = analyze(ref_s)
    an_src = analyze(work)
    an_res = analyze(result)
    limitations = _limitations(an_src, an_ref, mref_raw, msrc, similarity, ssim, p)
    report = {
        "loss_initial": round(l0, 5), "loss_after_initial_estimate": round(analytic_loss, 5),
        "loss_final": round(final_loss, 5),
        "loss_history": history, "renders": renders, "iterations": it, "converged": converged,
        "components_final": {k: round(v, 5) for k, v in final_comps.items()},
        "similarity": round(similarity, 3),
        "suggested_strength": round(0.35 + 0.65 * similarity, 3),
        "confidence": "high" if similarity > 0.75 else "medium" if similarity > 0.5 else "low",
        "color_distribution_distance": {"before": round(swd_before, 5), "after": round(swd_after, 5)},
        "content_preservation_ssim": round(ssim, 4),
        "skin_protection": p.skin_mask,
        "limitations": limitations,
        "reference": reference_path,
        "matching": matching,
        "descriptors": {"reference": an_ref["descriptors"], "source": an_src["descriptors"],
                        "result": an_res["descriptors"]},
    }
    if lut_info:
        report["residual_lut"] = lut_info
    return {"steps": steps, "report": report}


def _radial_falloff(L: np.ndarray) -> bool:
    """True when all four corners are darker than the centre (a real vignette, not dark content in one area)."""
    h, w = L.shape
    ch, cw = max(2, h // 6), max(2, w // 6)
    centre = float(np.median(L[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw]))
    corners = [L[:ch, :cw], L[:ch, -cw:], L[-ch:, :cw], L[-ch:, -cw:]]
    return all(float(np.median(c)) < centre * 0.88 for c in corners)


def _curve_from_quantiles(pre_y: np.ndarray, ref_yq: np.ndarray) -> np.ndarray:
    """Quantile mapping evaluated at the fixed knots, with endpoints from ref black/white points."""
    flat = pre_y.ravel()
    levels = np.searchsorted(np.sort(flat), KNOTS) / max(flat.size, 1)
    levels = np.clip(levels, LQ[0], LQ[-1])
    y = np.interp(levels, LQ, ref_yq)
    y[0] = float(np.clip(ref_yq[0] - np.quantile(flat, LQ[0]) * 0.9, 0, 0.5)) if ref_yq[0] > np.quantile(flat, LQ[0]) else 0.0
    y[-1] = float(np.clip(1 - max(0.0, np.quantile(flat, LQ[-1]) - ref_yq[-1]) * 1.1, 0.5, 1.0))
    y = KNOTS + 0.9 * (y - KNOTS)
    return _monotone(y)


def _curve_residual(pre_y: np.ndarray, cur_yq: np.ndarray, ref_yq: np.ndarray) -> np.ndarray:
    """Output-luma error at each knot: knots sit at the input quantile levels they map."""
    flat = pre_y.ravel()
    levels = np.clip(np.searchsorted(np.sort(flat), KNOTS) / max(flat.size, 1), LQ[0], LQ[-1])
    return np.interp(levels, LQ, ref_yq - cur_yq)


def _limitations(an_src, an_ref, mref, msrc, similarity, ssim, p) -> list[str]:
    out = []
    if mref["is_mono"]:
        out.append("Reference is monochrome: matched with a black & white conversion plus tint.")
    shared = sum(1 for b in mref["bands"] if mref["bands"][b][0] > 0.05 and msrc["bands"][b][0] > 0.05)
    if shared <= 1 and not mref["is_mono"]:
        out.append("Scene colours differ strongly between source and reference; the grade was inferred from "
                   "tonality and near-neutral tints, so exact colours are not copied.")
    if abs(an_ref["texture"]["sharpness"] - an_src["texture"]["sharpness"]) > 0.02:
        out.append("Reference sharpness/texture differs; global grading does not reproduce lens softness or "
                   "local contrast — consider effect.diffusion / adjust.clarity manually.")
    if similarity < 0.5:
        out.append("Low match confidence: the reference look may rely on local edits, compositing or lighting "
                   "that global adjustments cannot reproduce.")
    if ssim < 0.85:
        out.append(f"The grade changes luminance structure noticeably (SSIM {ssim:.2f}); review the tone curve.")
    if p.skin_mask:
        out.append("Skin tones were partially protected from the colour grade (editable mask on the grade steps).")
    return out


def _residual_lut(result: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Bake the Monge–Kantorovich OKLab transform result->ref into a 33³ .cube LUT."""
    from ..core.color import oklab_to_rgb
    from ..core.paths import data_dir, ensure
    from ..ops._util import identity_lut
    from .transfer import mkl_map, _sample

    A, ms, mr = mkl_map(_sample(rgb_to_oklab(np.clip(result, 0, 1))), _sample(rgb_to_oklab(np.clip(ref, 0, 1))))
    grid = identity_lut(33)
    lab = rgb_to_oklab(grid.reshape(-1, 1, 3))[:, 0]
    mapped = (lab - ms) @ A.T + mr
    rgb = np.clip(oklab_to_rgb(mapped.astype(F32)[:, None])[:, 0], 0, 1).reshape(33, 33, 33, 3)
    lines = ["TITLE \"LazyRoom residual match\"", "LUT_3D_SIZE 33"]
    for bi in range(33):
        for gi in range(33):
            for ri in range(33):
                r, g, b = rgb[ri, gi, bi]
                lines.append(f"{r:.6f} {g:.6f} {b:.6f}")
    text = "\n".join(lines) + "\n"
    h = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
    p = ensure(data_dir() / "luts" / "generated") / f"match-{h}.cube"
    p.write_text(text)
    return {"path": str(p), "method": "MKL (OKLab) residual, 33^3"}


def normalization_steps(img: np.ndarray, mode: str = "both", target_median: float = 0.45) -> list[dict[str, Any]]:
    """Per-image corrections for consistent series: exposure to a common median and neutral white balance."""
    work, _ = fit_within(img[..., :3], 384)
    steps: list[dict[str, Any]] = []
    if mode in ("white_balance", "both"):
        m0 = measure(work)["neutral_ab"]
        jac = np.zeros((2, 2))
        for j, key in enumerate(("temperature", "tint")):
            mj = measure(_render(work, [{"op": "adjust.white_balance", "params": {key: 20}}]))["neutral_ab"]
            jac[:, j] = (mj - m0) / 20.0
        try:
            tt = np.linalg.solve(jac, -m0 * 0.85)
        except np.linalg.LinAlgError:
            tt = np.zeros(2)
        steps.append({"op": "adjust.white_balance", "label": "Normalize: white balance", "origin": "style_match",
                      "params": {"temperature": round(float(np.clip(tt[0], -60, 60)), 2),
                                 "tint": round(float(np.clip(tt[1], -60, 60)), 2)}})
    if mode in ("exposure", "both"):
        cur = _render(work, steps) if steps else work
        med = float(np.median((np.clip(cur, 0, 1) @ LUMA_709)))
        lin_t = float(srgb_to_linear(np.array([target_median], F32))[0])
        lin_m = float(srgb_to_linear(np.array([max(med, 1e-3)], F32))[0])
        ev = float(np.clip(math.log2(lin_t / lin_m), -1.5, 1.5))
        steps.append({"op": "adjust.exposure", "label": "Normalize: exposure", "origin": "style_match",
                      "params": {"ev": round(ev, 3)}})
    return steps
