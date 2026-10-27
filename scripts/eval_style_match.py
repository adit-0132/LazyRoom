"""Quantitative evaluation of reference style matching with ground truth.

For each look L and photo pair (A, B): the reference is B graded with L; the
ideal answer is A graded with L.  We compare LazyRoom's style_match(A, ref)
against that ground truth and against statistical baselines (Reinhard, MKL,
histogram matching).  Lower ΔE / distribution distance is better.

Usage:  python scripts/eval_style_match.py [--out docs/eval/style_match.md]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("LAZYROOM_ITB_ONLY", "none")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lazyroom.core.context import RenderContext  # noqa: E402
from lazyroom.core.image import fit_within  # noqa: E402
from lazyroom.core.io import load_image  # noqa: E402
from lazyroom.core.pipeline import render, steps_from_list  # noqa: E402
from lazyroom.presets.store import PresetStore  # noqa: E402
from lazyroom.style.match import match_style  # noqa: E402
from lazyroom.style.metrics import delta_e76, sliced_wasserstein, ssim_gray  # noqa: E402
from lazyroom.style.transfer import transfer_colors  # noqa: E402

LOOKS = ["warm-70s-film", "teal-orange", "faded-analog", "cool-urban", "earthy-muted", "bw-classic", "moody-dark",
         "cross-process"]
PHOTOS = ["astronaut.png", "coffee.png", "chelsea.png", "rocket.jpg"]


def apply_look(img, look_id):
    steps = [s for s in PresetStore().get(look_id)["steps"] if s["op"] not in ("effect.grain", "draw.border",
                                                                               "effect.light_leak")]
    out, _ = render(img, steps_from_list(steps), RenderContext(seed=1))
    return np.clip(out, 0, 1)


def score(result, truth, source):
    de = float(delta_e76(np.clip(result, 0, 1), truth).mean())
    swd = sliced_wasserstein(result, truth, n_proj=24, n_samples=8000)
    y = np.array([0.2126, 0.7152, 0.0722], np.float32)
    ssim = ssim_gray(np.clip(source, 0, 1) @ y, np.clip(result, 0, 1) @ y)
    return de, swd, ssim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "eval" / "style_match.md"))
    args = ap.parse_args()
    photos = {p: fit_within(load_image(ROOT / "tests/data/photos" / p)[0][..., :3], 384)[0] for p in PHOTOS}
    rows = []
    totals: dict[str, list[float]] = {}
    for look in LOOKS:
        for a, b in itertools.permutations(PHOTOS, 2):
            if (PHOTOS.index(a) + PHOTOS.index(b)) % 2:  # subsample pairs to keep runtime modest
                continue
            A, B = photos[a], photos[b]
            ref = apply_look(B, look)
            truth = apply_look(A, look)
            res = {}
            t0 = time.time()
            m = match_style(A, ref, include_grain=False, max_iterations=40)
            out, _ = render(A, steps_from_list(m["steps"]), RenderContext(seed=1))
            w = m["report"]["suggested_strength"]  # what the API applies by default (strength=None)
            out = A + (np.clip(out, 0, 1) - A) * w
            res["lazyroom"] = score(out, truth, A) + (time.time() - t0,)
            for method in ("reinhard", "mkl", "histogram"):
                t0 = time.time()
                o = transfer_colors(A, ref, method=method)
                res[method] = score(o, truth, A) + (time.time() - t0,)
            res["unedited"] = score(A, truth, A) + (0.0,)
            rows.append((look, a, b, res, m["report"]["similarity"], m["report"]["confidence"]))
            for k, v in res.items():
                totals.setdefault(k, []).append(v[0])
            print(look, a, b, {k: round(v[0], 2) for k, v in res.items()}, flush=True)
    lines = ["# Reference style matching — evaluation", "",
             "Ground truth: the reference is photo B graded with a built-in look; the ideal result is photo A graded "
             "with the same look (grain, borders and light leaks excluded). Metric: mean CIE76 ΔE to the ground truth "
             "(lower is better); SSIM of luminance vs the source measures content preservation.", "",
             "| Method | mean ΔE (lower = better) | median ΔE |", "|---|---|---|"]
    for k, v in sorted(totals.items(), key=lambda kv: np.mean(kv[1])):
        lines.append(f"| {k} | {np.mean(v):.2f} | {np.median(v):.2f} |")
    lines += ["", "## Per case", "", "| Look | A | B | LazyRoom ΔE | Reinhard | MKL | Histogram | Unedited | "
              "PM SSIM | PM similarity |", "|---|---|---|---|---|---|---|---|---|---|"]
    for look, a, b, res, sim, conf in rows:
        lines.append(f"| {look} | {a} | {b} | {res['lazyroom'][0]:.2f} | {res['reinhard'][0]:.2f} | "
                     f"{res['mkl'][0]:.2f} | {res['histogram'][0]:.2f} | {res['unedited'][0]:.2f} | "
                     f"{res['lazyroom'][2]:.3f} | {sim} ({conf}) |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
