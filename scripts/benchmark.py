"""Performance benchmark for representative operations and looks.

Usage: python scripts/benchmark.py [--sizes 2,12,24] [--repeat 3] [--out docs/eval/benchmark.md]
Measures wall time per operation on synthetic photographic content at the given megapixel sizes, after a
warm-up run (numba compilation / caches excluded). Reports median of `repeat` runs.
"""
from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LAZYROOM_HOME", "/tmp/lazyroom-bench")

from lazyroom.core.context import RenderContext  # noqa: E402
from lazyroom.core.image import resize  # noqa: E402
from lazyroom.core.io import load_image  # noqa: E402
from lazyroom.core.pipeline import render, steps_from_list  # noqa: E402
from lazyroom.presets.store import PresetStore  # noqa: E402

CASES = {
    "adjust.exposure": [{"op": "adjust.exposure", "params": {"ev": 0.5}}],
    "adjust.white_balance": [{"op": "adjust.white_balance", "params": {"temperature": 20}}],
    "adjust.curves": [{"op": "adjust.curves", "params": {"rgb": [[0, 0.05], [0.5, 0.55], [1, 0.95]]}}],
    "adjust.hsl": [{"op": "adjust.hsl", "params": {"green_saturation": -40, "orange_luminance": 10}}],
    "adjust.color_grading": [{"op": "adjust.color_grading", "params": {"shadows_hue": 200, "shadows_saturation": 30}}],
    "adjust.tone": [{"op": "adjust.tone", "params": {"highlights": -40, "shadows": 30}}],
    "effect.grain": [{"op": "effect.grain", "params": {"amount": 20}}],
    "effect.sharpen": [{"op": "effect.sharpen", "params": {"amount": 80}}],
    "geometry.resize (lanczos3, 50%)": [{"op": "geometry.resize", "params": {"scale": 50, "method": "lanczos3"}}],
    "filter.sepia": [{"op": "filter.sepia", "params": {}}],
    "filter.gaussian_blur": [{"op": "filter.gaussian_blur", "params": {}}],
    "filter.floyd_steinberg_dithering": [{"op": "filter.floyd_steinberg_dithering", "params": {}}],
    "look: warm-70s-film (9 steps)": "warm-70s-film",
    "look: teal-orange (5 steps)": "teal-orange",
}


def make_image(mp: float) -> np.ndarray:
    src, _ = load_image(ROOT / "tests/data/photos/coffee.png")
    w = int(round((mp * 1e6 * 1.5) ** 0.5))
    h = int(round(w / 1.5))
    return np.ascontiguousarray(resize(src[..., :3], w, h))


def steps_for(spec):
    if isinstance(spec, str):
        return steps_from_list(PresetStore().get(spec)["steps"])
    return steps_from_list(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2,12,24")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "docs" / "eval" / "benchmark.md"))
    a = ap.parse_args()
    sizes = [float(s) for s in a.sizes.split(",")]
    results: dict[str, dict[float, float]] = {k: {} for k in CASES}
    for mp in sizes:
        img = make_image(mp)
        for name, spec in CASES.items():
            steps = steps_for(spec)
            render(img, steps, RenderContext(seed=1))  # warm-up
            times = []
            for _ in range(a.repeat):
                t0 = time.perf_counter()
                render(img, steps, RenderContext(seed=1))
                times.append(time.perf_counter() - t0)
            results[name][mp] = statistics.median(times)
            print(f"{mp:>5.1f} MP  {name:40s} {results[name][mp] * 1000:8.0f} ms", flush=True)
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    lines = ["# Performance benchmark", "",
             f"Machine: {platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
             f"{platform.system()} {platform.release()}, Python {platform.python_version()}. "
             f"Load average at the end: {load:.1f}. Median of {a.repeat} runs after warm-up.", "",
             "| Operation | " + " | ".join(f"{s:g} MP" for s in sizes) + " |",
             "|---|" + "---|" * len(sizes)]
    for name in CASES:
        lines.append(f"| {name} | " + " | ".join(f"{results[name][s] * 1000:.0f} ms" for s in sizes) + " |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
