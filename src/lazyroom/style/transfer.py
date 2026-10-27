"""Statistical colour transfer (non-parametric baselines used by style matching).

* ``reinhard`` — per-channel mean/std matching in OKLab (Reinhard et al. 2001,
  using OKLab instead of lαβ).
* ``mkl`` — Monge–Kantorovich linear transform: the closed-form optimal
  transport map between two Gaussians (Pitié & Kokaram 2007), in OKLab.
* ``histogram`` — per-channel CDF matching in OKLab.
"""
from __future__ import annotations

import numpy as np

from ..core.color import oklab_to_rgb, rgb_to_oklab

F32 = np.float32


def _sqrtm_psd(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    w = np.clip(w, 1e-10, None)
    return (v * np.sqrt(w)) @ v.T


def mkl_map(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (A, mu_src, mu_ref) so that x -> (x - mu_src) @ A.T + mu_ref."""
    mu_s, mu_r = src.mean(0), ref.mean(0)
    cs = np.cov((src - mu_s).T) + np.eye(3) * 1e-7
    cr = np.cov((ref - mu_r).T) + np.eye(3) * 1e-7
    cs_h = _sqrtm_psd(cs)
    cs_hi = np.linalg.inv(cs_h)
    mid = _sqrtm_psd(cs_h @ cr @ cs_h)
    A = cs_hi @ mid @ cs_hi
    return A, mu_s, mu_r


def _sample(lab: np.ndarray, n: int = 200_000, seed: int = 0) -> np.ndarray:
    flat = lab.reshape(-1, 3)
    if len(flat) <= n:
        return flat.astype(np.float64)
    idx = np.random.default_rng(seed).choice(len(flat), n, replace=False)
    return flat[idx].astype(np.float64)


def transfer_colors(img: np.ndarray, ref: np.ndarray, *, method: str = "reinhard",
                    preserve_luminance: bool = False) -> np.ndarray:
    src_lab = rgb_to_oklab(np.clip(img[..., :3], 0, 1))
    ref_lab = rgb_to_oklab(np.clip(ref[..., :3], 0, 1))
    s, r = _sample(src_lab), _sample(ref_lab)
    if method == "reinhard":
        ms, ss = s.mean(0), s.std(0) + 1e-6
        mr, sr = r.mean(0), r.std(0) + 1e-6
        out = (src_lab - ms) * (sr / ss) + mr
    elif method == "mkl":
        A, ms, mr = mkl_map(s, r)
        out = (src_lab.reshape(-1, 3) - ms) @ A.T + mr
        out = out.reshape(src_lab.shape)
    elif method == "histogram":
        out = np.empty_like(src_lab)
        qs = np.linspace(0, 1, 1025)
        for c in range(3):
            sq = np.quantile(s[:, c], qs)
            rq = np.quantile(r[:, c], qs)
            sq = np.maximum.accumulate(sq + np.arange(len(sq)) * 1e-9)
            out[..., c] = np.interp(src_lab[..., c], sq, rq)
    else:
        raise ValueError(f"unknown method {method}")
    out = out.astype(F32)
    if preserve_luminance:
        out[..., 0] = src_lab[..., 0]
    return oklab_to_rgb(out)
