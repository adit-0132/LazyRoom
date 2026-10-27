"""numba entry point for the texture kernels (shares LazyRoom's on-disk JIT cache)."""
from __future__ import annotations

from ..core import fast as _fast  # noqa: F401  (sets NUMBA_CACHE_DIR before numba is imported)

from numba import njit, prange  # noqa: E402

__all__ = ["njit", "prange"]
