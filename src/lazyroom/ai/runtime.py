"""ONNX Runtime plumbing shared by every AI operation.

* Sessions: CPU by default.  ``LAZYROOM_AI_PROVIDER`` = ``cpu`` (default) |
  ``auto`` | ``cuda`` | ``directml`` | ``coreml`` opts into an accelerator; it is
  used only when onnxruntime reports it as available, otherwise CPU is used
  (accelerators are never required).  Sessions are cached (LRU, at most
  ``LAZYROOM_AI_MAX_SESSIONS``, default 2) because they are expensive to build.
* Cancellation: every ``session.run`` goes through :func:`run`, which watches
  ``ctx.check()`` (cancel token + deadline) from a helper thread and terminates
  the native run (``RunOptions.terminate``) — the equivalent of ImageToolbox's
  ``runCancellable``.
* Generic image-to-image models (upscalers, denoisers, de-JPEG, colourisers...)
  run through :func:`process_image`, a port of ImageToolbox's ``AiProcessor`` /
  ``ModelInfo`` / ``TileGrid`` (feature/ai-tools/.../data): same tile grid,
  edge-extension padding to a multiple of 8, strength control tensors and
  smooth-step overlap blending.
* Resource limits: output pixel cap (``LAZYROOM_AI_MAX_OUTPUT_PIXELS``,
  default 200 MP) and tile size bounds.
"""
from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..core.errors import ErrorCode, LazyRoomError

F32 = np.float32

MODEL_ALIGNMENT = 8  # AiExtensions.MODEL_ALIGNMENT
DEFAULT_CHUNK = 512  # SavableNeuralParams.Default
DEFAULT_OVERLAP = 16
DEFAULT_STRENGTH = 65.0

_PROVIDER_ALIASES = {
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
_AUTO_ORDER = ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")


def ort():
    try:
        import onnxruntime  # noqa: PLC0415

        return onnxruntime
    except ImportError as exc:  # pragma: no cover - guarded by requires=["onnxruntime"]
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, "onnxruntime is not installed",
                              hint="pip install 'lazyroom[ai]' (installs onnxruntime).") from exc


def max_output_pixels() -> int:
    try:
        return int(os.environ.get("LAZYROOM_AI_MAX_OUTPUT_PIXELS", 200_000_000))
    except ValueError:
        return 200_000_000


def max_untiled_pixels() -> int:
    try:
        return int(os.environ.get("LAZYROOM_AI_MAX_UNTILED_PIXELS", 16_000_000))
    except ValueError:
        return 16_000_000


def check_output_size(w: int, h: int, what: str = "AI output") -> None:
    if w * h > max_output_pixels():
        raise LazyRoomError(ErrorCode.RESOURCE_LIMIT,
                              f"{what} would be {w}x{h} ({w * h / 1e6:.0f} MP), above the "
                              f"{max_output_pixels() / 1e6:.0f} MP limit",
                              hint="Downscale the input first or raise LAZYROOM_AI_MAX_OUTPUT_PIXELS.")


def select_providers() -> list[str]:
    """Execution providers for new sessions (always ends with CPU)."""
    want = os.environ.get("LAZYROOM_AI_PROVIDER", "cpu").strip().lower() or "cpu"
    available = set(ort().get_available_providers())
    chosen: list[str] = []
    if want == "auto":
        chosen = [p for p in _AUTO_ORDER if p in available][:1]
    elif want in _PROVIDER_ALIASES and want != "cpu":
        p = _PROVIDER_ALIASES[want]
        if p in available:
            chosen = [p]
    return chosen + ["CPUExecutionProvider"]


def _threads() -> int:
    n = os.cpu_count() or 1
    return 1 if n <= 2 else (n * 3) // 4  # AndroidAiToolsRepository.createSession


@dataclass
class Session:
    path: Path
    sess: Any
    providers: list[str]
    inputs: list[Any] = field(default_factory=list)
    outputs: list[Any] = field(default_factory=list)

    @property
    def input_names(self) -> list[str]:
        return [i.name for i in self.inputs]

    @property
    def output_names(self) -> list[str]:
        return [o.name for o in self.outputs]


_CACHE: "OrderedDict[tuple, Session]" = OrderedDict()
_LOCK = threading.Lock()


def _max_sessions() -> int:
    try:
        return max(1, int(os.environ.get("LAZYROOM_AI_MAX_SESSIONS", 2)))
    except ValueError:
        return 2


def load_session(path: str | os.PathLike, *, opt_level: str | None = None) -> Session:
    """Create (or reuse) an inference session.  ``opt_level``: none | basic | extended | all."""
    o = ort()
    p = Path(path)
    if not p.exists():
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, f"Model file missing: {p.name}")
    providers = select_providers()
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size, tuple(providers), opt_level)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    so = o.SessionOptions()
    so.log_severity_level = 3  # errors only (old TF exports warn about initializers)
    so.intra_op_num_threads = _threads()
    so.inter_op_num_threads = 4
    levels = {"none": o.GraphOptimizationLevel.ORT_DISABLE_ALL,
              "basic": o.GraphOptimizationLevel.ORT_ENABLE_BASIC,
              "extended": o.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
              "all": o.GraphOptimizationLevel.ORT_ENABLE_ALL}
    if opt_level is None and p.suffix.lower() == ".ort":
        opt_level = "none"  # .ort files are already optimised (ITB: prevent double optimisation)
    if opt_level:
        so.graph_optimization_level = levels[opt_level]
    try:
        sess = o.InferenceSession(str(p), sess_options=so, providers=providers)
    except Exception as exc:
        raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"Could not open model {p.name}: {exc}",
                              hint="The file may be corrupt; delete it and download again.") from exc
    s = Session(path=p, sess=sess, providers=sess.get_providers(), inputs=list(sess.get_inputs()),
                outputs=list(sess.get_outputs()))
    with _LOCK:
        _CACHE[key] = s
        while len(_CACHE) > _max_sessions():
            _CACHE.popitem(last=False)
    return s


def clear_sessions() -> None:
    with _LOCK:
        _CACHE.clear()


def run(session: Session, feeds: dict[str, np.ndarray], ctx=None, output_names: Sequence[str] | None = None
        ) -> list[np.ndarray]:
    """``session.run`` that honours ``ctx`` cancellation / deadline (terminates the native run)."""
    o = ort()
    ro = o.RunOptions()
    stop = threading.Event()
    watcher = None
    if ctx is not None and getattr(ctx, "cancel", None) is not None:
        ctx.check()

        def watch() -> None:
            while not stop.wait(0.05):
                try:
                    ctx.check()
                except LazyRoomError:
                    ro.terminate = True
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
    try:
        return session.sess.run(list(output_names) if output_names else None, feeds, ro)
    except Exception as exc:
        if ctx is not None:
            ctx.check()  # raises CANCELLED / TIMEOUT if that was the cause
        raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"Inference failed ({session.path.name}): {exc}") from exc
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=1)


# ---------------------------------------------------------------------------
# Port of data/model/ImageTensor.kt + ModelInfo.kt
# ---------------------------------------------------------------------------
_FLOAT_TYPES = ("tensor(float)", "tensor(float16)")


def _dim(v: Any) -> int | None:
    return int(v) if isinstance(v, int) and v > 0 else None


@dataclass
class ImageTensor:
    name: str
    channels: int
    is_fp16: bool
    height: int | None
    width: int | None


def first_image_tensor(nodes: Sequence[Any]) -> ImageTensor | None:
    for n in nodes:
        shape = list(n.shape or [])
        if n.type in _FLOAT_TYPES and len(shape) == 4:
            return ImageTensor(name=n.name, channels=1 if shape[1] == 1 else 3,
                               is_fp16=n.type == "tensor(float16)", height=_dim(shape[2]), width=_dim(shape[3]))
    return None


# ModelInfo.scaleMap: first of x1,1x,x2,2x,...,x16,16x contained in the name
_SCALE_KEYS = [k for s in range(1, 17) for k in (f"x{s}", f"{s}x")]


def name_scale(name: str) -> int:
    for k in _SCALE_KEYS:
        if k in name:
            return int(k.strip("x"))
    return 1


@dataclass
class ModelInfo:
    """Port of ImageToolbox ``ModelInfo`` (tensor layout, tile limit, scale)."""
    name: str
    session: Session
    strength: float = DEFAULT_STRENGTH
    chunk_size: int = DEFAULT_CHUNK
    overlap: int = DEFAULT_OVERLAP
    non_chunkable: bool = False

    def __post_init__(self) -> None:
        t = first_image_tensor(self.session.inputs)
        if t is None:
            raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"{self.name}: model has no NCHW image input tensor")
        self.input = t
        self.is_scunet = self.name.startswith("scunet_")
        self.is_scunet_color = self.name.startswith("scunet_color")
        self.min_spatial = 512 if "nafnet" in self.name else 256  # spatialSizeMap
        self.scale = name_scale(self.name)

    @property
    def tile_limit(self) -> int:
        if self.non_chunkable:
            return 1 << 30
        chunk = min(self.chunk_size, 256) if self.is_scunet else self.chunk_size
        if self.input.width and self.input.height:
            return min(chunk, self.input.width, self.input.height)
        return chunk

    def tensor_size(self, w: int, h: int) -> tuple[int, int]:
        use_min = self.is_scunet_color or self.min_spatial > 256

        def side(fixed: int | None, v: int) -> int:
            if fixed:
                return fixed
            a = -(-v // MODEL_ALIGNMENT) * MODEL_ALIGNMENT
            return max(a, self.min_spatial) if use_min else a

        return side(self.input.width, w), side(self.input.height, h)

    def control_inputs(self) -> dict[str, np.ndarray]:
        """Extra float inputs of shape [1, 1] receive ``strength / 100`` (appendControlInputs)."""
        out: dict[str, np.ndarray] = {}
        for n in self.session.inputs:
            if n.name == self.input.name or n.type not in _FLOAT_TYPES:
                continue
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in (n.shape or [])]
            if shape == [1, 1]:
                dt = np.float16 if n.type == "tensor(float16)" else np.float32
                out[n.name] = np.full((1, 1), self.strength / 100.0, dtype=dt)
        return out


# ---------------------------------------------------------------------------
# Port of data/model/TileGrid.kt
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tile:
    index: int
    x: int
    y: int
    width: int
    height: int
    column: int
    row: int


@dataclass(frozen=True)
class TileGrid:
    image_width: int
    image_height: int
    tile_limit: int
    overlap: int
    step: int
    columns: int
    rows: int

    @classmethod
    def from_size(cls, image_width: int, image_height: int, tile_limit: int, overlap: int) -> "TileGrid":
        if overlap >= tile_limit:
            raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"overlap ({overlap}) must be smaller than the tile size "
                                                            f"({tile_limit})")
        step = tile_limit - overlap
        columns = 1 if image_width <= tile_limit else math.ceil((image_width - overlap) / step)
        rows = 1 if image_height <= tile_limit else math.ceil((image_height - overlap) / step)
        return cls(image_width, image_height, tile_limit, overlap, step, columns, rows)

    def tiles(self) -> list[Tile]:
        out: list[Tile] = []
        for row in range(self.rows):
            for column in range(self.columns):
                x, y = column * self.step, row * self.step
                w = min(x + self.tile_limit, self.image_width) - x
                h = min(y + self.tile_limit, self.image_height) - y
                if w > 0 and h > 0:
                    out.append(Tile(len(out), x, y, w, h, column, row))
        return out


def smooth_step_ramp(length: int) -> np.ndarray:
    """``smoothStep(position, length)`` for position = 0..length-1 (AiExtensions.kt)."""
    pos = np.arange(length, dtype=F32)
    x = np.clip(pos / max(length - 1, 1), 0, 1)
    return (x * x * (3 - 2 * x)).astype(F32)


def draw_tile(canvas: np.ndarray, tile_out: np.ndarray, tile: Tile, overlap: int, scale: int) -> None:
    """Port of ``AiProcessor.drawTile``: blend the left/top ``overlap*scale`` band with smooth-step weights."""
    tx, ty = tile.x * scale, tile.y * scale
    h, w = tile_out.shape[:2]
    blend_w = overlap * scale
    left, top = tile.column > 0, tile.row > 0
    region = canvas[ty:ty + h, tx:tx + w]
    if not left and not top:
        region[...] = tile_out
        return
    wx = np.ones(w, F32)
    wy = np.ones(h, F32)
    if left:
        n = min(blend_w, w)
        wx[:n] = smooth_step_ramp(blend_w)[:n]
    if top:
        n = min(blend_w, h)
        wy[:n] = smooth_step_ramp(blend_w)[:n]
    a = np.minimum(wy[:, None], wx[None, :])[..., None]
    region[...] = region * (1 - a) + tile_out * a


def render_tiled(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], *, tile_limit: int, overlap: int,
                 ctx=None, progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Run ``fn`` per tile and compose like ImageToolbox (``renderByTiles`` + ``composeTiles``).

    ``fn`` maps an (h, w, C) tile to an (h*s, w*s, C') result; the scale ``s`` is
    taken from the first tile's output and must be an integer.
    """
    H, W = img.shape[:2]
    if W <= tile_limit and H <= tile_limit:
        return fn(img)
    grid = TileGrid.from_size(W, H, tile_limit, overlap)
    tiles = grid.tiles()
    canvas: np.ndarray | None = None
    scale = 1
    for i, t in enumerate(tiles):
        if ctx is not None:
            ctx.check()
        out = fn(img[t.y:t.y + t.height, t.x:t.x + t.width])
        if canvas is None:
            sy, sx = out.shape[0] / t.height, out.shape[1] / t.width
            if sy != sx or sy != int(sy) or sy < 1:
                raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"Model output scale {sx}x{sy} is not an integer")
            scale = int(sy)
            check_output_size(W * scale, H * scale)
            canvas = np.zeros((H * scale, W * scale, out.shape[2]), F32)
        draw_tile(canvas, out, t, overlap, scale)
        if progress:
            progress(i + 1, len(tiles))
    assert canvas is not None
    return canvas


# ---------------------------------------------------------------------------
# Generic image -> image model (port of AiProcessor.runModelOnBitmap)
# ---------------------------------------------------------------------------
def pad_edge(x: np.ndarray, w: int, h: int) -> np.ndarray:
    """``fitToTensorSize``: extend the right column / bottom row to (w, h)."""
    ph, pw = h - x.shape[0], w - x.shape[1]
    if ph <= 0 and pw <= 0:
        return x
    return np.pad(x, ((0, max(ph, 0)), (0, max(pw, 0)), (0, 0)), mode="edge")


def to_nchw(x: np.ndarray, channels: int, fp16: bool) -> np.ndarray:
    x = np.clip(x[..., :3], 0, 1)
    if channels == 1:
        t = x.mean(axis=2, keepdims=True)  # (r + g + b) / 3
    else:
        t = x
    t = np.ascontiguousarray(t.transpose(2, 0, 1)[None])
    return t.astype(np.float16 if fp16 else np.float32)


def from_nchw(y: np.ndarray) -> np.ndarray:
    """First output -> (H, W, 3) float32 (1-channel outputs become gray)."""
    y = np.asarray(y)
    while y.ndim > 4:
        y = y[0]
    if y.ndim == 4:
        y = y[0]
    if y.ndim == 2:
        y = y[None]
    y = y.astype(np.float32)
    if y.shape[0] == 1:
        y = np.repeat(y, 3, axis=0)
    return np.ascontiguousarray(y[:3].transpose(1, 2, 0))


def run_image_model(info: ModelInfo, x: np.ndarray, ctx=None) -> np.ndarray:
    """One forward pass on an (h, w, 3) float image; returns (h*s, w*s, 3) clipped to 0..1."""
    h, w = x.shape[:2]
    tw, th = info.tensor_size(w, h)
    resized = False
    if (info.input.width and info.input.height) and (w > tw or h > th):
        # Fixed-size model asked to run on a larger, non-chunked image: ImageToolbox would
        # crop; we resize to the model size and back instead (documented deviation).
        import cv2

        x = cv2.resize(x, (tw, th), interpolation=cv2.INTER_AREA)
        resized = True
    xin = pad_edge(x, tw, th)
    feeds = {info.input.name: to_nchw(xin, info.input.channels, info.input.is_fp16)}
    feeds.update(info.control_inputs())
    y = from_nchw(run(info.session, feeds, ctx)[0])
    s = y.shape[0] / th
    if resized:
        import cv2

        y = cv2.resize(y, (w * max(1, round(s)), h * max(1, round(s))), interpolation=cv2.INTER_CUBIC)
    else:
        si = int(round(s))
        y = y[: h * si, : w * si]
    return np.clip(y, 0, 1).astype(F32)


def process_image(model_name: str, path: str | os.PathLike, img: np.ndarray, ctx=None, *,
                  strength: float = DEFAULT_STRENGTH, chunk_size: int = DEFAULT_CHUNK,
                  overlap: int = DEFAULT_OVERLAP, enable_chunking: bool = True, non_chunkable: bool = False,
                  opt_level: str | None = None) -> np.ndarray:
    """Generic ImageToolbox model pipeline.  ``img`` is (H, W, 3|4) float; alpha is carried
    through with nearest-neighbour upscaling exactly like ``renderModelOutput``."""
    sess = load_session(path, opt_level=opt_level)
    info = ModelInfo(name=model_name, session=sess, strength=strength, chunk_size=chunk_size,
                     overlap=overlap, non_chunkable=non_chunkable or not enable_chunking)
    alpha = img[..., 3] if img.ndim == 3 and img.shape[2] == 4 else None
    rgb = np.ascontiguousarray(img[..., :3])
    fixed = bool(info.input.width and info.input.height)
    if info.non_chunkable and not fixed and rgb.shape[0] * rgb.shape[1] > max_untiled_pixels():
        raise LazyRoomError(ErrorCode.RESOURCE_LIMIT,
                              f"{model_name} would run on {rgb.shape[1]}x{rgb.shape[0]} in a single pass "
                              f"(limit {max_untiled_pixels() / 1e6:.0f} MP without tiling)",
                              hint="Enable chunking, downscale first, or raise LAZYROOM_AI_MAX_UNTILED_PIXELS.")
    out = render_tiled(rgb, lambda t: run_image_model(info, t, ctx), tile_limit=info.tile_limit,
                       overlap=info.overlap, ctx=ctx)
    if alpha is not None:
        s = out.shape[0] // rgb.shape[0]
        a = np.repeat(np.repeat(alpha, s, axis=0), s, axis=1)[: out.shape[0], : out.shape[1]]
        out = np.dstack([out, a])
    return out.astype(F32)


def opt_level_for(name: str) -> str | None:
    """AndroidAiToolsRepository.createSession optimisation choices."""
    if name.endswith(".ort"):
        return "none"
    if name.startswith("fbcnn_"):
        return "extended"
    if name.startswith("scunet_"):
        return "none"
    return None
