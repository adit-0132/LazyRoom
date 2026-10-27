"""Bridge between the QML user interface and the LazyRoom API.

The GUI holds no editing logic: every change becomes an edit action on the
same project model the CLI/MCP use.  Slider drags update the step live
(without history) and are committed as one undoable edit on release.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import (QFileSystemWatcher, QObject, QRunnable, QThreadPool, QTimer, QUrl, Property, Signal,
                            Slot)
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider

from ..core.errors import LazyRoomError, wrap_exception
from ..core.image import composite_on, has_alpha
from ..core.paths import config_dir
from ..core.pipeline import GROUP_OP, Step, find_step
from ..core.project import PROJECT_SUFFIXES
from ..core.registry import CATEGORIES, FILTER_GROUPS, get_op, list_ops, search_ops
from .models import DictListModel

CATEGORY_TITLES = [("looks", "Looks", "palette"), ("adjust", "Adjust", "tune"), ("effect", "Effects", "auto_awesome"),
                   ("filter", "Filters", "filter"), ("geometry", "Transform", "crop_rotate"), ("draw", "Draw", "draw"),
                   ("generate", "Generate", "gradient"), ("ai", "AI", "psychology"), ("favorites", "Favorites", "star"),
                   ("recent", "Recent", "history")]


def to_qimage(img: np.ndarray) -> QImage:
    if has_alpha(img):
        img = composite_on(img, checker=max(6, min(img.shape[:2]) // 48))
    u8 = np.ascontiguousarray((np.clip(img[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8))
    h, w = u8.shape[:2]
    q = QImage(u8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return q.copy()  # detach from the numpy buffer


class PreviewProvider(QQuickImageProvider):
    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.ImageType.Image)
        self.images: dict[str, QImage] = {}
        self._lock = threading.Lock()

    def put(self, key: str, img: QImage) -> None:
        with self._lock:
            self.images[key] = img

    def requestImage(self, id: str, size, requestedSize):  # noqa: N802
        key = id.split("/")[0]
        with self._lock:
            img = self.images.get(key)
        if img is None:
            img = QImage(1, 1, QImage.Format.Format_RGB888)
            img.fill(0)
        if size is not None:
            size.setWidth(img.width())
            size.setHeight(img.height())
        return img


class _Signals(QObject):
    done = Signal(object, object, float, int)  # image, error, ms, ticket


class RenderTask(QRunnable):
    def __init__(self, fn, ticket: int, signals: _Signals) -> None:
        super().__init__()
        self.fn, self.ticket, self.signals = fn, ticket, signals

    def run(self) -> None:
        t0 = time.perf_counter()
        try:
            img = self.fn()
            self.signals.done.emit(img, None, (time.perf_counter() - t0) * 1000, self.ticket)
        except Exception as exc:  # reported in the status bar
            self.signals.done.emit(None, wrap_exception(exc), 0.0, self.ticket)


class _ToolSignals(QObject):
    done = Signal(object, object)  # result, error


class Controller(QObject):
    toolChanged = Signal()
    batchChanged = Signal()
    changed = Signal()
    previewChanged = Signal()
    stepsChanged = Signal()
    selectionChanged = Signal()
    statusChanged = Signal()
    themeChanged = Signal()
    aiChanged = Signal()
    toast = Signal(str)

    def __init__(self, api, provider: PreviewProvider) -> None:
        super().__init__()
        self.api = api
        self.provider = provider
        self.pid: str | None = None
        self._preview_version = 0
        self._original_version = 0
        self._rendering = False
        self._dirty = False
        self._ticket = 0
        self._render_ms = 0.0
        self._status = "Open an image to start (drag & drop works too)."
        self._selected: str | None = None
        self._preview_size = 1600
        self._dark = True
        self._ai_text = ""
        self._category = "adjust"
        self._group = ""
        self._search = ""
        self._last_save = 0.0
        self._live = False  # while dragging a slider, render a half-size preview for responsiveness
        self._paint_mode = ""
        self._pick = False
        self._brush = 0.05
        self._show_mask = False
        self._mask_version = 0
        self.pool = QThreadPool.globalInstance()
        self.signals = _Signals()
        self.signals.done.connect(self._on_rendered)
        self._m_opsModel = DictListModel(["id", "title", "group", "favorite", "available", "kind"])
        self._m_stepsModel = DictListModel(["id", "op", "title", "label", "enabled", "origin", "depth", "isGroup",
                                         "hasMask", "opacity", "summary"])
        self._m_paramsModel = DictListModel(["name", "title", "type", "min", "max", "step", "value", "choices", "unit",
                                          "description", "advanced", "fileKind"])
        self._m_historyModel = DictListModel(["id", "description", "actor", "time", "current"])
        self._m_groupsModel = DictListModel(["id", "title"])
        self._m_categoriesModel = DictListModel(["id", "title", "icon"])
        self._m_categoriesModel.set_items([{"id": c, "title": t, "icon": i} for c, t, i in CATEGORY_TITLES])
        self._m_aiModel = DictListModel(["phrase", "effect"])
        self._m_toolsModel = DictListModel(["name", "title", "category", "summary", "available"])
        self._m_toolParamsModel = DictListModel(["name", "title", "type", "min", "max", "value", "choices",
                                                 "description", "fileKind"])
        self._tool_result = ""
        self._tool_output = ""
        self._tool_busy = False
        self._batch_job: str | None = None
        self._batch_status = ""
        self._batch_progress = 0.0
        self._batch_timer = QTimer(interval=500)
        self._batch_timer.timeout.connect(self._poll_batch)
        self.tool_signals = _ToolSignals()
        self.tool_signals.done.connect(self._on_tool_done)
        self.watcher = QFileSystemWatcher()
        self.watcher.fileChanged.connect(self._on_file_changed)
        self._settings_path = config_dir() / "gui.json"
        self._settings = self._load_settings()
        self._dark = bool(self._settings.get("dark", True))
        self._debounce = QTimer(singleShot=True, interval=30)
        self._debounce.timeout.connect(self._start_render)
        self._refresh_ops()
        threading.Thread(target=_warmup, daemon=True).start()

    # ------------------------------------------------------------------ settings
    def _load_settings(self) -> dict[str, Any]:
        try:
            return json.loads(self._settings_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {"favorites": [], "recent": [], "dark": True}

    def _save_settings(self) -> None:
        try:
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._settings_path.write_text(json.dumps(self._settings, indent=1))
        except OSError:
            pass

    # ------------------------------------------------------------------ properties
    @Property(QObject, constant=True)
    def toolsModel(self):
        return self._m_toolsModel

    @Property(QObject, constant=True)
    def toolParamsModel(self):
        return self._m_toolParamsModel

    @Property(QObject, constant=True)
    def opsModel(self):
        return self._m_opsModel

    @Property(QObject, constant=True)
    def stepsModel(self):
        return self._m_stepsModel

    @Property(QObject, constant=True)
    def paramsModel(self):
        return self._m_paramsModel

    @Property(QObject, constant=True)
    def historyModel(self):
        return self._m_historyModel

    @Property(QObject, constant=True)
    def groupsModel(self):
        return self._m_groupsModel

    @Property(QObject, constant=True)
    def categoriesModel(self):
        return self._m_categoriesModel

    @Property(QObject, constant=True)
    def aiModel(self):
        return self._m_aiModel

    def _session(self):
        return self.api.ws.session(self.pid) if self.pid else None

    @Property(bool, notify=changed)
    def projectOpen(self) -> bool:
        return self.pid is not None

    @Property(str, notify=changed)
    def projectName(self) -> str:
        s = self._session()
        return s.project.name if s else "LazyRoom"

    @Property(str, notify=changed)
    def projectPath(self) -> str:
        s = self._session()
        return (s.project.path or "") if s else ""

    @Property(str, notify=changed)
    def imageInfo(self) -> str:
        s = self._session()
        if not s:
            return ""
        src = s.project.source
        return f"{src.width}×{src.height} · {src.format or ''} · {Path(src.path).name}"

    @Property(bool, notify=stepsChanged)
    def canUndo(self) -> bool:
        s = self._session()
        return bool(s and s.project.history.can_undo)

    @Property(bool, notify=stepsChanged)
    def canRedo(self) -> bool:
        s = self._session()
        return bool(s and s.project.history.can_redo)

    @Property(str, notify=previewChanged)
    def previewSource(self) -> str:
        return f"image://pm/current/{self._preview_version}" if self.pid else ""

    @Property(str, notify=previewChanged)
    def originalSource(self) -> str:
        return f"image://pm/original/{self._original_version}" if self.pid else ""

    @Property(bool, notify=statusChanged)
    def rendering(self) -> bool:
        return self._rendering

    @Property(str, notify=statusChanged)
    def status(self) -> str:
        return self._status

    @Property(str, notify=selectionChanged)
    def selectedStepId(self) -> str:
        return self._selected or ""

    @Property(str, notify=selectionChanged)
    def selectedTitle(self) -> str:
        st = self._selected_step()
        if not st:
            return ""
        if st.op == GROUP_OP:
            return st.label or "Group"
        return get_op(st.op).title

    @Property(str, notify=selectionChanged)
    def selectedDescription(self) -> str:
        st = self._selected_step()
        if not st or st.op == GROUP_OP:
            return ""
        return get_op(st.op).description[:400]

    @Property(float, notify=selectionChanged)
    def selectedOpacity(self) -> float:
        st = self._selected_step()
        return float(st.opacity) if st else 1.0

    @Property(str, notify=selectionChanged)
    def selectedBlend(self) -> str:
        st = self._selected_step()
        return st.blend if st else "normal"

    @Property(str, notify=selectionChanged)
    def selectedMask(self) -> str:
        st = self._selected_step()
        return json.dumps(st.mask) if st and st.mask else ""

    @Property(bool, notify=selectionChanged)
    def selectedMaskable(self) -> bool:
        st = self._selected_step()
        return bool(st and not st.is_geometry)

    @Property(str, notify=selectionChanged)
    def overlayMode(self) -> str:
        """Interactive canvas overlay for the selected step ("crop", "pick" or "")."""
        if self._crop_index() is not None:
            return "crop"
        return "pick" if self._pick and self._top_index() is not None else ""

    @Property(bool, notify=selectionChanged)
    def canPickNeutral(self) -> bool:
        st = self._selected_step()
        return bool(st and st.op == "adjust.white_balance")

    def _top_index(self) -> int | None:
        """Top-level index of the selected step (or of the group containing it)."""
        s, st = self._session(), self._selected_step()
        if not s or not st:
            return None

        def contains(x: Step) -> bool:
            return x.id == st.id or any(contains(c) for c in (x.steps or []))

        for i, x in enumerate(s.project.steps):
            if contains(x):
                return i
        return None

    @Slot(bool)
    def setPickNeutral(self, on: bool) -> None:
        """Eyedropper: while on, the canvas shows the image *before* the selected white-balance step."""
        self._pick = bool(on) and self.canPickNeutral
        self.selectionChanged.emit()
        self.requestRender()

    @Slot(float, float)
    def pickNeutral(self, x: float, y: float) -> None:
        s, st, idx = self._session(), self._selected_step(), self._top_index()
        if not s or not st or st.op != "adjust.white_balance" or idx is None:
            return
        from ..ops.adjust import neutral_white_balance

        with s.lock:
            steps = [Step.from_dict(t.to_dict(), validate=False) for t in s.project.steps]
        img, _, _ = s.render(max_size=self._preview_size, steps=steps, upto=idx)
        h, w = img.shape[:2]
        cx, cy = int(min(w - 1, max(0, x * w))), int(min(h - 1, max(0, y * h)))
        r = max(1, round(min(w, h) * 0.004))
        patch = img[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1, :3].reshape(-1, 3).mean(0)
        temperature, tint = neutral_white_balance(patch)
        clamped = abs(temperature) >= 100 or abs(tint) >= 100
        self._pick = False
        self._edit([{"type": "update", "step_id": st.id, "params": {"temperature": temperature, "tint": tint}}],
                   "White balance from neutral point")
        self._refresh_params()
        self.selectionChanged.emit()
        msg = f"White balance from neutral point: temperature {temperature}, tint {tint}"
        if clamped:
            msg += " — the sampled colour is far from neutral (slider limit reached); pick a grey or white area"
        self._set_status(msg)

    @Property("QVariantList", notify=selectionChanged)
    def cropRect(self) -> list:
        st = self._selected_step()
        if not st or st.op != "geometry.crop":
            return [0.0, 0.0, 1.0, 1.0]
        p = st.params
        return [float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("width", 1)), float(p.get("height", 1))]

    def _crop_index(self) -> int | None:
        """Top-level index of the selected normalized crop step (preview is rendered *before* it)."""
        s, st = self._session(), self._selected_step()
        if not s or not st or st.op != "geometry.crop" or st.params.get("units", "normalized") != "normalized":
            return None
        for i, x in enumerate(s.project.steps):
            if x.id == st.id:
                return i
        return None

    @Slot(float, float, float, float)
    def setCropLive(self, x: float, y: float, w: float, h: float) -> None:
        s, st = self._session(), self._selected_step()
        if not s or not st or st.op != "geometry.crop":
            return
        x, y = max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
        w, h = max(0.01, min(1.0 - x, w)), max(0.01, min(1.0 - y, h))
        with s.lock:
            st.params = {**st.params, "x": round(x, 5), "y": round(y, 5), "width": round(w, 5), "height": round(h, 5)}
        self._pending_commit = True
        self.selectionChanged.emit()

    # ------------------------------------------------------------------ mask painting
    maskChanged = Signal()

    @Property(str, notify=maskChanged)
    def paintMode(self) -> str:
        return self._paint_mode

    @Property(float, notify=maskChanged)
    def brushSize(self) -> float:
        return self._brush

    @Property(bool, notify=maskChanged)
    def showMask(self) -> bool:
        return self._show_mask

    @Property(str, notify=maskChanged)
    def maskSource(self) -> str:
        return f"image://pm/mask/{self._mask_version}" if self._show_mask else ""

    @Slot(str)
    def setPaintMode(self, mode: str) -> None:
        self._paint_mode = mode if mode in ("paint", "erase") and self.selectedMaskable else ""
        if self._paint_mode:
            self._show_mask = True
            self._update_mask_overlay()
        self.maskChanged.emit()

    @Slot(float)
    def setBrushSize(self, v: float) -> None:
        self._brush = float(max(0.003, min(0.4, v)))
        self.maskChanged.emit()

    @Slot(bool)
    def setShowMask(self, on: bool) -> None:
        self._show_mask = bool(on)
        self._update_mask_overlay()
        self.maskChanged.emit()

    def _brush_shape(self, st: Step) -> dict:
        mask = st.mask or {"shapes": []}
        shapes = mask.setdefault("shapes", [])
        if not shapes or shapes[-1].get("type") != "brush":
            shapes.append({"type": "brush", "strokes": []})
        st.mask = mask
        return shapes[-1]

    @Slot(float, float)
    def strokeBegin(self, x: float, y: float) -> None:
        s, st = self._session(), self._selected_step()
        if not s or not st or not self._paint_mode:
            return
        with s.lock:
            shape = self._brush_shape(st)
            shape["strokes"].append({"points": [[round(x, 4), round(y, 4)]], "radius": round(self._brush, 4),
                                     "hardness": 0.5, "erase": self._paint_mode == "erase"})
        self._pending_commit = True
        self._live = True
        self._update_mask_overlay()
        self.requestRender()

    @Slot(float, float)
    def strokeTo(self, x: float, y: float) -> None:
        s, st = self._session(), self._selected_step()
        if not s or not st or not st.mask or not self._paint_mode:
            return
        with s.lock:
            stroke = st.mask["shapes"][-1]["strokes"][-1]
            lx, ly = stroke["points"][-1]
            if (x - lx) ** 2 + (y - ly) ** 2 < (self._brush * 0.25) ** 2:
                return
            stroke["points"].append([round(x, 4), round(y, 4)])
        self._update_mask_overlay()
        self.requestRender()

    @Slot()
    def strokeEnd(self) -> None:
        st = self._selected_step()
        if not st or not st.mask:
            return
        self._live = False
        self._pending_commit = False
        self._edit([{"type": "update", "step_id": st.id, "mask": st.mask}],
                   "Paint mask" if self._paint_mode == "paint" else "Erase mask")
        self._update_mask_overlay()

    def _update_mask_overlay(self) -> None:
        s, st = self._session(), self._selected_step()
        if not self._show_mask or not s or not st or not st.mask:
            empty = QImage(1, 1, QImage.Format.Format_RGBA8888)
            empty.fill(0)
            self.provider.put("mask", empty)
        else:
            from ..core.context import RenderContext
            from ..core.masks import render_mask

            base, _ = s.proxy(min(self._preview_size, 1024))
            try:
                m = render_mask(st.mask, base, RenderContext(resolver=self.api.ws.resolve_aux_file))
            except LazyRoomError as e:
                self._set_status(f"Mask: {e.message}")
                return
            h, w = m.shape
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., 0] = 255
            rgba[..., 1] = 60
            rgba[..., 2] = 80
            rgba[..., 3] = (np.clip(m, 0, 1) * 140).astype(np.uint8)
            q = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
            self.provider.put("mask", q)
        self._mask_version += 1
        self.maskChanged.emit()

    @Property(bool, notify=themeChanged)
    def darkTheme(self) -> bool:
        return self._dark

    @Property(str, notify=aiChanged)
    def aiText(self) -> str:
        return self._ai_text

    @Property(str, notify=changed)
    def category(self) -> str:
        return self._category

    @Property(list, constant=True)
    def blendModes(self) -> list:
        from ..core.blend import BLEND_MODES

        return list(BLEND_MODES)

    @Property(list, constant=True)
    def exportFormats(self) -> list:
        return ["jpg", "png", "webp", "avif", "tif", "heic", "jp2", "bmp", "gif", "qoi"]

    # ------------------------------------------------------------------ helpers
    def _run(self, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except LazyRoomError as e:
            self._set_status(f"{e.message}" + (f" — {e.hint}" if e.hint else ""))
            self.toast.emit(e.message)
        except Exception as e:  # pragma: no cover
            err = wrap_exception(e)
            self._set_status(err.message)
            self.toast.emit(err.message)
        return None

    def _set_status(self, text: str) -> None:
        self._status = text
        self.statusChanged.emit()

    def _selected_step(self) -> Step | None:
        s = self._session()
        if not s or not self._selected:
            return None
        try:
            lst, i = find_step(s.project.steps, self._selected)
            return lst[i]
        except LazyRoomError:
            return None

    # ------------------------------------------------------------------ catalogue
    @Slot(str)
    def setCategory(self, cat: str) -> None:
        self._category = cat
        self._group = ""
        if cat == "filter":
            self._m_groupsModel.set_items([{"id": "", "title": "All"}] +
                                       [{"id": g, "title": t} for g, t in FILTER_GROUPS.items()])
        else:
            self._m_groupsModel.set_items([])
        self._refresh_ops()
        self.changed.emit()

    @Slot(str)
    def setGroup(self, group: str) -> None:
        self._group = group
        self._refresh_ops()

    @Slot(str)
    def setSearch(self, text: str) -> None:
        self._search = text.strip()
        self._refresh_ops()

    def _refresh_ops(self) -> None:
        fav = set(self._settings.get("favorites", []))
        items: list[dict[str, Any]] = []
        if self._category == "looks" and not self._search:
            for p in self.api.presets_list()["presets"]:
                items.append({"id": p["id"], "title": p["name"], "group": p["kind"], "favorite": False,
                              "available": True, "kind": "preset"})
            self._m_opsModel.set_items(items)
            return
        if self._search:
            specs = search_ops(self._search, limit=200)
        elif self._category == "favorites":
            specs = [get_op(i) for i in self._settings.get("favorites", []) if _safe_op(i)]
        elif self._category == "recent":
            specs = [get_op(i) for i in self._settings.get("recent", []) if _safe_op(i)]
        else:
            specs = list_ops(self._category if self._category in CATEGORIES else None, self._group or None)
            specs.sort(key=lambda s: s.title.lower())
        for s in specs:
            items.append({"id": s.id, "title": s.title, "group": s.group or s.category, "favorite": s.id in fav,
                          "available": s.availability()[0], "kind": "op"})
        self._m_opsModel.set_items(items)

    @Slot(str)
    def toggleFavorite(self, op_id: str) -> None:
        favs = self._settings.setdefault("favorites", [])
        if op_id in favs:
            favs.remove(op_id)
        else:
            favs.append(op_id)
        self._save_settings()
        self._refresh_ops()

    # ------------------------------------------------------------------ files
    @Slot(str)
    def openFile(self, url: str) -> None:
        path = QUrl(url).toLocalFile() if url.startswith("file:") else url
        if not path:
            return
        if self.pid:
            self.api.project_close(self.pid)
            self.pid = None
        if path.lower().endswith(PROJECT_SUFFIXES):
            res = self._run(self.api.project_open, path)
        else:
            res = self._run(self.api.project_create, path)
        if not res:
            return
        self.pid = res["project_id"]
        self._selected = None
        self._watch()
        for w in res.get("warnings", []):
            self.toast.emit(w)
        self._set_status(f"Opened {Path(path).name}")
        self._after_change(original=True)
        self.changed.emit()

    def _watch(self) -> None:
        for f in self.watcher.files():
            self.watcher.removePath(f)
        s = self._session()
        if s and s.project.path:
            self.watcher.addPath(s.project.path)

    def _on_file_changed(self, path: str) -> None:
        if time.time() - self._last_save < 2.0:
            self._watch()
            return
        s = self._session()
        if not s or not s.project.path:
            return
        QTimer.singleShot(300, self._reload_from_disk)

    @Slot()
    def _reload_from_disk(self) -> None:
        s = self._session()
        if not s or not s.project.path or not Path(s.project.path).exists():
            return
        from ..core.project import Project

        try:
            fresh = Project.load(s.project.path)
        except LazyRoomError as e:
            self._set_status(f"Project changed on disk but could not be reloaded: {e.message}")
            return
        with s.lock:
            s.project.steps = fresh.steps
            s.project.history = fresh.history
        self._watch()
        self.toast.emit("Project updated from disk (edited by another app or agent)")
        self._after_change()

    @Slot(str)
    def saveProject(self, url: str = "") -> None:
        if not self.pid:
            return
        path = QUrl(url).toLocalFile() if url.startswith("file:") else url
        self._last_save = time.time()
        res = self._run(self.api.project_save, self.pid, path or None)
        if res:
            self._set_status(f"Saved {res['path']}")
            self._watch()
            self.changed.emit()

    @Slot(str, str, int, str, bool)
    def exportImage(self, url: str, fmt: str, quality: int, metadata: str, overwrite: bool) -> None:
        if not self.pid:
            return
        path = QUrl(url).toLocalFile() if url.startswith("file:") else url
        self._set_status("Exporting…")

        def work():
            return self.api.export(self.pid, path, format=fmt or None, quality=quality, metadata=metadata,
                                   overwrite=overwrite)

        res = self._run(work)
        if res:
            self._set_status(f"Exported {res['output']} ({res['width']}×{res['height']}, {res['bytes'] // 1024} KB)")
            self.toast.emit("Export complete")

    @Slot("QVariant", result=str)
    def urlToPath(self, url) -> str:
        u = url if isinstance(url, QUrl) else QUrl(str(url))
        return u.toLocalFile() if u.isLocalFile() else str(url)

    @Slot(result=str)
    def suggestedExportName(self) -> str:
        s = self._session()
        if not s:
            return ""
        p = Path(s.project.source.path)
        return str(p.with_name(p.stem + "-edit.jpg"))

    # ------------------------------------------------------------------ editing
    def _edit(self, actions: list[dict[str, Any]], desc: str | None = None) -> dict | None:
        if not self.pid:
            return None
        res = self._run(self.api.project_edit, self.pid, actions, actor="user", description=desc)
        if res is not None:
            self._after_change()
        return res

    @Slot(str)
    def addOperation(self, op_id: str) -> None:
        if not self.pid:
            self.toast.emit("Open an image first")
            return
        res = self._edit([{"type": "add", "op": op_id, "params": {}}], f"Add {op_id}")
        if res and res["created_step_ids"]:
            recent = [r for r in self._settings.get("recent", []) if r != op_id]
            self._settings["recent"] = [op_id] + recent[:29]
            self._save_settings()
            self.selectStep(res["created_step_ids"][0])

    @Slot(str, float)
    def applyPreset(self, preset_id: str, strength: float = 1.0) -> None:
        if not self.pid:
            self.toast.emit("Open an image first")
            return
        res = self._run(self.api.preset_apply, self.pid, preset_id, strength=strength)
        if res:
            self._after_change()
            if res.get("created_step_ids"):
                self.selectStep(res["created_step_ids"][0])

    @Slot(str)
    def selectStep(self, step_id: str) -> None:
        was_crop = self._crop_index() is not None or self._pick
        self._selected = step_id or None
        self._paint_mode = ""
        self._pick = False
        self._update_mask_overlay()
        self._refresh_params()
        self.selectionChanged.emit()
        self._refresh_steps()
        if was_crop or self._crop_index() is not None:
            self.requestRender()

    @Slot(str)
    def removeStep(self, step_id: str) -> None:
        if self._edit([{"type": "remove", "step_id": step_id}]) is not None and self._selected == step_id:
            self.selectStep("")

    @Slot(str)
    def toggleStep(self, step_id: str) -> None:
        self._edit([{"type": "toggle", "step_id": step_id}])

    @Slot(str)
    def duplicateStep(self, step_id: str) -> None:
        self._edit([{"type": "duplicate", "step_id": step_id}])

    @Slot(str, int)
    def moveStep(self, step_id: str, delta: int) -> None:
        s = self._session()
        if not s:
            return
        lst, i = find_step(s.project.steps, step_id)
        self._edit([{"type": "move", "step_id": step_id, "index": max(0, min(len(lst) - 1, i + delta))}])

    @Slot(str, int)
    def moveStepToRow(self, step_id: str, row: int) -> None:
        """Drag-and-drop reorder: move a step to where the (flattened) list row ``row`` is, within its own level."""
        s = self._session()
        if not s:
            return
        ids = [it.get("id") for it in self._m_stepsModel.items()]
        if not ids or step_id not in ids:
            return
        target = ids[max(0, min(len(ids) - 1, row))]
        lst, i = find_step(s.project.steps, step_id)

        def holds(x: Step, sid: str) -> bool:
            return x.id == sid or any(holds(c, sid) for c in (x.steps or []))

        j = next((k for k, x in enumerate(lst) if holds(x, target)), None)
        if j is None or j == i:
            return
        self._edit([{"type": "move", "step_id": step_id, "index": j}], "Reorder steps")

    @Slot()
    def undo(self) -> None:
        if self.pid and self._run(self.api.project_undo, self.pid) is not None:
            self._after_change()

    @Slot()
    def redo(self) -> None:
        if self.pid and self._run(self.api.project_redo, self.pid) is not None:
            self._after_change()

    @Slot(str)
    def revertTo(self, history_id: str) -> None:
        if self.pid and self._run(self.api.project_revert, self.pid, history_id) is not None:
            self._after_change()

    @Slot()
    def clearSteps(self) -> None:
        self._edit([{"type": "clear"}], "Clear all edits")
        self.selectStep("")

    # live parameter editing ------------------------------------------------
    @Slot(str, "QVariant")
    def setParamLive(self, name: str, value: Any) -> None:
        """Update the selected step immediately (no history entry) and re-render."""
        s = self._session()
        st = self._selected_step()
        if not s or not st or st.op == GROUP_OP:
            return
        params = dict(st.params)
        params[name] = _from_qml(value)
        try:
            spec = get_op(st.op)
            values = spec.validate(params)
        except LazyRoomError as e:
            self._set_status(e.message)
            return
        with s.lock:
            st.params = spec.params_json(values)
        self._pending_commit = True
        self._live = True
        self.requestRender()

    @Slot(str, "QVariant")
    def setParam(self, name: str, value: Any) -> None:
        """Set a parameter and commit it as one undoable edit."""
        st = self._selected_step()
        if not st:
            return
        self._edit([{"type": "update", "step_id": st.id, "params": {name: _from_qml(value)}}],
                   f"Adjust {name.replace('_', ' ')}")
        self._refresh_params()

    @Slot()
    def commitParams(self) -> None:
        st = self._selected_step()
        if not st or not getattr(self, "_pending_commit", False):
            return
        self._pending_commit = False
        self._live = False
        self._edit([{"type": "update", "step_id": st.id, "params": dict(st.params), "replace_params": True}],
                   f"Adjust {self.selectedTitle}")
        self._refresh_params()

    @Slot(float)
    def setOpacityLive(self, v: float) -> None:
        s, st = self._session(), self._selected_step()
        if s and st:
            with s.lock:
                st.opacity = float(max(0.0, min(1.0, v)))
            self._pending_commit = True
            self._live = True
            self.requestRender()

    @Slot()
    def commitOpacity(self) -> None:
        st = self._selected_step()
        self._live = False
        if st:
            self._edit([{"type": "update", "step_id": st.id, "opacity": st.opacity}], "Change opacity")

    @Slot(str)
    def setBlend(self, mode: str) -> None:
        st = self._selected_step()
        if st:
            self._edit([{"type": "update", "step_id": st.id, "blend": mode}], f"Blend {mode}")

    @Slot(str)
    def setMaskPreset(self, kind: str) -> None:
        st = self._selected_step()
        if not st:
            return
        presets = {
            "none": None,
            "radial": {"shapes": [{"type": "radial", "center": [0.5, 0.5], "radius": [0.35, 0.35], "feather": 0.6}]},
            "radial_inverted": {"shapes": [{"type": "radial", "center": [0.5, 0.5], "radius": [0.35, 0.35],
                                            "feather": 0.6}], "invert": True},
            "top_gradient": {"shapes": [{"type": "linear", "start": [0.5, 0.55], "end": [0.5, 0.0]}]},
            "bottom_gradient": {"shapes": [{"type": "linear", "start": [0.5, 0.45], "end": [0.5, 1.0]}]},
            "shadows": {"shapes": [{"type": "luminance", "min": 0.0, "max": 0.35, "softness": 0.15}]},
            "highlights": {"shapes": [{"type": "luminance", "min": 0.65, "max": 1.0, "softness": 0.15}]},
            "skin": {"shapes": [{"type": "skin"}]},
            "not_skin": {"shapes": [{"type": "all"}, {"type": "skin", "mode": "subtract"}]},
        }
        self._edit([{"type": "update", "step_id": st.id, "mask": presets.get(kind)}], f"Mask: {kind}")

    @Slot(str)
    def setMaskJson(self, text: str) -> None:
        st = self._selected_step()
        if not st:
            return
        try:
            mask = json.loads(text) if text.strip() else None
        except json.JSONDecodeError as e:
            self.toast.emit(f"Invalid mask JSON: {e}")
            return
        self._edit([{"type": "update", "step_id": st.id, "mask": mask}], "Edit mask")

    # ------------------------------------------------------------------ AI panel
    @Slot(str)
    def runIntent(self, text: str) -> None:
        if not self.pid:
            self.toast.emit("Open an image first")
            return
        res = self._run(self.api.intent_plan, text, project_id=self.pid, apply=True)
        if not res:
            return
        self._m_aiModel.set_items([{"phrase": i["phrase"], "effect": i["effect"]} for i in res["interpretation"]])
        notes = res.get("notes", [])
        self._ai_text = (f"Engine: {res['engine']}. " + " ".join(notes)).strip()
        self.aiChanged.emit()
        self._after_change()

    @Slot(str)
    def matchReference(self, url: str) -> None:
        if not self.pid:
            return
        path = QUrl(url).toLocalFile() if url.startswith("file:") else url
        self._set_status("Matching reference style…")
        res = self._run(self.api.style_match, self.pid, path)
        if not res:
            return
        rep = res["report"]
        self._m_aiModel.set_items([{"phrase": "similarity", "effect": f"{rep['similarity']} ({rep['confidence']})"}] +
                               [{"phrase": "limitation", "effect": l} for l in rep["limitations"]])
        self._ai_text = f"Matched {Path(path).name}: loss {rep['loss_initial']} → {rep['loss_final']}"
        self.aiChanged.emit()
        self._after_change()

    # ------------------------------------------------------------------ view
    @Slot()
    def toggleTheme(self) -> None:
        self._dark = not self._dark
        self._settings["dark"] = self._dark
        self._save_settings()
        self.themeChanged.emit()

    @Slot(int)
    def setPreviewSize(self, px: int) -> None:
        px = int(max(512, min(4096, px)))
        if abs(px - self._preview_size) > 64:
            self._preview_size = px
            if self.pid:
                self._after_change(original=True)

    # ------------------------------------------------------------------ utility tools
    @Property(str, notify=toolChanged)
    def toolResult(self) -> str:
        return self._tool_result

    @Property(str, notify=toolChanged)
    def toolOutput(self) -> str:
        return QUrl.fromLocalFile(self._tool_output).toString() if self._tool_output else ""

    @Property(bool, notify=toolChanged)
    def toolBusy(self) -> bool:
        return self._tool_busy

    @Slot()
    def refreshTools(self) -> None:
        tools = self.api.tools_list()["tools"]
        self._m_toolsModel.set_items([{**t, "available": t.get("available", True)} for t in tools])

    @Slot(str)
    def selectTool(self, name: str) -> None:
        d = self._run(self.api.tool_describe, name)
        if not d:
            return
        rows = []
        for p in d["params"]:
            default = p.get("default")
            if p["name"] in ("input", "a") and self.pid:
                default = self.api.ws.session(self.pid).project.source.path
            rows.append({"name": p["name"], "title": p.get("title") or p["name"].replace("_", " ").capitalize(),
                         "type": p["type"], "min": p.get("min"), "max": p.get("max"),
                         "value": default if not isinstance(default, (list, dict)) else json.dumps(default),
                         "choices": p.get("choices", []), "description": p.get("description", ""),
                         "fileKind": p.get("file_kind", "")})
        self._m_toolParamsModel.set_items(rows)
        self._tool_result, self._tool_output = d.get("description", ""), ""
        self.toolChanged.emit()

    @Slot(str, str)
    def runTool(self, name: str, params_json: str) -> None:
        try:
            raw = json.loads(params_json or "{}")
        except json.JSONDecodeError as e:
            self.toast.emit(f"Invalid parameters: {e}")
            return
        params: dict[str, Any] = {}
        for row in self._m_toolParamsModel.items():
            if row["name"] not in raw:
                continue
            v = raw[row["name"]]
            if row["type"] in ("string_list", "points", "float_list", "gradient", "point") and isinstance(v, str):
                v = [x.strip() for x in v.split(",") if x.strip()] if row["type"] == "string_list" and not v.strip().startswith("[") \
                    else json.loads(v)
            if v == "" and row["type"] in ("file", "text") and row["name"] not in ("output",):
                continue
            params[row["name"]] = v
        self._tool_busy = True
        self.toolChanged.emit()
        api, sig = self.api, self.tool_signals

        def work():
            try:
                sig.done.emit(api.tool_run(name, params), None)
            except Exception as exc:
                sig.done.emit(None, wrap_exception(exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_tool_done(self, res, err) -> None:
        self._tool_busy = False
        if err is not None:
            self._tool_result = json.dumps({"error": err.to_dict()}, indent=1)
            self._tool_output = ""
        else:
            self._tool_result = json.dumps(res, indent=1, default=str)[:20000]
            out = ""
            for key in ("output", "path", "swatch", "diff"):
                v = res.get(key)
                if isinstance(v, dict):
                    v = v.get("path")
                if isinstance(v, str) and Path(v).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
                    out = v
                    break
            self._tool_output = out
        self.toolChanged.emit()

    # ------------------------------------------------------------------ batch
    @Property(str, notify=batchChanged)
    def batchStatus(self) -> str:
        return self._batch_status

    @Property(float, notify=batchChanged)
    def batchProgress(self) -> float:
        return self._batch_progress

    @Property(bool, notify=batchChanged)
    def batchRunning(self) -> bool:
        return self._batch_job is not None

    @Slot(str, str, str, str, str, str, bool)
    def startBatch(self, input_dir: str, output_dir: str, preset: str, naming: str, fmt: str, normalize: str,
                   recursive: bool) -> None:
        conv = lambda u: QUrl(u).toLocalFile() if u.startswith("file:") else u
        kwargs: dict[str, Any] = {"recursive": recursive, "naming": naming or "{stem}{ext}",
                                  "format": fmt or None, "normalize": normalize or "none"}
        if preset == "__project__":
            if not self.pid:
                self.toast.emit("Open a photo and build an edit first, or choose a look")
                return
            kwargs["project_id"] = self.pid
        elif preset:
            kwargs["preset"] = preset
        res = self._run(self.api.batch_start, [conv(input_dir)], conv(output_dir), **kwargs)
        if res:
            self._batch_job = res["job_id"]
            self._batch_status = f"Started: {res['total']} images"
            self._batch_progress = 0.0
            self._batch_timer.start()
            self.batchChanged.emit()

    def _poll_batch(self) -> None:
        if not self._batch_job:
            self._batch_timer.stop()
            return
        st = self.api.job_status(self._batch_job)
        self._batch_progress = st["progress"]
        self._batch_status = (f"{st['status']}: {st['succeeded']} done, {st['failed']} failed, "
                              f"{st['skipped']} skipped of {st['total']}")
        if st["status"] in ("completed", "partial", "failed", "cancelled"):
            self._batch_timer.stop()
            if st.get("report_path"):
                self._batch_status += f" — report: {st['report_path']}"
            self._batch_job = None
            self.toast.emit("Batch " + st["status"])
        self.batchChanged.emit()

    @Slot()
    def cancelBatch(self) -> None:
        if self._batch_job:
            self.api.job_cancel(self._batch_job)

    @Slot(result=list)
    def presetChoices(self) -> list:
        return [{"id": p["id"], "name": p["name"]} for p in self.api.presets_list()["presets"]]

    # ------------------------------------------------------------------ settings
    @Slot(result=str)
    def settingsJson(self) -> str:
        cfg = self.api.ws.config
        return json.dumps({"llm_provider": cfg.llm.provider, "llm_base_url": cfg.llm.base_url,
                           "llm_model": cfg.llm.model, "llm_api_key_env": cfg.llm.api_key_env,
                           "llm_send_image": cfg.llm.send_preview_image, "approval_policy": cfg.approval_policy,
                           "read_roots": cfg.read_roots, "write_roots": cfg.write_roots,
                           "preview_size": cfg.preview_size, "backends": self.api.backends_status()["backends"]})

    @Slot(str)
    def saveSettings(self, text: str) -> None:
        try:
            d = json.loads(text)
        except json.JSONDecodeError as e:
            self.toast.emit(f"Invalid settings: {e}")
            return
        cfg = self.api.ws.config
        cfg.llm.provider = d.get("llm_provider", cfg.llm.provider) or "none"
        cfg.llm.base_url = d.get("llm_base_url", cfg.llm.base_url)
        cfg.llm.model = d.get("llm_model", cfg.llm.model)
        cfg.llm.api_key_env = d.get("llm_api_key_env", cfg.llm.api_key_env)
        cfg.llm.send_preview_image = bool(d.get("llm_send_image", cfg.llm.send_preview_image))
        cfg.approval_policy = d.get("approval_policy", cfg.approval_policy)
        roots = lambda v: [r.strip() for r in v.split(";") if r.strip()] if isinstance(v, str) else list(v or [])
        cfg.read_roots = roots(d.get("read_roots", cfg.read_roots))
        cfg.write_roots = roots(d.get("write_roots", cfg.write_roots))
        path = cfg.save()
        from ..app.security import PathPolicy

        self.api.ws.policy = PathPolicy(cfg.read_roots, cfg.write_roots)
        for sess in self.api.ws.sessions.values():
            self.api.ws.policy.protect(sess.project.source.path)
        self.toast.emit(f"Settings saved ({path})")

    # ------------------------------------------------------------------ refresh / render
    def _after_change(self, original: bool = False) -> None:
        self._refresh_steps()
        self._refresh_params()
        self._refresh_history()
        self.stepsChanged.emit()
        self.selectionChanged.emit()
        if original:
            self._render_original()
        self.requestRender()

    def _refresh_steps(self) -> None:
        s = self._session()
        rows: list[dict[str, Any]] = []
        if s:
            def walk(steps, depth):
                for st in steps:
                    title = st.label or (get_op(st.op).title if st.op != GROUP_OP else "Group")
                    summary = "" if st.op == GROUP_OP else ", ".join(
                        f"{k} {v}" for k, v in list(st.params.items())[:3] if not isinstance(v, (list, dict)))
                    rows.append({"id": st.id, "op": st.op, "title": title, "label": st.label or "",
                                 "enabled": st.enabled, "origin": st.origin, "depth": depth,
                                 "isGroup": st.op == GROUP_OP, "hasMask": bool(st.mask), "opacity": st.opacity,
                                 "summary": summary[:80]})
                    if st.op == GROUP_OP:
                        walk(st.steps or [], depth + 1)
            walk(s.project.steps, 0)
        self._m_stepsModel.set_items(rows)

    def _refresh_params(self) -> None:
        st = self._selected_step()
        if not st or st.op == GROUP_OP:
            self._m_paramsModel.set_items([])
            return
        spec = get_op(st.op)
        items = []
        for p in spec.params:
            d = p.describe()
            items.append({"name": p.name, "title": p.label, "type": p.kind, "min": d.get("min"), "max": d.get("max"),
                          "step": d.get("step"), "value": st.params.get(p.name, d["default"]),
                          "choices": d.get("choices", []), "unit": p.unit or "", "description": p.description,
                          "advanced": p.advanced, "fileKind": d.get("file_kind", "")})
        self._m_paramsModel.set_items(items)

    def _refresh_history(self) -> None:
        s = self._session()
        if not s:
            self._m_historyModel.set_items([])
            return
        h = s.project.history
        rows = [{"id": e.id, "description": e.description, "actor": e.actor,
                 "time": time.strftime("%H:%M:%S", time.localtime(e.time)), "current": i == h.cursor}
                for i, e in enumerate(h.entries)]
        self._m_historyModel.set_items(list(reversed(rows[-200:])))

    @Slot()
    def requestRender(self) -> None:
        if not self.pid:
            return
        self._dirty = True
        if not self._rendering:
            self._debounce.start()

    def _start_render(self) -> None:
        if not self.pid or self._rendering:
            return
        self._dirty = False
        self._rendering = True
        self._ticket += 1
        self.statusChanged.emit()
        pid = self.pid
        size = max(384, self._preview_size // 2) if self._live else self._preview_size
        api = self.api

        upto = self._crop_index()
        if upto is None and self._pick:
            upto = self._top_index()

        def job():
            sess = api.ws.session(pid)
            with sess.lock:
                steps = [Step.from_dict(s.to_dict(), validate=False) for s in sess.project.steps]
            img, _, _ = sess.render(max_size=size, steps=steps, upto=upto)
            return img

        self.pool.start(RenderTask(job, self._ticket, self.signals))

    def _on_rendered(self, img, err, ms: float, ticket: int) -> None:
        self._rendering = False
        if err is not None:
            self._set_status(f"Render failed: {err.message}")
        elif img is not None:
            self.provider.put("current", to_qimage(img))
            self._preview_version += 1
            self._render_ms = ms
            self._status = f"Rendered in {ms:.0f} ms · {img.shape[1]}×{img.shape[0]} preview"
            self.previewChanged.emit()
        self.statusChanged.emit()
        if self._dirty:
            self._debounce.start()

    def _render_original(self) -> None:
        s = self._session()
        if not s:
            return
        img, _ = s.proxy(self._preview_size)
        self.provider.put("original", to_qimage(img))
        self._original_version += 1
        self.previewChanged.emit()


def _warmup() -> None:
    """Load/compile the numba kernels off the UI thread so the first preview is fast."""
    try:
        from ..core import fast

        fast.warmup()
    except Exception:
        pass


def _safe_op(op_id: str) -> bool:
    try:
        get_op(op_id)
        return True
    except LazyRoomError:
        return False


def _from_qml(v: Any) -> Any:
    """Convert QJSValue/QVariant containers to plain Python."""
    if hasattr(v, "toVariant"):
        v = v.toVariant()
    if isinstance(v, (list, tuple)):
        return [_from_qml(x) for x in v]
    if isinstance(v, dict):
        return {k: _from_qml(x) for k, x in v.items()}
    return v
