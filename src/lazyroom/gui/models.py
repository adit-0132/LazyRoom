"""Generic Qt list models fed from Python dicts."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractListModel, QByteArray, QModelIndex, Qt, Signal, Property


class DictListModel(QAbstractListModel):
    """A list model whose rows are dicts; roles are the dict keys declared at construction."""

    countChanged = Signal()

    def __init__(self, roles: list[str], parent=None) -> None:
        super().__init__(parent)
        self._roles = list(roles)
        self._items: list[dict[str, Any]] = []

    def roleNames(self):  # noqa: N802 (Qt API)
        return {Qt.UserRole + 1 + i: QByteArray(r.encode()) for i, r in enumerate(self._roles)}

    def rowCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self._items)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._items)):
            return None
        i = role - Qt.UserRole - 1
        if role == Qt.DisplayRole:
            i = 0
        if 0 <= i < len(self._roles):
            return self._items[index.row()].get(self._roles[i])
        return None

    def set_items(self, items: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._items = list(items)
        self.endResetModel()
        self.countChanged.emit()

    def update_item(self, row: int, item: dict[str, Any]) -> None:
        if 0 <= row < len(self._items):
            self._items[row] = item
            idx = self.index(row)
            self.dataChanged.emit(idx, idx)

    def items(self) -> list[dict[str, Any]]:
        return self._items

    def _count(self) -> int:
        return len(self._items)

    count = Property(int, _count, notify=countChanged)
