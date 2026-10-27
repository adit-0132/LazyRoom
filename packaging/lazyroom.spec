# PyInstaller spec: one folder, two executables sharing the same runtime.
#   lazyroom      console CLI (also `lazyroom mcp` for the MCP server)
#   LazyRoom      windowed desktop GUI (no console window on Windows)
# Build:  pyinstaller packaging/lazyroom.spec --noconfirm --distpath dist --workpath build
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

# mcp.cli is the optional `mcp` developer CLI (needs typer, from mcp[cli]); LazyRoom never imports it
hidden = (collect_submodules("lazyroom") + collect_submodules("mcp", filter=lambda m: not m.startswith("mcp.cli"))
          + collect_submodules("mcp_types"))
hidden += ["PySide6.QtQuickControls2", "PySide6.QtSvg", "zxingcpp", "segno", "pypdf", "piexif"]
datas = collect_data_files("lazyroom", include_py_files=False)
datas += collect_data_files("mcp") + collect_data_files("mcp_types")
gmic_dir = os.environ.get("LAZYROOM_BUNDLE_GMIC")  # optional: folder with the G'MIC CLI to ship in bin/
binaries = []
if gmic_dir and os.path.isdir(gmic_dir):
    for f in os.listdir(gmic_dir):
        binaries.append((os.path.join(gmic_dir, f), "bin"))

common = dict(
    pathex=[SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyQt5", "PyQt6"],
    # numba's on-disk cache needs real .py files, so ship our package as source instead of in the PYZ archive
    module_collection_mode={"lazyroom": "py"},
    noarchive=False,
)
a_cli = Analysis([os.path.join(ROOT, "packaging", "entry_cli.py")], **common)
a_gui = Analysis([os.path.join(ROOT, "packaging", "entry_gui.py")], **common)
icon = os.path.join(ROOT, "packaging", "lazyroom.ico") if sys.platform == "win32" else None
exe_cli = EXE(PYZ(a_cli.pure), a_cli.scripts, [], exclude_binaries=True, name="lazyroom", console=True, icon=icon)
exe_gui = EXE(PYZ(a_gui.pure), a_gui.scripts, [], exclude_binaries=True, name="LazyRoom", console=False, icon=icon)
coll = COLLECT(exe_cli, exe_gui, a_cli.binaries, a_cli.datas, a_gui.binaries, a_gui.datas, name="LazyRoom")
