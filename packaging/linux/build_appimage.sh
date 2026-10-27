#!/usr/bin/env bash
# Build LazyRoom-<version>-x86_64.AppImage from a PyInstaller build.
# Requirements: the project venv with pyinstaller, appimagetool (downloaded to .tools/ if missing).
# Optional: LAZYROOM_BUNDLE_GMIC=<folder with a portable gmic CLI> to ship G'MIC inside the image.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
VERSION="$("$PY" -c 'import sys; sys.path.insert(0, "'"$ROOT"'/src"); import lazyroom; print(lazyroom.__version__)')"
cd "$ROOT"
if [ "${SKIP_PYINSTALLER:-0}" != "1" ]; then
  "$PY" -m PyInstaller packaging/lazyroom.spec --noconfirm --distpath dist --workpath build
fi
APPDIR="$ROOT/build/AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/lib" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -a dist/LazyRoom "$APPDIR/usr/lib/lazyroom"
cp packaging/linux/AppRun "$APPDIR/AppRun"
cp packaging/linux/lazyroom.desktop "$APPDIR/lazyroom.desktop"
cp packaging/linux/lazyroom.desktop "$APPDIR/usr/share/applications/"
cp packaging/icon-256.png "$APPDIR/lazyroom.png"
cp packaging/icon-256.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/lazyroom.png"
cp LICENSE NOTICE "$APPDIR/usr/lib/lazyroom/"
TOOL="${APPIMAGETOOL:-$ROOT/.tools/appimagetool}"
if [ ! -x "$TOOL" ]; then
  mkdir -p "$ROOT/.tools"
  curl -sSfL -o "$TOOL" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$TOOL"
fi
OUT="$ROOT/dist/LazyRoom-$VERSION-x86_64.AppImage"
ARCH=x86_64 "$TOOL" --appimage-extract-and-run --no-appstream "$APPDIR" "$OUT"
echo "built $OUT"
