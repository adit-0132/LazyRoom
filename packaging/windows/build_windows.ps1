# Build LazyRoom for Windows: PyInstaller folder + portable ZIP + Inno Setup installer.
# Usage (PowerShell, from the repository root):
#   py -3.13 -m venv .venv; .\.venv\Scripts\pip install -e ".[all]" pyinstaller
#   $env:LAZYROOM_BUNDLE_GMIC = "C:\path\to\gmic-cli-folder"   # optional: ship the G'MIC CLI
#   .\packaging\windows\build_windows.ps1
$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Root
$Py = if ($env:PYTHON) { $env:PYTHON } else { ".\.venv\Scripts\python.exe" }
$Version = & $Py -c "import sys; sys.path.insert(0, 'src'); import lazyroom; print(lazyroom.__version__)"
& $Py -m PyInstaller packaging\lazyroom.spec --noconfirm --distpath dist --workpath build
Copy-Item LICENSE, NOTICE dist\LazyRoom\
Compress-Archive -Path dist\LazyRoom\* -DestinationPath "dist\LazyRoom-$Version-windows-x64-portable.zip" -Force
$Iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (Test-Path $Iscc) {
  & $Iscc "/DAppVersion=$Version" packaging\windows\lazyroom.iss
} else {
  Write-Warning "Inno Setup 6 not found; skipped the installer (choco install innosetup)"
}
