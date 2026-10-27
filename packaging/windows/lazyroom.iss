; Inno Setup script for LazyRoom (per-user install by default, no admin rights needed).
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
[Setup]
AppId={{6E2B0C2A-6B6E-4C0F-9D7F-3A1F3C1B9A55}
AppName=LazyRoom
AppVersion={#AppVersion}
AppPublisher=LazyRoom contributors
DefaultDirName={autopf}\LazyRoom
DefaultGroupName=LazyRoom
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\..\dist
OutputBaseFilename=LazyRoom-{#AppVersion}-windows-x64-setup
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\lazyroom.ico
UninstallDisplayIcon={app}\lazyroom-gui.exe
LicenseFile=..\..\LICENSE
ChangesEnvironment=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; Flags: unchecked
Name: "addtopath"; Description: "Add the lazyroom command (CLI / MCP server) to PATH"; Flags: unchecked

[Files]
Source: "..\..\dist\LazyRoom\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\LazyRoom"; Filename: "{app}\lazyroom-gui.exe"
Name: "{autodesktop}\LazyRoom"; Filename: "{app}\lazyroom-gui.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\lazyroom-gui.exe"; Description: "Launch LazyRoom"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsAddPath(Param: string): boolean;
var OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then begin Result := True; exit; end;
  Result := Pos(';' + Uppercase(Param) + ';', ';' + Uppercase(OrigPath) + ';') = 0;
end;
