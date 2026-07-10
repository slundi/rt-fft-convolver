; Inno Setup script for the rt-ir plugins (Windows installer).
; Compile with: iscc /DMyAppVersion=0.1.0 packaging\windows\rt-ir-plugins.iss
; Expects the bundles built into target\bundled (run `just bundle-all` first).

#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif

[Setup]
AppId={{9C1D6E2A-4B3F-4E77-9A21-RTIRCABREV01}}
AppName=RT IR Plugins
AppVersion={#MyAppVersion}
AppPublisher=Sébastien Lundi
DefaultDirName={autopf}\RT IR Plugins
DisableDirPage=yes
DisableProgramGroupPage=yes
OutputDir=..\..\dist
OutputBaseFilename=rt-ir-plugins-{#MyAppVersion}-windows-x64
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin

[Files]
; CLAP → Common Files\CLAP ; VST3 → Common Files\VST3
Source: "..\..\target\bundled\*.clap"; DestDir: "{commoncf}\CLAP"; Flags: recursesubdirs ignoreversion
Source: "..\..\target\bundled\*.vst3\*"; DestDir: "{commoncf}\VST3\{#SourceName}"; \
    Flags: recursesubdirs createallsubdirs ignoreversion
