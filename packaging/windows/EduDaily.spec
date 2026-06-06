# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path.cwd()

datas = [
    (str(ROOT / "frontend"), "frontend"),
    (str(ROOT / "templates"), "templates"),
    (str(ROOT / "prompts"), "prompts"),
    (str(ROOT / ".env.example"), "."),
]

model_dir = ROOT / "models" / "bge-small-zh-v1.5"
if model_dir.exists():
    datas.append((str(model_dir), "models/bge-small-zh-v1.5"))

datas += collect_data_files("sentence_transformers")
datas += collect_data_files("chromadb")

hiddenimports = []
hiddenimports += collect_submodules("chromadb")
hiddenimports += collect_submodules("sentence_transformers")
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("keyring")


a = Analysis(
    [str(ROOT / "desktop_app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EduDaily",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "branding" / "app-icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EduDaily",
)
