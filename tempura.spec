# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Tempura Rhythm Lab. Torch is excluded: AI Studio stays optional.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
project_root = Path(SPECPATH).resolve()

datas = []
for pkg in ("librosa", "soundfile", "audioread"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

for rel_path, target_dir in (
    ("config.json", "."),
    ("data/charts", "data/charts"),
    ("data/models", "data/models"),
):
    src_path = project_root / rel_path
    if src_path.exists():
        datas.append((str(src_path), target_dir))

hiddenimports = [
    "scipy.special._cdflib",
    "scipy.linalg.cython_blas",
    "scipy.linalg.cython_lapack",
    "sklearn.utils._typedefs",
    "sounddevice",
    "soundfile",
]
try:
    hiddenimports += collect_submodules("src")
except Exception:
    pass

excludes = [
    "torch",
    "torchaudio",
    "torchvision",
    "pytest",
    "IPython",
    "matplotlib",
    "tkinter",
]

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TempuraRhythmLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TempuraRhythmLab",
)
