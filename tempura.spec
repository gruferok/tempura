# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Tempura Rhythm Lab (PyInstaller 6.x).
# Torch is excluded: AI Studio stays optional and is reported as unavailable in the build.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

project_root = Path(SPECPATH).resolve()

datas, binaries = [], []


def _collect(fn, package):
    """Native/data collection is best-effort per package; a missing optional package must not break the spec."""
    try:
        return fn(package)
    except Exception:
        return []


for pkg in ("librosa", "soundfile", "audioread"):
    datas += _collect(collect_data_files, pkg)

# Native audio runtimes: Windows wheels ship PortAudio/libsndfile in these helper packages.
for pkg in ("_sounddevice_data", "_soundfile_data"):
    datas += _collect(collect_data_files, pkg)
    binaries += _collect(collect_dynamic_libs, pkg)

for rel_path, target_dir in (
    ("config.json", "."),
    ("data/charts", "data/charts"),
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
hiddenimports += _collect(collect_submodules, "src")

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
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

# console=True keeps --diagnose / --headless-smoke output visible for beta builds.
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
    name="TempuraRhythmLab",
)
