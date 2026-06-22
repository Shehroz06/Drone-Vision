# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for rtsp-vision (Windows, --onedir)
#
# Build command (run from project root on a Windows machine):
#   pyinstaller rtsp_vision.spec
#
# Output: dist\rtsp-vision\rtsp-vision.exe  (+ sibling DLLs and data)

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_all

# ── Collect package data ──────────────────────────────────────────────────────

# EasyOCR: includes its own data files (charsets, language configs, etc.)
easyocr_datas, easyocr_binaries, easyocr_hidden = collect_all('easyocr')

# OpenCV: picks up the FFmpeg DLLs that ship inside the cv2 wheel
cv2_datas    = collect_data_files('cv2')
cv2_binaries = collect_dynamic_libs('cv2')

# Torch: copies the .pth initialisation files and JIT kernels
torch_datas  = collect_data_files('torch', include_py_files=True)

# ── Analysis ──────────────────────────────────────────────────────────────────

a = Analysis(
    ['run.py'],
    pathex=['.'],
    binaries=[
        *cv2_binaries,
        *easyocr_binaries,
    ],
    datas=[
        # Project data directories
        ('frontend',       'frontend'),        # Leaflet map UI
        ('assets',         'assets'),          # config_default.yaml, icon
        ('easyocr_models', 'easyocr_models'),  # pre-downloaded CRAFT + CRNN weights
        # tiles/ is optional — include only if the directory exists
        # ('tiles', 'tiles'),   # uncomment if offline map tiles are needed
        # Package data
        *cv2_datas,
        *easyocr_datas,
        *torch_datas,
    ],
    hiddenimports=[
        # ── uvicorn (all dynamic-import protocol/loop modules) ───────────────
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.http.httptools_impl',
        'uvicorn.protocols.websockets.websockets_impl',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.loops.asyncio',
        'uvicorn.loops.uvloop',
        # ── starlette / FastAPI ──────────────────────────────────────────────
        'anyio',
        'anyio._backends._asyncio',
        'starlette.routing',
        'starlette.staticfiles',
        'starlette.responses',
        'starlette.middleware',
        'starlette.datastructures',
        'starlette.websockets',
        # ── easyocr ─────────────────────────────────────────────────────────
        *easyocr_hidden,
        # ── other ───────────────────────────────────────────────────────────
        'yaml',
        'dotenv',
        'cv2',
        'scipy.special._cython_special',  # pulled in by torch/PIL on some builds
        'PIL._tkinter_finder',
    ],
    hookspath=[],
    runtime_hooks=[],
    # Strip test/notebook/GUI packages that are never used at runtime
    excludes=['tkinter', 'matplotlib', 'IPython', 'notebook', 'pytest'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir: binaries go in COLLECT below
    name='rtsp-vision',
    debug=False,
    strip=False,
    upx=False,               # UPX corrupts some torch DLLs — keep off
    console=True,            # show the terminal so the user can see the server URL
    icon='assets\\icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='rtsp-vision',
)
