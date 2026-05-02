# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller: один файл ScientificParser.exe (Windows, без установщика и без папки _internal).

Сборка из каталога scientific_parser:
  py -m PyInstaller scientific_parser_portable.spec --noconfirm
"""
import os

from PyInstaller.utils.hooks import collect_all

SPECDIR = os.path.dirname(os.path.abspath(SPEC))

sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all("sklearn")
pandas_datas, pandas_binaries, pandas_hiddenimports = collect_all("pandas")
certifi_datas, certifi_binaries, certifi_hiddenimports = collect_all("certifi")
webview_datas, webview_binaries, webview_hiddenimports = collect_all("webview")
proxy_datas, proxy_binaries, proxy_hiddenimports = collect_all("proxy_tools")

datas = (
    [(os.path.join(SPECDIR, "models"), "models")]
    + sklearn_datas
    + pandas_datas
    + certifi_datas
    + webview_datas
    + proxy_datas
)

binaries = sklearn_binaries + pandas_binaries + certifi_binaries + webview_binaries + proxy_binaries

hiddenimports = list(
    dict.fromkeys(
        sklearn_hiddenimports
        + pandas_hiddenimports
        + certifi_hiddenimports
        + webview_hiddenimports
        + proxy_hiddenimports
        + [
            "waitress",
            "waitress.channel",
            "flask_limiter",
            "rapidfuzz",
            "rapidfuzz._rapidfuzz",
            "lxml",
            "lxml.etree",
            "joblib",
            "inference_pipeline",
            "app",
            "net_security",
            "webview.platforms.edgechromium",
            "webview.platforms.winforms",
        ]
    )
)

a = Analysis(
    [os.path.join(SPECDIR, "portable_launcher.py")],
    pathex=[SPECDIR],
    binaries=binaries,
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ScientificParser",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
