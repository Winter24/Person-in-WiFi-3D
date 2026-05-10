#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/tmp/cuda-driver}"
export LD_LIBRARY_PATH="/tmp/cuda-driver:/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/tmp/cuda-driver:/usr/lib64-nvidia:${LIBRARY_PATH:-}"

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import glob
import importlib
import os
import subprocess
import sys


def unique(items):
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def candidate_dirs():
    dirs = []
    for key in ("TRITON_LIBCUDA_PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH"):
        dirs.extend(os.environ.get(key, "").split(":"))
    dirs.extend([
        "/tmp/cuda-driver",
        "/usr/lib64-nvidia",
        "/usr/local/cuda/compat",
        "/usr/lib/x86_64-linux-gnu",
    ])
    try:
        libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
        for line in libs.splitlines():
            if "libcuda.so" in line and "=>" in line:
                dirs.append(os.path.dirname(line.split("=>")[-1].strip()))
    except Exception:
        pass
    return unique(dirs)


def find_libcuda():
    for directory in candidate_dirs():
        exact = os.path.join(directory, "libcuda.so")
        versioned = os.path.join(directory, "libcuda.so.1")
        if os.path.exists(exact):
            return exact
        if os.path.exists(versioned):
            return versioned
    for pattern in (
        "/usr/lib*/**/libcuda.so",
        "/usr/lib*/**/libcuda.so.1",
        "/usr/local/cuda*/**/libcuda.so",
        "/usr/local/cuda*/**/libcuda.so.1",
    ):
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def ensure_tmp_symlink():
    target = find_libcuda()
    if not target:
        print("WARNING: libcuda.so/libcuda.so.1 not found; Triton patch will still be installed.")
        return
    tmp_dir = Path("/tmp/cuda-driver")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    link = tmp_dir / "libcuda.so"
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        print("Triton libcuda symlink:", link, "->", target)
    except OSError as exc:
        print("WARNING: Could not create /tmp/cuda-driver/libcuda.so symlink:", exc)


def triton_libcuda_dirs_works():
    try:
        build = importlib.import_module("triton.common.build")
    except ModuleNotFoundError:
        print("Triton is not installed; skipping Triton libcuda auto-fix.")
        return True
    try:
        print("Triton libcuda dirs:", build.libcuda_dirs())
        return True
    except Exception as exc:
        print("Triton libcuda_dirs() failed before patch:", repr(exc))
        return False


def patch_triton_build():
    build = importlib.import_module("triton.common.build")
    path = Path(build.__file__)
    source = path.read_text()

    marker = "# RESFES_TRITON_LIBCUDA_AUTOFIX"
    if marker in source:
        print("Triton build.py already patched:", path)
        return

    start = source.index("def libcuda_dirs():")
    end = source.find("\ndef ", start + 1)
    if end == -1:
        end = len(source)
        suffix = ""
    else:
        suffix = source[end + 1:]

    backup = path.with_suffix(".py.bak")
    if not backup.exists():
        backup.write_text(source)

    new_func = r'''def libcuda_dirs():
    # RESFES_TRITON_LIBCUDA_AUTOFIX
    import os
    import subprocess

    candidate_dirs = [
        "/tmp/cuda-driver",
        "/usr/lib64-nvidia",
        "/usr/local/cuda/compat",
        "/usr/lib/x86_64-linux-gnu",
    ]

    env_dirs = []
    for key in ("TRITON_LIBCUDA_PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH"):
        value = os.environ.get(key, "")
        env_dirs.extend([x for x in value.split(":") if x])

    dirs = []
    for path in env_dirs + candidate_dirs:
        if path and path not in dirs:
            dirs.append(path)

    try:
        libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
        for line in libs.splitlines():
            if "libcuda.so" in line and "=>" in line:
                path = os.path.dirname(line.split("=>")[-1].strip())
                if path not in dirs:
                    dirs.append(path)
    except Exception:
        pass

    for path in dirs:
        if os.path.exists(os.path.join(path, "libcuda.so")):
            return [path]

    msg = "libcuda.so cannot found! Searched: {}".format(dirs)
    raise AssertionError(msg)

'''

    path.write_text(source[:start] + new_func + suffix)
    print("Patched Triton build.py:", path)
    print("Backup:", backup)


ensure_tmp_symlink()
if triton_libcuda_dirs_works():
    sys.exit(0)

try:
    patch_triton_build()
except Exception as exc:
    print("WARNING: Could not patch Triton build.py:", repr(exc))
    sys.exit(0)

importlib.invalidate_caches()
try:
    import triton.common.build as build
    print("Triton libcuda dirs after patch:", build.libcuda_dirs())
except Exception as exc:
    print("WARNING: Triton libcuda auto-fix installed but verification failed:", repr(exc))
PY
