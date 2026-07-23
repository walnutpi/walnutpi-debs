from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
import os
import subprocess
import sys


def _compile_v4l_cam():
    """Compile v4l-cam.c → libv4l-cam.so if source is newer than binary."""
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(here, "k230_sensor", "v4l-cam")
    src = os.path.join(src_dir, "v4l-cam.c")
    dst = os.path.join(src_dir, "libv4l-cam.so")

    # skip if .so is already up-to-date
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        print(f"[v4l-cam] {dst} is up-to-date, skipping compile")
        return

    cc = os.environ.get("CC", "gcc")
    cflags = ["-fPIC", "-shared", "-O2"]

    print(f"[v4l-cam] compiling {src} → {dst}")
    try:
        subprocess.check_call([cc] + cflags + ["-o", dst, src])
    except subprocess.CalledProcessError as e:
        print(f"[v4l-cam] compilation failed: {e}", file=sys.stderr)
        raise


# compile at import time — covers both `pip install -e .` and `pip install .`
_compile_v4l_cam()


class build_py(_build_py):
    """Also compile during wheel builds (non-editable install)."""

    def run(self):
        _compile_v4l_cam()
        super().run()


setup(cmdclass={"build_py": build_py})
