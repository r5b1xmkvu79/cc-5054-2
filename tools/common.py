"""Shared paths, ffmpeg locations, and small helpers."""
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DEPS = os.path.join(ROOT, ".deps")

W, H, FPS, N = 320, 180, 24, 240
YUV_FRAME_BYTES = W * H * 3 // 2  # yuv420p


def _find_tool(name):
    ffdir = os.path.join(DEPS, "ffmpeg")
    for d in sorted(os.listdir(ffdir)):
        p = os.path.join(ffdir, d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise RuntimeError(f"{name} not found under {ffdir}")


FFMPEG = _find_tool("ffmpeg")
FFPROBE = _find_tool("ffprobe")


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def variant_dir(name, root=None):
    d = os.path.join(root or OUT, name)
    os.makedirs(d, exist_ok=True)
    return d
