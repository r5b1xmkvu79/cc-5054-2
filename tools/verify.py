"""Spot-check the positioning index in fresh decode sessions.

Ground truth: decode the whole mp4 from the head (fresh session), recording
every frame's pixels and pts.

For each target frame:
  * carve path  - the locator reads exactly the planned packets from the
                  bitstream; a fresh decoder session decodes them; the target
                  image must match the ground truth bit-exactly.
  * seek path   - a fresh container-level seek to the index's entry timestamp
                  decodes forward; the target image and its pts must match.

Also reports the packets/bytes each positioning had to read beyond the
ideal single-packet read.
"""
import os
import re
import subprocess
import time

import numpy as np

from common import FFMPEG, H, OUT, W, YUV_FRAME_BYTES, load_json, save_json, variant_dir
from locate import Locator

SHOWINFO = re.compile(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([0-9.]+)")

DEFAULT_TARGETS = [10, 41, 84, 90, 120, 121, 147, 156, 160, 189, 191, 205, 239]


def decode_all(mp4, out_yuv):
    """Fresh session: decode the whole file from the head; return pts list."""
    cmd = [FFMPEG, "-y", "-v", "info", "-i", mp4, "-vf", "showinfo",
           "-f", "rawvideo", "-pix_fmt", "yuv420p", out_yuv]
    p = subprocess.run(cmd, capture_output=True, text=True)
    pts = [(int(m.group(1)), float(m.group(3)))
           for m in SHOWINFO.finditer(p.stderr)]
    return pts


def decode_carve(carve_bytes, out_yuv):
    """Fresh session: decode exactly these bitstream bytes."""
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "h264", "-i", "pipe:0",
           "-f", "rawvideo", "-pix_fmt", "yuv420p", out_yuv]
    subprocess.run(cmd, input=carve_bytes, check=True, capture_output=True)


def decode_seek(mp4, entry_time, n_frames, out_yuv):
    """Fresh session: container seek to entry_time, decode n_frames."""
    cmd = [FFMPEG, "-y", "-v", "info", "-copyts",
           "-ss", f"{entry_time:.9f}", "-i", mp4,
           "-frames:v", str(n_frames), "-vf", "showinfo",
           "-f", "rawvideo", "-pix_fmt", "yuv420p", out_yuv]
    p = subprocess.run(cmd, capture_output=True, text=True)
    pts = [(int(m.group(1)), float(m.group(3)))
           for m in SHOWINFO.finditer(p.stderr)]
    return pts


def frame_bytes(yuv, i):
    return yuv[i * YUV_FRAME_BYTES:(i + 1) * YUV_FRAME_BYTES]


def pixels_match(a, b):
    if a == b:
        return True, 0
    if len(a) != len(b):
        return False, 255
    d = np.frombuffer(a, np.uint8).astype(np.int16) - np.frombuffer(b, np.uint8).astype(np.int16)
    return False, int(np.abs(d).max())


def label_for(frm, manifest):
    for t in manifest["transitions"]:
        if t["kind"] == "cut" and t["at"] == frm:
            return f"cut@{frm} (first of new shot)"
        if t["kind"] == "flash" and t["at"] == frm:
            return f"flash@{frm}"
        if t["kind"] == "dissolve" and t["start"] <= frm < t["end"]:
            return f"dissolve {t['name']} [{t['start']}..{t['end'] - 1}]"
    for s in manifest["shots"]:
        if s["start"] <= frm < s["end"]:
            return f"shot {s['name']}"
    return "?"


def naive_entry_check(loc, start_frm, target_frm, gt, vd):
    """Negative control: enter at a non-IDR frame (as a naive player that
    assumes 'scene change => keyframe' would). Even with SPS/PPS provided,
    the decode must fail to reproduce the ground truth."""
    from h264idx import split_nals
    frames = loc.frames
    start_dec = loc.frm_to_dec[start_frm]
    target_dec = loc.frm_to_dec[target_frm]
    assert not frames[start_dec]["idr"], "control must start at a non-IDR"

    with open(loc.h264_path, "rb") as f:
        head = f.read(4096)
    sps_pps = b""
    for s, h, e in split_nals(head):
        if data_ntype(head[h]) in (7, 8):
            sps_pps += head[s:e]
    start = frames[start_dec]["au_start"]
    end = frames[target_dec]["au_end"]
    with open(loc.h264_path, "rb") as f:
        f.seek(start)
        carve = sps_pps + f.read(end - start)

    out = os.path.join(vd, f"naive_{start_frm}_{target_frm}.yuv")
    p = subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "h264", "-i",
                        "pipe:0", "-f", "rawvideo", "-pix_fmt", "yuv420p", out],
                       input=carve, capture_output=True)
    n_out = os.path.getsize(out) // YUV_FRAME_BYTES if os.path.exists(out) else 0
    match = False
    if n_out:
        with open(out, "rb") as f:
            data = f.read()
        match, _ = pixels_match(frame_bytes(data, n_out - 1),
                                frame_bytes(gt, target_frm))
    if os.path.exists(out):
        os.remove(out)
    return {"start_frm": start_frm, "target_frm": target_frm,
            "frames_out": n_out, "reproduces_ground_truth": bool(match)}


def data_ntype(byte):
    return byte & 0x1F


def verify_variant(variant, targets=None):
    vd = variant_dir(variant)
    loc = Locator(variant)
    manifest = load_json(os.path.join(OUT, "manifest.json"))
    targets = targets or DEFAULT_TARGETS

    gt_yuv_path = os.path.join(vd, "gt.yuv")
    gt_pts = decode_all(loc.mp4_path, gt_yuv_path)
    with open(gt_yuv_path, "rb") as f:
        gt = f.read()
    n_gt = len(gt) // YUV_FRAME_BYTES
    assert n_gt == loc.index["n"] == len(gt_pts), (n_gt, len(gt_pts))
    assert [n for n, _ in gt_pts] == list(range(n_gt)), "gt pts order broken"

    checks = []
    for t in targets:
        plan = loc.locate(t)
        rank, total = loc.target_rank_in_carve(plan)

        carve_path = os.path.join(vd, f"carve_{t}.yuv")
        carve = loc.carve(plan)
        t0 = time.perf_counter()
        decode_carve(carve, carve_path)
        carve_ms = (time.perf_counter() - t0) * 1000.0
        with open(carve_path, "rb") as f:
            cy = f.read()
        n_carve = len(cy) // YUV_FRAME_BYTES
        carve_ok, carve_diff = pixels_match(
            frame_bytes(cy, rank) if n_carve == total else b"",
            frame_bytes(gt, t))

        seek_path = os.path.join(vd, f"seek_{t}.yuv")
        k = t - plan["entry_frm"] + 1
        pts = decode_seek(loc.mp4_path, plan["entry_pts_time"], k, seek_path)
        with open(seek_path, "rb") as f:
            sy = f.read()
        n_seek = len(sy) // YUV_FRAME_BYTES
        seek_ok, seek_diff = pixels_match(
            frame_bytes(sy, k - 1) if n_seek == k else b"",
            frame_bytes(gt, t))
        pts_ok = (len(pts) == k
                  and abs(pts[0][1] - plan["entry_pts_time"]) < 1e-3
                  and abs(pts[-1][1] - gt_pts[t][1]) < 1e-3)

        checks.append({
            "target": t,
            "label": label_for(t, manifest),
            "entry_frm": plan["entry_frm"],
            "entry_encoder_added": plan["entry_encoder_added"],
            "closure_min_frm": plan["closure_min_frm"],
            "n_packets": plan["n_packets"],
            "bytes": plan["bytes"],
            "extra_packets": plan["extra_packets"],
            "extra_bytes": plan["extra_bytes"],
            "decode_frames": plan["decode_frames"],
            "carve_decode_ms": round(carve_ms, 1),
            "carve_pixels_ok": carve_ok,
            "seek_pixels_ok": seek_ok,
            "seek_pts_ok": pts_ok,
            "ok": carve_ok and seek_ok and pts_ok,
        })
        os.remove(carve_path)
        os.remove(seek_path)

    naive = []
    if variant == "sparse":
        # a naive player assuming 'scene change => keyframe' would enter here
        for s, t in ((140, 147), (156, 160)):
            naive.append(naive_entry_check(loc, s, t, gt, vd))

    os.remove(gt_yuv_path)
    report = {
        "variant": variant,
        "n_frames": n_gt,
        "checks": checks,
        "naive_controls": naive,
        "all_ok": all(c["ok"] for c in checks)
                  and all(not n["reproduces_ground_truth"] for n in naive),
    }
    save_json(report, os.path.join(vd, "verify.json"))

    print(f"\n=== verify [{variant}]  all_ok={report['all_ok']} ===")
    hdr = (f"{'tgt':>4} {'label':<26} {'entry':>5} {'dep@':>4} {'pkts':>4} "
           f"{'bytes':>6} {'+pkts':>5} {'+bytes':>6} {'decf':>4} "
           f"{'carve':>5} {'seek':>5} {'pts':>4}")
    print(hdr)
    for c in checks:
        print(f"{c['target']:>4} {c['label']:<26} {c['entry_frm']:>5} "
              f"{c['closure_min_frm']:>4} {c['n_packets']:>4} {c['bytes']:>6} "
              f"{c['extra_packets']:>5} {c['extra_bytes']:>6} "
              f"{c['decode_frames']:>4} "
              f"{'OK' if c['carve_pixels_ok'] else 'BAD':>5} "
              f"{'OK' if c['seek_pixels_ok'] else 'BAD':>5} "
              f"{'OK' if c['seek_pts_ok'] else 'BAD':>4}")
    for nctl in naive:
        verdict = "corrupt (as expected)" if not nctl["reproduces_ground_truth"] \
            else "REPRODUCED?! entry would be unsafe"
        print(f"  naive entry at {nctl['start_frm']} -> target "
              f"{nctl['target_frm']}: {verdict} "
              f"(frames_out={nctl['frames_out']})")
    return report


def main():
    for variant in ("dense", "sparse"):
        verify_variant(variant)


if __name__ == "__main__":
    main()
