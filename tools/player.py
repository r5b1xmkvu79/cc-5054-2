"""Terminal 'player' for the archivist: click any frame, the locator walks
the real dependencies, reads the packets, decodes to the target, and shows
what the positioning cost.

  python3 tools/player.py <variant>            interactive prompt
  python3 tools/player.py <variant> --click N  single click
  python3 tools/player.py <variant> --sheet    write a contact sheet PNG
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import FFMPEG, H, OUT, W, YUV_FRAME_BYTES, load_json, run, variant_dir
from locate import Locator
from verify import decode_carve


def contact_sheet(variant):
    vd = variant_dir(variant)
    out = os.path.join(vd, "sheet.png")
    run([FFMPEG, "-y", "-v", "error", "-i", os.path.join(vd, "v.mp4"),
         "-vf", r"select='not(mod(n\,10))',scale=160:90,tile=6x4:nb_frames=24",
         "-frames:v", "1", out])
    print(f"contact sheet -> {out}")


def click(loc, frm, save_png=True):
    plan = loc.locate(frm)
    rank, total = loc.target_rank_in_carve(plan)
    vd = variant_dir(loc.variant)
    yuv = os.path.join(vd, f"click_{frm}.yuv")
    decode_carve(loc.carve(plan), yuv)
    with open(yuv, "rb") as f:
        data = f.read()
    os.remove(yuv)
    n = len(data) // YUV_FRAME_BYTES
    if n != total:
        print(f"!! decoded {n} frames, expected {total}")
        return
    frame = data[rank * YUV_FRAME_BYTES:(rank + 1) * YUV_FRAME_BYTES]

    print(f"click frame {frm} ({plan['target_pts_time']:.3f}s, "
          f"type {plan['target_type']})")
    print(f"  safe entry: frame {plan['entry_frm']} "
          f"({plan['entry_pts_time']:.3f}s"
          f"{', encoder-added IDR' if plan['entry_encoder_added'] else ''})")
    print(f"  dependencies reach back to frame {plan['closure_min_frm']}")
    print(f"  read {plan['n_packets']} packets / {plan['bytes']} B "
          f"(+{plan['extra_packets']} packets, +{plan['extra_bytes']} B over "
          f"the ideal 1-packet read), decode {plan['decode_frames']} frames")
    if save_png:
        png = os.path.join(vd, f"click_{frm}.png")
        p = subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-f", "rawvideo",
             "-pix_fmt", "yuv420p", "-s", f"{W}x{H}", "-i", "pipe:0", png],
            input=frame, capture_output=True)
        if p.returncode == 0:
            print(f"  frame -> {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["dense", "sparse"])
    ap.add_argument("--click", type=int, default=None)
    ap.add_argument("--sheet", action="store_true")
    args = ap.parse_args()

    loc = Locator(args.variant)
    if args.sheet:
        contact_sheet(args.variant)
    if args.click is not None:
        click(loc, args.click)
        return

    manifest = load_json(os.path.join(OUT, "manifest.json"))
    print(f"variant '{args.variant}', {loc.index['n']} frames, "
          f"IDRs at {loc.index['idr_frames']}")
    print("shots:", ", ".join(f"{s['name']}[{s['start']}..{s['end'] - 1}]"
                              for s in manifest["shots"]))
    print("type a frame number to locate, 'q' to quit")
    while True:
        try:
            line = input("frame> ").strip()
        except EOFError:
            break
        if line in ("q", "quit", ""):
            break
        try:
            frm = int(line)
        except ValueError:
            continue
        if not 0 <= frm < loc.index["n"]:
            print(f"out of range 0..{loc.index['n'] - 1}")
            continue
        click(loc, frm)


if __name__ == "__main__":
    main()
