"""Encode the planned variants with libx264, dump packet tables and the raw
Annex B stream for parsing. The encoder is free to deviate from the plan
(keyint limits etc.); the written bitstream is what the locator trusts."""
import json
import os
import subprocess

from common import FFMPEG, FFPROBE, FPS, H, OUT, W, load_json, run, variant_dir

X264_PARAMS = "aud=1:repeat-headers=1:scenecut=0"
PRESET = "medium"
CRF = "20"


def encode_variant(name, forced, g, raw=None, out_root=None):
    vd = variant_dir(name, out_root)
    raw = raw or os.path.join(OUT, "frames.rgb")
    h264 = os.path.join(vd, "v.h264")
    mp4 = os.path.join(vd, "v.mp4")

    expr = "+".join(f"eq(n,{f})" for f in forced) or "0"
    run([FFMPEG, "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", raw,
         "-an", "-c:v", "libx264", "-preset", PRESET, "-crf", CRF,
         "-pix_fmt", "yuv420p",
         "-g", str(g), "-keyint_min", "8",
         "-force_key_frames", f"expr:{expr}",
         "-x264-params", X264_PARAMS,
         mp4])
    run([FFMPEG, "-y", "-v", "error", "-i", mp4, "-c:v", "copy",
         "-bsf:v", "h264_mp4toannexb", "-an", "-f", "h264", h264])

    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts,pts_time,dts,dts_time,duration,pos,size,flags",
         "-of", "json", mp4],
        check=True, capture_output=True, text=True)
    packets = json.loads(probe.stdout)["packets"]
    with open(os.path.join(vd, "packets.json"), "w") as f:
        json.dump(packets, f)

    keys = [i for i, p in enumerate(packets) if "K" in p["flags"]]
    print(f"[{name}] forced={forced} g={g}")
    print(f"[{name}] packets={len(packets)} keyframe packets(dts order)={keys}")
    print(f"[{name}] mp4={os.path.getsize(mp4)}B h264={os.path.getsize(h264)}B")


def main():
    plan = load_json(os.path.join(OUT, "plan.json"))
    for name, v in plan["variants"].items():
        encode_variant(name, v["forced"], v["g"])


if __name__ == "__main__":
    main()
