"""Random-access locator: given a target presentation frame, walk the actual
reference graph parsed from the bitstream, find the safe entry IDR, and read
exactly the packets needed to decode up to the target."""
import os

from common import load_json, variant_dir


class Locator:
    def __init__(self, variant, vdir=None):
        self.variant = variant
        vd = vdir or variant_dir(variant)
        self.vdir = vd
        self.index = load_json(os.path.join(vd, "index.json"))
        self.frames = self.index["frames"]
        self.h264_path = os.path.join(vd, "v.h264")
        self.mp4_path = os.path.join(vd, "v.mp4")
        self.frm_to_dec = {f["frm"]: f["dec"] for f in self.frames}
        self.idrs = self.index["idrs"]
        self.added = set(self.index["plan_vs_actual"]["added_by_encoder"])

    def locate(self, target_frm):
        """Return the read+decode plan for presenting target_frm."""
        d = self.frm_to_dec[target_frm]
        frames = self.frames

        # walk the actual dependency graph from the bitstream
        needed = set()
        stack = [d]
        while stack:
            f = stack.pop()
            if f in needed:
                continue
            needed.add(f)
            stack.extend(frames[f]["refs"])
        min_needed = min(needed)

        entry = max(i for i in self.idrs if i <= d)
        assert entry <= min_needed, (
            f"dependency closure reaches dec {min_needed} before entry {entry}")

        pkts = frames[entry:d + 1]
        nbytes = sum(p["size"] for p in pkts)
        entry_frm = frames[entry]["frm"]
        return {
            "variant": self.variant,
            "target_frm": target_frm,
            "target_dec": d,
            "target_pts_time": frames[d]["pts_time"],
            "target_type": frames[d]["type"],
            "entry_frm": entry_frm,
            "entry_dec": entry,
            "entry_pts_time": frames[entry]["pts_time"],
            "entry_encoder_added": entry_frm in self.added,
            "closure_min_dec": min_needed,
            "closure_min_frm": frames[min_needed]["frm"],
            "n_packets": len(pkts),
            "bytes": nbytes,
            "extra_packets": len(pkts) - 1,
            "extra_bytes": nbytes - frames[d]["size"],
            "decode_frames": d - entry + 1,
            "packets": [{"dec": p["dec"], "pos": p["pos"], "size": p["size"]}
                        for p in pkts],
        }

    def carve(self, plan):
        """Read (from the written bitstream) the exact bytes the plan needs."""
        entry = plan["entry_dec"]
        d = plan["target_dec"]
        start = self.frames[entry]["au_start"]
        end = self.frames[d]["au_end"]
        with open(self.h264_path, "rb") as f:
            f.seek(start)
            return f.read(end - start)

    def target_rank_in_carve(self, plan):
        """Presentation rank of the target among frames decoded from the carve."""
        entry, d = plan["entry_dec"], plan["target_dec"]
        frms = sorted(self.frames[i]["frm"] for i in range(entry, d + 1))
        return frms.index(plan["target_frm"]), len(frms)
