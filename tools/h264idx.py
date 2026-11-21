"""Parse the written H.264 bitstream (Annex B) and build the positioning index.

Reads out/<variant>/v.h264 + packets.json, writes out/<variant>/index.json.

For every access unit we record: keyframe flag, slice type, frame_num, POC,
reference picture lists (built with a DPB simulation per H.264 8.2.4), plus
the container-level pts/dts/byte-range from ffprobe. The index is validated
against the container (keyframe flags, timestamps) - the bitstream wins.
"""
import json
import os

from common import FPS, load_json, save_json, variant_dir

HIGH_PROFILES = {100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135}
SLICE_NAMES = {0: "P", 1: "B", 2: "I", 3: "SP", 4: "SI"}


class BitReader:
    __slots__ = ("d", "p")

    def __init__(self, data):
        self.d = data
        self.p = 0

    def u(self, n):
        v = 0
        d = self.d
        for _ in range(n):
            v = (v << 1) | ((d[self.p >> 3] >> (7 - (self.p & 7))) & 1)
            self.p += 1
        return v

    def ue(self):
        lz = 0
        while self.u(1) == 0:
            lz += 1
        return (1 << lz) - 1 + (self.u(lz) if lz else 0)

    def se(self):
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)


def ebsp_to_rbsp(b):
    out = bytearray()
    z = 0
    for x in b:
        if z == 2 and x == 3:
            z = 0
            continue
        out.append(x)
        z = z + 1 if x == 0 else 0
    return bytes(out)


def split_nals(data):
    """Yield (nal_start, header_off, nal_end) for each NAL in Annex B data."""
    n = len(data)
    i = 0
    starts = []
    while True:
        p = data.find(b"\x00\x00\x01", i)
        if p < 0:
            break
        s = p if (p == 0 or data[p - 1] != 0) else p - 1
        starts.append((s, p + 3))
        i = p + 3
    for k, (s, h) in enumerate(starts):
        e = starts[k + 1][0] if k + 1 < len(starts) else n
        yield s, h, e


def parse_sps(rbsp):
    r = BitReader(rbsp)
    sps = {}
    profile = r.u(8)
    r.u(8)  # constraint flags + reserved
    sps["level"] = r.u(8)
    sps["id"] = r.ue()
    sps["chroma_format_idc"] = 1
    if profile in HIGH_PROFILES:
        sps["chroma_format_idc"] = r.ue()
        if sps["chroma_format_idc"] == 3:
            r.u(1)
        r.ue()  # bit_depth_luma_minus8
        r.ue()  # bit_depth_chroma_minus8
        r.u(1)  # qpprime_y_zero_transform_bypass
        if r.u(1):  # seq_scaling_matrix_present
            n = 8 if sps["chroma_format_idc"] != 3 else 12
            for i in range(n):
                if r.u(1):
                    size = 16 if i < 6 else 64
                    last, nxt = 8, 8
                    for _ in range(size):
                        if nxt:
                            nxt = (last + r.se() + 256) % 256
                        last = nxt or last
    sps["log2_max_frame_num_minus4"] = r.ue()
    sps["pic_order_cnt_type"] = r.ue()
    if sps["pic_order_cnt_type"] == 0:
        sps["log2_max_pic_order_cnt_lsb_minus4"] = r.ue()
    elif sps["pic_order_cnt_type"] == 1:
        sps["delta_pic_order_always_zero"] = r.u(1)
        r.se()
        r.se()
        for _ in range(r.ue()):
            r.se()
    sps["max_num_ref_frames"] = r.ue()
    sps["gaps_allowed"] = r.u(1)
    r.ue()  # pic_width_in_mbs_minus1
    r.ue()  # pic_height_in_map_units_minus1
    sps["frame_mbs_only"] = r.u(1)
    if not sps["frame_mbs_only"]:
        r.u(1)
    r.u(1)  # direct_8x8_inference
    if r.u(1):  # frame_cropping
        r.ue(); r.ue(); r.ue(); r.ue()
    return sps


def parse_pps(rbsp):
    r = BitReader(rbsp)
    pps = {}
    pps["id"] = r.ue()
    pps["sps_id"] = r.ue()
    pps["cabac"] = r.u(1)
    pps["bottom_field_pic_order"] = r.u(1)
    nsg = r.ue()
    if nsg != 0:
        raise ValueError("slice groups not supported")
    pps["num_ref_idx_l0_default"] = r.ue() + 1
    pps["num_ref_idx_l1_default"] = r.ue() + 1
    pps["weighted_pred"] = r.u(1)
    pps["weighted_bipred"] = r.u(2)
    r.se()  # pic_init_qp_minus26
    r.se()  # pic_init_qs_minus26
    r.se()  # chroma_qp_index_offset
    pps["deblocking_control_present"] = r.u(1)
    r.u(1)  # constrained_intra_pred
    pps["redundant_pic_cnt_present"] = r.u(1)
    return pps


def parse_slice_header(rbsp, nal_type, nal_ref_idc, spss, ppss):
    r = BitReader(rbsp)
    sh = {"nal_ref_idc": nal_ref_idc}
    sh["first_mb_in_slice"] = r.ue()
    st = r.ue()
    sh["slice_type"] = SLICE_NAMES[st % 5]
    pps = ppss[r.ue()]
    sps = spss[pps["sps_id"]]
    sh["frame_num"] = r.u(sps["log2_max_frame_num_minus4"] + 4)
    if not sps["frame_mbs_only"]:
        if r.u(1):
            r.u(1)
    idr = nal_type == 5
    sh["idr"] = idr
    if idr:
        r.ue()  # idr_pic_id
    poc_lsb = 0
    if sps["pic_order_cnt_type"] == 0:
        poc_lsb = r.u(sps["log2_max_pic_order_cnt_lsb_minus4"] + 4)
        if pps["bottom_field_pic_order"]:
            r.se()
    elif sps["pic_order_cnt_type"] == 1:
        raise ValueError("poc_type 1 not supported")
    sh["poc_lsb"] = poc_lsb
    if pps["redundant_pic_cnt_present"]:
        r.ue()
    st5 = sh["slice_type"]
    if st5 == "B":
        r.u(1)  # direct_spatial_mv_pred
    num_l0, num_l1 = pps["num_ref_idx_l0_default"], pps["num_ref_idx_l1_default"]
    if st5 in ("P", "SP", "B"):
        if r.u(1):  # num_ref_idx_active_override
            num_l0 = r.ue() + 1
            if st5 == "B":
                num_l1 = r.ue() + 1
    sh["num_ref_idx_l0"] = num_l0
    sh["num_ref_idx_l1"] = num_l1

    def mods():
        out = []
        if r.u(1):
            while True:
                idc = r.ue()
                if idc == 3:
                    break
                out.append((idc, r.ue()))
        return out

    sh["mods_l0"] = mods() if st5 not in ("I", "SI") else []
    sh["mods_l1"] = mods() if st5 == "B" else []

    # pred weight table (skipped, but must be walked over to reach marking)
    chroma = sps["chroma_format_idc"] != 0
    if (pps["weighted_pred"] and st5 in ("P", "SP")) or \
       (pps["weighted_bipred"] == 1 and st5 == "B"):
        r.ue()
        if chroma:
            r.ue()
        for i in range(num_l0):
            if r.u(1):
                r.se(); r.se()
            if chroma and r.u(1):
                r.se(); r.se(); r.se(); r.se()
        if st5 == "B":
            for i in range(num_l1):
                if r.u(1):
                    r.se(); r.se()
                if chroma and r.u(1):
                    r.se(); r.se(); r.se(); r.se()

    # dec_ref_pic_marking
    sh["mmco"] = []
    if nal_ref_idc:
        if idr:
            r.u(1)  # no_output_of_prior_pics
            r.u(1)  # long_term_reference_flag
        else:
            if r.u(1):  # adaptive_ref_pic_marking_mode
                while True:
                    op = r.ue()
                    if op == 0:
                        break
                    args = []
                    if op in (1, 3):
                        args.append(r.ue())
                    if op == 2:
                        args.append(r.ue())
                    if op in (3, 6):
                        args.append(r.ue())
                    if op == 4:
                        args.append(r.ue())
                    sh["mmco"].append((op, args))
    sh["_sps"] = sps
    return sh


class RefPic:
    __slots__ = ("frame_num", "poc", "dec")

    def __init__(self, frame_num, poc, dec):
        self.frame_num = frame_num
        self.poc = poc
        self.dec = dec


class StreamState:
    """POC + DPB bookkeeping across the stream in decode order."""

    def __init__(self):
        self.prev_poc_msb = 0
        self.prev_poc_lsb = 0
        self.dpb = []  # list[RefPic], short-term only

    def poc(self, sh):
        sps = sh["_sps"]
        if sps["pic_order_cnt_type"] == 2:
            fn = sh["frame_num"]
            return 2 * fn if sh["nal_ref_idc"] else 2 * fn - 1
        max_lsb = 1 << (sps["log2_max_pic_order_cnt_lsb_minus4"] + 4)
        if sh["idr"]:
            self.prev_poc_msb = 0
            self.prev_poc_lsb = 0
        lsb = sh["poc_lsb"]
        if lsb < self.prev_poc_lsb and self.prev_poc_lsb - lsb >= max_lsb // 2:
            msb = self.prev_poc_msb + max_lsb
        elif lsb > self.prev_poc_lsb and lsb - self.prev_poc_lsb > max_lsb // 2:
            msb = self.prev_poc_msb - max_lsb
        else:
            msb = self.prev_poc_msb
        if sh["nal_ref_idc"]:
            self.prev_poc_msb = msb
            self.prev_poc_lsb = lsb
        return msb + lsb

    def ref_lists(self, sh, poc):
        sps = sh["_sps"]
        max_fn = 1 << (sps["log2_max_frame_num_minus4"] + 4)
        cur_fn = sh["frame_num"]

        def fw(p):
            return p.frame_num - max_fn if p.frame_num > cur_fn else p.frame_num

        st = sh["slice_type"]
        l0 = l1 = []
        if st in ("P", "SP"):
            l0 = sorted(self.dpb, key=fw, reverse=True)
        elif st == "B":
            before = sorted((p for p in self.dpb if p.poc < poc),
                            key=lambda p: p.poc, reverse=True)
            after = sorted((p for p in self.dpb if p.poc > poc),
                           key=lambda p: p.poc)
            l0 = before + after
            l1 = after + before
            if len(l1) > 1 and [p.frame_num for p in l1] == [p.frame_num for p in l0]:
                l1[0], l1[1] = l1[1], l1[0]

        def apply_mods(lst, mods):
            if not mods:
                return lst
            lst = list(lst)
            pic_num_pred = cur_fn
            ref_idx = 0
            for idc, val in mods:
                if idc == 0:
                    pic_num_pred = pic_num_pred - (val + 1)
                    if pic_num_pred < 0:
                        pic_num_pred += max_fn
                elif idc == 1:
                    pic_num_pred = pic_num_pred + (val + 1)
                    if pic_num_pred >= max_fn:
                        pic_num_pred -= max_fn
                else:
                    raise ValueError("long-term modification not supported")
                tgt = None
                for j, p in enumerate(lst):
                    if fw(p) % max_fn == pic_num_pred % max_fn:
                        tgt = lst.pop(j)
                        break
                if tgt is None:
                    raise ValueError("modification target not found")
                lst.insert(ref_idx, tgt)
                ref_idx += 1
            return lst

        l0 = apply_mods(l0, sh["mods_l0"])[: sh["num_ref_idx_l0"]]
        l1 = apply_mods(l1, sh["mods_l1"])[: sh["num_ref_idx_l1"]] if st == "B" else []
        return l0, l1

    def store(self, sh, poc, dec):
        sps = sh["_sps"]
        max_fn = 1 << (sps["log2_max_frame_num_minus4"] + 4)
        cur_fn = sh["frame_num"]
        if sh["idr"]:
            self.dpb = []
        if not sh["nal_ref_idc"]:
            return
        self.dpb.append(RefPic(cur_fn, poc, dec))
        if sh["mmco"]:
            for op, args in sh["mmco"]:
                if op == 1:  # mark short-term picture as unused
                    pic_num_x = (cur_fn - (args[0] + 1)) % max_fn
                    self.dpb = [p for p in self.dpb if p.frame_num != pic_num_x]
                elif op == 5:  # reset
                    self.dpb = []
                else:
                    raise ValueError(f"mmco op {op} not supported")
        else:  # sliding window
            while len(self.dpb) > sps["max_num_ref_frames"]:
                self.dpb.sort(key=lambda p: p.frame_num - max_fn
                              if p.frame_num > cur_fn else p.frame_num)
                self.dpb.pop(0)


def build_index(variant, vdir=None, plan_path=None):
    vd = vdir or variant_dir(variant)
    with open(os.path.join(vd, "v.h264"), "rb") as f:
        data = f.read()
    packets = load_json(os.path.join(vd, "packets.json"))

    spss, ppss = {}, {}
    aus = []  # each: {start, end, vcl: (nal_type, nal_ref_idc, rbsp)}
    cur = None
    for s, h, e in split_nals(data):
        nal_ref_idc = (data[h] >> 5) & 3
        ntype = data[h] & 0x1F
        # an AU closes as soon as any NAL follows its VCL slice (x264 emits
        # AUD/SPS/PPS/SEI before the slice, one slice per AU)
        if cur is not None and cur["vcl"] is not None:
            aus.append(cur)
            cur = None
        if cur is None:
            cur = {"start": s, "vcl": None}
        if ntype in (1, 5):  # VCL slice
            cur["vcl"] = (ntype, nal_ref_idc, ebsp_to_rbsp(data[h + 1:e]))
        elif ntype == 7:
            sp = parse_sps(ebsp_to_rbsp(data[h + 1:e]))
            spss[sp["id"]] = sp
        elif ntype == 8:
            pp = parse_pps(ebsp_to_rbsp(data[h + 1:e]))
            ppss[pp["id"]] = pp
    if cur is not None:
        aus.append(cur)
    for i, au in enumerate(aus):
        au["end"] = aus[i + 1]["start"] if i + 1 < len(aus) else len(data)

    assert len(aus) == len(packets), \
        f"AU count {len(aus)} != packet count {len(packets)}"

    state = StreamState()
    frames = []
    for dec, au in enumerate(aus):
        ntype, nal_ref_idc, rbsp = au["vcl"]
        sh = parse_slice_header(rbsp, ntype, nal_ref_idc, spss, ppss)
        poc = state.poc(sh)
        l0, l1 = state.ref_lists(sh, poc)
        frames.append({
            "dec": dec,
            "type": sh["slice_type"],
            "idr": bool(sh["idr"]),
            "nal_ref_idc": nal_ref_idc,
            "frame_num": sh["frame_num"],
            "poc": poc,
            "refs_l0": [p.dec for p in l0],
            "refs_l1": [p.dec for p in l1],
            "au_start": au["start"],
            "au_end": au["end"],
        })
        state.store(sh, poc, dec)

    # merge container packet info (decode order)
    pts_list = [int(p["pts"]) for p in packets]
    pts_rank = {pts: r for r, pts in enumerate(sorted(pts_list))}
    for dec, fr in enumerate(frames):
        p = packets[dec]
        fr["pts"] = int(p["pts"])
        fr["dts"] = int(p["dts"])
        fr["pts_time"] = float(p["pts_time"])
        fr["frm"] = pts_rank[fr["pts"]]
        fr["pos"] = int(p["pos"])
        fr["size"] = int(p["size"])
        fr["container_key"] = "K" in p["flags"]
        fr["refs"] = sorted(set(fr["refs_l0"]) | set(fr["refs_l1"]))

    # validations: container vs bitstream
    idrs = [f["dec"] for f in frames if f["idr"]]
    cont_keys = [i for i, f in enumerate(frames) if f["container_key"]]
    assert idrs == cont_keys, f"IDR/container keyframe mismatch: {idrs} vs {cont_keys}"
    # POC ordering is only meaningful within a closed GOP (it restarts at
    # each IDR); validate it there against the container presentation order.
    gop = []
    for f in frames:
        if f["idr"] and gop:
            by_poc = sorted(gop, key=lambda x: x["poc"])
            for r, x in enumerate(by_poc):
                assert x["frm"] == min(y["frm"] for y in gop) + r, \
                    "POC order != PTS order within GOP"
            gop = []
        gop.append(f)
    by_frm = sorted(frames, key=lambda f: f["frm"])
    for i in range(1, len(by_frm)):
        dt = by_frm[i]["pts_time"] - by_frm[i - 1]["pts_time"]
        assert abs(dt - 1.0 / FPS) < 1e-6, f"non-constant frame interval {dt}"

    plan = load_json(plan_path or os.path.join(os.path.dirname(vd), "plan.json"))
    planned = plan["variants"][variant]["forced"]
    actual = [f["frm"] for f in frames if f["idr"]]
    index = {
        "variant": variant,
        "fps": FPS,
        "n": len(frames),
        "planned_forced": planned,
        "g": plan["variants"][variant]["g"],
        "idrs": idrs,
        "idr_frames": actual,
        "plan_vs_actual": {
            "planned": planned,
            "actual": actual,
            "added_by_encoder": sorted(set(actual) - set(planned)),
            "planned_but_missing": sorted(set(planned) - set(actual)),
        },
        "frames": frames,
    }
    save_json(index, os.path.join(vd, "index.json"))

    print(f"[{variant}] AUs={len(frames)} IDRs(frm)={actual}")
    pv = index["plan_vs_actual"]
    print(f"[{variant}] planned={pv['planned']}")
    print(f"[{variant}] encoder added={pv['added_by_encoder']} "
          f"missing={pv['planned_but_missing']}")
    return index


def main():
    for variant in ("dense", "sparse"):
        build_index(variant)


if __name__ == "__main__":
    main()
