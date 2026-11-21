"""End-to-end pipeline: footage -> cut detection -> plan -> encode -> index
-> fresh-session spot checks -> arrangement comparison report.

  python3 tools/pipeline.py            run everything
  python3 tools/pipeline.py verify     only re-run spot checks + report
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUT, load_json, save_json, variant_dir

VARIANTS = ("dense", "sparse")
PAIR_TARGETS = (84, 147, 90, 156)


def step_footage():
    import footage
    footage.main()


def step_cuts():
    import cuts
    cuts.main()


def step_encode():
    import encode
    encode.main()


def step_index():
    import h264idx
    h264idx.main()


def step_verify():
    import verify
    return {v: verify.verify_variant(v) for v in VARIANTS}


def step_incremental():
    import experiments
    return experiments.main()


def stats_row(variant):
    vd = variant_dir(variant)
    idx = load_json(os.path.join(vd, "index.json"))
    rep = load_json(os.path.join(vd, "verify.json"))
    frames = idx["frames"]
    idr_bytes = sum(f["size"] for f in frames if f["idr"])
    file_bytes = os.path.getsize(os.path.join(vd, "v.mp4"))
    checks = rep["checks"]

    def agg(key):
        vals = [c[key] for c in checks]
        return sum(vals) / len(vals), max(vals)

    mp, maxp = agg("n_packets")
    mb, maxb = agg("bytes")
    mf, maxf = agg("decode_frames")
    return {
        "variant": variant,
        "file_bytes": file_bytes,
        "n_idr": len(idx["idrs"]),
        "idr_bytes": idr_bytes,
        "idr_pct": round(100.0 * idr_bytes / file_bytes, 1),
        "mean_pkts": round(mp, 1), "max_pkts": maxp,
        "mean_bytes": round(mb), "max_bytes": maxb,
        "mean_decode_frames": round(mf, 1), "max_decode_frames": maxf,
        "all_ok": rep["all_ok"],
        "plan_vs_actual": idx["plan_vs_actual"],
        "naive_controls": rep["naive_controls"],
    }


def direct_refs(variant, frm):
    idx = load_json(os.path.join(variant_dir(variant), "index.json"))
    frames = idx["frames"]
    d = {f["frm"]: f["dec"] for f in frames}[frm]
    return sorted({frames[r]["frm"] for r in frames[d]["refs"]})


def report():
    rows = [stats_row(v) for v in VARIANTS]
    print("\n=== arrangement comparison (13 spot-checked positionings) ===")
    hdr = (f"{'variant':<7} {'file_B':>7} {'#IDR':>4} {'IDR_B':>6} {'IDR%':>5} "
           f"{'meanPk':>6} {'maxPk':>5} {'meanB':>6} {'maxB':>6} "
           f"{'meanF':>5} {'maxF':>4}")
    print(hdr)
    for r in rows:
        print(f"{r['variant']:<7} {r['file_bytes']:>7} {r['n_idr']:>4} "
              f"{r['idr_bytes']:>6} {r['idr_pct']:>5} {r['mean_pkts']:>6} "
              f"{r['max_pkts']:>5} {r['mean_bytes']:>6} {r['max_bytes']:>6} "
              f"{r['mean_decode_frames']:>5} {r['max_decode_frames']:>4}")
    print("(meanF/maxF = frames to decode before the target appears; "
          "the positioning wait)")

    print("\n=== plan vs written bitstream (index trusts the bitstream) ===")
    for r in rows:
        pv = r["plan_vs_actual"]
        print(f"[{r['variant']}] planned IDRs {pv['planned']}")
        print(f"[{r['variant']}] written IDRs {pv['actual']}  "
              f"(encoder added {pv['added_by_encoder']}, "
              f"missing {pv['planned_but_missing']})")

    print("\n=== the matched pair of scene changes ===")
    for v in VARIANTS:
        rep = load_json(os.path.join(variant_dir(v), "verify.json"))
        by_t = {c["target"]: c for c in rep["checks"]}
        print(f"[{v}]")
        for t in PAIR_TARGETS:
            c = by_t[t]
            print(f"  frame {t:>3} ({c['label']:<24}) entry={c['entry_frm']:>3} "
                  f"pkts={c['n_packets']:>3} bytes={c['bytes']:>6} "
                  f"decode_frames={c['decode_frames']:>3} "
                  f"direct_refs={direct_refs(v, t)}")

    print("\n=== naive-entry negative controls (sparse) ===")
    for r in rows:
        for nctl in r["naive_controls"]:
            verdict = "corrupt, as expected" \
                if not nctl["reproduces_ground_truth"] else "REPRODUCED?!"
            print(f"  enter at {nctl['start_frm']} -> target "
                  f"{nctl['target_frm']}: {verdict}")

    out = {"variants": rows,
           "all_ok": all(r["all_ok"] for r in rows)}
    save_json(out, os.path.join(OUT, "report.json"))
    print(f"\nreport -> {os.path.join(OUT, 'report.json')}  "
          f"all_checks_ok={out['all_ok']}")


def main():
    steps = sys.argv[1:] or ["all"]
    if "all" in steps or "footage" in steps:
        step_footage()
    if "all" in steps or "cuts" in steps:
        step_cuts()
    if "all" in steps or "encode" in steps:
        step_encode()
    if "all" in steps or "index" in steps:
        step_index()
    if "all" in steps or "verify" in steps:
        step_verify()
    if "all" in steps or "incremental" in steps:
        step_incremental()
    if "all" in steps or "report" in steps or "verify" in steps:
        report()


if __name__ == "__main__":
    main()
