"""
Summarize a step 4 results folder (from run_repeat.sh).

    python latency-variance/summarize.py latency-variance/results/<tag>

Prints, and saves to <folder>/summary.txt:
  1. sanity checks: same input/output token counts and text for every request
  2. latency distribution: all, cold (request 0), warm (requests 1+), per process
  3. within- vs between-process spread of warm latency
  4. latency, misses and MoE time by request index
  5. what tracks warm latency: correlation with cache counters and phase times
Also writes <folder>/latency.csv (one row per request) for plotting.
"""
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

GiB = 2**30


def pct(xs, q):
    """q-th percentile (0-100), linear interpolation."""
    xs = sorted(xs)
    i = (len(xs) - 1) * q / 100
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def corr(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def slope(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


def main():
    folder = Path(sys.argv[1])
    reqs = [json.loads(line) for line in (folder / "requests.jsonl").open()]
    reqs.sort(key=lambda r: (r["proc"], r["req"]))
    out = []

    def p(line=""):
        out.append(line)
        print(line)

    # Per-request prefill/decode split from the layer logs.
    phase = defaultdict(lambda: defaultdict(float))
    for path in sorted(folder.glob("proc*_layers.csv")):
        for row in csv.DictReader(path.open()):
            s = phase[(int(row["run"][4:]), int(row["request"][1:]))]
            s[row["phase"] + "_misses"] += int(row["misses"])
            s[row["phase"] + "_ms"] += float(row["moe_ms"])
    for r in reqs:
        s = phase[(r["proc"], r["req"])]
        for k in ("prefill_misses", "prefill_ms", "decode_misses", "decode_ms"):
            r[k] = s[k]

    procs = sorted({r["proc"] for r in reqs})
    cold = [r for r in reqs if r["req"] == 0]
    warm = [r for r in reqs if r["req"] > 0]

    p(f"Results: {folder}  ({len(procs)} processes, {len(reqs)} requests)")
    p()
    p("1. Sanity checks")
    p(f"  prompt: {reqs[0]['prompt']!r}")
    p(f"  input_tokens={sorted({r['input_tokens'] for r in reqs})} "
      f"output_tokens={sorted({r['output_tokens'] for r in reqs})} "
      f"distinct generated texts={len({r['generated_text'] for r in reqs})}")
    p(f"  nodes={sorted({r['node'] for r in reqs})} cpus={sorted({str(r['cpus']) for r in reqs})} "
      f"jobs={sorted({str(r['slurm_job']) for r in reqs})}")
    p(f"  cache capacity (max resident experts) per process: "
      + ", ".join(f"proc{pr}={max(r['resident_after'] for r in reqs if r['proc'] == pr)}"
                  for pr in procs))
    texts = defaultdict(list)
    for r in reqs:
        texts[r["generated_text"]].append(f"{r['proc']}/{r['req']}")
    for t, where in texts.items():
        p(f"     {len(where):>3}x {t!r}" + (f"  (proc/req: {' '.join(where[:10])})" if len(texts) > 1 else ""))

    p()
    p("2. Latency distribution (seconds)")
    p(f"  {'group':<14}{'n':>4}{'median':>8}{'mean':>8}{'sd':>7}{'cv %':>6}"
      f"{'min':>8}{'p5':>8}{'p95':>8}{'max':>8}")

    def row(name, rs):
        lat = [r["latency_s"] for r in rs]
        if not lat:
            return
        p(f"  {name:<14}{len(lat):>4}{st.median(lat):>8.3f}{st.mean(lat):>8.3f}{sd(lat):>7.3f}"
          f"{100 * sd(lat) / st.mean(lat):>6.1f}{min(lat):>8.3f}{pct(lat, 5):>8.3f}"
          f"{pct(lat, 95):>8.3f}{max(lat):>8.3f}")

    row("all", reqs)
    row("cold (req 0)", cold)
    row("warm (req 1+)", warm)
    row("warm (req 2+)", [r for r in warm if r["req"] > 1])
    for pr in procs:
        row(f"  proc{pr:02d} warm", [r for r in warm if r["proc"] == pr])

    if len(procs) > 1 and warm:
        p()
        p("3. Warm latency: within vs between processes")
        means = [st.mean(r["latency_s"] for r in warm if r["proc"] == pr) for pr in procs]
        within = st.mean(sd([r["latency_s"] for r in warm if r["proc"] == pr]) for pr in procs)
        p(f"  mean within-process sd = {within:.3f} s")
        p(f"  sd of process means    = {sd(means):.3f} s  (range {min(means):.3f}-{max(means):.3f})")

    p()
    p("4. By request index (mean over processes)")
    p(f"  {'req':>4}{'n':>3}{'latency s':>10}{'min':>8}{'max':>8}{'misses':>8}{'H2D GiB':>9}"
      f"{'pf miss':>8}{'pf ms':>7}{'dec miss':>9}{'dec ms':>7}")
    by_req = defaultdict(list)
    for r in reqs:
        by_req[r["req"]].append(r)
    for i, rs in sorted(by_req.items()):
        lat = [r["latency_s"] for r in rs]

        def m(k):
            return st.mean(r[k] for r in rs)
        p(f"  {i:>4}{len(rs):>3}{st.mean(lat):>10.3f}{min(lat):>8.3f}{max(lat):>8.3f}"
          f"{m('misses'):>8.0f}{m('h2d_bytes') / GiB:>9.2f}{m('prefill_misses'):>8.0f}"
          f"{m('prefill_ms'):>7.0f}{m('decode_misses'):>9.0f}{m('decode_ms'):>7.0f}")
    p("  (pf/dec ms = summed MoE-layer wall time; layers are not CUDA-synchronized, so approximate)")

    if len(warm) > 2:
        p()
        p("5. What tracks warm latency (requests 1+)")
        lat = [r["latency_s"] for r in warm]
        p(f"  {'variable':<16}{'mean':>10}{'sd':>10}{'corr w/ latency':>17}{'latency per unit':>18}")
        for k, scale, unit in [("misses", 1, "ms/miss"), ("h2d_bytes", GiB, "s/GiB"),
                               ("prefill_misses", 1, "ms/miss"), ("decode_misses", 1, "ms/miss"),
                               ("prefill_ms", 1, "s/s"), ("decode_ms", 1, "s/s"),
                               ("output_tokens", 1, "s/token"), ("resident_before", 1, "ms/expert")]:
            xs = [r[k] / scale for r in warm]
            b = slope(xs, lat)
            if unit.startswith("ms/"):
                b *= 1e3
            elif unit == "s/s":
                b *= 1e3  # x is in ms
            p(f"  {k:<16}{st.mean(xs):>10.2f}{sd(xs):>10.2f}{corr(xs, lat):>17.3f}"
              f"{b:>13.3f} {unit}")
        p("  (corr is nan when the variable never changes)")

    with (folder / "latency.csv").open("w", newline="") as f:
        fields = ["proc", "req", "latency_s", "output_tokens", "misses", "hits", "h2d_bytes",
                  "prefill_misses", "prefill_ms", "decode_misses", "decode_ms",
                  "resident_before", "resident_after", "node"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(reqs)

    (folder / "summary.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
