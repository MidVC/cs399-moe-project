"""
Summarize a step 5 results folder (from run_order_effect.sh).

    python order-effect/summarize.py order-effect/results/<tag>

Prints, and saves to <folder>/summary.txt:
  1. sanity checks: same text and token counts for every run of a prompt
  2. latency + cache counters per condition (prompt, what ran before it)
  3. misses and H2D bytes per condition, split into prefill and decode
  4. expert overlap between the two prompts (from the layer logs)
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

GiB = 2**30


def condition(pid, prev):
    return f"{pid} cold (first)" if prev is None else f"{pid} after {prev}"


def main():
    folder = Path(sys.argv[1])
    reqs = [json.loads(line) for line in (folder / "requests.jsonl").open()]
    out = []

    def p(line=""):
        out.append(line)
        print(line)

    p(f"Results: {folder}  ({len(reqs)} requests)")
    p()
    p("1. Sanity checks")
    for pid in sorted({r["prompt_id"] for r in reqs}):
        rs = [r for r in reqs if r["prompt_id"] == pid]
        p(f"  {pid}: {rs[0]['prompt']!r}")
        p(f"     input_tokens={sorted({r['input_tokens'] for r in rs})} "
          f"output_tokens={sorted({r['output_tokens'] for r in rs})} "
          f"distinct generated texts={len({r['generated_text'] for r in rs})} "
          f"gpus={sorted({r['gpu'] for r in rs})}")
        p(f"     text: {rs[0]['generated_text']!r}")

    p()
    p("2. Latency and cache counters per request")
    groups = defaultdict(list)
    for r in reqs:
        groups[(r["prompt_id"], r["prev_prompt_id"] or "")].append(r)
    p(f"  {'condition':<16}{'n':>3}{'median s':>10}{'mean s':>9}{'sd s':>8}"
      f"{'min s':>8}{'max s':>8}{'H2D GiB':>9}{'hits':>7}{'misses':>8}{'hit %':>7}")
    medians = {}
    for (pid, prev), rs in sorted(groups.items()):
        lat = [r["latency_s"] for r in rs]
        acc = sum(r["accesses"] for r in rs)
        hits = sum(r["hits"] for r in rs)
        medians[(pid, prev)] = (st.median(lat), st.mean(r["h2d_bytes"] for r in rs) / GiB)
        p(f"  {condition(pid, prev or None):<16}{len(rs):>3}{st.median(lat):>10.3f}"
          f"{st.mean(lat):>9.3f}{(st.stdev(lat) if len(lat) > 1 else 0):>8.3f}"
          f"{min(lat):>8.3f}{max(lat):>8.3f}"
          f"{medians[(pid, prev)][1]:>9.2f}{hits / len(rs):>7.0f}"
          f"{(acc - hits) / len(rs):>8.0f}{100 * hits / acc:>7.1f}")
    p()
    p("  Change vs. the same prompt run cold (median latency, mean H2D):")
    for (pid, prev), (med, h2d) in sorted(medians.items()):
        if prev and (pid, "") in medians:
            cmed, ch2d = medians[(pid, "")]
            p(f"    {condition(pid, prev):<14} {med - cmed:+.3f} s ({100 * (med - cmed) / cmed:+.1f}%)"
              f"   H2D {h2d - ch2d:+.2f} GiB")

    p()
    p("3. Per-request misses and H2D, by phase (mean over trials)")
    phase_sums = defaultdict(lambda: defaultdict(float))
    runs_per_cond = defaultdict(set)
    expert_sets = defaultdict(set)
    first_run = {}
    for path in sorted(folder.glob("trial*_layers.csv")):
        for row in csv.DictReader(path.open()):
            order = row["run"].split("_")[1]
            pos = int(row["request"][0])
            pid = order[pos - 1]
            prev = order[pos - 2] if pos > 1 else ""
            key = (pid, prev)
            runs_per_cond[key].add(row["run"])
            s = phase_sums[key]
            s[row["phase"] + "_misses"] += int(row["misses"])
            s[row["phase"] + "_h2d"] += int(row["h2d_bytes"])
            s[row["phase"] + "_ms"] += float(row["moe_ms"])
            # Routing is deterministic, so one run per prompt gives its expert set.
            if first_run.setdefault(pid, row["run"]) == row["run"]:
                expert_sets[pid].update(
                    (row["phase"], int(row["layer"]), int(e)) for e in row["experts"].split())
    p(f"  {'condition':<16}{'prefill miss':>13}{'prefill GiB':>12}{'prefill ms':>11}"
      f"{'decode miss':>12}{'decode GiB':>11}{'decode ms':>10}")
    for key, s in sorted(phase_sums.items()):
        n = len(runs_per_cond[key])
        p(f"  {condition(key[0], key[1] or None):<16}{s['prefill_misses'] / n:>13.0f}"
          f"{s['prefill_h2d'] / n / GiB:>12.2f}{s['prefill_ms'] / n:>11.0f}"
          f"{s['decode_misses'] / n:>12.0f}{s['decode_h2d'] / n / GiB:>11.2f}{s['decode_ms'] / n:>10.0f}")
    p("  (ms = summed MoE-layer wall time; layers are not CUDA-synchronized, so it is approximate)")

    if {"A", "B"} <= expert_sets.keys():
        p()
        p("4. Expert overlap between A and B (distinct (layer, expert) pairs routed to)")
        for phase in ("prefill", "decode", None):
            a = {(l, e) for ph, l, e in expert_sets["A"] if phase in (None, ph)}
            b = {(l, e) for ph, l, e in expert_sets["B"] if phase in (None, ph)}
            both = a & b
            p(f"  {phase or 'all':<8} |A|={len(a):>5} |B|={len(b):>5} shared={len(both):>5}"
              f"  Jaccard={len(both) / len(a | b):.3f}"
              f"  shared/|A|={len(both) / len(a):.3f} shared/|B|={len(both) / len(b):.3f}")
        p("  (the cache holds ~2368 experts; out of 48 layers x 128 = 6144)")

    (folder / "summary.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
