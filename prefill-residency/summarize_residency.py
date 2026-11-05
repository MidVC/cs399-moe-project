"""
Summarize a prefill-residency run.

    python prefill-residency/summarize_residency.py prefill-residency/results/<tag>

For each sequence it reads the probe request (the victim's re-run prefill) and
reports how many of the victim's prefill experts were still cache hits. Comparing
the no-attacker control with the attacker variants gives the experts the attacker
evicted. If resident-set snapshots or a per-expert debug log are present, it also
reports set-level survival and the shared-vs-unique split.
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def load_requests(folder):
    recs = [json.loads(l) for l in (folder / "requests.jsonl").open()]
    by_label = defaultdict(lambda: defaultdict(list))  # label -> trial -> [rec]
    for r in recs:
        by_label[r["label"]][r["trial"]].append(r)
    return by_label


def layer_rows(folder, label, trial):
    p = folder / f"{label}_trial{trial:02d}_layers.csv"
    return list(csv.DictReader(p.open())) if p.exists() else []


def expert_set(rows, request_tag, phase="prefill"):
    """Set of (layer, expert) the given request routed to in the given phase."""
    s = set()
    for row in rows:
        if row["request"] == request_tag and row["phase"] == phase:
            for e in row["experts"].split():
                s.add((int(row["layer"]), int(e)))
    return s


def probe_counts(rows, request_tag):
    hits = misses = acc = 0
    for row in rows:
        if row["request"] == request_tag:  # probe is ntok=1 -> all prefill
            hits += int(row["hits"]); misses += int(row["misses"]); acc += int(row["n_experts"])
    return hits, misses, acc


def decode_key(k):
    return (k >> 32, k & 0xFFFFFFFF) if isinstance(k, int) else tuple(k)


def main():
    folder = Path(sys.argv[1])
    by_label = load_requests(folder)
    out = []

    def pr(s=""):
        out.append(s); print(s)

    pr(f"Prefill-residency results: {folder}")
    pr()
    pr("Each sequence: victim runs fully, then (optionally) the attacker, then the")
    pr("victim's prefill is re-run as a probe. Probe HIT = that victim expert is still")
    pr("resident; probe MISS = it was evicted. Counts are summed over the 48 layers.")
    pr()

    # Describe each label and gather probe stats.
    summary = {}
    for label, trials in sorted(by_label.items()):
        any_trial = next(iter(trials.values()))
        steps = sorted(any_trial, key=lambda r: r["pos"])
        victim = steps[0]["pid"]
        probe = next(r for r in steps if r["role"] == "probe")
        attackers = [r for r in steps if r["role"] == "attacker"]
        if not attackers:
            atk = f"{victim} control (no attacker)"
        else:
            a = attackers[0]
            atk = f"{a['pid']} {'prefill' if a['ntok'] == 1 else 'full (%d tok)' % a['ntok']}"
        hits_l, miss_l, acc_l = [], [], []
        for trial in trials:
            rows = layer_rows(folder, label, trial)
            h, m, a = probe_counts(rows, probe["request_tag"])
            if h + m:
                hits_l.append(h); miss_l.append(m); acc_l.append(a)
        if not hits_l:
            continue
        summary[label] = dict(
            victim=victim, attacker=atk, is_control=not attackers,
            n=len(hits_l), hits=round(st.mean(hits_l)), misses=round(st.mean(miss_l)),
            acc=round(st.mean(acc_l)),
        )

    pr("Victim-expert survival (probe prefill):")
    pr(f"  {'sequence':<8}{'victim':>7}{'attacker':>22}{'trials':>7}"
       f"{'accesses':>10}{'survived':>10}{'evicted':>9}{'survived %':>12}")
    for label, s in sorted(summary.items(), key=lambda kv: (kv[1]["victim"], not kv[1]["is_control"])):
        pr(f"  {label:<8}{s['victim']:>7}{s['attacker']:>22}{s['n']:>7}"
           f"{s['acc']:>10}{s['hits']:>10}{s['misses']:>9}{100 * s['hits'] / s['acc']:>11.1f}%")

    pr()
    pr("Attacker's marginal effect (vs the same victim's no-attacker control):")
    controls = {s["victim"]: s for s in summary.values() if s["is_control"]}
    for label, s in sorted(summary.items(), key=lambda kv: kv[1]["victim"]):
        if s["is_control"] or s["victim"] not in controls:
            continue
        c = controls[s["victim"]]
        evicted = c["hits"] - s["hits"]
        pr(f"  victim {s['victim']}, attacker {s['attacker']:<18} "
           f"survived {s['hits']} vs {c['hits']} (control): attacker evicted "
           f"{evicted} more of the victim's experts "
           f"({100 * evicted / c['acc']:.1f}% of its {c['acc']}-expert prefill set)")

    # Optional: exact set-level survival from resident snapshots.
    snap_lines = []
    for label, trials in sorted(by_label.items()):
        for trial in sorted(trials):
            sp = folder / f"{label}_trial{trial:02d}_resident.json"
            if not sp.exists():
                continue
            snaps = {k: {decode_key(x) for x in v} for k, v in json.loads(sp.read_text()).items()}
            rows = layer_rows(folder, label, trial)
            steps = sorted(trials[trial], key=lambda r: r["pos"])
            vtag = steps[0]["request_tag"]
            vset = expert_set(rows, vtag)
            if not vset:
                continue
            line = f"  {label} trial {trial}: victim prefill set={len(vset)}"
            for r in steps:
                snap = snaps.get(r["request_tag"])
                if snap is not None:
                    line += f" | after {r['request_tag']}({r['role']}): {len(vset & snap)} of victim resident"
            snap_lines.append(line)
    if snap_lines:
        pr()
        pr("Exact set-level survival (engine exposed the resident set):")
        for l in snap_lines:
            pr(l)
    else:
        pr()
        pr("(No resident-set snapshots: the engine did not expose a way to list resident")
        pr(" experts, so survival is from the reload probe above. See api lines in driver.log.)")

    # Optional: shared-vs-unique split from the per-expert debug log.
    ev_lines = []
    for label, trials in sorted(by_label.items()):
        s = summary.get(label)
        if not s or s["is_control"]:
            continue
        for trial in sorted(trials):
            evp = folder / f"{label}_trial{trial:02d}_expert_events.csv"
            if not evp.exists():
                continue
            steps = sorted(trials[trial], key=lambda r: r["pos"])
            try:
                r = analyze_debug(evp, steps)
            except Exception as e:
                ev_lines.append(f"  {label} trial {trial}: debug parse failed ({e})")
                continue
            ev_lines.append(
                f"  {label} tr{trial}: victim used {r['victim']} experts (prefill+decode). "
                f"Attacker prefill hit {r['attacker_hits']}/{r['attacker']} experts already resident "
                f"({r['attacker_hits_in_victim']} of them victim's). "
                f"Probe: survived {r['surv_shared']}/{r['shared']} shared + "
                f"{r['surv_uniq']}/{r['uniq']} victim-unique")
    if ev_lines:
        pr()
        pr("Shared-vs-unique survival (per-expert debug log):")
        for l in ev_lines:
            pr(l)

    (folder / "summary.txt").write_text("\n".join(out) + "\n")


def analyze_debug(events_csv, steps):
    """From the raw per-expert debug stdout log (markers + C++ HIT/MISS lines),
    answer two things for a victim->attacker->probe sequence:
      * attacker segment: of the attacker's prefill accesses, how many HIT (were
        already resident from the victim) -- i.e. victim experts still resident
        when the attacker's prefill ran.
      * probe segment: of the victim's experts, how many survived (HIT), split by
        whether the attacker also used them (shared) vs victim-unique.
    Field order differs between hit and miss lines, so layer/expert are parsed
    independently rather than as one ordered match."""
    import re
    stdout = str(events_csv).replace("_expert_events.csv", "_stdout.log")
    victim_tag = steps[0]["request_tag"]
    attacker_tag = next(r["request_tag"] for r in steps if r["role"] == "attacker")
    probe_tag = next(r["request_tag"] for r in steps if r["role"] == "probe")
    lay = re.compile(r"layer_idx\s+(-?\d+)")
    exp = re.compile(r"expert_idx\s+(-?\d+)")

    cur = None
    touched = defaultdict(set)          # tag -> {(layer,expert) accessed}
    hit_set = defaultdict(set)          # tag -> {(layer,expert) that were HITs}
    for raw in open(stdout, errors="replace"):
        if raw.startswith("###REQ"):
            m = re.search(r"pos=(\d+) pid=(\w+) ntok=(\d+)", raw)
            cur = f"{m.group(1)}{m.group(2)}n{m.group(3)}" if m else None
            continue
        if cur is None:
            continue
        is_hit_now = "cache_hit_immediate" in raw
        is_fetch = "GPUFetchFunc: gpu_id" in raw
        if not (is_hit_now or is_fetch):
            continue
        ml, me = lay.search(raw), exp.search(raw)
        if not (ml and me):
            continue
        key = (int(ml.group(1)), int(me.group(1)))
        touched[cur].add(key)
        if is_hit_now or " cache_hit  1" in raw or " cache_hit 1" in raw:
            hit_set[cur].add(key)

    vset = touched[victim_tag]
    aset = touched[attacker_tag]
    a_hits = hit_set[attacker_tag]
    shared = vset & aset
    uniq = vset - aset
    probe_surv = hit_set[probe_tag]
    return dict(
        victim=len(vset), attacker=len(aset),
        attacker_hits=len(a_hits), attacker_hits_in_victim=len(a_hits & vset),
        shared=len(shared), uniq=len(uniq),
        surv_shared=len(shared & probe_surv), surv_uniq=len(uniq & probe_surv),
    )


if __name__ == "__main__":
    main()
