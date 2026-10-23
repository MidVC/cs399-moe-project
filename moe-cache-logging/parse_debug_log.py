"""
Turn MoE-Infinity's built-in C++ debug output (enabled with MUDUO_LOG_DEBUG=1)
into a per-expert event CSV.

Source lines (core/parallel/expert_dispatcher.cpp, v0.2.2):
  Enqueue()       "cache_hit_immediate: expert_idx E layer_idx L gpu_id G"        -> HIT
  GPUFetchFunc()  "ExpertDispatcher::GPUFetchFunc: gpu_id G layer_idx L expert_idx E
                   cache_hit 0|1 cache_size B incache count N"                     -> MISS / HIT_LATE
  GPUFetchFunc()  "evicting expert: gpu_id G cache size B incache count N
                   layer_idx L expert_idx E"                                       -> EVICT_FOR
                   (L/E are the *incoming* expert that needs the slot; the victim's
                    id is NOT printed by the stock build)
  GPUFetchFunc()  "overloading expert cache: ..."                                  -> OVERFLOW_LOAD
                   (prefill with a full cache: expert is loaded, used once, and
                    dropped again after execution instead of evicting another one)
  GPUFetchFunc()  "All cached expert locked, waiting ..."                          -> ALL_LOCKED

Usage:
  python parse_debug_log.py logs/run1_stdout.log logs/run1_expert_events.csv
"""

import csv
import re
import sys

LINE = re.compile(
    r"^\[(?P<date>\d{8}) (?P<time>\d{2}:\d{2}:\d{2}\.\d{6})Z?\s*\]"
    r"\[\s*(?P<tid>\d+)\s*\]\[(?P<lvl>\w+)\s*\](?:\[(?P<func>[^\]]*)\])?\s*(?P<msg>.*)$"
)


def kv(msg, key):
    m = re.search(re.escape(key) + r"\s+(-?\d+)", msg)
    return int(m.group(1)) if m else ""


def classify(msg):
    if "cache_hit_immediate" in msg:
        return "HIT"
    if "GPUFetchFunc: gpu_id" in msg:
        return "MISS" if kv(msg, "cache_hit") == 0 else "HIT_LATE"
    if "evicting expert" in msg:
        return "EVICT_FOR"
    if "overloading expert cache" in msg:
        return "OVERFLOW_LOAD"
    if "All cached expert locked" in msg:
        return "ALL_LOCKED"
    return None


def main(src, dst):
    n = 0
    with open(src, errors="replace") as fin, open(dst, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["seq", "time", "tid", "event", "layer", "expert", "gpu",
                    "cache_free_bytes", "incache_count"])
        for raw in fin:
            m = LINE.match(raw.rstrip("\n"))
            if not m:
                continue
            msg = m.group("msg")
            ev = classify(msg)
            if ev is None:
                continue
            free = kv(msg, "cache_size") if ev == "MISS" else kv(msg, "cache size")
            w.writerow([n, f"{m.group('date')} {m.group('time')}", m.group("tid"),
                        ev, kv(msg, "layer_idx"), kv(msg, "expert_idx"),
                        kv(msg, "gpu_id"), free, kv(msg, "incache count")])
            n += 1
    print(f"wrote {n} events -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
