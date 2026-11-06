"""
Task 3 — ANALYSIS. Joins the three per-token clocks with the cache misses and
scores whether the timing signal survives transport.

Inputs (in --logdir):
  <tag>_server_tokens.csv     step, t_gen, t_send      (server)
  <tag>_client_arrivals.csv   step, t_arrival, recv_idx (client)
  <layers-tag>_layers.csv     per-step misses          (ExpertCacheLogger)

Timing alignment (the one subtle bit):
  generate's prefill emits token 1; decode step j emits token j+1. So the gap
  arriving at token k reflects the forward pass that made it -- decode step k-1.
  We therefore pair gap[k] with the misses at fwd_step (k-1).

Prints three correlations, each as Pearson / Spearman:
  [1] gen vs misses     -- is the signal there?            (mechanism)
  [2] arrival vs gen     -- did transport keep the shape?  (fidelity)
  [3] arrival vs misses  -- can a remote attacker read it? (end-to-end)
plus how much the transport bundled tokens. Writes <tag>_task3_joined.csv.
"""
import argparse, csv, collections, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--layers-tag", default=None, help="RUN_TAG of _layers.csv (default: --tag)")
ap.add_argument("--logdir", default="logs")
a = ap.parse_args()
L = Path(a.logdir)
layers_tag = a.layers_tag or a.tag

def rd(p):
    with open(p) as f: return list(csv.DictReader(f))

srv = {int(r["step"]): r for r in rd(L / f"{a.tag}_server_tokens.csv")}
cli = {}
for r in rd(L / f"{a.tag}_client_arrivals.csv"):
    cli.setdefault(int(r["step"]), r)        # first arrival wins

# misses per decode step for this request
miss = collections.Counter()
for r in rd(L / f"{layers_tag}_layers.csv"):
    if r["request"] != a.tag or r["phase"] != "decode":
        continue
    miss[int(r["fwd_step"])] += int(r["misses"])

steps = sorted(s for s in srv if s in cli)   # tokens seen by both server and client
if len(steps) < 3:
    raise SystemExit(f"too few aligned steps ({len(steps)}); "
                     f"srv={len(srv)} cli={len(cli)} miss_steps={len(miss)}")

def gaps(key, src):
    """inter-token gap for each step = this token's time minus the previous one's."""
    out = {}
    for i, s in enumerate(steps):
        if i == 0: continue
        out[s] = float(src[s][key]) - float(src[steps[i-1]][key])
    return out

gen_gap = gaps("t_gen", srv)
send_gap = gaps("t_send", srv)
arr_gap = gaps("t_arrival", cli)

def pearson(xs, ys):                          # linear correlation, -1..+1
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    return cov/math.sqrt(vx*vy) if vx>0 and vy>0 else float("nan")

def spearman(xs, ys):                         # rank correlation (ignores outliers/scale)
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r=[0]*len(v)
        i=0
        while i < len(v):
            j=i
            while j+1<len(v) and v[order[j+1]]==v[order[i]]: j+=1
            avg=(i+j)/2+1
            for k in range(i,j+1): r[order[k]]=avg
            i=j+1
        return r
    return pearson(rank(xs), rank(ys))

# pair each token's gap with the misses from the step that produced it (k-1)
ds = [s for s in steps[1:] if (s-1) in miss]
if len(ds) < 3:
    raise SystemExit(f"too few steps with a matching decode fwd_step: {len(ds)}")
m  = [miss[s-1] for s in ds]
gg = [gen_gap[s]*1000 for s in ds]            # ms
sg = [send_gap[s]*1000 for s in ds]
ag = [arr_gap[s]*1000 for s in ds]

rows = []
for i,s in enumerate(ds):
    rows.append(dict(step=s, misses=miss[s-1],
        gen_gap_ms=f"{gg[i]:.3f}", send_gap_ms=f"{sg[i]:.3f}",
        arrival_gap_ms=f"{ag[i]:.3f}",
        recv_idx=cli[s]["recv_idx"], bytes_in_recv=cli[s]["bytes_in_recv"]))
with open(L / f"{a.tag}_task3_joined.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# bundling: tokens that shared one recv(), and gaps that collapsed to ~0
by_recv = collections.Counter(cli[s]["recv_idx"] for s in steps)
multi = sum(1 for v in by_recv.values() if v>1)
zero_gaps = sum(1 for x in ag if x < 0.1)

print(f"=== Task 3: {a.tag}  (aligned decode steps: {len(ds)}) ===")
print(f"mean misses/step       {sum(m)/len(m):.1f}   range {min(m)}..{max(m)}")
print(f"mean gen_gap     {sum(gg)/len(gg):.2f} ms")
print(f"mean send_gap    {sum(sg)/len(sg):.2f} ms")
print(f"mean arrival_gap {sum(ag)/len(ag):.2f} ms")
print(f"recvs carrying >1 token: {multi}/{len(by_recv)}   "
      f"arrival gaps <0.1ms (coalesced): {zero_gaps}/{len(ag)}")
print()
print("correlation (Pearson / Spearman):")
print(f"  [1] gen_gap   vs misses   {pearson(gg,m):+.3f} / {spearman(gg,m):+.3f}   (mechanism)")
print(f"  [2] arrival   vs gen_gap  {pearson(ag,gg):+.3f} / {spearman(ag,gg):+.3f}   (transport fidelity)")
print(f"  [3] arrival   vs misses   {pearson(ag,m):+.3f} / {spearman(ag,m):+.3f}   (end-to-end attacker)")
print(f"\njoined -> {L / (a.tag + '_task3_joined.csv')}")
