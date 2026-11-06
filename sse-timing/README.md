# SSE timing: does the expert-cache decode-timing channel survive transport?

Follow-on to the steps 1-2 logging. Inside the server, a token's decode time
tracks its **expert-cache misses** (a miss = fetch an expert to the GPU,
~0.4 ms each). A remote attacker never sees per-token time -- only **SSE
chunks**, which a server can bundle, delay, or let the network jitter. This
measures whether the per-token timing is still readable at the user's edge.

Reuses the `ExpertCacheLogger` (steps 1-2, `moe-cache-logging/`) for the misses.

## How it works -- three clocks on every token

All use `time.perf_counter()` (comparable on one host; across hosts compare only
gaps between tokens).

| clock | set where | meaning |
|---|---|---|
| `t_gen` | server, in `TimedStreamer.put` | token produced inside the model |
| `t_send` | server, at SSE flush | token pushed onto the network |
| `t_arrival` | client, at raw-socket `recv()` | token received by the user |

**Timing alignment:** prefill emits token 1, decode step *j* emits token *j+1*,
so the gap arriving at token *k* reflects decode step *k-1*. `analyze.py` pairs
`gap[k]` with the misses at step *k-1*.

## Files

| file | role |
|---|---|
| `sse_server.py` | runs the model, streams tokens, stamps `t_gen`/`t_send`, logs misses (one request at a time, so no server-side bundling) |
| `sse_client.py` | raw-socket reader (no library buffering); stamps `t_arrival` |
| `analyze.py` | joins the three clocks + misses, prints the scores |
| `run_sse_timing.sh` | runs it all: server -> client A (cold) -> client B (warm) -> analyze |

## Run it

Inside a GPU salloc session, from the repo root:

```bash
bash sse-timing/run_sse_timing.sh
# or with your own prompts / length:
MAX_NEW=128 PROMPT_A="..." PROMPT_B="..." bash sse-timing/run_sse_timing.sh
```

Results land in `sse-timing/results/<RUN_TAG>/`.

## Reading the output -- three scores (Pearson / Spearman)

- **[1] gen vs misses** -- is the signal there? (sanity check)
- **[2] arrival vs gen** -- did the trip keep the timing shape?
- **[3] arrival vs misses** -- can a remote attacker read the misses? (the real goal)

Also printed: how many tokens arrived **bundled** into one chunk.

- Strong **[3]** -> the leak is remotely observable.
- Weak **[3]** with strong **[1]** -> the transport destroyed it.

## Result so far (run 20261106-180410, loopback, one request at a time)

Transport was **transparent**: zero bundling, arrival matched internal timing to
+/-0.2 ms ([2] ~ 1.0). So [3] equals [1]. The limit is the **mechanism**: misses
only *moderately* predict per-token time (r ~ 0.4-0.6), because they are a
minority of the ~132 ms per-token cost. Conclusion: the leak reaches the user
intact, but the signal is only moderately informative.

## Caveats & next steps

1. **`cache=cold`** only resets the usage counters, not GPU residency (a truly
   empty GPU needs a fresh process). The warm-B-after-A case is the realistic
   cross-user scenario anyway.
2. **Same-host timing.** Loopback, so [2] is a best case; a real network adds
   jitter -- trust the rank-based Spearman across hosts.
3. **Clean baseline.** One request at a time, to isolate transport. The realistic
   follow-up is the batching server (`api_server_v2.py`), which adds its own
   bundling on top.
