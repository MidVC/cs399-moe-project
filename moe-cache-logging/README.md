# Expert-cache code map and logging: MoE-Infinity 0.2.2 + Qwen3-30B-A3B

Setup: `w00606/moe-setup` (commit df9c58a). The wheels are `moe_infinity-0.2.2` and `moe_store-0.2.2`. The wheel's Python files match `EfficientMoE/MoE-Infinity@v0.2.2` exactly, and that tag is the source for the C++ line numbers below.
Config from `smoke.py`: `use_native_engine=False`, `device_memory_ratio=0.5`, `num_threads=4`, with prefetch, adaptive precision and the phase policy all **off** (defaults). KV cache is on (`use_cache=True` → HF `DynamicCache`).

---

## Part 1: where residency is checked, and where experts are loaded and evicted

### Call path for one MoE layer

```
MoE.generate                         moe_infinity/entrypoints/big_modeling.py:1278-1282
  └ HF model.generate(...)           (use_native_engine=False → plain HF generate loop)
     └ Qwen3MoEBlock.forward          moe_store/wrappers/qwen.py:48-85
        router softmax → top-8 → router_mask
        executor.dispatch_local(...)  → executor.wait_dispatch_local()
           └ DistributedExpertExecutor   moe_infinity/distributed/expert_executor.py
              dispatch_local        :402   set_inputs → _dispatch_eager_local
              _dispatch_eager_local :353   for each selected expert: dispatcher.enqueue_expert(L, E, gpu)
              wait_dispatch_local   :580   dispatcher.wait_expert()  (barrier)
                 └ C++ ExpertDispatcher  core/parallel/expert_dispatcher.cpp  (built into moe_infinity/_store*.so)
```

The `Qwen3MoeSparseMoeBlock` → `Qwen3MoEBlock` swap happens in `moe_infinity/boundary/transformers_model_overrides.py:55-57`. The wiring (`module.expert_executor`, `layer_id`) is in `moe_infinity/runtime/model_offload.py:1324-1361`. The dispatcher is created at `model_offload.py:1234`.

### 1. Residency check: is the selected expert on the GPU?

| What | Where |
|---|---|
| **Main hit/miss check** | `ExpertDispatcher::Enqueue`, `expert_dispatcher.cpp:616`. Line **649**: `if (expert_node->node->device.is_cuda())` → **hit**: goes straight to `exec_queue_`, bumps `cache_hit_count_`, logs `cache_hit_immediate` (line 653). `else` → **miss**: goes to `input_queue_` for the fetch thread (line 697). |
| Re-check in the fetch worker | `GPUFetchFunc`, line **1155**: `bool cache_hit = expert_node->node->device.is_cuda();` |
| Residency state itself | `Node::device` (`core/model/model_topology.h`). It is CUDA when the expert's weights are on the GPU and CPU otherwise. The set of cached keys is in `cached_experts_[gpu]` (key = `layer<<32 \| expert`), and free budget is in `cache_sizes_[gpu]`. |
| Counters | `GetCacheHitRate` :917, `ClearExpertCacheCounts` :741. Python: `dispatcher.get_cache_hit_rate()`. `ArcherTopologyHandle::GetResidentSparseExpertCount` (model_topology.cpp:387) → Python `archer_engine.get_resident_expert_count()` |

### 2. Loading an expert (CPU → GPU)

| What | Where |
|---|---|
| Fetch worker thread (one per GPU) | `ExpertDispatcher::GPUFetchFunc`, `expert_dispatcher.cpp:1030` |
| Reserve a cache slot | lines 1295-1301: `cache_sizes_ -= bytes; cached_experts_.insert(key)` |
| **The copy** | line **1304**: `expert_node->node->SetDevice(device, true, stream, &transfer_done)` → `Node::SetDevice`, `core/model/model_topology.cpp:60`. That does `CudaMemcpyAsync(H2D)` from pinned host memory (line 195) on the fetch stream and returns a CUDA event. |
| Compute waits for the copy | `GPUExecFunc` :1370 does `cudaStreamWaitEvent(exec_stream, transfer_event)` (≈1435) |
| Where experts start | `model_topology.cpp:673` ("Moving sparse parameters to CPU"). **All 6,144 experts are loaded into host RAM at start-up, and the GPU expert cache starts empty.** Every miss is therefore a PCIe H2D copy, never a disk read. |
| Cache budget | `cache_sizes_` is initialised at :352 from `GetSparseCacheLimit` (model_topology.cpp:915) = `device_memory_ratio × total GPU memory − dense weights on GPU` ≈ 0.5 × 48 GB − dense. Each Qwen3 expert is 3 × 2048 × 768 × 2 B ≈ **9 MiB**, so roughly a third of the experts fit. |

### 3. Eviction (GPU → CPU)

| What | Where |
|---|---|
| Trigger | `GPUFetchFunc` line **1190**: on a miss with `cache_sizes_[gpu] < expert bytes` (budget full) |
| **Victim choice** | `ExpertDispatcher::FindExpertEvict`, `expert_dispatcher.cpp:931`. It picks the cached expert with the smallest `incache_visit_count` that is idle and has no pending dispatch. |
| Evict | line **1276**: `evict_node->SetDevice(evict_node->default_host)` frees the GPU buffer and points the tensors back at the host copy (the host copy is kept, so no D2H copy is needed). Then `cache_sizes_ += bytes` and `cached_experts_.erase` (1285). |
| Prefill overflow (no eviction) | If the cache is full **and the batch has >1 token** (prefill), lines 1191-1205 skip eviction. The expert is loaded as a one-off (`gpu_overload_`, log line "overloading expert cache") and dropped again after it runs, in `RetirementFunc` :2177 / `if (args.evict)` :2181. |
| Not active in your config | Prefetch path (`core/prefetch/task_scheduler.cpp`, `ArcherPrefetchHandle`), `ExpertResidencyManager` (`core/prefetch/expert_residency.cpp`, only when `phase_specific_expert_policy` / adaptive precision is on), and the Python `memory/offloading_policy.py` (LRU/ARC), which the C++ dispatcher doesn't use here. |

**Policy detail that matters for Part 5:** `incache_visit_count` is incremented only when an expert is *loaded* (lines 1118, 1305), not on a fast-path hit in `Enqueue`. So the eviction policy is "fewest times loaded", ties broken by `unordered_set` order. It is not true LFU over accesses. The counts also persist across requests and are never aged, so the GPU cache state left by prompt A directly changes the hits and evictions for prompt B.

---

## Part 2: logs

The stock build **does** have per-expert logs, but they are hidden at `DEBUG` level. There are three layers of logging, from zero effort to a rebuild:

| Level | How | You get |
|---|---|---|
| **A. Built-in C++ debug log** (no code change) | `export MUDUO_LOG_DEBUG=1` (read by `core/base/logging.cc:46`) | One stdout line per expert access: `HIT` (`cache_hit_immediate`), `MISS` (`GPUFetchFunc ... cache_hit 0`), `EVICT_FOR`, `OVERFLOW_LOAD`, with µs timestamps. **Limitation:** the evict line prints the *incoming* expert, not the victim, and there is no transfer timing. `parse_debug_log.py` turns it into CSV. |
| **B. `expert_cache_logger.py`** (Python, no rebuild) | wraps `executor.dispatch_local/wait_dispatch_local` | One CSV row per (request, forward step, MoE layer): selected experts, exact hit/miss counts, resident count before/after, inferred evictions, H2D bytes, MoE-layer time. |
| **C. `cpp_patch/`** (optional rebuild of `_store.so`) | `MOE_EXPERT_EVENT_LOG=path.csv` | Per-expert `HIT / MISS / EVICT` (**victim** layer/expert + its visit count) `/ OVERFLOW_LOAD / OVERFLOW_EVICT / H2D_SUBMIT / H2D_DONE` (**GPU-timed copy ms** from CUDA events). |

### Run A + B (works with your current install)

Copy this folder into your `moe-setup` checkout, next to `run.sh`, then in your `salloc` session:

```bash
bash moe-cache-logging/run_logged.sh                 # 3 repeats of the smoke prompt
N_REPEATS=5 SYNC_LAYERS=1 bash moe-cache-logging/run_logged.sh   # accurate per-layer ms
```

Outputs in `moe-cache-logging/logs/`:
- `<tag>_requests.csv`: latency, input/output tokens, H2D bytes, resident experts, generated text, per request
- `<tag>_layers.csv`: per-MoE-layer hit/miss/evictions/H2D bytes (logger B)
- `<tag>_stdout.log` → `<tag>_expert_events.csv`: per-expert events (debug log A)

Expected pattern: `r0` (cold cache) is all misses with large H2D. `r1`, `r2` should be mostly hits, until the ~⅓-of-experts budget fills and evictions start.

Notes:
- `SYNC_LAYERS=1` adds `cuda.synchronize()` around each MoE layer. Use it for per-layer attribution, not for latency distributions.
- The logger calls `clear_expert_cache_counts()` once when attached. That also resets `incache_visit_count`, which is harmless right after model load (everything is 0) but would perturb eviction if you attach mid-run.
- The hit/miss counts in B assume prefetch is off (true here), so dispatcher accesses = experts enqueued.

### Optional C: patched build

```bash
module load cuda/12.8            # whatever provides nvcc 12.8 on your cluster
bash moe-cache-logging/cpp_patch/build_patched_store.sh
MOE_EXPERT_EVENT_LOG=$PWD/moe-cache-logging/logs/events.csv bash moe-cache-logging/run_logged.sh
bash moe-cache-logging/cpp_patch/build_patched_store.sh --restore   # undo
```

The patch only adds logging calls behind an env-var check in `expert_dispatcher.cpp`, and it is a no-op when the variable is unset. It passes a syntax-only compile against the v0.2.2 sources but **has not been built or run on a GPU**, so check the `PATCH_PRESENT` / import lines the script prints.
