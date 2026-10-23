"""
Per-layer expert-cache logger for MoE-Infinity 0.2.2 (no rebuild needed).

It wraps the shared DistributedExpertExecutor that every Qwen3MoEBlock calls:

    Qwen3MoEBlock.forward
      -> executor.dispatch_local(layer_id, hidden, router_mask, ...)   # enqueue experts
      -> executor.wait_dispatch_local()                                # barrier

For every MoE-layer call it writes one CSV row with:
  * which experts the router selected for this layer
  * how many of them were GPU cache hits / misses   (from the C++ dispatcher's
    cache_hit_count_/cache_access_count_ via get_cache_hit_rate())
  * resident expert count before/after              (Node::device.is_cuda())
  * H2D expert bytes moved during this layer         (expert_h2d_bytes_total_)
  * inferred evictions = misses - delta(resident)
  * MoE-layer wall time (optionally CUDA-synchronised)

The per-*expert* hit/miss identity and eviction lines come from the C++ debug log
(MUDUO_LOG_DEBUG=1, see parse_debug_log.py). This file only gives exact counts.

Usage:
    from expert_cache_logger import ExpertCacheLogger
    log = ExpertCacheLogger(model, "logs/run1_layers.csv", sync_layers=False)
    with log.request("A", run=0):
        model.generate(...)
    log.close()
"""

from __future__ import annotations

import csv
import os
import time
from contextlib import contextmanager
from typing import Optional

import torch


class ExpertCacheLogger:
    FIELDS = [
        "run", "request", "fwd_step", "phase", "layer", "n_tokens",
        "n_experts", "experts",
        "hits", "misses", "cum_hit_rate",
        "resident_before", "resident_after", "inferred_evictions",
        "h2d_bytes", "occupancy_bytes_after",
        "t_start_s", "moe_ms",
    ]

    def __init__(self, moe_model, csv_path: str, sync_layers: bool = False):
        """
        moe_model   : the moe_infinity.MoE object (has .engine)
        sync_layers : if True, torch.cuda.synchronize() before/after each MoE
                      layer so moe_ms includes the async H2D copy + expert GEMMs.
                      This perturbs end-to-end latency, so keep it False for
                      latency experiments and True for per-layer attribution.
        """
        eng = moe_model.engine
        self.executor = eng.expert_executor
        self.dispatcher = eng.expert_dispatcher
        self.archer = eng.archer_engine
        self.sync = sync_layers

        if getattr(self.executor, "prefetcher", None) is not None:
            print("[ExpertCacheLogger] WARNING: speculative prefetch is ON; "
                  "hit counts include prefetched experts.", flush=True)

        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        self._f = open(csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()

        # Reset dispatcher counters so cum accesses == what we count here.
        self.dispatcher.clear_expert_cache_counts()
        self._cum_access = 0
        self._cum_hits = 0

        self.run = ""
        self.request_tag = ""
        self.fwd_step = -1
        self._last_layer = 10**9
        self._pending = None
        self._t0 = time.perf_counter()

        self._orig_dispatch = self.executor.dispatch_local
        self._orig_wait = self.executor.wait_dispatch_local
        self.executor.dispatch_local = self._dispatch_local
        self.executor.wait_dispatch_local = self._wait_dispatch_local

    # ---------------------------------------------------------------- helpers
    def _resident(self) -> int:
        return int(self.archer.get_resident_expert_count())

    def _h2d(self) -> int:
        return int(self.archer.get_expert_h2d_bytes_total())

    # ------------------------------------------------------------ the wraps
    def _dispatch_local(self, layer_id, hidden_states, router_mask,
                        router_weights, router_logits=None, prefetcher=None):
        if layer_id is not None and layer_id <= self._last_layer:
            self.fwd_step += 1            # new forward pass (prefill or 1 decode step)
        self._last_layer = layer_id if layer_id is not None else -1

        num_expert = router_mask.shape[-1]
        active = torch.any(router_mask.view(-1, num_expert).bool(), dim=0)
        experts = torch.nonzero(active).flatten().tolist()
        n_tokens = int(hidden_states.shape[0])

        if self.sync:
            torch.cuda.synchronize()
        self._pending = dict(
            layer=layer_id, n_tokens=n_tokens, experts=experts,
            resident_before=self._resident(), h2d_before=self._h2d(),
            t=time.perf_counter(),
        )
        return self._orig_dispatch(layer_id, hidden_states, router_mask,
                                   router_weights, router_logits=router_logits,
                                   prefetcher=prefetcher)

    def _wait_dispatch_local(self):
        out = self._orig_wait()
        if self.sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        p, self._pending = self._pending, None
        if p is None:
            return out

        n = len(p["experts"])
        self._cum_access += n
        rate = float(self.dispatcher.get_cache_hit_rate())
        cum_hits = int(round(rate * self._cum_access))
        hits = cum_hits - self._cum_hits
        self._cum_hits = cum_hits
        misses = n - hits

        res_after = self._resident()
        self._w.writerow(dict(
            run=self.run, request=self.request_tag, fwd_step=self.fwd_step,
            phase="prefill" if p["n_tokens"] > 1 else "decode",
            layer=p["layer"], n_tokens=p["n_tokens"], n_experts=n,
            experts=" ".join(map(str, p["experts"])),
            hits=hits, misses=misses, cum_hit_rate=f"{rate:.6f}",
            resident_before=p["resident_before"], resident_after=res_after,
            inferred_evictions=max(0, misses - (res_after - p["resident_before"])),
            h2d_bytes=self._h2d() - p["h2d_before"],
            occupancy_bytes_after=int(self.archer.get_expert_occupancy_bytes()),
            t_start_s=f"{p['t'] - self._t0:.6f}",
            moe_ms=f"{(t1 - p['t']) * 1e3:.4f}",
        ))
        return out

    # ------------------------------------------------------------- public API
    @contextmanager
    def request(self, tag: str, run=""):
        """Tag every layer row produced inside this block."""
        self.request_tag, self.run = tag, run
        self.fwd_step, self._last_layer = -1, 10**9
        try:
            yield self
        finally:
            self._f.flush()
            self.request_tag = ""

    def snapshot(self) -> dict:
        """Cheap global counters, e.g. to store next to each request's latency."""
        return dict(
            resident_experts=self._resident(),
            h2d_bytes_total=self._h2d(),
            occupancy_bytes=int(self.archer.get_expert_occupancy_bytes()),
            cum_hit_rate=float(self.dispatcher.get_cache_hit_rate()),
            cum_accesses=self._cum_access,
        )

    def close(self):
        self.executor.dispatch_local = self._orig_dispatch
        self.executor.wait_dispatch_local = self._orig_wait
        self._f.close()
