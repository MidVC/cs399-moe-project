"""
Task 3 — SERVER side.

Streams a model's answer over SSE and timestamps every token twice:
  t_gen  : when the model produces the token  (set in TimedStreamer.put)
  t_send : when that token's SSE frame is flushed to the socket

Both use time.perf_counter() (CLOCK_MONOTONIC on Linux), so they line up with
the client's arrival times on the same host.

Key design choice: ONE request at a time (a lock around generate). The server
never bundles tokens, so any blurring we later see comes from transport alone.
Model/engine config matches smoke_logged.py (use_native_engine=False), which
keeps the ExpertCacheLogger miss counts valid.

Outputs:
  <TAG>_server_tokens.csv   step, token_id, text, t_gen, t_send
  <RUN_TAG>_layers.csv      per-step cache misses (via ExpertCacheLogger)

Endpoint:  GET /generate?prompt=...&max_new_tokens=64&tag=run&cache=cold
"""
import csv, os, sys, threading, time, queue, json, urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from moe_infinity import MoE

# sse-timing/ is a top-level task folder. Reuse the shared logger from
# moe-cache-logging/ (steps 1-2), the same way the other task folders do.
_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_DIR))
sys.path.insert(0, str(_REPO_DIR / "moe-cache-logging"))
from expert_cache_logger import ExpertCacheLogger  # noqa: E402

LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8077"))

# ---- load the model once, with the validated smoke-run settings ----
assert torch.cuda.is_available(), "CUDA unavailable"
root = Path(os.environ["MOE_WORK_ROOT"])
checkpoint = root / "models" / "Qwen3-30B-A3B-ad44e777"
offload = root / "state" / "offload-qwen3-bf16-moe022"
offload.mkdir(parents=True, exist_ok=True)

print("loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
model = MoE(str(checkpoint), {
    "offload_path": str(offload), "device_memory_ratio": 0.5, "num_threads": 4,
    "use_native_engine": False, "adaptive_memory_enabled": False,
    "adaptive_expert_precision": False,
})

# One logger for the whole process. Its _layers.csv gathers every request
# (separated by tag) and records the per-step misses we correlate against timing.
LAYERS_CSV = LOG_DIR / f"{os.environ.get('RUN_TAG','sse-timing')}_layers.csv"
logger = ExpertCacheLogger(model, str(LAYERS_CSV), sync_layers=False)

_gen_lock = threading.Lock()   # one generation at a time -> no token bundling


class TimedStreamer(BaseStreamer):
    """Receives tokens from model.generate as they are produced.

    generate calls put() once with the prompt, then once per new token. We skip
    the prompt call, stamp t_gen on each token, and pass (step, id, t_gen) to the
    HTTP thread through a queue.
    """
    def __init__(self):
        self.q = queue.Queue()
        self.first = True
        self.step = 0

    def put(self, value):
        if self.first:                 # first call is the prompt, not a new token
            self.first = False
            return
        t = time.perf_counter()        # t_gen: token just produced
        for tid in value.view(-1).tolist():
            self.step += 1
            self.q.put((self.step, tid, t))

    def end(self):
        self.q.put(None)               # sentinel: generation finished


def warm_or_cold(cache):
    """cache='cold' resets the cache usage counters before the request.
    NOTE: clears the LFU counters only, not GPU residency. A truly empty GPU
    needs a fresh process (see README)."""
    if cache == "cold":
        try:
            model.engine.expert_dispatcher.clear_expert_cache_counts()
        except Exception:
            pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):         # silence default request logging
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/health":        # used by run_sse_timing.sh to wait for readiness
            self.send_response(200); self.send_header("Content-Length","2")
            self.end_headers(); self.wfile.write(b"ok"); return
        if u.path != "/generate":
            self.send_response(404); self.end_headers(); return

        # ---- parse the request ----
        qs = urllib.parse.parse_qs(u.query)
        prompt = qs.get("prompt", ["Explain what a GPU is."])[0]
        max_new = int(qs.get("max_new_tokens", ["64"])[0])
        tag = qs.get("tag", ["req"])[0]
        cache = qs.get("cache", ["warm"])[0]

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt",
                           add_special_tokens=False).to("cuda:0")

        # ---- open the SSE stream ----
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        # ---- generate on a worker thread; main thread streams the tokens out ----
        streamer = TimedStreamer()
        rows = []
        def worker():
            with _gen_lock:            # serialize: one request at a time
                warm_or_cold(cache)
                with logger.request(tag, run=os.environ.get("RUN_TAG","sse-timing")), \
                     torch.no_grad():
                    model.generate(inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new, do_sample=False, use_cache=True,
                        streamer=streamer, pad_token_id=tokenizer.eos_token_id)
        th = threading.Thread(target=worker, daemon=True)
        th.start()

        def chunk(b: bytes):           # write one HTTP chunked-transfer frame
            self.wfile.write(f"{len(b):X}\r\n".encode()); self.wfile.write(b)
            self.wfile.write(b"\r\n"); self.wfile.flush()

        # one SSE frame per token, flushed immediately (t_send stamped here)
        while True:
            item = streamer.q.get()
            if item is None:           # end sentinel
                break
            step, tid, t_gen = item
            piece = tokenizer.decode([tid], skip_special_tokens=True)
            t_send = time.perf_counter()
            frame = ("data: " + json.dumps(
                {"step": step, "token_id": tid, "text": piece,
                 "t_gen": t_gen, "t_send": t_send}) + "\n\n").encode()
            chunk(frame)
            rows.append((step, tid, piece, f"{t_gen:.9f}", f"{t_send:.9f}"))
        chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()   # end chunked stream
        th.join()

        # save this request's per-token timing
        with open(LOG_DIR / f"{tag}_server_tokens.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step","token_id","text","t_gen","t_send"])
            w.writerows(rows)


if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"SSE_SERVER_READY {HOST}:{PORT}", flush=True)
    srv.serve_forever()
