"""
Task 3 — CLIENT side (the "attacker"/user).

Receives the SSE stream over a RAW socket and stamps t_arrival the instant each
token's bytes land. Raw socket = no HTTP library in the way, so no hidden
buffering distorts the timing.
  - Same host as server -> t_arrival is directly comparable to t_gen/t_send.
  - Across hosts         -> only compare gaps between tokens, not absolute times.

It also records which recv() delivered each token (recv_idx). When several
tokens share one recv_idx, they arrived bundled -- that is the transport
blurring the experiment is looking for.

Usage:
  python sse_client.py --host 127.0.0.1 --port 8077 --tag req \
        --prompt "..." --max-new-tokens 64 --cache cold
Output: <tag>_client_arrivals.csv  (step, t_arrival, recv_idx, bytes_in_recv)
"""
import argparse, csv, json, socket, time, urllib.parse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8077)
ap.add_argument("--tag", default="req")
ap.add_argument("--prompt", default="Explain what a GPU is.")
ap.add_argument("--max-new-tokens", type=int, default=64)
ap.add_argument("--cache", default="warm")
ap.add_argument("--logdir", default="logs")
a = ap.parse_args()

# build and send the GET request by hand (we drive the raw socket ourselves)
q = urllib.parse.urlencode({"prompt": a.prompt, "max_new_tokens": a.max_new_tokens,
                            "tag": a.tag, "cache": a.cache})
req = (f"GET /generate?{q} HTTP/1.1\r\nHost: {a.host}:{a.port}\r\n"
       f"Accept: text/event-stream\r\nConnection: close\r\n\r\n")

s = socket.create_connection((a.host, a.port))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # ask the OS not to delay sends
s.sendall(req.encode())

buf = b""
rows = []
recv_idx = 0
# read until the end of the HTTP headers (the blank line), then drop them
while b"\r\n\r\n" not in buf:
    buf += s.recv(4096)
buf = buf.split(b"\r\n\r\n", 1)[1]


def dechunk(data):
    """Pull complete HTTP chunked-transfer pieces out of a growing buffer.
    Returns (decoded_bytes, leftover); a partial final chunk stays in leftover."""
    out = b""; i = 0
    while True:
        j = data.find(b"\r\n", i)
        if j < 0: return out, data[i:]
        try: size = int(data[i:j], 16)
        except ValueError: return out, data[i:]
        if size == 0: return out, b""
        start = j + 2; end = start + size
        if end + 2 > len(data): return out, data[i:]
        out += data[start:end]; i = end + 2


# main read loop: timestamp every recv(); emit one row per token frame
sse_buf = ""
raw = b""
while True:
    t = time.perf_counter()            # t_arrival: bytes just landed
    try: d = s.recv(8192)
    except OSError: break
    if not d: break
    recv_idx += 1
    nbytes = len(d)
    raw += d
    decoded, raw = dechunk(raw)
    if decoded:
        sse_buf += decoded.decode(errors="replace")
        while "\n\n" in sse_buf:       # one SSE frame = "data: {...}\n\n"
            frame, sse_buf = sse_buf.split("\n\n", 1)
            frame = frame.strip()
            if not frame.startswith("data: "): continue
            payload = frame[len("data: "):]
            if payload == "[DONE]":
                continue
            try: obj = json.loads(payload)
            except json.JSONDecodeError: continue
            rows.append((obj["step"], f"{t:.9f}", recv_idx, nbytes))

s.close()
Path(a.logdir).mkdir(parents=True, exist_ok=True)
out = Path(a.logdir) / f"{a.tag}_client_arrivals.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["step","t_arrival","recv_idx","bytes_in_recv"])
    w.writerows(rows)
print(f"CLIENT_DONE tokens={len(rows)} recvs={recv_idx} -> {out}")
