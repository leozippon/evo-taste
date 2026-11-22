"""vLLM server lifecycle, concurrent generation, and adapter versions (docs/dev/INFRA_NOTES.md).

Every adapter version gets a new name and a new directory; the previous version is unloaded and its directory
removed. Requests address adapters by exact version name, so a request can never run on stale weights.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)
HANDOFF_ROOT = Path("/dev/shm/evotaste")
READY_TIMEOUT_S = 1200
MAX_CONNECTIONS = 512
# A request whose connection fails at the transport level (reset, refused, dropped without a response) is resent
# up to RETRIES times, RETRY_DELAY_S x attempt apart: one bad connection among millions of requests must not end a
# job, while a dead server exhausts the bound within seconds. Timeouts and HTTP error statuses are never retried.
RETRIES = 3
RETRY_DELAY_S = 2.0
_TRANSIENT = (httpx.NetworkError, httpx.RemoteProtocolError)


def _die_with_parent():  # runs in the child before exec: SIGTERM it when the worker dies, even by SIGKILL
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


class Server:
    """A vLLM OpenAI-compatible server on the visible GPU, owned by this process."""

    def __init__(self, model: str, server_cfg, max_lora_rank: int, log_path: Path):
        self.model, self.cfg, self.rank, self.log_path = model, server_cfg, max_lora_rank, log_path
        self.proc: subprocess.Popen | None = None
        self.url = ""

    def start(self) -> None:
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("pin exactly one GPU with CUDA_VISIBLE_DEVICES before starting a server")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        cmd = [str(Path(sys.executable).with_name("vllm")), "serve", self.model, "--host", "127.0.0.1",
               "--port", str(port), "--kv-cache-memory-bytes", self.cfg.kv_cache,
               "--gpu-memory-utilization", str(self.cfg.gpu_memory_utilization), "--max-model-len", "8192",
               "--generation-config", "vllm", "--enable-prompt-tokens-details",
               "--enable-lora", "--max-lora-rank", str(self.rank), "--max-loras", str(self.cfg.max_loras),
               "--max-cpu-loras", str(max(64, 2 * self.cfg.max_loras)),
               "--max-num-seqs", str(self.cfg.max_num_seqs),
               "--max-num-batched-tokens", str(self.cfg.max_num_batched_tokens)]
        env = {**os.environ, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as out:
            self.proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                         start_new_session=True, preexec_fn=_die_with_parent)
        log.info("vLLM pid %d starting on %s (log %s)", self.proc.pid, self.url, self.log_path)

    def wait_ready(self) -> None:
        t0 = time.time()
        while time.time() - t0 < READY_TIMEOUT_S:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {self.proc.returncode}; see {self.log_path}")
            try:
                if httpx.get(f"{self.url}/health", timeout=5).status_code == 200:
                    log.info("vLLM ready after %.0f s", time.time() - t0)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM not ready after {READY_TIMEOUT_S} s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        log.info("vLLM pid %d stopped", self.proc.pid)


@dataclass
class Request:
    model: str  # adapter version name, or the base model id
    prompt: list[int]
    n: int
    max_tokens: int


@dataclass
class Sample:
    text: str
    token_ids: list[int]
    finish_reason: str
    token_logprobs: list[float]  # sampler log-probability of each sampled token (full softmax, temperature 1)

    @property
    def logprob(self) -> float:
        return float(sum(self.token_logprobs))


class Sampler:
    """Client of one server: generation at temperature 1 and adapter versions under a private /dev/shm directory."""

    def __init__(self, url: str, base_model: str, run_name: str):
        self.url, self.base_model = url, base_model
        served = httpx.get(f"{url}/v1/models", timeout=30).json()["data"]
        if not any(m["id"] == base_model and not m.get("parent") for m in served):
            raise RuntimeError(f"server at {url} does not serve {base_model}")
        self.root = HANDOFF_ROOT / run_name / str(os.getpid())
        self.root.mkdir(parents=True, exist_ok=True)
        for other in self.root.parent.iterdir():  # versions left behind by killed processes of this run
            if other != self.root and not Path(f"/proc/{other.name}").exists():
                shutil.rmtree(other, ignore_errors=True)
        self.current: dict[str, tuple[str, Path]] = {}  # member -> (version name, directory)
        self.counter = 0

    # -- adapters ---------------------------------------------------------------------------------------------
    def publish(self, member: str, save) -> str:
        """Save a new version with save(path), load it under a new name, then retire the previous version."""
        self.counter += 1
        name = f"{member}@{os.getpid()}-{self.counter}"
        path = self.root / name
        save(path)
        self._post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
        self.retire(member)
        self.current[member] = (name, path)
        return name

    def load_existing(self, label: str, path: Path) -> str:
        """Serve an immutable adapter directory (evaluation of a checkpoint) under a new name."""
        self.counter += 1
        name = f"{label}@{os.getpid()}-{self.counter}"
        self._post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(Path(path).resolve())})
        return name

    def retire(self, member: str) -> None:
        if member in self.current:
            name, path = self.current.pop(member)
            self._post("/v1/unload_lora_adapter", {"lora_name": name})
            shutil.rmtree(path)

    def close(self) -> None:
        for member in list(self.current):
            try:
                self.retire(member)
            except httpx.HTTPError:  # the server may already be gone; the directories are removed below
                pass
        shutil.rmtree(self.root, ignore_errors=True)

    def _post(self, route: str, body: dict) -> None:
        r = httpx.post(f"{self.url}{route}", json=body, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(f"{route} {body} failed: {r.status_code} {r.text[:300]}")

    # -- generation ---------------------------------------------------------------------------------------------
    def generate(self, requests: list[Request]) -> list[list[Sample]]:
        """All requests concurrently; per request its n samples in order. Any failed request raises."""
        return asyncio.run(self._generate(requests))

    async def _generate(self, requests):
        # Enough concurrent requests to fill the server's 512 sequence slots, but within the default soft limit of
        # 1024 open files (more simultaneous connections fail with "All connection attempts failed"). The semaphore
        # keeps requests out of httpx's own pool queue, whose bookkeeping is quadratic in the number of waiters.
        limits = httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=MAX_CONNECTIONS)
        slots = asyncio.Semaphore(MAX_CONNECTIONS)
        async with httpx.AsyncClient(base_url=self.url, timeout=7200, limits=limits) as client:
            return await asyncio.gather(*(self._one(client, slots, r) for r in requests))

    async def _one(self, client, slots, req: Request) -> list[Sample]:
        body = {"model": req.model, "prompt": req.prompt, "n": req.n, "max_tokens": req.max_tokens,
                "temperature": 1.0, "top_p": 1.0, "logprobs": 0, "return_token_ids": True}
        async with slots:
            for attempt in range(RETRIES + 1):
                try:
                    r = await client.post("/v1/completions", json=body)
                    break
                except _TRANSIENT as e:
                    if attempt == RETRIES:
                        raise RuntimeError(f"completion via {req.model}: transport failed {RETRIES + 1} times "
                                           f"({type(e).__name__}: {e}); is the server alive?") from e
                    log.warning("completion via %s: %s (%s); retry %d/%d", req.model, type(e).__name__, e,
                                attempt + 1, RETRIES)
                    await asyncio.sleep(RETRY_DELAY_S * (attempt + 1))
        if r.status_code != 200:
            raise RuntimeError(f"completion via {req.model} failed: {r.status_code} {r.text[:300]}")
        out = []
        for ch in sorted(r.json()["choices"], key=lambda c: c["index"]):
            ids, lps = ch["token_ids"], ch["logprobs"]["token_logprobs"]
            if len(ids) != len(lps) or not ids:
                raise RuntimeError(f"{req.model}: {len(ids)} token ids but {len(lps)} logprobs")
            out.append(Sample(ch["text"], ids, ch["finish_reason"], lps))
        if len(out) != req.n:
            raise RuntimeError(f"{req.model}: asked for {req.n} samples, got {len(out)}")
        return out
