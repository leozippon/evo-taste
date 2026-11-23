"""Transport faults: a transient one is retried within a bound; HTTP errors and timeouts are never retried."""

import httpx
import pytest

from evotaste import sampler
from evotaste.sampler import Request, Sampler

OK = {"choices": [{"index": 0, "text": "<answer>1</answer>", "token_ids": [5, 6], "finish_reason": "stop",
                   "logprobs": {"token_logprobs": [-0.5, -0.25]}}]}


def serve(monkeypatch, outcomes):
    """A Sampler whose requests meet `outcomes` in turn (an exception class, a status code, or 200 = success)."""
    calls = []

    def handler(request):
        outcome = outcomes[min(len(calls), len(outcomes) - 1)]
        calls.append(outcome)
        if isinstance(outcome, type):
            raise outcome("injected", request=request)
        return httpx.Response(outcome, json=OK if outcome == 200 else {"error": "bad request"})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(sampler, "RETRY_DELAY_S", 0.0)
    s = Sampler.__new__(Sampler)  # skips the server handshake
    s.url, s.base_model = "http://test", "m"
    return s, calls


def generate(s):
    return s.generate([Request("m", [1, 2, 3], 1, 8)])


def test_transient_fault_is_retried(monkeypatch):
    s, calls = serve(monkeypatch, [httpx.ReadError, httpx.RemoteProtocolError, 200])
    [[sample]] = generate(s)
    assert sample.token_ids == [5, 6] and sample.logprob == -0.75 and len(calls) == 3


def test_retry_bound_fails_loudly(monkeypatch):
    s, calls = serve(monkeypatch, [httpx.ConnectError])  # a dead server
    with pytest.raises(RuntimeError, match="transport failed"):
        generate(s)
    assert len(calls) == sampler.RETRIES + 1


@pytest.mark.parametrize("outcome", [400, 500, httpx.ReadTimeout])
def test_no_retry_on_http_errors_or_timeouts(monkeypatch, outcome):
    s, calls = serve(monkeypatch, [outcome, 200])
    with pytest.raises((RuntimeError, httpx.ReadTimeout)):
        generate(s)
    assert len(calls) == 1


def test_publish_reports_its_parts_and_retires_the_old_version(tmp_path):
    s = Sampler.__new__(Sampler)
    s.root, s.current, s.counter, posts = tmp_path, {}, 0, []
    s._post = lambda route, body: posts.append((route, body["lora_name"]))
    first, parts = s.publish("m0", lambda p: p.mkdir())
    assert set(parts) == {"save_s", "load_s", "unload_s", "cleanup_s"}
    second, _ = s.publish("m0", lambda p: p.mkdir())
    assert first != second and not (tmp_path / first).exists() and (tmp_path / second).exists()
    assert posts == [("/v1/load_lora_adapter", first), ("/v1/load_lora_adapter", second),
                     ("/v1/unload_lora_adapter", first)]  # the new version is loaded before the old one goes
