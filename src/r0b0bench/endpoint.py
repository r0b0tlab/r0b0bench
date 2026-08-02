from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator
from urllib.parse import urljoin

import httpx


class Endpoint:
    def __init__(self, base_url: str, model: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/") + "/"
        self.model = model
        self.timeout = timeout
        # long NIAH/prefill may exceed 600s; callers can raise via with_timeout
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0))

    def close(self) -> None:
        self._client.close()

    def with_timeout(self, timeout: float) -> "Endpoint":
        """Return a shallow clone using a different request timeout."""
        other = Endpoint.__new__(Endpoint)
        other.base_url = self.base_url
        other.model = self.model
        other.timeout = timeout
        other._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0))
        return other

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    def health(self) -> dict[str, Any]:
        errors = []
        for path in ("../health", "models"):
            try:
                r = self._client.get(self._url(path))
                if r.status_code == 200:
                    return {"ok": True, "path": path, "status_code": r.status_code, "body": _safe_json(r)}
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}:{exc}")
        try:
            origin = self.base_url.replace("/v1/", "/").rstrip("/") + "/health"
            r = self._client.get(origin)
            if r.status_code == 200:
                return {"ok": True, "path": origin, "status_code": 200}
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        return {"ok": False, "errors": errors}

    def models(self) -> dict[str, Any]:
        r = self._client.get(self._url("models"))
        r.raise_for_status()
        return r.json()

    def max_model_len(self) -> int | None:
        try:
            data = self.models()
        except Exception:
            return None
        for m in data.get("data") or []:
            if m.get("id") == self.model and m.get("max_model_len"):
                return int(m["max_model_len"])
        for m in data.get("data") or []:
            if m.get("max_model_len"):
                return int(m["max_model_len"])
        return None

    def metrics_text(self) -> str | None:
        """Best-effort scrape of Prometheus metrics from sibling /metrics."""
        candidates = []
        # base is usually .../v1/
        root = self.base_url
        if root.endswith("/v1/"):
            candidates.append(root[:-4] + "metrics")
            candidates.append(root + "../metrics")
        candidates.append(urljoin(root, "../metrics"))
        candidates.append(urljoin(root, "metrics"))
        for url in candidates:
            try:
                r = self._client.get(url)
                if r.status_code == 200 and ("kv_cache" in r.text or "vllm:" in r.text or "# HELP" in r.text):
                    return r.text
            except Exception:
                continue
        return None

    def kv_cache_size_tokens(self) -> int | None:
        text = self.metrics_text()
        if not text:
            return None
        m = re.search(r'kv_cache_size_tokens="(\d+)"', text)
        return int(m.group(1)) if m else None

    def chat_completions(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
        body = {"model": self.model, **payload}
        t0 = time.perf_counter()
        r = self._client.post(self._url("chat/completions"), json=body)
        elapsed = time.perf_counter() - t0
        try:
            data = r.json()
        except Exception:
            data = {"error": r.text}
        return r.status_code, data, elapsed

    def completions(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
        body = {"model": self.model, **payload}
        t0 = time.perf_counter()
        r = self._client.post(self._url("completions"), json=body)
        elapsed = time.perf_counter() - t0
        try:
            data = r.json()
        except Exception:
            data = {"error": r.text}
        return r.status_code, data, elapsed

    def chat_render(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Server-side chat template render → token_ids (vLLM render endpoint)."""
        body = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        r = self._client.post(self._url("chat/completions/render"), json=body)
        r.raise_for_status()
        return r.json()

    def chat_completions_stream(
        self,
        payload: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        """Streaming chat completion; returns status + latency stats + assembled text."""
        body = {"model": self.model, "stream": True, "stream_options": {"include_usage": True}, **payload}
        t0 = time.perf_counter()
        ttft = None
        token_times: list[float] = []
        texts: list[str] = []
        finish = None
        usage: dict[str, Any] = {}
        status = 0
        with self._client.stream("POST", self._url("chat/completions"), json=body) as r:
            status = r.status_code
            if status != 200:
                try:
                    err = r.read().decode(errors="replace")
                except Exception:
                    err = ""
                return status, {
                    "ok": False,
                    "error": err[:1000],
                    "elapsed_s": time.perf_counter() - t0,
                }
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                else:
                    continue
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                now = time.perf_counter()
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                # also count reasoning if present for timing only
                if not content and delta.get("reasoning_content"):
                    content = ""  # don't measure reasoning as completion tokens by default
                if content:
                    if ttft is None:
                        ttft = now - t0
                    token_times.append(now)
                    texts.append(content)
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
        elapsed = time.perf_counter() - t0
        itls = []
        for i in range(1, len(token_times)):
            itls.append((token_times[i] - token_times[i - 1]) * 1000.0)
        return status, {
            "ok": True,
            "text": "".join(texts),
            "finish_reason": finish,
            "usage": usage,
            "ttft_ms": (ttft * 1000.0) if ttft is not None else None,
            "itl_ms_mean": (sum(itls) / len(itls)) if itls else None,
            "itl_ms_p50": _percentile(itls, 50) if itls else None,
            "itl_ms_p95": _percentile(itls, 95) if itls else None,
            "stream_completion_chunks": len(token_times),
            "elapsed_s": elapsed,
            "e2el_ms": elapsed * 1000.0,
        }


def _percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _safe_json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return r.text[:500]
