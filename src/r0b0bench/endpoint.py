from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin

import httpx


class Endpoint:
    def __init__(self, base_url: str, model: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/") + "/"
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    def health(self) -> dict[str, Any]:
        # try /health on parent origin and /v1/models
        errors = []
        for path in ("../health", "models"):
            try:
                r = self._client.get(self._url(path))
                if r.status_code == 200:
                    return {"ok": True, "path": path, "status_code": r.status_code, "body": _safe_json(r)}
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}:{exc}")
        # base_url might already be origin without /v1
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

    def chat_completions(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
        import time

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
        import time

        body = {"model": self.model, **payload}
        t0 = time.perf_counter()
        r = self._client.post(self._url("completions"), json=body)
        elapsed = time.perf_counter() - t0
        try:
            data = r.json()
        except Exception:
            data = {"error": r.text}
        return r.status_code, data, elapsed


def _safe_json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return r.text[:500]
