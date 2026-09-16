#!/usr/bin/env python3
"""Hardened Q200-v2 text-180 runner; native thinking and no inline model code.

The runner freezes dataset identity before any request, persists one complete
row per ID, and refuses to call a partial/duplicate/missing-usage run complete.
HumanEval grading runs generated code in a separate process group with a hard
wall-clock timeout.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import time
import unicodedata
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

MODEL = "qwen38-flash-next-w4a16"
DATASET_SHA256 = "66a75701cbeea69f212e1c8be92aab9efaf3fa4d7af3c6911c8f7864a17d8d14"
EXPECTED_COUNT = 180
EXPECTED_FAMILY_COUNTS = {"gsm8k": 80, "humaneval": 40, "ifeval": 40, "hard_reasoning": 20}
DEFAULT_CHAT_KWARGS = {"enable_thinking": True, "thinking": True, "reasoning_effort": "low"}
Q200_FINISH_REASONS = frozenset({"stop"})
MANUAL_EVIDENCE_SCHEMA = "r0b0tlab.qwen38.manual_evidence.v1"
MANUAL_REVIEW_METHOD = "independent_manual_review"

# The executable-code lane is a claim-bearing boundary.  These values are
# duplicated in the repository-owned in-image driver and are hash-bound in
# every request/receipt; changing one requires rebuilding and requalifying the
# production image rather than silently falling back to host execution.
Q200_SANDBOX_DRIVER_VERSION = "r0b0tlab.qwen38.q200_sandbox_driver.v1"
Q200_SANDBOX_REQUEST_SCHEMA = "r0b0tlab.qwen38.q200_sandbox_request.v1"
Q200_SANDBOX_RECEIPT_SCHEMA = "r0b0tlab.qwen38.q200_sandbox_receipt.v1"
Q200_SANDBOX_DRIVER_PATH = "/opt/r0b0tlab/q200_sandbox_driver.py"
Q200_SANDBOX_IMAGE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
Q200_SANDBOX_MAX_REQUEST_BYTES = 1024 * 1024
Q200_SANDBOX_MAX_OUTPUT_BYTES = 64 * 1024
Q200_SANDBOX_MAX_OUTPUT_TAIL_BYTES = 4096
Q200_SANDBOX_RUNTIME_MANIFEST: dict[str, Any] = {
    "schema": "r0b0tlab.qwen38.q200_sandbox_runtime.v1",
    "cpu_seconds": 9,
    "address_space_bytes": 256 * 1024 * 1024,
    "file_size_bytes": 1024 * 1024,
    "nproc": 32,
    "nofile": 64,
    "output_bytes": Q200_SANDBOX_MAX_OUTPUT_BYTES,
    "tmpfs_bytes": 64 * 1024 * 1024,
    "network": "none",
    "read_only_root": True,
    "uid_gid": "65532:65532",
}
Q200_SANDBOX_RECEIPT_KEYS = {
    "schema",
    "driver_version",
    "driver_source_sha256",
    "image_id",
    "request_sha256",
    "candidate_sha256",
    "reference_sha256",
    "runtime_manifest_sha256",
    "runtime_manifest",
    "status",
    "passed",
    "failure_class",
    "returncode",
    "stdout_bytes",
    "stderr_bytes",
    "stdout_tail",
    "stderr_tail",
    "reason",
    "stdin_isolated",
}


def _sandbox_driver_source_sha256() -> str:
    path = Path(__file__).resolve().parents[1] / "docker" / "q200_sandbox_driver.py"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sandbox_canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sandbox_sha256_json(value: Any) -> str:
    return hashlib.sha256(_sandbox_canonical(value)).hexdigest()

try:
    from score_flex_gsm8k import final_answer, normalize_number
except ImportError:  # pragma: no cover - direct package imports use scripts/
    from scripts.score_flex_gsm8k import final_answer, normalize_number

try:
    from niah_common import strict_json_loads
except ImportError:  # pragma: no cover - direct package imports use scripts/
    from scripts.niah_common import strict_json_loads

try:
    from admission_control import AdmissionCoordinator, AdmissionError, AdmissionNotGranted
except ImportError:  # pragma: no cover - direct package imports use scripts/
    from scripts.admission_control import AdmissionCoordinator, AdmissionError, AdmissionNotGranted


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_quality_set(path: str | Path, *, require_frozen_hash: bool = True) -> tuple[list[dict[str, Any]], str]:
    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if require_frozen_hash and digest != DATASET_SHA256:
        raise ValueError(f"quality set SHA-256 mismatch: {digest} != {DATASET_SHA256}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank quality-set line {line_number}")
            record = strict_json_loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise ValueError(f"invalid quality-set row {line_number}")
            if record["id"] in seen:
                raise ValueError(f"duplicate quality-set id: {record['id']}")
            seen.add(record["id"])
            for key in ("family", "grade", "prompt", "reference"):
                if key not in record:
                    raise ValueError(f"quality-set row {record['id']} missing {key}")
            rows.append(record)
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(f"quality-set count {len(rows)} != {EXPECTED_COUNT}")
    from collections import Counter
    counts = dict(Counter(row["family"] for row in rows))
    if counts != EXPECTED_FAMILY_COUNTS:
        raise ValueError(f"quality-set family counts {counts} != {EXPECTED_FAMILY_COUNTS}")
    return rows, digest


def dataset_manifest(path: str | Path) -> dict[str, Any]:
    rows, digest = read_quality_set(path)
    from collections import Counter
    return {
        "schema": "r0b0tlab.qwen38.quality_text_180_manifest.v2",
        "dataset": "quality-text-180-v2.jsonl",
        "sha256": digest,
        "bytes": Path(path).stat().st_size,
        "rows": len(rows),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "ids": [row["id"] for row in rows],
    }


def validate_native_thinking(chat_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(chat_kwargs)
    if values.get("enable_thinking") is not True or values.get("thinking") is not True or values.get("reasoning_effort") != "low":
        raise ValueError("Q200 requires native thinking with reasoning_effort=low")
    return values


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _request_max_tokens(source: Mapping[str, Any], max_tokens: int) -> int:
    return max(max_tokens, 1536 if source["family"] in {"humaneval", "agentic_coding"} else 1024)


def _run_identity(*, model: str, image_id: str, profile_id: str, candidate_id: str, dataset_sha256: str, chat_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "image_id": image_id,
        "profile_id": profile_id,
        "candidate_id": candidate_id,
        "dataset_sha256": dataset_sha256,
        "chat_template_kwargs": dict(chat_kwargs),
    }


def _row_identity(*, identity: Mapping[str, Any], source: Mapping[str, Any], max_tokens: int) -> dict[str, Any]:
    request_max = _request_max_tokens(source, max_tokens)
    request = _request_payload(identity["model"], str(source["prompt"]), request_max, identity["chat_template_kwargs"])
    return {
        **identity,
        "prompt_sha256": _prompt_sha256(str(source["prompt"])),
        "request_max_tokens": request_max,
        "request_sha256": _sha256_json(request),
    }


def _request_payload(model: str, prompt: str, max_tokens: int, chat_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": dict(chat_kwargs),
    }


def chat(base: str, prompt: str, max_tokens: int, timeout: int, *, model: str = MODEL, chat_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    kwargs = dict(DEFAULT_CHAT_KWARGS if chat_kwargs is None else chat_kwargs)
    payload = _request_payload(model, prompt, max_tokens, kwargs)
    request_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    request = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=request_bytes, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    status = 0
    body: dict[str, Any] = {}
    error = None
    response_status = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_status = int(response.status)
            status = response_status
            parsed = strict_json_loads(response.read())
            if isinstance(parsed, dict):
                body = parsed
            else:
                error = "response JSON is not an object"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    raw_choices = body.get("choices")
    choices: list[Any] = raw_choices if isinstance(raw_choices, list) else []
    choice = choices[0] if len(choices) == 1 and isinstance(choices[0], dict) else {}
    raw_message = choice.get("message")
    message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
    raw_usage = body.get("usage")
    usage: dict[str, Any] | None = raw_usage if isinstance(raw_usage, dict) else None
    raw_content = message.get("content")
    content = raw_content if isinstance(raw_content, str) else ""
    raw_reasoning = message.get("reasoning_content")
    reasoning_content = raw_reasoning if isinstance(raw_reasoning, str) else ""
    finish_reason = choice.get("finish_reason")
    if error is None:
        response_failures: list[str] = []
        if response_status != 200:
            response_failures.append("HTTP status is not 200")
        if len(choices) != 1 or not isinstance(choices[0], dict):
            response_failures.append("response must contain exactly one object choice")
        if not isinstance(raw_message, dict):
            response_failures.append("choice.message must be an object")
        if not isinstance(raw_content, str) or not raw_content.strip():
            response_failures.append("choice.message.content must be non-empty text")
        if raw_reasoning is not None and not isinstance(raw_reasoning, str):
            response_failures.append("choice.message.reasoning_content must be text or null")
        if finish_reason not in Q200_FINISH_REASONS:
            response_failures.append("finish_reason is missing or unsupported")
        if usage is None:
            response_failures.append("usage must be an object")
        else:
            for key in ("prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    response_failures.append(f"usage.{key} must be a positive integer")
        if response_failures:
            error = "invalid completion response: " + "; ".join(response_failures)
    return {
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "request": payload,
        "status": status,
        "response_status": response_status,
        "error": error,
        "content": content,
        "reasoning_content": reasoning_content,
        "text": content,  # compatibility alias; never truncated
        "finish_reason": finish_reason,
        "finish": finish_reason,
        "usage": usage,
        "model_id": body.get("model") or model,
        "elapsed_seconds": elapsed,
        "elapsed": elapsed,
        "response_sha256": hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if body else None,
    }


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    code = match.group(1) if match else text
    return re.sub(r"^python\s*$", "", code, flags=re.IGNORECASE | re.MULTILINE).strip() + "\n"


def _human_eval_prelude(prompt: str, entry_point: str) -> str:
    """Return trusted helper/import context preceding a HumanEval target.

    Frozen prompts can define helpers before the function the model is asked to
    complete. The response contract asks for only the completed target
    definition, so the grader must restore that prompt context rather than
    falsely rejecting calls to the supplied helpers.
    """
    if not isinstance(prompt, str) or not isinstance(entry_point, str) or not entry_point:
        raise ValueError("HumanEval prompt and entry point must be non-empty strings")
    match = re.search(r"```(?:python|py)?\s*(.*?)```", prompt, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ValueError("HumanEval prompt lacks one fenced Python context")
    source = match.group(1)
    tree = ast.parse(source, mode="exec")
    targets = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
    ]
    if len(targets) != 1:
        raise ValueError("HumanEval prompt must define the target exactly once")
    target = targets[0]
    start_line = min([target.lineno, *(item.lineno for item in target.decorator_list)])
    prelude = "".join(source.splitlines(keepends=True)[: start_line - 1]).strip()
    if prelude:
        ast.parse(prelude, mode="exec")
        return prelude + "\n\n"
    return ""


def _sandbox_failure(
    reason: str,
    *,
    container_name: str | None = None,
    cleanup: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "grader_error",
        "passed": None,
        "failure_class": "HARNESS_BLOCK",
        "reason": reason[:256],
    }
    if container_name is not None:
        result["container_name"] = container_name
    if cleanup is not None:
        result["cleanup"] = dict(cleanup)
    return result


class _DockerSandbox:
    """Small Docker seam; production code has no alternate executor."""

    _env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/nonexistent",
        "DOCKER_CONFIG": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }

    def _small(self, argv: list[str], *, timeout: float = 15.0, limit: int = 64 * 1024) -> tuple[int, bytes, bytes]:
        """Run a bounded Docker control command without buffered reads."""
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            start_new_session=True,
        )
        if process.stdout is None or process.stderr is None:
            _terminate_host_process(process)
            raise RuntimeError("Docker control command did not provide output pipes")
        selector = selectors.DefaultSelector()
        streams = ((process.stdout, "stdout"), (process.stderr, "stderr"))
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        try:
            for stream, label in streams:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, label)
            deadline = time.monotonic() + timeout
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _terminate_host_process(process)
                    raise RuntimeError("Docker control command timed out")
                for key, _mask in selector.select(min(0.05, remaining)):
                    stream = key.fileobj
                    fd = stream if isinstance(stream, int) else stream.fileno()
                    try:
                        chunk = os.read(fd, 4096)
                    except BlockingIOError:
                        continue
                    except OSError:
                        chunk = b""
                    if not chunk:
                        try:
                            selector.unregister(stream)
                        except Exception:
                            pass
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                        continue
                    buffer = buffers[key.data]
                    buffer.extend(chunk)
                    if len(buffer) > limit:
                        _terminate_host_process(process)
                        raise RuntimeError("Docker control command exceeded its bound")
            if process.poll() is None:
                try:
                    process.wait(timeout=0.20)
                except subprocess.TimeoutExpired as exc:
                    _terminate_host_process(process)
                    raise RuntimeError("Docker control command did not exit") from exc
            return process.returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])
        finally:
            try:
                selector.close()
            except Exception:
                pass
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    def image_identity(self, image_id: str) -> str:
        returncode, stdout, stderr = self._small(["docker", "image", "inspect", "--format={{.Id}}", image_id])
        if returncode != 0:
            raise RuntimeError(f"Docker image is unavailable: {stderr.decode(errors='replace')[:256]}")
        values = stdout.decode("utf-8", errors="replace").splitlines()
        if len(values) != 1 or not Q200_SANDBOX_IMAGE_RE.fullmatch(values[0]):
            raise RuntimeError("Docker image inspect did not return one immutable image ID")
        return values[0]

    def run(self, argv: list[str]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            start_new_session=True,
        )

    def force_remove(self, name: str) -> None:
        returncode, _stdout, stderr = self._small(["docker", "container", "rm", "--force", name], timeout=15.0)
        if returncode != 0 and "no such container" not in stderr.decode("utf-8", errors="replace").lower():
            raise RuntimeError(f"Docker cleanup failed: {stderr.decode(errors='replace')[:256]}")

    def container_present(self, name: str) -> bool:
        returncode, _stdout, stderr = self._small(["docker", "container", "inspect", name], timeout=15.0)
        if returncode == 0:
            return True
        if "no such object" in stderr.decode("utf-8", errors="replace").lower() or "no such container" in stderr.decode("utf-8", errors="replace").lower():
            return False
        raise RuntimeError(f"Docker cleanup inspection failed: {stderr.decode(errors='replace')[:256]}")


_DOCKER = _DockerSandbox()


def _terminate_host_process(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=0.20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            process.wait(timeout=0.20)
        except subprocess.TimeoutExpired:
            pass


def _docker_command(name: str, image_id: str) -> list[str]:
    return [
        "docker",
        "run",
        "--interactive",
        "--name",
        name,
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--user=65532:65532",
        "--pids-limit=64",
        "--cpus=1.0",
        "--memory=256m",
        "--memory-swap=256m",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--tmpfs=/workspace:rw,noexec,nosuid,nodev,size=64m,mode=1777",
        "--workdir=/workspace",
        "--env=PATH=/usr/local/bin:/usr/bin:/bin",
        "--env=PYTHONPATH=",
        "--env=PYTHONNOUSERSITE=1",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=HOME=/nonexistent",
        "--env=LANG=C.UTF-8",
        "--env=LC_ALL=C.UTF-8",
        image_id,
        "python3",
        Q200_SANDBOX_DRIVER_PATH,
    ]


def _drain_docker(
    process: subprocess.Popen[bytes],
    payload: bytes,
    *,
    timeout: float,
) -> dict[str, Any]:
    """Drain Docker's two pipes incrementally with independent live caps."""
    if process.stdin is None or process.stdout is None or process.stderr is None:
        raise RuntimeError("Docker run did not provide standard pipes")
    selector = selectors.DefaultSelector()
    stdout_tail = bytearray()
    stderr_tail = bytearray()
    stdout_bytes = stderr_bytes = 0
    input_offset = 0
    status = "ok"
    reason: str | None = None
    try:
        for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, label)
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        deadline = time.monotonic() + timeout
        while selector.get_map():
            now = time.monotonic()
            if now >= deadline:
                status, reason = "timeout", "Docker foreground run exceeded its bound"
                _terminate_host_process(process)
                break
            if process.poll() is not None:
                try:
                    selector.unregister(process.stdin)
                except Exception:
                    pass
                try:
                    process.stdin.close()
                except OSError:
                    pass
            events = selector.select(max(0.01, min(0.05, deadline - now)))
            for key, _mask in events:
                label = key.data
                fileobj = key.fileobj
                fd = fileobj.fileno()
                if label == "stdin":
                    if input_offset >= len(payload) or process.poll() is not None:
                        try:
                            selector.unregister(fileobj)
                        except Exception:
                            pass
                        fileobj.close()
                        continue
                    try:
                        written = os.write(fd, payload[input_offset : input_offset + 16384])
                    except BrokenPipeError:
                        status, reason = "stdin_error", "Docker driver closed stdin"
                        try:
                            selector.unregister(fileobj)
                        except Exception:
                            pass
                        fileobj.close()
                        continue
                    input_offset += written
                    if input_offset >= len(payload):
                        try:
                            selector.unregister(fileobj)
                        except Exception:
                            pass
                        fileobj.close()
                    continue
                try:
                    chunk = os.read(fd, 4096)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""
                if not chunk:
                    try:
                        selector.unregister(fileobj)
                    except Exception:
                        pass
                    fileobj.close()
                    continue
                if label == "stdout":
                    stdout_bytes += len(chunk)
                    stdout_tail.extend(chunk)
                    if len(stdout_tail) > Q200_SANDBOX_MAX_OUTPUT_TAIL_BYTES:
                        del stdout_tail[: len(stdout_tail) - Q200_SANDBOX_MAX_OUTPUT_TAIL_BYTES]
                    if stdout_bytes > Q200_SANDBOX_MAX_OUTPUT_BYTES:
                        status, reason = "output_limit", "Docker stdout exceeded live bound"
                        _terminate_host_process(process)
                        break
                else:
                    stderr_bytes += len(chunk)
                    stderr_tail.extend(chunk)
                    if len(stderr_tail) > Q200_SANDBOX_MAX_OUTPUT_TAIL_BYTES:
                        del stderr_tail[: len(stderr_tail) - Q200_SANDBOX_MAX_OUTPUT_TAIL_BYTES]
                    if stderr_bytes > Q200_SANDBOX_MAX_OUTPUT_BYTES:
                        status, reason = "output_limit", "Docker stderr exceeded live bound"
                        _terminate_host_process(process)
                        break
            if status != "ok":
                break
        if process.poll() is None:
            _terminate_host_process(process)
        return {
            "status": status,
            "reason": reason,
            "returncode": process.returncode,
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
            "stdout": bytes(stdout_tail),
            "stderr": bytes(stderr_tail),
        }
    finally:
        try:
            selector.close()
        except Exception:
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


def _cleanup_sandbox(name: str) -> dict[str, Any]:
    """Force-remove repeatedly, then require two stable absent inspections."""
    deadline = time.monotonic() + 2.0
    absent_checks = 0
    attempts = 0
    last_error: str | None = None
    while time.monotonic() < deadline:
        attempts += 1
        try:
            _DOCKER.force_remove(name)
            present = _DOCKER.container_present(name)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.05)
            continue
        if not present:
            absent_checks += 1
            if absent_checks >= 2:
                return {"verified_absent": True, "attempts": attempts}
        else:
            absent_checks = 0
        time.sleep(0.05)
    return {"verified_absent": False, "attempts": attempts, "error": last_error or "container remained present"}


def _validate_sandbox_receipt(
    receipt: Any,
    *,
    image_id: str,
    request_sha256: str,
    candidate_sha256: str,
    reference_sha256: str,
    driver_source_sha256: str,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping) or set(receipt) != Q200_SANDBOX_RECEIPT_KEYS:
        raise ValueError("sandbox receipt schema is incomplete or has unknown fields")
    if receipt.get("schema") != Q200_SANDBOX_RECEIPT_SCHEMA or receipt.get("driver_version") != Q200_SANDBOX_DRIVER_VERSION:
        raise ValueError("sandbox receipt schema/version mismatch")
    if receipt.get("driver_source_sha256") != driver_source_sha256 or receipt.get("image_id") != image_id:
        raise ValueError("sandbox receipt image/driver binding mismatch")
    if receipt.get("request_sha256") != request_sha256 or receipt.get("candidate_sha256") != candidate_sha256 or receipt.get("reference_sha256") != reference_sha256:
        raise ValueError("sandbox receipt request/candidate/reference binding mismatch")
    if receipt.get("runtime_manifest") != Q200_SANDBOX_RUNTIME_MANIFEST or receipt.get("runtime_manifest_sha256") != _sandbox_sha256_json(Q200_SANDBOX_RUNTIME_MANIFEST):
        raise ValueError("sandbox receipt runtime manifest mismatch")
    status = receipt.get("status")
    passed = receipt.get("passed")
    failure_class = receipt.get("failure_class")
    model_statuses = {"failed", "timeout", "output_limit", "oom"}
    if status == "ok":
        if passed is not True or failure_class is not None:
            raise ValueError("successful sandbox receipt has invalid verdict")
    elif status in model_statuses:
        if passed is not False or failure_class != "MODEL_REJECT":
            raise ValueError("candidate-failure receipt has invalid verdict")
    elif status == "grader_error":
        if passed is not None or failure_class != "HARNESS_BLOCK":
            raise ValueError("harness receipt has invalid verdict")
    else:
        raise ValueError("sandbox receipt has an unknown status")
    if not (isinstance(receipt.get("returncode"), int) and not isinstance(receipt.get("returncode"), bool)) and receipt.get("returncode") is not None:
        raise ValueError("sandbox returncode is invalid")
    for key in ("stdout_bytes", "stderr_bytes"):
        value = receipt.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > Q200_SANDBOX_MAX_OUTPUT_BYTES + 4096:
            raise ValueError(f"sandbox {key} is outside its bound")
    for key in ("stdout_tail", "stderr_tail", "reason"):
        value = receipt.get(key)
        if not isinstance(value, str) or len(value.encode("utf-8")) > (Q200_SANDBOX_MAX_OUTPUT_TAIL_BYTES if key != "reason" else 256):
            raise ValueError(f"sandbox {key} is invalid")
    if receipt.get("stdin_isolated") is not True:
        raise ValueError("sandbox stdin was not isolated")
    return dict(receipt)


def grade_exec_result(
    text: str,
    ref: Mapping[str, Any] | None,
    timeout: float = 8.0,
    *,
    image_id: str | None = None,
    prelude: str = "",
) -> dict[str, Any]:
    """Execute generated Python only through the repository-owned Docker driver."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not isinstance(image_id, str) or not Q200_SANDBOX_IMAGE_RE.fullmatch(image_id):
        return _sandbox_failure("an immutable sandbox image ID is required")
    if not isinstance(text, str) or not isinstance(ref, Mapping) or not isinstance(prelude, str):
        return _sandbox_failure("candidate/reference/prelude has an invalid type")
    try:
        candidate_code = _extract_code(text)
        if prelude:
            ast.parse(prelude, mode="exec")
        code = prelude + candidate_code
        reference = dict(ref)
        driver_source_sha256 = _sandbox_driver_source_sha256()
        candidate_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
        reference_sha256 = _sandbox_sha256_json(reference)
        request_without_hash: dict[str, Any] = {
            "schema": Q200_SANDBOX_REQUEST_SCHEMA,
            "driver_version": Q200_SANDBOX_DRIVER_VERSION,
            "driver_source_sha256": driver_source_sha256,
            "image_id": image_id,
            "candidate": code,
            "reference": reference,
            "timeout_seconds": float(timeout),
            "runtime_manifest": Q200_SANDBOX_RUNTIME_MANIFEST,
        }
        request_sha256 = _sandbox_sha256_json(request_without_hash)
        request = {**request_without_hash, "request_sha256": request_sha256}
        payload = _sandbox_canonical(request) + b"\n"
        if len(payload) > Q200_SANDBOX_MAX_REQUEST_BYTES:
            return _sandbox_failure("sandbox request exceeds its byte bound")
    except (OSError, TypeError, ValueError):
        return _sandbox_failure("candidate/reference cannot be represented as a strict sandbox request")

    name = f"q200-sandbox-{uuid.uuid4().hex}"
    primary: dict[str, Any] | None = None
    cleanup: dict[str, Any]
    try:
        try:
            observed_image_id = _DOCKER.image_identity(image_id)
            if observed_image_id != image_id:
                primary = _sandbox_failure("local Docker image ID does not match the supplied immutable ID", container_name=name)
            elif not code.strip():
                primary = {"status": "empty_code", "passed": False, "failure_class": "MODEL_REJECT", "reason": "empty candidate"}
            else:
                process = _DOCKER.run(_docker_command(name, image_id))
                transport = _drain_docker(process, payload, timeout=max(1.0, timeout + 2.0))
                if transport["status"] != "ok" or transport["returncode"] != 0:
                    primary = _sandbox_failure(transport.get("reason") or "Docker sandbox did not produce a clean exit", container_name=name)
                else:
                    lines = transport["stdout"].splitlines()
                    if len(lines) != 1:
                        primary = _sandbox_failure("sandbox did not emit exactly one receipt", container_name=name)
                    else:
                        parsed = strict_json_loads(lines[0])
                        primary = _validate_sandbox_receipt(
                            parsed,
                            image_id=image_id,
                            request_sha256=request_sha256,
                            candidate_sha256=candidate_sha256,
                            reference_sha256=reference_sha256,
                            driver_source_sha256=driver_source_sha256,
                        )
        except Exception as exc:
            primary = _sandbox_failure(f"sandbox runtime error: {type(exc).__name__}", container_name=name)
    finally:
        cleanup = _cleanup_sandbox(name)

    if not cleanup.get("verified_absent"):
        return _sandbox_failure("sandbox cleanup was not verified", container_name=name, cleanup=cleanup)
    if primary is None:
        return _sandbox_failure("sandbox produced no result", container_name=name, cleanup=cleanup)
    primary["cleanup"] = cleanup
    primary["container_name"] = name
    return primary


def grade_exec(text: str, ref: Mapping[str, Any] | None, timeout: float = 8.0, *, image_id: str | None = None) -> bool:
    """Compatibility bool wrapper; no host execution fallback exists."""
    return grade_exec_result(text, ref, timeout=timeout, image_id=image_id).get("passed") is True


# Fixed external tests for the 20 frozen agentic-coding rows.  Generated
# self-tests are allowed to run, but never count as ground truth by themselves.
AGENTIC_RUBRIC_VERSION = "r0b0tlab.qwen38.agentic_external.v2"
AGENTIC_EXTERNAL_TESTS: dict[str, str] = {
    "agentic-00": """
assert top_k_frequent(["i", "love", "leetcode", "i", "love", "coding"], 2) == ["i", "love"]
assert top_k_frequent(["b", "a", "c", "b", "a", "c"], 3) == ["a", "b", "c"]
assert top_k_frequent([], 0) == []
""",
    "agentic-01": """
assert flatten([1, [2, [3], []], 4]) == [1, 2, 3, 4]
assert flatten([]) == []
_deep = 0
for _ in range(2500):
    _deep = [_deep]
assert flatten(_deep) == [0]
""",
    "agentic-02": """
_factory = globals().get("lru_cache") or globals().get("LRUCache")
assert callable(_factory)
_cache = _factory(2)
_cache.put("a", 1); _cache.put("b", 2)
assert _cache.get("a") == 1
_cache.put("c", 3)
assert _cache.get("b") in (None, -1)
assert _cache.get("a") == 1 and _cache.get("c") == 3
""",
    "agentic-03": """
def _normalize_intervals(value): return [list(interval) for interval in value]
assert _normalize_intervals(merge_intervals([[1, 3], [2, 4], [8, 9]])) == [[1, 4], [8, 9]]
assert _normalize_intervals(merge_intervals([[1, 2], [2, 3]])) == [[1, 2], [2, 3]]
assert _normalize_intervals(merge_intervals([])) == []
""",
    "agentic-04": """
assert word_break("leetcode", {"leet", "code"}) is True
assert word_break("catsandog", {"cats", "dog", "sand", "and", "cat"}) is False
assert word_break("", {"x"}) is True
""",
    "agentic-05": """
assert min_window("ADOBECODEBANC", "ABC") == "BANC"
assert min_window("aa", "aa") == "aa"
assert min_window("a", "aa") == ""
""",
    "agentic-06": """
class _ExternalListNode:
    def __init__(self, value): self.val, self.next = value, None
_a, _b, _c = _ExternalListNode(1), _ExternalListNode(2), _ExternalListNode(3)
_a.next = _b; _b.next = _c; _c.next = _b
assert bool(detect_cycle(_a)) is True
_c.next = None
assert bool(detect_cycle(_a)) is False
assert bool(detect_cycle(None)) is False
""",
    "agentic-07": """
class _ExternalTreeNode:
    def __init__(self, value, left=None, right=None): self.val, self.left, self.right = value, left, right
_root = _ExternalTreeNode(1, _ExternalTreeNode(2), _ExternalTreeNode(3, _ExternalTreeNode(4), None))
_encoded = serialize(_root)
_decoded = deserialize(_encoded)
assert serialize(_decoded) == _encoded
assert getattr(_decoded, "val", None) == 1
assert deserialize(serialize(None)) is None
""",
    "agentic-08": """
assert trapping_rain_water([0,1,0,2,1,0,1,3,2,1,2,1]) == 6
assert trapping_rain_water([4,2,0,3,2,5]) == 9
assert trapping_rain_water([]) == 0
""",
    "agentic-09": """
assert edit_distance("horse", "ros") == 3
assert edit_distance("intention", "execution") == 5
assert edit_distance("", "abc") == 3
""",
    "agentic-10": """
_limiter = RateLimiter(60)
assert sum(bool(_limiter.allow()) for _ in range(70)) == 60
assert _limiter.allow() is False
""",
    "agentic-11": """
assert parse_csv_line('a,"b,c","d""e"') == ["a", "b,c", 'd"e']
assert parse_csv_line(",x,") == ["", "x", ""]
assert parse_csv_line('"a"') == ["a"]
""",
    "agentic-12": """
_state = {"n": 0}
def _flaky():
    _state["n"] += 1
    if _state["n"] < 3: raise ValueError("retry")
    return "ok"
_wrapped = retry(_flaky, 3, 0)
assert callable(_wrapped) and _wrapped() == "ok" and _state["n"] == 3
_state2 = {"n": 0}
def _always():
    _state2["n"] += 1
    raise RuntimeError("stop")
_wrapped2 = retry(_always, 2, 0)
try:
    _wrapped2()
    raise AssertionError("retry swallowed terminal failure")
except RuntimeError:
    pass
assert _state2["n"] == 2
""",
    "agentic-13": """
_result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
assert all(group == sorted(group) for group in _result)
assert sorted(tuple(group) for group in _result) == [("ate", "eat", "tea"), ("bat",), ("nat", "tan")]
assert group_anagrams([]) == []
""",
    "agentic-14": """
_matrix = [[1,2,3],[4,5,6],[7,8,9]]
matrix_rotate_90(_matrix)
assert _matrix == [[7,4,1],[8,5,2],[9,6,3]]
_single = [[1]]; matrix_rotate_90(_single); assert _single == [[1]]
""",
    "agentic-15": """
_obj = {"a": {"b": [{"c": 7}, {"c": 9}]}, "x": [1, 2]}
assert jsonpath_get(_obj, "a.b[1].c") == 9
assert jsonpath_get(_obj, "x[0]") == 1
""",
    "agentic-16": """
assert validate_parentheses("([]{})") is True
assert validate_parentheses("([)]") is False
assert validate_parentheses("") is True
""",
    "agentic-17": """
def _lis_ok(nums, value, expected):
    if isinstance(value, int): return value == expected
    seq = list(value)
    if len(seq) != expected or any(a >= b for a, b in zip(seq, seq[1:])): return False
    pos = 0
    for item in nums:
        if pos < len(seq) and item == seq[pos]: pos += 1
    return pos == len(seq)
_nums = [10,9,2,5,3,7,101,18]
assert _lis_ok(_nums, longest_increasing_subsequence(_nums), 4)
assert _lis_ok([], longest_increasing_subsequence([]), 0)
""",
    "agentic-18": """
import inspect as _inspect
import re as _re
_candidate_sources = []
for _name, _value in list(globals().items()):
    if _name.startswith("_External") or _name.startswith("_candidate"):
        continue
    if (_inspect.isfunction(_value) or _inspect.isclass(_value)) and getattr(_value, "__module__", None) == "__main__":
        try: _candidate_sources.append(_inspect.getsource(_value))
        except (OSError, TypeError): pass
_candidate_source = "".join(_candidate_sources)
assert "sha256" in _candidate_source.lower()
assert _re.search(r"(?<![0-9])100(?![0-9])", _candidate_source)
_first = [shard_key(f"user-{i}", 10) for i in range(300)]
_second = [shard_key(f"user-{i}", 10) for i in range(300)]
assert _first == _second
assert all(isinstance(value, int) and 0 <= value < 10 for value in _first)
assert len(set(_first)) == 10
_grown = [shard_key(f"user-{i}", 11) for i in range(300)]
_moved = sum(a != b for a, b in zip(_first, _grown))
assert 0 < _moved < 105
assert all(a == b or b == 10 for a, b in zip(_first, _grown))
""",
    "agentic-19": """
import inspect as _inspect
_signature = _inspect.signature(debounce)
assert "fn" in _signature.parameters and "wait_ms" in _signature.parameters
assert any(name in _signature.parameters for name in ("time_source", "clock", "now"))
_clock = [0.0]
_calls = []
_kwargs = {}
for _name in ("time_source", "clock", "now"):
    if _name in _signature.parameters:
        _kwargs[_name] = lambda: _clock[0]
        break
_wrapped = debounce(lambda value: _calls.append(value), 100, **_kwargs)
assert callable(_wrapped)
_wrapped(1); _clock[0] = 0.05; _wrapped(2)
assert len(_calls) <= 1
_clock[0] = 0.20; _wrapped(3)
if hasattr(_wrapped, "flush"): _wrapped.flush()
assert len(_calls) <= 2 and _calls and _calls[-1] in (2, 3)
""",
}


def grade_agentic_result(
    text: str,
    row_id: Any,
    timeout: float = 8.0,
    *,
    image_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(row_id, str) or row_id not in AGENTIC_EXTERNAL_TESTS:
        return {"status": "grader_error", "passed": None, "failure_class": "HARNESS_BLOCK", "reason": f"missing_external_rubric:{row_id}"}
    result = grade_exec_result(
        text,
        {"test": AGENTIC_EXTERNAL_TESTS[row_id]},
        timeout=timeout,
        image_id=image_id,
    )
    result["rubric"] = AGENTIC_RUBRIC_VERSION
    result["rubric_id"] = row_id
    return result


IFEVAL_SUPPORTED_INSTRUCTIONS = frozenset(
    {
        "change_case:capital_word_frequency",
        "change_case:english_capital",
        "change_case:english_lowercase",
        "combination:repeat_prompt",
        "combination:two_responses",
        "detectable_content:number_placeholders",
        "detectable_content:postscript",
        "detectable_format:json_format",
        "detectable_format:multiple_sections",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections",
        "detectable_format:title",
        "keywords:existence",
        "keywords:forbidden_words",
        "keywords:frequency",
        "keywords:letter_frequency",
        "language:response_language",
        "length_constraints:number_paragraphs",
        "length_constraints:number_sentences",
        "length_constraints:number_words",
        "punctuation:no_comma",
        "startend:end_checker",
        "startend:quotation",
    }
)


def _ifeval_int(args: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _ifeval_text(args: Mapping[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _ifeval_relation(actual: int, expected: int, relation: Any) -> bool:
    if relation == "less than":
        return actual < expected
    if relation == "at least":
        return actual >= expected
    raise ValueError("relation must be exactly 'less than' or 'at least'")


def _ifeval_word_count(text: str) -> int:
    # IFEval's reference uses RegexpTokenizer(r"\\w+").  Python's Unicode
    # regex has the same relevant contract for this frozen local subset.
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def _ifeval_sentence_count(text: str) -> int:
    # Dependency-free frozen-subset approximation of Punkt.  This scorer is
    # intentionally labeled local IFEval rather than official full IFEval.
    return sum(bool(chunk.strip(" \t\r\n.!?")) for chunk in re.findall(r"[^.!?]+(?:[.!?]+|$)", text))


def _ifeval_only_language(text: str, language: str) -> bool:
    ranges = {"kn": (0x0C80, 0x0CFF), "pa": (0x0A00, 0x0A7F)}
    if language not in ranges:
        raise ValueError(f"unsupported frozen response language: {language}")
    low, high = ranges[language]
    letters = [char for char in text if unicodedata.category(char).startswith("L")]
    return bool(letters) and all(low <= ord(char) <= high for char in letters)


def _ifeval_english_case(text: str, *, upper: bool) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for char in letters):
        return False
    return text.isupper() if upper else text.islower()


def _check_ifeval_instruction(instruction: str, text: str, args: Mapping[str, Any]) -> bool:
    if instruction == "keywords:existence":
        keywords = args.get("keywords")
        if not isinstance(keywords, list) or not keywords or any(not isinstance(word, str) or not word for word in keywords):
            raise ValueError("keywords must be a non-empty list of non-empty strings")
        return all(re.search(re.escape(word), text, flags=re.IGNORECASE) is not None for word in keywords)

    if instruction == "keywords:forbidden_words":
        words = args.get("forbidden_words")
        if not isinstance(words, list) or not words or any(not isinstance(word, str) or not word for word in words):
            raise ValueError("forbidden_words must be a non-empty list of non-empty strings")
        return all(re.search(r"\b" + re.escape(word) + r"\b", text, flags=re.IGNORECASE) is None for word in words)

    if instruction == "keywords:frequency":
        keyword = _ifeval_text(args, "keyword")
        frequency = _ifeval_int(args, "frequency", minimum=0)
        actual = len(re.findall(re.escape(keyword), text, flags=re.IGNORECASE))
        return _ifeval_relation(actual, frequency, args.get("relation"))

    if instruction == "keywords:letter_frequency":
        letter = _ifeval_text(args, "letter")
        if len(letter) != 1:
            raise ValueError("letter must contain exactly one character")
        frequency = _ifeval_int(args, "let_frequency", minimum=0)
        actual = text.lower().count(letter.lower())
        return _ifeval_relation(actual, frequency, args.get("let_relation"))

    if instruction == "length_constraints:number_words":
        expected = _ifeval_int(args, "num_words", minimum=0)
        return _ifeval_relation(_ifeval_word_count(text), expected, args.get("relation"))

    if instruction == "length_constraints:number_sentences":
        expected = _ifeval_int(args, "num_sentences", minimum=0)
        return _ifeval_relation(_ifeval_sentence_count(text), expected, args.get("relation"))

    if instruction == "length_constraints:number_paragraphs":
        expected = _ifeval_int(args, "num_paragraphs", minimum=1)
        paragraphs = re.split(r"\s?\*\*\*\s?", text)
        actual = len(paragraphs)
        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index in {0, len(paragraphs) - 1}:
                    actual -= 1
                else:
                    return False
        return actual == expected

    if instruction == "punctuation:no_comma":
        return "," not in text

    if instruction == "startend:quotation":
        value = text.strip()
        return len(value) > 1 and value.startswith('"') and value.endswith('"')

    if instruction == "startend:end_checker":
        phrase = _ifeval_text(args, "end_phrase").lower()
        return text.strip().strip('"').lower().endswith(phrase)

    if instruction == "detectable_content:number_placeholders":
        expected = _ifeval_int(args, "num_placeholders", minimum=1)
        return len(re.findall(r"\[.*?\]", text)) >= expected

    if instruction == "detectable_content:postscript":
        marker = _ifeval_text(args, "postscript_marker")
        if marker == "P.P.S":
            pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif marker == "P.S.":
            pattern = r"\s*p\.\s?s\..*$"
        else:
            pattern = r"\s*" + re.escape(marker.lower()) + r".*$"
        return re.search(pattern, text.lower(), flags=re.MULTILINE) is not None

    if instruction == "detectable_format:number_bullet_lists":
        expected = _ifeval_int(args, "num_bullets", minimum=1)
        stars = re.findall(r"^\s*\*[^\*].*$", text, flags=re.MULTILINE)
        dashes = re.findall(r"^\s*-.*$", text, flags=re.MULTILINE)
        return len(stars) + len(dashes) == expected

    if instruction == "detectable_format:number_highlighted_sections":
        expected = _ifeval_int(args, "num_highlights", minimum=1)
        singles = re.findall(r"\*[^\n\*]*\*", text)
        doubles = re.findall(r"\*\*[^\n\*]*\*\*", text)
        actual = sum(bool(value.strip("*").strip()) for value in singles)
        actual += sum(bool(value.removeprefix("**").removesuffix("**").strip()) for value in doubles)
        return actual >= expected

    if instruction == "detectable_format:multiple_sections":
        splitter = _ifeval_text(args, "section_spliter")
        expected = _ifeval_int(args, "num_sections", minimum=1)
        sections = re.split(r"\s?" + re.escape(splitter) + r"\s?\d+\s?", text)
        return len(sections) - 1 >= expected

    if instruction == "detectable_format:title":
        return any(title.lstrip("<").rstrip(">").strip() for title in re.findall(r"<<[^\n]+>>", text))

    if instruction == "detectable_format:json_format":
        candidate = text.strip()
        for prefix in ("```json", "```Json", "```JSON", "```"):
            if candidate.startswith(prefix):
                candidate = candidate.removeprefix(prefix)
                break
        candidate = candidate.removesuffix("```").strip()
        try:
            strict_json_loads(candidate)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return False
        return True

    if instruction == "combination:two_responses":
        valid: list[str] = []
        responses = text.split("******")
        for index, response in enumerate(responses):
            if response.strip():
                valid.append(response.strip())
            elif index not in {0, len(responses) - 1}:
                return False
        return len(valid) == 2 and valid[0] != valid[1]

    if instruction == "combination:repeat_prompt":
        prompt = _ifeval_text(args, "prompt_to_repeat")
        return text.strip().lower().startswith(prompt.strip().lower())

    if instruction == "change_case:english_capital":
        return _ifeval_english_case(text, upper=True)

    if instruction == "change_case:english_lowercase":
        return _ifeval_english_case(text, upper=False)

    if instruction == "change_case:capital_word_frequency":
        expected = _ifeval_int(args, "capital_frequency", minimum=0)
        words = re.findall(r"\b\w+(?:[-']\w+)*\b", text, flags=re.UNICODE)
        actual = sum(word.isupper() for word in words)
        return _ifeval_relation(actual, expected, args.get("capital_relation"))

    if instruction == "language:response_language":
        return _ifeval_only_language(text, _ifeval_text(args, "language"))

    raise ValueError(f"unsupported instruction: {instruction}")


def grade_ifeval_result(text: str, ref: Mapping[str, Any]) -> dict[str, Any]:
    """Grade the exact frozen local IFEval subset and fail closed on drift."""
    if not isinstance(text, str) or not text.strip():
        return {"status": "scored", "passed": False, "checks": [], "reason": "empty_response"}
    if not isinstance(ref, Mapping):
        return {"status": "grader_error", "passed": None, "checks": [], "reason": "reference_not_object"}
    ids = ref.get("instruction_id_list")
    kwargs = ref.get("kwargs")
    if not isinstance(ids, list) or not ids or not isinstance(kwargs, list) or len(ids) != len(kwargs):
        return {"status": "grader_error", "passed": None, "checks": [], "reason": "invalid_instruction_vectors"}
    checks: list[dict[str, Any]] = []
    for index, instruction in enumerate(ids):
        if not isinstance(instruction, str) or instruction not in IFEVAL_SUPPORTED_INSTRUCTIONS:
            return {
                "status": "grader_error",
                "passed": None,
                "checks": checks,
                "reason": f"unsupported_instruction:{instruction}",
                "instruction_index": index,
            }
        args = kwargs[index]
        if not isinstance(args, Mapping):
            return {
                "status": "grader_error",
                "passed": None,
                "checks": checks,
                "reason": "instruction_kwargs_not_object",
                "instruction": instruction,
                "instruction_index": index,
            }
        try:
            followed = _check_ifeval_instruction(instruction, text, args)
        except (TypeError, ValueError, re.error, json.JSONDecodeError) as exc:
            return {
                "status": "grader_error",
                "passed": None,
                "checks": checks,
                "reason": f"invalid_instruction:{instruction}:{type(exc).__name__}:{exc}",
                "instruction": instruction,
                "instruction_index": index,
            }
        checks.append({"instruction": instruction, "passed": bool(followed)})
    return {"status": "scored", "passed": all(check["passed"] for check in checks), "checks": checks}


def grade_ifeval(text: str, ref: Mapping[str, Any]) -> bool:
    """Compatibility bool wrapper; unsupported/malformed references never pass."""
    return grade_ifeval_result(text, ref).get("passed") is True


def load_resume_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank resume line {line_number}")
            row = strict_json_loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"invalid resume row {line_number}")
            if row["id"] in rows:
                raise ValueError(f"duplicate resume id {row['id']}")
            rows[row["id"]] = row
    return rows


def load_manual_evidence(
    path: str | Path,
    *,
    dataset: Iterable[Mapping[str, Any]],
    responses: Iterable[Mapping[str, Any]],
    dataset_sha256: str,
    run_identity_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load independent hard/agentic grades bound to exact response bytes."""
    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    value = strict_json_loads(raw)
    required_top = {
        "schema",
        "dataset_sha256",
        "run_identity_sha256",
        "reviewer",
        "method",
        "rows",
    }
    if not isinstance(value, Mapping) or set(value) != required_top:
        raise ValueError("manual evidence top-level schema is incomplete or has unknown fields")
    if value.get("schema") != MANUAL_EVIDENCE_SCHEMA:
        raise ValueError("manual evidence schema mismatch")
    if value.get("dataset_sha256") != dataset_sha256:
        raise ValueError("manual evidence dataset hash mismatch")
    if value.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError("manual evidence run identity mismatch")
    reviewer = value.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip() or any(char in reviewer for char in ("\x00", "\n", "\r")):
        raise ValueError("manual evidence reviewer must be a non-empty single-line string")
    if value.get("method") != MANUAL_REVIEW_METHOD:
        raise ValueError("manual evidence method mismatch")
    expected_ids = {
        str(row["id"])
        for row in dataset
        if row.get("family") in {"hard_reasoning", "agentic_coding"}
    }
    response_map = {str(row.get("id")): row for row in responses if isinstance(row.get("id"), str)}
    evidence_rows = value.get("rows")
    if not isinstance(evidence_rows, list):
        raise ValueError("manual evidence rows must be an array")
    observed: dict[str, dict[str, Any]] = {}
    required_row = {"id", "content_sha256", "passed", "rationale"}
    for index, row in enumerate(evidence_rows):
        if not isinstance(row, Mapping) or set(row) != required_row:
            raise ValueError(f"manual evidence row {index} is malformed")
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id not in expected_ids or row_id in observed:
            raise ValueError(f"manual evidence row ID is unknown or duplicate: {row_id}")
        content_sha256 = row.get("content_sha256")
        if not isinstance(content_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", content_sha256):
            raise ValueError(f"manual evidence content hash is invalid: {row_id}")
        response = response_map.get(row_id)
        content = response.get("content") if isinstance(response, Mapping) else None
        if not isinstance(content, str) or hashlib.sha256(content.encode("utf-8")).hexdigest() != content_sha256:
            raise ValueError(f"manual evidence response binding mismatch: {row_id}")
        if not isinstance(row.get("passed"), bool):
            raise ValueError(f"manual evidence passed flag is invalid: {row_id}")
        rationale = row.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 4000:
            raise ValueError(f"manual evidence rationale is invalid: {row_id}")
        observed[row_id] = dict(row)
    if set(observed) != expected_ids:
        missing = sorted(expected_ids - set(observed))
        extra = sorted(set(observed) - expected_ids)
        raise ValueError(f"manual evidence ID set mismatch; missing={missing}, extra={extra}")
    return observed, {
        "path": str(source.resolve()),
        "sha256": digest,
        "reviewer": reviewer,
        "method": MANUAL_REVIEW_METHOD,
        "row_count": len(observed),
        "run_identity_sha256": run_identity_sha256,
    }


def validate_completeness(rows: Iterable[Mapping[str, Any]], dataset: Iterable[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    expected = {str(row["id"]): row for row in dataset}
    observed: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str):
            errors.append("row missing id")
            continue
        if row_id in observed:
            errors.append(f"duplicate result id: {row_id}")
        observed[row_id] = row
    missing = sorted(set(expected) - set(observed))
    unknown = sorted(set(observed) - set(expected))
    errors.extend(f"missing result id: {value}" for value in missing)
    errors.extend(f"unknown result id: {value}" for value in unknown)
    for row_id, row in observed.items():
        if row_id not in expected:
            continue
        source = expected[row_id]
        if row.get("family") != source.get("family") or row.get("grade") != source.get("grade"):
            errors.append(f"row metadata mismatch: {row_id}")
        if row.get("status") != 200 or row.get("response_status") != 200 or row.get("error") is not None:
            errors.append(f"transport failure: {row_id}")
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"missing/empty content: {row_id}")
        else:
            content_sha256 = row.get("content_sha256")
            expected_content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_sha256 != expected_content_sha256:
                errors.append(f"missing/invalid content hash: {row_id}")
        finish_reason = row.get("finish_reason")
        if finish_reason not in Q200_FINISH_REASONS:
            errors.append(f"missing/invalid finish_reason: {row_id}")
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            errors.append(f"missing usage: {row_id}")
        else:
            for key in ("prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    errors.append(f"missing/invalid usage.{key}: {row_id}")
    return not errors and len(observed) == len(expected), errors


def validate_resume_identity(rows: Iterable[Mapping[str, Any]], dataset: Iterable[Mapping[str, Any]], *, identity: Mapping[str, Any], max_tokens: int) -> list[str]:
    expected = {str(row["id"]): row for row in dataset}
    errors: list[str] = []
    for row in rows:
        row_id = row.get("id")
        source = expected.get(row_id) if isinstance(row_id, str) else None
        if source is None:
            continue
        expected_row = _row_identity(identity=identity, source=source, max_tokens=max_tokens)
        for key in ("model", "image_id", "profile_id", "candidate_id", "dataset_sha256", "prompt_sha256", "request_max_tokens", "request_sha256"):
            if row.get(key) != expected_row[key]:
                errors.append(f"resume identity mismatch ({key}): {row_id}")
        if row.get("chat_template_kwargs") != expected_row["chat_template_kwargs"]:
            errors.append(f"resume identity mismatch (chat_template_kwargs): {row_id}")
        if row.get("identity_sha256") != _sha256_json(expected_row):
            errors.append(f"resume identity hash mismatch: {row_id}")
    return errors


def grade_summary(
    rows: Iterable[Mapping[str, Any]],
    dataset: Iterable[Mapping[str, Any]],
    *,
    human_eval_timeout: float = 8.0,
    recompute: bool = False,
    manual_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    sandbox_image_id: str | None = None,
) -> dict[str, Any]:
    expected = {str(row["id"]): row for row in dataset}
    families: dict[str, dict[str, Any]] = {family: {"n": 0, "transported": 0, "grade_complete": 0, "correct": 0, "incorrect": 0, "ungraded": 0, "grader_errors": 0, "accuracy_pct": None} for family in EXPECTED_FAMILY_COUNTS}
    correct = incorrect = ungraded = grader_errors = transported = 0
    for row in rows:
        row_id = row.get("id")
        source = expected.get(row_id) if isinstance(row_id, str) else None
        if source is None:
            continue
        family = str(source["family"])
        stats = families.setdefault(family, {"n": 0, "transported": 0, "grade_complete": 0, "correct": 0, "incorrect": 0, "ungraded": 0, "grader_errors": 0, "accuracy_pct": None})
        stats["n"] += 1
        transported_row = row.get("status") == 200 and not row.get("error") and isinstance(row.get("usage"), Mapping)
        if transported_row:
            transported += 1
            stats["transported"] += 1
        else:
            ungraded += 1
            stats["ungraded"] += 1
            continue
        grader = _grade_row(source, row, human_eval_timeout=human_eval_timeout, manual_evidence=manual_evidence, sandbox_image_id=sandbox_image_id) if recompute else row.get("grader")
        passed = grader.get("passed") if isinstance(grader, Mapping) else None
        grader_status = str(grader.get("status", "")) if isinstance(grader, Mapping) else "missing"
        failure_class = grader.get("failure_class") if isinstance(grader, Mapping) else None
        if grader_status in {"error", "grader_error"} or failure_class == "HARNESS_BLOCK":
            grader_errors += 1
            stats["grader_errors"] += 1
        elif passed is True and grader_status in {"scored", "ok"}:
            correct += 1
            stats["correct"] += 1
            stats["grade_complete"] += 1
        elif passed is False and grader_status in {"scored", "ok", "failed", "timeout", "output_limit", "oom", "empty_code"} and failure_class != "HARNESS_BLOCK":
            incorrect += 1
            stats["incorrect"] += 1
            stats["grade_complete"] += 1
        else:
            ungraded += 1
            stats["ungraded"] += 1
    for stats in families.values():
        graded = stats["correct"] + stats["incorrect"]
        stats["accuracy_pct"] = round(100.0 * stats["correct"] / graded, 2) if graded else None
    return {
        "transported": transported,
        "correct": correct,
        "incorrect": incorrect,
        "ungraded": ungraded,
        "grader_errors": grader_errors,
        "grade_complete": ungraded == 0 and grader_errors == 0 and correct + incorrect == sum(stats["n"] for stats in families.values()),
        "families": families,
    }


def _manual_grade(
    source: Mapping[str, Any],
    response: Mapping[str, Any],
    manual_evidence: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    row_id = source.get("id")
    evidence = manual_evidence.get(row_id) if manual_evidence is not None and isinstance(row_id, str) else None
    if evidence is None:
        return {"status": "ungraded", "passed": None, "reason": "manual_review_required"}
    content = str(response.get("content") or "")
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if evidence.get("content_sha256") != content_sha256:
        return {
            "status": "grader_error",
            "passed": None,
            "failure_class": "HARNESS_BLOCK",
            "reason": "manual evidence response binding mismatch",
        }
    return {
        "status": "scored",
        "passed": evidence["passed"],
        "review_method": MANUAL_REVIEW_METHOD,
        "content_sha256": content_sha256,
        "rationale": evidence["rationale"],
    }


def _grade_row(
    source: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    human_eval_timeout: float,
    manual_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    sandbox_image_id: str | None = None,
) -> Any:
    grade = source.get("grade")
    content = str(response.get("content") or "")
    reference = source.get("reference")
    if grade == "numeric_exact":
        expected = normalize_number(reference)
        got = normalize_number(final_answer(content))
        return {"status": "scored", "passed": expected is not None and got is not None and expected == got, "expected": expected, "got": got}
    if grade == "exec":
        if source.get("family") == "agentic_coding":
            # External code execution remains available as a diagnostic, but
            # the frozen agentic contracts contain semantic constraints that
            # cannot be made claim-bearing by finite examples alone.  Require
            # independent, response-bound adjudication for the published score.
            return _manual_grade(source, response, manual_evidence)
        reference_map = reference if isinstance(reference, Mapping) else {}
        entry_point = reference_map.get("entry_point")
        if not isinstance(entry_point, str) or not entry_point:
            return _sandbox_failure("HumanEval entry point is malformed")
        try:
            prelude = _human_eval_prelude(str(source.get("prompt") or ""), entry_point)
        except (SyntaxError, ValueError, TypeError):
            return _sandbox_failure("HumanEval prompt context is malformed")
        return grade_exec_result(
            content,
            reference_map,
            timeout=human_eval_timeout,
            image_id=sandbox_image_id,
            prelude=prelude,
        )
    if grade == "ifeval_strict":
        return grade_ifeval_result(content, reference if isinstance(reference, Mapping) else {})
    if grade == "manual":
        return _manual_grade(source, response, manual_evidence)
    return {"status": "error", "passed": None, "reason": f"unsupported_grade:{grade}"}


def run_quality(*, base_url: str, run_id: str, dataset_path: str | Path, admission: AdmissionCoordinator, model: str = MODEL, max_tokens: int = 4096, timeout: int = 600, workers: int = 1, human_eval_timeout: float = 8.0, image_id: str | None = None, profile_id: str | None = None, candidate_id: str | None = None, chat_kwargs: Mapping[str, Any] | None = None, manual_evidence_path: str | Path | None = None) -> dict[str, Any]:
    dataset, dataset_hash = read_quality_set(dataset_path)
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4 for the frozen production profile")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if not all(isinstance(value, str) and value for value in (image_id, profile_id, candidate_id)):
        raise ValueError("image_id, profile_id, and candidate_id are required for Q200 identity binding")
    if not Q200_SANDBOX_IMAGE_RE.fullmatch(str(image_id)):
        raise ValueError("image_id must be a full immutable sha256 image ID")
    effective_chat_kwargs = validate_native_thinking(DEFAULT_CHAT_KWARGS if chat_kwargs is None else chat_kwargs)
    identity = _run_identity(model=model, image_id=image_id, profile_id=profile_id, candidate_id=candidate_id, dataset_sha256=dataset_hash, chat_kwargs=effective_chat_kwargs)
    rows_path = Path(f"{run_id}.rows.jsonl")
    existing = load_resume_rows(rows_path)
    expected_ids = {row["id"] for row in dataset}
    if set(existing) - expected_ids:
        raise ValueError("resume contains IDs outside the frozen dataset")
    identity_errors = validate_resume_identity(existing.values(), dataset, identity=identity, max_tokens=max_tokens)
    if identity_errors:
        raise ValueError("; ".join(identity_errors[:8]))
    # Existing successful rows are immutable evidence; failed rows are also
    # retained and make the completeness gate fail rather than being hidden.
    pending = [row for row in dataset if row["id"] not in existing]
    rows_by_id = dict(existing)
    rows_path.parent.mkdir(parents=True, exist_ok=True)

    def one(source: Mapping[str, Any]) -> dict[str, Any]:
        row_id = str(source["id"])
        row_identity = _row_identity(identity=identity, source=source, max_tokens=max_tokens)
        request_max = row_identity["request_max_tokens"]
        response = chat(base_url, str(source["prompt"]), request_max, timeout, model=model, chat_kwargs=effective_chat_kwargs)
        grader = _grade_row(source, response, human_eval_timeout=human_eval_timeout, sandbox_image_id=identity["image_id"])
        usage = response.get("usage")
        return {
            "id": row_id,
            "family": source["family"],
            "grade": source["grade"],
            "model": model,
            "model_id": response.get("model_id"),
            "request_sha256": row_identity["request_sha256"],
            "prompt_sha256": row_identity["prompt_sha256"],
            "request_max_tokens": request_max,
            "dataset_sha256": dataset_hash,
            "chat_template_kwargs": dict(effective_chat_kwargs),
            "identity_sha256": _sha256_json(row_identity),
            "status": response.get("status"),
            "response_status": response.get("response_status"),
            "error": response.get("error"),
            "content": response.get("content") or "",
            "content_sha256": hashlib.sha256(str(response.get("content") or "").encode("utf-8")).hexdigest(),
            "reasoning_content": response.get("reasoning_content") or "",
            "text": response.get("content") or "",  # compatibility alias; never truncated
            "usage": usage,
            "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, Mapping) else None,
            "completion_tokens": usage.get("completion_tokens") if isinstance(usage, Mapping) else None,
            "finish_reason": response.get("finish_reason"),
            "finish": response.get("finish_reason"),
            "elapsed_seconds": response.get("elapsed_seconds"),
            "elapsed": response.get("elapsed_seconds"),
            "response_sha256": response.get("response_sha256"),
            "grader": grader,
            "passed": grader.get("passed") if isinstance(grader, Mapping) else grader,
            "image_id": identity["image_id"],
            "profile_id": identity["profile_id"],
            "candidate_id": identity["candidate_id"],
        }

    not_admitted_ids: list[str] = []
    not_admitted_event: dict[str, Any] | None = None
    with rows_path.open("a", encoding="utf-8", buffering=1) as output:
        for batch_offset in range(0, len(pending), workers):
            batch_sources = pending[batch_offset : batch_offset + workers]
            row_id = f"q200-batch-{batch_offset // workers:03d}-{batch_sources[0]['id']}-{batch_sources[-1]['id']}"
            try:
                with admission.request(row_id) as lease:
                    if len(batch_sources) == 1:
                        batch_results = [one(batch_sources[0])]
                    else:
                        with ThreadPoolExecutor(max_workers=len(batch_sources)) as pool:
                            futures = {pool.submit(one, source): source for source in batch_sources}
                            batch_results = [future.result() for future in as_completed(futures)]
            except AdmissionNotGranted as exc:
                not_admitted_ids = [str(source["id"]) for source in pending[batch_offset:]]
                not_admitted_event = {"row_id": row_id, "error": str(exc)}
                break
            for result in batch_results:
                result["admission_lease"] = lease
                rows_by_id[result["id"]] = result
                output.write(json.dumps(result, sort_keys=True) + "\n")
                output.flush()

    ordered = [rows_by_id[row["id"]] for row in dataset if row["id"] in rows_by_id]
    transport_complete, errors = validate_completeness(ordered, dataset)
    manual_evidence: dict[str, dict[str, Any]] | None = None
    manual_evidence_meta: dict[str, Any] | None = None
    run_identity_sha256 = _sha256_json(identity)
    if manual_evidence_path is not None:
        manual_evidence, manual_evidence_meta = load_manual_evidence(
            manual_evidence_path,
            dataset=dataset,
            responses=ordered,
            dataset_sha256=dataset_hash,
            run_identity_sha256=run_identity_sha256,
        )
    grades = grade_summary(
        ordered,
        dataset,
        human_eval_timeout=human_eval_timeout,
        recompute=True,
        manual_evidence=manual_evidence,
        sandbox_image_id=identity["image_id"],
    )
    grade_complete = bool(grades["grade_complete"])
    scored = transport_complete and grade_complete and grades["transported"] == EXPECTED_COUNT
    finish_reason_counts: dict[str, int] = {}
    for row in ordered:
        reason = str(row.get("finish_reason"))
        finish_reason_counts[reason] = finish_reason_counts.get(reason, 0) + 1
    observed_completion_tokens = [
        int(row["completion_tokens"])
        for row in ordered
        if isinstance(row.get("completion_tokens"), int) and not isinstance(row.get("completion_tokens"), bool)
    ]
    ceiling_contact_ids = [
        str(row.get("id"))
        for row in ordered
        if isinstance(row.get("completion_tokens"), int)
        and not isinstance(row.get("completion_tokens"), bool)
        and isinstance(row.get("request_max_tokens"), int)
        and not isinstance(row.get("request_max_tokens"), bool)
        and int(row["completion_tokens"]) >= int(row["request_max_tokens"])
    ]
    summary = {
        "schema": "r0b0tlab.qwen38.quality_text_180_run.v2",
        "status": "SCORED" if scored else "INCOMPLETE",
        "run_id": run_id,
        "base_url": base_url,
        "model": model,
        "dataset_sha256": dataset_hash,
        "dataset_count": len(dataset),
        "chat_template_kwargs": dict(effective_chat_kwargs),

        "response_budget": {
            "configured_max_tokens": max_tokens,
            "finish_reason_counts": finish_reason_counts,
            "max_observed_completion_tokens": max(observed_completion_tokens, default=None),
            "ceiling_contact_ids": ceiling_contact_ids,
            "all_rows_stopped": len(ordered) == EXPECTED_COUNT and finish_reason_counts == {"stop": EXPECTED_COUNT},
        },
        "families": grades["families"],
        "rows": len(ordered),
        "transport_count": grades["transported"],
        "transport_complete": transport_complete and grades["transported"] == EXPECTED_COUNT,
        "grade_complete": grade_complete,
        "correct_count": grades["correct"],
        "incorrect_count": grades["incorrect"],
        "ungraded_count": grades["ungraded"],
        "grader_error_count": grades["grader_errors"],
        "not_admitted_count": len(not_admitted_ids),
        "not_admitted_ids": not_admitted_ids,
        "not_admitted_event": not_admitted_event,
        "quality_policy": {"name": "external_baseline_comparator_required", "applied": False, "passed": None},
        "completeness_errors": errors,
        "image_id": image_id,
        "profile_id": profile_id,
        "candidate_id": candidate_id,
        "run_identity_sha256": run_identity_sha256,
        "manual_evidence": manual_evidence_meta,
    }
    Path(f"{run_id}.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--set", default="artifacts/quality-text-180-v2.jsonl")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--human-eval-timeout", type=float, default=8.0)
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--profile-id", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--admission-config", type=Path, required=True)
    ap.add_argument("--manual-evidence", type=Path)
    ap.add_argument("--chat-template-kwargs", help="JSON object; production runs should retain thinking=true and reasoning_effort=low")
    args = ap.parse_args()
    try:
        kwargs = strict_json_loads(args.chat_template_kwargs) if args.chat_template_kwargs else dict(DEFAULT_CHAT_KWARGS)
        if not isinstance(kwargs, dict):
            raise ValueError("chat-template-kwargs must be a JSON object")
        admission = AdmissionCoordinator.from_path(args.admission_config)
        summary = run_quality(base_url=args.base_url, run_id=args.run_id, dataset_path=args.set, admission=admission, model=args.model, max_tokens=args.max_tokens, timeout=args.timeout, workers=args.workers, human_eval_timeout=args.human_eval_timeout, image_id=args.image_id, profile_id=args.profile_id, candidate_id=args.candidate_id, chat_kwargs=kwargs, manual_evidence_path=args.manual_evidence)
    except (AdmissionError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"quality runner failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "SCORED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
