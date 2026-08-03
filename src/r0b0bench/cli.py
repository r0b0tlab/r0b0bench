from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from r0b0bench import OPTIONAL_LANES, PROFILES, SYSTEMS_LANES, __version__
from r0b0bench.config import ensure_outside_checkout, load_profile, write_json
from r0b0bench.endpoint import Endpoint
from r0b0bench.lanes.bfcl import run_bfcl_ast, run_bfcl_mt
from r0b0bench.lanes.canary import run_canary
from r0b0bench.lanes.concurrency import run_concurrency
from r0b0bench.lanes.latency import run_latency
from r0b0bench.lanes.niah import run_niah
from r0b0bench.lanes.perf import run_perf
from r0b0bench.lanes.quality import run_gsm8k, run_humaneval, run_ifeval, run_qa
from r0b0bench.lanes.throughput import run_throughput


def cmd_profiles(_: argparse.Namespace) -> int:
    for p in PROFILES:
        print(p)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"r0b0bench {__version__}")
    print(f"profiles: {', '.join(PROFILES)}")
    print(f"systems_lanes: {', '.join(SYSTEMS_LANES)}")
    ok = True
    if args.base_url and args.model:
        ep = Endpoint(args.base_url, args.model)
        try:
            h = ep.health()
            print("health:", json.dumps(h)[:300])
            if not h.get("ok"):
                ok = False
            m = ep.max_model_len()
            print("max_model_len:", m)
            kv = ep.kv_cache_size_tokens()
            print("kv_cache_size_tokens:", kv)
            models = ep.models()
            ids = [x.get("id") for x in models.get("data") or []]
            print("models:", ids)
            if args.model not in ids:
                print(f"WARNING: model {args.model!r} not in /v1/models list")
        except Exception as exc:  # noqa: BLE001
            print("endpoint error:", exc)
            ok = False
        finally:
            ep.close()
    try:
        import bfcl_eval  # noqa: F401

        print("bfcl_eval: available")
    except Exception:
        print("bfcl_eval: not installed (systems BFCL lanes need import or [bfcl] extra)")
    return 0 if ok else 2


def _expand_lanes(profile: dict, only: list[str] | None) -> list[str]:
    order: list[str] = []
    for item in profile.get("lane_order") or []:
        if item == "systems":
            order.extend(SYSTEMS_LANES)
        else:
            order.append(item)
    if only:
        allow = set(only)
        order = [x for x in order if x in allow]
        # allow optional lanes not in profile order
        for x in only:
            if x in OPTIONAL_LANES and x not in order:
                order.append(x)
    # de-dupe preserving order
    seen = set()
    out = []
    for x in order:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _run_lane(lane: str, ep: Endpoint, lane_dir: Path, systems_cfg: dict, profile: dict, tokenizer: str):
    if lane == "canary":
        return run_canary(ep, lane_dir, systems_cfg.get("canary"))
    if lane == "bfcl_mt":
        return run_bfcl_mt(ep, lane_dir, systems_cfg.get("bfcl_mt") or {})
    if lane == "bfcl_ast":
        return run_bfcl_ast(ep, lane_dir, systems_cfg.get("bfcl_ast") or {})
    if lane == "latency":
        return run_latency(ep, lane_dir, systems_cfg.get("latency") or {})
    if lane == "concurrency":
        return run_concurrency(ep, lane_dir, systems_cfg.get("concurrency") or {})
    if lane == "throughput":
        return run_throughput(ep, lane_dir, systems_cfg.get("throughput") or {})
    if lane == "perf":
        return run_perf(ep, lane_dir, systems_cfg.get("perf") or {})
    if lane == "niah":
        return run_niah(ep, lane_dir, systems_cfg.get("niah") or {}, tokenizer_path=tokenizer or None)
    if lane == "qa":
        q = (profile.get("quality") or {}).get("qa") or {}
        return run_qa(ep, lane_dir, q)
    if lane == "ifeval":
        q = (profile.get("quality") or {}).get("ifeval") or {}
        return run_ifeval(ep, lane_dir, q)
    if lane == "humaneval":
        q = (profile.get("quality") or {}).get("humaneval") or {}
        return run_humaneval(ep, lane_dir, q)
    if lane == "gsm8k":
        q = (profile.get("quality") or {}).get("gsm8k") or {}
        return run_gsm8k(ep, lane_dir, q)
    from r0b0bench.lanes.bfcl import run_quality_stub

    return run_quality_stub(lane, lane_dir, {"error": "unknown_lane"})


def cmd_run(args: argparse.Namespace) -> int:
    try:
        profile = load_profile(args.profile)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 4
    try:
        out_root = ensure_outside_checkout(Path(args.output))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 4

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    only = args.only.split(",") if args.only else None
    if only:
        only = [x.strip() for x in only if x.strip()]
    if args.skip_systems:
        only = only or []
        only = [x for x in (_expand_lanes(profile, None)) if x not in SYSTEMS_LANES] if not args.only else [
            x for x in only if x not in SYSTEMS_LANES
        ]
        invalid_for_publish = True
    else:
        invalid_for_publish = False

    lanes = _expand_lanes(profile, only)
    timeout = float(args.timeout) if args.timeout else 600.0
    ep = Endpoint(args.base_url, args.model, timeout=timeout)
    systems_cfg = profile.get("systems") or {}
    results = []
    t0 = time.perf_counter()
    try:
        for lane in lanes:
            lane_dir = run_dir / "lanes" / lane
            print(f"=== lane {lane} ===", flush=True)
            res = _run_lane(lane, ep, lane_dir, systems_cfg, profile, args.tokenizer)
            results.append(res.model_dump())
            write_json(lane_dir / "lane_result.json", res.model_dump())
            print(json.dumps({"lane": lane, "status": res.status, "infra_errors": res.infra_errors}), flush=True)
            if res.infra_errors and lane == "canary":
                print("canary infra failure — stopping", flush=True)
                break
    finally:
        ep.close()

    report = {
        "schema_version": 2,
        "r0b0bench_version": __version__,
        "run_id": run_id,
        "profile": profile.get("profile"),
        "base_url": args.base_url,
        "model": args.model,
        "systems_lanes": list(SYSTEMS_LANES),
        "invalid_for_publish": invalid_for_publish or bool(args.skip_systems),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.perf_counter() - t0,
        "lanes": results,
        "infra_errors_total": sum(int(r.get("infra_errors") or 0) for r in results),
    }
    write_json(run_dir / "report.json", report)
    lines = [
        f"# r0b0bench report `{run_id}`",
        "",
        f"- profile: **{report['profile']}**",
        f"- model: `{args.model}` @ `{args.base_url}`",
        f"- invalid_for_publish: {report['invalid_for_publish']}",
        f"- infra_errors_total: {report['infra_errors_total']}",
        "",
        "| Lane | Status | Infra |",
        "|------|--------|------:|",
    ]
    for r in results:
        lines.append(f"| {r['lane_id']} | {r['status']} | {r.get('infra_errors', 0)} |")
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {run_dir / 'report.json'}")
    if report["infra_errors_total"]:
        return 2
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.run_dir) / "report.json"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 4
    print(path.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="r0b0bench", description="r0b0bench endpoint benchmark client")
    p.add_argument("--version", action="version", version=f"r0b0bench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("profiles", help="list profiles")
    sp.set_defaults(func=cmd_profiles)

    sd = sub.add_parser("doctor", help="environment and endpoint checks")
    sd.add_argument("--base-url", default="")
    sd.add_argument("--model", default="")
    sd.set_defaults(func=cmd_doctor)

    sr = sub.add_parser("run", help="run a profile")
    sr.add_argument("--profile", required=True, choices=list(PROFILES))
    sr.add_argument("--base-url", required=True)
    sr.add_argument("--model", required=True)
    sr.add_argument("--output", required=True, help="output root OUTSIDE the git checkout")
    sr.add_argument("--tokenizer", default="", help="local tokenizer/model path for NIAH")
    sr.add_argument(
        "--only",
        default="",
        help="comma-separated lane filter (canary,bfcl_mt,bfcl_ast,latency,concurrency,throughput,niah[,perf])",
    )
    sr.add_argument("--skip-systems", action="store_true", help="debug only; marks report invalid")
    sr.add_argument("--run-id", default="")
    sr.add_argument("--timeout", type=float, default=600.0, help="default HTTP timeout seconds per request")
    sr.set_defaults(func=cmd_run)

    srep = sub.add_parser("report", help="print report.json")
    srep.add_argument("--run-dir", required=True)
    srep.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
