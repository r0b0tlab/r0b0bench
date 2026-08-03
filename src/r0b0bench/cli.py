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
    full_lanes = _expand_lanes(profile, None)
    if args.skip_systems:
        only = only or []
        only = [x for x in (_expand_lanes(profile, None)) if x not in SYSTEMS_LANES] if not args.only else [
            x for x in only if x not in SYSTEMS_LANES
        ]
        invalid_for_publish = True
    else:
        invalid_for_publish = bool(only and set(_expand_lanes(profile, only)) != set(full_lanes))

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

    lane_statuses_complete = bool(results) and len(results) == len(lanes) and all(
        r.get("status") == "PASS" and not int(r.get("infra_errors") or 0) for r in results
    )
    report = {
        "schema_version": 2,
        "r0b0bench_version": __version__,
        "run_id": run_id,
        "profile": profile.get("profile"),
        "base_url": args.base_url,
        "model": args.model,
        "systems_lanes": list(SYSTEMS_LANES),
        "invalid_for_publish": invalid_for_publish or bool(args.skip_systems) or not lane_statuses_complete,
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


def cmd_results(args: argparse.Namespace) -> int:
    from r0b0bench import results_ledger as rl

    action = args.results_cmd
    if action == "list":
        rows = rl.list_entries()
        if not rows:
            print("(no entries)")
            return 0
        for e in rows:
            m = (e.get("model") or {}).get("display_name") or (e.get("model") or {}).get("id")
            print(f"{e.get('entry_id')}\t{m}\t{(e.get('harness') or {}).get('profile')}\tinvalid={e.get('invalid_for_publish')}")
        return 0
    if action == "show":
        e = rl.show_entry(args.entry_id)
        print(json.dumps(e, indent=2))
        return 0
    if action == "compare":
        print(rl.compare_entries(args.entry_a, args.entry_b), end="")
        return 0
    if action == "rebuild-index":
        idx, board = rl.rebuild_index()
        print(f"index: {idx}")
        print(f"leaderboard: {board}")
        return 0
    if action == "add":
        report_path = Path(args.report)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entry = rl.entry_from_report(
            report,
            entry_id=args.entry_id,
            model_display=args.model_display or None,
            hardware=args.hardware or None,
            notes=args.notes or None,
        )
        if args.runtime_json:
            entry["runtime"] = {**(entry.get("runtime") or {}), **json.loads(Path(args.runtime_json).read_text())}
        path = rl.write_entry(entry, force=args.force)
        rl.rebuild_index()
        print(f"wrote {path}")
        return 0
    print(f"unknown results action {action}", file=sys.stderr)
    return 2


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
        help="comma-separated lane filter (canary,bfcl_mt,bfcl_ast,latency,concurrency,throughput,niah[,perf,qa,...])",
    )
    sr.add_argument("--skip-systems", action="store_true", help="debug only; marks report invalid")
    sr.add_argument("--run-id", default="")
    sr.add_argument("--timeout", type=float, default=600.0, help="default HTTP timeout seconds per request")
    sr.set_defaults(func=cmd_run)

    srep = sub.add_parser("report", help="print report.json")
    srep.add_argument("--run-dir", required=True)
    srep.set_defaults(func=cmd_report)

    sres = sub.add_parser("results", help="record / list / compare package results")
    sres_sub = sres.add_subparsers(dest="results_cmd", required=True)

    sres_list = sres_sub.add_parser("list", help="list recorded entries")
    sres_list.set_defaults(func=cmd_results)

    sres_show = sres_sub.add_parser("show", help="print one entry JSON")
    sres_show.add_argument("entry_id")
    sres_show.set_defaults(func=cmd_results)

    sres_cmp = sres_sub.add_parser("compare", help="compare two entry_ids")
    sres_cmp.add_argument("entry_a")
    sres_cmp.add_argument("entry_b")
    sres_cmp.set_defaults(func=cmd_results)

    sres_rb = sres_sub.add_parser("rebuild-index", help="rebuild index.json + LEADERBOARD.md")
    sres_rb.set_defaults(func=cmd_results)

    sres_add = sres_sub.add_parser("add", help="import a package report.json into results/entries")
    sres_add.add_argument("--report", required=True, help="path to report.json")
    sres_add.add_argument("--entry-id", required=True)
    sres_add.add_argument("--model-display", default="")
    sres_add.add_argument("--hardware", default="")
    sres_add.add_argument("--notes", default="")
    sres_add.add_argument("--runtime-json", default="", help="optional JSON merged into runtime")
    sres_add.add_argument("--force", action="store_true")
    sres_add.set_defaults(func=cmd_results)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
