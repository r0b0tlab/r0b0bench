# Campaign procedures

Frozen, reproducible protocols used alongside the four CLI profiles in
[`PROFILES.md`](PROFILES.md). Each section names the exact artifacts, the fail-closed rules,
and a reference result. Record the identity bindings (dataset revision, image id, profile id,
template kwargs) in every ledger entry.

---

## 1. Q200v2 — frozen quality kit (text180 + BFCL structural-hard20)

**What it is.** 180 frozen text-quality rows (`quality-text-180-v2`: gsm8k 80, humaneval 40,
ifeval 40, hard_reasoning 20) plus the official BFCL v4 `multi_turn_base` structural-hard20
subset (20 rows). Dataset sha256 `74623ab9b075120cd6f7a93059cc16d8817a6039dd20118b8f0350279f8b1ed6`.

**Identity.** Every row binds `dataset_sha256`, `run_identity_sha256` (image id + profile id +
candidate id + chat kwargs + max_tokens), `request_sha256`, `response_sha256`. The run summary
carries `grade_complete`, per-family `accuracy_pct`, and the response-budget audit
(`finish_reason_counts`, ceiling contacts).

**Chat kwargs are model-specific.** The kit defaults to Qwen-family kwargs; a model whose
template exposes only one thinking switch needs a thin adapter that overrides
`DEFAULT_CHAT_KWARGS`, replaces the thinking validator, and injects
`extra_body.chat_template_kwargs` for BFCL. Ling-3.0-flash-VL needs exactly
`{"enable_thinking": true}` — it has no `reasoning_effort` / `thinking_token_budget`.

**Sandbox image (humaneval grading).** `grade_exec_result` runs
`docker run … <image_id> python3 /opt/r0b0tlab/q200_sandbox_driver.py` under
`--memory=256m --memory-swap=256m` with **no** `--entrypoint`. The campaign image therefore
needs (a) the kit's driver baked at `/opt/r0b0tlab/q200_sandbox_driver.py` (mode 644) and
(b) `ENTRYPOINT []` — vLLM images set `ENTRYPOINT ["vllm"]`, so the args are eaten by the vLLM
CLI and the 256 MB cap turns that into **rc 137 (SIGKILL) with empty stdout** → every
humaneval row reports `HARNESS_BLOCK` / "Docker sandbox did not produce a clean exit".

**hard_reasoning is manual-review by design.** Rows ship `grade: manual` → `ungraded` until
`--manual-evidence <file>` is supplied. Schema is exact:
`{schema, dataset_sha256, run_identity_sha256, reviewer, method, rows}`, `method ==
"independent_manual_review"`, each row exactly `{id, content_sha256, passed, rationale}` bound
to the response bytes. Re-running with the same `--run-id` RESUMES (existing rows are
immutable) and re-grades everything, so the evidence is applied without regenerating
responses.

**Budget rule (fail closed).** The frozen contract accepts only `finish_reason == "stop"`.
A length-truncated row is transport-failed even when its grader passes — disclose it
(transported-correct / transported), never silently raise the cap.

**BFCL lane.** Serve `max_model_len ≥ prompt + BFCL_MAX_TOKENS` (32 K used here; a 16 K serve
400s with "maximum context length is 16384"). `resume` is disabled for claim-bearing hard20
runs → fresh `BFCL_PROJECT_ROOT` and timing sidecar per attempt. "Failed to decode the model
response" printed once per turn is the normal text-only end-of-turn path (count ≈ cases ×
turns), not a harness failure. Run the client off the serve node when other work owns it.

**Reference (Ling-3.0-flash-VL-NVFP4-MP, 2026-09-09).** text180 176/179 transported (98.3 %:
gsm8k 78/80, humaneval 40/40, ifeval 38/40, hard_reasoning 20/20 manual) + BFCL-hard20 17/20
(85.0 %) = **193/199 (97.0 %)**. One ifeval row hit the 8192 ceiling (disclosed).

---

## 2. NIAH — advertised-window ladder

**Depths.** 25 % / 50 % / 90 % of the advertised window **plus** multi-key 33/66 (three
needles at 33/66/90, answer = last). Target tokens = advertised window − 64. Client timeout
≥ 43,200 s, launched under `setsid`/tmux on the serve-side host.

**GB10 single-node wedge-safe serve (hard requirement).** A single 131,072-token prefill at
`--gpu-memory-utilization 0.85` with swap enabled **hard-wedged a GB10** (ICMP + TCP-22 accept
but sshd could not complete a banner exchange; both tailnet and LAN dead; recovery required a
power cycle). The same window passes with:

```bash
sudo swapoff -a && sudo sysctl -w vm.swappiness=0
# serve flags
--gpu-memory-utilization 0.78 --max-num-batched-tokens 4096 --max-num-seqs 1
```

and a **ladder** (32 K → 65 K → 131 K) that stops on the first failure so a wedge cannot be
compounded by queued work. ~120 K-token prompts then prefill in ~44 s.

**Reference (Ling-3.0-flash-VL-NVFP4-MP).** 15/15 across the ladder, 0 infra errors.

---

## 3. r0b0bench-vision v1.0

Frozen vision benchmark: 4 suites / 4,703 rows, deterministic graders, single-image requests,
thinking-off. Contract + runner + graders: [`scripts/vision/`](../scripts/vision/README.md)
(`benchmark.json` is hash-bound into every run summary).

| suite | rows | grading | license |
|---|---:|---|---|
| cvbench | 2,638 | MC letter | Apache-2.0 |
| mmvp | 300 (150 pairs) | 2-choice + paired metric | MIT |
| realworldqa | 765 | letter or normalized containment | CC-BY-ND-4.0 |
| ocrbench | 1,000 | official containment (HME100k strips spaces) | MIT |

```bash
python3 scripts/vision/run_vision.py --base-url http://127.0.0.1:8000 \
  --data-dir ~/rbv-data --out-dir ~/rbv-out/<run-id> --model <served-model> \
  --image-id sha256:<id> --profile-id <id> --candidate-id <label> --workers 4
```

**Reference (Ling-3.0-flash-VL-NVFP4-MP, 2026-09-09).** 3,801/4,703 = **80.82 %**
(Wilson 95 % 79.67–81.92 %): cvbench 78.92 % (Count 67.1 / Relation 84.5 / Depth 92.8 /
Distance 74.5), mmvp 82.0 % (paired 100/150 = 66.7 %), realworldqa 78.3 %, ocrbench 874/1,000.
Mean 1.85 s/row, 36 min wall at 4 workers.

---

## 4. End-to-end throughput + telemetry

**e2e throughput.** `e2e_tok_s = completion_tokens / elapsed_seconds` **per row**, reported as
mean, p50 and aggregate (Σ completion tokens / Σ elapsed). Never derive it from wrapper fields
that may be empty; never use content length as a token proxy. Report the client location
(same host as the serve vs remote) and the worker count.

**Telemetry.** Sample on the serve host: `nvidia-smi --query-gpu=power.draw,temperature.gpu,
utilization.gpu,clocks.current.graphics,clocks_throttle_reasons.active` plus
`/proc/meminfo` (MemAvailable/Swap*) at a 2 s cadence. Summarize **load-only** samples
(util > 0) separately from the idle baseline and publish mean + max — never a mixed-idle peak.
On GB10, `utilization.gpu` is effectively binary (0 or ~96); power is the better load
discriminator, so gate on both. Raw JSONL stays private; the sanitized summary is published.

**Reference (Q200v2 text180 lane).** mean 42.64 / p50 42.73 / aggregate 43.13 tok/s (n=180);
load power mean 38.05 W / max 42.17 W, temp mean 68.3 °C / max 75 °C, min MemAvailable
5.07 GiB.

**Clock hygiene.** A mid-run NTP jump (a node corrected ~5 h during a request) makes the
OpenAI SDK compute a bogus elapsed and raise `APITimeoutError`. Verify node clocks agree
(`date -u` on every participant) before a claim run, and note any correction in disclosures.
