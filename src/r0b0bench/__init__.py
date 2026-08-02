"""r0b0bench — reproducible OpenAI-compatible endpoint benchmarks."""

__version__ = "1.0.0rc2"

PROFILES = ("core", "core-subset", "systems")

# Standard systems package (expanded from profile lane_order entry "systems")
SYSTEMS_LANES = (
    "canary",
    "bfcl_mt",
    "bfcl_ast",
    "latency",
    "concurrency",
    "throughput",
    "niah",
)

# Optional composite kept for --only perf
OPTIONAL_LANES = ("perf",)

QUALITY_LANES_CORE = ("qa", "ifeval", "humaneval", "gsm8k")
QUALITY_LANES_SUBSET = ("qa", "ifeval", "humaneval", "gsm8k")
