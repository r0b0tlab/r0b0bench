"""r0b0bench — reproducible OpenAI-compatible endpoint benchmarks."""

__version__ = "1.0.0rc1"

PROFILES = ("core", "core-subset")
SYSTEMS_LANES = ("canary", "bfcl_mt", "bfcl_ast", "perf", "niah")
QUALITY_LANES_CORE = ("qa", "ifeval", "humaneval", "gsm8k")
QUALITY_LANES_SUBSET = ("qa", "ifeval", "humaneval", "gsm8k")
