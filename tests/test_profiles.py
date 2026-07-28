from r0b0bench.config import load_profile, wilson_ci
from r0b0bench import PROFILES


def test_profiles_only_two():
    assert PROFILES == ("core", "core-subset")
    for name in PROFILES:
        p = load_profile(name)
        assert p["includes_systems"] is True or p.get("profile") in ("core", "core-subset")
        assert "systems" in p
        assert p["systems"]["lanes"] == ["canary", "bfcl_mt", "bfcl_ast", "perf", "niah"]


def test_wilson():
    ci = wilson_ci(50, 100)
    assert ci is not None
    assert ci["low"] < 0.5 < ci["high"]
