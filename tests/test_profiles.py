from r0b0bench.config import load_profile, wilson_ci
from r0b0bench import PROFILES, SYSTEMS_LANES


def test_profiles_include_public_rc2_profiles():
    assert PROFILES == ("core", "core-subset", "systems", "hard-subset")
    for name in PROFILES:
        p = load_profile(name)
        if p.get("profile") in ("core", "core-subset"):
            assert p.get("includes_systems") is True
        else:
            assert p.get("includes_systems", False) is False
        assert "systems" in p
        assert p["systems"]["lanes"] == list(SYSTEMS_LANES)


def test_wilson():
    ci = wilson_ci(50, 100)
    assert ci is not None
    assert ci["low"] < 0.5 < ci["high"]
