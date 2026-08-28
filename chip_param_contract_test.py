#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_compare():
    spec = importlib.util.spec_from_file_location("chip_param_compare", ROOT / "chip_param_compare.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    compare = load_compare()
    assert len(compare.CONFIGS) == 6
    assert compare.CONFIGS[0]["name"] == "winrate_1.5"
    assert compare.CONFIGS[-1]["name"] == "vol_0.9"
    assert compare.effective_thresholds({"CHIP_PROFILE": "balanced"}) == {"vol": 1.3, "conc": 0.2}
    assert compare.effective_thresholds({"CHIP_PROFILE": "balanced", "CHIP_VOL_MULTIPLIER": "1.0"})["vol"] == 1.0

    rows = [
        {"is_approaching": True, "volume_ratio": 1.4, "conc90_pct": 15, "trend": True, "is_tradeable": True},
        {"is_approaching": True, "volume_ratio": 0.8, "conc90_pct": 35, "trend": False, "is_tradeable": False},
    ]
    funnel = compare.build_funnel(rows, {"vol": 1.3, "conc": 0.2})
    assert funnel["total"] == 2
    assert funnel["in_zone"] == 2
    assert funnel["vol_ok_in_zone"] == 1
    assert funnel["conc_ok_in_zone"] == 1
    assert funnel["vol_and_conc_ok"] == 1
    assert funnel["trend_ok_among_vol_and_conc"] == 1
    assert funnel["final_tradeable"] == 1

    workflow = (ROOT / ".github/workflows/chip-param-comparison.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "shard_total" in workflow and "shard_index" in workflow
    assert "chip-param-comparison-${{ inputs.shard_index }}" in workflow
    assert "SENDKEY" not in workflow and "SERVERCHAN" not in workflow
    base = (ROOT / "chip_param_base_scan.py").read_text(encoding="utf-8")
    assert "SERVERCHAN_SENDKEY" not in base
    assert "SCT_KEY" not in base
    assert "PUSH_ON_SHARD = False" in base
    assert "volume_ratio" in base and "is_spike_tradeable" in base
    print("chip parameter comparison contract: PASS")


if __name__ == "__main__":
    main()
