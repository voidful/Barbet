#!/usr/bin/env python3
"""Regenerate Barbet config.json files."""

from __future__ import annotations

from pathlib import Path

from barbet import BarbetConfig


ROOT = Path(__file__).resolve().parents[1]


def write_config(config: BarbetConfig, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config.architectures = ["BarbetForCausalLM"]
    config.auto_map = {
        "AutoConfig": "configuration_barbet.BarbetConfig",
        "AutoModel": "modeling_barbet.BarbetModel",
        "AutoModelForCausalLM": "modeling_barbet.BarbetForCausalLM",
    }
    config.torch_dtype = "bfloat16"
    config.save_pretrained(path)


def main() -> None:
    write_config(BarbetConfig.barbet_300m(), ROOT / "configs" / "barbet_300m")
    write_config(BarbetConfig.barbet_1b(), ROOT / "configs" / "barbet_1b")


if __name__ == "__main__":
    main()
