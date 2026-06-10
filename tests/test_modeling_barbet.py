from __future__ import annotations

import torch

from barbet import BarbetConfig, BarbetForCausalLM


def tiny_config() -> BarbetConfig:
    return BarbetConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window_size=16,
        global_attention_layers=[0],
        mamba_layers=[2],
        mtp_enabled=True,
    )


def test_tiny_forward_and_loss() -> None:
    model = BarbetForCausalLM(tiny_config())
    input_ids = torch.randint(0, 128, (2, 12))
    out = model(input_ids=input_ids, labels=input_ids)
    assert out.logits.shape == (2, 12, 128)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_factory_configs_validate() -> None:
    assert BarbetConfig.barbet_300m().num_hidden_layers == 12
    assert BarbetConfig.barbet_1b().num_hidden_layers == 24
