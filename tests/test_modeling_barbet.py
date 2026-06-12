from __future__ import annotations

import torch

from barbet import BarbetConfig, BarbetForCausalLM


def tiny_config(tie_word_embeddings: bool = True) -> BarbetConfig:
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
        tie_word_embeddings=tie_word_embeddings,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=0,
    )


def test_tiny_forward_and_loss() -> None:
    model = BarbetForCausalLM(tiny_config())
    input_ids = torch.randint(0, 128, (2, 12))
    out = model(input_ids=input_ids, labels=input_ids)
    assert out.logits.shape == (2, 12, 128)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_tied_embeddings() -> None:
    model = BarbetForCausalLM(tiny_config(tie_word_embeddings=True))
    assert model.lm_head.weight is model.model.embed_tokens.weight

    untied = BarbetForCausalLM(tiny_config(tie_word_embeddings=False))
    assert untied.lm_head.weight is not untied.model.embed_tokens.weight


def test_save_load_roundtrip_preserves_tie_and_logits(tmp_path) -> None:
    model = BarbetForCausalLM(tiny_config(tie_word_embeddings=True)).eval()
    input_ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        reference = model(input_ids=input_ids).logits
    model.save_pretrained(tmp_path)
    reloaded = BarbetForCausalLM.from_pretrained(tmp_path).eval()
    assert reloaded.lm_head.weight is reloaded.model.embed_tokens.weight
    with torch.no_grad():
        restored = reloaded(input_ids=input_ids).logits
    assert torch.allclose(reference, restored, atol=1e-6)


def test_factory_configs_validate() -> None:
    config_300m = BarbetConfig.barbet_300m()
    assert config_300m.num_hidden_layers == 20
    assert config_300m.global_attention_layers == [0, 4, 8, 12, 16]
    assert config_300m.mamba_layers == [3, 7, 11, 15, 19]
    assert config_300m.max_position_embeddings == 8192
    assert config_300m.sliding_window_size == 2048

    config_1b = BarbetConfig.barbet_1b()
    assert config_1b.num_hidden_layers == 28
    assert config_1b.global_attention_layers == [0, 4, 8, 12, 16, 20, 24]
    assert config_1b.mamba_layers == [3, 7, 11, 15, 19, 23, 27]
    assert config_1b.max_position_embeddings == 262144
    assert config_1b.sliding_window_size == 8192

    for config in (config_300m, config_1b):
        assert config.vocab_size == 114944
        assert config.tie_word_embeddings is True
        assert config.qk_logit_clip is False
        assert config.attention_sink is False
        assert config.unk_token_id == 114688
        assert config.bos_token_id == 114689
        assert config.eos_token_id == 114690
        assert config.pad_token_id == 114691


def test_1m_extension_rope_scaling() -> None:
    config = BarbetConfig.barbet_1b_1m_extension()
    assert config.max_position_embeddings == 1048576
    assert config.rope_scaling == {
        "type": "linear",
        "factor": 4.0,
        "original_context_length": 262144,
    }
