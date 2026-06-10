# Configuration

Barbet uses `BarbetConfig`, a standard Transformers `PretrainedConfig`.

## Presets

| Config | Layers | Hidden | FFN | Heads | KV Heads | Context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `barbet_300m` | 12 | 1024 | 2816 | 8 | 2 | 262144 |
| `barbet_1b` | 24 | 1536 | 5120 | 16 | 2 | 262144 |

Both shipped configs currently use:

- `vocab_size=114822`
- `head_dim=128`
- `rope_theta=10000000`
- `sliding_window_size=8192`
- `rms_norm_eps=1e-6`
- untied embeddings and LM head

The vocabulary size matches the tokenizer used by the active Barbet/Open
Formosa proxy runs. If the final tokenizer vocabulary changes, update both
config files and regenerate any converted checkpoints.

## Files

- `configs/barbet_300m/config.json`
- `configs/barbet_1b/config.json`

Each config contains:

```json
"auto_map": {
  "AutoConfig": "configuration_barbet.BarbetConfig",
  "AutoModel": "modeling_barbet.BarbetModel",
  "AutoModelForCausalLM": "modeling_barbet.BarbetForCausalLM"
}
```

This is required for Hugging Face remote-code loading.

## Regenerating Configs

```bash
PYTHONPATH=src python scripts/write_model_configs.py
```

This rewrites the config folders from the Python factory methods:

- `BarbetConfig.barbet_300m()`
- `BarbetConfig.barbet_1b()`

## Important Fields

`global_attention_layers`

Layers that use full causal attention.

`mamba_layers`

Layers that use the Mamba-style mixer.

`sliding_window_size`

Window size for all non-global, non-Mamba layers.

`rope_scaling`

Optional RoPE scaling metadata for research extensions. The default 300M and 1B
configs do not use scaling because they target 256K directly.

`mtp_enabled`, `mtp_offsets`, `mtp_loss_weights`

Controls auxiliary multi-token prediction training loss. Inference does not
require MTP outputs.
