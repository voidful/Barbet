# Barbet

Barbet is a Hugging Face Transformers implementation of the Barbet causal
language model family. The repository provides remote-code compatible modeling
classes and two production-oriented configuration presets: Barbet 300M and
Barbet 1B.

This repository is intentionally lightweight. It contains model code and config
metadata, not training checkpoints or Megatron runtime artifacts.

## Contents

- `BarbetConfig`
- `BarbetModel`
- `BarbetForCausalLM`
- `configs/barbet_300m/config.json`
- `configs/barbet_1b/config.json`
- remote-code files for Hugging Face Hub loading:
  - `configuration_barbet.py`
  - `modeling_barbet.py`

## Model Summary

Barbet is a decoder-only hybrid language model with:

- grouped-query attention
- QK RMSNorm
- RoPE with large-context theta
- scheduled global attention layers
- local sliding-window attention layers
- scheduled Mamba-style mixer layers
- SwiGLU feed-forward layers
- untied token embeddings and LM head
- optional multi-token prediction loss for training

The 300M config is the current proxy model family used for systems validation.
The 1B config is the target family configuration.

## Quick Start

```bash
pip install -e ".[dev]"
pytest -q
```

```python
from barbet import BarbetConfig, BarbetForCausalLM

config = BarbetConfig.barbet_300m()
model = BarbetForCausalLM(config)
```

## Hugging Face Loading

After a config folder and the remote-code files are uploaded to a Hugging Face
model repository, the model can be loaded with:

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("voidful/Barbet-300M", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("voidful/Barbet-300M", trust_remote_code=True)
```

The config files under `configs/` already include the `auto_map` fields required
for remote-code loading.

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Transformers Usage](docs/transformers_usage.md)
- [Checkpoint Conversion](docs/checkpoint_conversion.md)
- [Long Context](docs/long_context.md)
- [Development](docs/development.md)

## Current Limitations

- The Mamba-style mixer is a deterministic PyTorch fallback so the HF model is
  self-contained. Kernel-backed Mamba can be wired later behind the same module
  interface.
- Megatron HybridModel checkpoints require a dedicated conversion script before
  they can be loaded by this Hugging Face implementation.
- Native 1M-context training is not a default config. The intended path is to
  validate a 256K base first, then use a verified external long-context
  expansion method.
