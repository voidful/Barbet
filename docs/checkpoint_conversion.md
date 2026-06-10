# Checkpoint Conversion

This repository provides the Hugging Face model class. It does not yet include
the Megatron-to-HF conversion script.

## Why Conversion Is Needed

The training stack uses Megatron HybridModel with a mixed attention/Mamba layer
schedule. Its checkpoint layout differs from the Hugging Face module layout in
this repository.

Conversion must map:

- token embeddings;
- final RMSNorm;
- LM head;
- attention query/key/value/output projections;
- QK RMSNorm parameters;
- SwiGLU gate/up/down projections;
- Mamba-style mixer parameters;
- optional MTP projection parameters.

## Expected Output

A converted Hugging Face checkpoint should contain:

```text
config.json
configuration_barbet.py
modeling_barbet.py
model.safetensors
```

It should load with:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "path/to/converted-barbet",
    trust_remote_code=True,
)
```

## Conversion Gates

A conversion script should verify:

- all expected HF keys are present;
- no unexpected Megatron shards are silently ignored;
- embedding and LM head shapes match `vocab_size`;
- 300M and 1B configs produce the expected parameter shapes;
- logits from a tiny deterministic fixture match before and after conversion
  where a reference path is available;
- `save_pretrained` and `from_pretrained` both work;
- `generate()` smoke test runs.

## Current Status

The HF implementation has been smoke-tested for:

- construction;
- forward pass;
- training loss;
- `save_pretrained`;
- remote-code `AutoConfig`;
- remote-code `AutoModelForCausalLM`;
- `generate()`.

The production Megatron checkpoint conversion script remains a separate task.
