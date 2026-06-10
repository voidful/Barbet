# Development

## Install

```bash
pip install -e ".[dev]"
```

## Tests

```bash
pytest -q
```

The tests cover:

- tiny config construction;
- forward pass;
- causal LM loss;
- 300M and 1B factory config validation.

## Smoke Test

```python
import torch
from barbet import BarbetConfig, BarbetForCausalLM

config = BarbetConfig(
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
)
model = BarbetForCausalLM(config)
ids = torch.randint(0, 128, (1, 8))
out = model(input_ids=ids, labels=ids)
assert out.logits.shape == (1, 8, 128)
```

## Style

Keep this repository independent from the training runtime:

- do not commit checkpoints;
- do not commit Slurm logs;
- do not commit Megatron work directories;
- keep Hugging Face remote-code files at repository root;
- keep package source under `src/barbet`.
