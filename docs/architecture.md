# Architecture

Barbet is a decoder-only hybrid causal language model. The Hugging Face
implementation follows the architecture used by the Open Formosa training
experiments while keeping the code self-contained for local loading, testing,
and checkpoint packaging.

## Stack

Each decoder layer contains:

1. RMSNorm before the mixer.
2. A mixer block:
   - global grouped-query attention,
   - sliding-window grouped-query attention, or
   - Mamba-style sequence mixer.
3. Residual connection.
4. RMSNorm before the MLP.
5. SwiGLU MLP.
6. Residual connection.

The final hidden state is normalized before the untied LM head.

## Attention

Attention layers use:

- grouped-query attention;
- separate query, key, value, and output projections;
- QK RMSNorm when enabled by config;
- RoPE with `rope_theta=10000000`;
- optional QK logit clipping;
- optional attention sink normalization;
- causal masking;
- optional local sliding-window masking.

Global attention layers use full causal attention. Sliding layers use the local
window configured by `sliding_window_size`.

## Layer Schedule

The layer type is determined by config:

- `global_attention_layers`
- `mamba_layers`
- all other layers are sliding-window attention

Barbet 300M:

- layers: 12
- global attention: 0, 4, 8
- Mamba-style: 3, 7, 11

Barbet 1B:

- layers: 24
- global attention: 0, 4, 8, 12, 16, 20
- Mamba-style: 3, 7, 11, 15, 19, 23

## Mamba-Style Mixer

The current HF implementation includes a deterministic PyTorch fallback mixer.
It is shape-compatible with the model stack and useful for:

- remote-code loading;
- CPU smoke tests;
- checkpoint packaging tests;
- reference inference.

Production Megatron checkpoints may use kernel-backed Mamba modules. Those
weights require explicit conversion into the HF module layout.

## Multi-Token Prediction

`BarbetForCausalLM` optionally computes auxiliary next-token losses at configured
future offsets. The default offsets are:

- `t+2` with weight `0.2`
- `t+3` with weight `0.1`

The auxiliary heads reuse the LM head weight through functional projection, so
the model can be saved with `save_pretrained` without duplicate shared tensors.

## Generation

The implementation inherits `GenerationMixin` and supports standard
Transformers generation calls. KV cache is not implemented yet; `use_cache` is
accepted for API compatibility but the current forward path recomputes the full
sequence.
