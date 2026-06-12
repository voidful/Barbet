# Architecture

Barbet is a decoder-only hybrid causal language model. The Hugging Face
implementation follows the R2 architecture used by the Open Formosa training
stack (Taiwan-Omni-300M-R2 / Taiwan-Omni-1B-R2) while keeping the code
self-contained for local loading, testing, and checkpoint packaging.

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

The final hidden state is normalized before the LM head. The LM head is tied
to the token embedding (R2 rebalance).

## Body vs Vocab Parameter Balance (R2)

R1 spent a disproportionate share of its parameter budget on the untied
embedding and LM head over the 114k vocabulary. R2 ties the two tables and
reinvests the savings into depth:

| Model | Layers | Tied | Total | Vocab params | Body params |
| --- | ---: | --- | ---: | ---: | ---: |
| 300M R1 | 12 | no | 384M | 235M (61%) | 149M (39%) |
| 300M R2 | 20 | yes | 365M | 118M (32%) | 247M (68%) |
| 1B R1 | 24 | no | 1138M | 352M (31%) | 786M (69%) |
| 1B R2 | 28 | yes | 1093M | 177M (16%) | 916M (84%) |

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

QK logit clipping and the learnable attention sink are part of the long-term
design but ship disabled in both R2 configs, matching the validated upstream
recipe (the corresponding Megatron/Transformer Engine features are
compatibility-gated).

Global attention layers use full causal attention. Sliding layers use the local
window configured by `sliding_window_size`.

## Layer Schedule

The layer type is determined by config:

- `global_attention_layers`
- `mamba_layers`
- all other layers are sliding-window attention

The block motif repeats every 4 layers: `global, sliding, sliding, mamba`.

Barbet 300M (5 repeats):

- layers: 20
- global attention: 0, 4, 8, 12, 16
- Mamba-style: 3, 7, 11, 15, 19

Barbet 1B (7 repeats):

- layers: 28
- global attention: 0, 4, 8, 12, 16, 20, 24
- Mamba-style: 3, 7, 11, 15, 19, 23, 27

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
