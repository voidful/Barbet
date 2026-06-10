# Long Context

Barbet targets a 256K product context before any 1M extension.

## 256K Base

The 300M and 1B configs both set:

```json
"max_position_embeddings": 262144
```

The current training curriculum is:

```text
32K -> 64K -> 128K -> 256K
```

Sliding attention layers keep a local window of 8192 tokens. Global attention
layers remain full causal attention.

## 1M Extension

The intended 1M path is not normal pretraining. It is a guarded research
extension from a validated 256K base.

The planned direction is:

```text
256K base -> external sparse/compressed-memory expansion -> 1M research model
```

A native full-attention 1M run is not a default config because global attention
layers would still be expensive and could create misleading systems results.

## RoPE Scaling

For a 1M research extension from 256K, linear RoPE scaling uses:

```json
"rope_scaling": {
  "type": "linear",
  "factor": 4.0,
  "original_context_length": 262144
}
```

This metadata is supported by `BarbetConfig`, but the shipped 300M and 1B base
configs do not enable scaling.
