# Qwen3.5-0.8B on a single Ascend chip: eager text adapter

`tokenspeed.runtime.models.qwen3_5_npu` is an experimental batch-1 text
runner. It reuses Transformers 5.17.0's decoder/rotary embedding and
caller-owned `DynamicCache`, strictly loads all text checkpoint parameters,
and replaces GDN, convolution, zero-centered RMSNorm, and full attention with
Ascend implementations. Vision and MTP checkpoint parameters are excluded;
the complete text model, including its tied LM head, is evaluated.

The kernel boundary remains `tokenspeed_kernel`; its explicit Ascend facade
imports the vendor package. GDN kernels use TokenSpeed's K-last state layout
`[B,H,V,K]`; the adapter converts to/from Transformers' `[B,H,K,V]` cache.
No tensor computation is moved to the CPU during model inference.

This runner does not implement the TokenSpeed serving scheduler, paged GDN
state allocation, batching, padding, prefix reuse, retraction, or ACL Graph.
Do not attach its request cache to scheduler slots. Integrating it with the
serving runtime requires the shared LCM cache ownership/export contracts in
`docs/design/cache-concepts.md` and `docs/design/unified_path.md`.

## Validated environment

- Ascend 910C host, CANN 9.0.0, one logical chip visible.
- Python 3.11.16, PyTorch 2.9.0+cpu with torch_npu 2.9.0.
- Transformers 5.17.0, safetensors 0.8.0, Triton-Ascend 3.2.0.
- BF16 projections, FP32 normalization/decay/recurrent state.

The existing nanoGPT conda environment can supply the packages through an
isolated venv:

```bash
cd /path/to/AI_compute_homework
/opt/miniconda3/envs/aics/bin/python -m venv --system-site-packages .venv
```

The wrapper sources CANN and makes only physical chip 2 visible by default;
set `HW1_CHIP` to select another available chip. It maps to `npu:0` inside
the process. Full package versions are recorded by `experiment.py`.

```bash
bash hw1/scripts/run_env.sh hw1/scripts/smoke.py \
  --model-dir hw1/models/Qwen3.5-0.8B

bash hw1/scripts/run_env.sh tokenspeed/tokenspeed-kernel-npu/test/test_gdn.py \
  --output hw1/results/operators.json

bash hw1/scripts/run_env.sh hw1/scripts/experiment.py \
  --model-dir hw1/models/Qwen3.5-0.8B --output hw1/results \
  --mode all --repeats 3
```

Model revision: `2fc06364715b967f1860aea9cf38778875588b17`. The checkpoint
SHA256 is `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
Metadata and tokenizer must come from the same pinned Hugging Face revision.
The identical weight file can be downloaded from ModelScope if Hugging Face
is inaccessible; verify SHA256 before using it.

## Numerical contracts

The convolution stencil includes every tap and updates the last four raw
inputs. GDN prefill uses a chunk-64 triangular inverse built by stable
recursive block substitution, followed by exact inter-chunk state propagation. Decode uses
the same delta recurrence one token at a time. Padding has zero beta/decay
and cannot change the final state. The recursive inverse includes every delta update and avoids the floating
point cancellation of a global matrix power series on correlated keys.

CANN 9.0.0 rejects head dimension 256 in TND inference attention; the NPU
MHA prefill wrapper converts each packed request to BNSD. Dense cached
attention also uses BNSD and an offset causal mask for multi-token extension.

Tests compare against unmodified Transformers CPU references, including
nonzero states, chunk boundaries, and packed attention requests. The model
experiment additionally checks teacher-forced logits, cached vs fresh-prefix
inference, every recurrent/conv cache, and two interleaved requests.
Latency measurements synchronize at every measured boundary and exclude
loading, tokenization, and warmup; output counts are forced for comparability.
