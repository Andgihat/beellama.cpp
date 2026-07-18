# BeeLlama + PrismML Ternary Bonsai (Q2_0)

A fork of [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) that loads
**[PrismML Ternary-Bonsai-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)**
(`Q2_0`, 2.125 bits/weight) so the model can use BeeLlama's **TurboQuant / KVarN KV-cache
compression** — which the upstream PrismML fork does not have.

**Why bother:** on a 16 GB card, Bonsai at 192K context with `q8_0` KV leaves ~0.6 GB of
headroom. With `turbo3` KV it leaves ~5.4 GB, and **384K context fits more comfortably than
192K used to**.

**Base:** BeeLlama `preview-v0.3.2` (commit [`a620cbd`](https://github.com/Anbeeld/beellama.cpp/commit/a620cbd481d16d3ccb5c8c96fa2fbd70191bea38), 17 Jun 2026).
Releases here are tagged `v0.3.2-ternary-bonsai[.N]`: the version names the upstream base it
was built from, the suffix counts revisions of this fork. Upstream version numbers belong to
Anbeeld and are not claimed here.

---

## Two 2-bit types, side by side

Both projects ship a type called "Q2_0" and they are **not compatible**:

| | on-disk id | block size | bytes/block | bit layout |
|---|---:|---:|---:|---|
| BeeLlama native `Q2_0` | 53 | 32 | 10 | planar — byte `j` holds elements `j, j+8, j+16, j+24` |
| PrismML Bonsai `Q2_0` | 42 | 128 | 34 | sequential — element `i` at byte `i/4`, bits `(i%4)*2` |

Since `.2` they **coexist**, so this is a drop-in replacement for upstream BeeLlama:

* BeeLlama's own type keeps id 53 **and its user-visible name `q2_0`** — existing models and
  existing command lines (`-ctk q2_0`, `-ctk kvarn2`, …) behave exactly as before.
* The ternary type takes id 42 and is displayed as `q2_0_t`.

The two have different capabilities, which follows from the formats themselves:

| | weights | KV cache |
|---|---|---|
| `q2_0` (53) | yes | yes |
| `q2_0_t` (42) | yes | **no** |

The ternary format is produced offline by PrismML's quantizer and has no runtime encoder, so
it cannot back a KV cache. Asking for a 2-bit KV cache gives you the native planar type, as
it always did.

**Note on the first release (`v0.3.2-ternary-bonsai`):** it replaced type 53 rather than coexisting with it,
which silently broke both native `Q2_0` models and 2-bit KV cache options. If you are on
that release, upgrade.

---

## Verified working

Tested on **RTX 5060 Ti 16 GB** (`sm_120`, CUDA 12.8, driver 610.62), Windows.

**Numerically identical to the reference implementation.** Perplexity on the same corpus,
same model file, same settings (`-c 4096 -fa on --chunks 12`):

| build | PPL |
|---|---|
| upstream PrismML (reference) | `3.1125 ± 0.04339` |
| **this fork** | **`3.1125 ± 0.04339`** |

Bit-for-bit the same result — the ported dequant, `vec_dot` and MMQ paths are not merely
"close enough", they are exact. This matters because a wrong bit layout produces plausible
garbage with no error at all; matching perplexity is the only way to rule that out.

| Metric | Upstream PrismML | This fork | |
|---|---:|---:|---|
| `pp512` (prefill) | 956.9 ± 28.8 | **1079.5 ± 35.8** | +13% |
| `tg128` (generation) | 44.3 ± 0.2 | **51.6 ± 0.1** | +16% |

VRAM at 192K context: `q8_0` → 15677 MiB, **`turbo3` → 10939 MiB** (≈2.9× smaller KV).

Maximum context with `turbo3` on 16 GB:

| context | VRAM | headroom |
|---:|---:|---:|
| 262144 | 11867 MiB | 4443 MiB |
| **393216** | 13595 MiB | **2715 MiB** |

---

## Working configurations

Model: `Ternary-Bonsai-27B-Q2_0.gguf` (6.66 GiB, the `g128` file — **not** `Q2_g64`).

**Sampling matters.** Bonsai is a thinking model; use the values from the model card
(`temp 0.7 / top-p 0.95 / top-k 20`). Greedy decoding (`--temp 0`) can make thinking models
loop and never close `</think>`.

### Long context — 384K, TurboQuant KV

```bash
llama-server -m Ternary-Bonsai-27B-Q2_0.gguf \
  --ctx-size 393216 --n-gpu-layers 99 --flash-attn on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --temp 0.7 --top-p 0.95 --top-k 20 --jinja
```

### Maximum speed — short context, `q8_0` KV

`turbo3` trades ~7% generation speed for memory (measured at 32K real context: `q8_0`
51.3 t/s vs `turbo3` 47.9 t/s, prefill identical). If the context fits without compression,
don't compress:

```bash
llama-server -m Ternary-Bonsai-27B-Q2_0.gguf \
  --ctx-size 32768 --n-gpu-layers 99 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --temp 0.7 --top-p 0.95 --top-k 20 --jinja
```

Other KV types available: `turbo2/3/4`, `turbo2/3/4_tcq`, `kvarn2/3/4/5`.

### Speculative decoding — the stock Qwen3.6-27B DFlash head works

A DFlash drafter trained for **plain Qwen3.6-27B accelerates Ternary Bonsai as well**. The
head is tied to the architecture, not to the target's quantization: it carries no
`token_embd` / `output` tensors (those are shared from the target at runtime) and
cross-attends to the target's hidden states.

```
dflash: contract ok: block_size=16 target_layer_ids=[1,16,31,46,61] n_embd=5120
dflash: target_vocab=248320 drafter_vocab=248320 vocab_match=1
```

**Set `--spec-draft-n-max 3` explicitly.** DFlash's adaptive controller settles on `n_max=15`,
which is tuned for ordinary quants that accept ~30% of drafted tokens. Bonsai is a much
coarser approximation of the model the head was trained on, so acceptance falls and long
draft blocks are mostly wasted work:

| `--spec-draft-n-max` | decode | acceptance |
|---:|---:|---:|
| 2 | 62.5 t/s | 69.4% |
| **3** | **64.4 t/s** | **63.1%** |
| 4 | 60.3 t/s | 52.6% |
| 5 | 54.5 t/s | 42.3% |
| 6 | 52.9 t/s | 40.5% |
| 8 | 55.7 t/s | 34.4% |
| 10 | 55.5 t/s | 24.5% |
| 15 (adaptive default) | 51.3 t/s | 15.1% |

Baseline without a drafter: **47.4 t/s**. So `n_max=3` is **+36%**, while leaving the
controller to its own devices gives only +8%. Note that 3 is simply llama.cpp's own default
for the flag — the DFlash controller overrides it.

```bash
llama-server -m Ternary-Bonsai-27B-Q2_0.gguf \
  --spec-draft-model Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-type dflash --spec-draft-n-max 3 \
  --spec-dflash-cross-ctx 512 --spec-draft-ngl all \
  --ctx-size 32768 --n-gpu-layers 999 --flash-attn on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --temp 0.7 --top-p 0.95 --top-k 20 --jinja
```

Measured at 32K context with a short prompt and greedy decoding, on one card and one drafter.
Acceptance normally drops as context grows, which would push the optimum shorter still; that
was not measured. Between `n_max` 5 and 8 the ordering is within run-to-run noise (~3 t/s) —
the peak at 2-4 is not.

### Build

```bash
cmake -S . -B build -G "Ninja Multi-Config" \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_NATIVE=ON -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CCACHE=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build --config Release -j
```

---

## Not supported

- **MTP speculation for Bonsai.** The model has no built-in MTP tensors
  (`n_layer == n_layer_all == 64`). Use a DFlash drafter instead — see above.
- **2-bit KV cache for the ternary type.** It has no runtime encoder; `-ctk q2_0` uses the
  native planar type instead.

---

## What was changed

The root cause of earlier failed porting attempts was that `Q2_0` was declared as the
ternary type while `QK2_0` still held BeeLlama's value of 32. The loader then computed
tensor offsets in a ratio of exactly **20/17** — the ratio of the two geometries
(10 bytes/32 weights vs 34 bytes/128 weights) — and rejected the file.

Fixing the constant alone is not enough: the **bit layouts also differ** (planar vs
sequential), so a build with correct geometry but the original kernels would load the file
and silently produce garbage.

| File | Change |
|---|---|
| `ggml-common.h` | `QK2_0` 32→128, `QI2_0`→`QK2_0/32`, `QR2_0` 4→1 |
| `ggml-cuda/dequantize.cuh` | `dequantize_q2_0` — sequential layout |
| `ggml-cuda/vecdotq.cuh` | `vec_dot_q2_0_q8_1`, `VDR_Q2_0_Q8_1_MMQ` 2→4 |
| `ggml-cuda/mmq.cuh` | `load_tiles_q2_0` + traits (this is the prefill path) |
| `ggml-cpu/quants.c` | `quantize_row_q2_0`, `ggml_vec_dot_q2_0_q8_0_generic` |
| `ggml-quants.c` | `quantize_row_q2_0_ref`, `dequantize_row_q2_0` |

The follow-up release splits the two formats apart, giving BeeLlama's native type back its
own id (53), name (`q2_0`), CPU and CUDA kernels, KV-cache encoder and flash-attention
instances, while the ternary type keeps id 42 and the weight-only path above.

All implementations are taken verbatim from
[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp). The MMQ infrastructure of
the two forks turned out to be compatible, and both already shared the same
`vec_dot_q8_0_q8_1_mma/dp4a` path, so only the tile loader needed porting.

---

## Credits & license

MIT, same as upstream. This work builds on:

- [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) — TurboQuant / KVarN / DFlash
- [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) — the ternary `Q2_0` implementation
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — upstream

See also [jarkevithwlad/turboquant-prismml-cuda](https://github.com/jarkevithwlad/turboquant-prismml-cuda),
which solves the same problem on a different base (AmesianX/TurboQuant) and handles the type
conflict by remapping only for `qwen35` models.
