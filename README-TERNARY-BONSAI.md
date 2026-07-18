# BeeLlama + PrismML Ternary Bonsai (Q2_0)

A fork of [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) that loads
**[PrismML Ternary-Bonsai-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)**
(`Q2_0`, 2.125 bits/weight) so the model can use BeeLlama's **TurboQuant / KVarN KV-cache
compression** — which the upstream PrismML fork does not have.

**Why bother:** on a 16 GB card, Bonsai at 192K context with `q8_0` KV leaves ~0.6 GB of
headroom. With `turbo3` KV it leaves ~5.4 GB, and **384K context fits more comfortably than
192K used to**.

---

## ⚠️ Work in progress: BeeLlama's own `Q2_0` (type 53) is currently broken

This fork **replaces** BeeLlama's native `Q2_0` instead of coexisting with it. Both formats
are called "Q2_0" but they are **not compatible**:

| | block size | bytes/block | bit layout |
|---|---:|---:|---|
| BeeLlama native `Q2_0` | 32 | 10 | planar — byte `j` holds elements `j, j+8, j+16, j+24` |
| PrismML Bonsai `Q2_0` | 128 | 34 | sequential — element `i` at byte `i/4`, bits `(i%4)*2` |

**If you have a model in BeeLlama's native `Q2_0` format, do not use this build yet** — it
will produce garbage, silently, with no error. Every other quantization type is untouched
and works normally.

Splitting these into two coexisting types (`Q2_0` = 42 ternary, `Q2_0_B` = 53 native) is in
progress and will land as a follow-up commit and a second release.

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

- **DFlash / speculative decoding for Bonsai.** Drafters are target-specific — each one only
  accelerates the exact model it was trained against. The only drafter that exists for
  Ternary Bonsai is PrismML's own DSpark head, and Bonsai has no built-in MTP tensors
  (`n_layer == n_layer_all == 64`). Heads trained for stock Qwen3.6 will not work.
- BeeLlama's native `Q2_0` — see the warning above.

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
