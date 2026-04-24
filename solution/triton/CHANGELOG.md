# Triton Kernel Changelog

## v4 — Fused routing kernel (current)

**Optimization:** Replace 6+ sequential PyTorch routing ops with a single Triton kernel (`_routing_kernel`) — one GPU program per token.

**What changed:**
- Added `_routing_kernel`: each CTA handles one token and executes the full DeepSeek-V3 routing pipeline in registers:
  1. `sigmoid(logits) + bias` → biased scores [256]
  2. Reshape to [8, 32] groups, sort each group descending, sum top-2 → group scores [8]
  3. Sort group scores, take threshold at rank 4 → group selection mask [8]
  4. Broadcast mask to [256], zero out non-selected experts
  5. `argsort` → take top-8 expert indices
  6. Gather unbiased sigmoid values for selected experts via [8, 256] broadcast comparison
  7. Normalize and scale weights
- Routing output is now compact **[T, 8]** (int32 indices + float32 weights) instead of sparse [T, 256].
  For T=1024: 32 KB vs 1 MB — eliminates 32× larger intermediate tensor.
- Per-expert weight extraction: `(topk_w[tok_idx] * (topk_idx[tok_idx] == ge)).sum(1)` replaces `weights[tok_idx, ge]`.

**GEMM kernels and SwiGLU are unchanged from v3 (still autotuned).**

---

## v3 — Autotuned GEMM tile sizes (superseded by v4)

**Optimization:** Autotuning over BLOCK_M, num_warps, and num_stages for both GEMM kernels.

**What changed:** Added `@triton.autotune` to both `_fp8_fp8_gemm` and `_f32_fp8_gemm`. A search space of 16 configs is explored per distinct (M, N, K) shape:
- `BLOCK_M` ∈ {16, 32, 64, 128}
- `num_warps` ∈ {4, 8}
- `num_stages` ∈ {3, 4}

`BLOCK_N=128` and `BLOCK_K=128` are held fixed to keep FP8 block-scale indexing exact. The grid uses `lambda meta:` to read the autotuned `BLOCK_M` at launch time.

**Algorithm is identical to v1** — FP8 GEMM kernels + SwiGLU in PyTorch + Python routing. This is a pure tile-size optimization.

**Results (19 workloads, all PASSED):**
- Latency — min: 2.20 ms, max: 17.80 ms, median: 7.01 ms
- Speedup — min: 1.807x, max: 4.945x, **mean: 2.486x**
- Improvement over v1: +0.10x mean speedup (2.385x → 2.486x), median latency 7.60 ms → 7.01 ms

---

## v2 — Fuse SwiGLU into GEMM1 epilogue (superseded — regressed vs v1)

**Optimization:** §3.3 / checklist item "SwiGLU: Fuse with GEMM2 (don't write GEMM1 output to memory)".

**What changed:** Replaced `_fp8_fp8_gemm` + `F.silu` + `F.mul` with a single new kernel `_fp8_fp8_gemm_swiglu`. Each CTA now holds **two** accumulators (`acc_gate`, `acc_up`) and processes both the gate half and the up half of W1 in the same K-loop. At the end of the loop, the SwiGLU is applied in-register before the result is stored:

```
z = silu(acc_up) * acc_gate   →   write [Tk, I] float32
```

**What this eliminates:** In v1, GEMM1 wrote `[Tk, 4096]` float32 to HBM, then PyTorch read it back to apply SwiGLU, then wrote `[Tk, 2048]` float32 as GEMM2's input. v2 skips that intermediate `[Tk, 4096]` write+read — for Tk=1024 that's ~16 MB saved per expert per forward pass.

**Tile size adjustment:** `BLOCK_M` for GEMM1 was reduced from 64 → 32 because each CTA now carries two accumulators (doubling register pressure). `BLOCK_M` for GEMM2 stays at 64 (single accumulator, unchanged).

**Everything else is identical to v1.**

---

## v1 — Basic Triton FP8 GEMM (superseded by v3)

**Strategy:** Keep routing and accumulation in PyTorch; replace the two GEMMs with custom Triton kernels that fuse FP8 block-scale dequantization directly into the matrix multiply.

**Routing** is identical to the Python reference — sigmoid gating, group-based top-K selection, softmax-normalized weights — all in PyTorch float32. No change here yet.

**GEMM1** (`_fp8_fp8_gemm`): computes `dequant(hidden) @ dequant(W1).T → [Tk, 4096]` in a single Triton kernel. Tiles are 64×128 over (tokens, output-features), iterating over K in 128-element chunks that align exactly with the FP8 block-scale granularity. A-scales are per-token-per-k-block `[56, Tk]`; B-scales are per-block `[32, 56]`. Everything accumulates in float32.

**SwiGLU** is still PyTorch (`F.silu(x2) * x1`), applied to the float32 GEMM1 output.

**GEMM2** (`_f32_fp8_gemm`): computes `float32_intermediate @ dequant(W2).T → [Tk, 7168]`. Same tiling strategy; only B needs FP8 dequantization since A is already float32 after SwiGLU.

**Accumulation** uses PyTorch `index_add_` into a float32 buffer, then copies to the bfloat16 output at the end.

**What's better than the Python reference:**
- The GEMMs run as GPU-parallel Triton kernels rather than sequential PyTorch matmuls over float32-expanded weights — avoids materializing the full dequantized `[E, 2I, H]` weight tensors in memory.

**Results (19 workloads, all PASSED):**
- Latency — min: 2.25 ms, max: 19.59 ms, median: 7.60 ms
- Speedup — min: 1.752x, max: 4.877x, **mean: 2.385x**

**Known limitations / next steps:**
- Routing is still pure PyTorch with sequential Python overhead.
- The expert loop is sequential in Python (32 iterations), each launching separate Triton kernels. A grouped/batched GEMM across all experts at once would be significantly faster.
- SwiGLU could be fused into the GEMM1 epilogue to save a read-write roundtrip.
- GEMM2 input is float32; quantizing it to FP8 before the matmul (like the FlashInfer baseline does) would halve memory bandwidth.
- No software pipelining or persistent kernel yet.
