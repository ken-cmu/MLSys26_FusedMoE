# Triton Kernel Changelog

## v1 — Basic Triton FP8 GEMM (current)

**Strategy:** Keep routing and accumulation in PyTorch; replace the two GEMMs with custom Triton kernels that fuse FP8 block-scale dequantization directly into the matrix multiply.

**Routing** is identical to the Python reference — sigmoid gating, group-based top-K selection, softmax-normalized weights — all in PyTorch float32. No change here yet.

**GEMM1** (`_fp8_fp8_gemm`): computes `dequant(hidden) @ dequant(W1).T → [Tk, 4096]` in a single Triton kernel. Tiles are 64×128 over (tokens, output-features), iterating over K in 128-element chunks that align exactly with the FP8 block-scale granularity. A-scales are per-token-per-k-block `[56, Tk]`; B-scales are per-block `[32, 56]`. Everything accumulates in float32.

**SwiGLU** is still PyTorch (`F.silu(x2) * x1`), applied to the float32 GEMM1 output.

**GEMM2** (`_f32_fp8_gemm`): computes `float32_intermediate @ dequant(W2).T → [Tk, 7168]`. Same tiling strategy; only B needs FP8 dequantization since A is already float32 after SwiGLU.

**Accumulation** uses PyTorch `index_add_` into a float32 buffer, then copies to the bfloat16 output at the end.

**What's better than the Python reference:**
- The GEMMs run as GPU-parallel Triton kernels rather than sequential PyTorch matmuls over float32-expanded weights — avoids materializing the full dequantized `[E, 2I, H]` weight tensors in memory.

**Known limitations / next steps:**
- Routing is still pure PyTorch with sequential Python overhead.
- The expert loop is sequential in Python (32 iterations), each launching separate Triton kernels. A grouped/batched GEMM across all experts at once would be significantly faster.
- SwiGLU could be fused into the GEMM1 epilogue to save a read-write roundtrip.
- GEMM2 input is float32; quantizing it to FP8 before the matmul (like the FlashInfer baseline does) would halve memory bandwidth.
- No software pipelining or persistent kernel yet.
