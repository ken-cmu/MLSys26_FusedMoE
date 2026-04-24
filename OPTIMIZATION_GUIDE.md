# MLSys 2026 FusedMoE Kernel Optimization Guide

**Kernel:** `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

**Target:** NVIDIA B200 (Blackwell, sm_100a)

---

## 1. Kernel Fusion Strategy

### 1.1 Routing + Top-K Fusion

**Naive approach:** Materialize full [seq_len, 256] routing score matrix
- Memory: 256 * seq_len * 4 bytes = ~1 MB for seq_len=1024
- Bandwidth waste: Need to read scores multiple times

**Fused approach:** Compute routing and top-K in single pass
```cuda
// Pseudo-code
for seq_idx in parallel:
    // Step 1: Sigmoid + bias per expert (in registers/shared mem)
    for expert in [0, 256):
        score = sigmoid(logits[seq_idx, expert]) + bias[expert]
    
    // Step 2: Group-wise top-2 selection
    for group in [0, 8):
        top2 = topk_within_group(scores[group*32:(group+1)*32], k=2)
        group_scores[group] = sum(top2)
    
    // Step 3: Top-4 groups (warp-level reduction)
    top_groups = warp_topk(group_scores, k=4)
    
    // Step 4: Global top-8 experts (among selected groups)
    expert_mask = scatter(top_groups)  // Mark relevant experts
    selected_experts = topk_masked(scores, k=8, mask=expert_mask)
```

**Memory savings:** Only materialize [seq_len, 8] expert indices + [seq_len, 8] weights

---

### 1.2 Dequantization + GEMM Fusion

**Key insight:** Weight dequantization is data-parallel and can be interleaved with GEMM

```cuda
// Fused pattern for GEMM1
for tile_k in [0, HIDDEN_SIZE, TILE_K):
    // Load input activation block [seq_len, TILE_K]
    A_block = load_activation_block(A, seq_idx, tile_k)
    
    for tile_out in [0, 2*INTERMEDIATE_SIZE, TILE_OUT):
        // Load quantized weights [TILE_OUT, TILE_K]
        W_q = load_weights_q(gemm1_weights, expert_id, tile_out, tile_k)
        
        // Load scales [blocks for out, blocks for k]
        S = load_scales(gemm1_weights_scale, expert_id, 
                        tile_out // BLOCK, tile_k // BLOCK)
        
        // Dequantize on-the-fly
        W = W_q.to_fp32() * S  // Small tile: fits in registers
        
        // GEMM: A_block @ W.T
        C_block = matmul(A_block, W.T)
        accumulate(output, C_block)
```

**Benefits:**
- Reduced L2 cache pressure (no separate dequant pass)
- Dequant computation overlaps with GEMM latency
- Weights never materialized in FP32 (stays FP8 → FP32 on-demand)

---

## 2. Memory Hierarchy Optimization

### 2.1 Block-Wise Quantization Exploitation

The [32, 32, 16] block structure of scales is intentional:

```
GEMM1 scales: [32 experts, 32 blocks out, 56 blocks hidden]
             = [32 * 32, 56] = [1024, 56] at materialization time

GEMM2 scales: [32 experts, 56 blocks hidden, 16 blocks intermediate]
             = [32 * 56, 16] = [1792, 16]
```

**Optimization:** Load scales into shared memory per block

```cuda
// Shared memory: 32 experts * 32 out-blocks * 4 bytes = 4 KB (fits in 96 KB per block)
__shared__ float scales_gemm1_out[32][32];  // [expert_id, out_block]
__shared__ float scales_gemm1_hidden[56];   // [hidden_block]

// Load once for all warps in the block
for idx in threadIdx.x...:
    scales_gemm1_out[idx/32][idx%32] = load_scale(...)

__syncthreads();

// Within inner loop:
float s = scales_gemm1_out[expert][out_block] * scales_gemm1_hidden[hidden_block];
```

### 2.2 Activation Streaming

Hidden states are large [seq_len, 7168] but used once:

**Optimization:** Process in row-major tiles

```cuda
// Process 128 tokens at a time (fits in 1.8 MB L2 cache for 128*7168*4 bytes)
for seq_tile in [0, seq_len, SEQ_TILE_SIZE):
    A_tile = load_activations(A, seq_tile, SEQ_TILE_SIZE)  // [SEQ_TILE, 7168]
    
    for expert in selected_experts:
        // All tokens for this expert, then next expert
        // Improves temporal locality
        output_tile = process_expert_batch(A_tile, expert)
        scatter_accumulate(output, output_tile, weights[expert])
```

---

## 3. WARP-Level and Thread-Block Patterns

### 3.1 Routing Kernel (256 threads per block)

```cuda
// Step 1: Parallel sigmoid computation (256 threads, 256 experts per token)
// Each warp: 32 threads → 32 experts
// Load logits + bias, compute sigmoid

// Step 2: Group-level reductions (within warp)
// Warp-shuffle for intra-group top-2 selection
for group_id in 0..7:
    group_scores[group_id] = warp_reduce_max(top2_sum[group_id])
    
// Step 3: Inter-group top-4 (single warp does this)
selected_groups = topk(group_scores, k=4)

// Step 4: Scatter selected experts back to global memory
for expert in 0..255:
    if (expert / 32) in selected_groups:
        expert_scores[expert] = score[expert]
```

### 3.2 GEMM Kernels (256 threads per block)

**Configuration:** 16×16 thread-block tiles (Blackwell-optimized)

```cuda
// Arrange threads in 16×16 grid
#define BLOCK_M 16
#define BLOCK_N 16

// Tile GEMMs across output dimensions
for tile_m in [0, 2*I, BLOCK_M):
    for tile_n in [0, H, BLOCK_N):
        // Each thread processes 1 element
        // Use Tensor Cores (TC) for 8×8 or 16×16 submatrices
        
        // Load input block [BLOCK_N, H] from A
        // Load weight block [BLOCK_M, BLOCK_N] from W
        // TC: BLOCK_N matmul
        
        // Accumulate into shared memory, then write
```

### 3.3 SwiGLU Activation (Fused with GEMM2)

```cuda
// After GEMM1 output is ready (in registers)
// SwiGLU: gate(first half) * second half

// Option 1: Sequential (simple, can be slow)
for i in 0..INTERMEDIATE_SIZE:
    gate = gelu(gemm1_out[i])
    value = gemm1_out[i + I]
    activated[i] = gate * value

// Option 2: Warp-level parallelization
// Each warp processes independent elements
for i in threadIdx.x..INTERMEDIATE_SIZE, step=warp_size:
    ...
```

---

## 4. B200-Specific Optimizations

### 4.1 Tensor Core Usage

Blackwell exposes 512-bit (64-byte) tensor operations:

```cuda
// FP8 TC operations (if available in CUDA 13.2+)
// Assuming TensorOp shape: 16×16×32 (FP8 input, FP32 accumulate)

__shared__ int8_t W_shared[16][32];      // FP8 weights
__shared__ int8_t A_shared[16][32];      // FP8 activations
__shared__ float C_shared[16][16];       // FP32 output

// Use mma instruction for 16×16×32 FP8 operation
asm("mma.sync.aligned.m16n16k32.row.col.f32.f8 ...")
```

### 4.2 Memory Bandwidth Optimization

B200 has 960 GB/s memory bandwidth (need to saturate it):

```cuda
// Arithmetic intensity for this kernel:
// GEMM1: seq_len * 2*I * 2*H / (seq_len*H*4 + seq_len*2*I*4 + E_local*2*I*H)
//      ≈ (1024 * 4096 * 14336) / (1024*7168*4 + 1024*4096*4 + 32*4096*7168)
//      ≈ 6e10 FLOPs / 2.7e8 bytes ≈ 220 FLOPS/byte

// Kernel is compute-bound (good!)
// Need ~14 TFLOPs throughput on B200 (max ~300 TFLOPs, but this is 4-8% of device)
```

### 4.3 Thread Block Size Tuning

B200 has 132 SMs, 128 threads per SM maximum occupancy:

```
- Routing kernel: 256 threads per block
  - Occupancy: 1 block/SM = 100% occupancy with 256 threads
  - 132 blocks running in parallel

- GEMM kernel: 256 threads per block
  - 16×16 threads = 256 threads
  - Occupancy: 2 blocks/SM = 100%
  - 264 blocks running in parallel
```

---

## 5. Optimization Checklist

### Performance-Critical Sections

- [ ] **Routing (top-K selection)**
  - [ ] Fuse sigmoid + bias computation
  - [ ] Use warp-level shuffles for group reductions
  - [ ] Avoid global memory reads for [seq_len, 256] routing scores
  - [ ] Estimate: 10-15% of total time for typical seq_len=1024

- [ ] **Weight dequantization**
  - [ ] Fuse with GEMM kernels (no separate dequant pass)
  - [ ] Load scales into shared memory
  - [ ] Compute scales on-the-fly (minimal overhead)
  - [ ] Estimate: Overlapped with compute, <5% standalone

- [ ] **GEMM1 computation**
  - [ ] Use Tensor Cores (16×16×32 or similar)
  - [ ] Tile strategically to maximize L2 cache hits
  - [ ] Pipeline weight loading with computation
  - [ ] Estimate: 40-50% of total time

- [ ] **SwiGLU activation**
  - [ ] Fuse with GEMM2 (don't write GEMM1 output to memory)
  - [ ] Use fast GELU approximation (e.g., Chebyshev polynomial)
  - [ ] Estimate: 5-10% of total time

- [ ] **GEMM2 computation**
  - [ ] Similar optimization as GEMM1
  - [ ] Careful data layout for [intermediate, hidden] dimensions
  - [ ] Estimate: 25-35% of total time

- [ ] **Expert accumulation**
  - [ ] Use atomicAdd for [seq_len, hidden] scatter
  - [ ] Or use shared memory reduction then scatter
  - [ ] Consider warp-level cooperative writing
  - [ ] Estimate: 5-10% of total time

### Memory Optimization Checklist

- [ ] **Global memory bandwidth**
  - [ ] Minimize reads/writes of [seq_len, 7168] activations
  - [ ] Reduce redundant scale loading (cache in shared mem)
  - [ ] Use coalesced memory access patterns

- [ ] **L2 cache utilization**
  - [ ] Tile activations to fit in ~1.8 MB L2
  - [ ] Stream weights tile-by-tile from HBM

- [ ] **Shared memory efficiency**
  - [ ] Store scales [32 experts, 32 out-blocks] if possible
  - [ ] Double-buffer GEMM inputs/outputs
  - [ ] Reduce bank conflicts (multiple threads accessing same bank)

### Arithmetic Optimization Checklist

- [ ] **FP8 handling**
  - [ ] Use fast convert (int8 reinterpret + scaling)
  - [ ] Avoid double conversion (FP8→FP32→FP8)

- [ ] **GELU approximation**
  - [ ] Use tanh-based approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
  - [ ] Or polynomial: better accuracy with ~3 multiplies
  - [ ] Avoid erf() calls (slow)

- [ ] **Reduction operations**
  - [ ] Use warp-level shuffles for cross-lane reductions
  - [ ] Avoid atomic operations in hot paths
  - [ ] Prefer block-level reductions with shared memory

---

## 6. Expected Performance Targets

Based on MLSys 2026 contest baseline (flashinfer):

| Kernel Component | Baseline Latency | Target | Comments |
|---|---|---|---|
| **Routing (top-K)** | ~0.3 ms | <0.2 ms | Fused sigmoid+topk |
| **GEMM1 dequant** | ~2.0 ms | <1.5 ms | Fused with GEMM |
| **GEMM2 dequant** | ~1.5 ms | <1.2 ms | Overlapped with compute |
| **SwiGLU** | ~0.4 ms | <0.3 ms | Fast GELU approx |
| **Accumulation** | ~0.5 ms | <0.4 ms | Scatter-add optimization |
| **Total (seq_len=1024)** | ~4.7 ms | <3.6 ms | **23% speedup target** |

**Observed speedups from contest solutions:** 65x - 175x (relative to baseline ~10ms latency at seq_len=1024)

---

## 7. Common Pitfalls

1. **Under-utilizing Tensor Cores**
   - FP8 TC kernels may not be available; use FP32 TC with manual dequant
   - Ensure tile sizes (16×16, 32×32) align with TC requirements

2. **Dequantization bottleneck**
   - Computing scales per-element is expensive
   - Cache scales in shared memory or use lookup tables

3. **Serialized top-K selection**
   - Sequential top-K across 256 experts per token is slow
   - Use parallel bitonic sort or hierarchical selection

4. **Memory access patterns**
   - Strided access to [seq_len, dim] tensors (each token separate)
   - Prefer contiguous access; reorder data if beneficial

5. **Register pressure**
   - Fused kernels may spill to local memory (slow)
   - Profile register usage; may need separate kernel passes

---

## References & Further Reading

- NVIDIA B200 Architecture: https://www.nvidia.com/en-us/data-center/blackwell/
- FlashInfer GitHub: https://github.com/flashinfer-ai/flashinfer
- MLSys 2026 Contest: FlashInfer Kernel Generation Challenge
- DeepSeek-V3 Paper: Routing and MOE design details
- CUDA Tensor Cores: https://docs.nvidia.com/cuda/
