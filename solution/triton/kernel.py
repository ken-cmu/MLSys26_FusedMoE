"""
Basic Triton implementation of the FP8 Fused MoE kernel.

Design:
  - Routing : pure PyTorch (same algorithm as the Python reference)
  - GEMM1   : Triton kernel — dequant(FP8 hidden) @ dequant(FP8 W1).T
  - SwiGLU  : PyTorch F.silu
  - GEMM2   : Triton kernel — F32 intermediate @ dequant(FP8 W2).T
  - Accum   : PyTorch index_add_

Both Triton kernels process a tile of BLOCK_M tokens × BLOCK_N output-features
at a time, iterating over K in chunks of BLOCK_K=128 (== FP8 quant-block size).
This alignment lets us load exactly one A-scale per (token, k-block) and one
B-scale per (out-block, k-block) without any splitting or interpolation.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Model constants
# ─────────────────────────────────────────────────────────────────────────────
H          = 7168   # hidden size
I          = 2048   # intermediate size (GEMM1 output = 2*I = 4096)
E_GLOBAL   = 256    # total experts
E_LOCAL    = 32     # local experts on this rank
TOP_K      = 8      # experts selected per token
N_GROUP    = 8      # number of routing groups
TOPK_GROUP = 4      # top routing groups to keep
QUANT_BLOCK = 128   # FP8 block-scale granularity

# Triton tile sizes — BLOCK_K must equal QUANT_BLOCK so k-block index == kb
BLOCK_M = 64
BLOCK_N = 128   # must equal QUANT_BLOCK for B-scale indexing to work cleanly
BLOCK_K = 128   # must equal QUANT_BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 1: dequant(FP8 A) @ dequant(FP8 B).T  →  float32 C
#
# Tensor layouts
#   A      : [M, K]          fp8_e4m3fn   (gathered hidden states)
#   Scale_A: [K//128, M]     float32      (per k-block, per token)
#   B      : [N, K]          fp8_e4m3fn   (weight matrix W1, stored as [2I, H])
#   Scale_B: [N//128, K//128] float32     (per out-block, per k-block)
#   C      : [M, N]          float32
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fp8_fp8_gemm(
    A_ptr,  SA_ptr,
    B_ptr,  SB_ptr,
    C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_sA_kb, stride_sA_m,
    stride_bn, stride_bk,
    stride_sB_nb, stride_sB_kb,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M] row offsets
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N] col offsets

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K is always a multiple of 128 (H=7168=56*128), so no tail
    for kb in range(K // BLOCK_K):
        rk = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        # ── A tile: [BLOCK_M, BLOCK_K] fp8 ──────────────────────────────
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        # Per-token, per-k-block scale: Scale_A[kb, tok]
        sa = tl.load(
            SA_ptr + kb * stride_sA_kb + rm * stride_sA_m,
            mask=rm < M, other=1.0,
        )                                           # [BLOCK_M]
        a = a.to(tl.float32) * sa[:, None]         # [BLOCK_M, BLOCK_K]

        # ── B tile: [BLOCK_N, BLOCK_K] fp8 ──────────────────────────────
        b = tl.load(
            B_ptr + rn[:, None] * stride_bn + rk[None, :] * stride_bk,
            mask=(rn[:, None] < N) & (rk[None, :] < K),
            other=0.0,
        )
        # One scale per (out-block, k-block): Scale_B[pid_n, kb]
        # Valid because BLOCK_N == QUANT_BLOCK, so pid_n == n_block index
        sb = tl.load(SB_ptr + pid_n * stride_sB_nb + kb * stride_sB_kb)
        b = b.to(tl.float32) * sb                  # [BLOCK_N, BLOCK_K]

        # C += A @ B.T
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 2: float32 A @ dequant(FP8 B).T  →  float32 C
#
# Tensor layouts
#   A      : [M, K]          float32      (SwiGLU output, intermediate)
#   B      : [N, K]          fp8_e4m3fn   (weight matrix W2, stored as [H, I])
#   Scale_B: [N//128, K//128] float32
#   C      : [M, N]          float32
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _f32_fp8_gemm(
    A_ptr,
    B_ptr,  SB_ptr,
    C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sB_nb, stride_sB_kb,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K always a multiple of 128 (I=2048=16*128)
    for kb in range(K // BLOCK_K):
        rk = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        ).to(tl.float32)                            # [BLOCK_M, BLOCK_K]

        b = tl.load(
            B_ptr + rn[:, None] * stride_bn + rk[None, :] * stride_bk,
            mask=(rn[:, None] < N) & (rk[None, :] < K),
            other=0.0,
        )
        sb = tl.load(SB_ptr + pid_n * stride_sB_nb + kb * stride_sB_kb)
        b = b.to(tl.float32) * sb                  # [BLOCK_N, BLOCK_K]

        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Python launchers for the two Triton kernels
# ─────────────────────────────────────────────────────────────────────────────

def _gemm1(
    x:        torch.Tensor,   # [Tk, H]  fp8_e4m3fn
    x_scale:  torch.Tensor,   # [H//128, Tk]  float32  — Scale_A
    w:        torch.Tensor,   # [2I, H]  fp8_e4m3fn   — W1 for one expert
    w_scale:  torch.Tensor,   # [2I//128, H//128]  float32
) -> torch.Tensor:            # [Tk, 2I]  float32
    Tk, K = x.shape
    N = w.shape[0]            # 2I = 4096
    out = torch.empty((Tk, N), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(Tk, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fp8_fp8_gemm[grid](
        x,       x_scale,
        w,       w_scale,
        out,
        Tk, N, K,
        x.stride(0),       x.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        w.stride(0),       w.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0),     out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def _gemm2(
    z:        torch.Tensor,   # [Tk, I]  float32  — SwiGLU output
    w:        torch.Tensor,   # [H, I]   fp8_e4m3fn  — W2 for one expert
    w_scale:  torch.Tensor,   # [H//128, I//128]  float32
) -> torch.Tensor:            # [Tk, H]  float32
    Tk, K = z.shape
    N = w.shape[0]            # H = 7168
    out = torch.empty((Tk, N), dtype=torch.float32, device=z.device)

    grid = (triton.cdiv(Tk, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _f32_fp8_gemm[grid](
        z,
        w,       w_scale,
        out,
        Tk, N, K,
        z.stride(0),       z.stride(1),
        w.stride(0),       w.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0),     out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Routing (pure PyTorch — identical algorithm to the Python reference)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_routing(
    routing_logits:       torch.Tensor,   # [T, 256] float32
    routing_bias:         torch.Tensor,   # [256] bfloat16
    routed_scaling_factor: float,
):
    """Returns topk_idx [T, 8] and per-expert weights [T, 256]."""
    T = routing_logits.shape[0]

    s = torch.sigmoid(routing_logits)                               # [T, 256]
    s_bias = s + routing_bias.to(torch.float32)                     # [T, 256]

    # Group scoring: for each group pick top-2, sum → group score
    s_grouped = s_bias.view(T, N_GROUP, E_GLOBAL // N_GROUP)        # [T, 8, 32]
    top2, _   = torch.topk(s_grouped, k=2, dim=2, sorted=False)
    group_scores = top2.sum(dim=2)                                  # [T, 8]

    # Keep top-TOPK_GROUP groups, mask out the rest
    _, grp_idx  = torch.topk(group_scores, k=TOPK_GROUP, dim=1, sorted=False)
    grp_mask    = torch.zeros_like(group_scores)
    grp_mask.scatter_(1, grp_idx, 1.0)
    score_mask  = (
        grp_mask.unsqueeze(2)
        .expand(T, N_GROUP, E_GLOBAL // N_GROUP)
        .reshape(T, E_GLOBAL)
    )                                                               # [T, 256]

    # Global top-K within kept groups
    masked = s_bias.masked_fill(score_mask == 0, float("-inf"))
    _, topk_idx = torch.topk(masked, k=TOP_K, dim=1, sorted=False) # [T, 8]

    # Routing weights: normalize s (without bias) over selected experts
    weight_mask = torch.zeros_like(s)
    weight_mask.scatter_(1, topk_idx, 1.0)
    weights     = s * weight_mask
    weights     = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-20)
    weights     = weights * routed_scaling_factor                   # [T, 256]

    return topk_idx, weights


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  (destination-passing style: output is the last argument)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def kernel(
    routing_logits:        torch.Tensor,   # float32        [T, 256]
    routing_bias:          torch.Tensor,   # bfloat16       [256]
    hidden_states:         torch.Tensor,   # fp8_e4m3fn     [T, 7168]
    hidden_states_scale:   torch.Tensor,   # float32        [56, T]
    gemm1_weights:         torch.Tensor,   # fp8_e4m3fn     [32, 4096, 7168]
    gemm1_weights_scale:   torch.Tensor,   # float32        [32, 32, 56]
    gemm2_weights:         torch.Tensor,   # fp8_e4m3fn     [32, 7168, 2048]
    gemm2_weights_scale:   torch.Tensor,   # float32        [32, 56, 16]
    local_expert_offset:   int,
    routed_scaling_factor: float,
    output:                torch.Tensor,   # bfloat16       [T, 7168]  (DPS)
):
    T      = hidden_states.shape[0]
    device = hidden_states.device

    # ── 1. Routing ────────────────────────────────────────────────────────
    topk_idx, weights = _compute_routing(
        routing_logits, routing_bias, routed_scaling_factor
    )
    # topk_idx : [T, 8]   global expert indices selected per token
    # weights  : [T, 256] normalized routing weights (0 for unselected)

    # ── 2. Float32 accumulation buffer ───────────────────────────────────
    out_f32 = torch.zeros((T, H), dtype=torch.float32, device=device)

    # ── 3. Per-local-expert loop ──────────────────────────────────────────
    local_start = int(local_expert_offset)
    for le in range(E_LOCAL):
        ge = local_start + le
        if ge >= E_GLOBAL:
            continue

        # Tokens that routed to this global expert
        sel     = (topk_idx == ge).any(dim=1)                 # [T] bool
        if not sel.any():
            continue
        tok_idx = sel.nonzero(as_tuple=False).squeeze(1)      # [Tk]

        # ── Gather ───────────────────────────────────────────────────
        x       = hidden_states[tok_idx].contiguous()         # [Tk, H] fp8
        x_scale = hidden_states_scale[:, tok_idx].contiguous()# [56, Tk]

        w1      = gemm1_weights[le]                           # [4096, H] fp8
        w1_sc   = gemm1_weights_scale[le]                     # [32, 56]

        w2      = gemm2_weights[le]                           # [H, 2048] fp8
        w2_sc   = gemm2_weights_scale[le]                     # [56, 16]

        # ── GEMM1 + SwiGLU ───────────────────────────────────────────
        g1     = _gemm1(x, x_scale, w1, w1_sc)               # [Tk, 4096] f32
        x1, x2 = g1[:, :I], g1[:, I:]                        # each [Tk, 2048]
        z      = F.silu(x2) * x1                              # [Tk, 2048]  (matches reference)

        # ── GEMM2 ────────────────────────────────────────────────────
        o = _gemm2(z, w2, w2_sc)                              # [Tk, H] f32

        # ── Weighted accumulate ───────────────────────────────────────
        w_tok = weights[tok_idx, ge].unsqueeze(1)             # [Tk, 1]
        out_f32.index_add_(0, tok_idx, o * w_tok)

    # ── 4. Write result into pre-allocated bfloat16 output ───────────────
    output.copy_(out_f32)
