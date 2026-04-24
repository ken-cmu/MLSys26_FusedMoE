"""
Triton FP8 Fused MoE kernel — v2.

Design:
  - Routing : pure PyTorch (same algorithm as the Python reference)
  - GEMM1   : Triton kernel — dequant(FP8 hidden) @ dequant(FP8 W1).T
              with SwiGLU fused into the epilogue (no intermediate [Tk,4096] write)
  - GEMM2   : Triton kernel — F32 intermediate @ dequant(FP8 W2).T
  - Accum   : PyTorch index_add_

Key constraint: BLOCK_K = BLOCK_N = 128 (== FP8 quant-block size) so that each
Triton tile column maps to exactly one B-scale entry and each K-iteration maps to
exactly one A-scale entry.
"""

import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Model constants
# ─────────────────────────────────────────────────────────────────────────────
H          = 7168
I          = 2048   # intermediate size; GEMM1 produces 2*I = 4096 cols
E_GLOBAL   = 256
E_LOCAL    = 32
TOP_K      = 8
N_GROUP    = 8
TOPK_GROUP = 4
QUANT_BLOCK = 128

# Tile sizes shared by both kernels
BLOCK_N = 128   # must equal QUANT_BLOCK for clean B-scale indexing
BLOCK_K = 128   # must equal QUANT_BLOCK for clean A-scale indexing

# GEMM1+SwiGLU uses a smaller BLOCK_M because each CTA holds two accumulators
# (gate + up), which doubles register pressure.
BLOCK_M_GEMM1 = 32
# GEMM2 has a single accumulator so we can use a larger tile.
BLOCK_M_GEMM2 = 64


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 1: dequant(FP8 A) @ dequant(FP8 B).T  with SwiGLU epilogue
#
# W1 layout: [2*I, H] — first I rows are "gate", last I rows are "up".
# Each CTA (pid_m, pid_n) computes one tile of the final [Tk, I] output:
#   - acc_gate accumulates  A @ B_gate.T   (B rows pid_n*BN .. (pid_n+1)*BN - 1)
#   - acc_up   accumulates  A @ B_up.T     (B rows (NI+pid_n)*BN .. (NI+pid_n+1)*BN - 1)
#   - epilogue: z = silu(acc_up) * acc_gate   written to C
#
# This eliminates the global-memory round-trip of the [Tk, 2*I] GEMM1 output.
#
# Tensor layouts
#   A      : [M, K]               fp8_e4m3fn
#   Scale_A: [K//128, M]          float32     (per k-block, per token)
#   B      : [2*I, K]             fp8_e4m3fn  (W1 for one expert)
#   Scale_B: [2*I//128, K//128]   float32
#   C      : [M, I]               float32     (already SwiGLU-activated)
#   NI     : I // BLOCK_N  (= 16, number of gate tiles)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fp8_fp8_gemm_swiglu(
    A_ptr, SA_ptr,
    B_ptr, SB_ptr,
    C_ptr,
    M, K,
    NI,
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
    pid_n = tl.program_id(1)   # 0 .. NI-1

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # B row offsets for gate and up tiles
    rn_gate = pid_n         * BLOCK_N + tl.arange(0, BLOCK_N)
    rn_up   = (NI + pid_n)  * BLOCK_N + tl.arange(0, BLOCK_N)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(K // BLOCK_K):
        rk = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        # ── A tile + dequant ───────────────────────────────────────────
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0,
        )
        sa = tl.load(
            SA_ptr + kb * stride_sA_kb + rm * stride_sA_m,
            mask=rm < M, other=1.0,
        )
        a = a.to(tl.float32) * sa[:, None]

        # ── Gate tile + dequant ────────────────────────────────────────
        b_g = tl.load(
            B_ptr + rn_gate[:, None] * stride_bn + rk[None, :] * stride_bk,
            mask=(rn_gate[:, None] < NI * BLOCK_N) & (rk[None, :] < K), other=0.0,
        )
        sb_g = tl.load(SB_ptr + pid_n * stride_sB_nb + kb * stride_sB_kb)
        b_g  = b_g.to(tl.float32) * sb_g
        acc_gate = tl.dot(a, tl.trans(b_g), acc_gate, out_dtype=tl.float32)

        # ── Up tile + dequant ──────────────────────────────────────────
        b_u = tl.load(
            B_ptr + rn_up[:, None] * stride_bn + rk[None, :] * stride_bk,
            mask=(rn_up[:, None] < 2 * NI * BLOCK_N) & (rk[None, :] < K), other=0.0,
        )
        sb_u = tl.load(SB_ptr + (NI + pid_n) * stride_sB_nb + kb * stride_sB_kb)
        b_u  = b_u.to(tl.float32) * sb_u
        acc_up = tl.dot(a, tl.trans(b_u), acc_up, out_dtype=tl.float32)

    # ── SwiGLU epilogue: z = silu(x2) * x1  (x1=gate, x2=up) ─────────
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    z = (acc_up * (1.0 / (1.0 + tl.exp(-acc_up)))) * acc_gate

    rn_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn_out[None, :] * stride_cn,
        z,
        mask=(rm[:, None] < M) & (rn_out[None, :] < NI * BLOCK_N),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 2: float32 A @ dequant(FP8 B).T  →  float32 C
#
# Tensor layouts
#   A      : [M, K]               float32     (SwiGLU output)
#   B      : [N, K]               fp8_e4m3fn  (W2 for one expert, [H, I])
#   Scale_B: [N//128, K//128]     float32
#   C      : [M, N]               float32
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _f32_fp8_gemm(
    A_ptr,
    B_ptr, SB_ptr,
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

    for kb in range(K // BLOCK_K):
        rk = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0,
        ).to(tl.float32)

        b = tl.load(
            B_ptr + rn[:, None] * stride_bn + rk[None, :] * stride_bk,
            mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0,
        )
        sb = tl.load(SB_ptr + pid_n * stride_sB_nb + kb * stride_sB_kb)
        b  = b.to(tl.float32) * sb

        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Python launchers
# ─────────────────────────────────────────────────────────────────────────────

def _gemm1_swiglu(
    x:        torch.Tensor,   # [Tk, H]       fp8_e4m3fn
    x_scale:  torch.Tensor,   # [H//128, Tk]  float32
    w:        torch.Tensor,   # [2*I, H]      fp8_e4m3fn
    w_scale:  torch.Tensor,   # [2*I//128, H//128]  float32
) -> torch.Tensor:            # [Tk, I]       float32  (already SwiGLU-activated)
    Tk, K = x.shape
    NI    = I // BLOCK_N                      # = 16
    out   = torch.empty((Tk, I), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(Tk, BLOCK_M_GEMM1), NI)
    _fp8_fp8_gemm_swiglu[grid](
        x,       x_scale,
        w,       w_scale,
        out,
        Tk, K, NI,
        x.stride(0),       x.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        w.stride(0),       w.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0),     out.stride(1),
        BLOCK_M=BLOCK_M_GEMM1, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def _gemm2(
    z:        torch.Tensor,   # [Tk, I]       float32
    w:        torch.Tensor,   # [H, I]        fp8_e4m3fn
    w_scale:  torch.Tensor,   # [H//128, I//128]  float32
) -> torch.Tensor:            # [Tk, H]       float32
    Tk, K = z.shape
    N     = w.shape[0]        # H = 7168
    out   = torch.empty((Tk, N), dtype=torch.float32, device=z.device)

    grid = (triton.cdiv(Tk, BLOCK_M_GEMM2), triton.cdiv(N, BLOCK_N))
    _f32_fp8_gemm[grid](
        z,
        w,       w_scale,
        out,
        Tk, N, K,
        z.stride(0),       z.stride(1),
        w.stride(0),       w.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0),     out.stride(1),
        BLOCK_M=BLOCK_M_GEMM2, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Routing (pure PyTorch — identical to the Python reference)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_routing(
    routing_logits:        torch.Tensor,   # [T, 256] float32
    routing_bias:          torch.Tensor,   # [256] bfloat16
    routed_scaling_factor: float,
):
    T = routing_logits.shape[0]

    s      = torch.sigmoid(routing_logits)
    s_bias = s + routing_bias.to(torch.float32)

    s_grouped    = s_bias.view(T, N_GROUP, E_GLOBAL // N_GROUP)
    top2, _      = torch.topk(s_grouped, k=2, dim=2, sorted=False)
    group_scores = top2.sum(dim=2)

    _, grp_idx  = torch.topk(group_scores, k=TOPK_GROUP, dim=1, sorted=False)
    grp_mask    = torch.zeros_like(group_scores)
    grp_mask.scatter_(1, grp_idx, 1.0)
    score_mask  = (
        grp_mask.unsqueeze(2)
        .expand(T, N_GROUP, E_GLOBAL // N_GROUP)
        .reshape(T, E_GLOBAL)
    )

    masked      = s_bias.masked_fill(score_mask == 0, float("-inf"))
    _, topk_idx = torch.topk(masked, k=TOP_K, dim=1, sorted=False)

    weight_mask = torch.zeros_like(s)
    weight_mask.scatter_(1, topk_idx, 1.0)
    weights     = s * weight_mask
    weights     = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-20)
    weights     = weights * routed_scaling_factor

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

    # ── 2. Float32 accumulation buffer ───────────────────────────────────
    out_f32 = torch.zeros((T, H), dtype=torch.float32, device=device)

    # ── 3. Per-local-expert loop ──────────────────────────────────────────
    local_start = int(local_expert_offset)
    for le in range(E_LOCAL):
        ge = local_start + le
        if ge >= E_GLOBAL:
            continue

        sel     = (topk_idx == ge).any(dim=1)
        if not sel.any():
            continue
        tok_idx = sel.nonzero(as_tuple=False).squeeze(1)

        x        = hidden_states[tok_idx].contiguous()
        x_scale  = hidden_states_scale[:, tok_idx].contiguous()

        w1       = gemm1_weights[le]
        w1_sc    = gemm1_weights_scale[le]
        w2       = gemm2_weights[le]
        w2_sc    = gemm2_weights_scale[le]

        # ── GEMM1 + SwiGLU (fused, no intermediate [Tk,4096] write) ──
        z = _gemm1_swiglu(x, x_scale, w1, w1_sc)    # [Tk, I] float32

        # ── GEMM2 ─────────────────────────────────────────────────────
        o = _gemm2(z, w2, w2_sc)                     # [Tk, H] float32

        # ── Weighted accumulate ────────────────────────────────────────
        w_tok = weights[tok_idx, ge].unsqueeze(1)
        out_f32.index_add_(0, tok_idx, o * w_tok)

    # ── 4. Write to pre-allocated bfloat16 output ─────────────────────────
    output.copy_(out_f32)
