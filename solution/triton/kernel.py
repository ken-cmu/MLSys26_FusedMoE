"""
Triton FP8 Fused MoE kernel — v3.

Design: same as v1 (FP8 GEMM kernels, SwiGLU in PyTorch) but both GEMM
kernels are decorated with @triton.autotune so Triton searches over
BLOCK_M, num_warps, and num_stages and picks the best config per
distinct (M, N, K) shape it encounters.

BLOCK_N=128 and BLOCK_K=128 are held fixed so that the B-scale and
A-scale indexing (pid_n == n_block, kb == k_block) stays exact without
any extra integer arithmetic.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Model constants
# ─────────────────────────────────────────────────────────────────────────────
H          = 7168
I          = 2048
E_GLOBAL   = 256
E_LOCAL    = 32
TOP_K      = 8
N_GROUP    = 8
TOPK_GROUP = 4
QUANT_BLOCK = 128   # FP8 block-scale granularity

# BLOCK_N and BLOCK_K are fixed; BLOCK_M is autotuned.
BLOCK_N = 128
BLOCK_K = 128

# ─────────────────────────────────────────────────────────────────────────────
# Autotune config lists
# Vary BLOCK_M (tile rows = tokens), num_warps, and num_stages.
# num_stages controls Triton's software prefetch pipeline depth.
# ─────────────────────────────────────────────────────────────────────────────
_GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': 128, 'BLOCK_K': 128},
                  num_warps=NW, num_stages=NS)
    for BM in [16, 32, 64, 128]
    for NW in [4, 8]
    for NS in [3, 4]
]


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel 1 (autotuned): dequant(FP8 A) @ dequant(FP8 B).T → float32 C
#
# A      : [M, K]          fp8_e4m3fn
# Scale_A: [K//128, M]     float32     (per k-block, per token)
# B      : [N, K]          fp8_e4m3fn
# Scale_B: [N//128, K//128] float32
# C      : [M, N]          float32
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(configs=_GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _fp8_fp8_gemm(
    A_ptr, SA_ptr,
    B_ptr, SB_ptr,
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

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(K // BLOCK_K):
        rk = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0,
        )
        sa = tl.load(
            SA_ptr + kb * stride_sA_kb + rm * stride_sA_m,
            mask=rm < M, other=1.0,
        )
        a = a.to(tl.float32) * sa[:, None]

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
# Triton kernel 2 (autotuned): float32 A @ dequant(FP8 B).T → float32 C
#
# A      : [M, K]          float32     (SwiGLU output)
# B      : [N, K]          fp8_e4m3fn
# Scale_B: [N//128, K//128] float32
# C      : [M, N]          float32
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(configs=_GEMM_CONFIGS, key=['M', 'N', 'K'])
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
# Python launchers — grid uses lambda meta so the autotuned BLOCK_M is picked up
# ─────────────────────────────────────────────────────────────────────────────

def _gemm1(
    x:       torch.Tensor,   # [Tk, H]          fp8_e4m3fn
    x_scale: torch.Tensor,   # [H//128, Tk]      float32
    w:       torch.Tensor,   # [2*I, H]          fp8_e4m3fn
    w_scale: torch.Tensor,   # [2*I//128, H//128] float32
) -> torch.Tensor:           # [Tk, 2*I]         float32
    Tk, K = x.shape
    N     = w.shape[0]       # 2*I = 4096
    out   = torch.empty((Tk, N), dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(Tk, meta['BLOCK_M']),
                         triton.cdiv(N,  meta['BLOCK_N']))
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
    )
    return out


def _gemm2(
    z:       torch.Tensor,   # [Tk, I]           float32
    w:       torch.Tensor,   # [H, I]            fp8_e4m3fn
    w_scale: torch.Tensor,   # [H//128, I//128]   float32
) -> torch.Tensor:           # [Tk, H]            float32
    Tk, K = z.shape
    N     = w.shape[0]       # H = 7168
    out   = torch.empty((Tk, N), dtype=torch.float32, device=z.device)

    grid = lambda meta: (triton.cdiv(Tk, meta['BLOCK_M']),
                         triton.cdiv(N,  meta['BLOCK_N']))
    _f32_fp8_gemm[grid](
        z,
        w,       w_scale,
        out,
        Tk, N, K,
        z.stride(0),       z.stride(1),
        w.stride(0),       w.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0),     out.stride(1),
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Routing (pure PyTorch — identical to v1 / Python reference)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_routing(
    routing_logits:        torch.Tensor,
    routing_bias:          torch.Tensor,
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
    routing_logits:        torch.Tensor,
    routing_bias:          torch.Tensor,
    hidden_states:         torch.Tensor,
    hidden_states_scale:   torch.Tensor,
    gemm1_weights:         torch.Tensor,
    gemm1_weights_scale:   torch.Tensor,
    gemm2_weights:         torch.Tensor,
    gemm2_weights_scale:   torch.Tensor,
    local_expert_offset:   int,
    routed_scaling_factor: float,
    output:                torch.Tensor,
):
    T      = hidden_states.shape[0]
    device = hidden_states.device

    topk_idx, weights = _compute_routing(
        routing_logits, routing_bias, routed_scaling_factor
    )

    out_f32 = torch.zeros((T, H), dtype=torch.float32, device=device)

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

        w1, w1_sc = gemm1_weights[le], gemm1_weights_scale[le]
        w2, w2_sc = gemm2_weights[le], gemm2_weights_scale[le]

        g1     = _gemm1(x, x_scale, w1, w1_sc)
        x1, x2 = g1[:, :I], g1[:, I:]
        z      = F.silu(x2) * x1

        o = _gemm2(z, w2, w2_sc)

        w_tok = weights[tok_idx, ge].unsqueeze(1)
        out_f32.index_add_(0, tok_idx, o * w_tok)

    output.copy_(out_f32)
