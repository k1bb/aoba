"""Hardware-portable fused batched tridiagonal Thomas solver (Metal / CUDA / CPU).

Solves M independent K x K tridiagonal linear systems simultaneously:
    dl[k, m] * x[k - 1, m] + d[k, m] * x[k, m] + du[k, m] * x[k + 1, m] = rhs[k, m]
for k = 0, ..., K - 1 and m = 0, ..., M - 1, with dl[0, m] = 0 and du[K - 1, m] = 0.

Used identically by:
1. Vertically implicit acoustic/gravity HEVI dynamics (Skamarock et al., 2012, Sec. 3;
   Nishizawa et al., 2025, SCALE-RM Sec. 6; K = N_z or N_z + 1).
2. Generalized multi-layer two-stream radiative transfer (Toon et al., 1989, Eqs. 39-47;
   K = 2 * N_z).

On Apple Silicon GPUs (`mps`), compiles a fused Metal compute shader (1 GPU thread per
column m) to eliminate Python loop kernel-launch overhead while retaining an identical
pure-PyTorch implementation for CPU and CUDA backends.
"""

from __future__ import annotations

import torch

_MAX_METAL_K: int = 128
_METAL_SHADER_LIB = None

_METAL_THOMAS_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void batched_thomas(
    device const float* dl  [[buffer(0)]],
    device const float* d   [[buffer(1)]],
    device const float* du  [[buffer(2)]],
    device const float* rhs [[buffer(3)]],
    device float* out       [[buffer(4)]],
    constant uint& K        [[buffer(5)]],
    constant uint& M        [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= M) return;

    float c_star[128];
    float d_star[128];

    float inv_d0 = 1.0f / d[tid];
    c_star[0] = du[tid] * inv_d0;
    d_star[0] = rhs[tid] * inv_d0;

    for (uint k = 1; k < K; ++k) {
        uint idx = k * M + tid;
        float denom = 1.0f / (d[idx] - dl[idx] * c_star[k - 1]);
        c_star[k] = du[idx] * denom;
        d_star[k] = (rhs[idx] - dl[idx] * d_star[k - 1]) * denom;
    }

    uint last_idx = (K - 1) * M + tid;
    float x_next = d_star[K - 1];
    out[last_idx] = x_next;

    for (int k = int(K) - 2; k >= 0; --k) {
        uint idx = uint(k) * M + tid;
        x_next = d_star[k] - c_star[k] * x_next;
        out[idx] = x_next;
    }
}
"""


def _get_metal_lib():
    global _METAL_SHADER_LIB
    if _METAL_SHADER_LIB is None and hasattr(torch.mps, "compile_shader"):
        _METAL_SHADER_LIB = torch.mps.compile_shader(_METAL_THOMAS_SRC)
    return _METAL_SHADER_LIB


@torch.jit.script
def solve_tridiagonal_pytorch(
    dl: torch.Tensor,
    d: torch.Tensor,
    du: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch JIT-compiled batched Thomas algorithm along dim=0 for shape [K, ...].

    Reference: Toon et al. (1989, J. Geophys. Res., 94, Eqs. 45-47).
    """
    K = d.shape[0]
    c_list: list[torch.Tensor] = []
    d_list: list[torch.Tensor] = []

    inv_b0 = 1.0 / d[0]
    c_prev = du[0] * inv_b0
    d_prev = rhs[0] * inv_b0
    c_list.append(c_prev)
    d_list.append(d_prev)

    for k in range(1, K):
        inv_denom = 1.0 / (d[k] - dl[k] * c_prev)
        c_prev = du[k] * inv_denom if k < K - 1 else torch.zeros_like(d_prev)
        d_prev = (rhs[k] - dl[k] * d_prev) * inv_denom
        c_list.append(c_prev)
        d_list.append(d_prev)

    x_list: list[torch.Tensor] = [d_prev] * K
    x_next = d_list[K - 1]
    x_list[K - 1] = x_next
    for k in range(K - 2, -1, -1):
        x_next = d_list[k] - c_list[k] * x_next
        x_list[k] = x_next

    return torch.stack(x_list, dim=0)


def solve_tridiagonal_batched(
    dl: torch.Tensor,
    d: torch.Tensor,
    du: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Solve batched tridiagonal systems of shape [K, ...] along leading dimension K.

    Automatically dispatches to the fused Metal compute shader when running in Float32
    on Apple Silicon (`mps`) with K <= 128 and no autograd tracking; otherwise uses
    the identical pure-PyTorch Thomas sweep.
    """
    orig_shape = rhs.shape
    K = orig_shape[0]
    if K == 1:
        return rhs / d

    use_metal = (
        rhs.device.type == "mps"
        and rhs.dtype == torch.float32
        and K <= _MAX_METAL_K
        and not (
            dl.requires_grad or d.requires_grad or du.requires_grad or rhs.requires_grad
        )
    )
    if use_metal:
        lib = _get_metal_lib()
        if lib is not None:
            M = rhs[0].numel()
            if M > 0:
                dl_c = dl.contiguous().view(K, M)
                d_c = d.contiguous().view(K, M)
                du_c = du.contiguous().view(K, M)
                rhs_c = rhs.contiguous().view(K, M)
                out = torch.empty_like(rhs_c)
                k_buf = torch.tensor([K], dtype=torch.uint32, device=rhs.device)
                m_buf = torch.tensor([M], dtype=torch.uint32, device=rhs.device)
                lib.batched_thomas(dl_c, d_c, du_c, rhs_c, out, k_buf, m_buf, threads=M)
                return out.view(orig_shape)

    return solve_tridiagonal_pytorch(dl, d, du, rhs)
