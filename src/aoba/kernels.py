"""Hardware-portable fused batched tridiagonal Thomas solver (Metal / CUDA / CPU).

Solves M independent K x K tridiagonal linear systems simultaneously:
    dl[k, m] * x[k - 1, m] + d[k, m] * x[k, m] + du[k, m] * x[k + 1, m] = rhs[k, m]
for k = 0, ..., K - 1 and m = 0, ..., M - 1, with dl[0, m] = 0 and du[K - 1, m] = 0.

Used identically by:
1. Vertically implicit acoustic/gravity HEVI dynamics (Skamarock et al., 2012, Sec. 3;
   Nishizawa et al., 2025, SCALE-RM Sec. 6; K = N_z or N_z + 1).
2. Generalized multi-layer two-stream radiative transfer (Toon et al., 1989, Eqs. 39-47;
   K = 2 * N_z).

On Apple Silicon GPUs (`mps`), compiles fused Metal compute shaders (1 GPU thread per
column m or 32-lane SIMD warp). On NVIDIA GPUs (`cuda`), automatically transpiles the
identical single-source shaders into native CUDA C++17 kernels via `aoba.kernels_cuda`
and PyTorch's JIT extension compiler, while retaining an identical pure-PyTorch fallback
for CPU backends.
"""

from __future__ import annotations

from typing import Any
import torch


def mean_2d(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """Fast 1D reduction over the trailing `(ny, nx)` dimensions."""
    m = x.reshape(*x.shape[:-2], -1).mean(dim=-1)
    return m.unsqueeze(-1).unsqueeze(-1) if keepdim else m


def sum_2d(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """Fast 1D sum over the trailing `(ny, nx)` dimensions."""
    s = x.reshape(*x.shape[:-2], -1).sum(dim=-1)
    return s.unsqueeze(-1).unsqueeze(-1) if keepdim else s

from aoba.kernels_cuda import get_cuda_lib
from aoba.kernels_extra import (
    EXTRA_METAL_SHADER_SRC,
    _get_scvt_mesh_cache,
    run_crm_fct_advect_metal,
    run_crm_fct_pair_metal,
    run_crm_fct_quad_metal,
    run_crm_microphysics_2m_metal,
    run_crm_microphysics_metal,
    run_crm_mynn_metal,
    run_crm_sgs_metal,
    run_scvt_div_metal,
    run_scvt_fast_acoustic_metal,
    run_scvt_scalar_lap_metal,
    run_scvt_storm_smag_metal,
    run_scvt_trisk_rot_metal,
    run_scvt_vec_lap_metal,
    run_scvt_vorticity_metal,
)
from aoba.kernels_physics import (
    PHYSICS_METAL_SHADER_SRC,
    run_crm_distill_closure_metal,
    run_crm_distill_extract_metal,
    run_crm_student_tile_update_metal,
    run_dust_column_metal,
    run_natural_aerosol_4sp_metal,
    run_ngwd_propagate_metal,
    run_ocean_bgc_column_metal,
    run_ogcm_baroclinic_f_fixed_metal,
    run_ogcm_baroclinic_rk3_update_metal,
    run_ogcm_barotropic_rk3_metal,
    run_ogcm_fct_zstar_metal,
    run_ogcm_gm_bolus_edge_metal,
    run_ogcm_kpp_vdiff_metal,
    run_ogcm_mom_vdiff_metal,
    run_ogcm_pre_kpp_metal,
    run_ogcm_tracer_vdiff_freeze_metal,
    run_oro_gwd_metal,
    run_rad_ckd_metal,
    run_rad_distill_closure_metal,
    run_rad_distill_extract_metal,
    run_rad_four_band_metal,
    run_scvt_fct_advect_metal,
    run_scvt_project_winds_metal,
    run_scvt_reconstruct_winds_metal,
    run_scvt_rk3_prep_metal,
    run_seaice_evp_subcycle_metal,
    run_seaice_itd_sync_metal,
    run_surface_albedos_metal,
    run_surface_land_newton_jac_metal,
    run_surface_land_pre_bio_metal,
    run_surface_ocean_seaice_post_metal,
    run_surface_transfer_metal,
    run_two_leaf_gas_exchange_metal,
)
from aoba.kernels_esm import (
    ESM_METAL_SHADER_SRC,
    run_dynamic_lake_step_metal,
    run_microphysics_6class_metal,
    run_ozone_chem_metal,
    run_rad_ckd_hd_metal,
    run_rad_rrtmgp_metal,
)

_MAX_METAL_K: int = 128
_MAX_COL_METAL_K: int = 64
_METAL_SHADER_LIB = None
_CRM_WS_CACHE: dict[tuple[str, int, int, int, int, int, int], dict[str, torch.Tensor]] = {}

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

kernel void crm_fast_uv_flux(
    device const float* rho_x        [[buffer(0)]],
    device const float* rho_y        [[buffer(1)]],
    device const float* rtot_x_inv   [[buffer(2)]],
    device const float* rtot_y_inv   [[buffer(3)]],
    device const float* a_p_s        [[buffer(4)]],   // pres_s - dp_dThm_s * Thm_ref
    device const float* dp_dThm_s    [[buffer(5)]],
    device const float* tend_u_slow  [[buffer(6)]],
    device const float* tend_v_slow  [[buffer(7)]],
    device const float* Thm_cur      [[buffer(8)]],
    device const float* u_cur        [[buffer(9)]],
    device const float* v_cur        [[buffer(10)]],
    device float* u_nxt              [[buffer(11)]],
    device float* v_nxt              [[buffer(12)]],
    device float* fx_buf             [[buffer(13)]],
    device float* fy_buf             [[buffer(14)]],
    device float* fx_acc             [[buffer(15)]],
    device float* fy_acc             [[buffer(16)]],
    constant ulong4& u_dim           [[buffer(17)]],  // [nz, n_col, ny, nx]
    constant uint& accum             [[buffer(18)]],
    constant float4& f_p             [[buffer(19)]],  // [dt_ac, dx, dy, 0]
    uint gid [[thread_position_in_grid]])
{
    uint nz = uint(u_dim.x);
    uint n_col = uint(u_dim.y);
    uint ny = uint(u_dim.z);
    uint nx = uint(u_dim.w);
    uint stride_c = ny * nx;
    uint M = n_col * stride_c;
    if (gid >= nz * M) return;

    uint k = gid / M;
    uint tid = gid - k * M;
    uint col = tid / stride_c;
    uint rem = tid - col * stride_c;
    uint iy = rem / nx;
    uint ix = rem - iy * nx;

    uint ix_p1 = (ix + 1 < nx) ? (ix + 1) : 0;
    uint iy_p1 = (iy + 1 < ny) ? (iy + 1) : 0;
    uint idx_xp = k * M + col * stride_c + iy * nx + ix_p1;
    uint idx_yp = k * M + col * stride_c + iy_p1 * nx + ix;

    float p_self = fma(dp_dThm_s[gid],    Thm_cur[gid],    a_p_s[gid]);
    float p_xp   = fma(dp_dThm_s[idx_xp], Thm_cur[idx_xp], a_p_s[idx_xp]);
    float p_yp   = fma(dp_dThm_s[idx_yp], Thm_cur[idx_yp], a_p_s[idx_yp]);

    float dt_ac = f_p.x;
    float inv_dx = 1.0f / f_p.y;
    float inv_dy = 1.0f / f_p.z;

    float pgf_u = -(p_xp - p_self) * inv_dx * rtot_x_inv[gid];
    float pgf_v = -(p_yp - p_self) * inv_dy * rtot_y_inv[gid];

    float u0_v = u_cur[gid];
    float v0_v = v_cur[gid];
    float u1_v = u0_v + dt_ac * (tend_u_slow[gid] + pgf_u);
    float v1_v = v0_v + dt_ac * (tend_v_slow[gid] + pgf_v);

    float fx_v = rho_x[gid] * (1.10f * u1_v - 0.10f * u0_v);
    float fy_v = rho_y[gid] * (1.10f * v1_v - 0.10f * v0_v);

    u_nxt[gid] = u1_v;
    v_nxt[gid] = v1_v;
    fx_buf[gid] = fx_v;
    fy_buf[gid] = fy_v;
    if (accum == 1u) {
        float w_acc = f_p.w;
        fx_acc[gid] = fx_v * w_acc;
        fy_acc[gid] = fy_v * w_acc;
    } else if (accum == 2u) {
        float w_acc = f_p.w;
        fx_acc[gid] += fx_v * w_acc;
        fy_acc[gid] += fy_v * w_acc;
    }
}

kernel void crm_fast_hevi_col(
    device const float* rho_cur       [[buffer(0)]],
    device const float* Thm_cur       [[buffer(1)]],
    device const float* w_cur         [[buffer(2)]],
    device const float* fx_buf        [[buffer(3)]],
    device const float* fy_buf        [[buffer(4)]],
    device const float* thm_x_eff     [[buffer(5)]],
    device const float* thm_y_eff     [[buffer(6)]],
    device const float* a_p_s         [[buffer(7)]],
    device const float* dp_dThm_s     [[buffer(8)]],
    device const float* q_load        [[buffer(9)]],
    device const float* tend_w_slow   [[buffer(10)]],
    device const float* tend_Thm_slow [[buffer(11)]],
    device const float* rho_dw        [[buffer(12)]],
    device const float* rho_tot_w     [[buffer(13)]],
    device const float* thm_w         [[buffer(14)]],
    device const float* dl            [[buffer(15)]],
    device const float* inv_den_buf   [[buffer(16)]],  // pre-factored inv_den in diag
    device const float* c_star_buf    [[buffer(17)]],  // pre-factored c_star in du
    device const float* dz_c          [[buffer(18)]],
    device const float* dz_w_int      [[buffer(19)]],
    device const float* sponge_w      [[buffer(20)]],
    device float* rho_nxt             [[buffer(21)]],
    device float* Thm_nxt             [[buffer(22)]],
    device float* w_nxt               [[buffer(23)]],
    device float* fz_acc              [[buffer(24)]],
    constant ulong4& u_dim            [[buffer(25)]],  // [nz, n_col, ny, nx]
    constant ulong4& u_flg            [[buffer(26)]],  // [dz_cols, accum (0=off,1=init,2=add), 0, 0]
    constant float4& f_p0             [[buffer(27)]],  // [dt_ac, beta, gravity, dx]
    constant float4& f_p1             [[buffer(28)]],  // [dy, inv_nac, 0, 0]
    uint tid [[thread_position_in_grid]])
{
    uint K = uint(u_dim.x);
    uint n_col = uint(u_dim.y);
    uint ny = uint(u_dim.z);
    uint nx = uint(u_dim.w);
    uint dz_cols = uint(u_flg.x);
    uint accum = uint(u_flg.y);
    uint stride_c = ny * nx;
    uint M = n_col * stride_c;
    if (tid >= M || K < 2u || K > 128u) return;

    uint col = tid / stride_c;
    uint rem = tid - col * stride_c;
    uint iy = rem / nx;
    uint ix = rem - iy * nx;
    uint col_dz = (dz_cols > 1u) ? col : 0u;

    uint ix_m1 = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint iy_m1 = (iy > 0u) ? (iy - 1u) : (ny - 1u);
    uint tid_xm = col * stride_c + iy * nx + ix_m1;
    uint tid_ym = col * stride_c + iy_m1 * nx + ix;

    float dt_ac = f_p0.x;
    float beta = f_p0.y;
    float gravity = f_p0.z;
    float inv_dx = 1.0f / f_p0.w;
    float inv_dy = 1.0f / f_p1.x;
    float inv_nac = f_p1.y;

    float d_star[128];

    float p_blend_prev = 0.0f;
    float rtot_blend_prev = 0.0f;
    float fz_old_below = 0.0f;
    float thm_w_below = 0.0f;
    float d_prev = 0.0f;

    for (uint k = 0u; k < K; ++k) {
        uint idx = k * M + tid;
        uint idx_xm = k * M + tid_xm;
        uint idx_ym = k * M + tid_ym;

        float fx_0 = fx_buf[idx];
        float fx_m = fx_buf[idx_xm];
        float fy_0 = fy_buf[idx];
        float fy_m = fy_buf[idx_ym];

        float div_h_rho = (fx_0 - fx_m) * inv_dx + (fy_0 - fy_m) * inv_dy;
        float fTh_x0 = fx_0 * thm_x_eff[idx];
        float fTh_xm = fx_m * thm_x_eff[idx_xm];
        float fTh_y0 = fy_0 * thm_y_eff[idx];
        float fTh_ym = fy_m * thm_y_eff[idx_ym];
        float div_h_Thm = (fTh_x0 - fTh_xm) * inv_dx + (fTh_y0 - fTh_ym) * inv_dy;

        float fz_old_above = 0.0f;
        float thm_w_above = 0.0f;
        float r_cur = rho_cur[idx];
        float T_cur = Thm_cur[idx];
        float thm_k = T_cur / max(r_cur, 1.0e-5f);
        if (k < K - 1u) {
            float w_up = w_cur[(k + 1u) * M + tid];
            fz_old_above = rho_dw[idx] * w_up;
            uint idx_p = (k + 1u) * M + tid;
            float thm_kp = Thm_cur[idx_p] / max(rho_cur[idx_p], 1.0e-5f);
            float sgn_up = (w_up > 0.0f) ? 1.0f : ((w_up < 0.0f) ? -1.0f : 0.0f);
            float dz_wu = dz_w_int[k * dz_cols + col_dz];
            float cfl_up = fabs(w_up) * (dt_ac / max(dz_wu, 1.0f));
            float beta_up = min(1.0f, max(0.0f, (cfl_up - 0.15f) * 4.0f));
            thm_w_above = 0.5f * (thm_k + thm_kp) - 0.5f * beta_up * sgn_up * (thm_kp - thm_k);
        }

        float dz_val = dz_c[k * dz_cols + col_dz];
        float r_dag = r_cur - dt_ac * (div_h_rho + (fz_old_above - fz_old_below) / dz_val);
        float T_dag = T_cur + dt_ac * (
            -div_h_Thm
            - (thm_w_above * fz_old_above - thm_w_below * fz_old_below) / dz_val
            + tend_Thm_slow[idx]
        );
        rho_nxt[idx] = r_dag;
        Thm_nxt[idx] = T_dag;

        float ap = a_p_s[idx];
        float dp = dp_dThm_s[idx];
        float T_blend = (1.0f - beta) * T_cur + beta * T_dag;
        float p_blend = fma(dp, T_blend, ap);
        float rtot_blend = ((1.0f - beta) * r_cur + beta * r_dag) * q_load[idx];

        if (k >= 1u) {
            uint um = k - 1u;
            uint idx_m = um * M + tid;
            float dz_w_val = dz_w_int[um * dz_cols + col_dz];
            float nh_imb = (p_blend - p_blend_prev) / dz_w_val + 0.5f * (rtot_blend + rtot_blend_prev) * gravity;
            float rdw_m = rho_dw[idx_m];
            float rtw_m = max(rho_tot_w[idx_m], 1.0e-5f);
            float rhs_m = dt_ac * (
                rdw_m * tend_w_slow[idx_m]
                - (rdw_m / rtw_m) * nh_imb
                - sponge_w[um] * fz_old_below
            );
            float inv_den = inv_den_buf[idx_m];
            d_prev = (um == 0u) ? (rhs_m * inv_den) : ((rhs_m - dl[idx_m] * d_prev) * inv_den);
            d_star[um] = d_prev;
        }
        p_blend_prev = p_blend;
        rtot_blend_prev = rtot_blend;
        fz_old_below = fz_old_above;
        thm_w_below = thm_w_above;
    }

    w_nxt[tid] = 0.0f;
    w_nxt[K * M + tid] = 0.0f;
    if (accum == 1u) {
        fz_acc[tid] = 0.0f;
        fz_acc[K * M + tid] = 0.0f;
    }

    float dfz_above = 0.0f;
    float thm_w_above = 0.0f;
    for (int m = int(K) - 2; m >= 0; --m) {
        uint um = uint(m);
        uint idx_m = um * M + tid;
        float dfz_cur = (m == int(K) - 2) ? d_star[um] : (d_star[um] - c_star_buf[idx_m] * dfz_above);
        float rdw_m = rho_dw[idx_m];
        float fz_old_m = rdw_m * w_cur[(um + 1u) * M + tid];
        float fz_new_m = fz_old_m + dfz_cur;
        w_nxt[(um + 1u) * M + tid] = fz_new_m / max(rdw_m, 1.0e-5f);
        if (accum == 1u) {
            fz_acc[(um + 1u) * M + tid] = (fz_old_m + beta * dfz_cur) * inv_nac;
        } else if (accum == 2u) {
            fz_acc[(um + 1u) * M + tid] += (fz_old_m + beta * dfz_cur) * inv_nac;
        }

        uint k_layer = um + 1u;
        uint idx_l = k_layer * M + tid;
        float thm_w_m = thm_w[idx_m];
        float dz_val = dz_c[k_layer * dz_cols + col_dz];
        rho_nxt[idx_l] = max(rho_nxt[idx_l] - beta * dt_ac * (dfz_above - dfz_cur) / dz_val, 1.0e-5f);
        Thm_nxt[idx_l] = Thm_nxt[idx_l] - beta * dt_ac * (thm_w_above * dfz_above - thm_w_m * dfz_cur) / dz_val;

        dfz_above = dfz_cur;
        thm_w_above = thm_w_m;
    }

    float dz_0 = dz_c[col_dz];
    rho_nxt[tid] = max(rho_nxt[tid] - beta * dt_ac * dfz_above / dz_0, 1.0e-5f);
    Thm_nxt[tid] = Thm_nxt[tid] - beta * dt_ac * (thm_w_above * dfz_above) / dz_0;
}

inline float ws3_face(float qm2, float qm1, float q0, float qp1, float vel) {
    float sgn = (vel > 0.0f) ? 1.0f : ((vel < 0.0f) ? -1.0f : 0.0f);
    return ((-qm2 + 7.0f * qm1 + 7.0f * q0 - qp1)
            + sgn * (qp1 - 3.0f * q0 + 3.0f * qm1 - qm2)) * (1.0f / 12.0f);
}

kernel void crm_3d_fast_hevi_loop(
    device float* st4_a               [[buffer(0)]],   // [4, nz, M]: rho, Thm, u, v (in/out)
    device float* w_a                 [[buffer(1)]],   // [nz+1, M] (in/out)
    device float* fxy_buf             [[buffer(2)]],   // [2, nz, M]: fx, fy scratch
    device float* prep_c0             [[buffer(3)]],   // [4, nz, M]: [p_blend_scratch, a_p_s, dp_dThm_s, rtot_blend_scratch]
    device const float* prep_c1       [[buffer(4)]],   // [8, nz, M]
    device const float* prep_w        [[buffer(5)]],   // [7, nz-1, M]
    device const float* q_load        [[buffer(6)]],
    device const float* tend_Thm_slow [[buffer(7)]],
    device const float* dz_c          [[buffer(8)]],
    device const float* dz_w_int      [[buffer(9)]],
    device const float* sponge_w      [[buffer(10)]],
    device float* fxy_acc             [[buffer(11)]],  // [2, nz, M]: fx_acc, fy_acc
    device float* fz_acc              [[buffer(12)]],  // [nz+1, M]
    constant ulong4& u_pack           [[buffer(13)]],  // [nz, n_col, (ny<<16)|nx, (dz_cols<<16)|(n_ac<<1)|accum]
    constant float4& f_p0             [[buffer(14)]],  // [dt_ac, beta, gravity, dx]
    constant float4& f_p1             [[buffer(15)]],  // [dy, inv_nac, 0, 0]
    uint col [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    uint K = uint(u_pack.x);
    uint n_col = uint(u_pack.y);
    uint ny = uint(u_pack.z >> 16u);
    uint nx = uint(u_pack.z & 0xFFFFu);
    uint dz_cols = uint(u_pack.w >> 16u);
    uint n_ac = uint((u_pack.w >> 1u) & 0x7FFFu);
    uint accum = uint(u_pack.w & 1u);
    uint stride_c = ny * nx;
    uint M = n_col * stride_c;
    if (col >= n_col || K < 2u || K > 64u || stride_c > 64u) return;

    uint kz = ltid / stride_c;
    uint rem = ltid - kz * stride_c;
    if (kz >= 4u) return;
    uint tid = col * stride_c + rem;
    uint iy = rem / nx;
    uint ix = rem - iy * nx;
    uint col_dz = (dz_cols > 1u) ? col : 0u;

    uint ix_p1 = (ix + 1u < nx) ? (ix + 1u) : 0u;
    uint iy_p1 = (iy + 1u < ny) ? (iy + 1u) : 0u;
    uint ix_m1 = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint iy_m1 = (iy > 0u) ? (iy - 1u) : (ny - 1u);
    uint base_col = col * stride_c;
    uint tid_xp = base_col + iy * nx + ix_p1;
    uint tid_yp = base_col + iy_p1 * nx + ix;
    uint tid_xm = base_col + iy * nx + ix_m1;
    uint tid_ym = base_col + iy_m1 * nx + ix;

    uint total = K * M;
    uint n_int_pts = (K - 1u) * M;
    device float* rho_a = st4_a;
    device float* Thm_a = st4_a + total;
    device float* u_a   = st4_a + 2u * total;
    device float* v_a   = st4_a + 3u * total;
    device float* fx_buf = fxy_buf;
    device float* fy_buf = fxy_buf + total;
    device float* fx_acc = fxy_acc;
    device float* fy_acc = fxy_acc + total;

    device float* p_blend_buf       = prep_c0;
    device const float* a_p_s       = prep_c0 + total;
    device const float* dp_dThm_s   = prep_c0 + 2u * total;
    device float* rtot_blend_buf    = prep_c0 + 3u * total;
    device const float* rho_x       = prep_c1;
    device const float* rho_y       = prep_c1 + total;
    device const float* rtot_x_inv  = prep_c1 + 2u * total;
    device const float* rtot_y_inv  = prep_c1 + 3u * total;
    device const float* thm_x_eff   = prep_c1 + 4u * total;
    device const float* thm_y_eff   = prep_c1 + 5u * total;
    device const float* tend_u_slow = prep_c1 + 6u * total;
    device const float* tend_v_slow = prep_c1 + 7u * total;

    device const float* tend_w_slow = prep_w;
    device const float* rho_dw      = prep_w + n_int_pts;
    device const float* rho_tot_w   = prep_w + 2u * n_int_pts;
    device const float* thm_w       = prep_w + 3u * n_int_pts;
    device const float* dl          = prep_w + 4u * n_int_pts;
    device const float* diag        = prep_w + 5u * n_int_pts;
    device const float* du          = prep_w + 6u * n_int_pts;

    float dt_ac = f_p0.x;
    float beta = f_p0.y;
    float gravity = f_p0.z;
    float inv_dx = 1.0f / f_p0.w;
    float inv_dy = 1.0f / f_p1.x;
    float inv_nac = f_p1.y;

    threadgroup float sm_dstar[64u * 64u];

    for (uint ac = 0u; ac < n_ac; ++ac) {
        // Step A (4x k-parallel, 256 threads/block): Horizontal acoustic PGF + u/v + mass flux
        for (uint k = kz; k < K; k += 4u) {
            uint kM = k * M;
            uint idx = kM + tid;
            uint idx_xp = kM + tid_xp;
            uint idx_yp = kM + tid_yp;

            float p_self = fma(dp_dThm_s[idx],    Thm_a[idx],    a_p_s[idx]);
            float p_xp   = fma(dp_dThm_s[idx_xp], Thm_a[idx_xp], a_p_s[idx_xp]);
            float p_yp   = fma(dp_dThm_s[idx_yp], Thm_a[idx_yp], a_p_s[idx_yp]);

            float pgf_u = -(p_xp - p_self) * inv_dx * rtot_x_inv[idx];
            float pgf_v = -(p_yp - p_self) * inv_dy * rtot_y_inv[idx];

            float u0_v = u_a[idx];
            float v0_v = v_a[idx];
            float u1_v = u0_v + dt_ac * (tend_u_slow[idx] + pgf_u);
            float v1_v = v0_v + dt_ac * (tend_v_slow[idx] + pgf_v);

            float fx_v = rho_x[idx] * (1.10f * u1_v - 0.10f * u0_v);
            float fy_v = rho_y[idx] * (1.10f * v1_v - 0.10f * v0_v);

            u_a[idx] = u1_v;
            v_a[idx] = v1_v;
            fx_buf[idx] = fx_v;
            fy_buf[idx] = fy_v;
            if (accum != 0u) {
                if (ac == 0u) {
                    fx_acc[idx] = fx_v * inv_nac;
                    fy_acc[idx] = fy_v * inv_nac;
                } else {
                    fx_acc[idx] += fx_v * inv_nac;
                    fy_acc[idx] += fy_v * inv_nac;
                }
            }
            // Cache live theta_m in idle sm_dstar for race-free conservative upwind vertical Thm flux in Step B-1
            sm_dstar[k * stride_c + rem] = Thm_a[idx] / max(rho_a[idx], 1.0e-5f);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // Step B-1 (4x k-parallel): Explicit horizontal + live upwind vertical-old divergence -> r_dag, T_dag
        for (uint k = kz; k < K; k += 4u) {
            uint kM = k * M;
            uint idx = kM + tid;
            uint idx_xm = kM + tid_xm;
            uint idx_ym = kM + tid_ym;

            float fx_0 = fx_buf[idx];
            float fx_m = fx_buf[idx_xm];
            float fy_0 = fy_buf[idx];
            float fy_m = fy_buf[idx_ym];

            float div_h_rho = (fx_0 - fx_m) * inv_dx + (fy_0 - fy_m) * inv_dy;
            float div_h_Thm =
                (fx_0 * thm_x_eff[idx] - fx_m * thm_x_eff[idx_xm]) * inv_dx
                + (fy_0 * thm_y_eff[idx] - fy_m * thm_y_eff[idx_ym]) * inv_dy;

            float thm_k = sm_dstar[k * stride_c + rem];
            float fz_old_below = 0.0f;
            float thm_w_below = 0.0f;
            if (k > 0u) {
                float w_dn = w_a[idx];
                fz_old_below = rho_dw[(k - 1u) * M + tid] * w_dn;
                float thm_km = sm_dstar[(k - 1u) * stride_c + rem];
                float sgn_dn = (w_dn > 0.0f) ? 1.0f : ((w_dn < 0.0f) ? -1.0f : 0.0f);
                float dz_wd = dz_w_int[(k - 1u) * dz_cols + col_dz];
                float beta_dn = min(1.0f, 0.25f + 1.5f * fabs(w_dn) * (dt_ac / max(dz_wd, 1.0f)));
                thm_w_below = 0.5f * (thm_km + thm_k) - 0.5f * beta_dn * sgn_dn * (thm_k - thm_km);
            }
            float fz_old_above = 0.0f;
            float thm_w_above = 0.0f;
            if (k < K - 1u) {
                float w_up = w_a[(k + 1u) * M + tid];
                fz_old_above = rho_dw[idx] * w_up;
                float thm_kp = sm_dstar[(k + 1u) * stride_c + rem];
                float sgn_up = (w_up > 0.0f) ? 1.0f : ((w_up < 0.0f) ? -1.0f : 0.0f);
                float dz_wu = dz_w_int[k * dz_cols + col_dz];
                float beta_up = min(1.0f, 0.25f + 1.5f * fabs(w_up) * (dt_ac / max(dz_wu, 1.0f)));
                thm_w_above = 0.5f * (thm_k + thm_kp) - 0.5f * beta_up * sgn_up * (thm_kp - thm_k);
            }

            float dz_val = dz_c[k * dz_cols + col_dz];
            float r_cur = rho_a[idx];
            float T_cur = Thm_a[idx];
            float r_dag = r_cur - dt_ac * (div_h_rho + (fz_old_above - fz_old_below) / dz_val);
            float T_dag = T_cur + dt_ac * (
                -div_h_Thm
                - (thm_w_above * fz_old_above - thm_w_below * fz_old_below) / dz_val
                + tend_Thm_slow[idx]
            );
            rho_a[idx] = r_dag;
            Thm_a[idx] = T_dag;

            float T_blend = (1.0f - beta) * T_cur + beta * T_dag;
            p_blend_buf[idx]    = fma(dp_dThm_s[idx], T_blend, a_p_s[idx]);
            rtot_blend_buf[idx] = ((1.0f - beta) * r_cur + beta * r_dag) * q_load[idx];
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // Step B-2 (4x k-parallel): Assemble vertical HEVI Helmholtz RHS directly into shared memory
        for (uint um = kz; um < K - 1u; um += 4u) {
            uint idx_m = um * M + tid;
            uint idx_p = (um + 1u) * M + tid;
            float p_prev  = p_blend_buf[idx_m];
            float p_cur   = p_blend_buf[idx_p];
            float rt_prev = rtot_blend_buf[idx_m];
            float rt_cur  = rtot_blend_buf[idx_p];
            float dz_w_val = dz_w_int[um * dz_cols + col_dz];
            float nh_imb = (p_cur - p_prev) / dz_w_val + 0.5f * (rt_cur + rt_prev) * gravity;
            float rdw_m = rho_dw[idx_m];
            float rtw_m = max(rho_tot_w[idx_m], 1.0e-5f);
            float fz_old_m = rdw_m * w_a[idx_p];
            sm_dstar[um * stride_c + rem] = dt_ac * (
                rdw_m * tend_w_slow[idx_m]
                - (rdw_m / rtw_m) * nh_imb
                - sponge_w[um] * fz_old_m
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step B-3: Lean register-only Thomas forward + backward sweep in shared memory
        if (kz == 0u) {
            float d_prev = 0.0f;
            for (uint um = 0u; um < K - 1u; ++um) {
                uint idx_m = um * M + tid;
                float rhs_m = sm_dstar[um * stride_c + rem];
                float id = diag[idx_m];
                d_prev = (um == 0u) ? (rhs_m * id) : ((rhs_m - dl[idx_m] * d_prev) * id);
                sm_dstar[um * stride_c + rem] = d_prev;
            }
            float dfz_above = 0.0f;
            for (int m = int(K) - 2; m >= 0; --m) {
                uint um = uint(m);
                uint idx_m = um * M + tid;
                float ds_val = sm_dstar[um * stride_c + rem];
                float dfz_cur = (m == int(K) - 2) ? ds_val : (ds_val - du[idx_m] * dfz_above);
                sm_dstar[um * stride_c + rem] = dfz_cur;
                dfz_above = dfz_cur;
            }
            w_a[tid] = 0.0f;
            w_a[K * M + tid] = 0.0f;
            if (accum != 0u && ac == 0u) {
                fz_acc[tid] = 0.0f;
                fz_acc[K * M + tid] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step B-4 (4x k-parallel): Apply implicit vertical flux increments to (rho, Thm, w, fz_acc)
        for (uint k = kz; k < K; k += 4u) {
            uint idx = k * M + tid;
            float dfz_below = (k > 0u) ? sm_dstar[(k - 1u) * stride_c + rem] : 0.0f;
            float dfz_above = (k < K - 1u) ? sm_dstar[k * stride_c + rem] : 0.0f;
            float thm_w_below = (k > 0u) ? thm_w[(k - 1u) * M + tid] : 0.0f;
            float thm_w_above = (k < K - 1u) ? thm_w[k * M + tid] : 0.0f;
            float dz_val = dz_c[k * dz_cols + col_dz];
            rho_a[idx] = max(rho_a[idx] - beta * dt_ac * (dfz_above - dfz_below) / dz_val, 1.0e-5f);
            Thm_a[idx] = Thm_a[idx] - beta * dt_ac * (thm_w_above * dfz_above - thm_w_below * dfz_below) / dz_val;
            if (k > 0u) {
                uint um = k - 1u;
                uint idx_m = um * M + tid;
                float rdw_m = rho_dw[idx_m];
                float fz_old_m = rdw_m * w_a[idx];
                w_a[idx] = (fz_old_m + dfz_below) / max(rdw_m, 1.0e-5f);
                if (accum != 0u) {
                    float fz_bar = (fz_old_m + beta * dfz_below) * inv_nac;
                    if (ac == 0u) {
                        fz_acc[idx] = fz_bar;
                    } else {
                        fz_acc[idx] += fz_bar;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void crm_copy_state5(
    device const float* rho0 [[buffer(0)]],
    device const float* Thm0 [[buffer(1)]],
    device const float* u0   [[buffer(2)]],
    device const float* v0   [[buffer(3)]],
    device const float* w0   [[buffer(4)]],
    device float* rho_a      [[buffer(5)]],
    device float* Thm_a      [[buffer(6)]],
    device float* u_a        [[buffer(7)]],
    device float* v_a        [[buffer(8)]],
    device float* w_a        [[buffer(9)]],
    constant uint& n_c       [[buffer(10)]],
    constant uint& n_w       [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n_c) {
        rho_a[gid] = rho0[gid];
        Thm_a[gid] = Thm0[gid];
        u_a[gid]   = u0[gid];
        v_a[gid]   = v0[gid];
    }
    if (gid < n_w) {
        w_a[gid] = w0[gid];
    }
}

kernel void crm_stage_hevi_factor(
    device const float* dl      [[buffer(0)]],
    device float* diag_inv      [[buffer(1)]],
    device float* du_cstar      [[buffer(2)]],
    constant uint& n_int        [[buffer(3)]],
    constant uint& M            [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= M || n_int == 0u) return;
    float inv_b0 = 1.0f / diag_inv[tid];
    float c_prev = (n_int > 1u) ? (du_cstar[tid] * inv_b0) : 0.0f;
    diag_inv[tid] = inv_b0;
    du_cstar[tid] = c_prev;
    for (uint um = 1u; um < n_int; ++um) {
        uint idx = um * M + tid;
        float inv_den = 1.0f / (diag_inv[idx] - dl[idx] * c_prev);
        c_prev = (um + 1u < n_int) ? (du_cstar[idx] * inv_den) : 0.0f;
        diag_inv[idx] = inv_den;
        du_cstar[idx] = c_prev;
    }
}

kernel void crm_stage_thermo(
    device const float* rho_s     [[buffer(0)]],
    device const float* Thm_s     [[buffer(1)]],
    device const float* q_load    [[buffer(2)]],
    device float* thm_s           [[buffer(3)]],
    device float* a_p_s           [[buffer(4)]],   // pres_s - dp_dThm_s * Thm_s
    device float* dp_dThm_s       [[buffer(5)]],
    device float* rho_tot_s       [[buffer(6)]],
    constant uint& total_pts      [[buffer(7)]],
    constant float4& f_p          [[buffer(8)]],   // [p_ref, r_d, gamma_d, 0]
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_pts) return;
    float p_ref = f_p.x;
    float r_d = f_p.y;
    float gamma_d = f_p.z;

    float r = rho_s[gid];
    float T = Thm_s[gid];
    thm_s[gid] = T / max(r, 1.0e-5f);
    float arg = max((r_d * T) / p_ref, 1.0e-7f);
    float p = p_ref * pow(arg, gamma_d);
    float dp = gamma_d * p / max(T, 1.0e-3f);
    dp_dThm_s[gid] = dp;
    a_p_s[gid] = p - dp * T;
    rho_tot_s[gid] = r * q_load[gid];
}

kernel void crm_stage_uv_flux(
    device const float* rho_s      [[buffer(0)]],
    device const float* rho_tot_s  [[buffer(1)]],
    device const float* thm_s      [[buffer(2)]],
    device const float* u_s        [[buffer(3)]],
    device const float* v_s        [[buffer(4)]],
    device const float* w_s        [[buffer(5)]],
    device const float* du_sgs     [[buffer(6)]],
    device const float* dv_sgs     [[buffer(7)]],
    device const float* u_mean     [[buffer(8)]],   // [nz, n_col]
    device const float* v_mean     [[buffer(9)]],   // [nz, n_col]
    device const float* dz_w_int   [[buffer(10)]],  // [nz-1, dz_cols]
    device const float* sponge_c   [[buffer(11)]],  // [nz]
    device const float* f_cor      [[buffer(12)]],  // [n_col] or [1]
    device float* rho_x            [[buffer(13)]],
    device float* rho_y            [[buffer(14)]],
    device float* rtot_x_inv       [[buffer(15)]],
    device float* rtot_y_inv       [[buffer(16)]],
    device float* thm_x_eff        [[buffer(17)]],
    device float* thm_y_eff        [[buffer(18)]],
    device float* tend_u_slow      [[buffer(19)]],
    device float* tend_v_slow      [[buffer(20)]],
    constant ulong4& u_dim         [[buffer(21)]],  // [nz, n_col, ny, nx]
    constant ulong4& u_flg         [[buffer(22)]],  // [dz_cols, has_cor, k_sp0, 0]
    constant float4& f_p           [[buffer(23)]],  // [dx, dy, 0, 0]
    uint gid [[thread_position_in_grid]])
{
    uint nz = uint(u_dim.x);
    uint n_col = uint(u_dim.y);
    uint ny = uint(u_dim.z);
    uint nx = uint(u_dim.w);
    uint dz_cols = uint(u_flg.x);
    uint has_cor = uint(u_flg.y);
    uint k_sp0 = uint(u_flg.z);
    uint stride_c = ny * nx;
    uint M = n_col * stride_c;
    if (gid >= nz * M) return;

    uint k = gid / M;
    uint tid = gid - k * M;
    uint col = tid / stride_c;
    uint rem = tid - col * stride_c;
    uint iy = rem / nx;
    uint ix = rem - iy * nx;
    uint col_dz = (dz_cols > 1u) ? col : 0u;

    uint ix_m2 = (ix >= 2u) ? (ix - 2u) : ((nx == 1u) ? 0u : (ix + nx - 2u));
    uint ix_m1 = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint ix_p1 = (ix + 1u < nx) ? (ix + 1u) : 0u;
    uint ix_p2 = (ix + 2u < nx) ? (ix + 2u) : ((nx == 1u) ? 0u : (ix + 2u - nx));

    uint iy_m2 = (iy >= 2u) ? (iy - 2u) : ((ny == 1u) ? 0u : (iy + ny - 2u));
    uint iy_m1 = (iy > 0u) ? (iy - 1u) : (ny - 1u);
    uint iy_p1 = (iy + 1u < ny) ? (iy + 1u) : 0u;
    uint iy_p2 = (iy + 2u < ny) ? (iy + 2u) : ((ny == 1u) ? 0u : (iy + 2u - ny));

    uint base_k = k * M + col * stride_c;
    uint idx_xm2 = base_k + iy * nx + ix_m2;
    uint idx_xm1 = base_k + iy * nx + ix_m1;
    uint idx_xp1 = base_k + iy * nx + ix_p1;
    uint idx_xp2 = base_k + iy * nx + ix_p2;
    uint idx_ym2 = base_k + iy_m2 * nx + ix;
    uint idx_ym1 = base_k + iy_m1 * nx + ix;
    uint idx_yp1 = base_k + iy_p1 * nx + ix;
    uint idx_yp2 = base_k + iy_p2 * nx + ix;
    uint idx_yp1_xm1 = base_k + iy_p1 * nx + ix_m1;
    uint idx_ym1_xp1 = base_k + iy_m1 * nx + ix_p1;

    float inv_dx = 1.0f / f_p.x;
    float inv_dy = 1.0f / f_p.y;

    rho_x[gid] = 0.5f * (rho_s[gid] + rho_s[idx_xp1]);
    rho_y[gid] = 0.5f * (rho_s[gid] + rho_s[idx_yp1]);
    rtot_x_inv[gid] = 1.0f / max(0.5f * (rho_tot_s[gid] + rho_tot_s[idx_xp1]), 1.0e-5f);
    rtot_y_inv[gid] = 1.0f / max(0.5f * (rho_tot_s[gid] + rho_tot_s[idx_yp1]), 1.0e-5f);

    float th0 = thm_s[gid];
    float th_xm1 = thm_s[idx_xm1];
    float th_xp1 = thm_s[idx_xp1];
    float th_xp2 = thm_s[idx_xp2];
    float th_ym1 = thm_s[idx_ym1];
    float th_yp1 = thm_s[idx_yp1];
    float th_yp2 = thm_s[idx_yp2];

    float u0 = u_s[gid];
    float v0 = v_s[gid];
    float sgn_u = (u0 > 0.0f) ? 1.0f : ((u0 < 0.0f) ? -1.0f : 0.0f);
    float sgn_v = (v0 > 0.0f) ? 1.0f : ((v0 < 0.0f) ? -1.0f : 0.0f);

    thm_x_eff[gid] = (
        (-th_xm1 + 7.0f * th0 + 7.0f * th_xp1 - th_xp2)
        + 0.25f * sgn_u * (th_xp2 - 3.0f * th_xp1 + 3.0f * th0 - th_xm1)
    ) / 12.0f;
    thm_y_eff[gid] = (
        (-th_ym1 + 7.0f * th0 + 7.0f * th_yp1 - th_yp2)
        + 0.25f * sgn_v * (th_yp2 - 3.0f * th_yp1 + 3.0f * th0 - th_ym1)
    ) / 12.0f;

    float u_xm2 = u_s[idx_xm2];
    float u_xm1 = u_s[idx_xm1];
    float u_xp1 = u_s[idx_xp1];
    float u_xp2 = u_s[idx_xp2];
    float u_ym2 = u_s[idx_ym2];
    float u_ym1 = u_s[idx_ym1];
    float u_yp1 = u_s[idx_yp1];
    float u_yp2 = u_s[idx_yp2];

    float v_xm2 = v_s[idx_xm2];
    float v_xm1 = v_s[idx_xm1];
    float v_xp1 = v_s[idx_xp1];
    float v_xp2 = v_s[idx_xp2];
    float v_ym2 = v_s[idx_ym2];
    float v_ym1 = v_s[idx_ym1];
    float v_yp1 = v_s[idx_yp1];
    float v_yp2 = v_s[idx_yp2];

    float u_c_left  = 0.5f * (u0 + u_xm1);
    float u_c_right = 0.5f * (u0 + u_xp1);
    float v_u_bot   = 0.5f * (v_ym1 + v_s[idx_ym1_xp1]);
    float v_u_top   = 0.5f * (v0 + v_xp1);

    float u_v_left  = 0.5f * (u_xm1 + u_s[idx_yp1_xm1]);
    float u_v_right = 0.5f * (u0 + u_yp1);
    float v_c_bot   = 0.5f * (v0 + v_ym1);
    float v_c_top   = 0.5f * (v0 + v_yp1);

    float adv_u =
        -(u_c_right * (ws3_face(u_xm1, u0, u_xp1, u_xp2, u_c_right) - u0)
          - u_c_left * (ws3_face(u_xm2, u_xm1, u0, u_xp1, u_c_left) - u0)) * inv_dx
        - (v_u_top * (ws3_face(u_ym1, u0, u_yp1, u_yp2, v_u_top) - u0)
           - v_u_bot * (ws3_face(u_ym2, u_ym1, u0, u_yp1, v_u_bot) - u0)) * inv_dy;

    float adv_v =
        -(u_v_right * (ws3_face(v_xm1, v0, v_xp1, v_xp2, u_v_right) - v0)
          - u_v_left * (ws3_face(v_xm2, v_xm1, v0, v_xp1, u_v_left) - v0)) * inv_dx
        - (v_c_top * (ws3_face(v_ym1, v0, v_yp1, v_yp2, v_c_top) - v0)
           - v_c_bot * (ws3_face(v_ym2, v_ym1, v0, v_yp1, v_c_bot) - v0)) * inv_dy;

    float v_at_u = 0.5f * (v_u_bot + v_u_top);
    float u_at_v = 0.5f * (u_v_left + u_v_right);

    float vadv_u = 0.0f;
    float vadv_v = 0.0f;
    float dtau_v = f_p.z;
    if (k < nz - 1u) {
        float dzw = dz_w_int[k * dz_cols + col_dz];
        uint kp1 = (k + 1u) * M;
        float w_ux = 0.5f * (w_s[kp1 + tid] + w_s[kp1 + col * stride_c + iy * nx + ix_p1]);
        float w_vy = 0.5f * (w_s[kp1 + tid] + w_s[kp1 + col * stride_c + iy_p1 * nx + ix]);
        if (w_ux < 0.0f) {
            float lim = (dtau_v > 0.0f) ? max(1.0f, (-w_ux) * (dtau_v / 0.45f) / dzw) : 1.0f;
            vadv_u += (-w_ux / lim) * (u_s[kp1 + tid] - u0) / dzw;
        }
        if (w_vy < 0.0f) {
            float lim = (dtau_v > 0.0f) ? max(1.0f, (-w_vy) * (dtau_v / 0.45f) / dzw) : 1.0f;
            vadv_v += (-w_vy / lim) * (v_s[kp1 + tid] - v0) / dzw;
        }
    }
    if (k > 0u) {
        float dzw = dz_w_int[(k - 1u) * dz_cols + col_dz];
        uint km1 = (k - 1u) * M;
        float w_ux = 0.5f * (w_s[k * M + tid] + w_s[k * M + col * stride_c + iy * nx + ix_p1]);
        float w_vy = 0.5f * (w_s[k * M + tid] + w_s[k * M + col * stride_c + iy_p1 * nx + ix]);
        if (w_ux > 0.0f) {
            float lim = (dtau_v > 0.0f) ? max(1.0f, w_ux * (dtau_v / 0.45f) / dzw) : 1.0f;
            vadv_u += (-w_ux / lim) * (u0 - u_s[km1 + tid]) / dzw;
        }
        if (w_vy > 0.0f) {
            float lim = (dtau_v > 0.0f) ? max(1.0f, w_vy * (dtau_v / 0.45f) / dzw) : 1.0f;
            vadv_v += (-w_vy / lim) * (v0 - v_s[km1 + tid]) / dzw;
        }
    }

    float fc = (has_cor != 0u) ? f_cor[col] : 0.0f;
    float sp = sponge_c[k];
    float um_k = 0.0f;
    float vm_k = 0.0f;
    if ((has_cor != 0u || sp > 0.0f) && k >= k_sp0) {
        uint sp_idx = (k - k_sp0) * n_col + col;
        um_k = u_mean[sp_idx];
        vm_k = v_mean[sp_idx];
    }
    float damp_u = (sp > 0.0f) ? (sp * (u0 - um_k)) : 0.0f;
    float damp_v = (sp > 0.0f) ? (sp * (v0 - vm_k)) : 0.0f;
    float cor_u = (has_cor != 0u) ? (fc * (v_at_u - vm_k)) : 0.0f;
    float cor_v = (has_cor != 0u) ? (-fc * (u_at_v - um_k)) : 0.0f;

    tend_u_slow[gid] = adv_u + vadv_u + cor_u + du_sgs[gid] - damp_u;
    tend_v_slow[gid] = adv_v + vadv_v + cor_v + dv_sgs[gid] - damp_v;
}

kernel void crm_stage_w_hevi(
    device const float* rho_s      [[buffer(0)]],
    device const float* rho_tot_s  [[buffer(1)]],
    device const float* thm_s      [[buffer(2)]],
    device const float* dp_dThm_s  [[buffer(3)]],
    device const float* q_load     [[buffer(4)]],
    device const float* u_s        [[buffer(5)]],
    device const float* v_s        [[buffer(6)]],
    device const float* w_s        [[buffer(7)]],
    device const float* dw_sgs     [[buffer(8)]],
    device const float* dz_c       [[buffer(9)]],   // [nz, dz_cols]
    device const float* dz_w_int   [[buffer(10)]],  // [nz-1, dz_cols]
    device const float* sponge_w   [[buffer(11)]],  // [nz-1]
    device float* tend_w_slow      [[buffer(12)]],
    device float* rho_dw           [[buffer(13)]],
    device float* rho_tot_w        [[buffer(14)]],
    device float* thm_w            [[buffer(15)]],
    device float* dl               [[buffer(16)]],
    device float* diag             [[buffer(17)]],
    device float* du               [[buffer(18)]],
    constant ulong4& u_dim         [[buffer(19)]],  // [nz, n_col, ny, nx]
    constant uint& dz_cols         [[buffer(20)]],
    constant float4& f_p0          [[buffer(21)]],  // [dt_ac, beta_hevi, gravity, dx]
    constant float4& f_p1          [[buffer(22)]],  // [dy, dtau, 0, 0]
    uint gid [[thread_position_in_grid]])
{
    uint nz = uint(u_dim.x);
    uint n_col = uint(u_dim.y);
    uint ny = uint(u_dim.z);
    uint nx = uint(u_dim.w);
    uint stride_c = ny * nx;
    uint M = n_col * stride_c;
    uint n_int = nz - 1u;
    if (gid >= n_int * M) return;

    uint m = gid / M;
    uint tid = gid - m * M;
    uint col = tid / stride_c;
    uint rem = tid - col * stride_c;
    uint iy = rem / nx;
    uint ix = rem - iy * nx;
    uint col_dz = (dz_cols > 1u) ? col : 0u;

    uint ix_m2 = (ix >= 2u) ? (ix - 2u) : ((nx == 1u) ? 0u : (ix + nx - 2u));
    uint ix_m1 = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint ix_p1 = (ix + 1u < nx) ? (ix + 1u) : 0u;
    uint ix_p2 = (ix + 2u < nx) ? (ix + 2u) : ((nx == 1u) ? 0u : (ix + 2u - nx));
    uint iy_m2 = (iy >= 2u) ? (iy - 2u) : ((ny == 1u) ? 0u : (iy + ny - 2u));
    uint iy_m1 = (iy > 0u) ? (iy - 1u) : (ny - 1u);
    uint iy_p1 = (iy + 1u < ny) ? (iy + 1u) : 0u;
    uint iy_p2 = (iy + 2u < ny) ? (iy + 2u) : ((ny == 1u) ? 0u : (iy + 2u - ny));

    uint tid_xm2 = col * stride_c + iy * nx + ix_m2;
    uint tid_xm1 = col * stride_c + iy * nx + ix_m1;
    uint tid_xp1 = col * stride_c + iy * nx + ix_p1;
    uint tid_xp2 = col * stride_c + iy * nx + ix_p2;
    uint tid_ym2 = col * stride_c + iy_m2 * nx + ix;
    uint tid_ym1 = col * stride_c + iy_m1 * nx + ix;
    uint tid_yp1 = col * stride_c + iy_p1 * nx + ix;
    uint tid_yp2 = col * stride_c + iy_p2 * nx + ix;

    uint idx_m = m * M + tid;
    uint idx_mp1 = (m + 1u) * M + tid;
    uint idx_mp2 = (m + 2u) * M + tid;

    float inv_dx = 1.0f / f_p0.w;
    float inv_dy = 1.0f / f_p1.x;
    float dtau_w = f_p1.y;

    float u_w_left  = 0.5f * (u_s[m * M + tid_xm1] + u_s[(m + 1u) * M + tid_xm1]);
    float u_w_right = 0.5f * (u_s[idx_m]           + u_s[idx_mp1]);
    float v_w_bot   = 0.5f * (v_s[m * M + tid_ym1] + v_s[(m + 1u) * M + tid_ym1]);
    float v_w_top   = 0.5f * (v_s[idx_m]           + v_s[idx_mp1]);

    float w_int = w_s[idx_mp1];
    uint wm1_base = (m + 1u) * M;
    float w_xm2 = w_s[wm1_base + tid_xm2];
    float w_xm1 = w_s[wm1_base + tid_xm1];
    float w_xp1 = w_s[wm1_base + tid_xp1];
    float w_xp2 = w_s[wm1_base + tid_xp2];
    float w_ym2 = w_s[wm1_base + tid_ym2];
    float w_ym1 = w_s[wm1_base + tid_ym1];
    float w_yp1 = w_s[wm1_base + tid_yp1];
    float w_yp2 = w_s[wm1_base + tid_yp2];

    float adv_w_h =
        -(u_w_right * (ws3_face(w_xm1, w_int, w_xp1, w_xp2, u_w_right) - w_int)
          - u_w_left * (ws3_face(w_xm2, w_xm1, w_int, w_xp1, u_w_left) - w_int)) * inv_dx
        - (v_w_top * (ws3_face(w_ym1, w_int, w_yp1, w_yp2, v_w_top) - w_int)
           - v_w_bot * (ws3_face(w_ym2, w_ym1, w_int, w_yp1, v_w_bot) - w_int)) * inv_dy;

    float dz_m = dz_c[m * dz_cols + col_dz];
    float dz_mp1 = dz_c[(m + 1u) * dz_cols + col_dz];
    float dz_wm = dz_w_int[m * dz_cols + col_dz];

    float dw_dz_dn = (w_int - w_s[idx_m]) / dz_m;
    float dw_dz_up = (w_s[idx_mp2] - w_int) / dz_mp1;
    float w_pos = max(w_int, 0.0f);
    float w_neg = min(w_int, 0.0f);
    float lim_pos = (dtau_w > 0.0f) ? max(1.0f, w_pos * (dtau_w / 0.50f) / dz_m) : 1.0f;
    float lim_neg = (dtau_w > 0.0f) ? max(1.0f, (-w_neg) * (dtau_w / 0.50f) / dz_mp1) : 1.0f;
    float adv_w_v = -((w_pos / lim_pos) * dw_dz_dn + (w_neg / lim_neg) * dw_dz_up);
    tend_w_slow[gid] = adv_w_h + adv_w_v + dw_sgs[idx_mp1];

    float rdw = 0.5f * (rho_s[idx_m] + rho_s[idx_mp1]);
    float rtw = 0.5f * (rho_tot_s[idx_m] + rho_tot_s[idx_mp1]);
    float thw = 0.5f * (thm_s[idx_m] + thm_s[idx_mp1]);
    rho_dw[gid] = rdw;
    rho_tot_w[gid] = rtw;
    thm_w[gid] = thw;

    float dt_ac = f_p0.x;
    float beta = f_p0.y;
    float gravity = f_p0.z;
    float bdt2 = (beta * dt_ac) * (beta * dt_ac);

    float fac_p = bdt2 * (rdw / (rtw * dz_wm));
    float fac_g = bdt2 * (0.5f * gravity * (rdw / rtw));
    float c_dn = dp_dThm_s[idx_m] / dz_m;
    float c_up = dp_dThm_s[idx_mp1] / dz_mp1;
    float q_m = q_load[idx_m];
    float q_mp1 = q_load[idx_mp1];

    diag[gid] = max(
        1.0f + dt_ac * sponge_w[m] + fac_p * thw * (c_up + c_dn) + fac_g * (q_mp1 / dz_mp1 - q_m / dz_m),
        0.5f
    );
    if (m > 0u) {
        float thw_prev = 0.5f * (thm_s[(m - 1u) * M + tid] + thm_s[idx_m]);
        dl[gid] = -fac_p * c_dn * thw_prev + fac_g * (q_m / dz_m);
    } else {
        dl[gid] = 0.0f;
    }
    if (m + 1u < n_int) {
        float thw_next = 0.5f * (thm_s[idx_mp1] + thm_s[idx_mp2]);
        du[gid] = -fac_p * c_up * thw_next - fac_g * (q_mp1 / dz_mp1);
    } else {
        du[gid] = 0.0f;
    }
}
"""


def _all_shader_src() -> str:
    return _METAL_THOMAS_SRC + EXTRA_METAL_SHADER_SRC + PHYSICS_METAL_SHADER_SRC + ESM_METAL_SHADER_SRC


def _get_metal_lib():
    global _METAL_SHADER_LIB
    if _METAL_SHADER_LIB is None and hasattr(torch.mps, "compile_shader"):
        _METAL_SHADER_LIB = torch.mps.compile_shader(_all_shader_src())
    return _METAL_SHADER_LIB


def _get_gpu_lib(device: torch.device | str | None = None):
    """Return the active fused GPU kernel library for `device` (`mps` Metal or `cuda` CUDA)."""
    dev = torch.device(device) if device is not None else None
    if dev is None:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            return None
    if dev.type == "mps":
        return _get_metal_lib()
    if dev.type == "cuda":
        return get_cuda_lib(_all_shader_src())
    return None


def _first_tensor_device(*args: Any, **kwargs: Any) -> torch.device | None:
    for a in args:
        if isinstance(a, torch.Tensor):
            return a.device
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return None


def get_active_gpu_backend(device: torch.device | str | None = None) -> str:
    """Return active fused kernel backend name: `'cuda'`, `'mps'`, or `'pytorch'`."""
    dev = torch.device(device) if device is not None else None
    if dev is None:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            return "pytorch"
    if dev.type == "cuda" and _get_gpu_lib(dev) is not None:
        return "cuda"
    if dev.type == "mps" and _get_gpu_lib(dev) is not None:
        return "mps"
    return "pytorch"


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

    use_gpu = (
        rhs.device.type in ("mps", "cuda")
        and rhs.dtype == torch.float32
        and K <= _MAX_METAL_K
        and not (
            dl.requires_grad or d.requires_grad or du.requires_grad or rhs.requires_grad
        )
    )
    if use_gpu:
        lib = _get_gpu_lib(rhs.device)
        if lib is not None:
            M = rhs[0].numel()
            if M > 0:
                dl_c = dl.contiguous().view(K, M)
                d_c = d.contiguous().view(K, M)
                du_c = du.contiguous().view(K, M)
                rhs_c = rhs.contiguous().view(K, M)
                out = torch.empty_like(rhs_c)
                lib.batched_thomas(dl_c, d_c, du_c, rhs_c, out, int(K), int(M), threads=M)
                return out.view(orig_shape)

    return solve_tridiagonal_pytorch(dl, d, du, rhs)


def can_use_crm_metal(rho0: torch.Tensor) -> bool:
    """Return True when fused CRM/SCVT GPU kernels (Metal on MPS or CUDA on NVIDIA) can run."""
    nz = rho0.shape[0]
    return (
        rho0.device.type in ("mps", "cuda")
        and rho0.dtype == torch.float32
        and 2 <= nz <= _MAX_COL_METAL_K
        and not rho0.requires_grad
        and _get_gpu_lib(rho0.device) is not None
    )


def can_use_elem_metal(t: torch.Tensor) -> bool:
    """Return True when single-level/elementwise GPU kernels (Metal or CUDA) can run."""
    return (
        t.device.type in ("mps", "cuda")
        and t.dtype == torch.float32
        and t.numel() > 0
        and not t.requires_grad
        and _get_gpu_lib(t.device) is not None
    )


can_use_crm_gpu = can_use_crm_metal
can_use_elem_gpu = can_use_elem_metal


def dispatch_crm_fct_advect_metal(
    q_batch: torch.Tensor,
    rho_old: torch.Tensor,
    rho_new: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    dx: float,
    dy: float,
    dz_c: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    lib = _get_gpu_lib(q_batch.device)
    assert lib is not None
    return run_crm_fct_advect_metal(lib, q_batch, rho_old, rho_new, fx, fy, fz, dx, dy, dz_c, dt)


def dispatch_crm_fct_pair_metal(
    q0: torch.Tensor,
    q1: torch.Tensor,
    rho_old: torch.Tensor,
    rho_new: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    dx: float,
    dy: float,
    dz_c: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(q0.device)
    assert lib is not None
    return run_crm_fct_pair_metal(lib, q0, q1, rho_old, rho_new, fx, fy, fz, dx, dy, dz_c, dt)


def dispatch_crm_fct_quad_metal(
    q0: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    q3: torch.Tensor,
    rho_old: torch.Tensor,
    rho_new: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    dx: float,
    dy: float,
    dz_c: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(q0.device)
    assert lib is not None
    return run_crm_fct_quad_metal(lib, q0, q1, q2, q3, rho_old, rho_new, fx, fy, fz, dx, dy, dz_c, dt)


def dispatch_crm_sgs_metal(
    rho_d: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    theta_m: torch.Tensor,
    dx: float,
    dy: float,
    dz_c: torch.Tensor,
    dz_w_int: torch.Tensor,
    gravity: float,
    c_s: float = 0.18,
    nu_max: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(rho_d.device)
    assert lib is not None
    return run_crm_sgs_metal(lib, rho_d, u, v, w, theta_m, dx, dy, dz_c, dz_w_int, gravity, c_s, nu_max=nu_max)


def dispatch_crm_microphysics_metal(
    rho_d: torch.Tensor,
    theta_m_c: torch.Tensor,
    q_n: torch.Tensor,
    q_p: torch.Tensor,
    q_v_prev: torch.Tensor,
    dz_c: torch.Tensor,
    dt: float,
    mp_consts: list[float],
) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(rho_d.device)
    assert lib is not None
    return run_crm_microphysics_metal(lib, rho_d, theta_m_c, q_n, q_p, q_v_prev, dz_c, dt, mp_consts)


def dispatch_crm_microphysics_2m_metal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_crm_microphysics_2m_metal(lib, **kwargs)


def dispatch_rad_rrtmgp_metal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_rad_rrtmgp_metal(lib, **kwargs)


def dispatch_rad_ckd_hd_metal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_rad_ckd_hd_metal(lib, **kwargs)


def dispatch_microphysics_6class_metal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_microphysics_6class_metal(lib, **kwargs)


def dispatch_ozone_chem_metal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_ozone_chem_metal(lib, **kwargs)


def dispatch_dynamic_lake_step_metal(**kwargs: Any) -> None:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    run_dynamic_lake_step_metal(lib, **kwargs)


def dispatch_scvt_fast_acoustic_metal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_scvt_fast_acoustic_metal(lib, **kwargs)


def dispatch_crm_mynn_metal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_crm_mynn_metal(lib, **kwargs)


def dispatch_scvt_div_metal(mesh: Any, flux_edge: torch.Tensor) -> torch.Tensor:
    lib = _get_gpu_lib(flux_edge.device)
    assert lib is not None
    return run_scvt_div_metal(lib, mesh, flux_edge)


def dispatch_scvt_vorticity_metal(mesh: Any, u_edge: torch.Tensor) -> torch.Tensor:
    lib = _get_gpu_lib(u_edge.device)
    assert lib is not None
    return run_scvt_vorticity_metal(lib, mesh, u_edge)


def dispatch_scvt_vec_lap_metal(mesh: Any, u_edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(u_edge.device)
    assert lib is not None
    return run_scvt_vec_lap_metal(lib, mesh, u_edge)


def dispatch_scvt_scalar_lap_metal(mesh: Any, phi_cell: torch.Tensor) -> torch.Tensor:
    lib = _get_gpu_lib(phi_cell.device)
    assert lib is not None
    return run_scvt_scalar_lap_metal(lib, mesh, phi_cell)


def dispatch_scvt_trisk_rot_metal(mesh: Any, zeta_v: torch.Tensor, u_edge: torch.Tensor) -> torch.Tensor:
    lib = _get_gpu_lib(u_edge.device)
    assert lib is not None
    return run_scvt_trisk_rot_metal(lib, mesh, zeta_v, u_edge)


def dispatch_scvt_storm_smag_metal(
    mesh: Any,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    q_n: torch.Tensor,
    theta_m: torch.Tensor,
    u_edge: torch.Tensor,
    w: torch.Tensor,
    dt: float,
    cs_scalar: float,
    cs_div: float,
    k_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(u_edge.device)
    assert lib is not None
    return run_scvt_storm_smag_metal(
        lib, mesh, rho_d, dz_c, q_n, theta_m, u_edge, w, dt, cs_scalar, cs_div, k_cap,
    )


def dispatch_scvt_fct_advect_metal(
    mesh: Any,
    q_batch: torch.Tensor,
    mass_old: torch.Tensor,
    mass_new: torch.Tensor,
    uh_edge: torch.Tensor,
    omega_z: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    lib = _get_gpu_lib(q_batch.device)
    assert lib is not None
    mc = _get_scvt_mesh_cache(mesh)
    return run_scvt_fct_advect_metal(lib, mesh, mc, q_batch, mass_old, mass_new, uh_edge, omega_z, dt)


def dispatch_dust_column_metal(**kwargs: Any) -> torch.Tensor:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_dust_column_metal(lib, **kwargs)


def dispatch_oro_gwd_metal(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_oro_gwd_metal(lib, **kwargs)


def dispatch_ngwd_propagate_metal(**kwargs: Any) -> torch.Tensor:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_ngwd_propagate_metal(lib, **kwargs)


def dispatch_rad_four_band_metal(
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_rad_four_band_metal(lib, **kwargs)


def dispatch_rad_ckd_metal(
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_rad_ckd_metal(lib, **kwargs)


def dispatch_surface_albedos_metal(**kwargs: Any) -> torch.Tensor:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_surface_albedos_metal(lib, **kwargs)


def dispatch_surface_transfer_metal(**kwargs: Any) -> torch.Tensor:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_surface_transfer_metal(lib, **kwargs)


def dispatch_two_leaf_gas_exchange_metal(**kwargs: Any) -> torch.Tensor:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_two_leaf_gas_exchange_metal(lib, **kwargs)


def dispatch_natural_aerosol_4sp_metal(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_natural_aerosol_4sp_metal(lib, **kwargs)


def dispatch_seaice_itd_sync_metal(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_seaice_itd_sync_metal(lib, **kwargs)


def dispatch_seaice_evp_subcycle_metal(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_seaice_evp_subcycle_metal(lib, **kwargs)


def dispatch_ogcm_barotropic_rk3_metal(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_ogcm_barotropic_rk3_metal(lib, **kwargs)


def dispatch_ogcm_fct_zstar_metal(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_ogcm_fct_zstar_metal(lib, **kwargs)


def dispatch_scvt_rk3_prep_metal(mesh: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    mc = _get_scvt_mesh_cache(mesh)
    return run_scvt_rk3_prep_metal(lib, mc, **kwargs)


def dispatch_ogcm_baroclinic_f_fixed_metal(mesh: Any, **kwargs: Any) -> torch.Tensor:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    mc = _get_scvt_mesh_cache(mesh)
    return run_ogcm_baroclinic_f_fixed_metal(lib, mc, **kwargs)


def dispatch_ogcm_baroclinic_rk3_update_metal(**kwargs: Any) -> torch.Tensor:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_ogcm_baroclinic_rk3_update_metal(lib, **kwargs)


def dispatch_ogcm_mom_vdiff_metal(mesh: Any, **kwargs: Any) -> torch.Tensor:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    mc = _get_scvt_mesh_cache(mesh)
    return run_ogcm_mom_vdiff_metal(lib, mc, **kwargs)


def dispatch_ogcm_tracer_vdiff_freeze_metal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_ogcm_tracer_vdiff_freeze_metal(lib, **kwargs)


def dispatch_ogcm_kpp_vdiff_metal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_ogcm_kpp_vdiff_metal(lib, **kwargs)


def dispatch_ogcm_pre_kpp_metal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_ogcm_pre_kpp_metal(lib, **kwargs)


def dispatch_ogcm_gm_bolus_edge_metal(mesh: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    mc = _get_scvt_mesh_cache(mesh)
    return run_ogcm_gm_bolus_edge_metal(lib, mc, **kwargs)


def dispatch_crm_student_tile_update_metal(**kwargs: Any) -> None:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    run_crm_student_tile_update_metal(lib, **kwargs)


def dispatch_surface_land_pre_bio_metal(**kwargs: Any) -> None:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    run_surface_land_pre_bio_metal(lib, **kwargs)


def dispatch_surface_land_newton_jac_metal(**kwargs: Any) -> None:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    run_surface_land_newton_jac_metal(lib, **kwargs)


def dispatch_surface_ocean_seaice_post_metal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_surface_ocean_seaice_post_metal(lib, **kwargs)


def dispatch_crm_distill_extract_metal(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_crm_distill_extract_metal(lib, **kwargs)


def dispatch_crm_distill_closure_metal(**kwargs: Any) -> dict[str, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_crm_distill_closure_metal(lib, **kwargs)


def dispatch_rad_distill_extract_metal(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_rad_distill_extract_metal(lib, **kwargs)


def dispatch_rad_distill_closure_metal(**kwargs: Any) -> dict[str, torch.Tensor]:
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_rad_distill_closure_metal(lib, **kwargs)


_WEIGHTED_EDGE_SUM_UP_CACHE: dict[tuple[str, int, int, int, int], torch.Tensor] = {}


def dispatch_scvt_weighted_edge_sum_metal(
    mesh: Any,
    edge_field: torch.Tensor,
    weights_cell: torch.Tensor,
) -> torch.Tensor:
    """Compute `(edge_field[:, eoc] * weights_cell).sum(dim=-1)` via fused `scvt_div_cell`."""
    lib = _get_gpu_lib(edge_field.device)
    assert lib is not None
    mc = _get_scvt_mesh_cache(mesh)
    nz = int(edge_field.shape[0])
    nc = int(mc["n_cells"])
    ne = int(mc["n_edges"])
    out = torch.empty((nz, nc), device=edge_field.device, dtype=edge_field.dtype)
    up_key = (edge_field.device.type, edge_field.device.index or 0, nz, nc, ne)
    up = _WEIGHTED_EDGE_SUM_UP_CACHE.get(up_key)
    if up is None:
        up = torch.tensor([nz, nc, ne], device=edge_field.device, dtype=torch.uint32)
        _WEIGHTED_EDGE_SUM_UP_CACHE[up_key] = up
    lib.scvt_div_cell(mc["eoc"], weights_cell.contiguous(), edge_field.contiguous(), out, up, threads=nc)
    return out


def dispatch_ocean_bgc_column_metal(**kwargs: Any):
    lib = _get_gpu_lib(_first_tensor_device(**kwargs))
    assert lib is not None
    return run_ocean_bgc_column_metal(lib, **kwargs)


def dispatch_scvt_reconstruct_winds_metal(mesh: Any, u_edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _get_gpu_lib(u_edge.device)
    assert lib is not None
    mc = _get_scvt_mesh_cache(mesh)
    return run_scvt_reconstruct_winds_metal(lib, mc, u_edge)


def dispatch_scvt_project_winds_metal(
    mesh: Any,
    vec_east_cell: torch.Tensor,
    vec_north_cell: torch.Tensor,
) -> torch.Tensor:
    lib = _get_gpu_lib(vec_east_cell.device)
    assert lib is not None
    mc = _get_scvt_mesh_cache(mesh)
    return run_scvt_project_winds_metal(lib, mc, vec_east_cell, vec_north_cell)


def run_crm_rk3_hevi_metal(
    dt_dyn: float,
    beta_hevi: float,
    dt_ac_max: float,
    p_ref: float,
    r_d: float,
    gamma_d: float,
    gravity: float,
    dx: float,
    dy: float,
    rho0: torch.Tensor,
    Thm0: torch.Tensor,
    u0: torch.Tensor,
    v0: torch.Tensor,
    w0: torch.Tensor,
    q_load: torch.Tensor | None,
    du_sgs: torch.Tensor,
    dv_sgs: torch.Tensor,
    dw_sgs: torch.Tensor,
    dThm_sgs: torch.Tensor,
    dz_c: torch.Tensor,
    dz_w_int: torch.Tensor,
    sponge_c: torch.Tensor,
    sponge_w: torch.Tensor,
    f_cor: torch.Tensor | None,
    q_n: torch.Tensor | None = None,
    q_p: torch.Tensor | None = None,
    clone_out: bool = True,
    n_sub_grp: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the full 3-stage Split-Explicit Wicker-Skamarock RK3-HEVI step on GPU (Metal or CUDA)."""
    lib = _get_gpu_lib(rho0.device)
    assert lib is not None

    nz, n_col, ny, nx = rho0.shape
    M = n_col * ny * nx
    total_pts = nz * M
    n_int_pts = (nz - 1) * M
    dev = rho0.device
    dz_cols = dz_c.shape[1]

    rho0_c = rho0.contiguous()
    Thm0_c = Thm0.contiguous()
    u0_c = u0.contiguous()
    v0_c = v0.contiguous()
    w0_c = w0.contiguous()
    du_s_c = du_sgs.contiguous()
    dv_s_c = dv_sgs.contiguous()
    dw_s_c = dw_sgs.contiguous()
    tTh_c = dThm_sgs.contiguous()
    dzc_c = dz_c.view(nz, dz_cols).contiguous()
    dzw_c = dz_w_int.view(nz - 1, dz_cols).contiguous()
    spc_c = sponge_c.view(nz).contiguous()
    spw_c = sponge_w.view(nz - 1).contiguous()

    has_cor = 1 if f_cor is not None else 0
    fcor_c = f_cor.view(n_col).contiguous() if f_cor is not None else torch.zeros(1, device=dev, dtype=torch.float32)

    ws_key = (dev.type, dev.index or 0, nz, n_col, ny, nx, dz_cols)
    ws = _CRM_WS_CACHE.get(ws_key)
    if ws is None:
        if len(_CRM_WS_CACHE) >= 8:
            _CRM_WS_CACHE.pop(next(iter(_CRM_WS_CACHE)))
        shape_w = (nz - 1, n_col, ny, nx)
        sp_nz = (spc_c > 0.0).nonzero(as_tuple=False)
        k_sp0 = int(sp_nz[0].item()) if sp_nz.numel() > 0 else nz
        st4_a = torch.empty((4, nz, n_col, ny, nx), device=dev, dtype=torch.float32)
        st4_b = torch.empty((4, nz, n_col, ny, nx), device=dev, dtype=torch.float32)
        st4_c = torch.empty((4, nz, n_col, ny, nx), device=dev, dtype=torch.float32)
        fxy_buf = torch.empty((2, nz, n_col, ny, nx), device=dev, dtype=torch.float32)
        fxy_acc = torch.empty((2, nz, n_col, ny, nx), device=dev, dtype=torch.float32)
        prep_c0 = torch.empty((4, nz, n_col, ny, nx), device=dev, dtype=torch.float32)
        prep_c1 = torch.empty((8, nz, n_col, ny, nx), device=dev, dtype=torch.float32)
        prep_w = torch.empty((7, nz - 1, n_col, ny, nx), device=dev, dtype=torch.float32)
        ws = {
            "st4_a": st4_a,
            "st4_b": st4_b,
            "st4_c": st4_c,
            "fxy_buf": fxy_buf,
            "fxy_acc": fxy_acc,
            "prep_c0": prep_c0,
            "prep_c1": prep_c1,
            "prep_w": prep_w,
            "rho_a": st4_a[0],
            "Thm_a": st4_a[1],
            "u_a": st4_a[2],
            "v_a": st4_a[3],
            "w_a": torch.empty_like(w0_c),
            "rho_b": st4_b[0],
            "Thm_b": st4_b[1],
            "u_b": st4_b[2],
            "v_b": st4_b[3],
            "w_b": torch.empty_like(w0_c),
            "rho_c": st4_c[0],
            "Thm_c": st4_c[1],
            "u_c": st4_c[2],
            "v_c": st4_c[3],
            "w_c": torch.empty_like(w0_c),
            "fx_buf": fxy_buf[0],
            "fy_buf": fxy_buf[1],
            "fx_acc": fxy_acc[0],
            "fy_acc": fxy_acc[1],
            "fz_acc": torch.empty_like(w0_c),
            "dummy": torch.zeros(1, device=dev, dtype=torch.float32),
            "thm_s": prep_c0[0],
            "a_p_s": prep_c0[1],
            "dp_dThm_s": prep_c0[2],
            "rho_tot_s": prep_c0[3],
            "rho_x": prep_c1[0],
            "rho_y": prep_c1[1],
            "rtot_x_inv": prep_c1[2],
            "rtot_y_inv": prep_c1[3],
            "thm_x_eff": prep_c1[4],
            "thm_y_eff": prep_c1[5],
            "tend_u_slow": prep_c1[6],
            "tend_v_slow": prep_c1[7],
            "tend_w_slow": prep_w[0],
            "rho_dw": prep_w[1],
            "rho_tot_w": prep_w[2],
            "thm_w": prep_w[3],
            "dl": prep_w[4],
            "diag": prep_w[5],
            "du": prep_w[6],
            "k_sp0": k_sp0,
        }
        _CRM_WS_CACHE[ws_key] = ws

    rho_a, Thm_a, u_a, v_a, w_a = ws["rho_a"], ws["Thm_a"], ws["u_a"], ws["v_a"], ws["w_a"]
    rho_b, Thm_b, u_b, v_b, w_b = ws["rho_b"], ws["Thm_b"], ws["u_b"], ws["v_b"], ws["w_b"]
    rho_c_ws, Thm_c_ws, u_c_ws, v_c_ws, w_c_ws = ws["rho_c"], ws["Thm_c"], ws["u_c"], ws["v_c"], ws["w_c"]
    fx_buf, fy_buf = ws["fx_buf"], ws["fy_buf"]
    fx_acc, fy_acc, fz_acc = ws["fx_acc"], ws["fy_acc"], ws["fz_acc"]
    dummy = ws["dummy"]

    use_simd_2d = (ny == 1 and 1 < nx <= 32 and nz <= 64 and (32 % nx == 0) and (f_cor is None))
    if use_simd_2d:
        c_tg = 64 // nx
        n_tg = (n_col + c_tg - 1) // c_tg
        grid_64 = (n_tg * 64,)
        # Ping-pong 3 banks so rho0_c..w0_c, scratch bank2, and rho_out..w_out never alias
        ptr0 = rho0_c.data_ptr()
        if ptr0 == rho_a.data_ptr():
            rho_out, Thm_out, u_out, v_out, w_out = rho_b, Thm_b, u_b, v_b, w_b
            rho_b2, Thm_b2, u_b2, v_b2, w_b2 = rho_c_ws, Thm_c_ws, u_c_ws, v_c_ws, w_c_ws
        elif ptr0 == rho_b.data_ptr():
            rho_out, Thm_out, u_out, v_out, w_out = rho_a, Thm_a, u_a, v_a, w_a
            rho_b2, Thm_b2, u_b2, v_b2, w_b2 = rho_c_ws, Thm_c_ws, u_c_ws, v_c_ws, w_c_ws
        else:
            rho_out, Thm_out, u_out, v_out, w_out = rho_a, Thm_a, u_a, v_a, w_a
            rho_b2, Thm_b2, u_b2, v_b2, w_b2 = rho_b, Thm_b, u_b, v_b, w_b

        qn_c = q_n.contiguous() if q_n is not None else (q_load - 1.0).contiguous()
        qp_c = q_p.contiguous() if q_p is not None else ws.setdefault("fy_zero", torch.zeros_like(rho0_c))
        if "fy_zero" not in ws:
            ws["fy_zero"] = torch.zeros_like(rho0_c)
        f_prep0 = [float(p_ref), float(r_d), float(gamma_d), float(dx)]

        import math as _math
        rk_taus = (dt_dyn / 3.0, 0.5 * dt_dyn, dt_dyn)

        n_ac0 = max(1, int(_math.ceil(rk_taus[0] / max(dt_ac_max, 0.1))))
        n_ac1 = max(1, int(_math.ceil(rk_taus[1] / max(dt_ac_max, 0.1))))
        n_ac2 = max(1, int(_math.ceil(rk_taus[2] / max(dt_ac_max, 0.1))))
        if dev.type == "mps" and n_ac0 <= 63 and n_ac1 <= 63 and n_ac2 <= 63:
            # Full-occupancy single-launch multi-substep 3-stage WS-RK3 + HEVI kernel
            # with direct 3-bank ping-pong, 1-float4 Backward read, and cached inv_dz_c/inv_dz_w.
            if "stage_buf_bc" not in ws:
                ws["stage_buf_bc"] = torch.empty((n_tg, nz, 4, 64, 4), device=dev, dtype=torch.float32)
            stage_buf = ws["stage_buf_bc"]
            dz_ptr = dzc_c.data_ptr()
            if ws.get("inv_dz_ptr") != dz_ptr or ws.get("inv_dz_ver") != getattr(dzc_c, "_version", 0):
                ws["inv_dz_c"] = (1.0 / dzc_c).contiguous()
                ws["inv_dz_w"] = (1.0 / dzw_c).contiguous()
                ws["inv_dz_ptr"] = dz_ptr
                ws["inv_dz_ver"] = getattr(dzc_c, "_version", 0)
            u_fused0 = [nz, n_col, nx, (int(dz_cols) << 18) | (n_ac0 << 12) | (n_ac1 << 6) | n_ac2]
            f_fused1 = [float(dt_dyn), float(beta_hevi), float(gravity), float(max(int(n_sub_grp), 1))]
            lib.crm_2d_rk3_stage_fused_simd(
                rho0_c, Thm0_c, u0_c, v0_c, w0_c,
                rho_out, Thm_out, u_out, v_out, w_out,
                rho_b2, Thm_b2, u_b2, v_b2, w_b2,
                qn_c, qp_c, du_s_c, dv_s_c, dw_s_c, tTh_c,
                ws["inv_dz_c"], ws["inv_dz_w"], spc_c, spw_c,
                stage_buf, fx_acc, fz_acc,
                u_fused0, f_prep0, f_fused1,
                threads=grid_64, group_size=(64,),
            )
            return rho_out, Thm_out, u_out, v_out, w_out, fx_acc, ws["fy_zero"], fz_acc

        # NVIDIA CUDA (A100/H100) achieves higher SM occupancy with the low-register 2-kernel
        # split (`d_star[64]` only, avoiding per-thread local-memory spill of 6x[64] arrays).
        if "stage_c0" not in ws:
            ws["stage_c0"] = torch.empty((nz, M, 4), device=dev, dtype=torch.float32)
            ws["stage_c1"] = torch.empty((nz, M, 4), device=dev, dtype=torch.float32)
            ws["stage_w0"] = torch.empty((nz - 1, M, 4), device=dev, dtype=torch.float32)
            ws["stage_w1"] = torch.empty((nz - 1, M, 4), device=dev, dtype=torch.float32)
        stage_c0, stage_c1, stage_w0, stage_w1 = (
            ws["stage_c0"], ws["stage_c1"], ws["stage_w0"], ws["stage_w1"]
        )
        u_prep = [nz, n_col, nx, dz_cols]
        rho_s, Thm_s, u_s, v_s, w_s = rho0_c, Thm0_c, u0_c, v0_c, w0_c
        for stage_idx, dtau in enumerate(rk_taus):
            n_ac = max(1, int(_math.ceil(dtau / max(dt_ac_max, 0.1))))
            dt_ac = dtau / float(n_ac)
            f_prep1 = [float(dt_ac), float(beta_hevi), float(gravity), float(dtau)]
            lib.crm_2d_stage_prep_simd(
                rho_s, Thm_s, u_s, v_s, w_s,
                qn_c, qp_c, du_s_c, dv_s_c, dw_s_c, dzc_c, dzw_c, spc_c, spw_c,
                stage_c0, stage_c1, stage_w0, stage_w1,
                u_prep, f_prep0, f_prep1,
                threads=grid_64, group_size=(64,),
            )
            accum_i = 1 if stage_idx == 2 else 0
            u_fast = [nz, n_col, nx, (int(dz_cols) << 16) | (int(n_ac) << 1) | accum_i]
            f_fast = [float(dt_ac), float(beta_hevi), float(gravity), float(dx)]
            lib.crm_2d_fast_hevi_simd(
                rho0_c, Thm0_c, u0_c, v0_c, w0_c,
                rho_out, Thm_out, u_out, v_out, w_out,
                stage_c0, stage_c1, stage_w0, stage_w1,
                tTh_c, dzc_c, dzw_c, spw_c,
                fx_acc, fz_acc, u_fast, f_fast,
                threads=grid_64, group_size=(64,),
            )
            rho_s, Thm_s, u_s, v_s, w_s = rho_out, Thm_out, u_out, v_out, w_out
        return rho_s, Thm_s, u_s, v_s, w_s, fx_acc, ws["fy_zero"], fz_acc

    ql_c = q_load.contiguous() if q_load is not None else (1.0 + q_n + q_p).contiguous()

    # Select two scratch state banks disjoint from rho0_c (so ac==0 reads rho0_c directly with zero copy
    # and the final output is also disjoint from rho0_c, enabling clone_out=False across RK3 substeps).
    ptr0 = rho0_c.data_ptr()
    bank_a = (rho_a, Thm_a, u_a, v_a, w_a)
    bank_b = (rho_b, Thm_b, u_b, v_b, w_b)
    bank_c = (rho_c_ws, Thm_c_ws, u_c_ws, v_c_ws, w_c_ws)
    if ptr0 == rho_a.data_ptr():
        pp0, pp1 = bank_b, bank_c
        st4_pp0 = ws["st4_b"]
    elif ptr0 == rho_b.data_ptr():
        pp0, pp1 = bank_a, bank_c
        st4_pp0 = ws["st4_a"]
    else:
        pp0, pp1 = bank_a, bank_b
        st4_pp0 = ws["st4_a"]

    thm_s, a_p_s, dp_dThm_s, rho_tot_s = (
        ws["thm_s"], ws["a_p_s"], ws["dp_dThm_s"], ws["rho_tot_s"]
    )
    rho_x, rho_y, rtot_x_inv, rtot_y_inv = ws["rho_x"], ws["rho_y"], ws["rtot_x_inv"], ws["rtot_y_inv"]
    thm_x_eff, thm_y_eff, tend_u_slow, tend_v_slow = (
        ws["thm_x_eff"], ws["thm_y_eff"], ws["tend_u_slow"], ws["tend_v_slow"]
    )
    tend_w_slow, rho_dw, rho_tot_w, thm_w, dl, diag, du = (
        ws["tend_w_slow"], ws["rho_dw"], ws["rho_tot_w"], ws["thm_w"], ws["dl"], ws["diag"], ws["du"]
    )
    k_sp0 = 0 if has_cor else int(ws["k_sp0"])
    u_dim4 = [nz, n_col, ny, nx]
    f_th4 = [float(p_ref), float(r_d), float(gamma_d), 0.0]
    u_suv_flg = [dz_cols, has_cor, k_sp0, 0]
    f_suv4 = [float(dx), float(dy), 0.0, 0.0]
    n_w_pts = (nz + 1) * M
    n_int = nz - 1

    use_2d_tg_loop = (ny == 1 and 1 < nx <= 128 and nz <= 64 and (128 % nx == 0))
    use_3d_tg_loop = (ny > 1 and nx > 1 and (ny * nx) <= 64 and nz <= 64)
    if use_2d_tg_loop:
        fx_acc.zero_()
        fy_acc.zero_()
        fz_acc.zero_()

    rho_s, Thm_s, u_s, v_s, w_s = rho0_c, Thm0_c, u0_c, v0_c, w0_c
    fx_final = dummy
    fy_final = dummy
    fz_final = dummy

    import math as _math
    rk_taus = (dt_dyn / 3.0, 0.5 * dt_dyn, dt_dyn)
    for stage_idx, dtau in enumerate(rk_taus):
        n_ac = max(1, int(_math.ceil(dtau / max(dt_ac_max, 0.1))))
        dt_ac = dtau / float(n_ac)
        inv_nac = 1.0 / float(n_ac)

        lib.crm_stage_thermo(rho_s, Thm_s, ql_c, thm_s, a_p_s, dp_dThm_s, rho_tot_s, total_pts, f_th4, threads=total_pts)
        if k_sp0 < nz:
            u_mean = mean_2d(u_s[k_sp0:]).contiguous()
            v_mean = mean_2d(v_s[k_sp0:]).contiguous()
        else:
            u_mean = dummy
            v_mean = dummy
        f_suv4_s = [float(dx), float(dy), float(dtau), 0.0]
        lib.crm_stage_uv_flux(
            rho_s, rho_tot_s, thm_s, u_s, v_s, w_s, du_s_c, dv_s_c, u_mean, v_mean,
            dzw_c, spc_c, fcor_c,
            rho_x, rho_y, rtot_x_inv, rtot_y_inv, thm_x_eff, thm_y_eff, tend_u_slow, tend_v_slow,
            u_dim4, u_suv_flg, f_suv4_s, threads=total_pts,
        )
        f_hevi4 = [float(dt_ac), float(beta_hevi), float(gravity), float(dx)]
        f_w_p1 = [float(dy), float(dtau), 0.0, 0.0]
        lib.crm_stage_w_hevi(
            rho_s, rho_tot_s, thm_s, dp_dThm_s, ql_c, u_s, v_s, w_s, dw_s_c,
            dzc_c, dzw_c, spw_c,
            tend_w_slow, rho_dw, rho_tot_w, thm_w, dl, diag, du,
            u_dim4, int(dz_cols), f_hevi4, f_w_p1, threads=n_int_pts,
        )

        accum = stage_idx == 2
        lib.crm_stage_hevi_factor(dl, diag, du, int(n_int), int(M), threads=M)
        if use_3d_tg_loop:
            lib.crm_copy_state5(
                rho0_c, Thm0_c, u0_c, v0_c, w0_c,
                pp0[0], pp0[1], pp0[2], pp0[3], pp0[4],
                total_pts, n_w_pts, threads=n_w_pts,
            )
            accum_i = 1 if accum else 0
            u_pack = [nz, n_col, (ny << 16) | nx, (int(dz_cols) << 16) | (int(n_ac) << 1) | accum_i]
            f_p1 = [float(dy), float(inv_nac), 0.0, 0.0]
            grp = ny * nx
            lib.crm_3d_fast_hevi_loop(
                st4_pp0, pp0[4], ws["fxy_buf"],
                ws["prep_c0"], ws["prep_c1"], ws["prep_w"],
                ql_c, tTh_c, dzc_c, dzw_c, spw_c,
                ws["fxy_acc"], fz_acc, u_pack, f_hevi4, f_p1,
                threads=(n_col * grp * 4,), group_size=(grp * 4,),
            )
            rho_s, Thm_s, u_s, v_s, w_s = pp0
            if accum:
                fx_final, fy_final, fz_final = fx_acc, fy_acc, fz_acc
        elif use_2d_tg_loop:
            accum_i = 1 if accum else 0
            lib.crm_copy_state5(
                rho0_c, Thm0_c, u0_c, v0_c, w0_c,
                pp0[0], pp0[1], pp0[2], pp0[3], pp0[4],
                total_pts, n_w_pts, threads=n_w_pts,
            )
            u_2d4 = [nz, n_col, nx, (int(dz_cols) << 16) | (int(n_ac) << 1) | accum_i]
            f_2d4 = [float(dt_ac), float(beta_hevi), float(gravity), float(dx)]
            if nx <= 64 and nz <= 48 and (64 % nx == 0):
                c_tg = 64 // nx
                n_tg = (n_col + c_tg - 1) // c_tg
                kern_2d = lib.crm_2d_fast_hevi_loop if nz <= 32 else lib.crm_2d_fast_hevi_loop_64
                grp = 64
            else:
                c_tg = 128 // nx
                n_tg = (n_col + c_tg - 1) // c_tg
                kern_2d = lib.crm_2d_fast_hevi_loop_tg128
                grp = 128
            kern_2d(
                pp0[0], pp0[1], pp0[2], pp0[3], pp0[4],
                rho_x, rtot_x_inv, thm_x_eff, a_p_s, dp_dThm_s, ql_c,
                tend_u_slow, tend_v_slow, tend_w_slow, tTh_c,
                rho_dw, rho_tot_w, thm_w, dl, diag, du, dzc_c, dzw_c, spw_c,
                fx_acc, fy_acc, fz_acc, u_2d4, f_2d4,
                threads=(n_tg * grp,), group_size=(grp,),
            )
            rho_s, Thm_s, u_s, v_s, w_s = pp0
            if accum:
                fx_final = fx_acc * inv_nac
                fy_final = fy_acc * inv_nac
                fz_final = fz_acc * inv_nac
        else:
            fx_out = fx_acc if accum else dummy
            fy_out = fy_acc if accum else dummy
            fz_out = fz_acc if accum else dummy

            f_uv4 = [float(dt_ac), float(dx), float(dy), float(inv_nac)]
            f_hevi_p1 = [float(dy), float(inv_nac), 0.0, 0.0]

            # Substep 0 reads (rho0_c..w0_c) directly -> pp0, avoiding crm_copy_state5 altogether!
            cur = (rho0_c, Thm0_c, u0_c, v0_c, w0_c)
            nxt = pp0
            alt = pp1

            for ac_i in range(n_ac):
                acc_mode = 0 if not accum else (1 if ac_i == 0 else 2)
                u_flg4 = [int(dz_cols), acc_mode, 0, 0]
                rho_c, Thm_c, u_c, v_c, w_c = cur
                rho_n, Thm_n, u_n, v_n, w_n = nxt
                lib.crm_fast_uv_flux(
                    rho_x, rho_y, rtot_x_inv, rtot_y_inv, a_p_s, dp_dThm_s,
                    tend_u_slow, tend_v_slow, Thm_c, u_c, v_c,
                    u_n, v_n, fx_buf, fy_buf, fx_out, fy_out,
                    u_dim4, acc_mode, f_uv4, threads=total_pts,
                )
                lib.crm_fast_hevi_col(
                    rho_c, Thm_c, w_c, fx_buf, fy_buf, thm_x_eff, thm_y_eff,
                    a_p_s, dp_dThm_s, ql_c, tend_w_slow, tTh_c,
                    rho_dw, rho_tot_w, thm_w, dl, diag, du, dzc_c, dzw_c, spw_c,
                    rho_n, Thm_n, w_n, fz_out,
                    u_dim4, u_flg4, f_hevi4, f_hevi_p1, threads=M,
                )
                if ac_i == 0:
                    cur, nxt = pp0, pp1
                else:
                    cur, nxt = nxt, cur

            rho_s, Thm_s, u_s, v_s, w_s = cur
            if accum:
                fx_final = fx_acc
                fy_final = fy_acc
                fz_final = fz_acc

    if clone_out:
        return rho_s.clone(), Thm_s.clone(), u_s.clone(), v_s.clone(), w_s.clone(), fx_final, fy_final, fz_final
    return rho_s, Thm_s, u_s, v_s, w_s, fx_final, fy_final, fz_final
