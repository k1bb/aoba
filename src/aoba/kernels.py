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

kernel void crm_fast_uv_flux(
    device const float* rho_x        [[buffer(0)]],
    device const float* rho_y        [[buffer(1)]],
    device const float* rtot_x_inv   [[buffer(2)]],
    device const float* rtot_y_inv   [[buffer(3)]],
    device const float* pres_s       [[buffer(4)]],
    device const float* dp_dThm_s    [[buffer(5)]],
    device const float* Thm_s        [[buffer(6)]],
    device const float* tend_u_slow  [[buffer(7)]],
    device const float* tend_v_slow  [[buffer(8)]],
    device const float* Thm_cur      [[buffer(9)]],
    device const float* u_cur        [[buffer(10)]],
    device const float* v_cur        [[buffer(11)]],
    device float* u_nxt              [[buffer(12)]],
    device float* v_nxt              [[buffer(13)]],
    device float* fx_buf             [[buffer(14)]],
    device float* fy_buf             [[buffer(15)]],
    device float* fx_acc             [[buffer(16)]],
    device float* fy_acc             [[buffer(17)]],
    device const uint* u_params      [[buffer(18)]],  // [nz, n_col, ny, nx, accum_flux, 0]
    device const float* f_params     [[buffer(19)]],  // [dt_ac, dx, dy]
    uint gid [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_col = u_params[1];
    uint ny = u_params[2];
    uint nx = u_params[3];
    uint accum = u_params[4];
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

    float p_self = pres_s[gid] + dp_dThm_s[gid] * (Thm_cur[gid] - Thm_s[gid]);
    float p_xp   = pres_s[idx_xp] + dp_dThm_s[idx_xp] * (Thm_cur[idx_xp] - Thm_s[idx_xp]);
    float p_yp   = pres_s[idx_yp] + dp_dThm_s[idx_yp] * (Thm_cur[idx_yp] - Thm_s[idx_yp]);

    float dt_ac = f_params[0];
    float inv_dx = 1.0f / f_params[1];
    float inv_dy = 1.0f / f_params[2];

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
    if (accum != 0u) {
        fx_acc[gid] += fx_v;
        fy_acc[gid] += fy_v;
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
    device const float* pres_s        [[buffer(7)]],
    device const float* dp_dThm_s     [[buffer(8)]],
    device const float* Thm_s         [[buffer(9)]],
    device const float* q_load        [[buffer(10)]],
    device const float* tend_w_slow   [[buffer(11)]],
    device const float* tend_Thm_slow [[buffer(12)]],
    device const float* rho_dw        [[buffer(13)]],
    device const float* rho_tot_w     [[buffer(14)]],
    device const float* thm_w         [[buffer(15)]],
    device const float* dl            [[buffer(16)]],
    device const float* diag          [[buffer(17)]],
    device const float* du            [[buffer(18)]],
    device const float* dz_c          [[buffer(19)]],
    device const float* dz_w_int      [[buffer(20)]],
    device const float* sponge_w      [[buffer(21)]],
    device float* rho_nxt             [[buffer(22)]],
    device float* Thm_nxt             [[buffer(23)]],
    device float* w_nxt               [[buffer(24)]],
    device float* fz_acc              [[buffer(25)]],
    device const uint* u_params       [[buffer(26)]],  // [nz, n_col, ny, nx, dz_cols, accum_flux]
    device const float* f_params      [[buffer(27)]],  // [dt_ac, beta_hevi, gravity, dx, dy]
    uint tid [[thread_position_in_grid]])
{
    uint K = u_params[0];
    uint n_col = u_params[1];
    uint ny = u_params[2];
    uint nx = u_params[3];
    uint dz_cols = u_params[4];
    uint accum = u_params[5];
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

    float dt_ac = f_params[0];
    float beta = f_params[1];
    float gravity = f_params[2];
    float inv_dx = 1.0f / f_params[3];
    float inv_dy = 1.0f / f_params[4];

    float c_star[128];
    float d_star[128];

    float p_blend_prev = 0.0f;
    float rtot_blend_prev = 0.0f;
    float fz_old_below = 0.0f;
    float thm_w_below = 0.0f;
    float c_prev = 0.0f;
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
        if (k < K - 1u) {
            fz_old_above = rho_dw[idx] * w_cur[(k + 1u) * M + tid];
            thm_w_above = thm_w[idx];
        }

        float dz_val = dz_c[k * dz_cols + col_dz];
        float r_cur = rho_cur[idx];
        float T_cur = Thm_cur[idx];
        float r_dag = r_cur - dt_ac * (div_h_rho + (fz_old_above - fz_old_below) / dz_val);
        float T_dag = T_cur + dt_ac * (
            -div_h_Thm
            - (thm_w_above * fz_old_above - thm_w_below * fz_old_below) / dz_val
            + tend_Thm_slow[idx]
        );
        rho_nxt[idx] = r_dag;
        Thm_nxt[idx] = T_dag;

        float ps = pres_s[idx];
        float dp = dp_dThm_s[idx];
        float Ts = Thm_s[idx];
        float p_cur = ps + dp * (T_cur - Ts);
        float p_dag = ps + dp * (T_dag - Ts);
        float p_blend = (1.0f - beta) * p_cur + beta * p_dag;
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
            if (um == 0u) {
                float inv_b0 = 1.0f / diag[idx_m];
                c_prev = du[idx_m] * inv_b0;
                d_prev = rhs_m * inv_b0;
            } else {
                float inv_den = 1.0f / (diag[idx_m] - dl[idx_m] * c_prev);
                c_prev = (um < K - 2u) ? (du[idx_m] * inv_den) : 0.0f;
                d_prev = (rhs_m - dl[idx_m] * d_prev) * inv_den;
            }
            c_star[um] = c_prev;
            d_star[um] = d_prev;
        }
        p_blend_prev = p_blend;
        rtot_blend_prev = rtot_blend;
        fz_old_below = fz_old_above;
        thm_w_below = thm_w_above;
    }

    w_nxt[tid] = 0.0f;
    w_nxt[K * M + tid] = 0.0f;

    float dfz_above = 0.0f;
    float thm_w_above = 0.0f;
    for (int m = int(K) - 2; m >= 0; --m) {
        uint um = uint(m);
        uint idx_m = um * M + tid;
        float dfz_cur = (m == int(K) - 2) ? d_star[um] : (d_star[um] - c_star[um] * dfz_above);
        float rdw_m = rho_dw[idx_m];
        float fz_old_m = rdw_m * w_cur[(um + 1u) * M + tid];
        float fz_new_m = fz_old_m + dfz_cur;
        w_nxt[(um + 1u) * M + tid] = fz_new_m / max(rdw_m, 1.0e-5f);
        if (accum != 0u) {
            fz_acc[(um + 1u) * M + tid] += (fz_old_m + beta * dfz_cur);
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

kernel void crm_stage_thermo(
    device const float* rho_s     [[buffer(0)]],
    device const float* Thm_s     [[buffer(1)]],
    device const float* q_load    [[buffer(2)]],
    device float* thm_s           [[buffer(3)]],
    device float* pres_s          [[buffer(4)]],
    device float* dp_dThm_s       [[buffer(5)]],
    device float* rho_tot_s       [[buffer(6)]],
    device const uint* u_params   [[buffer(7)]],  // [total_pts]
    device const float* f_params  [[buffer(8)]],  // [p_ref, r_d, gamma_d]
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u_params[0]) return;
    float p_ref = f_params[0];
    float r_d = f_params[1];
    float gamma_d = f_params[2];

    float r = rho_s[gid];
    float T = Thm_s[gid];
    thm_s[gid] = T / max(r, 1.0e-5f);
    float arg = max((r_d * T) / p_ref, 1.0e-7f);
    float p = p_ref * pow(arg, gamma_d);
    pres_s[gid] = p;
    dp_dThm_s[gid] = gamma_d * p / max(T, 1.0e-3f);
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
    device const uint* u_params    [[buffer(21)]],  // [nz, n_col, ny, nx, dz_cols, has_cor]
    device const float* f_params   [[buffer(22)]],  // [dx, dy]
    uint gid [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_col = u_params[1];
    uint ny = u_params[2];
    uint nx = u_params[3];
    uint dz_cols = u_params[4];
    uint has_cor = u_params[5];
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

    uint ix_m1 = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint ix_p1 = (ix + 1u < nx) ? (ix + 1u) : 0u;
    uint ix_p2 = (ix + 2u < nx) ? (ix + 2u) : (ix + 2u - nx);
    if (nx == 1u) ix_p2 = 0u;

    uint iy_m1 = (iy > 0u) ? (iy - 1u) : (ny - 1u);
    uint iy_p1 = (iy + 1u < ny) ? (iy + 1u) : 0u;
    uint iy_p2 = (iy + 2u < ny) ? (iy + 2u) : (iy + 2u - ny);
    if (ny == 1u) iy_p2 = 0u;

    uint base_k = k * M + col * stride_c;
    uint idx_xm1 = base_k + iy * nx + ix_m1;
    uint idx_xp1 = base_k + iy * nx + ix_p1;
    uint idx_xp2 = base_k + iy * nx + ix_p2;
    uint idx_ym1 = base_k + iy_m1 * nx + ix;
    uint idx_yp1 = base_k + iy_p1 * nx + ix;
    uint idx_yp2 = base_k + iy_p2 * nx + ix;
    uint idx_yp1_xm1 = base_k + iy_p1 * nx + ix_m1;
    uint idx_ym1_xp1 = base_k + iy_m1 * nx + ix_p1;

    float inv_dx = 1.0f / f_params[0];
    float inv_dy = 1.0f / f_params[1];

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

    float uc_self = 0.5f * (u0 + u_s[idx_xm1]);
    float uc_yp1  = 0.5f * (u_s[idx_yp1] + u_s[idx_yp1_xm1]);
    float vc_self = 0.5f * (v0 + v_s[idx_ym1]);
    float vc_xp1  = 0.5f * (v_s[idx_xp1] + v_s[idx_ym1_xp1]);

    float v_at_u = 0.5f * (vc_self + vc_xp1);
    float u_at_v = 0.5f * (uc_self + uc_yp1);

    float adv_u = -u0 * (u_s[idx_xp1] - u_s[idx_xm1]) * (0.5f * inv_dx)
                  - v_at_u * (u_s[idx_yp1] - u_s[idx_ym1]) * (0.5f * inv_dy);
    float adv_v = -u_at_v * (v_s[idx_xp1] - v_s[idx_xm1]) * (0.5f * inv_dx)
                  - v0 * (v_s[idx_yp1] - v_s[idx_ym1]) * (0.5f * inv_dy);

    float du_dz = 0.0f;
    float dv_dz = 0.0f;
    if (k < nz - 1u) {
        float dzw = dz_w_int[k * dz_cols + col_dz];
        uint kp1 = (k + 1u) * M;
        float w_ux = 0.5f * (w_s[kp1 + tid] + w_s[kp1 + col * stride_c + iy * nx + ix_p1]);
        float w_vy = 0.5f * (w_s[kp1 + tid] + w_s[kp1 + col * stride_c + iy_p1 * nx + ix]);
        du_dz += 0.5f * w_ux * (u_s[kp1 + tid] - u0) / dzw;
        dv_dz += 0.5f * w_vy * (v_s[kp1 + tid] - v0) / dzw;
    }
    if (k > 0u) {
        float dzw = dz_w_int[(k - 1u) * dz_cols + col_dz];
        uint km1 = (k - 1u) * M;
        float w_ux = 0.5f * (w_s[k * M + tid] + w_s[k * M + col * stride_c + iy * nx + ix_p1]);
        float w_vy = 0.5f * (w_s[k * M + tid] + w_s[k * M + col * stride_c + iy_p1 * nx + ix]);
        du_dz += 0.5f * w_ux * (u0 - u_s[km1 + tid]) / dzw;
        dv_dz += 0.5f * w_vy * (v0 - v_s[km1 + tid]) / dzw;
    }

    float fc = (has_cor != 0u) ? f_cor[col] : 0.0f;
    float sp = sponge_c[k];
    float um = u_mean[k * n_col + col];
    float vm = v_mean[k * n_col + col];

    tend_u_slow[gid] = adv_u - du_dz + fc * v_at_u + du_sgs[gid] - sp * (u0 - um);
    tend_v_slow[gid] = adv_v - dv_dz - fc * u_at_v + dv_sgs[gid] - sp * (v0 - vm);
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
    device const uint* u_params    [[buffer(19)]],  // [nz, n_col, ny, nx, dz_cols, 0]
    device const float* f_params   [[buffer(20)]],  // [dt_ac, beta_hevi, gravity, dx, dy]
    uint gid [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_col = u_params[1];
    uint ny = u_params[2];
    uint nx = u_params[3];
    uint dz_cols = u_params[4];
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

    uint ix_m1 = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint ix_p1 = (ix + 1u < nx) ? (ix + 1u) : 0u;
    uint iy_m1 = (iy > 0u) ? (iy - 1u) : (ny - 1u);
    uint iy_p1 = (iy + 1u < ny) ? (iy + 1u) : 0u;

    uint tid_xm1 = col * stride_c + iy * nx + ix_m1;
    uint tid_xp1 = col * stride_c + iy * nx + ix_p1;
    uint tid_ym1 = col * stride_c + iy_m1 * nx + ix;
    uint tid_yp1 = col * stride_c + iy_p1 * nx + ix;

    uint idx_m = m * M + tid;
    uint idx_mp1 = (m + 1u) * M + tid;
    uint idx_mp2 = (m + 2u) * M + tid;

    float inv_dx = 1.0f / f_params[3];
    float inv_dy = 1.0f / f_params[4];

    float uc_m   = 0.5f * (u_s[m * M + tid] + u_s[m * M + tid_xm1]);
    float uc_mp1 = 0.5f * (u_s[(m + 1u) * M + tid] + u_s[(m + 1u) * M + tid_xm1]);
    float vc_m   = 0.5f * (v_s[m * M + tid] + v_s[m * M + tid_ym1]);
    float vc_mp1 = 0.5f * (v_s[(m + 1u) * M + tid] + v_s[(m + 1u) * M + tid_ym1]);
    float u_w = 0.5f * (uc_m + uc_mp1);
    float v_w = 0.5f * (vc_m + vc_mp1);

    float w_int = w_s[idx_mp1];
    float dw_dx = (w_s[(m + 1u) * M + tid_xp1] - w_s[(m + 1u) * M + tid_xm1]) * (0.5f * inv_dx);
    float dw_dy = (w_s[(m + 1u) * M + tid_yp1] - w_s[(m + 1u) * M + tid_ym1]) * (0.5f * inv_dy);

    float dz_m = dz_c[m * dz_cols + col_dz];
    float dz_mp1 = dz_c[(m + 1u) * dz_cols + col_dz];
    float dz_wm = dz_w_int[m * dz_cols + col_dz];

    float dw_dz_dn = (w_int - w_s[idx_m]) / dz_m;
    float dw_dz_up = (w_s[idx_mp2] - w_int) / dz_mp1;
    float adv_w = -u_w * dw_dx - v_w * dw_dy - w_int * 0.5f * (dw_dz_dn + dw_dz_up);
    tend_w_slow[gid] = adv_w + dw_sgs[idx_mp1];

    float rdw = 0.5f * (rho_s[idx_m] + rho_s[idx_mp1]);
    float rtw = 0.5f * (rho_tot_s[idx_m] + rho_tot_s[idx_mp1]);
    float thw = 0.5f * (thm_s[idx_m] + thm_s[idx_mp1]);
    rho_dw[gid] = rdw;
    rho_tot_w[gid] = rtw;
    thm_w[gid] = thw;

    float dt_ac = f_params[0];
    float beta = f_params[1];
    float gravity = f_params[2];
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


def can_use_crm_metal(rho0: torch.Tensor) -> bool:
    """Return True when the fused CRM acoustic+HEVI Metal shader can be dispatched."""
    nz = rho0.shape[0]
    return (
        rho0.device.type == "mps"
        and rho0.dtype == torch.float32
        and 2 <= nz <= _MAX_METAL_K
        and not rho0.requires_grad
        and _get_metal_lib() is not None
    )


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
    q_load: torch.Tensor,
    du_sgs: torch.Tensor,
    dv_sgs: torch.Tensor,
    dw_sgs: torch.Tensor,
    dThm_sgs: torch.Tensor,
    dz_c: torch.Tensor,
    dz_w_int: torch.Tensor,
    sponge_c: torch.Tensor,
    sponge_w: torch.Tensor,
    f_cor: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the full 3-stage Split-Explicit Wicker-Skamarock RK3-HEVI step on Metal GPU."""
    lib = _get_metal_lib()
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
    ql_c = q_load.contiguous()
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

    # Persistent single-allocation ping-pong & stage-prep buffers across all 3 RK3 stages
    rho_a = torch.empty_like(rho0_c)
    Thm_a = torch.empty_like(Thm0_c)
    u_a = torch.empty_like(u0_c)
    v_a = torch.empty_like(v0_c)
    w_a = torch.empty_like(w0_c)

    rho_b = torch.empty_like(rho0_c)
    Thm_b = torch.empty_like(Thm0_c)
    u_b = torch.empty_like(u0_c)
    v_b = torch.empty_like(v0_c)
    w_b = torch.empty_like(w0_c)

    fx_buf = torch.empty_like(rho0_c)
    fy_buf = torch.empty_like(rho0_c)
    fx_acc = torch.zeros_like(rho0_c)
    fy_acc = torch.zeros_like(rho0_c)
    fz_acc = torch.zeros_like(w0_c)
    dummy = torch.zeros(1, device=dev, dtype=torch.float32)

    thm_s = torch.empty_like(rho0_c)
    Thm_ref = torch.empty_like(rho0_c)
    pres_s = torch.empty_like(rho0_c)
    dp_dThm_s = torch.empty_like(rho0_c)
    rho_tot_s = torch.empty_like(rho0_c)

    rho_x = torch.empty_like(rho0_c)
    rho_y = torch.empty_like(rho0_c)
    rtot_x_inv = torch.empty_like(rho0_c)
    rtot_y_inv = torch.empty_like(rho0_c)
    thm_x_eff = torch.empty_like(rho0_c)
    thm_y_eff = torch.empty_like(rho0_c)
    tend_u_slow = torch.empty_like(rho0_c)
    tend_v_slow = torch.empty_like(rho0_c)

    shape_w = (nz - 1, n_col, ny, nx)
    tend_w_slow = torch.empty(shape_w, device=dev, dtype=torch.float32)
    rho_dw = torch.empty(shape_w, device=dev, dtype=torch.float32)
    rho_tot_w = torch.empty(shape_w, device=dev, dtype=torch.float32)
    thm_w = torch.empty(shape_w, device=dev, dtype=torch.float32)
    dl = torch.empty(shape_w, device=dev, dtype=torch.float32)
    diag = torch.empty(shape_w, device=dev, dtype=torch.float32)
    du = torch.empty(shape_w, device=dev, dtype=torch.float32)

    u_th = torch.tensor([total_pts], dtype=torch.uint32, device=dev)
    f_th = torch.tensor([p_ref, r_d, gamma_d], dtype=torch.float32, device=dev)
    u_suv = torch.tensor([nz, n_col, ny, nx, dz_cols, has_cor], dtype=torch.uint32, device=dev)
    f_suv = torch.tensor([dx, dy], dtype=torch.float32, device=dev)
    u_sw = torch.tensor([nz, n_col, ny, nx, dz_cols, 0], dtype=torch.uint32, device=dev)

    rho_s, Thm_s, u_s, v_s, w_s = rho0_c, Thm0_c, u0_c, v0_c, w0_c
    fx_final = torch.zeros_like(rho0_c)
    fy_final = torch.zeros_like(rho0_c)
    fz_final = torch.zeros_like(w0_c)

    import math as _math
    rk_taus = (dt_dyn / 3.0, 0.5 * dt_dyn, dt_dyn)
    for stage_idx, dtau in enumerate(rk_taus):
        n_ac = max(1, int(_math.ceil(dtau / max(dt_ac_max, 0.1))))
        dt_ac = dtau / float(n_ac)

        # Stage reference thermodynamics + Slow RHS + HEVI tridiagonal coefficients (3 fused kernels)
        lib.crm_stage_thermo(rho_s, Thm_s, ql_c, thm_s, pres_s, dp_dThm_s, rho_tot_s, u_th, f_th, threads=total_pts)
        u_mean = u_s.mean(dim=(-2, -1)).contiguous()
        v_mean = v_s.mean(dim=(-2, -1)).contiguous()
        lib.crm_stage_uv_flux(
            rho_s, rho_tot_s, thm_s, u_s, v_s, w_s, du_s_c, dv_s_c, u_mean, v_mean,
            dzw_c, spc_c, fcor_c,
            rho_x, rho_y, rtot_x_inv, rtot_y_inv, thm_x_eff, thm_y_eff, tend_u_slow, tend_v_slow,
            u_suv, f_suv, threads=total_pts,
        )
        f_hevi = torch.tensor([dt_ac, beta_hevi, gravity, dx, dy], dtype=torch.float32, device=dev)
        lib.crm_stage_w_hevi(
            rho_s, rho_tot_s, thm_s, dp_dThm_s, ql_c, u_s, v_s, w_s, dw_s_c,
            dzc_c, dzw_c, spw_c,
            tend_w_slow, rho_dw, rho_tot_w, thm_w, dl, diag, du,
            u_sw, f_hevi, threads=n_int_pts,
        )

        # Snapshot stage reference Thm_s BEFORE resetting ping-pong buffer A (in case Thm_s aliases Thm_a)
        Thm_ref.copy_(Thm_s)

        # Initialize ping-pong state A from t^n (U^0)
        rho_a.copy_(rho0_c)
        Thm_a.copy_(Thm0_c)
        u_a.copy_(u0_c)
        v_a.copy_(v0_c)
        w_a.copy_(w0_c)

        accum = stage_idx == 2
        accum_u = 1 if accum else 0
        fx_out = fx_acc if accum else dummy
        fy_out = fy_acc if accum else dummy
        fz_out = fz_acc if accum else dummy

        u_uv = torch.tensor([nz, n_col, ny, nx, accum_u, 0], dtype=torch.uint32, device=dev)
        f_uv = torch.tensor([dt_ac, dx, dy], dtype=torch.float32, device=dev)
        u_hevi = torch.tensor([nz, n_col, ny, nx, dz_cols, accum_u], dtype=torch.uint32, device=dev)

        cur = (rho_a, Thm_a, u_a, v_a, w_a)
        nxt = (rho_b, Thm_b, u_b, v_b, w_b)

        for _ in range(n_ac):
            rho_c, Thm_c, u_c, v_c, w_c = cur
            rho_n, Thm_n, u_n, v_n, w_n = nxt
            lib.crm_fast_uv_flux(
                rho_x, rho_y, rtot_x_inv, rtot_y_inv, pres_s, dp_dThm_s, Thm_ref,
                tend_u_slow, tend_v_slow, Thm_c, u_c, v_c,
                u_n, v_n, fx_buf, fy_buf, fx_out, fy_out,
                u_uv, f_uv, threads=total_pts,
            )
            lib.crm_fast_hevi_col(
                rho_c, Thm_c, w_c, fx_buf, fy_buf, thm_x_eff, thm_y_eff,
                pres_s, dp_dThm_s, Thm_ref, ql_c, tend_w_slow, tTh_c,
                rho_dw, rho_tot_w, thm_w, dl, diag, du, dzc_c, dzw_c, spw_c,
                rho_n, Thm_n, w_n, fz_out,
                u_hevi, f_hevi, threads=M,
            )
            cur, nxt = nxt, cur

        rho_s, Thm_s, u_s, v_s, w_s = cur
        if accum:
            inv_nac = 1.0 / float(n_ac)
            fx_final = fx_acc * inv_nac
            fy_final = fy_acc * inv_nac
            fz_final = fz_acc * inv_nac

    return rho_s, Thm_s, u_s, v_s, w_s, fx_final, fy_final, fz_final
