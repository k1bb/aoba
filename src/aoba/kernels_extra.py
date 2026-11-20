"""Extra Metal shaders and wrappers for AOBA GPU acceleration (FCT, SGS, microphysics, SCVT)."""

from __future__ import annotations

import math
from typing import Any

import torch

from aoba import constants as C

_MAX_WS_CACHE: int = 8
_FCT_WS_CACHE: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
_SGS_WS_CACHE: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
_SCVT_WS_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _cache_put(cache: dict[tuple[Any, ...], Any], key: tuple[Any, ...], val: Any) -> None:
    if len(cache) >= _MAX_WS_CACHE:
        cache.pop(next(iter(cache)))
    cache[key] = val


EXTRA_METAL_SHADER_SRC: str = r"""
// =========================================================================
// 5. Batched 3D Zalesak Flux-Corrected Transport (FCT) Kernels
// =========================================================================

kernel void crm_fct_flux_td(
    device const float* q          [[buffer(0)]],   // [B, nz, n_col, ny, nx]
    device const float* rho_old    [[buffer(1)]],   // [nz, n_col, ny, nx]
    device const float* rho_new    [[buffer(2)]],   // [nz, n_col, ny, nx]
    device const float* fx         [[buffer(3)]],   // [nz, n_col, ny, nx]
    device const float* fy         [[buffer(4)]],   // [nz, n_col, ny, nx]
    device const float* fz         [[buffer(5)]],   // [nz+1, n_col, ny, nx]
    device const float* dz_c       [[buffer(6)]],   // [nz, dz_cols]
    device float* ax_out           [[buffer(7)]],   // [B, nz, n_col, ny, nx]
    device float* ay_out           [[buffer(8)]],   // [B, nz, n_col, ny, nx]
    device float* az_out           [[buffer(9)]],   // [B, nz+1, n_col, ny, nx]
    device float* q_td             [[buffer(10)]],  // [B, nz, n_col, ny, nx]
    device const uint* u_params    [[buffer(11)]],  // [B, nz, n_col, ny, nx, dz_cols]
    device const float* f_params   [[buffer(12)]],  // [dt, dx, dy]
    uint gid [[thread_position_in_grid]])
{
    uint B = u_params[0];
    uint nz = u_params[1];
    uint n_col = u_params[2];
    uint ny = u_params[3];
    uint nx = u_params[4];
    uint dz_cols = u_params[5];

    uint stride_c = ny * nx;
    uint M = n_col * stride_c;
    uint vol_pts = nz * M;
    if (gid >= B * vol_pts) return;

    uint b = gid / vol_pts;
    uint loc = gid - b * vol_pts;
    uint k = loc / M;
    uint tid = loc - k * M;
    uint col = tid / stride_c;
    uint rem = tid - col * stride_c;
    uint iy = rem / nx;
    uint ix = rem - iy * nx;
    uint col_dz = (dz_cols > 1u) ? col : 0u;

    uint ix_m1 = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint ix_p1 = (ix + 1u < nx) ? (ix + 1u) : 0u;
    uint ix_p2 = (ix + 2u < nx) ? (ix + 2u) : ((nx == 1u) ? 0u : (ix + 2u - nx));
    uint iy_m1 = (iy > 0u) ? (iy - 1u) : (ny - 1u);
    uint iy_p1 = (iy + 1u < ny) ? (iy + 1u) : 0u;
    uint iy_p2 = (iy + 2u < ny) ? (iy + 2u) : ((ny == 1u) ? 0u : (iy + 2u - ny));

    uint q_k_base = b * vol_pts + k * M + col * stride_c;
    uint f_k_base = k * M + col * stride_c;

    float q0 = q[gid];
    float q_xm1 = q[q_k_base + iy * nx + ix_m1];
    float q_xp1 = q[q_k_base + iy * nx + ix_p1];
    float q_xp2 = q[q_k_base + iy * nx + ix_p2];
    float q_ym1 = q[q_k_base + iy_m1 * nx + ix];
    float q_yp1 = q[q_k_base + iy_p1 * nx + ix];
    float q_yp2 = q[q_k_base + iy_p2 * nx + ix];

    float fx_self = fx[loc];
    float fx_xm1  = fx[f_k_base + iy * nx + ix_m1];
    float fl_x_self = max(fx_self, 0.0f) * q0    + min(fx_self, 0.0f) * q_xp1;
    float fl_x_xm1  = max(fx_xm1,  0.0f) * q_xm1 + min(fx_xm1,  0.0f) * q0;
    float fh_x_self = fx_self * ((-q_xm1 + 7.0f * q0 + 7.0f * q_xp1 - q_xp2) * (1.0f / 12.0f));
    ax_out[gid] = fh_x_self - fl_x_self;

    float fy_self = fy[loc];
    float fy_ym1  = fy[f_k_base + iy_m1 * nx + ix];
    float fl_y_self = max(fy_self, 0.0f) * q0    + min(fy_self, 0.0f) * q_yp1;
    float fl_y_ym1  = max(fy_ym1,  0.0f) * q_ym1 + min(fy_ym1,  0.0f) * q0;
    float fh_y_self = fy_self * ((-q_ym1 + 7.0f * q0 + 7.0f * q_yp1 - q_yp2) * (1.0f / 12.0f));
    ay_out[gid] = fh_y_self - fl_y_self;

    uint az_base = b * (nz + 1u) * M;
    if (k == 0u) {
        az_out[az_base + tid] = 0.0f;
    }
    if (k == nz - 1u) {
        az_out[az_base + nz * M + tid] = 0.0f;
    }

    float fl_z_dn = 0.0f;
    float fl_z_up = 0.0f;
    if (k > 0u) {
        float fz_dn = fz[k * M + tid];
        float q_km1 = q[gid - M];
        fl_z_dn = max(fz_dn, 0.0f) * q_km1 + min(fz_dn, 0.0f) * q0;
    }
    if (k + 1u < nz) {
        uint m = k + 1u;
        float fz_up = fz[m * M + tid];
        float q_kp1 = q[gid + M];
        fl_z_up = max(fz_up, 0.0f) * q0 + min(fz_up, 0.0f) * q_kp1;

        float fh_z_up;
        if (nz >= 4u && m >= 2u && m <= nz - 2u) {
            float q_km1 = q[gid - M];
            float q_kp2 = q[gid + 2u * M];
            fh_z_up = fz_up * ((-q_km1 + 7.0f * q0 + 7.0f * q_kp1 - q_kp2) * (1.0f / 12.0f));
        } else {
            fh_z_up = fz_up * (0.5f * (q0 + q_kp1));
        }
        az_out[az_base + m * M + tid] = fh_z_up - fl_z_up;
    }

    float dt = f_params[0];
    float inv_dx = 1.0f / f_params[1];
    float inv_dy = 1.0f / f_params[2];
    float inv_dz = 1.0f / dz_c[k * dz_cols + col_dz];

    float div_fl = (fl_x_self - fl_x_xm1) * inv_dx
                 + (fl_y_self - fl_y_ym1) * inv_dy
                 + (fl_z_up - fl_z_dn) * inv_dz;
    float rho_safe = max(rho_new[loc], 1.0e-5f);
    q_td[gid] = max((rho_old[loc] * q0 - dt * div_fl) / rho_safe, 0.0f);
}

kernel void crm_fct_bounds_ratios(
    device const float* q          [[buffer(0)]],
    device const float* q_td       [[buffer(1)]],
    device const float* rho_new    [[buffer(2)]],
    device const float* ax         [[buffer(3)]],
    device const float* ay         [[buffer(4)]],
    device const float* az         [[buffer(5)]],
    device const float* dz_c       [[buffer(6)]],
    device float* r_plus           [[buffer(7)]],
    device float* r_minus          [[buffer(8)]],
    device const uint* u_params    [[buffer(9)]],
    device const float* f_params   [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    uint B = u_params[0];
    uint nz = u_params[1];
    uint n_col = u_params[2];
    uint ny = u_params[3];
    uint nx = u_params[4];
    uint dz_cols = u_params[5];

    uint stride_c = ny * nx;
    uint M = n_col * stride_c;
    uint vol_pts = nz * M;
    if (gid >= B * vol_pts) return;

    uint b = gid / vol_pts;
    uint loc = gid - b * vol_pts;
    uint k = loc / M;
    uint tid = loc - k * M;
    uint col = tid / stride_c;
    uint rem = tid - col * stride_c;
    uint iy = rem / nx;
    uint ix = rem - iy * nx;
    uint col_dz = (dz_cols > 1u) ? col : 0u;

    uint ix_m1 = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint ix_p1 = (ix + 1u < nx) ? (ix + 1u) : 0u;
    uint iy_m1 = (iy > 0u) ? (iy - 1u) : (ny - 1u);
    uint iy_p1 = (iy + 1u < ny) ? (iy + 1u) : 0u;
    uint km1 = (k > 0u) ? (k - 1u) : 0u;
    uint kp1 = (k + 1u < nz) ? (k + 1u) : (nz - 1u);

    uint q_base = b * vol_pts;
    uint base_k = q_base + k * M + col * stride_c;
    uint idx_xm1 = base_k + iy * nx + ix_m1;
    uint idx_xp1 = base_k + iy * nx + ix_p1;
    uint idx_ym1 = base_k + iy_m1 * nx + ix;
    uint idx_yp1 = base_k + iy_p1 * nx + ix;
    uint idx_km1 = q_base + km1 * M + tid;
    uint idx_kp1 = q_base + kp1 * M + tid;

    float q0 = q[gid];
    float qtd0 = q_td[gid];
    float q_max = max(q0, qtd0);
    float q_min = min(q0, qtd0);

    q_max = max(q_max, max(q[idx_xm1], q_td[idx_xm1]));
    q_max = max(q_max, max(q[idx_xp1], q_td[idx_xp1]));
    q_max = max(q_max, max(q[idx_ym1], q_td[idx_ym1]));
    q_max = max(q_max, max(q[idx_yp1], q_td[idx_yp1]));
    q_max = max(q_max, max(q[idx_km1], q_td[idx_km1]));
    q_max = max(q_max, max(q[idx_kp1], q_td[idx_kp1]));

    q_min = min(q_min, min(q[idx_xm1], q_td[idx_xm1]));
    q_min = min(q_min, min(q[idx_xp1], q_td[idx_xp1]));
    q_min = min(q_min, min(q[idx_ym1], q_td[idx_ym1]));
    q_min = min(q_min, min(q[idx_yp1], q_td[idx_yp1]));
    q_min = min(q_min, min(q[idx_km1], q_td[idx_km1]));
    q_min = min(q_min, min(q[idx_kp1], q_td[idx_kp1]));

    float ax_s = ax[gid];
    float ax_m = ax[idx_xm1];
    float ay_s = ay[gid];
    float ay_m = ay[idx_ym1];

    uint az_base = b * (nz + 1u) * M;
    float az_dn = az[az_base + k * M + tid];
    float az_up = az[az_base + (k + 1u) * M + tid];

    float dt = f_params[0];
    float inv_dx = 1.0f / f_params[1];
    float inv_dy = 1.0f / f_params[2];
    float inv_dz = 1.0f / dz_c[k * dz_cols + col_dz];

    float p_plus = dt * (
        (max(ax_m, 0.0f) - min(ax_s, 0.0f)) * inv_dx
        + (max(ay_m, 0.0f) - min(ay_s, 0.0f)) * inv_dy
        + (max(az_dn, 0.0f) - min(az_up, 0.0f)) * inv_dz
    );
    float p_minus = dt * (
        (max(ax_s, 0.0f) - min(ax_m, 0.0f)) * inv_dx
        + (max(ay_s, 0.0f) - min(ay_m, 0.0f)) * inv_dy
        + (max(az_up, 0.0f) - min(az_dn, 0.0f)) * inv_dz
    );

    float rho_safe = max(rho_new[loc], 1.0e-5f);
    float q_plus = rho_safe * max(q_max - qtd0, 0.0f);
    float q_minus = rho_safe * max(qtd0 - q_min, 0.0f);
    float eps = 1.0e-20f;

    r_plus[gid]  = (p_plus  > eps) ? clamp(q_plus  / (p_plus  + eps), 0.0f, 1.0f) : 1.0f;
    r_minus[gid] = (p_minus > eps) ? clamp(q_minus / (p_minus + eps), 0.0f, 1.0f) : 1.0f;
}

kernel void crm_fct_limit_apply(
    device const float* q_td       [[buffer(0)]],
    device const float* rho_new    [[buffer(1)]],
    device const float* ax         [[buffer(2)]],
    device const float* ay         [[buffer(3)]],
    device const float* az         [[buffer(4)]],
    device const float* r_plus     [[buffer(5)]],
    device const float* r_minus    [[buffer(6)]],
    device const float* dz_c       [[buffer(7)]],
    device float* q_out            [[buffer(8)]],
    device const uint* u_params    [[buffer(9)]],
    device const float* f_params   [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    uint B = u_params[0];
    uint nz = u_params[1];
    uint n_col = u_params[2];
    uint ny = u_params[3];
    uint nx = u_params[4];
    uint dz_cols = u_params[5];

    uint stride_c = ny * nx;
    uint M = n_col * stride_c;
    uint vol_pts = nz * M;
    if (gid >= B * vol_pts) return;

    uint b = gid / vol_pts;
    uint loc = gid - b * vol_pts;
    uint k = loc / M;
    uint tid = loc - k * M;
    uint col = tid / stride_c;
    uint rem = tid - col * stride_c;
    uint iy = rem / nx;
    uint ix = rem - iy * nx;
    uint col_dz = (dz_cols > 1u) ? col : 0u;

    uint ix_m1 = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint ix_p1 = (ix + 1u < nx) ? (ix + 1u) : 0u;
    uint iy_m1 = (iy > 0u) ? (iy - 1u) : (ny - 1u);
    uint iy_p1 = (iy + 1u < ny) ? (iy + 1u) : 0u;

    uint base_k = b * vol_pts + k * M + col * stride_c;
    uint idx_xm1 = base_k + iy * nx + ix_m1;
    uint idx_xp1 = base_k + iy * nx + ix_p1;
    uint idx_ym1 = base_k + iy_m1 * nx + ix;
    uint idx_yp1 = base_k + iy_p1 * nx + ix;

    float rp_0 = r_plus[gid];
    float rm_0 = r_minus[gid];

    float ax_s = ax[gid];
    float ax_m = ax[idx_xm1];
    float rp_xp1 = r_plus[idx_xp1];
    float rm_xp1 = r_minus[idx_xp1];
    float rp_xm1 = r_plus[idx_xm1];
    float rm_xm1 = r_minus[idx_xm1];
    float cx_s = (ax_s >= 0.0f) ? min(rm_0, rp_xp1) : min(rp_0, rm_xp1);
    float cx_m = (ax_m >= 0.0f) ? min(rm_xm1, rp_0) : min(rp_xm1, rm_0);

    float ay_s = ay[gid];
    float ay_m = ay[idx_ym1];
    float rp_yp1 = r_plus[idx_yp1];
    float rm_yp1 = r_minus[idx_yp1];
    float rp_ym1 = r_plus[idx_ym1];
    float rm_ym1 = r_minus[idx_ym1];
    float cy_s = (ay_s >= 0.0f) ? min(rm_0, rp_yp1) : min(rp_0, rm_yp1);
    float cy_m = (ay_m >= 0.0f) ? min(rm_ym1, rp_0) : min(rp_ym1, rm_0);

    uint az_base = b * (nz + 1u) * M;
    float az_dn_lim = 0.0f;
    float az_up_lim = 0.0f;
    if (k > 0u) {
        float az_dn = az[az_base + k * M + tid];
        float rp_km1 = r_plus[gid - M];
        float rm_km1 = r_minus[gid - M];
        float cz_dn = (az_dn >= 0.0f) ? min(rm_km1, rp_0) : min(rp_km1, rm_0);
        az_dn_lim = cz_dn * az_dn;
    }
    if (k + 1u < nz) {
        float az_up = az[az_base + (k + 1u) * M + tid];
        float rp_kp1 = r_plus[gid + M];
        float rm_kp1 = r_minus[gid + M];
        float cz_up = (az_up >= 0.0f) ? min(rm_0, rp_kp1) : min(rp_0, rm_kp1);
        az_up_lim = cz_up * az_up;
    }

    float dt = f_params[0];
    float inv_dx = 1.0f / f_params[1];
    float inv_dy = 1.0f / f_params[2];
    float inv_dz = 1.0f / dz_c[k * dz_cols + col_dz];

    float div_anti = (cx_s * ax_s - cx_m * ax_m) * inv_dx
                   + (cy_s * ay_s - cy_m * ay_m) * inv_dy
                   + (az_up_lim - az_dn_lim) * inv_dz;
    float rho_safe = max(rho_new[loc], 1.0e-5f);
    q_out[gid] = max(q_td[gid] - (dt / rho_safe) * div_anti, 0.0f);
}

// =========================================================================
// 6. 3D Smagorinsky-Lilly SGS Turbulence Kernels
// =========================================================================

inline float scotti_p2(float z) {
    return z * (0.98f + z * (0.073f + z * (-0.418f + 0.120f * z)));
}

inline float scotti_p3(float z) {
    return z * (0.976f + z * (0.188f + z * (1.169f + z * (0.755f - 0.151f * z))));
}

inline float scotti_p1(float z) {
    return 2.5f * scotti_p2(z) - 1.5f * pow(max(cos(z), 0.0f), 2.0f / 3.0f) * sin(z);
}

inline float scotti_aspect(float aspect) {
    float a = max(aspect, 1.0f);
    float b1 = atan(1.0f / a);
    float b2 = 1.57079632679f - b1;
    float bracket =
        4.0f * scotti_p1(b1) * pow(a, 1.0f / 3.0f)
        + 0.222f * scotti_p2(b1) * pow(a, -5.0f / 3.0f)
        + 0.077f * scotti_p3(b1) * pow(a, -11.0f / 3.0f)
        - 3.0f * b1
        + 4.0f * scotti_p1(b2)
        + 0.222f * scotti_p2(b2)
        + 0.077f * scotti_p3(b2)
        - 3.0f * b2;
    return 1.736f * pow(a, 1.0f / 3.0f) * pow(max(bracket, 0.1f), -0.75f);
}

kernel void crm_sgs_eddy_uv(
    device const float* u          [[buffer(0)]],
    device const float* v          [[buffer(1)]],
    device const float* w          [[buffer(2)]],
    device const float* theta_m    [[buffer(3)]],
    device const float* dz_c       [[buffer(4)]],
    device const float* dz_w_int   [[buffer(5)]],
    device float* nu_sgs_out       [[buffer(6)]],
    device float* d_sgs_out        [[buffer(7)]],
    device float* du_sgs_out       [[buffer(8)]],
    device float* dv_sgs_out       [[buffer(9)]],
    device const uint* u_params    [[buffer(10)]],  // [nz, n_col, ny, nx, dz_cols]
    device const float* f_params   [[buffer(11)]],  // [dx, dy, gravity, c_s]
    uint gid [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_col = u_params[1];
    uint ny = u_params[2];
    uint nx = u_params[3];
    uint dz_cols = u_params[4];
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
    uint iy_m1 = (iy > 0u) ? (iy - 1u) : (ny - 1u);
    uint iy_p1 = (iy + 1u < ny) ? (iy + 1u) : 0u;

    uint base_k = k * M + col * stride_c;
    uint idx_xm1 = base_k + iy * nx + ix_m1;
    uint idx_xp1 = base_k + iy * nx + ix_p1;
    uint idx_ym1 = base_k + iy_m1 * nx + ix;
    uint idx_yp1 = base_k + iy_p1 * nx + ix;
    uint idx_ym1_xm1 = base_k + iy_m1 * nx + ix_m1;
    uint idx_yp1_xm1 = base_k + iy_p1 * nx + ix_m1;
    uint idx_ym1_xp1 = base_k + iy_m1 * nx + ix_p1;

    float dx = f_params[0];
    float dy = f_params[1];
    float gravity = f_params[2];
    float c_s = f_params[3];
    float inv_dx = 1.0f / dx;
    float inv_dy = 1.0f / dy;
    float dz = dz_c[k * dz_cols + col_dz];

    float u0 = u[gid];
    float u_xm1 = u[idx_xm1];
    float u_xp1 = u[idx_xp1];
    float u_ym1 = u[idx_ym1];
    float u_yp1 = u[idx_yp1];

    float v0 = v[gid];
    float v_xm1 = v[idx_xm1];
    float v_xp1 = v[idx_xp1];
    float v_ym1 = v[idx_ym1];
    float v_yp1 = v[idx_yp1];

    float du_dx = (u0 - u_xm1) * inv_dx;
    float dv_dy = (v0 - v_ym1) * inv_dy;
    float dw_dz = (w[(k + 1u) * M + tid] - w[k * M + tid]) / dz;

    float uc_0   = 0.5f * (u0 + u_xm1);
    float uc_yp1 = 0.5f * (u_yp1 + u[idx_yp1_xm1]);
    float uc_ym1 = 0.5f * (u_ym1 + u[idx_ym1_xm1]);
    float du_dy = 0.5f * (uc_yp1 - uc_ym1) * inv_dy;

    float vc_0   = 0.5f * (v0 + v_ym1);
    float vc_xp1 = 0.5f * (v_xp1 + v[idx_ym1_xp1]);
    float vc_xm1 = 0.5f * (v_xm1 + v[idx_ym1_xm1]);
    float dv_dx = 0.5f * (vc_xp1 - vc_xm1) * inv_dx;

    float du_dz = 0.0f;
    float dv_dz = 0.0f;
    float n2 = 0.0f;
    float th0 = theta_m[gid];

    if (k + 1u < nz) {
        float dzw = dz_w_int[k * dz_cols + col_dz];
        uint kp1_base = (k + 1u) * M + col * stride_c;
        float uc_kp1 = 0.5f * (u[kp1_base + iy * nx + ix] + u[kp1_base + iy * nx + ix_m1]);
        float vc_kp1 = 0.5f * (v[kp1_base + iy * nx + ix] + v[kp1_base + iy_m1 * nx + ix]);
        du_dz += 0.5f * (uc_kp1 - uc_0) / dzw;
        dv_dz += 0.5f * (vc_kp1 - vc_0) / dzw;
        float th_kp1 = theta_m[(k + 1u) * M + tid];
        n2 += 0.5f * (gravity / max(0.5f * (th_kp1 + th0), 200.0f)) * ((th_kp1 - th0) / dzw);
    }
    if (k > 0u) {
        float dzw = dz_w_int[(k - 1u) * dz_cols + col_dz];
        uint km1_base = (k - 1u) * M + col * stride_c;
        float uc_km1 = 0.5f * (u[km1_base + iy * nx + ix] + u[km1_base + iy * nx + ix_m1]);
        float vc_km1 = 0.5f * (v[km1_base + iy * nx + ix] + v[km1_base + iy_m1 * nx + ix]);
        du_dz += 0.5f * (uc_0 - uc_km1) / dzw;
        dv_dz += 0.5f * (vc_0 - vc_km1) / dzw;
        float th_km1 = theta_m[(k - 1u) * M + tid];
        n2 += 0.5f * (gravity / max(0.5f * (th0 + th_km1), 200.0f)) * ((th0 - th_km1) / dzw);
    }

    float s2 = 2.0f * (du_dx * du_dx + dv_dy * dv_dy + dw_dz * dw_dz)
             + (du_dy + dv_dx) * (du_dy + dv_dx)
             + du_dz * du_dz + dv_dz * dv_dz;
    float strain_mag = sqrt(max(s2, 1.0e-12f));

    float dh = sqrt(dx * dy);
    float dz_safe = max(dz, 1.0f);
    float aspect = max(dh / dz_safe, dz_safe / dh);
    float delta_f = pow(dx * dy * dz_safe, 1.0f / 3.0f);
    float lam = scotti_aspect(aspect) * delta_f;
    float cs_lam2_s = (c_s * lam) * (c_s * lam) * strain_mag;

    float ri = n2 / max(strain_mag * strain_mag, 1.0e-10f);
    float pr_n = 0.70f;
    float ri_c = 0.25f;
    float g_coef = (1.0f - pr_n) / ri_c;

    float nu_sgs = 0.0f;
    float d_sgs = 0.0f;
    if (ri < 0.0f) {
        nu_sgs = cs_lam2_s * sqrt(1.0f - 16.0f * ri);
        d_sgs  = (cs_lam2_s / pr_n) * sqrt(1.0f - 40.0f * ri);
    } else if (ri < ri_c) {
        float sf = 1.0f - ri / ri_c;
        float stab_fac = sf * sf * sf * sf;
        nu_sgs = cs_lam2_s * stab_fac;
        d_sgs  = (cs_lam2_s / pr_n) * stab_fac * (1.0f - g_coef * ri);
    }

    nu_sgs_out[gid] = nu_sgs;
    d_sgs_out[gid] = d_sgs;

    float nu_eff = nu_sgs + 0.004f * (dx * dy) / 10.0f;
    float lap_u = (u_xp1 - 2.0f * u0 + u_xm1) * (inv_dx * inv_dx)
                + (u_yp1 - 2.0f * u0 + u_ym1) * (inv_dy * inv_dy);
    float lap_v = (v_xp1 - 2.0f * v0 + v_xm1) * (inv_dx * inv_dx)
                + (v_yp1 - 2.0f * v0 + v_ym1) * (inv_dy * inv_dy);

    du_sgs_out[gid] = nu_eff * lap_u;
    dv_sgs_out[gid] = nu_eff * lap_v;
}

kernel void crm_sgs_heat_w(
    device const float* rho_d      [[buffer(0)]],
    device const float* w          [[buffer(1)]],
    device const float* theta_m    [[buffer(2)]],
    device const float* nu_sgs     [[buffer(3)]],
    device const float* d_sgs      [[buffer(4)]],
    device float* dw_sgs_out       [[buffer(5)]],
    device float* dThm_sgs_out     [[buffer(6)]],
    device const uint* u_params    [[buffer(7)]],
    device const float* f_params   [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_col = u_params[1];
    uint ny = u_params[2];
    uint nx = u_params[3];
    uint stride_c = ny * nx;
    uint M = n_col * stride_c;
    if (gid >= nz * M) return;

    uint k = gid / M;
    uint tid = gid - k * M;
    uint col = tid / stride_c;
    uint rem = tid - col * stride_c;
    uint iy = rem / nx;
    uint ix = rem - iy * nx;

    uint ix_m1 = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint ix_p1 = (ix + 1u < nx) ? (ix + 1u) : 0u;
    uint iy_m1 = (iy > 0u) ? (iy - 1u) : (ny - 1u);
    uint iy_p1 = (iy + 1u < ny) ? (iy + 1u) : 0u;

    float dx = f_params[0];
    float dy = f_params[1];
    float inv_dx = 1.0f / dx;
    float inv_dy = 1.0f / dy;
    float nu_bg = 0.004f * (dx * dy) / 10.0f;

    uint base_k = k * M + col * stride_c;
    uint idx_xm1 = base_k + iy * nx + ix_m1;
    uint idx_xp1 = base_k + iy * nx + ix_p1;
    uint idx_ym1 = base_k + iy_m1 * nx + ix;
    uint idx_yp1 = base_k + iy_p1 * nx + ix;

    if (k == 0u) {
        dw_sgs_out[tid] = 0.0f;
        dw_sgs_out[nz * M + tid] = 0.0f;
    } else {
        float w0 = w[gid];
        float w_lap = (w[idx_xp1] - 2.0f * w0 + w[idx_xm1]) * (inv_dx * inv_dx)
                    + (w[idx_yp1] - 2.0f * w0 + w[idx_ym1]) * (inv_dy * inv_dy);
        float nu_w = 0.5f * (nu_sgs[gid - M] + nu_sgs[gid]) + nu_bg;
        dw_sgs_out[gid] = nu_w * w_lap;
    }

    float th0   = theta_m[gid];
    float th_x1 = theta_m[idx_xp1];
    float th_xm = theta_m[idx_xm1];
    float th_y1 = theta_m[idx_yp1];
    float th_ym = theta_m[idx_ym1];

    float d0   = d_sgs[gid];
    float d_x1 = d_sgs[idx_xp1];
    float d_xm = d_sgs[idx_xm1];
    float d_y1 = d_sgs[idx_yp1];
    float d_ym = d_sgs[idx_ym1];

    float r0   = rho_d[gid];
    float r_x1 = rho_d[idx_xp1];
    float r_xm = rho_d[idx_xm1];
    float r_y1 = rho_d[idx_yp1];
    float r_ym = rho_d[idx_ym1];

    float fth_xp = -0.5f * (r0 + r_x1) * 0.5f * (d0 + d_x1) * (th_x1 - th0) * inv_dx;
    float fth_xm = -0.5f * (r_xm + r0) * 0.5f * (d_xm + d0) * (th0 - th_xm) * inv_dx;
    float fth_yp = -0.5f * (r0 + r_y1) * 0.5f * (d0 + d_y1) * (th_y1 - th0) * inv_dy;
    float fth_ym = -0.5f * (r_ym + r0) * 0.5f * (d_ym + d0) * (th0 - th_ym) * inv_dy;

    dThm_sgs_out[gid] = -(fth_xp - fth_xm) * inv_dx - (fth_yp - fth_ym) * inv_dy;
}

// =========================================================================
// 7. Fused Column-Local Bulk Microphysics & Implicit Sedimentation Kernel
// =========================================================================

inline float mp_esat_w(float T, float e_tr, float t_frz) {
    float tc = max(T, 160.0f) - t_frz;
    return e_tr * exp(17.67f * tc / (tc + 243.50f));
}

inline float mp_esat_i(float T, float e_tr, float t_frz) {
    float ts = max(T, 120.0f);
    return e_tr * exp(21.8745584f * (ts - t_frz) / (ts - 7.66f));
}

inline void mp_qsat_mix(
    float T, float pres, float e_tr, float t_frz, float t_ice, float t_liq, float eps_w,
    thread float& qsat, thread float& dqsat_dT)
{
    float omega = clamp((T - t_ice) / (t_liq - t_ice), 0.0f, 1.0f);
    float ew = mp_esat_w(T, e_tr, t_frz);
    float ei = mp_esat_i(T, e_tr, t_frz);
    float dw = max(pres - ew, 0.05f * pres);
    float di = max(pres - ei, 0.05f * pres);
    float qsw = eps_w * ew / dw;
    float qsi = eps_w * ei / di;
    qsat = omega * qsw + (1.0f - omega) * qsi;

    float t_w = max(T, 160.0f);
    float t_i = max(T, 120.0f);
    float cw = t_w - t_frz + 243.50f;
    float ci = t_i - 7.66f;
    float dew = ew * (17.67f * 243.50f) / (cw * cw);
    float dei = ei * (21.8745584f * (t_frz - 7.66f)) / (ci * ci);
    float dqsw = eps_w * pres * dew / (dw * dw);
    float dqsi = eps_w * pres * dei / (di * di);
    float domega = (T > t_ice && T < t_liq) ? (1.0f / (t_liq - t_ice)) : 0.0f;
    dqsat_dT = omega * dqsw + (1.0f - omega) * dqsi + domega * (qsw - qsi);
}

inline void mp_sat_adj(
    float t_li, float pres, float qn_in,
    float e_tr, float t_frz, float t_ice, float t_liq, float eps_w,
    float lv0, float ls0, float cpd,
    thread float& T_out, thread float& qv_out, thread float& qc_out, thread float& qi_out)
{
    float qn = max(qn_in, 0.0f);
    float qsat0, dq0;
    mp_qsat_mix(t_li, pres, e_tr, t_frz, t_ice, t_liq, eps_w, qsat0, dq0);
    if (qn <= qsat0) {
        T_out = t_li;
        qv_out = qn;
        qc_out = 0.0f;
        qi_out = 0.0f;
        return;
    }
    float T = t_li;
    for (int it = 0; it < 4; ++it) {
        float qsat, dqsat;
        mp_qsat_mix(T, pres, e_tr, t_frz, t_ice, t_liq, eps_w, qsat, dqsat);
        float qc_tmp = max(qn - qsat, 0.0f);
        float om = clamp((T - t_ice) / (t_liq - t_ice), 0.0f, 1.0f);
        float leff = om * lv0 + (1.0f - om) * ls0;
        float res = T - t_li - (leff / cpd) * qc_tmp;
        float dl = (T > t_ice && T < t_liq) ? ((lv0 - ls0) / (t_liq - t_ice)) : 0.0f;
        float df = 1.0f + (leff / cpd) * dqsat - (dl / cpd) * qc_tmp;
        T = T - res / max(df, 1.0f);
    }
    float qsat_f, dq_f;
    mp_qsat_mix(T, pres, e_tr, t_frz, t_ice, t_liq, eps_w, qsat_f, dq_f);
    float qcond = max(qn - qsat_f, 0.0f);
    float om_f = clamp((T - t_ice) / (t_liq - t_ice), 0.0f, 1.0f);
    T_out = T;
    qv_out = qn - qcond;
    qc_out = om_f * qcond;
    qi_out = (1.0f - om_f) * qcond;
}

kernel void crm_microphysics_col(
    device const float* rho_d      [[buffer(0)]],
    device const float* theta_m_c  [[buffer(1)]],
    device const float* q_n_in     [[buffer(2)]],
    device const float* q_p_in     [[buffer(3)]],
    device const float* q_v_prev   [[buffer(4)]],
    device const float* dz_c       [[buffer(5)]],
    device float* thm_out          [[buffer(6)]],
    device float* qn_out           [[buffer(7)]],
    device float* qp_out           [[buffer(8)]],
    device float* temp_out         [[buffer(9)]],
    device float* qv_out           [[buffer(10)]],
    device float* qc_out           [[buffer(11)]],
    device float* qi_out           [[buffer(12)]],
    device float* rain_out         [[buffer(13)]],
    device float* snow_out         [[buffer(14)]],
    device const uint* u_params    [[buffer(15)]],  // [nz, n_col, stride_c, dz_cols]
    device const float* mp         [[buffer(16)]],  // precomputed constants & dt
    uint m [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_col = u_params[1];
    uint stride_c = u_params[2];
    uint dz_cols = u_params[3];
    uint M = n_col * stride_c;
    if (m >= M) return;

    uint col = m / stride_c;
    uint col_dz = (dz_cols > 1u) ? col : 0u;

    float dt        = mp[0];
    float p_ref     = mp[1];
    float r_d       = mp[2];
    float r_v       = mp[3];
    float cp_d      = mp[4];
    float gamma_d   = mp[5];
    float kappa_d   = mp[6];
    float rv_ov_rd  = mp[7];
    float eps_w     = mp[8];
    float lv0       = mp[9];
    float ls0       = mp[11];
    float e_tr      = mp[12];
    float t_frz     = mp[13];
    float t_tri     = mp[14];
    float t_ice     = mp[15];
    float t_liq     = mp[16];
    float qc_thresh = mp[17];
    float qi_thresh = mp[18];
    float r_auto    = mp[19];
    float r_aggr    = mp[20];
    float vt_ci     = mp[21];
    float t0_p      = mp[22];
    float t00_p     = mp[23];
    float t0_g      = mp[24];
    float t00_g     = mp[25];
    float diff_v    = mp[26];
    float k_th_a    = mp[27];
    float c_r       = mp[28];
    float c_s       = mp[29];
    float c_g       = mp[30];
    float n0_r      = mp[31];
    float n0_s      = mp[32];
    float n0_g      = mp[33];
    float b_r       = mp[34];
    float b_s       = mp[35];
    float b_g       = mp[36];
    float fp_r      = mp[37];
    float fp_s      = mp[38];
    float fp_g      = mp[39];
    float ac_rc     = mp[40];
    float ac_sc     = mp[41];
    float ac_gc     = mp[42];
    float ac_ri     = mp[43];
    float ac_si     = mp[44];
    float ac_gi     = mp[45];
    float aem_r     = mp[46];
    float bem_r     = mp[47];
    float aem_s     = mp[48];
    float bem_s     = mp[49];
    float aem_g     = mp[50];
    float bem_g     = mp[51];

    float eps_small = 1.0e-15f;
    float exp_r = 0.25f * (3.0f + b_r);
    float exp_s = 0.25f * (3.0f + b_s);
    float exp_g = 0.25f * (3.0f + b_g);
    float ev_exp_r = 0.125f * (5.0f + b_r);
    float ev_exp_s = 0.125f * (5.0f + b_s);
    float ev_exp_g = 0.125f * (5.0f + b_g);
    float f_exp_r = 1.0f + 0.25f * b_r;
    float f_exp_s = 1.0f + 0.25f * b_s;
    float f_exp_g = 1.0f + 0.25f * b_g;

    float eps_acc_r = pow(eps_small, exp_r);
    float eps_acc_s = pow(eps_small, exp_s);
    float eps_acc_g = pow(eps_small, exp_g);
    float eps_ev_r  = pow(eps_small, ev_exp_r);
    float eps_ev_s  = pow(eps_small, ev_exp_s);
    float eps_ev_g  = pow(eps_small, ev_exp_g);
    float eps_fl_r  = pow(eps_small, f_exp_r);
    float eps_fl_s  = pow(eps_small, f_exp_s);
    float eps_fl_g  = pow(eps_small, f_exp_g);

    float tli_a[64];
    float pres_a[64];
    float exn_a[64];
    float qn_a[64];
    float qp_a[64];
    float qi_a[64];
    float mass_a[64];
    float ratep_a[64];
    float rho_a[64];
    float omega_p_s0 = 0.0f;

    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * M + m;
        float rd = rho_d[idx];
        float thm = theta_m_c[idx];
        float qn = max(q_n_in[idx], 0.0f);
        float qp = max(q_p_in[idx], 0.0f);
        float qvg = min(max(q_v_prev[idx], 0.0f), qn);

        float pres = p_ref * pow(max(r_d * rd * thm / p_ref, 1.0e-8f), gamma_d);
        float exner = pow(pres / p_ref, kappa_d);
        float temp_old = exner * (thm / (1.0f + rv_ov_rd * qvg));
        float q_cond_old = max(qn - qvg, 0.0f);
        float om_n_old = clamp((temp_old - t_ice) / (t_liq - t_ice), 0.0f, 1.0f);
        float t_li = temp_old - (om_n_old * lv0 + (1.0f - om_n_old) * ls0) * q_cond_old / cp_d;

        float temp, qv, qc, qi;
        mp_sat_adj(t_li, pres, qn, e_tr, t_frz, t_ice, t_liq, eps_w, lv0, ls0, cp_d, temp, qv, qc, qi);

        float om_p_old = clamp((temp - t00_p) / (t0_p - t00_p), 0.0f, 1.0f);
        float om_g_old = clamp((temp - t00_g) / (t0_g - t00_g), 0.0f, 1.0f);
        float qr = om_p_old * qp;
        float qs = (1.0f - om_p_old) * (1.0f - om_g_old) * qp;
        float qg = (1.0f - om_p_old) * om_g_old * qp;

        float ice_tfac = exp(0.025f * (temp - t_tri));
        float dq_auto = r_auto * max(qc - qc_thresh, 0.0f);
        float dq_aggr = r_aggr * ice_tfac * max(qi - qi_thresh, 0.0f);

        float qr_pow = pow(max(rd * qr, 0.0f) + eps_small, exp_r) - eps_acc_r;
        float qs_pow = pow(max(rd * qs, 0.0f) + eps_small, exp_s) - eps_acc_s;
        float qg_pow = pow(max(rd * qg, 0.0f) + eps_small, exp_g) - eps_acc_g;
        float inv_sqrt_rd = rsqrt(max(rd, 0.01f));

        float accr_c = inv_sqrt_rd * qc * (ac_rc * qr_pow + ac_sc * qs_pow + ac_gc * qg_pow);
        float accr_i = inv_sqrt_rd * ice_tfac * qi * (ac_ri * qr_pow + ac_si * qs_pow + ac_gi * qg_pow);

        float conv_delta = min((dq_auto + dq_aggr + accr_c + accr_i) * dt, qc + qi);
        float qn_mid = qn - conv_delta;
        float qp_mid = qp + conv_delta;

        float qsat_mix, dq_unused;
        mp_qsat_mix(temp, pres, e_tr, t_frz, t_ice, t_liq, eps_w, qsat_mix, dq_unused);
        float subsat = max(1.0f - qv / max(qsat_mix, 1.0e-8f), 0.0f);

        float ew = mp_esat_w(temp, e_tr, t_frz);
        float ei = mp_esat_i(temp, e_tr, t_frz);
        float ab_liq = (lv0 / (k_th_a * temp)) * (lv0 / (r_v * temp) - 1.0f) + (r_v * temp) / (diff_v * max(ew, 1.0e-3f));
        float ab_ice = (ls0 / (k_th_a * temp)) * (ls0 / (r_v * temp) - 1.0f) + (r_v * temp) / (diff_v * max(ei, 1.0e-4f));

        float qrm = om_p_old * max(qp_mid, 0.0f);
        float qsm = (1.0f - om_p_old) * (1.0f - om_g_old) * max(qp_mid, 0.0f);
        float qgm = (1.0f - om_p_old) * om_g_old * max(qp_mid, 0.0f);

        float rrq = max(rd * qrm, 0.0f);
        float rsq = max(rd * qsm, 0.0f);
        float rgq = max(rd * qgm, 0.0f);
        float rd_m025 = pow(rd, -0.25f);

        float vent_r = aem_r * sqrt(rrq) + bem_r * (pow(rrq + eps_small, ev_exp_r) - eps_ev_r) * rd_m025;
        float vent_s = aem_s * sqrt(rsq) + bem_s * (pow(rsq + eps_small, ev_exp_s) - eps_ev_s) * rd_m025;
        float vent_g = aem_g * sqrt(rgq) + bem_g * (pow(rgq + eps_small, ev_exp_g) - eps_ev_g) * rd_m025;

        float evap_r = (2.0f * 3.14159265f * c_r * n0_r / (rd * ab_liq)) * vent_r * subsat;
        float subl_s = (2.0f * 3.14159265f * c_s * n0_s / (rd * ab_ice)) * vent_s * subsat;
        float subl_g = (2.0f * 3.14159265f * c_g * n0_g / (rd * ab_ice)) * vent_g * subsat;

        float max_evap = min(qp_mid, max(qsat_mix - qn_mid, 0.0f));
        float evap_delta = min((evap_r + subl_s + subl_g) * dt, max_evap);

        float qn_after = qn_mid + evap_delta;
        float qp_after = qp_mid - evap_delta;
        float leff_prec = om_p_old * lv0 + (1.0f - om_p_old) * ls0;
        float dt_li_micro = (leff_prec / cp_d) * (conv_delta - evap_delta);
        float t_li_after = t_li + dt_li_micro;

        float qrs = om_p_old * max(qp_after, 0.0f);
        float qss = (1.0f - om_p_old) * (1.0f - om_g_old) * max(qp_after, 0.0f);
        float qgs = (1.0f - om_p_old) * om_g_old * max(qp_after, 0.0f);
        float flux_p0 =
            fp_r * inv_sqrt_rd * (pow(max(rd * qrs, 0.0f) + eps_small, f_exp_r) - eps_fl_r)
            + fp_s * inv_sqrt_rd * (pow(max(rd * qss, 0.0f) + eps_small, f_exp_s) - eps_fl_s)
            + fp_g * inv_sqrt_rd * (pow(max(rd * qgs, 0.0f) + eps_small, f_exp_g) - eps_fl_g);
        float rate_p = (qp_after > 1.0e-12f) ? min(flux_p0 / max(qp_after, 1.0e-12f), 15.0f * rd) : 0.0f;

        if (k == 0u) omega_p_s0 = om_p_old;
        tli_a[k] = t_li_after;
        pres_a[k] = pres;
        exn_a[k] = exner;
        qn_a[k] = qn_after;
        qp_a[k] = qp_after;
        qi_a[k] = qi;
        mass_a[k] = max(rd * dz_c[k * dz_cols + col_dz], 1.0e-4f);
        ratep_a[k] = rate_p;
        rho_a[k] = rd;
    }

    float qp_above = 0.0f;
    float qi_above = 0.0f;
    float ratep_above = 0.0f;
    float rateqi_above = 0.0f;

    for (int k_i = int(nz) - 1; k_i >= 0; --k_i) {
        uint k = uint(k_i);
        float inv_mass = 1.0f / mass_a[k];

        float diag_p = 1.0f + dt * ratep_a[k] * inv_mass;
        float du_qp = (k + 1u < nz) ? (-dt * ratep_above * inv_mass) * qp_above : 0.0f;
        float qp_sed = max((qp_a[k] - du_qp) / diag_p, 0.0f);
        qp_above = qp_sed;
        ratep_above = ratep_a[k];

        float rate_qi = rho_a[k] * vt_ci;
        float diag_qi = 1.0f + dt * rate_qi * inv_mass;
        float du_qi = (k + 1u < nz) ? (-dt * rateqi_above * inv_mass) * qi_above : 0.0f;
        float qi_sed = max((qi_a[k] - du_qi) / diag_qi, 0.0f);
        qi_above = qi_sed;
        rateqi_above = rate_qi;

        if (k == 0u) {
            float p_sfc_tot = ratep_a[0] * qp_sed;
            rain_out[m] = omega_p_s0 * p_sfc_tot;
            snow_out[m] = (1.0f - omega_p_s0) * p_sfc_tot + rate_qi * qi_sed;
        }

        float dqn_sed = qi_sed - qi_a[k];
        float qn_final = max(qn_a[k] + dqn_sed, 0.0f);
        float t_li_final = tli_a[k] - (ls0 / cp_d) * dqn_sed;

        float t_new, qv_new, qc_new, qi_new;
        mp_sat_adj(t_li_final, pres_a[k], qn_final, e_tr, t_frz, t_ice, t_liq, eps_w, lv0, ls0, cp_d, t_new, qv_new, qc_new, qi_new);

        uint idx = k * M + m;
        thm_out[idx] = (t_new / exn_a[k]) * (1.0f + rv_ov_rd * qv_new);
        qn_out[idx] = qn_final;
        qp_out[idx] = qp_sed;
        temp_out[idx] = t_new;
        qv_out[idx] = qv_new;
        qc_out[idx] = qc_new;
        qi_out[idx] = qi_new;
    }
}

// =========================================================================
// 8. Global SCVT Fast Acoustic-HEVI Sub-Step Kernels
// =========================================================================

kernel void scvt_fast_edge(
    device const int* c0_arr       [[buffer(0)]],   // [N_edges]
    device const int* c1_arr       [[buffer(1)]],   // [N_edges]
    device const float* dc_e       [[buffer(2)]],   // [N_edges]
    device const float* rtot_e     [[buffer(3)]],   // [nz, N_edges]
    device const float* fac_z0     [[buffer(4)]],   // [nz, N_edges]
    device const float* fac_z1     [[buffer(5)]],   // [nz, N_edges]
    device const float* m_edge     [[buffer(6)]],   // [nz, N_edges]
    device const float* thm_e      [[buffer(7)]],   // [nz, N_edges]
    device const float* tend_u_sl  [[buffer(8)]],   // [nz, N_edges]
    device const float* sponge_c   [[buffer(9)]],   // [nz]
    device const float* pres_s     [[buffer(10)]],  // [nz, N_cells]
    device const float* dp_dThm_s  [[buffer(11)]],  // [nz, N_cells]
    device const float* Thm_ref    [[buffer(12)]],  // [nz, N_cells]
    device const float* Thm_c      [[buffer(13)]],  // [nz, N_cells]
    device const float* u_c        [[buffer(14)]],  // [nz, N_edges]
    device float* u_n              [[buffer(15)]],  // [nz, N_edges]
    device float* uh_k             [[buffer(16)]],  // [nz, N_edges]
    device float* fTh_h_k          [[buffer(17)]],  // [nz, N_edges]
    device float* uh_accum         [[buffer(18)]],  // [nz, N_edges]
    device const uint* u_params    [[buffer(19)]],  // [nz, N_cells, N_edges, accum]
    device const float* f_params   [[buffer(20)]],  // [dt_ac, beta_hevi, gravity]
    uint gid [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_cells = u_params[1];
    uint n_edges = u_params[2];
    uint accum = u_params[3];
    if (gid >= nz * n_edges) return;

    uint k = gid / n_edges;
    uint e = gid - k * n_edges;
    uint c0 = uint(c0_arr[e]);
    uint c1 = uint(c1_arr[e]);
    uint idx_c0 = k * n_cells + c0;
    uint idx_c1 = k * n_cells + c1;

    float p0 = pres_s[idx_c0] + dp_dThm_s[idx_c0] * (Thm_c[idx_c0] - Thm_ref[idx_c0]);
    float p1 = pres_s[idx_c1] + dp_dThm_s[idx_c1] * (Thm_c[idx_c1] - Thm_ref[idx_c1]);
    float pgf = -(p1 * fac_z1[gid] - p0 * fac_z0[gid]) / (rtot_e[gid] * dc_e[e]);

    float u0 = u_c[gid];
    float dt_ac = f_params[0];
    float un = u0 + dt_ac * (tend_u_sl[gid] + pgf - sponge_c[k] * u0);
    float uh = m_edge[gid] * (1.10f * un - 0.10f * u0);

    u_n[gid] = un;
    uh_k[gid] = uh;
    fTh_h_k[gid] = uh * thm_e[gid];
    if (accum != 0u) {
        uh_accum[gid] += uh;
    }
}

kernel void scvt_fast_cell_hevi(
    device const int* eoc          [[buffer(0)]],   // [N_cells, 6]
    device const float* esign      [[buffer(1)]],   // [N_cells, 6]
    device const float* area_cell  [[buffer(2)]],   // [N_cells]
    device const float* dv_edge    [[buffer(3)]],   // [N_edges]
    device const float* area_edge  [[buffer(4)]],   // [N_edges]
    device const float* zw_slope_e [[buffer(5)]],   // [nz-1, N_edges]
    device const float* u_n        [[buffer(6)]],   // [nz, N_edges]
    device const float* uh_k       [[buffer(7)]],   // [nz, N_edges]
    device const float* fTh_h_k    [[buffer(8)]],   // [nz, N_edges]
    device const float* m_c        [[buffer(9)]],   // [nz, N_cells]
    device const float* Thm_c      [[buffer(10)]],  // [nz, N_cells]
    device const float* w_c        [[buffer(11)]],  // [nz+1, N_cells]
    device const float* udotz_c    [[buffer(12)]],  // [nz-1, N_cells]
    device const float* pres_s     [[buffer(13)]],  // [nz, N_cells]
    device const float* dp_dThm_s  [[buffer(14)]],  // [nz, N_cells]
    device const float* Thm_ref    [[buffer(15)]],  // [nz, N_cells]
    device const float* tend_Thm   [[buffer(16)]],  // [nz, N_cells]
    device const float* dz_c       [[buffer(17)]],  // [nz, N_cells]
    device const float* dz_w_int   [[buffer(18)]],  // [nz-1, N_cells]
    device const float* q_load     [[buffer(19)]],  // [nz, N_cells]
    device const float* hevi_c     [[buffer(20)]],  // [6, nz-1, N_cells]: rho_dw, rtot_w, thm_w, dl, diag, du
    device const float* sponge_w   [[buffer(21)]],  // [nz-1]
    device float* m_n              [[buffer(22)]],  // [nz, N_cells]
    device float* Thm_n            [[buffer(23)]],  // [nz, N_cells]
    device float* w_n              [[buffer(24)]],  // [nz+1, N_cells]
    device float* udotz_n          [[buffer(25)]],  // [nz-1, N_cells]
    device float* om_accum         [[buffer(26)]],  // [nz+1, N_cells]
    device const uint* u_params    [[buffer(27)]],  // [nz, N_cells, N_edges, accum]
    device const float* f_params   [[buffer(28)]],  // [dt_ac, beta_hevi, gravity]
    uint c [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_cells = u_params[1];
    uint n_edges = u_params[2];
    uint accum = u_params[3];
    if (c >= n_cells) return;

    uint n_int = nz - 1u;
    uint int_stride = n_int * n_cells;
    device const float* rho_dw = hevi_c;
    device const float* rtot_w = hevi_c + int_stride;
    device const float* thm_w  = hevi_c + 2u * int_stride;
    device const float* dl     = hevi_c + 3u * int_stride;
    device const float* diag   = hevi_c + 4u * int_stride;
    device const float* du     = hevi_c + 5u * int_stride;

    float dt_ac = f_params[0];
    float beta = f_params[1];
    float gravity = f_params[2];

    float inv_Ac = 1.0f / area_cell[c];
    uint e_id[6];
    float div_w[6];
    float uz_w[6];
    for (uint d = 0u; d < 6u; ++d) {
        uint eid = uint(eoc[c * 6u + d]);
        float sgn = esign[c * 6u + d];
        e_id[d] = eid;
        div_w[d] = sgn * dv_edge[eid] * inv_Ac;
        uz_w[d] = (sgn != 0.0f) ? (area_edge[eid] * inv_Ac) : 0.0f;
    }

    float m_dag[64];
    float Thm_dag[64];
    float p_blend[64];
    float rtot_blend[64];
    float om_old[64];
    float udotz_next[64];
    float cp[64];
    float dp[64];

    for (uint m = 0u; m < nz - 1u; ++m) {
        uint midx = m * n_cells + c;
        float uz_sum = 0.0f;
        uint eb0 = m * n_edges;
        uint eb1 = (m + 1u) * n_edges;
        for (uint d = 0u; d < 6u; ++d) {
            if (uz_w[d] != 0.0f) {
                uint eid = e_id[d];
                float u_int = 0.5f * (u_n[eb0 + eid] + u_n[eb1 + eid]);
                uz_sum += uz_w[d] * u_int * zw_slope_e[eb0 + eid];
            }
        }
        udotz_next[m] = uz_sum;
        udotz_n[midx] = uz_sum;
        om_old[m] = rho_dw[midx] * (w_c[(m + 1u) * n_cells + c] - udotz_c[midx]);
    }

    for (uint k = 0u; k < nz; ++k) {
        uint cidx = k * n_cells + c;
        uint ebase = k * n_edges;
        float d_m = 0.0f;
        float d_Th = 0.0f;
        for (uint d = 0u; d < 6u; ++d) {
            if (div_w[d] != 0.0f) {
                uint eid = e_id[d];
                d_m  += div_w[d] * uh_k[ebase + eid];
                d_Th += div_w[d] * fTh_h_k[ebase + eid];
            }
        }
        float om_dn = (k > 0u) ? om_old[k - 1u] : 0.0f;
        float om_up = (k + 1u < nz) ? om_old[k] : 0.0f;
        float thw_dn = (k > 0u) ? thm_w[(k - 1u) * n_cells + c] : 0.0f;
        float thw_up = (k + 1u < nz) ? thm_w[k * n_cells + c] : 0.0f;

        float mk = m_c[cidx];
        float Thmk = Thm_c[cidx];
        float md = mk - dt_ac * (d_m + (om_up - om_dn));
        float Td = Thmk + dt_ac * (-d_Th - (thw_up * om_up - thw_dn * om_dn) + tend_Thm[cidx]);

        float ps = pres_s[cidx];
        float dpT = dp_dThm_s[cidx];
        float Tr = Thm_ref[cidx];
        float pk = ps + dpT * (Thmk - Tr);
        float pd = ps + dpT * (Td - Tr);
        p_blend[k] = (1.0f - beta) * pk + beta * pd;

        float qd_dz = q_load[cidx] / dz_c[cidx];
        rtot_blend[k] = ((1.0f - beta) * mk + beta * md) * qd_dz;
        m_dag[k] = md;
        Thm_dag[k] = Td;
    }

    for (uint m = 0u; m < n_int; ++m) {
        uint midx = m * n_cells + c;
        float rdw = rho_dw[midx];
        float rtw = rtot_w[midx];
        float dzw = dz_w_int[midx];
        float rtot_bw = 0.5f * (rtot_blend[m] + rtot_blend[m + 1u]);
        float nh_imbal = (p_blend[m + 1u] - p_blend[m]) / dzw + rtot_bw * gravity;
        float fz_phys_k = rdw * w_c[(m + 1u) * n_cells + c];
        float rhs = -rdw * (udotz_next[m] - udotz_c[midx])
                    + dt_ac * (-(rdw / rtw) * nh_imbal - sponge_w[m] * fz_phys_k);

        float diag_m = diag[midx];
        if (m == 0u) {
            float inv = 1.0f / diag_m;
            cp[0] = (n_int > 1u) ? (du[midx] * inv) : 0.0f;
            dp[0] = rhs * inv;
        } else {
            float dlm = dl[midx];
            float inv = 1.0f / (diag_m - dlm * cp[m - 1u]);
            cp[m] = (m + 1u < n_int) ? (du[midx] * inv) : 0.0f;
            dp[m] = (rhs - dlm * dp[m - 1u]) * inv;
        }
    }

    float dom[64];
    dom[n_int - 1u] = dp[n_int - 1u];
    for (int m_i = int(n_int) - 2; m_i >= 0; --m_i) {
        uint m = uint(m_i);
        dom[m] = dp[m] - cp[m] * dom[m + 1u];
    }

    w_n[c] = 0.0f;
    w_n[nz * n_cells + c] = 0.0f;
    for (uint m = 0u; m < n_int; ++m) {
        uint midx = m * n_cells + c;
        float om_new = om_old[m] + dom[m];
        w_n[(m + 1u) * n_cells + c] = udotz_next[m] + om_new / max(rho_dw[midx], 1.0e-5f);
        if (accum != 0u) {
            om_accum[(m + 1u) * n_cells + c] += (1.0f - beta) * om_old[m] + beta * om_new;
        }
    }

    for (uint k = 0u; k < nz; ++k) {
        uint cidx = k * n_cells + c;
        float dom_dn = (k > 0u) ? dom[k - 1u] : 0.0f;
        float dom_up = (k < n_int) ? dom[k] : 0.0f;
        float thw_dn = (k > 0u) ? thm_w[(k - 1u) * n_cells + c] : 0.0f;
        float thw_up = (k < n_int) ? thm_w[k * n_cells + c] : 0.0f;

        m_n[cidx]   = max(m_dag[k] - beta * dt_ac * (dom_up - dom_dn), 1.0e-3f);
        Thm_n[cidx] = Thm_dag[k] - beta * dt_ac * (thw_up * dom_up - thw_dn * dom_dn);
    }
}

// =========================================================================
// 9. Fused Column-Local MYNN Level 2.5 Boundary-Layer Turbulence Kernel
// =========================================================================

inline float mynn_erff(float x) {
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t) * exp(-ax * ax);
    return sign * y;
}

inline void mynn_thomas_solve(
    uint nz,
    thread const float* dl,
    thread const float* d,
    thread const float* du,
    thread const float* rhs,
    thread float* out)
{
    float cp[64];
    float dp[64];
    float inv0 = 1.0f / d[0];
    cp[0] = (nz > 1u) ? (du[0] * inv0) : 0.0f;
    dp[0] = rhs[0] * inv0;
    for (uint k = 1u; k < nz; ++k) {
        float inv = 1.0f / (d[k] - dl[k] * cp[k - 1u]);
        cp[k] = (k + 1u < nz) ? (du[k] * inv) : 0.0f;
        dp[k] = (rhs[k] - dl[k] * dp[k - 1u]) * inv;
    }
    out[nz - 1u] = dp[nz - 1u];
    for (int k_i = int(nz) - 2; k_i >= 0; --k_i) {
        uint k = uint(k_i);
        out[k] = dp[k] - cp[k] * out[k + 1u];
    }
}

kernel void crm_mynn_col(
    device const float* rho_d      [[buffer(0)]],
    device const float* u_in       [[buffer(1)]],
    device const float* v_in       [[buffer(2)]],
    device const float* thm_in     [[buffer(3)]],
    device const float* qn_in      [[buffer(4)]],
    device const float* qv_in      [[buffer(5)]],
    device const float* qc_in      [[buffer(6)]],
    device const float* qi_in      [[buffer(7)]],
    device const float* qke_in     [[buffer(8)]],
    device const float* z_c        [[buffer(9)]],   // [nz, dz_cols]
    device const float* z_w        [[buffer(10)]],  // [nz+1, dz_cols]
    device const float* dz_c       [[buffer(11)]],  // [nz, dz_cols]
    device const float* sfc6       [[buffer(12)]],  // [6, M]
    device float* u_out            [[buffer(13)]],
    device float* v_out            [[buffer(14)]],
    device float* thm_out          [[buffer(15)]],
    device float* qn_out           [[buffer(16)]],
    device float* qke_out          [[buffer(17)]],
    device float* km_out           [[buffer(18)]],  // [nz-1, M]
    device float* kh_out           [[buffer(19)]],  // [nz-1, M]
    device float* qv_out           [[buffer(20)]],
    device float* qc_out           [[buffer(21)]],
    device float* qi_out           [[buffer(22)]],
    device const uint* u_params    [[buffer(23)]],  // [nz, M, dz_cols]
    device const float* mp         [[buffer(24)]],  // [dt, constants...]
    uint m [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint M = u_params[1];
    uint dz_cols = u_params[2];
    if (m >= M) return;
    uint col_dz = (dz_cols > 1u) ? m : 0u;

    float dt       = mp[0];
    float p_ref    = mp[1];
    float r_d      = mp[2];
    float cp_d     = mp[3];
    float cp_v     = mp[4];
    float gamma_d  = mp[5];
    float kappa_d  = mp[6];
    float rv_ov_rd = mp[7];
    float eps_w    = mp[8];
    float zvir     = mp[9];
    float gravity  = mp[10];
    float kappa_vk = mp[11];
    float lv0      = mp[12];
    float ls0      = mp[13];
    float e_tr     = mp[14];
    float t_frz    = mp[15];
    float t_ice    = mp[16];
    float t_liq    = mp[17];
    float tke_min  = mp[18];
    float B1       = mp[19];
    float B2       = mp[20];
    float C1       = mp[21];
    float C2       = mp[22];
    float C3       = mp[23];
    float C5       = mp[24];
    float A1       = mp[25];
    float A2       = mp[26];
    float GAM1     = mp[27];
    float GAM2     = mp[28];
    float F1       = mp[29];
    float F2       = mp[30];
    float RF1      = mp[31];
    float RF2      = mp[32];
    float RFC      = mp[33];

    float u_star       = sfc6[m];
    float sfc_buoy     = sfc6[M + m];
    float tau_x_sfc    = sfc6[2u * M + m];
    float tau_y_sfc    = sfc6[3u * M + m];
    float shf_sfc      = sfc6[4u * M + m];
    float evap_sfc     = sfc6[5u * M + m];

    float rd_a[64], u_a[64], v_a[64], qn_a[64], q2_a[64], qv_a[64], qc_a[64], qi_a[64];
    float pres_a[64], exn_a[64], temp_a[64], th_a[64], thl_a[64], zc_a[64], dzc_a[64], qvel_a[64];

    float zw0 = z_w[col_dz];
    float num_lt = 0.0f;
    float den_lt = 0.0f;
    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * M + m;
        uint dz_idx = k * dz_cols + col_dz;
        float rd = rho_d[idx];
        float thm = thm_in[idx];
        float qn = qn_in[idx];
        float qv = max(qv_in[idx], 0.0f);
        float qc = qc_in[idx];
        float qi = qi_in[idx];
        float q2 = max(qke_in[idx], 2.0f * tke_min);
        float qvel = sqrt(q2);

        float pres = p_ref * pow(max(r_d * rd * thm / p_ref, 1.0e-8f), gamma_d);
        float exner = pow(pres / p_ref, kappa_d);
        float temp = exner * (thm / (1.0f + rv_ov_rd * qv));
        float theta = temp / exner;
        float theta_l = theta - (lv0 * qc + ls0 * qi) / (cp_d * exner);

        float zc = z_c[dz_idx];
        float dzc = dz_c[dz_idx];
        num_lt += qvel * (zc - zw0) * dzc;
        den_lt += qvel * dzc;

        rd_a[k] = rd;
        u_a[k] = u_in[idx];
        v_a[k] = v_in[idx];
        qn_a[k] = qn;
        q2_a[k] = q2;
        qv_a[k] = qv;
        qc_a[k] = qc;
        qi_a[k] = qi;
        pres_a[k] = pres;
        exn_a[k] = exner;
        temp_a[k] = temp;
        th_a[k] = theta;
        thl_a[k] = theta_l;
        zc_a[k] = zc;
        dzc_a[k] = dzc;
        qvel_a[k] = qvel;
    }

    float l_t_col = clamp(0.23f * num_lt / max(den_lt, 1.0e-4f), 20.0f, 2500.0f);
    float us_safe = max(u_star, 0.02f);
    float b_sfc_nz = (fabs(sfc_buoy) < 1.0e-9f) ? 1.0e-9f : sfc_buoy;
    float l_mo = -(us_safe * us_safe * us_safe) / (kappa_vk * b_sfc_nz);
    float q_conv = pow(max(sfc_buoy, 0.0f) * l_t_col, 1.0f / 3.0f);
    float af_ratio = (A1 * F1) / (A2 * F2);
    float inv_sqrt2 = 0.70710678f;
    float inv_sqrt2pi = 0.39894228f;
    float sqrt_b2_h = sqrt(B2 * 0.5f);

    float km_a[64], kh_a[64], dzint_a[64], rho_int_a[64];
    float prod_c[64], l_c[64];
    for (uint k = 0u; k < nz; ++k) prod_c[k] = 0.0f;

    uint n_int = nz - 1u;
    for (uint k = 0u; k < n_int; ++k) {
        float dz_int = max(zc_a[k + 1u] - zc_a[k], 1.0f);
        float z_int = z_w[(k + 1u) * dz_cols + col_dz] - zw0;
        dzint_a[k] = dz_int;

        float du_dz = (u_a[k + 1u] - u_a[k]) / dz_int;
        float dv_dz = (v_a[k + 1u] - v_a[k]) / dz_int;
        float dthl_dz = (thl_a[k + 1u] - thl_a[k]) / dz_int;
        float dqn_dz = (qn_a[k + 1u] - qn_a[k]) / dz_int;
        float shear2 = du_dz * du_dz + dv_dz * dv_dz + 1.0e-10f;

        float q_int = 0.5f * (qvel_a[k + 1u] + qvel_a[k]);
        float q2_int = q_int * q_int;
        float temp_int = 0.5f * (temp_a[k + 1u] + temp_a[k]);
        float pres_int = 0.5f * (pres_a[k + 1u] + pres_a[k]);
        float theta_int = 0.5f * (th_a[k + 1u] + th_a[k]);
        float qn_int = 0.5f * (qn_a[k + 1u] + qn_a[k]);
        float qc_int = 0.5f * (qc_a[k + 1u] + qc_a[k]) + 0.5f * (qi_a[k + 1u] + qi_a[k]);

        float qsl, dqsl;
        mp_qsat_mix(temp_int, pres_int, e_tr, t_frz, t_ice, t_liq, eps_w, qsl, dqsl);
        float a_cond = 1.0f / (1.0f + (lv0 / cp_d) * dqsl);
        float b_cond = (temp_int / theta_int) * dqsl;
        float c_cond = (1.0f + zvir * qn_int - (1.0f + zvir) * qc_int) * (theta_int * lv0 / (temp_int * cp_d)) - (1.0f + zvir) * theta_int;

        float r_grid = (qc_int > 1.0e-6f) ? 1.0f : 0.0f;
        float beta_th0 = 1.0f + zvir * qn_int - (1.0f + zvir) * qc_int - 0.5f * r_grid * a_cond * b_cond * c_cond;
        float beta_q0 = zvir * theta_int + 0.5f * r_grid * a_cond * c_cond;
        float n2_init = (gravity / theta_int) * (beta_th0 * dthl_dz + beta_q0 * dqn_dz);

        float zeta = clamp(z_int / l_mo, -10.0f, 5.0f);
        float l_surf;
        if (zeta < 0.0f) {
            l_surf = kappa_vk * z_int * pow(max(1.0f - 100.0f * zeta, 1.0f), 0.20f);
        } else if (zeta < 1.0f) {
            l_surf = kappa_vk * z_int / (1.0f + 2.7f * zeta);
        } else {
            l_surf = kappa_vk * z_int / 3.7f;
        }

        float inv_lb = 0.0f;
        if (n2_init > 1.0e-8f) {
            float n_bv = sqrt(n2_init);
            float lb_base = q_int / n_bv;
            float lb = (zeta < 0.0f) ? ((1.0f + 5.0f * sqrt(q_conv / max(l_t_col * n_bv, 1.0e-6f))) * lb_base) : lb_base;
            inv_lb = 1.0f / max(lb, 1.0f);
        }

        float l_master = clamp(1.0f / (1.0f / max(l_surf, 0.5f) + 1.0f / l_t_col + inv_lb), 0.5f, 1500.0f);
        if (k == 0u) l_c[0] = l_master;
        if (k > 0u)  l_c[k] = 0.5f * (l_c[k] + l_master);
        if (k + 1u < n_int) l_c[k + 1u] = l_master;
        else l_c[nz - 1u] = l_master;

        float dq_li = dqn_dz - b_cond * dthl_dz;
        float sigma_s = max(0.5f * a_cond * l_master * sqrt_b2_h * fabs(dq_li), 1.0e-7f);
        float q1 = clamp(a_cond * (qn_int - qsl) / (2.0f * sigma_s), -6.0f, 6.0f);
        float r_sub = 0.5f * (1.0f + mynn_erff(q1 * inv_sqrt2));
        float gauss_q1 = inv_sqrt2pi * exp(-0.5f * q1 * q1);
        float ql_sub = 2.0f * sigma_s * (r_sub * q1 + gauss_q1);
        float r_tilde = clamp(r_sub - (ql_sub / (2.0f * sigma_s)) * gauss_q1, 0.0f, 1.0f);

        float beta_th = 1.0f + zvir * qn_int - (1.0f + zvir) * ql_sub - r_tilde * a_cond * b_cond * c_cond;
        float beta_q = zvir * theta_int + r_tilde * a_cond * c_cond;
        float n2 = (gravity / theta_int) * (beta_th * dthl_dz + beta_q * dqn_dz);

        float ri = clamp(n2 / shear2, -10.0f, 5.0f);
        float disc = max(ri * ri + 2.0f * af_ratio * (RF1 - 2.0f * RF2) * ri + (af_ratio * RF1) * (af_ratio * RF1), 0.0f);
        float rf = min(0.5f * (1.0f / af_ratio) * (ri + af_ratio * RF1 - sqrt(disc)), RFC - 1.0e-4f);
        float sh2 = max(3.0f * A2 * (GAM1 + GAM2) * (RFC - rf) / (1.0f - rf), 0.0f);
        float sm2 = max(af_ratio * ((RF1 - rf) / max(RF2 - rf, 1.0e-5f)) * sh2, 0.0f);
        float q2_lev2 = max(B1 * (l_master * l_master) * sm2 * (1.0f - rf) * shear2, 0.0f);

        float alpha_c = (q2_lev2 > q2_int) ? (q_int / sqrt(max(q2_lev2, 1.0e-8f))) : 1.0f;
        float ac2 = alpha_c * alpha_c;
        float g_m = min((l_master * l_master / q2_int) * shear2, 1.0e4f);
        float g_h = clamp(-(l_master * l_master / q2_int) * n2, -4.0f, 0.028f);

        float phi1 = 1.0f - 3.0f * ac2 * A2 * B2 * (1.0f - C3) * g_h;
        float phi2 = 1.0f - 9.0f * ac2 * A1 * A2 * (1.0f - C2) * g_h;
        float phi3 = phi1 + 9.0f * ac2 * (A2 * A2) * (1.0f - C2) * (1.0f - C5) * g_h;
        float phi4 = phi1 - 12.0f * ac2 * A1 * A2 * (1.0f - C2) * g_h;
        float phi5 = 6.0f * ac2 * (A1 * A1) * g_m;
        float denom25 = max(phi2 * phi4 + phi5 * phi3, 1.0e-5f);

        float s_m = clamp(alpha_c * A1 * (phi3 - 3.0f * C1 * phi4) / denom25, 0.0f, 2.5f);
        float s_h = clamp(alpha_c * A2 * (phi2 + 3.0f * C1 * phi5) / denom25, 0.0f, 3.0f);

        float km = l_master * q_int * s_m;
        float kh = l_master * q_int * s_h;
        km_a[k] = km;
        kh_a[k] = kh;
        km_out[k * M + m] = km;
        kh_out[k * M + m] = kh;
        rho_int_a[k] = 0.5f * (rd_a[k + 1u] + rd_a[k]);

        float prod_i = 2.0f * (km * shear2 - kh * n2);
        prod_c[k] += 0.5f * prod_i;
        prod_c[k + 1u] += 0.5f * prod_i;
    }

    float zrel0 = max(zc_a[0] - zw0, 1.0f);
    prod_c[0] += 2.0f * max((u_star * u_star * u_star) / (kappa_vk * zrel0) + sfc_buoy, 0.0f);

    float mass_c[64], dl_t[64], du_t[64], diag_t[64], rhs_t[64], sol_a[64];
    for (uint k = 0u; k < nz; ++k) {
        mass_c[k] = rd_a[k] * dzc_a[k];
        dl_t[k] = 0.0f;
        du_t[k] = 0.0f;
    }

    for (uint k = 0u; k < n_int; ++k) {
        float fct = dt * 3.0f * rho_int_a[k] * km_a[k] / dzint_a[k];
        dl_t[k + 1u] = -fct / mass_c[k + 1u];
        du_t[k]      = -fct / mass_c[k];
    }
    for (uint k = 0u; k < nz; ++k) {
        float prod_pos = max(prod_c[k], 0.0f);
        float buoy_sink = max(-prod_c[k], 0.0f) / q2_a[k];
        diag_t[k] = 1.0f - dl_t[k] - du_t[k] + dt * (2.0f * qvel_a[k] / (B1 * l_c[k]) + buoy_sink);
        rhs_t[k] = q2_a[k] + dt * prod_pos;
    }
    mynn_thomas_solve(nz, dl_t, diag_t, du_t, rhs_t, sol_a);
    for (uint k = 0u; k < nz; ++k) qke_out[k * M + m] = max(sol_a[k], 2.0f * tke_min);

    for (uint k = 0u; k < n_int; ++k) {
        float fcm = dt * rho_int_a[k] * km_a[k] / dzint_a[k];
        dl_t[k + 1u] = -fcm / mass_c[k + 1u];
        du_t[k]      = -fcm / mass_c[k];
    }
    float lim0 = 0.5f * mass_c[0] / max(dt, 1.0f);
    for (int comp = 0; comp < 2; ++comp) {
        float tau_sfc = (comp == 0) ? tau_x_sfc : tau_y_sfc;
        float v0 = (comp == 0) ? u_a[0] : v_a[0];
        bool opp = (tau_sfc * v0) < 0.0f;
        float r_drag = opp ? (fabs(tau_sfc) / max(fabs(v0), 0.10f)) : 0.0f;
        float exp_rem = opp ? (tau_sfc + r_drag * v0) : clamp(tau_sfc, -lim0, lim0);
        for (uint k = 0u; k < nz; ++k) {
            diag_t[k] = 1.0f - dl_t[k] - du_t[k];
            rhs_t[k] = (comp == 0) ? u_a[k] : v_a[k];
        }
        diag_t[0] += dt * r_drag / mass_c[0];
        rhs_t[0]  += dt * exp_rem / mass_c[0];
        mynn_thomas_solve(nz, dl_t, diag_t, du_t, rhs_t, sol_a);
        for (uint k = 0u; k < nz; ++k) {
            if (comp == 0) u_out[k * M + m] = sol_a[k];
            else           v_out[k * M + m] = sol_a[k];
        }
    }

    for (uint k = 0u; k < n_int; ++k) {
        float fch = dt * rho_int_a[k] * kh_a[k] / dzint_a[k];
        dl_t[k + 1u] = -fch / mass_c[k + 1u];
        du_t[k]      = -fch / mass_c[k];
    }
    for (uint k = 0u; k < nz; ++k) {
        diag_t[k] = 1.0f - dl_t[k] - du_t[k];
        rhs_t[k] = thl_a[k];
    }
    float cp_eff0 = cp_d + cp_v * qv_a[0];
    rhs_t[0] += dt * (shf_sfc / (cp_eff0 * exn_a[0])) / mass_c[0];
    mynn_thomas_solve(nz, dl_t, diag_t, du_t, rhs_t, sol_a);

    float thl_new_a[64];
    for (uint k = 0u; k < nz; ++k) {
        thl_new_a[k] = sol_a[k];
        rhs_t[k] = qn_a[k];
    }
    rhs_t[0] += dt * evap_sfc / mass_c[0];
    mynn_thomas_solve(nz, dl_t, diag_t, du_t, rhs_t, sol_a);

    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * M + m;
        float qn_new = max(sol_a[k], 0.0f);
        float t_li_new = thl_new_a[k] * exn_a[k];
        float t_new, qv_new, qc_new, qi_new;
        mp_sat_adj(t_li_new, pres_a[k], qn_new, e_tr, t_frz, t_ice, t_liq, eps_w, lv0, ls0, cp_d, t_new, qv_new, qc_new, qi_new);
        qn_out[idx]  = qn_new;
        qv_out[idx]  = qv_new;
        qc_out[idx]  = qc_new;
        qi_out[idx]  = qi_new;
        thm_out[idx] = (t_new / exn_a[k]) * (1.0f + rv_ov_rd * qv_new);
    }
}

// =========================================================================
// 10. 2D (ny == 1, nx <= 32, nz <= 32) Threadgroup-Fused CRM Fast Acoustic-HEVI Loop
// =========================================================================

kernel void crm_2d_fast_hevi_loop(
    device float* rho_s            [[buffer(0)]],   // [nz, n_col, 1, nx] (initialized to rho0, updated in-place)
    device float* Thm_s            [[buffer(1)]],   // [nz, n_col, 1, nx]
    device float* u_s              [[buffer(2)]],   // [nz, n_col, 1, nx]
    device float* v_s              [[buffer(3)]],   // [nz, n_col, 1, nx]
    device float* w_s              [[buffer(4)]],   // [nz+1, n_col, 1, nx]
    device const float* rho_x      [[buffer(5)]],
    device const float* rtot_x_inv [[buffer(6)]],
    device const float* thm_x_eff  [[buffer(7)]],
    device const float* pres_s     [[buffer(8)]],
    device const float* dp_dThm_s  [[buffer(9)]],
    device const float* Thm_ref    [[buffer(10)]],
    device const float* q_load     [[buffer(11)]],
    device const float* tend_u_sl  [[buffer(12)]],
    device const float* tend_v_sl  [[buffer(13)]],
    device const float* tend_w_sl  [[buffer(14)]],  // [nz-1, n_col, 1, nx]
    device const float* tend_Thm   [[buffer(15)]],
    device const float* rho_dw     [[buffer(16)]],  // [nz-1, ...]
    device const float* rho_tot_w  [[buffer(17)]],
    device const float* thm_w      [[buffer(18)]],
    device const float* dl         [[buffer(19)]],
    device const float* diag       [[buffer(20)]],
    device const float* du         [[buffer(21)]],
    device const float* dz_c       [[buffer(22)]],  // [nz, dz_cols]
    device const float* dz_w_int   [[buffer(23)]],  // [nz-1, dz_cols]
    device const float* sponge_w   [[buffer(24)]],  // [nz-1]
    device float* fx_acc           [[buffer(25)]],
    device float* fy_acc           [[buffer(26)]],
    device float* fz_acc           [[buffer(27)]],
    device const uint* u_params    [[buffer(28)]],  // [nz, n_col, nx, dz_cols, n_ac, accum]
    device const float* f_params   [[buffer(29)]],  // [dt_ac, beta_hevi, gravity, dx]
    uint lid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_col = u_params[1];
    uint nx = u_params[2];
    uint dz_cols = u_params[3];
    uint n_ac = u_params[4];
    uint accum = u_params[5];

    uint local_c = lid / nx;
    uint ix = lid - local_c * nx;
    uint c_tg = 64u / nx;
    uint col = tg_id * c_tg + local_c;
    bool active = (col < n_col);

    uint ix_m1 = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint ix_p1 = (ix + 1u < nx) ? (ix + 1u) : 0u;
    uint lid_xm1 = local_c * nx + ix_m1;
    uint lid_xp1 = local_c * nx + ix_p1;
    uint M = n_col * nx;
    uint tid = active ? (col * nx + ix) : 0u;
    uint col_dz = (dz_cols > 1u && active) ? col : 0u;

    float dt_ac = f_params[0];
    float beta = f_params[1];
    float gravity = f_params[2];
    float inv_dx = 1.0f / f_params[3];

    threadgroup float tg_Thm[2048];
    threadgroup float tg_fx[2048];

    float c_star[32], d_star[32];

    for (uint ac = 0u; ac < n_ac; ++ac) {
        if (active) {
            for (uint k = 0u; k < nz; ++k) {
                tg_Thm[k * 64u + lid] = Thm_s[k * M + tid];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (uint k = 0u; k < nz; ++k) {
                uint idx = k * M + tid;
                float ps0 = pres_s[idx];
                float dp0 = dp_dThm_s[idx];
                float Tr0 = Thm_ref[idx];
                float p_k0 = ps0 + dp0 * (tg_Thm[k * 64u + lid] - Tr0);

                uint idx_xp1 = k * M + col * nx + ix_p1;
                float p_k1 = pres_s[idx_xp1] + dp_dThm_s[idx_xp1] * (tg_Thm[k * 64u + lid_xp1] - Thm_ref[idx_xp1]);

                float pgf_u = -(p_k1 - p_k0) * inv_dx * rtot_x_inv[idx];
                float u0 = u_s[idx];
                float v0 = v_s[idx];
                float u_n = u0 + dt_ac * (tend_u_sl[idx] + pgf_u);
                float v_n = v0 + dt_ac * tend_v_sl[idx];

                float fx_k = rho_x[idx] * (1.10f * u_n - 0.10f * u0);
                tg_fx[k * 64u + lid] = fx_k;
                if (accum != 0u) {
                    fx_acc[idx] += fx_k;
                    fy_acc[idx] += rho_s[idx] * (1.10f * v_n - 0.10f * v0);
                }
                u_s[idx] = u_n;
                v_s[idx] = v_n;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            float p_blend_prev = 0.0f;
            float rtot_blend_prev = 0.0f;
            float fz_old_below = 0.0f;
            float thm_w_below = 0.0f;
            float c_prev = 0.0f;
            float d_prev = 0.0f;

            for (uint k = 0u; k < nz; ++k) {
                uint idx = k * M + tid;
                uint idx_xm = k * M + col * nx + ix_m1;

                float fx_0 = tg_fx[k * 64u + lid];
                float fx_m = tg_fx[k * 64u + lid_xm1];
                float div_h_rho = (fx_0 - fx_m) * inv_dx;
                float div_h_Thm = (fx_0 * thm_x_eff[idx] - fx_m * thm_x_eff[idx_xm]) * inv_dx;

                float fz_old_above = 0.0f;
                float thm_w_above = 0.0f;
                if (k < nz - 1u) {
                    fz_old_above = rho_dw[idx] * w_s[(k + 1u) * M + tid];
                    thm_w_above = thm_w[idx];
                }

                float dz_val = dz_c[k * dz_cols + col_dz];
                float r_cur = rho_s[idx];
                float T_cur = Thm_s[idx];
                float r_dag = r_cur - dt_ac * (div_h_rho + (fz_old_above - fz_old_below) / dz_val);
                float T_dag = T_cur + dt_ac * (
                    -div_h_Thm
                    - (thm_w_above * fz_old_above - thm_w_below * fz_old_below) / dz_val
                    + tend_Thm[idx]
                );
                rho_s[idx] = r_dag;
                Thm_s[idx] = T_dag;

                float ps = pres_s[idx];
                float dp = dp_dThm_s[idx];
                float Ts = Thm_ref[idx];
                float p_blend = (1.0f - beta) * (ps + dp * (T_cur - Ts)) + beta * (ps + dp * (T_dag - Ts));
                float rtot_blend = ((1.0f - beta) * r_cur + beta * r_dag) * q_load[idx];

                if (k >= 1u) {
                    uint um = k - 1u;
                    uint idx_m = um * M + tid;
                    float dz_w_val = dz_w_int[um * dz_cols + col_dz];
                    float nh_imb = (p_blend - p_blend_prev) / dz_w_val + 0.5f * (rtot_blend + rtot_blend_prev) * gravity;
                    float rdw_m = rho_dw[idx_m];
                    float rtw_m = max(rho_tot_w[idx_m], 1.0e-5f);
                    float rhs_m = dt_ac * (
                        rdw_m * tend_w_sl[idx_m]
                        - (rdw_m / rtw_m) * nh_imb
                        - sponge_w[um] * fz_old_below
                    );
                    if (um == 0u) {
                        float inv_b0 = 1.0f / diag[idx_m];
                        c_prev = du[idx_m] * inv_b0;
                        d_prev = rhs_m * inv_b0;
                    } else {
                        float inv_den = 1.0f / (diag[idx_m] - dl[idx_m] * c_prev);
                        c_prev = (um < nz - 2u) ? (du[idx_m] * inv_den) : 0.0f;
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

            w_s[tid] = 0.0f;
            w_s[nz * M + tid] = 0.0f;

            float dfz_above = 0.0f;
            float thm_w_above = 0.0f;
            for (int m = int(nz) - 2; m >= 0; --m) {
                uint um = uint(m);
                uint idx_m = um * M + tid;
                float dfz_cur = (m == int(nz) - 2) ? d_star[um] : (d_star[um] - c_star[um] * dfz_above);
                float rdw_m = rho_dw[idx_m];
                uint w_idx = (um + 1u) * M + tid;
                float fz_old_m = rdw_m * w_s[w_idx];
                float fz_new_m = fz_old_m + dfz_cur;
                w_s[w_idx] = fz_new_m / max(rdw_m, 1.0e-5f);
                if (accum != 0u) {
                    fz_acc[w_idx] += (fz_old_m + beta * dfz_cur);
                }

                uint k_layer = um + 1u;
                uint idx_l = k_layer * M + tid;
                float thm_w_m = thm_w[idx_m];
                float dz_val = dz_c[k_layer * dz_cols + col_dz];
                rho_s[idx_l] = max(rho_s[idx_l] - beta * dt_ac * (dfz_above - dfz_cur) / dz_val, 1.0e-5f);
                Thm_s[idx_l] = Thm_s[idx_l] - beta * dt_ac * (thm_w_above * dfz_above - thm_w_m * dfz_cur) / dz_val;

                dfz_above = dfz_cur;
                thm_w_above = thm_w_m;
            }
            float dz_0 = dz_c[col_dz];
            rho_s[tid] = max(rho_s[tid] - beta * dt_ac * dfz_above / dz_0, 1.0e-5f);
            Thm_s[tid] = Thm_s[tid] - beta * dt_ac * (thm_w_above * dfz_above) / dz_0;
        }
    }
}

// =========================================================================
// 11. SCVT Mesh Operators (Divergence, Vorticity, Vector/Scalar Laplacian, TRiSK)
// =========================================================================

kernel void scvt_div_cell(
    device const int* eoc          [[buffer(0)]],   // [N_cells, 6]
    device const float* div_w      [[buffer(1)]],   // [N_cells, 6] pre-scaled s*dv_edge/A_c
    device const float* flux_e     [[buffer(2)]],   // [nz, N_edges]
    device float* div_out          [[buffer(3)]],   // [nz, N_cells]
    device const uint* u_params    [[buffer(4)]],   // [nz, N_cells, N_edges]
    uint gid [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_cells = u_params[1];
    uint n_edges = u_params[2];
    if (gid >= nz * n_cells) return;
    uint k = gid / n_cells;
    uint c = gid - k * n_cells;
    uint c6 = c * 6u;
    uint ebase = k * n_edges;
    float acc = 0.0f;
    for (uint d = 0u; d < 6u; ++d) {
        acc += div_w[c6 + d] * flux_e[ebase + uint(eoc[c6 + d])];
    }
    div_out[gid] = acc;
}

kernel void scvt_vorticity_vert(
    device const int* eov          [[buffer(0)]],   // [N_verts, 3]
    device const float* vort_w     [[buffer(1)]],   // [N_verts, 3] pre-scaled s*dc_edge/A_v
    device const float* u_e        [[buffer(2)]],   // [nz, N_edges]
    device float* zeta_out         [[buffer(3)]],   // [nz, N_verts]
    device const uint* u_params    [[buffer(4)]],   // [nz, N_verts, N_edges]
    uint gid [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_verts = u_params[1];
    uint n_edges = u_params[2];
    if (gid >= nz * n_verts) return;
    uint k = gid / n_verts;
    uint v = gid - k * n_verts;
    uint v3 = v * 3u;
    uint ebase = k * n_edges;
    zeta_out[gid] = vort_w[v3]      * u_e[ebase + uint(eov[v3])]
                  + vort_w[v3 + 1u] * u_e[ebase + uint(eov[v3 + 1u])]
                  + vort_w[v3 + 2u] * u_e[ebase + uint(eov[v3 + 2u])];
}

kernel void scvt_vec_lap_edge(
    device const int* c0_arr       [[buffer(0)]],   // [N_edges]
    device const int* c1_arr       [[buffer(1)]],   // [N_edges]
    device const int* v0_arr       [[buffer(2)]],   // [N_edges]
    device const int* v1_arr       [[buffer(3)]],   // [N_edges]
    device const float* inv_dc     [[buffer(4)]],   // [N_edges]
    device const float* inv_dv     [[buffer(5)]],   // [N_edges] (1 / max(dv_edge, 100))
    device const float* div_c      [[buffer(6)]],   // [nz, N_cells]
    device const float* zeta_v     [[buffer(7)]],   // [nz, N_verts]
    device float* lap_out          [[buffer(8)]],   // [nz, N_edges]
    device float* grad_div_out     [[buffer(9)]],   // [nz, N_edges]
    device const uint* u_params    [[buffer(10)]],  // [nz, N_cells, N_verts, N_edges]
    uint gid [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_cells = u_params[1];
    uint n_verts = u_params[2];
    uint n_edges = u_params[3];
    if (gid >= nz * n_edges) return;
    uint k = gid / n_edges;
    uint e = gid - k * n_edges;

    uint cbase = k * n_cells;
    uint vbase = k * n_verts;
    float gd = (div_c[cbase + uint(c1_arr[e])] - div_c[cbase + uint(c0_arr[e])]) * inv_dc[e];
    float rc = -(zeta_v[vbase + uint(v1_arr[e])] - zeta_v[vbase + uint(v0_arr[e])]) * inv_dv[e];
    grad_div_out[gid] = gd;
    lap_out[gid] = gd + rc;
}

kernel void scvt_scalar_grad_edge(
    device const int* c0_arr       [[buffer(0)]],   // [N_edges]
    device const int* c1_arr       [[buffer(1)]],   // [N_edges]
    device const float* inv_dc     [[buffer(2)]],   // [N_edges]
    device const float* phi_c      [[buffer(3)]],   // [nz, N_cells]
    device float* grad_out         [[buffer(4)]],   // [nz, N_edges]
    device const uint* u_params    [[buffer(5)]],   // [nz, N_cells, N_edges]
    uint gid [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_cells = u_params[1];
    uint n_edges = u_params[2];
    if (gid >= nz * n_edges) return;
    uint k = gid / n_edges;
    uint e = gid - k * n_edges;
    uint cbase = k * n_cells;
    grad_out[gid] = (phi_c[cbase + uint(c1_arr[e])] - phi_c[cbase + uint(c0_arr[e])]) * inv_dc[e];
}

kernel void scvt_trisk_rot_edge(
    device const int* v0_arr       [[buffer(0)]],   // [N_edges]
    device const int* v1_arr       [[buffer(1)]],   // [N_edges]
    device const int* ep_arr       [[buffer(2)]],   // [N_edges, 10]
    device const float* w_trisk    [[buffer(3)]],   // [N_edges, 10]
    device const float* zeta_v     [[buffer(4)]],   // [nz, N_verts]
    device const float* f_vert     [[buffer(5)]],   // [N_verts]
    device const float* u_e        [[buffer(6)]],   // [nz, N_edges]
    device float* rot_out          [[buffer(7)]],   // [nz, N_edges]
    device const uint* u_params    [[buffer(8)]],   // [nz, N_verts, N_edges]
    uint gid [[thread_position_in_grid]])
{
    uint nz = u_params[0];
    uint n_verts = u_params[1];
    uint n_edges = u_params[2];
    if (gid >= nz * n_edges) return;
    uint k = gid / n_edges;
    uint e = gid - k * n_edges;

    uint vbase = k * n_verts;
    uint ebase = k * n_edges;
    uint v0e = uint(v0_arr[e]);
    uint v1e = uint(v1_arr[e]);
    float eta_self = 0.5f * (zeta_v[vbase + v0e] + f_vert[v0e] + zeta_v[vbase + v1e] + f_vert[v1e]);

    uint e10 = e * 10u;
    float acc = 0.0f;
    for (uint j = 0u; j < 10u; ++j) {
        uint ep = uint(ep_arr[e10 + j]);
        uint vp0 = uint(v0_arr[ep]);
        uint vp1 = uint(v1_arr[ep]);
        float eta_ep = 0.5f * (zeta_v[vbase + vp0] + f_vert[vp0] + zeta_v[vbase + vp1] + f_vert[vp1]);
        acc += w_trisk[e10 + j] * 0.5f * (eta_self + eta_ep) * u_e[ebase + ep];
    }
    rot_out[gid] = acc;
}
"""


def run_crm_fct_advect_metal(
    lib: Any,
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
    """Execute batched 3D Zalesak FCT across `[B, nz, n_col, ny, nx]` on Metal GPU."""
    B, nz, n_col, ny, nx = q_batch.shape
    M = n_col * ny * nx
    vol_pts = nz * M
    total_q = B * vol_pts
    dev = q_batch.device
    dz_2d = dz_c.view(nz, -1).contiguous()
    dz_cols = dz_2d.shape[1]

    key = (dev.index or 0, B, nz, n_col, ny, nx, dz_cols)
    ws = _FCT_WS_CACHE.get(key)
    if ws is None:
        ws = {
            "ax": torch.empty((B, nz, n_col, ny, nx), device=dev, dtype=torch.float32),
            "ay": torch.empty((B, nz, n_col, ny, nx), device=dev, dtype=torch.float32),
            "az": torch.empty((B, nz + 1, n_col, ny, nx), device=dev, dtype=torch.float32),
            "q_td": torch.empty((B, nz, n_col, ny, nx), device=dev, dtype=torch.float32),
            "r_plus": torch.empty((B, nz, n_col, ny, nx), device=dev, dtype=torch.float32),
            "r_minus": torch.empty((B, nz, n_col, ny, nx), device=dev, dtype=torch.float32),
            "u_params": torch.tensor([B, nz, n_col, ny, nx, dz_cols], dtype=torch.uint32, device=dev),
        }
        _cache_put(_FCT_WS_CACHE, key, ws)

    q_c = q_batch.contiguous()
    ro_c = rho_old.contiguous()
    rn_c = rho_new.contiguous()
    fx_c = fx.contiguous()
    fy_c = fy.contiguous()
    fz_c = fz.contiguous()
    q_out = torch.empty_like(q_c)
    f_params = torch.tensor([dt, dx, dy], dtype=torch.float32, device=dev)

    lib.crm_fct_flux_td(
        q_c, ro_c, rn_c, fx_c, fy_c, fz_c, dz_2d,
        ws["ax"], ws["ay"], ws["az"], ws["q_td"],
        ws["u_params"], f_params, threads=total_q,
    )
    lib.crm_fct_bounds_ratios(
        q_c, ws["q_td"], rn_c, ws["ax"], ws["ay"], ws["az"], dz_2d,
        ws["r_plus"], ws["r_minus"],
        ws["u_params"], f_params, threads=total_q,
    )
    lib.crm_fct_limit_apply(
        ws["q_td"], rn_c, ws["ax"], ws["ay"], ws["az"], ws["r_plus"], ws["r_minus"], dz_2d,
        q_out, ws["u_params"], f_params, threads=total_q,
    )
    return q_out


def run_crm_sgs_metal(
    lib: Any,
    rho_d: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    theta_m: torch.Tensor,
    dx: float,
    dy: float,
    dz_c: torch.Tensor,
    dz_w_int: torch.Tensor,
    gravity: float = C.GRAVITY,
    c_s: float = 0.18,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute 3D Smagorinsky-Lilly SGS tendencies in 2 fused Metal kernels."""
    nz, n_col, ny, nx = rho_d.shape
    M = n_col * ny * nx
    total_pts = nz * M
    dev = rho_d.device
    dz_2d = dz_c.view(nz, -1).contiguous()
    dzw_2d = dz_w_int.view(nz - 1, -1).contiguous()
    dz_cols = dz_2d.shape[1]

    key = (dev.index or 0, nz, n_col, ny, nx, dz_cols)
    ws = _SGS_WS_CACHE.get(key)
    if ws is None:
        ws = {
            "nu": torch.empty((nz, n_col, ny, nx), device=dev, dtype=torch.float32),
            "d": torch.empty((nz, n_col, ny, nx), device=dev, dtype=torch.float32),
            "du": torch.empty((nz, n_col, ny, nx), device=dev, dtype=torch.float32),
            "dv": torch.empty((nz, n_col, ny, nx), device=dev, dtype=torch.float32),
            "dw": torch.empty((nz + 1, n_col, ny, nx), device=dev, dtype=torch.float32),
            "dThm": torch.empty((nz, n_col, ny, nx), device=dev, dtype=torch.float32),
            "u_params": torch.tensor([nz, n_col, ny, nx, dz_cols], dtype=torch.uint32, device=dev),
        }
        _cache_put(_SGS_WS_CACHE, key, ws)

    f_eddy = torch.tensor([dx, dy, gravity, c_s], dtype=torch.float32, device=dev)
    f_hw = torch.tensor([dx, dy], dtype=torch.float32, device=dev)

    r_c = rho_d.contiguous()
    u_c = u.contiguous()
    v_c = v.contiguous()
    w_c = w.contiguous()
    th_c = theta_m.contiguous()

    lib.crm_sgs_eddy_uv(
        u_c, v_c, w_c, th_c, dz_2d, dzw_2d,
        ws["nu"], ws["d"], ws["du"], ws["dv"],
        ws["u_params"], f_eddy, threads=total_pts,
    )
    lib.crm_sgs_heat_w(
        r_c, w_c, th_c, ws["nu"], ws["d"],
        ws["dw"], ws["dThm"],
        ws["u_params"], f_hw, threads=total_pts,
    )
    return ws["du"], ws["dv"], ws["dw"], ws["dThm"]


def run_crm_microphysics_metal(
    lib: Any,
    rho_d: torch.Tensor,
    theta_m_c: torch.Tensor,
    q_n: torch.Tensor,
    q_p: torch.Tensor,
    q_v_prev: torch.Tensor,
    dz_c: torch.Tensor,
    dt: float,
    mp_consts: list[float],
) -> tuple[torch.Tensor, ...]:
    """Execute column-local mixed-phase microphysics + sedimentation in 1 Metal kernel."""
    orig_shape = rho_d.shape
    nz = orig_shape[0]
    if rho_d.ndim == 4:
        _, n_col, ny, nx = orig_shape
        stride_c = ny * nx
    else:
        n_col = orig_shape[1]
        stride_c = 1
    M = n_col * stride_c
    dev = rho_d.device

    dz_2d = dz_c.view(nz, -1).contiguous()
    dz_cols = dz_2d.shape[1]

    r_c = rho_d.contiguous()
    th_c = theta_m_c.contiguous()
    qn_c = q_n.contiguous()
    qp_c = q_p.contiguous()
    qv_c = q_v_prev.contiguous()

    thm_out = torch.empty_like(r_c)
    qn_out = torch.empty_like(r_c)
    qp_out = torch.empty_like(r_c)
    temp_out = torch.empty_like(r_c)
    qv_out = torch.empty_like(r_c)
    qc_out = torch.empty_like(r_c)
    qi_out = torch.empty_like(r_c)
    sfc_shape = orig_shape[1:]
    rain_out = torch.empty(sfc_shape, device=dev, dtype=torch.float32)
    snow_out = torch.empty(sfc_shape, device=dev, dtype=torch.float32)

    u_params = torch.tensor([nz, n_col, stride_c, dz_cols], dtype=torch.uint32, device=dev)
    mp_buf = torch.tensor([dt, *mp_consts], dtype=torch.float32, device=dev)

    lib.crm_microphysics_col(
        r_c, th_c, qn_c, qp_c, qv_c, dz_2d,
        thm_out, qn_out, qp_out, temp_out, qv_out, qc_out, qi_out, rain_out, snow_out,
        u_params, mp_buf, threads=M,
    )
    return thm_out, qn_out, qp_out, temp_out, qv_out, qc_out, qi_out, rain_out, snow_out


def run_scvt_fast_acoustic_metal(
    lib: Any,
    n_ac: int,
    dt_ac: float,
    beta_hevi: float,
    gravity: float,
    accum_fluxes: bool,
    m0: torch.Tensor,
    Thm0: torch.Tensor,
    u0: torch.Tensor,
    w0: torch.Tensor,
    udotz0: torch.Tensor,
    pres_s: torch.Tensor,
    dp_dThm_s: torch.Tensor,
    Thm_ref: torch.Tensor,
    fac_z0: torch.Tensor,
    fac_z1: torch.Tensor,
    rtot_e: torch.Tensor,
    dc_e: torch.Tensor,
    tend_u_slow: torch.Tensor,
    tend_Thm_slow: torch.Tensor,
    sponge_c: torch.Tensor,
    sponge_w: torch.Tensor,
    area_e: torch.Tensor,
    zw_slope_e: torch.Tensor,
    dv_edge: torch.Tensor,
    area_cell: torch.Tensor,
    edges_on_cell: torch.Tensor,
    edge_sign_on_cell: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    m_edge: torch.Tensor,
    thm_e: torch.Tensor,
    rho_dw: torch.Tensor,
    rtot_w: torch.Tensor,
    thm_w: torch.Tensor,
    dz_c: torch.Tensor,
    dz_w_int: torch.Tensor,
    q_load: torch.Tensor,
    dl: torch.Tensor,
    diag: torch.Tensor,
    du: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute global SCVT horizontal acoustic + vertical HEVI loop on Metal GPU."""
    nz, n_cells = m0.shape
    n_edges = u0.shape[1]
    dev = m0.device
    accum_u = 1 if accum_fluxes else 0

    key = (dev.index or 0, nz, n_cells, n_edges)
    ws = _SCVT_WS_CACHE.get(key)
    if ws is None:
        ws = {
            "m_a": torch.empty_like(m0),
            "m_b": torch.empty_like(m0),
            "Thm_a": torch.empty_like(Thm0),
            "Thm_b": torch.empty_like(Thm0),
            "u_a": torch.empty_like(u0),
            "u_b": torch.empty_like(u0),
            "w_a": torch.empty_like(w0),
            "w_b": torch.empty_like(w0),
            "udz_a": torch.empty_like(udotz0),
            "udz_b": torch.empty_like(udotz0),
            "uh_k": torch.empty_like(u0),
            "fTh_h_k": torch.empty_like(u0),
            "uh_acc": torch.zeros_like(u0),
            "om_acc": torch.zeros_like(w0),
        }
        _cache_put(_SCVT_WS_CACHE, key, ws)

    ws["m_a"].copy_(m0)
    ws["Thm_a"].copy_(Thm0)
    ws["u_a"].copy_(u0)
    ws["w_a"].copy_(w0)
    ws["udz_a"].copy_(udotz0)
    if accum_fluxes:
        ws["uh_acc"].zero_()
        ws["om_acc"].zero_()

    c0_i32 = c0.to(torch.int32).contiguous()
    c1_i32 = c1.to(torch.int32).contiguous()
    eoc_i32 = edges_on_cell.to(torch.int32).contiguous()
    esgn_c = edge_sign_on_cell.contiguous()
    area_c_c = area_cell.contiguous()
    dv_c = dv_edge.contiguous()
    ae_c = area_e.view(-1).contiguous()
    dce_c = dc_e.view(-1).contiguous()
    spc_c = sponge_c.view(-1).contiguous()
    spw_c = sponge_w.view(-1).contiguous()

    u_params = torch.tensor([nz, n_cells, n_edges, accum_u], dtype=torch.uint32, device=dev)
    f_params = torch.tensor([dt_ac, beta_hevi, gravity], dtype=torch.float32, device=dev)
    hevi_c = torch.stack([rho_dw, rtot_w, thm_w, dl, diag, du], dim=0).contiguous()

    cur = (ws["m_a"], ws["Thm_a"], ws["u_a"], ws["w_a"], ws["udz_a"])
    nxt = (ws["m_b"], ws["Thm_b"], ws["u_b"], ws["w_b"], ws["udz_b"])

    for _ in range(n_ac):
        mc, Tc, uc, wc, uzc = cur
        mn, Tn, un, wn, uzn = nxt
        lib.scvt_fast_edge(
            c0_i32, c1_i32, dce_c, rtot_e.contiguous(), fac_z0.contiguous(), fac_z1.contiguous(),
            m_edge.contiguous(), thm_e.contiguous(), tend_u_slow.contiguous(), spc_c,
            pres_s.contiguous(), dp_dThm_s.contiguous(), Thm_ref.contiguous(), Tc, uc,
            un, ws["uh_k"], ws["fTh_h_k"], ws["uh_acc"],
            u_params, f_params, threads=nz * n_edges,
        )
        lib.scvt_fast_cell_hevi(
            eoc_i32, esgn_c, area_c_c, dv_c, ae_c, zw_slope_e.contiguous(), un, ws["uh_k"], ws["fTh_h_k"],
            mc, Tc, wc, uzc, pres_s.contiguous(), dp_dThm_s.contiguous(), Thm_ref.contiguous(),
            tend_Thm_slow.contiguous(), dz_c.contiguous(), dz_w_int.contiguous(), q_load.contiguous(),
            hevi_c, spw_c, mn, Tn, wn, uzn, ws["om_acc"],
            u_params, f_params, threads=n_cells,
        )
        cur, nxt = nxt, cur

    mc, Tc, uc, wc, _ = cur
    return mc.clone(), Tc.clone(), uc.clone(), wc.clone(), ws["uh_acc"].clone(), ws["om_acc"].clone()


def run_crm_mynn_metal(
    lib: Any,
    rho_d: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    theta_m_c: torch.Tensor,
    q_n: torch.Tensor,
    q_v: torch.Tensor,
    q_c: torch.Tensor,
    q_i: torch.Tensor,
    qke: torch.Tensor,
    z_c: torch.Tensor,
    z_w: torch.Tensor,
    dz_c: torch.Tensor,
    u_star: torch.Tensor,
    sfc_buoy_flux: torch.Tensor,
    tau_x_sfc: torch.Tensor,
    tau_y_sfc: torch.Tensor,
    shf_sfc: torch.Tensor,
    evap_sfc: torch.Tensor,
    dt: float,
    mynn_consts: list[float],
) -> tuple[torch.Tensor, ...]:
    """Execute column-local MYNN 2.5 PBL turbulence in 1 fused Metal kernel."""
    orig_shape = rho_d.shape
    nz = orig_shape[0]
    M = rho_d[0].numel()
    dev = rho_d.device

    zc_2d = z_c.view(nz, -1).contiguous()
    zw_2d = z_w.view(nz + 1, -1).contiguous()
    dz_2d = dz_c.view(nz, -1).contiguous()
    dz_cols = dz_2d.shape[1]

    r_c = rho_d.contiguous()
    u_c = u.contiguous()
    v_c = v.contiguous()
    th_c = theta_m_c.contiguous()
    qn_c = q_n.contiguous()
    qv_c = q_v.contiguous()
    qc_c = q_c.contiguous()
    qi_c = q_i.contiguous()
    qke_c = qke.contiguous()

    sfc6 = torch.stack([
        u_star.reshape(M),
        sfc_buoy_flux.reshape(M),
        tau_x_sfc.reshape(M),
        tau_y_sfc.reshape(M),
        shf_sfc.reshape(M),
        evap_sfc.reshape(M),
    ], dim=0).contiguous()

    u_out = torch.empty_like(r_c)
    v_out = torch.empty_like(r_c)
    thm_out = torch.empty_like(r_c)
    qn_out = torch.empty_like(r_c)
    qke_out = torch.empty_like(r_c)
    km_out = torch.empty((nz - 1, *orig_shape[1:]), device=dev, dtype=torch.float32)
    kh_out = torch.empty_like(km_out)
    qv_out = torch.empty_like(r_c)
    qc_out = torch.empty_like(r_c)
    qi_out = torch.empty_like(r_c)

    u_params = torch.tensor([nz, M, dz_cols], dtype=torch.uint32, device=dev)
    mp_buf = torch.tensor([dt, *mynn_consts], dtype=torch.float32, device=dev)

    lib.crm_mynn_col(
        r_c, u_c, v_c, th_c, qn_c, qv_c, qc_c, qi_c, qke_c,
        zc_2d, zw_2d, dz_2d, sfc6,
        u_out, v_out, thm_out, qn_out, qke_out, km_out, kh_out, qv_out, qc_out, qi_out,
        u_params, mp_buf, threads=M,
    )
    return u_out, v_out, thm_out, qn_out, qke_out, km_out, kh_out, qv_out, qc_out, qi_out


def _get_scvt_mesh_cache(mesh: Any) -> dict[str, Any]:
    mc = getattr(mesh, "_metal_ops_cache", None)
    if mc is None:
        c0 = mesh.cells_on_edge[:, 0].to(torch.int32).contiguous()
        c1 = mesh.cells_on_edge[:, 1].to(torch.int32).contiguous()
        v0 = mesh.verts_on_edge[:, 0].to(torch.int32).contiguous()
        v1 = mesh.verts_on_edge[:, 1].to(torch.int32).contiguous()
        eoc = mesh.edges_on_cell.to(torch.int32).contiguous()
        eov = mesh.edges_on_vert.to(torch.int32).contiguous()
        ep = mesh.trisk_edges.to(torch.int32).contiguous()
        div_w = (mesh.edge_sign_on_cell * mesh.dv_edge[mesh.edges_on_cell] / mesh.area_cell.unsqueeze(-1)).contiguous()
        vort_w = (mesh.edge_sign_on_vert * mesh.dc_edge[mesh.edges_on_vert] / mesh.area_vert.unsqueeze(-1)).contiguous()
        inv_dc = (1.0 / mesh.dc_edge).contiguous()
        inv_dv = (1.0 / torch.clamp(mesh.dv_edge, min=100.0)).contiguous()
        mc = {
            "c0": c0, "c1": c1, "v0": v0, "v1": v1,
            "eoc": eoc, "eov": eov, "ep": ep,
            "w_trisk": mesh.trisk_weights.contiguous(),
            "f_vert": mesh.f_vert.contiguous(),
            "div_w": div_w, "vort_w": vort_w,
            "inv_dc": inv_dc, "inv_dv": inv_dv,
            "n_cells": int(mesh.n_cells),
            "n_edges": int(mesh.n_edges),
            "n_verts": int(mesh.n_verts),
            "u_cache": {},
        }
        mesh._metal_ops_cache = mc
    return mc


def _scvt_u_param(mc: dict[str, Any], key: tuple[Any, ...], vals: list[int], dev: torch.device) -> torch.Tensor:
    t = mc["u_cache"].get(key)
    if t is None:
        t = torch.tensor(vals, dtype=torch.uint32, device=dev)
        mc["u_cache"][key] = t
    return t


def run_scvt_div_metal(lib: Any, mesh: Any, flux_edge: torch.Tensor) -> torch.Tensor:
    mc = _get_scvt_mesh_cache(mesh)
    nz = flux_edge.shape[0]
    nc = mc["n_cells"]
    out = torch.empty((nz, nc), device=flux_edge.device, dtype=torch.float32)
    up = _scvt_u_param(mc, ("div", nz), [nz, nc, mc["n_edges"]], flux_edge.device)
    lib.scvt_div_cell(mc["eoc"], mc["div_w"], flux_edge.contiguous(), out, up, threads=nz * nc)
    return out


def run_scvt_vorticity_metal(lib: Any, mesh: Any, u_edge: torch.Tensor) -> torch.Tensor:
    mc = _get_scvt_mesh_cache(mesh)
    nz = u_edge.shape[0]
    nv = mc["n_verts"]
    out = torch.empty((nz, nv), device=u_edge.device, dtype=torch.float32)
    up = _scvt_u_param(mc, ("vor", nz), [nz, nv, mc["n_edges"]], u_edge.device)
    lib.scvt_vorticity_vert(mc["eov"], mc["vort_w"], u_edge.contiguous(), out, up, threads=nz * nv)
    return out


def run_scvt_vec_lap_metal(lib: Any, mesh: Any, u_edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mc = _get_scvt_mesh_cache(mesh)
    nz = u_edge.shape[0]
    ne = mc["n_edges"]
    div_c = run_scvt_div_metal(lib, mesh, u_edge)
    zeta_v = run_scvt_vorticity_metal(lib, mesh, u_edge)
    lap = torch.empty((nz, ne), device=u_edge.device, dtype=torch.float32)
    gd = torch.empty((nz, ne), device=u_edge.device, dtype=torch.float32)
    up = _scvt_u_param(mc, ("vlap", nz), [nz, mc["n_cells"], mc["n_verts"], ne], u_edge.device)
    lib.scvt_vec_lap_edge(
        mc["c0"], mc["c1"], mc["v0"], mc["v1"], mc["inv_dc"], mc["inv_dv"],
        div_c, zeta_v, lap, gd, up, threads=nz * ne,
    )
    return lap, gd


def run_scvt_scalar_lap_metal(lib: Any, mesh: Any, phi_cell: torch.Tensor) -> torch.Tensor:
    mc = _get_scvt_mesh_cache(mesh)
    nz = phi_cell.shape[0]
    ne = mc["n_edges"]
    grad_e = torch.empty((nz, ne), device=phi_cell.device, dtype=torch.float32)
    up = _scvt_u_param(mc, ("sgrad", nz), [nz, mc["n_cells"], ne], phi_cell.device)
    lib.scvt_scalar_grad_edge(mc["c0"], mc["c1"], mc["inv_dc"], phi_cell.contiguous(), grad_e, up, threads=nz * ne)
    return run_scvt_div_metal(lib, mesh, grad_e)


def run_scvt_trisk_rot_metal(lib: Any, mesh: Any, zeta_v: torch.Tensor, u_edge: torch.Tensor) -> torch.Tensor:
    mc = _get_scvt_mesh_cache(mesh)
    nz = u_edge.shape[0]
    ne = mc["n_edges"]
    rot_out = torch.empty((nz, ne), device=u_edge.device, dtype=torch.float32)
    up = _scvt_u_param(mc, ("rot", nz), [nz, mc["n_verts"], ne], u_edge.device)
    lib.scvt_trisk_rot_edge(
        mc["v0"], mc["v1"], mc["ep"], mc["w_trisk"], zeta_v.contiguous(), mc["f_vert"],
        u_edge.contiguous(), rot_out, up, threads=nz * ne,
    )
    return rot_out
