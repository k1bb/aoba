"""Single-source Metal/CUDA fused compute kernels for AOBA's universal ESM modules.

Accelerates AOBA's newly added high-complexity ESM physics modules on both Apple Silicon (`mps`)
and NVIDIA GPUs (`cuda`, via automatic transpilation in `aoba.kernels_cuda`), eliminating
GPU-starving Python vertical/g-point loops and mid-step CPU-GPU synchronizations:

1. **`rad_rrtmgp_lw_grp` & `rad_rrtmgp_sw_grp`** (`rad_scheme = "rrtmgp"`):
   Official BSD-3 HITRAN + MT_CKD 4.1 RRTMGP radiative transfer (128 LW + 112 SW = 240 g-points).
   Dispatches one 128-thread threadgroup per atmospheric column (`g = 0..127` LW / `0..111` SW)
   with CSR-indexed minor/continuum gases, exact Pincus et al. (2019) trilinear `(T, ln p, eta)`
   interpolation, exact Toon (1989) delta-Eddington SW 2-stream, zero-divergence nighttime early
   exit, and single-barrier SIMD-warp reductions.
2. **`rad_ckd_hd_col`** (`rad_scheme = "ckd_hd"`):
   29-band / 111-g-point MSTRN11-class correlated-k radiation (72 LW + 39 SW) with Chou et al.
   (1999) LW cloud scattering and 5-species aerosol optics.
3. **`microphysics_6class_col`** (`microphysics_scheme = "six_class_2m"`):
   Single-pass top-down fused 6-class 2-moment Morrison (2005) / KK00 microphysics with coupled
   3-species implicit upwind sedimentation (`q_r/n_r, q_s/n_s, q_g/n_g`) in registers.
4. **`ozone_chem_col`** (`o3_mode = "prognostic"`):
   Fused 3D prognostic ozone photochemistry (Chapman + NOx + HOx + PSC + tropospheric CH4/LNOx/
   wildfire + exact Riccati-exponential + Wesely dry deposition + 6-layer PBL mixing).
"""

from __future__ import annotations

import math
from typing import Any
import torch

from aoba import constants as C


_ESM_TABLE_CACHE: dict[tuple[Any, ...], Any] = {}


ESM_METAL_SHADER_SRC: str = r"""
// =========================================================================
// 20. Universal ESM Fused Kernels (RRTMGP 240g, CKD-HD 111g, 6-Class 2M, O3)
// =========================================================================

inline float esm_expm1(float x) {
    if (fabs(x) < 1.0e-4f) {
        return x * (1.0f + x * (0.5f + x * (1.0f / 6.0f)));
    }
    return exp(x) - 1.0f;
}

inline float esm_warp_sum(float v) {
    v += simd_shuffle_xor(v, ushort(16));
    v += simd_shuffle_xor(v, ushort(8));
    v += simd_shuffle_xor(v, ushort(4));
    v += simd_shuffle_xor(v, ushort(2));
    v += simd_shuffle_xor(v, ushort(1));
    return v;
}

inline float esm_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -40.0f, 40.0f)));
}

inline float esm_totplnk(device const float* totplnk, uint b, float tk, float pi_val) {
    float t_pos = clamp(tk - 160.0f, 0.0f, 194.999f);
    uint i0 = uint(floor(t_pos));
    float w = t_pos - float(i0);
    uint base = b * 196u + i0;
    return pi_val * ((1.0f - w) * totplnk[base] + w * totplnk[base + 1u]);
}

// -------------------------------------------------------------------------
// 20a. Official HITRAN / MT_CKD 4.1 RRTMGP Longwave (16 bands / 128 g-pts)
//      1 Threadgroup (128 threads = 128 g-points) per column
// -------------------------------------------------------------------------
kernel void rad_rrtmgp_lw_grp(
    device const float* pres_c      [[buffer(0)]],
    device const float* temp_c      [[buffer(1)]],
    device const float* temp_w      [[buffer(2)]],
    device const float* dz_c        [[buffer(3)]],
    device const float* rho_d_c     [[buffer(4)]],
    device const float* qv_c        [[buffer(5)]],
    device const float* qc_c        [[buffer(6)]],
    device const float* qi_c        [[buffer(7)]],
    device const float* o3_vmr_c    [[buffer(8)]],
    device const float* q_dust_c    [[buffer(9)]],
    device const float* re_liq_c    [[buffer(10)]],
    device const float* re_ice_c    [[buffer(11)]],
    device const float* aero_lw_c   [[buffer(12)]],
    device const float* sfc_in      [[buffer(13)]],
    device float* lw_up_out         [[buffer(14)]],
    device float* lw_dn_out         [[buffer(15)]],
    device float* q_lw_out          [[buffer(16)]],
    device const float* kmajor      [[buffer(17)]],
    device const float* plank_frac  [[buffer(18)]],
    device const float* kminor_all  [[buffer(19)]],
    device const float* tbl_f       [[buffer(20)]],
    device const int* tbl_i         [[buffer(21)]],
    constant ulong4& u_p0           [[buffer(22)]], // [N, M, dz_cols, flags]
    constant ulong4& u_p1           [[buffer(23)]], // [n_temp, n_press, n_eta, n_contrib_tot]
    constant float4& f_p0           [[buffer(24)]], // [co2_vmr, ch4_vmr, n2o_vmr, o2_vmr]
    constant float4& f_p1           [[buffer(25)]], // [re_l_sc, re_i_sc, co_vmr, n2_vmr]
    uint g                          [[thread_position_in_threadgroup]],
    uint col                        [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    uint N = uint(u_p0.x);
    uint M = uint(u_p0.y);
    uint dz_cols = uint(u_p0.z);
    uint flags = uint(u_p0.w);
    if (col >= M || N == 0u || N > RAD_MAX_NZ) return;

    uint n_temp = uint(u_p1.x);
    uint n_press = uint(u_p1.y);
    uint n_eta = uint(u_p1.z);
    uint n_ctot = uint(u_p1.w);
    uint n_p1 = n_press + 1u;

    bool has_dust   = (flags & 1u) != 0u;
    bool has_o3     = (flags & 2u) != 0u;
    bool o3_lw_cpl  = (flags & 4u) != 0u;
    bool has_re_l3d = (flags & 8u) != 0u;
    bool has_re_i3d = (flags & 16u) != 0u;
    bool has_aero   = (flags & 32u) != 0u;
    uint dz_col = (dz_cols == 1u) ? 0u : col;

    float pr_ln0  = tbl_f[0];
    float pr_dln  = tbl_f[1];
    float pr_trop = tbl_f[2];
    float tr_0    = tbl_f[3];
    float tr_d    = tbl_f[4];
    float ref_p   = tbl_f[5];
    float ref_t   = tbl_f[6];
    float m_air_m = tbl_f[7];
    float cp_d    = tbl_f[8];
    float o3_g1   = tbl_f[9];
    float o3_g2   = tbl_f[10];
    float o3_g3   = tbl_f[11];
    float pi_val  = tbl_f[12];

    uint b = uint(tbl_i[g]); // 0..15
    float diff_b   = tbl_f[16u + b];
    float k_liq_b  = tbl_f[32u + b];
    float k_liq_r  = tbl_f[48u + b];
    float k_ice_b  = tbl_f[64u + b];
    float k_ice_r  = tbl_f[80u + b];
    float k_dust_b = tbl_f[96u + b];
    device const float* totplnk = tbl_f + 112u;
    device const float* vmr_ref = tbl_f + 3248u; // [n_temp, 20, 2]

    uint ks_base = 128u + b * 4u;
    uint s1_l = uint(tbl_i[ks_base + 0u]);
    uint s2_l = uint(tbl_i[ks_base + 1u]);
    uint s1_u = uint(tbl_i[ks_base + 2u]);
    uint s2_u = uint(tbl_i[ks_base + 3u]);

    uint moff_l0 = uint(tbl_i[192u + g]);
    uint moff_l1 = uint(tbl_i[192u + g + 1u]);
    uint moff_u0 = uint(tbl_i[321u + g]);
    uint moff_u1 = uint(tbl_i[321u + g + 1u]);
    device const int* ment = tbl_i + 450u;

    float co2_vmr = f_p0.x;
    float ch4_vmr = f_p0.y;
    float n2o_vmr = f_p0.z;
    float o2_vmr  = f_p0.w;
    float re_l_sc = f_p1.x;
    float re_i_sc = f_p1.y;
    float co_vmr  = f_p1.z;
    float n2_vmr  = f_p1.w;

    float ts_c      = max(sfc_in[0u * M + col], 100.0f);
    float emiss_sfc = clamp(sfc_in[7u * M + col], 0.0f, 1.0f);

    uint warp = g >> 5u;
    threadgroup float sh_warp[4];

    float tr_arr[RAD_MAX_NZ];
    float sup_arr[RAD_MAX_NZ];

    if (g == 0u) {
        lw_dn_out[N * M + col] = 0.0f;
    }

    float f_dn = 0.0f;
    float pfrac_bot = 0.0f;
    float b_lev_prev = esm_totplnk(totplnk, b, max(temp_w[N * M + col], 100.0f), pi_val);

    uint km_stride_e = 128u;
    uint km_stride_p = n_eta * 128u;
    uint km_stride_t = n_p1 * km_stride_p;
    uint mn_stride_e = n_ctot;
    uint mn_stride_t = n_eta * n_ctot;
    float eta_max = float(n_eta - 1u);

    for (uint n = 0u; n < N; ++n) {
        uint k = N - 1u - n;
        uint idx = k * M + col;

        float pk = pres_c[idx];
        float tk = temp_c[idx];
        float dzk = dz_c[k * dz_cols + dz_col];
        float dm_dry = max(rho_d_c[idx] * dzk, 1.0e-5f);
        float col_dry_e20 = (dm_dry / m_air_m) * 1.0e-24f;
        float qvk = max(qv_c[idx], 0.0f);
        float qck = max(qc_c[idx], 0.0f);
        float qik = max(qi_c[idx], 0.0f);

        float o3v;
        if (has_o3) {
            o3v = max(o3_vmr_c[idx], 0.0f);
        } else {
            float ph = max(pk * 0.01f, 1.0e-3f);
            o3v = o3_g1 * pow(ph, o3_g2) * exp(-ph / o3_g3) * 1.0e-6f;
        }
        float strat_fall = pow(min(pk / 8000.0f, 1.0f), 0.38f);
        float vmr[9];
        vmr[0] = 1.0f;
        vmr[1] = max(qvk * (28.9647f / 18.01528f), 1.0e-7f);
        vmr[2] = co2_vmr;
        vmr[3] = o3_lw_cpl ? o3v : 0.0f;
        vmr[4] = n2o_vmr * strat_fall;
        vmr[5] = co_vmr;
        vmr[6] = ch4_vmr * strat_fall;
        vmr[7] = o2_vmr;
        vmr[8] = n2_vmr;

        float t_pos = (tk - tr_0) / tr_d;
        int jt0_i = clamp(int(floor(t_pos)), 0, int(n_temp) - 2);
        uint jt0 = uint(jt0_i);
        uint jt1 = jt0 + 1u;
        float ft = clamp(t_pos - float(jt0_i), 0.0f, 1.0f);

        float ln_p = log(max(pk, 0.5f));
        float p_pos = (ln_p - pr_ln0) / pr_dln;
        bool is_upper = (pk <= pr_trop);
        uint tropo = is_upper ? 1u : 0u;
        uint jp0_km;
        float fp;
        if (is_upper) {
            int jp_u = clamp(int(floor(p_pos)), 12, int(n_press) - 2);
            jp0_km = uint(jp_u + 1);
            fp = clamp(p_pos - float(jp_u), 0.0f, 1.0f);
        } else {
            int jp_l = clamp(int(floor(p_pos)), 0, 11);
            jp0_km = uint(jp_l);
            fp = clamp(p_pos - float(jp_l), 0.0f, 1.0f);
        }
        uint jp1_km = jp0_km + 1u;

        uint s1 = is_upper ? s1_u : s1_l;
        uint s2 = is_upper ? s2_u : s2_l;
        bool has_key = (s1 > 0u) || (s2 > 0u);
        float v1 = vmr[s1];
        float v2 = vmr[s2];

        float vr1_0 = vmr_ref[(jt0 * 20u + s1) * 2u + tropo];
        float vr2_0 = vmr_ref[(jt0 * 20u + s2) * 2u + tropo];
        float vmix0 = v1 + (vr1_0 / max(vr2_0, 1.0e-30f)) * v2;
        float eta0 = (has_key && vmix0 > 1.0e-30f) ? (v1 / max(vmix0, 1.0e-30f)) : 0.0f;
        float es0 = clamp(eta0 * eta_max, 0.0f, eta_max - 1.0e-5f);
        uint je0_0 = uint(clamp(int(floor(es0)), 0, int(n_eta) - 2));
        float fe_0 = clamp(es0 - float(je0_0), 0.0f, 1.0f);
        float cmix0 = has_key ? (col_dry_e20 * vmix0) : 0.0f;

        float vr1_1 = vmr_ref[(jt1 * 20u + s1) * 2u + tropo];
        float vr2_1 = vmr_ref[(jt1 * 20u + s2) * 2u + tropo];
        float vmix1 = v1 + (vr1_1 / max(vr2_1, 1.0e-30f)) * v2;
        float eta1 = (has_key && vmix1 > 1.0e-30f) ? (v1 / max(vmix1, 1.0e-30f)) : 0.0f;
        float es1 = clamp(eta1 * eta_max, 0.0f, eta_max - 1.0e-5f);
        uint je0_1 = uint(clamp(int(floor(es1)), 0, int(n_eta) - 2));
        float fe_1 = clamp(es1 - float(je0_1), 0.0f, 1.0f);
        float cmix1 = has_key ? (col_dry_e20 * vmix1) : 0.0f;

        uint idx00 = jt0 * km_stride_t + jp0_km * km_stride_p + je0_0 * km_stride_e + g;
        uint idx01 = jt0 * km_stride_t + jp1_km * km_stride_p + je0_0 * km_stride_e + g;
        uint idx10 = jt1 * km_stride_t + jp0_km * km_stride_p + je0_1 * km_stride_e + g;
        uint idx11 = jt1 * km_stride_t + jp1_km * km_stride_p + je0_1 * km_stride_e + g;

        float k_t0 = (1.0f - fp) * ((1.0f - fe_0) * kmajor[idx00] + fe_0 * kmajor[idx00 + km_stride_e])
                   + fp          * ((1.0f - fe_0) * kmajor[idx01] + fe_0 * kmajor[idx01 + km_stride_e]);
        float k_t1 = (1.0f - fp) * ((1.0f - fe_1) * kmajor[idx10] + fe_1 * kmajor[idx10 + km_stride_e])
                   + fp          * ((1.0f - fe_1) * kmajor[idx11] + fe_1 * kmajor[idx11 + km_stride_e]);
        float tau_gas = max((1.0f - ft) * (k_t0 * cmix0) + ft * (k_t1 * cmix1), 0.0f);

        float pf0 = (1.0f - fp) * ((1.0f - fe_0) * plank_frac[idx00] + fe_0 * plank_frac[idx00 + km_stride_e])
                  + fp          * ((1.0f - fe_0) * plank_frac[idx01] + fe_0 * plank_frac[idx01 + km_stride_e]);
        float pf1 = (1.0f - fp) * ((1.0f - fe_1) * plank_frac[idx10] + fe_1 * plank_frac[idx10 + km_stride_e])
                  + fp          * ((1.0f - fe_1) * plank_frac[idx11] + fe_1 * plank_frac[idx11 + km_stride_e]);
        float pfrac = max((1.0f - ft) * pf0 + ft * pf1, 0.0f);
        if (n == N - 1u) {
            pfrac_bot = pfrac;
        }

        uint m0 = is_upper ? moff_u0 : moff_l0;
        uint m1 = is_upper ? moff_u1 : moff_l1;
        if (m0 < m1) {
            float dens_fac = (pk / max(tk, 100.0f)) * (ref_t / ref_p);
            float inv_1ph2o = 1.0f / (1.0f + vmr[1]);
            for (uint me = m0; me < m1; ++me) {
                uint e4 = me * 4u;
                uint g_idx = uint(ment[e4 + 0u]);
                uint s_idx = uint(ment[e4 + 1u]);
                uint mflg  = uint(ment[e4 + 2u]);
                uint k_col = uint(ment[e4 + 3u]);

                float g_col = col_dry_e20 * vmr[g_idx];
                if ((mflg & 2u) != 0u) g_col *= dens_fac;
                if (s_idx > 0u) {
                    float vs = vmr[s_idx];
                    g_col *= ((mflg & 1u) != 0u) ? ((1.0f - vs) * inv_1ph2o) : (vs * inv_1ph2o);
                }
                uint mi0 = jt0 * mn_stride_t + je0_0 * mn_stride_e + k_col;
                uint mi1 = jt1 * mn_stride_t + je0_1 * mn_stride_e + k_col;
                float km0 = (1.0f - fe_0) * kminor_all[mi0] + fe_0 * kminor_all[mi0 + mn_stride_e];
                float km1 = (1.0f - fe_1) * kminor_all[mi1] + fe_1 * kminor_all[mi1 + mn_stride_e];
                tau_gas += max(((1.0f - ft) * km0 + ft * km1) * g_col, 0.0f);
            }
        }

        float tau_cld = 0.0f;
        if (qck > 0.0f) {
            float rel = has_re_l3d ? clamp(re_liq_c[idx], 3.0f, 40.0f) : re_l_sc;
            tau_cld += (k_liq_b + k_liq_r / rel) * (1000.0f * qck * dm_dry);
        }
        if (qik > 0.0f) {
            float rei = has_re_i3d ? clamp(re_ice_c[idx], 10.0f, 150.0f) : re_i_sc;
            tau_cld += (k_ice_b + k_ice_r / rei) * (1000.0f * qik * dm_dry);
        }
        if (has_dust) {
            tau_cld += k_dust_b * (max(q_dust_c[idx], 0.0f) * dm_dry);
        }
        if (has_aero) {
            tau_cld += max(aero_lw_c[idx], 0.0f);
        }

        float dtau = clamp(diff_b * (tau_gas + tau_cld), 0.0f, 50.0f);
        float trans = exp(-dtau);
        float em_l = -esm_expm1(-dtau);

        float b_lev_bot = esm_totplnk(totplnk, b, max(temp_w[k * M + col], 100.0f), pi_val);
        float b_top_g = b_lev_prev * pfrac;
        float b_bot_g = b_lev_bot * pfrac;
        b_lev_prev = b_lev_bot;

        float w_lin = (dtau > 1.0e-3f)
            ? ((1.0f / max(dtau, 1.0e-3f)) - trans / max(em_l, 1.0e-6f))
            : (0.5f - dtau * (1.0f / 12.0f));
        float s_dn = em_l * ((1.0f - w_lin) * b_bot_g + w_lin * b_top_g);
        float s_up = em_l * ((1.0f - w_lin) * b_top_g + w_lin * b_bot_g);

        f_dn = f_dn * trans + s_dn;
        tr_arr[n] = trans;
        sup_arr[n] = s_up;

        float wsum = esm_warp_sum(f_dn);
        if (lane == 0u) sh_warp[warp] = wsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) {
            lw_dn_out[k * M + col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float b_sfc_g = esm_totplnk(totplnk, b, ts_c, pi_val) * pfrac_bot;
    float f_up = emiss_sfc * b_sfc_g + (1.0f - emiss_sfc) * f_dn;
    {
        float wsum = esm_warp_sum(f_up);
        if (lane == 0u) sh_warp[warp] = wsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) {
            lw_up_out[col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int in = int(N) - 1; in >= 0; --in) {
        uint n = uint(in);
        f_up = f_up * tr_arr[n] + sup_arr[n];
        float wsum = esm_warp_sum(f_up);
        if (lane == 0u) sh_warp[warp] = wsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) {
            lw_up_out[(N - n) * M + col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (g < N) {
        uint idx = g * M + col;
        uint up_idx = (g + 1u) * M + col;
        float net_bot = lw_dn_out[idx] - lw_up_out[idx];
        float net_top = lw_dn_out[up_idx] - lw_up_out[up_idx];
        float dm_bu = max(rho_d_c[idx] * dz_c[g * dz_cols + dz_col], 1.0e-5f);
        q_lw_out[idx] = (net_top - net_bot) / (cp_d * dm_bu);
    }
}

// -------------------------------------------------------------------------
// 20b. Official HITRAN / MT_CKD 4.1 RRTMGP Shortwave (14 bands / 112 g-pts)
//      1 Threadgroup (128 threads; g=0..111 active) per column
// -------------------------------------------------------------------------
kernel void rad_rrtmgp_sw_grp(
    device const float* pres_c      [[buffer(0)]],
    device const float* temp_c      [[buffer(1)]],
    device const float* dz_c        [[buffer(2)]],
    device const float* rho_d_c     [[buffer(3)]],
    device const float* qv_c        [[buffer(4)]],
    device const float* qc_c        [[buffer(5)]],
    device const float* qi_c        [[buffer(6)]],
    device const float* o3_vmr_c    [[buffer(7)]],
    device const float* q_dust_c    [[buffer(8)]],
    device const float* re_liq_c    [[buffer(9)]],
    device const float* re_ice_c    [[buffer(10)]],
    device const float* aero_tau_sw [[buffer(11)]],
    device const float* aero_sct_sw [[buffer(12)]],
    device const float* aero_asy_sw [[buffer(13)]],
    device const float* sfc_in      [[buffer(14)]],
    device float* sw_up_out         [[buffer(15)]],
    device float* sw_dn_out         [[buffer(16)]],
    device float* q_sw_out          [[buffer(17)]],
    device float* sfc_out           [[buffer(18)]],
    device const float* kmajor      [[buffer(19)]],
    device const float* kminor_all  [[buffer(20)]],
    device const float* rayl_all    [[buffer(21)]],
    device const float* tbl_f       [[buffer(22)]],
    device const int* tbl_i         [[buffer(23)]],
    constant ulong4& u_p0           [[buffer(24)]], // [N, M, dz_cols, flags]
    constant ulong4& u_p1           [[buffer(25)]], // [n_temp, n_press, n_eta, n_contrib_tot]
    constant float4& f_p0           [[buffer(26)]], // [co2_vmr, ch4_vmr, n2o_vmr, o2_vmr]
    constant float4& f_p1           [[buffer(27)]], // [re_l_sc, re_i_sc, co_vmr, n2_vmr]
    constant float4& f_p2           [[buffer(28)]], // [chi_sw, 0, 0, 0]
    uint g                          [[thread_position_in_threadgroup]],
    uint col                        [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    uint N = uint(u_p0.x);
    uint M = uint(u_p0.y);
    uint dz_cols = uint(u_p0.z);
    uint flags = uint(u_p0.w);
    if (col >= M || N == 0u || N > RAD_MAX_NZ) return;

    float mu0_in    = sfc_in[5u * M + col];
    float s_toa_col = sfc_in[6u * M + col];
    if (mu0_in <= 1.0e-4f || s_toa_col <= 1.0e-4f) {
        if (g <= N) {
            sw_up_out[g * M + col] = 0.0f;
            sw_dn_out[g * M + col] = 0.0f;
        }
        if (g < N) {
            q_sw_out[g * M + col] = 0.0f;
        }
        if (g < 5u) {
            sfc_out[g * M + col] = 0.0f;
        }
        return;
    }

    uint n_temp = uint(u_p1.x);
    uint n_press = uint(u_p1.y);
    uint n_eta = uint(u_p1.z);
    uint n_ctot = uint(u_p1.w);
    uint n_p1 = n_press + 1u;

    bool has_dust   = (flags & 1u) != 0u;
    bool has_o3     = (flags & 2u) != 0u;
    bool has_re_l3d = (flags & 8u) != 0u;
    bool has_re_i3d = (flags & 16u) != 0u;
    bool has_aero   = (flags & 32u) != 0u;
    uint dz_col = (dz_cols == 1u) ? 0u : col;

    float pr_ln0  = tbl_f[0];
    float pr_dln  = tbl_f[1];
    float pr_trop = tbl_f[2];
    float tr_0    = tbl_f[3];
    float tr_d    = tbl_f[4];
    float ref_p   = tbl_f[5];
    float ref_t   = tbl_f[6];
    float m_air_m = tbl_f[7];
    float cp_d    = tbl_f[8];
    float o3_g1   = tbl_f[9];
    float o3_g2   = tbl_f[10];
    float o3_g3   = tbl_f[11];

    float co2_vmr = f_p0.x;
    float ch4_vmr = f_p0.y;
    float n2o_vmr = f_p0.z;
    float o2_vmr  = f_p0.w;
    float re_l_sc = f_p1.x;
    float re_i_sc = f_p1.y;
    float co_vmr  = f_p1.z;
    float n2_vmr  = f_p1.w;
    float chi_sw  = f_p2.x;

    float alb_vis_d = clamp(sfc_in[1u * M + col], 0.0f, 1.0f);
    float alb_nir_d = clamp(sfc_in[2u * M + col], 0.0f, 1.0f);
    float alb_vis_f = clamp(sfc_in[3u * M + col], 0.0f, 1.0f);
    float alb_nir_f = clamp(sfc_in[4u * M + col], 0.0f, 1.0f);

    uint warp = g >> 5u;
    threadgroup float sh_warp[4];

    float c_star[2 * RAD_MAX_NZ];
    float d_star[2 * RAD_MAX_NZ];
    float bg_a[RAD_MAX_NZ];
    float el_a[RAD_MAX_NZ];
    float att_a[RAD_MAX_NZ];
    float cp_a[RAD_MAX_NZ];
    float cm_a[RAD_MAX_NZ];

    float mu0_ts = max(mu0_in, 1.0e-4f);
    float inv_mu0 = 1.0f / mu0_ts;
    float sqrt3 = 1.73205081f;

    bool active = (g < 112u);
    bool is_vis = false;
    bool is_par = false;
    float f_beam0 = 0.0f;
    float cp_c0 = 0.0f;

    if (active) {
        uint b = uint(tbl_i[g]); // 0..13
        is_vis = (tbl_i[112u + g] != 0);
        is_par = (tbl_i[224u + g] != 0);

        float solar_f = tbl_f[16u + g];
        float aero_sc = tbl_f[128u + b];
        float dust_ke = tbl_f[142u + b];
        float dust_ss = tbl_f[156u + b];
        float cld_oml = tbl_f[170u + b];
        float cld_omi = tbl_f[184u + b];
        device const float* vmr_ref = tbl_f + 198u;

        uint ks_base = 336u + b * 4u;
        uint s1_l = uint(tbl_i[ks_base + 0u]);
        uint s2_l = uint(tbl_i[ks_base + 1u]);
        uint s1_u = uint(tbl_i[ks_base + 2u]);
        uint s2_u = uint(tbl_i[ks_base + 3u]);

        uint moff_l0 = uint(tbl_i[392u + g]);
        uint moff_l1 = uint(tbl_i[392u + g + 1u]);
        uint moff_u0 = uint(tbl_i[505u + g]);
        uint moff_u1 = uint(tbl_i[505u + g + 1u]);
        device const int* ment = tbl_i + 618u;

        f_beam0 = s_toa_col * solar_f;
        float r_dir = is_vis ? alb_vis_d : alb_nir_d;
        float r_sfc = is_vis ? alb_vis_f : alb_nir_f;

        uint km_stride_e = 112u;
        uint km_stride_p = n_eta * 112u;
        uint km_stride_t = n_p1 * km_stride_p;
        uint mn_stride_e = n_ctot;
        uint mn_stride_t = n_eta * n_ctot;
        uint ry_stride_e = 112u;
        uint ry_stride_t = n_eta * 112u;
        uint ry_half = n_temp * ry_stride_t;
        float eta_max = float(n_eta - 1u);

        float tau_cum = 0.0f;
        float att_prev = 1.0f;
        float e1_p = 0.0f, e2_p = 0.0f, e3_p = 0.0f, e4_p = 0.0f;
        float cp_t_p = 0.0f, cm_t_p = 0.0f;

        for (uint n = 0u; n < N; ++n) {
            uint k = N - 1u - n;
            uint idx = k * M + col;

            float pk = pres_c[idx];
            float tk = temp_c[idx];
            float dzk = dz_c[k * dz_cols + dz_col];
            float dm_dry = max(rho_d_c[idx] * dzk, 1.0e-5f);
            float col_dry_e20 = (dm_dry / m_air_m) * 1.0e-24f;
            float qvk = max(qv_c[idx], 0.0f);
            float qck = max(qc_c[idx], 0.0f);
            float qik = max(qi_c[idx], 0.0f);

            float o3v;
            if (has_o3) {
                o3v = max(o3_vmr_c[idx], 0.0f);
            } else {
                float ph = max(pk * 0.01f, 1.0e-3f);
                o3v = o3_g1 * pow(ph, o3_g2) * exp(-ph / o3_g3) * 1.0e-6f;
            }
            float strat_fall = pow(min(pk / 8000.0f, 1.0f), 0.38f);
            float vmr[9];
            vmr[0] = 1.0f;
            vmr[1] = max(qvk * (28.9647f / 18.01528f), 1.0e-7f);
            vmr[2] = co2_vmr;
            vmr[3] = o3v;
            vmr[4] = n2o_vmr * strat_fall;
            vmr[5] = co_vmr;
            vmr[6] = ch4_vmr * strat_fall;
            vmr[7] = o2_vmr;
            vmr[8] = n2_vmr;

            float t_pos = (tk - tr_0) / tr_d;
            int jt0_i = clamp(int(floor(t_pos)), 0, int(n_temp) - 2);
            uint jt0 = uint(jt0_i);
            uint jt1 = jt0 + 1u;
            float ft = clamp(t_pos - float(jt0_i), 0.0f, 1.0f);

            float ln_p = log(max(pk, 0.5f));
            float p_pos = (ln_p - pr_ln0) / pr_dln;
            bool is_upper = (pk <= pr_trop);
            uint tropo = is_upper ? 1u : 0u;
            uint jp0_km;
            float fp;
            if (is_upper) {
                int jp_u = clamp(int(floor(p_pos)), 12, int(n_press) - 2);
                jp0_km = uint(jp_u + 1);
                fp = clamp(p_pos - float(jp_u), 0.0f, 1.0f);
            } else {
                int jp_l = clamp(int(floor(p_pos)), 0, 11);
                jp0_km = uint(jp_l);
                fp = clamp(p_pos - float(jp_l), 0.0f, 1.0f);
            }
            uint jp1_km = jp0_km + 1u;

            uint s1 = is_upper ? s1_u : s1_l;
            uint s2 = is_upper ? s2_u : s2_l;
            bool has_key = (s1 > 0u) || (s2 > 0u);
            float v1 = vmr[s1];
            float v2 = vmr[s2];

            float vr1_0 = vmr_ref[(jt0 * 20u + s1) * 2u + tropo];
            float vr2_0 = vmr_ref[(jt0 * 20u + s2) * 2u + tropo];
            float vmix0 = v1 + (vr1_0 / max(vr2_0, 1.0e-30f)) * v2;
            float eta0 = (has_key && vmix0 > 1.0e-30f) ? (v1 / max(vmix0, 1.0e-30f)) : 0.0f;
            float es0 = clamp(eta0 * eta_max, 0.0f, eta_max - 1.0e-5f);
            uint je0_0 = uint(clamp(int(floor(es0)), 0, int(n_eta) - 2));
            float fe_0 = clamp(es0 - float(je0_0), 0.0f, 1.0f);
            float cmix0 = has_key ? (col_dry_e20 * vmix0) : 0.0f;

            float vr1_1 = vmr_ref[(jt1 * 20u + s1) * 2u + tropo];
            float vr2_1 = vmr_ref[(jt1 * 20u + s2) * 2u + tropo];
            float vmix1 = v1 + (vr1_1 / max(vr2_1, 1.0e-30f)) * v2;
            float eta1 = (has_key && vmix1 > 1.0e-30f) ? (v1 / max(vmix1, 1.0e-30f)) : 0.0f;
            float es1 = clamp(eta1 * eta_max, 0.0f, eta_max - 1.0e-5f);
            uint je0_1 = uint(clamp(int(floor(es1)), 0, int(n_eta) - 2));
            float fe_1 = clamp(es1 - float(je0_1), 0.0f, 1.0f);
            float cmix1 = has_key ? (col_dry_e20 * vmix1) : 0.0f;

            uint idx00 = jt0 * km_stride_t + jp0_km * km_stride_p + je0_0 * km_stride_e + g;
            uint idx01 = jt0 * km_stride_t + jp1_km * km_stride_p + je0_0 * km_stride_e + g;
            uint idx10 = jt1 * km_stride_t + jp0_km * km_stride_p + je0_1 * km_stride_e + g;
            uint idx11 = jt1 * km_stride_t + jp1_km * km_stride_p + je0_1 * km_stride_e + g;

            float k_t0 = (1.0f - fp) * ((1.0f - fe_0) * kmajor[idx00] + fe_0 * kmajor[idx00 + km_stride_e])
                       + fp          * ((1.0f - fe_0) * kmajor[idx01] + fe_0 * kmajor[idx01 + km_stride_e]);
            float k_t1 = (1.0f - fp) * ((1.0f - fe_1) * kmajor[idx10] + fe_1 * kmajor[idx10 + km_stride_e])
                       + fp          * ((1.0f - fe_1) * kmajor[idx11] + fe_1 * kmajor[idx11 + km_stride_e]);
            float tau_gas = max((1.0f - ft) * (k_t0 * cmix0) + ft * (k_t1 * cmix1), 0.0f);

            uint m0 = is_upper ? moff_u0 : moff_l0;
            uint m1 = is_upper ? moff_u1 : moff_l1;
            if (m0 < m1) {
                float dens_fac = (pk / max(tk, 100.0f)) * (ref_t / ref_p);
                float inv_1ph2o = 1.0f / (1.0f + vmr[1]);
                for (uint me = m0; me < m1; ++me) {
                    uint e4 = me * 4u;
                    uint g_idx = uint(ment[e4 + 0u]);
                    uint s_idx = uint(ment[e4 + 1u]);
                    uint mflg  = uint(ment[e4 + 2u]);
                    uint k_col = uint(ment[e4 + 3u]);

                    float g_col = col_dry_e20 * vmr[g_idx];
                    if ((mflg & 2u) != 0u) g_col *= dens_fac;
                    if (s_idx > 0u) {
                        float vs = vmr[s_idx];
                        g_col *= ((mflg & 1u) != 0u) ? ((1.0f - vs) * inv_1ph2o) : (vs * inv_1ph2o);
                    }
                    uint mi0 = jt0 * mn_stride_t + je0_0 * mn_stride_e + k_col;
                    uint mi1 = jt1 * mn_stride_t + je0_1 * mn_stride_e + k_col;
                    float km0 = (1.0f - fe_0) * kminor_all[mi0] + fe_0 * kminor_all[mi0 + mn_stride_e];
                    float km1 = (1.0f - fe_1) * kminor_all[mi1] + fe_1 * kminor_all[mi1 + mn_stride_e];
                    tau_gas += max(((1.0f - ft) * km0 + ft * km1) * g_col, 0.0f);
                }
            }

            uint r_off = is_upper ? ry_half : 0u;
            uint ri0 = r_off + jt0 * ry_stride_t + je0_0 * ry_stride_e + g;
            uint ri1 = r_off + jt1 * ry_stride_t + je0_1 * ry_stride_e + g;
            float r0 = (1.0f - fe_0) * rayl_all[ri0] + fe_0 * rayl_all[ri0 + ry_stride_e];
            float r1 = (1.0f - fe_1) * rayl_all[ri1] + fe_1 * rayl_all[ri1 + ry_stride_e];
            float tau_rayl = max(((1.0f - ft) * r0 + ft * r1) * col_dry_e20, 0.0f);

            float tau_ext = 0.0f;
            float scat_ext = 0.0f;
            float asym_ext = 0.0f;
            if (qck > 0.0f) {
                float rel = has_re_l3d ? clamp(re_liq_c[idx], 3.0f, 40.0f) : re_l_sc;
                float tl = chi_sw * (1000.0f * qck * dm_dry) * (0.0028f + 1.50f / rel);
                tau_ext += tl;
                scat_ext += tl * cld_oml;
                asym_ext += tl * cld_oml * 0.85f;
            }
            if (qik > 0.0f) {
                float rei = has_re_i3d ? clamp(re_ice_c[idx], 10.0f, 150.0f) : re_i_sc;
                float ti = chi_sw * (1000.0f * qik * dm_dry) * (0.0034f + 2.40f / rei);
                tau_ext += ti;
                scat_ext += ti * cld_omi;
                asym_ext += ti * cld_omi * 0.80f;
            }
            if (has_dust) {
                float qdk = max(q_dust_c[idx], 0.0f);
                if (qdk > 0.0f) {
                    float td = dust_ke * (qdk * dm_dry);
                    tau_ext += td;
                    scat_ext += td * dust_ss;
                    asym_ext += td * dust_ss * 0.72f;
                }
            }
            if (has_aero) {
                tau_ext  += aero_sc * max(aero_tau_sw[idx], 0.0f);
                scat_ext += aero_sc * max(aero_sct_sw[idx], 0.0f);
                asym_ext += aero_sc * aero_asy_sw[idx];
            }

            float tau_lay = tau_gas + tau_rayl + tau_ext;
            float scat_lay = tau_rayl + scat_ext;
            float w0 = clamp(scat_lay / max(tau_lay, 1.0e-8f), 0.0f, 0.999999f);
            float g0 = clamp(asym_ext / max(scat_lay, 1.0e-8f), -0.95f, 0.95f);

            float f_del = g0 * g0;
            float denom_del = max(1.0f - w0 * f_del, 1.0e-7f);
            float tau = max(tau_lay * (1.0f - w0 * f_del), 1.0e-7f);
            float w = clamp((w0 * (1.0f - f_del)) / denom_del, 0.0f, 0.999999f);
            float gg = (g0 - f_del) / max(1.0f - f_del, 1.0e-7f);

            float gamma1 = 0.5f * sqrt3 * (2.0f - w * (1.0f + gg));
            float gamma2 = 0.5f * sqrt3 * w * (1.0f - gg);
            float gamma3 = 0.5f * (1.0f - sqrt3 * gg * mu0_ts);
            float gamma4 = 1.0f - gamma3;

            float lam = sqrt(max(gamma1 * gamma1 - gamma2 * gamma2, 1.0e-12f));
            float big_gamma = gamma2 / (gamma1 + lam);

            tau_cum += tau;
            float denom_res = lam * lam - inv_mu0 * inv_mu0;
            if (fabs(denom_res) < 1.0e-5f) {
                denom_res = (denom_res >= 0.0f) ? 1.0e-5f : -1.0e-5f;
            }
            float att_bot = exp(-min(tau_cum * inv_mu0, 60.0f));
            float cp_c = w * f_beam0 * ((gamma1 - inv_mu0) * gamma3 + gamma4 * gamma2) / denom_res;
            float cm_c = w * f_beam0 * ((gamma1 + inv_mu0) * gamma4 + gamma2 * gamma3) / denom_res;
            float cp_0 = cp_c * att_prev;
            float cp_t = cp_c * att_bot;
            float cm_0 = cm_c * att_prev;
            float cm_t = cm_c * att_bot;
            if (n == 0u) cp_c0 = cp_c;

            float exp_lt = exp(-min(lam * tau, 50.0f));
            float e1 = 1.0f + big_gamma * exp_lt;
            float e2 = 1.0f - big_gamma * exp_lt;
            float e3 = big_gamma + exp_lt;
            float e4 = big_gamma - exp_lt;

            bg_a[n]  = big_gamma;
            el_a[n]  = exp_lt;
            att_a[n] = att_bot;
            cp_a[n]  = cp_t;
            cm_a[n]  = cm_t;

            if (n == 0u) {
                float inv_b0 = 1.0f / e1;
                c_star[0] = (-e2) * inv_b0;
                d_star[0] = (-cm_0) * inv_b0;
            } else {
                uint r1 = 2u * n - 1u;
                float A1 = e1_p * e2 - e3_p * e4;
                float B1 = e2_p * e2 - e4_p * e4;
                float D1 = e1 * e4 - e2 * e3;
                float E1 = (cp_0 - cp_t_p) * e2 + (cm_t_p - cm_0) * e4;
                float inv1 = 1.0f / (B1 - A1 * c_star[r1 - 1u]);
                c_star[r1] = D1 * inv1;
                d_star[r1] = (E1 - A1 * d_star[r1 - 1u]) * inv1;

                uint r2 = 2u * n;
                float A2 = e2_p * e3_p - e4_p * e1_p;
                float B2 = e1_p * e1 - e3_p * e3;
                float D2 = e3_p * e4 - e1_p * e2;
                float E2 = e3_p * (cp_0 - cp_t_p) + e1_p * (cm_t_p - cm_0);
                float inv2 = 1.0f / (B2 - A2 * c_star[r2 - 1u]);
                c_star[r2] = D2 * inv2;
                d_star[r2] = (E2 - A2 * d_star[r2 - 1u]) * inv2;
            }

            e1_p = e1; e2_p = e2; e3_p = e3; e4_p = e4;
            cp_t_p = cp_t; cm_t_p = cm_t;
            att_prev = att_bot;
        }

        {
            uint r_last = 2u * N - 1u;
            float direct_sfc = mu0_ts * f_beam0 * att_prev;
            float Al = e1_p - r_sfc * e3_p;
            float Bl = e2_p - r_sfc * e4_p;
            float El = r_dir * direct_sfc - cp_t_p + r_sfc * cm_t_p;
            float inv_l = 1.0f / (Bl - Al * c_star[r_last - 1u]);
            d_star[r_last] = (El - Al * d_star[r_last - 1u]) * inv_l;
        }
    }

    float y_next = 0.0f;
    float s1_dir_g = 0.0f, s1_dif_g = 0.0f, s2_dir_g = 0.0f, s2_dif_g = 0.0f, par_g = 0.0f;
    float toa_up_g = 0.0f, toa_dn_g = 0.0f;

    for (int in = int(N) - 1; in >= 0; --in) {
        uint n = uint(in);
        float f_up_b = 0.0f;
        float f_dn_b = 0.0f;
        if (active) {
            float y2 = (n == N - 1u) ? d_star[2u * N - 1u] : (d_star[2u * n + 1u] - c_star[2u * n + 1u] * y_next);
            float y1 = d_star[2u * n] - c_star[2u * n] * y2;
            y_next = y1;

            float G = bg_a[n];
            float E = el_a[n];
            float att_b = att_a[n];
            float e1 = 1.0f + G * E;
            float e2 = 1.0f - G * E;
            float e3 = G + E;
            float e4 = G - E;

            float f_dir = mu0_ts * f_beam0 * att_b;
            f_up_b = max(y1 * e1 + y2 * e2 + cp_a[n], 0.0f);
            f_dn_b = max(y1 * e3 + y2 * e4 + cm_a[n] + f_dir, 0.0f);

            if (n == N - 1u) {
                float sd = max(f_dir, 0.0f);
                float sf_d = max(f_dn_b - sd, 0.0f);
                if (is_vis) {
                    s1_dir_g = sd;
                    s1_dif_g = sf_d;
                } else {
                    s2_dir_g = sd;
                    s2_dif_g = sf_d;
                }
                if (is_par) {
                    par_g = sd + sf_d;
                }
            }
            if (n == 0u) {
                toa_up_g = max(y1 * e3 - y2 * e4 + cp_c0, 0.0f);
                toa_dn_g = mu0_ts * f_beam0;
            }
        }

        uint k_bot = N - 1u - n;
        float w_up = esm_warp_sum(f_up_b);
        if (lane == 0u) sh_warp[warp] = w_up;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) sw_up_out[k_bot * M + col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float w_dn = esm_warp_sum(f_dn_b);
        if (lane == 0u) sh_warp[warp] = w_dn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) sw_dn_out[k_bot * M + col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        float w = esm_warp_sum(toa_up_g);
        if (lane == 0u) sh_warp[warp] = w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) sw_up_out[N * M + col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        w = esm_warp_sum(toa_dn_g);
        if (lane == 0u) sh_warp[warp] = w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) sw_dn_out[N * M + col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        w = esm_warp_sum(s1_dir_g);
        if (lane == 0u) sh_warp[warp] = w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) sfc_out[0u * M + col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        w = esm_warp_sum(s1_dif_g);
        if (lane == 0u) sh_warp[warp] = w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) sfc_out[1u * M + col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        w = esm_warp_sum(s2_dir_g);
        if (lane == 0u) sh_warp[warp] = w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) sfc_out[2u * M + col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        w = esm_warp_sum(s2_dif_g);
        if (lane == 0u) sh_warp[warp] = w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) sfc_out[3u * M + col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        w = esm_warp_sum(par_g);
        if (lane == 0u) sh_warp[warp] = w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0u) sfc_out[4u * M + col] = sh_warp[0] + sh_warp[1] + sh_warp[2] + sh_warp[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (g < N) {
        uint idx = g * M + col;
        uint up_idx = (g + 1u) * M + col;
        float net_bot = sw_dn_out[idx] - sw_up_out[idx];
        float net_top = sw_dn_out[up_idx] - sw_up_out[up_idx];
        float dm_bu = max(rho_d_c[idx] * dz_c[g * dz_cols + dz_col], 1.0e-5f);
        q_sw_out[idx] = (net_top - net_bot) / (cp_d * dm_bu);
    }
}

// -------------------------------------------------------------------------
// 20c. MSTRN11-Class 29-Band (111 g-point) CKD-HD with Chou LW Scattering
//      & 5-Species Aerosol Optics (1 thread per column)
// -------------------------------------------------------------------------
kernel void rad_ckd_hd_col(
    device const float* pres_c       [[buffer(0)]],
    device const float* pres_w       [[buffer(1)]],
    device const float* temp_c       [[buffer(2)]],
    device const float* temp_w       [[buffer(3)]],
    device const float* dz_c         [[buffer(4)]],
    device const float* rho_d_c      [[buffer(5)]],
    device const float* qv_c         [[buffer(6)]],
    device const float* qc_c         [[buffer(7)]],
    device const float* qi_c         [[buffer(8)]],
    device const float* q_dust_c     [[buffer(9)]],
    device const float* o3_vmr_c     [[buffer(10)]],
    device const float* re_liq_c     [[buffer(11)]],
    device const float* re_ice_c     [[buffer(12)]],
    device const float* aero_tau_sw  [[buffer(13)]],
    device const float* aero_sct_sw  [[buffer(14)]],
    device const float* aero_asy_sw  [[buffer(15)]],
    device const float* aero_tau_lw  [[buffer(16)]],
    device const float* sfc_in       [[buffer(17)]],
    device float* sw_up_out          [[buffer(18)]],
    device float* sw_dn_out          [[buffer(19)]],
    device float* lw_up_out          [[buffer(20)]],
    device float* lw_dn_out          [[buffer(21)]],
    device float* q_rad_3d           [[buffer(22)]],
    device float* sfc_out            [[buffer(23)]],
    device const float* tbl          [[buffer(24)]],
    constant ulong4& u_p             [[buffer(25)]], // [N, M, dz_cols, flags]
    constant float4& f_p0            [[buffer(26)]], // [co2_ppmv, ch4_ppmv, n2o_ppmv, chi_sw]
    constant float4& f_p1            [[buffer(27)]], // [re_l_sc, re_i_sc, 0, 0]
    uint col                         [[thread_position_in_grid]]
) {
    uint N = uint(u_p.x);
    uint M = uint(u_p.y);
    uint dz_cols = uint(u_p.z);
    uint flags = uint(u_p.w);
    if (col >= M || N == 0u || N > RAD_MAX_NZ) return;

    bool has_dust   = (flags & 1u) != 0u;
    bool has_o3     = (flags & 2u) != 0u;
    bool o3_lw_cpl  = (flags & 4u) != 0u;
    bool has_re_l3d = (flags & 8u) != 0u;
    bool has_re_i3d = (flags & 16u) != 0u;
    bool has_aero   = (flags & 32u) != 0u;
    uint dz_col = (dz_cols == 1u) ? 0u : col;

    float STEFAN_B   = tbl[1602];
    float O3_G1      = tbl[1603];
    float O3_G2      = tbl[1604];
    float O3_G3      = tbl[1605];
    float CP_D       = tbl[1606];
    float EPS_WATER  = tbl[1608];
    float P_REF      = tbl[1609];
    float GRAVITY    = tbl[1610];
    float O2_VMR     = tbl[1611];
    float C2_PLANCK  = tbl[1612];
    float LW_DIFF    = tbl[1613];
    float SW1_PAR_F  = tbl[1614];
    float R_CO2      = tbl[1615];
    float R_O3       = tbl[1616];
    float R_CH4      = tbl[1617];
    float R_N2O      = tbl[1618];
    float R_O2       = tbl[1619];

    float co2_mass_f = max(f_p0.x, 0.0f) * 1.0e-6f * R_CO2;
    float ch4_mass_f = max(f_p0.y, 0.0f) * 1.0e-6f * R_CH4;
    float n2o_mass_f = max(f_p0.z, 0.0f) * 1.0e-6f * R_N2O;
    float o2_mass_f  = O2_VMR * R_O2;
    float chi_sw     = f_p0.w;
    float re_l_sc    = clamp(f_p1.x, 3.0f, 40.0f);
    float re_i_sc    = clamp(f_p1.y, 10.0f, 150.0f);

    float ts_c       = max(sfc_in[0u * M + col], 100.0f);
    float alb_vis_d  = clamp(sfc_in[1u * M + col], 0.0f, 1.0f);
    float alb_nir_d  = clamp(sfc_in[2u * M + col], 0.0f, 1.0f);
    float alb_vis_f  = clamp(sfc_in[3u * M + col], 0.0f, 1.0f);
    float alb_nir_f  = clamp(sfc_in[4u * M + col], 0.0f, 1.0f);
    float mu0_in     = sfc_in[5u * M + col];
    float s_toa_col  = sfc_in[6u * M + col];
    float emiss_sfc  = clamp(sfc_in[7u * M + col], 0.0f, 1.0f);

    float c_star[2 * RAD_MAX_NZ];
    float d_star[2 * RAD_MAX_NZ];

    for (uint k = 0u; k <= N; ++k) {
        sw_up_out[k * M + col] = 0.0f;
        sw_dn_out[k * M + col] = 0.0f;
    }
    float s_vis_dir = 0.0f, s_vis_dif = 0.0f;
    float s_nir_dir = 0.0f, s_nir_dif = 0.0f;
    float s_par_wm2 = 0.0f;

    // A. Shortwave 39-g-point Delta-Quadrature Two-Stream with 5-species aerosols
    if ((mu0_in > 1.0e-4f) && (s_toa_col > 1.0e-4f)) {
        float mu0_safe = max(mu0_in, 1.0e-3f);
        float mu0_ts = max(mu0_in, 1.0e-4f);
        float inv_mu0 = 1.0f / mu0_ts;
        float sqrt3 = 1.73205081f;

        uint k_top = N - 1u;
        float p_top_w = pres_w[N * M + col];
        float o3_top;
        if (has_o3) {
            o3_top = max(o3_vmr_c[k_top * M + col], 0.0f);
        } else {
            float ph_top = max(pres_c[k_top * M + col] * 0.01f, 1.0e-3f);
            o3_top = O3_G1 * pow(ph_top, O3_G2) * exp(-ph_top / O3_G3) * 1.0e-6f;
        }
        float u_o3_above = (o3_top * R_O3 * (p_top_w / GRAVITY)) / (O3_G2 + 1.0f);

        uint nq1 = N * M;
        uint nq2 = 2u * N * M;

        for (uint g_idx = 0u; g_idx < 39u; ++g_idx) {
            uint goff = 1212u + 10u * g_idx;
            uint b = uint(tbl[goff] + 0.5f);
            float w_g = tbl[goff + 1u];
            float ko  = tbl[goff + 2u];
            float kh  = tbl[goff + 3u];
            float gh  = tbl[goff + 4u];
            float kc  = tbl[goff + 5u];
            float gc  = tbl[goff + 6u];
            float km  = tbl[goff + 7u];
            float gm  = tbl[goff + 8u];
            float ko2 = tbl[goff + 9u];

            float s_frac = tbl[1098u + b];
            float ray_t0 = tbl[1104u + b];
            bool is_vis  = (tbl[1110u + b] > 0.5f);
            float la = tbl[1116u + b], lb = tbl[1122u + b], lc = tbl[1128u + b];
            float ld = tbl[1134u + b], le = tbl[1140u + b], lf = tbl[1146u + b];
            float ia = tbl[1152u + b], ib = tbl[1158u + b], ic = tbl[1164u + b];
            float id = tbl[1170u + b], ie = tbl[1176u + b], i_f = tbl[1182u + b];
            float dk = tbl[1188u + b], ds = tbl[1194u + b], da = tbl[1200u + b];
            float asc = tbl[1206u + b];

            float f_beam0 = (w_g * s_frac) * s_toa_col;
            if (ko > 0.0f) {
                f_beam0 *= exp(-min(ko * u_o3_above / mu0_safe, 40.0f));
            }
            float r_dir = is_vis ? alb_vis_d : alb_nir_d;
            float r_sfc = is_vis ? alb_vis_f : alb_nir_f;

            float tau_cum = 0.0f;
            float att_prev = 1.0f;
            float e1_p = 0.0f, e2_p = 0.0f, e3_p = 0.0f, e4_p = 0.0f;
            float cp_t_p = 0.0f, cm_t_p = 0.0f;

            for (uint n = 0u; n < N; ++n) {
                uint k = N - 1u - n;
                uint idx = k * M + col;
                float pk = pres_c[idx];
                float dzk = dz_c[k * dz_cols + dz_col];
                float dm_dry = rho_d_c[idx] * dzk;
                float dp_frac = max(pres_w[k * M + col] - pres_w[(k + 1u) * M + col], 1.0f) / P_REF;

                float tau_lay = ray_t0 * dp_frac;
                float scat_lay = tau_lay;
                float asym_scat_lay = 0.0f;

                float qck = max(qc_c[idx], 0.0f);
                float qik = max(qi_c[idx], 0.0f);
                if (qck > 0.0f) {
                    float rel = has_re_l3d ? clamp(re_liq_c[idx], 3.0f, 40.0f) : re_l_sc;
                    float lwp = 1000.0f * qck * dm_dry;
                    float tl = chi_sw * lwp * (la + lb / rel);
                    float ol = clamp(1.0f - (lc + ld * rel), 0.0f, 0.999999f);
                    float gl = clamp(le + lf * rel, -0.99f, 0.99f);
                    tau_lay += tl;
                    scat_lay += tl * ol;
                    asym_scat_lay += tl * ol * gl;
                }
                if (qik > 0.0f) {
                    float rei = has_re_i3d ? clamp(re_ice_c[idx], 10.0f, 150.0f) : re_i_sc;
                    float iwp = 1000.0f * qik * dm_dry;
                    float ti = chi_sw * iwp * (ia + ib / rei);
                    float oi = clamp(1.0f - (ic + id * rei), 0.0f, 0.999999f);
                    float gi = clamp(ie + i_f * rei, -0.99f, 0.99f);
                    tau_lay += ti;
                    scat_lay += ti * oi;
                    asym_scat_lay += ti * oi * gi;
                }
                if (has_dust) {
                    float qdk = max(q_dust_c[idx], 0.0f);
                    if (qdk > 0.0f) {
                        float td = dk * (qdk * dm_dry);
                        tau_lay += td;
                        scat_lay += td * ds;
                        asym_scat_lay += td * ds * da;
                    }
                }
                if (has_aero) {
                    tau_lay += asc * max(aero_tau_sw[idx], 0.0f);
                    scat_lay += asc * max(aero_sct_sw[idx], 0.0f);
                    asym_scat_lay += asc * aero_asy_sw[idx];
                }
                if (ko > 0.0f) {
                    float o3v;
                    if (has_o3) {
                        o3v = max(o3_vmr_c[idx], 0.0f);
                    } else {
                        float ph = max(pk * 0.01f, 1.0e-3f);
                        o3v = O3_G1 * pow(ph, O3_G2) * exp(-ph / O3_G3) * 1.0e-6f;
                    }
                    tau_lay += ko * (o3v * R_O3 * dm_dry);
                }
                if (kh > 0.0f) {
                    tau_lay += kh * pow(max((pk + 80.0f) / P_REF, 1.0e-4f), gh) * (max(qv_c[idx], 0.0f) * dm_dry);
                }
                if (kc > 0.0f) {
                    tau_lay += kc * pow(max((pk + 150.0f) / P_REF, 1.0e-4f), gc) * (co2_mass_f * dm_dry);
                }
                if (km > 0.0f) {
                    float sf = pow(min(pk / 8000.0f, 1.0f), 0.38f);
                    tau_lay += km * pow(max((pk + 100.0f) / P_REF, 1.0e-4f), gm) * (ch4_mass_f * sf * dm_dry);
                }
                if (ko2 > 0.0f) {
                    tau_lay += ko2 * pow(max(pk / P_REF, 1.0e-4f), 0.85f) * (o2_mass_f * dm_dry);
                }

                float w0 = clamp(scat_lay / max(tau_lay, 1.0e-8f), 0.0f, 0.999999f);
                float g0 = clamp(asym_scat_lay / max(scat_lay, 1.0e-8f), -0.99f, 0.99f);
                float f_del = g0 * g0;
                float denom_del = max(1.0f - w0 * f_del, 1.0e-7f);
                float tau = max(tau_lay * (1.0f - w0 * f_del), 1.0e-7f);
                float w = clamp((w0 * (1.0f - f_del)) / denom_del, 0.0f, 0.999999f);
                float gg = (g0 - f_del) / max(1.0f - f_del, 1.0e-7f);

                float gamma1 = 0.5f * sqrt3 * (2.0f - w * (1.0f + gg));
                float gamma2 = 0.5f * sqrt3 * w * (1.0f - gg);
                float gamma3 = 0.5f * (1.0f - sqrt3 * gg * mu0_ts);
                float gamma4 = 1.0f - gamma3;

                float lam = sqrt(max(gamma1 * gamma1 - gamma2 * gamma2, 1.0e-12f));
                float big_gamma = gamma2 / (gamma1 + lam);

                tau_cum += tau;
                float denom_res = lam * lam - inv_mu0 * inv_mu0;
                if (fabs(denom_res) < 1.0e-5f) {
                    denom_res = (denom_res >= 0.0f) ? 1.0e-5f : -1.0e-5f;
                }
                float att_bot = exp(-min(tau_cum * inv_mu0, 60.0f));
                float cp_c = w * f_beam0 * ((gamma1 - inv_mu0) * gamma3 + gamma4 * gamma2) / denom_res;
                float cm_c = w * f_beam0 * ((gamma1 + inv_mu0) * gamma4 + gamma2 * gamma3) / denom_res;
                float cp_0 = cp_c * att_prev;
                float cp_t = cp_c * att_bot;
                float cm_0 = cm_c * att_prev;
                float cm_t = cm_c * att_bot;

                float exp_lt = exp(-min(lam * tau, 50.0f));
                float e1 = 1.0f + big_gamma * exp_lt;
                float e2 = 1.0f - big_gamma * exp_lt;
                float e3 = big_gamma + exp_lt;
                float e4 = big_gamma - exp_lt;

                uint nidx = n * M + col;
                lw_up_out[nidx] = big_gamma;
                lw_dn_out[nidx] = exp_lt;
                q_rad_3d[nidx] = att_bot;
                q_rad_3d[nq1 + nidx] = cp_c;
                q_rad_3d[nq2 + nidx] = cm_c;

                if (n == 0u) {
                    float inv_b0 = 1.0f / e1;
                    c_star[0] = (-e2) * inv_b0;
                    d_star[0] = (-cm_0) * inv_b0;
                } else {
                    uint r1 = 2u * n - 1u;
                    float A1 = e1_p * e2 - e3_p * e4;
                    float B1 = e2_p * e2 - e4_p * e4;
                    float D1 = e1 * e4 - e2 * e3;
                    float E1 = (cp_0 - cp_t_p) * e2 + (cm_t_p - cm_0) * e4;
                    float inv1 = 1.0f / (B1 - A1 * c_star[r1 - 1u]);
                    c_star[r1] = D1 * inv1;
                    d_star[r1] = (E1 - A1 * d_star[r1 - 1u]) * inv1;

                    uint r2 = 2u * n;
                    float A2 = e2_p * e3_p - e4_p * e1_p;
                    float B2 = e1_p * e1 - e3_p * e3;
                    float D2 = e3_p * e4 - e1_p * e2;
                    float E2 = e3_p * (cp_0 - cp_t_p) + e1_p * (cm_t_p - cm_0);
                    float inv2 = 1.0f / (B2 - A2 * c_star[r2 - 1u]);
                    c_star[r2] = D2 * inv2;
                    d_star[r2] = (E2 - A2 * d_star[r2 - 1u]) * inv2;
                }

                e1_p = e1; e2_p = e2; e3_p = e3; e4_p = e4;
                cp_t_p = cp_t; cm_t_p = cm_t;
                att_prev = att_bot;
            }

            {
                uint r_last = 2u * N - 1u;
                float direct_sfc = mu0_ts * f_beam0 * att_prev;
                float Al = e1_p - r_sfc * e3_p;
                float Bl = e2_p - r_sfc * e4_p;
                float El = r_dir * direct_sfc - cp_t_p + r_sfc * cm_t_p;
                float inv_l = 1.0f / (Bl - Al * c_star[r_last - 1u]);
                d_star[r_last] = (El - Al * d_star[r_last - 1u]) * inv_l;
            }

            float y_next = 0.0f;
            for (int in = int(N) - 1; in >= 0; --in) {
                uint n = uint(in);
                float y2 = (n == N - 1u) ? d_star[2u * N - 1u] : (d_star[2u * n + 1u] - c_star[2u * n + 1u] * y_next);
                float y1 = d_star[2u * n] - c_star[2u * n] * y2;
                y_next = y1;

                uint nidx = n * M + col;
                float G = lw_up_out[nidx];
                float E = lw_dn_out[nidx];
                float att_b = q_rad_3d[nidx];
                float cp_c = q_rad_3d[nq1 + nidx];
                float cm_c = q_rad_3d[nq2 + nidx];

                float e1 = 1.0f + G * E;
                float e2 = 1.0f - G * E;
                float e3 = G + E;
                float e4 = G - E;
                float cp_t = cp_c * att_b;
                float cm_t = cm_c * att_b;

                float f_dir = mu0_ts * f_beam0 * att_b;
                float f_up_b = max(y1 * e1 + y2 * e2 + cp_t, 0.0f);
                float f_dn_b = max(y1 * e3 + y2 * e4 + cm_t + f_dir, 0.0f);

                uint k_bot = N - 1u - n;
                uint out_b = k_bot * M + col;
                sw_up_out[out_b] += f_up_b;
                sw_dn_out[out_b] += f_dn_b;
                if (n == N - 1u) {
                    float sd = max(f_dir, 0.0f);
                    float sf_d = max(f_dn_b - sd, 0.0f);
                    if (is_vis) {
                        s_vis_dir += sd;
                        s_vis_dif += sf_d;
                        if (b == 1u) s_par_wm2 += SW1_PAR_F * (sd + sf_d);
                    } else {
                        s_nir_dir += sd;
                        s_nir_dif += sf_d;
                    }
                }
                if (n == 0u) {
                    uint out_t = N * M + col;
                    sw_up_out[out_t] += max(y1 * e3 - y2 * e4 + cp_c, 0.0f);
                    sw_dn_out[out_t] += mu0_ts * f_beam0;
                }
            }
        }
    }
    sfc_out[0u * M + col] = s_vis_dir;
    sfc_out[1u * M + col] = s_vis_dif;
    sfc_out[2u * M + col] = s_nir_dir;
    sfc_out[3u * M + col] = s_nir_dif;
    sfc_out[4u * M + col] = s_par_wm2;

    // B. Longwave 72-g-point with Chou (1999) LW Cloud Scattering & Aerosols
    for (uint k = 0u; k <= N; ++k) {
        lw_up_out[k * M + col] = 0.0f;
        lw_dn_out[k * M + col] = 0.0f;
    }

    float ts2 = ts_c * ts_c;
    float eb_sfc = STEFAN_B * (ts2 * ts2);
    float r_lw = 1.0f - emiss_sfc;

    for (uint b = 0u; b < 10u; ++b) {
        float nu1 = tbl[2u * b];
        float nu2 = tbl[2u * b + 1u];
        float c_self = tbl[20u + b];
        float c_for  = tbl[30u + b];
        float cla = tbl[40u + b], clb = tbl[50u + b];
        float cia = tbl[60u + b], cib = tbl[70u + b];
        float kd_lw = tbl[80u + b];

        float bsfc_b = rad_ckd_band_frac(nu1, nu2, ts_c, C2_PLANCK) * eb_sfc;
        for (uint n = 0u; n <= N; ++n) {
            uint kw = N - n;
            float tw = max(temp_w[kw * M + col], 100.0f);
            float tw2 = tw * tw;
            float ebw = STEFAN_B * (tw2 * tw2);
            q_rad_3d[n * M + col] = rad_ckd_band_frac(nu1, nu2, tw, C2_PLANCK) * ebw;
        }
        for (uint n = 0u; n < N; ++n) {
            uint k = N - 1u - n;
            uint idx = k * M + col;
            float pk = pres_c[idx];
            float tk = temp_c[idx];
            float dzk = dz_c[k * dz_cols + dz_col];
            float dm_dry = rho_d_c[idx] * dzk;
            float qvk = max(qv_c[idx], 0.0f);
            float qck = max(qc_c[idx], 0.0f);
            float qik = max(qi_c[idx], 0.0f);

            float p_atm = pk / 101325.0f;
            float e_atm = (qvk * p_atm) / (EPS_WATER + qvk);
            float t_self = exp(clamp(1800.0f * (1.0f / max(tk, 180.0f) - 1.0f / 296.0f), -3.0f, 6.0f));
            float tb = (c_self * e_atm * t_self + c_for * p_atm) * (qvk * dm_dry);
            if (qck > 0.0f) {
                float rel = has_re_l3d ? clamp(re_liq_c[idx], 3.0f, 40.0f) : re_l_sc;
                tb += (cla + clb / rel) * (1000.0f * qck * dm_dry);
            }
            if (qik > 0.0f) {
                float rei = has_re_i3d ? clamp(re_ice_c[idx], 10.0f, 150.0f) : re_i_sc;
                tb += (cia + cib / rei) * (1000.0f * qik * dm_dry);
            }
            if (has_dust) {
                tb += kd_lw * (max(q_dust_c[idx], 0.0f) * dm_dry);
            }
            if (has_aero) {
                tb += max(aero_tau_lw[idx], 0.0f);
            }
            d_star[RAD_MAX_NZ + n] = tb;
        }

        uint g0_b = uint(tbl[1620u + b]);
        uint g1_b = uint(tbl[1620u + b + 1u]);
        for (uint g_idx = g0_b; g_idx < g1_b; ++g_idx) {
            uint goff = 90u + 14u * g_idx;
            float w_g = tbl[goff + 1u];
            float kh  = tbl[goff + 2u];
            float gh  = tbl[goff + 3u];
            float kc  = tbl[goff + 4u];
            float gc  = tbl[goff + 5u];
            float pvc = tbl[goff + 6u];
            float btc = tbl[goff + 7u];
            float ko  = tbl[goff + 8u];
            float go  = tbl[goff + 9u];
            float km  = tbl[goff + 10u];
            float gm  = tbl[goff + 11u];
            float kn  = tbl[goff + 12u];
            float gn  = tbl[goff + 13u];

            float cur_dn = 0.0f;
            float fb0 = q_rad_3d[col];
            for (uint n = 0u; n < N; ++n) {
                uint k = N - 1u - n;
                uint idx = k * M + col;
                float pk = pres_c[idx];
                float tk = temp_c[idx];
                float dzk = dz_c[k * dz_cols + dz_col];
                float dm_dry = rho_d_c[idx] * dzk;

                float tau = d_star[RAD_MAX_NZ + n];
                if (kh > 0.0f) {
                    tau += kh * pow(max((pk + 80.0f) / P_REF, 1.0e-4f), gh) * (max(qv_c[idx], 0.0f) * dm_dry);
                }
                if (kc > 0.0f) {
                    float pf = pow(max((pk + pvc) / P_REF, 1.0e-4f), gc);
                    float tf = (btc != 0.0f)
                        ? exp(clamp(btc * ((1.0f / 260.0f) - (1.0f / max(tk, 160.0f))), -4.0f, 4.0f))
                        : 1.0f;
                    tau += (kc * pf * tf) * (co2_mass_f * dm_dry);
                }
                if (ko > 0.0f && o3_lw_cpl) {
                    float o3v;
                    if (has_o3) {
                        o3v = max(o3_vmr_c[idx], 0.0f);
                    } else {
                        float ph = max(pk * 0.01f, 1.0e-3f);
                        o3v = O3_G1 * pow(ph, O3_G2) * exp(-ph / O3_G3) * 1.0e-6f;
                    }
                    tau += ko * pow(max((pk + 50.0f) / P_REF, 1.0e-4f), go) * (o3v * R_O3 * dm_dry);
                }
                if (km > 0.0f || kn > 0.0f) {
                    float sf = pow(min(pk / 8000.0f, 1.0f), 0.38f);
                    float pn = max((pk + 80.0f) / P_REF, 1.0e-4f);
                    if (km > 0.0f) tau += km * pow(pn, gm) * (ch4_mass_f * sf * dm_dry);
                    if (kn > 0.0f) tau += kn * pow(pn, gn) * (n2o_mass_f * sf * dm_dry);
                }

                float dtau = clamp(LW_DIFF * tau, 0.0f, 50.0f);
                float trans = exp(-dtau);
                float em_l = -esm_expm1(-dtau);
                float w_lin = (dtau > 1.0e-3f)
                    ? ((1.0f / max(dtau, 1.0e-3f)) - trans / max(em_l, 1.0e-6f))
                    : (0.5f - dtau * (1.0f / 12.0f));
                float fb1 = q_rad_3d[(n + 1u) * M + col];
                float s_dn = em_l * ((1.0f - w_lin) * fb1 + w_lin * fb0);
                float s_up = em_l * ((1.0f - w_lin) * fb0 + w_lin * fb1);

                cur_dn = cur_dn * trans + s_dn;
                c_star[n] = trans;
                d_star[n] = s_up;
                lw_dn_out[k * M + col] += w_g * cur_dn;
                fb0 = fb1;
            }

            float cur_up = emiss_sfc * bsfc_b + r_lw * cur_dn;
            lw_up_out[col] += w_g * cur_up;
            for (int in = int(N) - 1; in >= 0; --in) {
                uint n = uint(in);
                cur_up = cur_up * c_star[n] + d_star[n];
                lw_up_out[(N - n) * M + col] += w_g * cur_up;
            }
        }
    }

    // C. Bottom-up Radiative Heating Rates [3, nz, M]
    uint plane = N * M;
    uint plane2 = 2u * plane;
    float fnet_sw_prev = sw_dn_out[col] - sw_up_out[col];
    float fnet_lw_prev = lw_dn_out[col] - lw_up_out[col];
    for (uint k = 0u; k < N; ++k) {
        uint idx = k * M + col;
        uint up_idx = (k + 1u) * M + col;
        float fnet_sw_cur = sw_dn_out[up_idx] - sw_up_out[up_idx];
        float fnet_lw_cur = lw_dn_out[up_idx] - lw_up_out[up_idx];
        float dm_bu = max(rho_d_c[idx] * dz_c[k * dz_cols + dz_col], 1.0e-5f);
        float qsw = (fnet_sw_cur - fnet_sw_prev) / (CP_D * dm_bu);
        float qlw = (fnet_lw_cur - fnet_lw_prev) / (CP_D * dm_bu);
        q_rad_3d[idx] = qsw;
        q_rad_3d[plane + idx] = qlw;
        q_rad_3d[plane2 + idx] = qsw + qlw;
        fnet_sw_prev = fnet_sw_cur;
        fnet_lw_prev = fnet_lw_cur;
    }
}

// -------------------------------------------------------------------------
// 20d. Single-Pass Top-Down Fused 6-Class 2-Moment Cloud Microphysics
//      (Morrison 2005 / KK00 / Coupled 3-Species Implicit Sedimentation)
// -------------------------------------------------------------------------
kernel void microphysics_6class_col(
    device float* theta_m        [[buffer(0)]],
    device float* q_v            [[buffer(1)]],
    device float* q_c            [[buffer(2)]],
    device float* q_i            [[buffer(3)]],
    device float* q_r            [[buffer(4)]],
    device float* q_s            [[buffer(5)]],
    device float* q_g            [[buffer(6)]],
    device float* n_c            [[buffer(7)]],
    device float* n_i            [[buffer(8)]],
    device float* n_r            [[buffer(9)]],
    device float* n_s            [[buffer(10)]],
    device float* n_g            [[buffer(11)]],
    device const float* rho_d    [[buffer(12)]],
    device const float* dz_c     [[buffer(13)]],
    device const float* is_ocn_s [[buffer(14)]],
    device const float* u_sfc_s  [[buffer(15)]],
    device const float* q_dust_s [[buffer(16)]],
    device const float* ccn_a_s  [[buffer(17)]],
    device float* pr_rain_out    [[buffer(18)]],
    device float* pr_snow_out    [[buffer(19)]],
    constant ulong4& u_dim       [[buffer(20)]], // [nz, M, sub_per_cell, flags]
    constant float4& f_p0        [[buffer(21)]], // [dt, R_D, CV_D, P_REF]
    constant float4& f_p1        [[buffer(22)]], // [RV_OVER_RD, T_FREEZE, L_F0, L_V0]
    constant float4& f_p2        [[buffer(23)]], // [CP_D, EPS_WATER, CCN_OCN, CCN_LND]
    constant float4& f_p3        [[buffer(24)]], // [CCN_DUST_SC, INP_DUST_ENH, KK00_AUTO, 0]
    uint col                     [[thread_position_in_grid]]
) {
    uint nz = uint(u_dim.x);
    uint M  = uint(u_dim.y);
    uint sub_per_cell = uint(u_dim.z);
    uint flags = uint(u_dim.w);
    if (col >= M || nz == 0u) return;

    bool has_ocn  = (flags & 1u) != 0u;
    bool has_usfc = (flags & 2u) != 0u;
    bool has_dust = (flags & 4u) != 0u;
    bool has_ccn  = (flags & 8u) != 0u;
    bool dz_is_2d = (flags & 16u) != 0u;

    uint dz_stride = dz_is_2d ? (M / max(sub_per_cell, 1u)) : 1u;
    uint dz_col    = dz_is_2d ? (col / max(sub_per_cell, 1u)) : 0u;

    float dt = f_p0.x;
    float R_D = f_p0.y;
    float R_OVER_CV = f_p0.y / f_p0.z;
    float P_REF = f_p0.w;
    float RV_OVER_RD = f_p1.x;
    float T_FREEZE = f_p1.y;
    float L_F0 = f_p1.z;
    float L_V0 = f_p1.w;
    float CP_D = f_p2.x;
    float EPS_W = f_p2.y;
    float CCN_OCN = f_p2.z;
    float CCN_LND = f_p2.w;
    float CCN_DUST = f_p3.x;
    float INP_DUST = f_p3.y;
    float KK00_AUTO = (f_p3.z > 0.0f) ? f_p3.z : 13500.0f;

    float is_ocn = has_ocn ? is_ocn_s[col] : 0.7f;
    float ccn_col_base = is_ocn * CCN_OCN + (1.0f - is_ocn) * CCN_LND;
    if (has_usfc) {
        ccn_col_base += is_ocn * 18.0f * clamp(u_sfc_s[col] / 10.0f, 0.0f, 2.5f);
    }
    if (has_ccn) {
        ccn_col_base += clamp(ccn_a_s[col] * 1.0e-6f, 0.0f, 250.0f);
    }

    // Running top-down implicit upwind sedimentation fluxes (zero register arrays!)
    float fq_i = 0.0f, fn_i = 0.0f;
    float fq_r = 0.0f, fn_r = 0.0f;
    float fq_s = 0.0f, fn_s = 0.0f;
    float fq_g = 0.0f, fn_g = 0.0f;

    for (int ik = int(nz) - 1; ik >= 0; --ik) {
        uint k = uint(ik);
        uint idx = k * M + col;

        float rd = rho_d[idx];
        float rd_safe = max(rd, 0.05f);
        float dzk = dz_c[k * dz_stride + dz_col];
        float mk = max(rd * dzk, 1.0e-4f);

        float thm = theta_m[idx];
        float qv_in = q_v[idx];
        float qv = max(qv_in, 1.0e-7f);
        float qc = max(q_c[idx], 0.0f);
        float qi = max(q_i[idx], 0.0f);
        float qr = max(q_r[idx], 0.0f);
        float qs = max(q_s[idx], 0.0f);
        float qg = max(q_g[idx], 0.0f);

        // Compressible EOS: diagnose_pressure_temperature
        float arg = max(R_D * rd * thm / P_REF, 1.0e-8f);
        float pi_ex = pow(arg, R_OVER_CV);
        float pres = P_REF * arg * pi_ex;
        float temp = pi_ex * (thm / (1.0f + RV_OVER_RD * max(qv_in, 0.0f)));
        float pi_safe = max(pi_ex, 0.1f);

        // 1. Aerosol-aware droplet & ice activation
        float ccn_cm3 = ccn_col_base;
        float dust_fac = 0.0f;
        if (has_dust) {
            dust_fac = clamp(q_dust_s[idx] * 1.0e7f, 0.0f, 5.0f);
            ccn_cm3 += CCN_DUST * dust_fac;
        }
        float ccn_m3 = clamp(ccn_cm3 * 1.0e6f, 1.5e7f, 6.0e8f);
        float nc_cur = 0.0f;
        if (qc > 1.0e-7f) {
            float nc_tgt = ccn_m3 / rd_safe;
            nc_cur = clamp(0.5f * max(n_c[idx], 0.0f) + 0.5f * nc_tgt, 5.0e6f, 1.0e9f);
        }
        float ni_cur = 0.0f;
        if (qi > 1.0e-8f) {
            float t_sub = clamp(T_FREEZE - temp, 0.0f, 40.0f);
            float inp_fac = 1.0f + (has_dust ? (INP_DUST * dust_fac) : 0.0f);
            float ni_tgt = clamp(5.0f * exp(0.304f * t_sub) * inp_fac / rd_safe, 1.0e2f, 5.0e6f);
            ni_cur = clamp(0.5f * max(n_i[idx], 0.0f) + 0.5f * ni_tgt, 1.0e2f, 5.0e6f);
        }
        float nr_cur = max(n_r[idx], 0.0f);
        float ns_cur = max(n_s[idx], 0.0f);
        float ng_cur = max(n_g[idx], 0.0f);

        // 2. Warm-rain KK00 autoconversion & accretion (AOBA Formulation Eq. 5.13)
        if (qc > 1.0e-6f) {
            float nc_cm3 = clamp(nc_cur * rd * 1.0e-6f, 10.0f, 800.0f);
            float k_auto = KK00_AUTO * pow(max(qc, 1.0e-9f), 1.47f) * pow(nc_cm3, -1.79f);
            float k_accr = 67.0f * pow(qr, 1.15f) * pow(max(qc, 1.0e-9f), 0.15f);
            float dq_c2r = (-esm_expm1(-min((k_auto + k_accr) * dt, 10.0f))) * qc;
            qc -= dq_c2r;
            qr += dq_c2r;
            nr_cur += dq_c2r / 2.5e-7f;
        }

        // 3a. Ice crystal aggregation (q_i -> q_s): excess above qi_thresh = 6e-5 kg/kg (Morrison et al., 2005)
        if (qi > 6.0e-5f) {
            float k_agg = 1.5e-3f * exp(clamp(0.025f * (temp - T_FREEZE), -3.0f, 0.0f));
            float dq_i2s = (-esm_expm1(-min(k_agg * dt, 5.0f))) * (qi - 6.0e-5f);
            qi -= dq_i2s;
            qs += dq_i2s;
            ns_cur += dq_i2s / 4.0e-8f;
        }

        // 3b. Riming onto snow/graupel -> graupel
        float dq_c2g = 0.0f;
        if (temp < T_FREEZE) {
            if (qc > 1.0e-5f && (qs + qg) > 1.0e-6f) {
                float k_rm = 0.6f * pow(qs, 0.80f) + 1.2f * pow(qg, 0.80f);
                dq_c2g = (-esm_expm1(-min(k_rm * dt, 8.0f))) * qc;
                qc -= dq_c2g;
            }
            float dq_s2g = 0.0f;
            if (qc + dq_c2g > 2.0e-4f && qs > 1.0e-5f) {
                // Match PyTorch order: qc before subtracting dq_c2g was checked
                float qc_pre = qc + dq_c2g;
                dq_s2g = (-esm_expm1(-min(2.5f * qc_pre * dt * 100.0f, 5.0f))) * qs;
                qs -= dq_s2g;
            }
            qg += dq_c2g + dq_s2g;
            ng_cur += (dq_c2g + dq_s2g) / 2.0e-7f;
        }

        // 3c. Freezing of supercooled rain -> graupel
        float dq_r2g = 0.0f;
        if (temp < 268.15f && qr > 0.0f) {
            float f_frz = 0.015f * exp(min(0.25f * (268.15f - temp), 6.0f));
            dq_r2g = (-esm_expm1(-min(f_frz * dt, 10.0f))) * qr;
            qr -= dq_r2g;
            qg += dq_r2g;
            ng_cur += dq_r2g / 2.5e-7f;
        }
        float dtemp_lat = (L_F0 / CP_D) * (dq_c2g + dq_r2g);

        // 3d. Melting of snow & graupel -> rain
        if (temp > T_FREEZE) {
            float t_abv = temp - T_FREEZE;
            float dq_s2r = (-esm_expm1(-min(0.008f * t_abv * dt, 10.0f))) * qs;
            float dq_g2r = (-esm_expm1(-min(0.004f * t_abv * dt, 10.0f))) * qg;
            qs -= dq_s2r;
            qg -= dq_g2r;
            qr += dq_s2r + dq_g2r;
            nr_cur += (dq_s2r + dq_g2r) / 3.0e-7f;
            dtemp_lat -= (L_F0 / CP_D) * (dq_s2r + dq_g2r);
        }

        // 3e. Sub-cloud rain evaporation (over liquid) & snow/graupel sublimation (over ice!)
        if (qr > 0.0f) {
            float e_sat_w = 611.2f * exp(clamp(17.67f * (temp - T_FREEZE) / max(temp - 29.65f, 20.0f), -30.0f, 15.0f));
            float q_sat_w = clamp(EPS_W * e_sat_w / max(pres - e_sat_w, 100.0f), 1.0e-6f, 0.15f);
            float subsat_w = max(q_sat_w - qv, 0.0f);
            if (subsat_w > 0.0f) {
                float k_evap = 0.004f * pow(max(qr * rd, 0.0f), 0.55f);
                float dq_ev = min((-esm_expm1(-min(k_evap * dt, 5.0f))) * qr, 0.5f * subsat_w);
                qr -= dq_ev;
                qv += dq_ev;
                dtemp_lat -= (L_V0 / CP_D) * dq_ev;
            }
        }
        if (qs > 0.0f || qg > 0.0f) {
            float e_sat_i = 611.2f * exp(clamp(21.875f * (temp - T_FREEZE) / max(temp - 7.66f, 20.0f), -35.0f, 15.0f));
            float q_sat_i = clamp(EPS_W * e_sat_i / max(pres - e_sat_i, 100.0f), 1.0e-6f, 0.15f);
            float sub_rem_i = 0.5f * max(q_sat_i - qv, 0.0f);
            if (sub_rem_i > 0.0f) {
                if (qs > 0.0f) {
                    float k_sub_s = 0.003f * pow(max(qs * rd, 0.0f), 0.55f);
                    float dq_ss = min((-esm_expm1(-min(k_sub_s * dt, 5.0f))) * qs, sub_rem_i);
                    qs -= dq_ss;
                    qv += dq_ss;
                    sub_rem_i = max(sub_rem_i - dq_ss, 0.0f);
                    dtemp_lat -= ((L_V0 + L_F0) / CP_D) * dq_ss;
                }
                if (qg > 0.0f && sub_rem_i > 0.0f) {
                    float k_sub_g = 0.002f * pow(max(qg * rd, 0.0f), 0.55f);
                    float dq_sg = min((-esm_expm1(-min(k_sub_g * dt, 5.0f))) * qg, sub_rem_i);
                    qg -= dq_sg;
                    qv += dq_sg;
                    dtemp_lat -= ((L_V0 + L_F0) / CP_D) * dq_sg;
                }
            }
        }

        // 4. Class-specific terminal velocities & 5. Top-down implicit sedimentation
        float rho_fac = sqrt(clamp(1.2f / rd_safe, 0.5f, 6.0f));
        float vi = clamp(0.22f + 0.45f * pow(max(qi / 1.0e-4f, 0.0f), 0.25f), 0.18f, 0.80f);
        float vr = clamp(6.5f * pow(max(rd * qr * 1.0e3f, 1.0e-6f), 0.14f) * rho_fac, 1.0f, 9.5f);
        float vs = clamp(1.1f * pow(max(rd * qs * 1.0e3f, 1.0e-6f), 0.10f) * rho_fac, 0.3f, 2.2f);
        float vg = clamp(4.2f * pow(max(rd * qg * 1.0e3f, 1.0e-6f), 0.18f) * rho_fac, 0.8f, 8.5f);

        float denom_i = mk + dt * rd * vi;
        float qi_new = (mk * qi + dt * fq_i) / denom_i;
        float ni_new = (mk * ni_cur + dt * fn_i) / denom_i;
        fq_i = rd * vi * qi_new;
        fn_i = rd * vi * ni_new;

        float denom_r = mk + dt * rd * vr;
        float qr_new = (mk * qr + dt * fq_r) / denom_r;
        float nr_new = (mk * nr_cur + dt * fn_r) / denom_r;
        fq_r = rd * vr * qr_new;
        fn_r = rd * vr * nr_new;

        float denom_s = mk + dt * rd * vs;
        float qs_new = (mk * qs + dt * fq_s) / denom_s;
        float ns_new = (mk * ns_cur + dt * fn_s) / denom_s;
        fq_s = rd * vs * qs_new;
        fn_s = rd * vs * ns_new;

        float denom_g = mk + dt * rd * vg;
        float qg_new = (mk * qg + dt * fq_g) / denom_g;
        float ng_new = (mk * ng_cur + dt * fn_g) / denom_g;
        fq_g = rd * vg * qg_new;
        fn_g = rd * vg * ng_new;

        ni_new = (qi_new > 1.0e-10f) ? clamp(ni_new, 1.0e2f, 5.0e6f) : 0.0f;
        nr_new = (qr_new > 1.0e-10f) ? clamp(nr_new, 1.0f, 1.0e6f) : 0.0f;
        ns_new = (qs_new > 1.0e-10f) ? clamp(ns_new, 1.0f, 1.0e6f) : 0.0f;
        ng_new = (qg_new > 1.0e-10f) ? clamp(ng_new, 1.0f, 1.0e5f) : 0.0f;

        float temp_new = clamp(temp + dtemp_lat, 120.0f, 500.0f);
        theta_m[idx] = clamp((temp_new / max(pi_ex, 0.05f)) * (1.0f + RV_OVER_RD * qv), 150.0f, 15000.0f);
        q_v[idx] = qv;
        q_c[idx] = qc;
        q_i[idx] = qi_new;
        q_r[idx] = qr_new;
        q_s[idx] = qs_new;
        q_g[idx] = qg_new;
        n_c[idx] = nc_cur;
        n_i[idx] = ni_new;
        n_r[idx] = nr_new;
        n_s[idx] = ns_new;
        n_g[idx] = ng_new;
    }

    pr_rain_out[col] = fq_r;
    pr_snow_out[col] = fq_s + fq_g + fq_i;
}

// -------------------------------------------------------------------------
// 20e. Fused 3D Prognostic Ozone Photochemistry + Dry Deposition + PBL Mix
// -------------------------------------------------------------------------
kernel void ozone_chem_col(
    device float* o3_vmr           [[buffer(0)]],  // [nz, nc] in-place
    device const float* pres_c     [[buffer(1)]],
    device const float* temp_c     [[buffer(2)]],
    device const float* rho_d      [[buffer(3)]],
    device const float* dz_c       [[buffer(4)]],
    device const float* q_v        [[buffer(5)]],
    device const float* mu0_arr    [[buffer(6)]],
    device const int* surf_type    [[buffer(7)]],
    device const float* lai_arr    [[buffer(8)]],
    device const float* fsnow_arr  [[buffer(9)]],
    device const float* lnox_arr   [[buffer(10)]],
    device const float* fire_arr   [[buffer(11)]],
    device float* diag_out         [[buffer(12)]], // [4, nc]: tco_du, sfc_ppbv, drydep, chem_net
    constant ulong4& u_dim         [[buffer(13)]], // [nz, nc, flags, surf_land]
    constant float4& f_p0          [[buffer(14)]], // [dt, n2o_ppmv, ch4_ppmv, ecl_ppbv]
    constant float4& f_p1          [[buffer(15)]], // [gravity, surf_ocean_f, 0, 0]
    uint col                       [[thread_position_in_grid]]
) {
    uint nz = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    uint flags = uint(u_dim.z);
    int SURF_LAND_ID = int(u_dim.w);
    if (col >= nc || nz == 0u || nz > RAD_MAX_NZ) return;

    bool has_lnox = (flags & 1u) != 0u;
    bool has_fire = (flags & 2u) != 0u;
    int SURF_OCEAN_ID = int(f_p1.y + 0.5f);

    float dt = f_p0.x;
    float n2o_ppmv = f_p0.y;
    float ch4_ppmv = f_p0.z;
    float ecl_ppbv = f_p0.w;
    float GRAVITY  = f_p1.x;

    // Physical constants matching src/aoba/ozone_chemistry.py
    float K_BOLTZ = 1.380649e-23f;
    float M_AIR_M = 28.9647e-3f / 6.02214076e23f;
    float M_O3_AIR = 48.0f / 28.9647f;
    float O2_VMR = 0.20946f;
    float DU_CONV = 2.6867e16f;

    float mu0 = clamp(mu0_arr[col], 0.0f, 1.0f);
    float is_day = (mu0 > 1.0e-4f) ? 1.0f : 0.0f;
    float amf = 35.0f / sqrt(1224.0f * mu0 * mu0 + 1.0f);

    // Surface dry deposition rate in k=0 (Wesely, 1989)
    int st = surf_type[col];
    float is_land = (st == SURF_LAND_ID) ? 1.0f : 0.0f;
    float is_ocn  = (st == SURF_OCEAN_ID) ? 1.0f : 0.0f;
    float f_veg = 1.0f - exp(-0.5f * max(lai_arr[col], 0.0f));
    float f_snw = clamp(fsnow_arr[col], 0.0f, 1.0f);
    float vd_land = f_snw * 0.0008f + (1.0f - f_snw) * (
        (1.0f - f_veg) * 0.0008f + f_veg * (0.0018f + 0.0042f * mu0)
    );
    float vd_sfc = is_land * vd_land + is_ocn * 0.0005f + (1.0f - is_land - is_ocn) * 0.0008f;
    float k_dep_0 = vd_sfc / max(dz_c[col], 10.0f);

    float n2o_fac = max(n2o_ppmv, 0.01f) / 0.273f;
    float ch4_fac = sqrt(max(ch4_ppmv, 0.05f) / 0.808f);
    float ecl_rat = max(ecl_ppbv, 0.0f) / 3.5f;
    float psc_coef = (1.0f / (12.0f * 86400.0f)) * (ecl_rat * ecl_rat) * is_day;

    float lnox_val = has_lnox ? max(lnox_arr[col], 0.0f) : 0.0f;
    float fire_val = has_fire ? max(fire_arr[col], 0.0f) : 0.0f;

    float chi0_arr[RAD_MAX_NZ];
    float chi_arr[RAD_MAX_NZ];
    float dm_arr[RAD_MAX_NZ];
    float dn_air_arr[RAD_MAX_NZ];
    float n_o3_abv[RAD_MAX_NZ];

    // Pass 1 (top-down): overhead O3 column & LNOx/fire profile normalizers
    float w_lnox_sum = 0.0f;
    float w_fire_sum = 0.0f;
    float o3_cum = 0.0f;
    for (int ik = int(nz) - 1; ik >= 0; --ik) {
        uint k = uint(ik);
        uint idx = k * nc + col;
        float c0 = clamp(o3_vmr[idx], 1.0e-10f, 4.0e-5f);
        float pk = max(pres_c[idx], 1.0f);
        float dm = max(rho_d[idx] * dz_c[idx], 1.0e-5f);
        float dna = (dm / M_AIR_M) * 1.0e-4f;
        float dno3 = c0 * dna;
        if (k == nz - 1u) {
            o3_cum = c0 * (pk / (M_AIR_M * GRAVITY)) * 1.0e-4f * 0.5f;
        }
        n_o3_abv[k] = o3_cum + 0.5f * dno3;
        o3_cum += dno3;

        chi0_arr[k] = c0;
        dm_arr[k] = dm;
        dn_air_arr[k] = dna;

        float ph = pk * 0.01f;
        float f_trop = 1.0f - esm_sigmoid((130.0f - ph) / 25.0f);
        if (has_lnox) {
            float d = (ph - 350.0f) / 180.0f;
            w_lnox_sum += exp(-d * d) * f_trop * dm;
        }
        if (has_fire) {
            w_fire_sum += esm_sigmoid((ph - 650.0f) / 80.0f) * f_trop * dm;
        }
    }
    float inv_lnox_sum = 1.0f / max(w_lnox_sum, 1.0f);
    float inv_fire_sum = 1.0f / max(w_fire_sum, 1.0f);

    // Pass 2: exact Riccati-exponential chemistry at every level k
    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * nc + col;
        float c0 = chi0_arr[k];
        float dm = dm_arr[k];
        float pk = max(pres_c[idx], 1.0f);
        float ph = pk * 0.01f;
        float tk = clamp(temp_c[idx], 140.0f, 350.0f);
        float m_cm3 = (pk / (K_BOLTZ * tk)) * 1.0e-6f;

        float n_o2_above = O2_VMR * (pk / (M_AIR_M * GRAVITY)) * 1.0e-4f;
        float s_o2 = n_o2_above * amf;
        float s_o3 = n_o3_abv[k] * amf;

        float j_o2 = is_day * (
            1.55e-9f * exp(-min(5.0e-24f * s_o2 + 7.5e-19f * s_o3, 60.0f))
          + 4.2e-10f * exp(-min(2.5e-22f * s_o2 + 4.5e-19f * s_o3, 60.0f))
        );
        float j_o3 = is_day * (
            8.5e-3f * exp(-min(2.6e-18f * s_o3, 60.0f))
          + 4.5e-4f * exp(-min(1.0e-20f * s_o3, 60.0f))
        );
        float j_o1d_ht = is_day * 3.0e-4f * exp(-min(1.8e-18f * s_o3, 60.0f));
        float j_o1d_uvb = is_day * 4.5e-5f * exp(-min(2.8e-19f * s_o3, 60.0f)) * sqrt(mu0 + 1.0e-6f);
        float j_o1d = j_o1d_ht + j_o1d_uvb;

        float k2 = 6.0e-34f * pow(300.0f / tk, 2.4f);
        float k3 = 8.0e-12f * exp(-2060.0f / tk);
        float o_over_o3 = j_o3 / max(k2 * (O2_VMR * m_cm3) * m_cm3, 1.0e-30f);
        float p_chap = 2.0f * j_o2 * O2_VMR;
        float l2_tot = 2.0f * k3 * o_over_o3 * m_cm3;

        float f_strat = esm_sigmoid((130.0f - ph) / 25.0f);
        float f_trop = 1.0f - f_strat;

        float lp = log(max(ph, 0.3f) / 6.0f) / 1.65f;
        float noy_vmr = 12.0e-9f * n2o_fac * exp(-lp * lp) * f_strat;
        float l1_nox = 2.0f * (5.1e-12f * exp(210.0f / tk)) * (0.45f * noy_vmr * m_cm3) * o_over_o3;

        float h2o_vmr = max(q_v[idx] * (28.9647f / 18.015f), 1.0e-6f);
        float h2o_st = max(h2o_vmr, 4.5e-6f);
        float ho2_vmr = 2.5e-10f * sqrt(max((h2o_st / 5.0e-6f) * (j_o1d_ht / 1.0e-4f), 0.0f)) * f_strat;
        float l1_hox = 2.0f * (3.0e-11f * exp(200.0f / tk)) * (ho2_vmr * m_cm3) * o_over_o3;

        float psc_p = esm_sigmoid((220.0f - ph) / 20.0f) * esm_sigmoid((ph - 8.0f) / 3.0f);
        float psc_t = esm_sigmoid((195.0f - tk) / 2.0f);
        float l1_psc = psc_coef * psc_p * psc_t;

        float p_trop = f_trop * 1.8e-14f * ch4_fac * mu0;
        float mol_air = max(dm / 0.0289647f, 1.0e-3f);
        if (has_lnox && lnox_val > 0.0f) {
            float d = (ph - 350.0f) / 180.0f;
            float wn = (exp(-d * d) * f_trop * dm) * inv_lnox_sum;
            p_trop += (10.0f * ((lnox_val / 0.014007f) * wn) / mol_air) * (0.25f + 0.75f * is_day);
        }
        if (has_fire && fire_val > 0.0f) {
            float wn = (esm_sigmoid((ph - 650.0f) / 80.0f) * f_trop * dm) * inv_fire_sum;
            p_trop += (0.04f * ((fire_val / 0.012011f) * wn) / mol_air) * is_day;
        }

        float f_o1d = (7.0f * h2o_vmr) / (1.0f + 7.0f * h2o_vmr);
        float l1_trop = f_trop * (j_o1d * f_o1d + 3.5e-7f * (0.3f + 0.7f * mu0));

        float p_tot = p_chap + p_trop;
        float l1_tot = l1_nox + l1_hox + l1_psc + l1_trop + ((k == 0u) ? k_dep_0 : 0.0f);

        float disc = sqrt(max(l1_tot * l1_tot + 4.0f * p_tot * l2_tot, 0.0f));
        float denom = l1_tot + disc;
        float chi_eq = (denom > 1.0e-30f) ? ((2.0f * p_tot) / denom) : c0;
        float l_eff = l1_tot + l2_tot * (c0 + chi_eq);
        float relax = -esm_expm1(-clamp(l_eff * dt, 0.0f, 50.0f));
        chi_arr[k] = clamp(c0 + (chi_eq - c0) * relax, 1.0e-10f, 4.0e-5f);
    }

    // Pass 3: 6-layer mass-conserving PBL vertical turbulent mixing
    uint n_mix = (nz > 1u) ? min(6u, nz - 1u) : 0u;
    if (n_mix > 0u) {
        float z_rel[8];
        float dz_loc[8];
        for (uint k = 0u; k <= n_mix; ++k) {
            dz_loc[k] = dz_c[k * nc + col];
        }
        z_rel[0] = 0.5f * dz_loc[0];
        for (uint k = 1u; k <= n_mix; ++k) {
            z_rel[k] = z_rel[k - 1u] + 0.5f * (dz_loc[k - 1u] + dz_loc[k]);
        }
        float h_pbl = 1500.0f;
        for (uint k = 0u; k < n_mix; ++k) {
            float z_int = 0.5f * (z_rel[k] + z_rel[k + 1u]);
            float kz = (z_int < h_pbl)
                ? (12.0f * (z_int / h_pbl) * ((1.0f - z_int / h_pbl) * (1.0f - z_int / h_pbl)) + 2.0f)
                : 0.5f;
            float dz_int = max(0.5f * (dz_loc[k] + dz_loc[k + 1u]), 10.0f);
            float rho_int = 0.5f * (rho_d[k * nc + col] + rho_d[(k + 1u) * nc + col]);
            float exch = (dt * rho_int * kz) / dz_int;
            float frac = exch / (1.0f + exch * (1.0f / dm_arr[k] + 1.0f / dm_arr[k + 1u]));
            float dflux = frac * (chi_arr[k] - chi_arr[k + 1u]);
            chi_arr[k]     -= dflux / dm_arr[k];
            chi_arr[k + 1u] += dflux / dm_arr[k + 1u];
        }
    }

    // Pass 4: clamp, write out o3_vmr & accumulate diagnostics
    float tco_sum = 0.0f;
    float chem_sum = 0.0f;
    for (uint k = 0u; k < nz; ++k) {
        float cn = clamp(chi_arr[k], 1.0e-10f, 4.0e-5f);
        o3_vmr[k * nc + col] = cn;
        tco_sum += cn * dn_air_arr[k];
        chem_sum += (cn - chi0_arr[k]) * M_O3_AIR * dm_arr[k];
        chi_arr[k] = cn;
    }
    float chi_mid_0 = 0.5f * (chi0_arr[0] + chi_arr[0]);
    diag_out[0u * nc + col] = tco_sum / DU_CONV;
    diag_out[1u * nc + col] = chi_arr[0] * 1.0e9f;
    diag_out[2u * nc + col] = (chi_mid_0 * M_O3_AIR * dm_arr[0]) * k_dep_0;
    diag_out[3u * nc + col] = chem_sum / max(dt, 1.0f);
}

// -------------------------------------------------------------------------
// 20e. Map-Free Autonomous Dynamic Lake 2-Layer FLake + Ice + Kahan Kernel
// -------------------------------------------------------------------------
inline void esm_qsat_lake(float temp_k, float pres_pa, bool over_ice, float t_freeze, float eps_w, float l_val, float rv, thread float& qs_out, thread float& dqs_out) {
    float tc = clamp(temp_k - t_freeze, -90.0f, 60.0f);
    float a = over_ice ? 21.875f : 17.269f;
    float b = over_ice ? 265.5f : 237.3f;
    float es = 610.94f * exp(a * tc / (tc + b));
    float p_eff = max(pres_pa - (1.0f - eps_w) * es, 100.0f);
    float qs = clamp(eps_w * es / p_eff, 1.0e-8f, 0.5f);
    float tk = max(temp_k, 180.0f);
    qs_out = qs;
    dqs_out = qs * l_val / (rv * tk * tk);
}

kernel void dynamic_lake_step_col(
    device const float* geom_in     [[buffer(0)]], // [3, nc]: f_cap, h_cap, v_cap_m3
    device const float* force_in    [[buffer(1)]], // [17, nc]
    device float* state_io          [[buffer(2)]], // [8, nc]: v_lake, v_comp, t_ml, t_bot, h_ml, h_ice, swe_ice, t_ice
    device float* diag_out          [[buffer(3)]], // [13, nc]: f_lake, h_lake, f_ice, t_rad, alb4, shf, lhf, evap, taux, tauy
    constant ulong4& u_p0           [[buffer(4)]], // [nc, 0, 0, 0]
    constant float4& f_p0           [[buffer(5)]], // [dt, k_ext_m1, T_FREEZE, SIGMA_SB]
    constant float4& f_p1           [[buffer(6)]], // [CP_D, L_V0, L_S0, L_F0]
    constant float4& f_p2           [[buffer(7)]], // [GRAVITY, KAPPA, EPS_WATER, R_V]
    uint col                        [[thread_position_in_grid]]
) {
    const uint nc = uint(u_p0.x);
    if (col >= nc) return;

    const float dt = f_p0.x;
    const float k_ext = f_p0.y;
    const float T_FREEZE = f_p0.z;
    const float SIGMA_SB = f_p0.w;
    const float CP_D = f_p1.x;
    const float L_V0 = f_p1.y;
    const float L_S0 = f_p1.z;
    const float L_F0 = f_p1.w;
    const float GRAVITY = f_p2.x;
    const float KAPPA = f_p2.y;
    const float EPS_W = f_p2.z;
    const float R_V = f_p2.w;

    const float RHO_W = 1000.0f;
    const float CP_W = 4186.0f;
    const float RHO_ICE = 917.0f;
    const float CP_ICE = 2100.0f;
    const float K_ICE = 2.2f;
    const float K_SNOW = 0.25f;
    const float RHO_SNOW = 300.0f;
    const float EMISS = 0.985f;
    const float T_MD = 277.13f;
    const float ALPHA_RHO = 6.8e-6f;
    const float H_MIN_ACT = 0.05f;
    const float H_ML_MIN = 1.0f;
    const float KZ_TH = 2.0e-5f;
    const float GAMMA_W = 5.0e-5f;

    float f_cap = geom_in[0u * nc + col];
    float h_cap = geom_in[1u * nc + col];
    float v_cap = geom_in[2u * nc + col];
    bool is_basin = (v_cap > 0.0f) && (f_cap > 1.0e-4f);

    float u1 = force_in[0u * nc + col];
    float v1 = force_in[1u * nc + col];
    float temp1 = force_in[2u * nc + col];
    float qv1 = force_in[3u * nc + col];
    float pres_sfc = force_in[4u * nc + col];
    float rho1 = force_in[5u * nc + col];
    float z_ref = force_in[6u * nc + col];
    float sw1_dir = force_in[7u * nc + col];
    float sw1_dif = force_in[8u * nc + col];
    float sw2_dir = force_in[9u * nc + col];
    float sw2_dif = force_in[10u * nc + col];
    float lw_dn = force_in[11u * nc + col];
    float pr_rain = max(force_in[12u * nc + col], 0.0f);
    float pr_snow = max(force_in[13u * nc + col], 0.0f);
    float mu0 = force_in[14u * nc + col];
    float area_c = force_in[15u * nc + col];
    float f_cor = force_in[16u * nc + col];

    float v_lake = state_io[0u * nc + col];
    float v_comp = state_io[1u * nc + col];
    float t_ml = state_io[2u * nc + col];
    float t_bot = state_io[3u * nc + col];
    float h_ml = state_io[4u * nc + col];
    float h_ice = state_io[5u * nc + col];
    float swe_ice = state_io[6u * nc + col];
    float t_ice = state_io[7u * nc + col];

    // 1. Hypsometry before step
    float phi = is_basin ? clamp(v_lake / max(v_cap, 1.0f), 0.0f, 1.0f) : 0.0f;
    float phi13 = (phi > 0.0f) ? pow(phi, 1.0f / 3.0f) : 0.0f;
    float f_raw = f_cap * phi13 * phi13;
    float h_raw = h_cap * phi13;
    bool active = is_basin && (h_raw >= H_MIN_ACT) && (f_raw > 1.0e-6f);
    float f_lake = active ? f_raw : 0.0f;
    float h_lake = active ? h_raw : 0.0f;

    // 2. Surface albedos
    float mu_s = clamp(mu0, 0.03f, 1.0f);
    float alb_open_dir = clamp(0.026f / (pow(mu_s, 1.7f) + 0.065f) + 0.15f * (mu_s - 0.1f) * (mu_s - 0.5f) * (mu_s - 1.0f), 0.04f, 0.45f);
    float alb_open_dif = 0.06f;
    float f_sn_i = swe_ice / (swe_ice + 20.0f);
    float alb_ice_v = 0.55f + (0.82f - 0.55f) * f_sn_i;
    float alb_ice_n = 0.30f + (0.52f - 0.30f) * f_sn_i;
    bool is_frozen = active && (h_ice > 1.0e-4f);

    float alb_vd = is_frozen ? alb_ice_v : alb_open_dir;
    float alb_vf = is_frozen ? alb_ice_v : alb_open_dif;
    float alb_nd = is_frozen ? alb_ice_n : alb_open_dir;
    float alb_nf = is_frozen ? alb_ice_n : alb_open_dif;

    if (!active) {
        float t_dry = clamp(temp1, T_FREEZE, 320.0f);
        state_io[0u * nc + col] = is_basin ? max(v_lake, 0.0f) : 0.0f;
        state_io[1u * nc + col] = is_basin ? v_comp : 0.0f;
        state_io[2u * nc + col] = t_dry;
        state_io[3u * nc + col] = t_dry;
        state_io[4u * nc + col] = 0.0f;
        state_io[5u * nc + col] = 0.0f;
        state_io[6u * nc + col] = 0.0f;
        state_io[7u * nc + col] = clamp(temp1, 180.0f, T_FREEZE);

        diag_out[0u * nc + col] = 0.0f;
        diag_out[1u * nc + col] = 0.0f;
        diag_out[2u * nc + col] = 0.0f;
        diag_out[3u * nc + col] = t_dry;
        diag_out[4u * nc + col] = alb_vd;
        diag_out[5u * nc + col] = alb_vf;
        diag_out[6u * nc + col] = alb_nd;
        diag_out[7u * nc + col] = alb_nf;
        diag_out[8u * nc + col] = 0.0f;
        diag_out[9u * nc + col] = 0.0f;
        diag_out[10u * nc + col] = 0.0f;
        diag_out[11u * nc + col] = 0.0f;
        diag_out[12u * nc + col] = 0.0f;
        return;
    }

    float area_lake = max(f_lake * area_c, 1.0f);
    float sw_abs = max(
        sw1_dir * (1.0f - alb_vd) + sw1_dif * (1.0f - alb_vf) +
        sw2_dir * (1.0f - alb_nd) + sw2_dif * (1.0f - alb_nf),
        0.0f
    );

    // 3. Bulk aerodynamic transfer
    float z0 = is_frozen ? 1.0e-3f : 1.0e-4f;
    float zr = max(z_ref, 2.0f);
    float ln_z = log(zr / z0);
    float cd_neut = (KAPPA / ln_z) * (KAPPA / ln_z);
    float u_eff = sqrt(u1 * u1 + v1 * v1 + 0.25f);
    float t_skin0 = is_frozen ? t_ice : t_ml;
    float ri_b = GRAVITY * zr * (temp1 - t_skin0) / (max(temp1, 180.0f) * u_eff * u_eff);
    float stab = (ri_b >= 0.0f) ? (1.0f / (1.0f + 5.0f * min(ri_b, 2.0f))) : min(1.0f - 2.0f * max(ri_b, -2.0f), 2.0f);
    float cd = clamp(cd_neut * stab, 2.0e-4f, 5.0e-3f);
    float ga = max(rho1, 0.1f) * cd * u_eff;
    float tau_x = -ga * u1;
    float tau_y = -ga * v1;

    float h_liq = max(h_lake - (RHO_ICE / RHO_W) * h_ice, 0.0f);
    float h_ml_c = clamp(h_ml, min(H_ML_MIN, h_liq), max(h_liq, 0.05f));
    float h_bot_c = max(h_liq - h_ml_c, 0.0f);
    float c_ml = RHO_W * CP_W * max(h_ml_c, H_MIN_ACT);

    float shf_out = 0.0f, lhf_out = 0.0f, evap_out = 0.0f, dv_m3 = 0.0f;
    float t_ml_n = t_ml, t_bot_n = t_bot, h_ice_n = h_ice, swe_n = swe_ice, t_ice_n = t_ice;

    if (!is_frozen) {
        float qs_w, dqs_w;
        esm_qsat_lake(t_ml, pres_sfc, false, T_FREEZE, EPS_W, L_V0, R_V, qs_w, dqs_w);
        float f_sw_ml = (h_bot_c > 1.0e-3f) ? (-esm_expm1(-k_ext * h_ml_c)) : 1.0f;
        float sw_ml = f_sw_ml * sw_abs;
        float sw_bot = (1.0f - f_sw_ml) * sw_abs;

        float t0 = t_ml;
        float t03 = t0 * t0 * t0;
        float f_net0 = sw_ml + EMISS * lw_dn - EMISS * SIGMA_SB * t03 * t0 - CP_D * ga * (t0 - temp1) - L_V0 * ga * (qs_w - qv1) - L_F0 * pr_snow;
        float df_dt = 4.0f * EMISS * SIGMA_SB * t03 + CP_D * ga + L_V0 * ga * dqs_w;
        float dt_ml = (f_net0 * dt) / (c_ml + dt * df_dt);
        float t_ml_star = t0 + dt_ml;

        shf_out = CP_D * ga * (t_ml_star - temp1);
        float evap_raw = ga * (qs_w + dqs_w * dt_ml - qv1);
        float evap_max = (RHO_W * h_lake) / max(dt, 1.0f) + pr_rain + pr_snow;
        evap_out = min(evap_raw, evap_max);
        lhf_out = L_V0 * evap_out;
        dv_m3 = ((pr_rain + pr_snow - evap_out) * area_lake * dt) / RHO_W;

        t_bot_n = (h_bot_c > 1.0e-3f) ? (t_bot + (sw_bot * dt) / (RHO_W * CP_W * h_bot_c)) : t_ml_star;
        float freeze_def = max(T_FREEZE - t_ml_star, 0.0f);
        h_ice_n = min((c_ml * freeze_def) / (RHO_ICE * L_F0), (RHO_W / RHO_ICE) * h_lake);
        t_ml_n = clamp(t_ml_star, T_FREEZE, 330.0f);
        t_bot_n = clamp(t_bot_n, T_FREEZE, 330.0f);
        swe_n = 0.0f;
        t_ice_n = T_FREEZE;
    } else {
        swe_n = swe_ice + pr_snow * dt;
        h_ice_n = h_ice;
        float k_eff = 1.0f / (max(h_ice_n, 1.0e-3f) / K_ICE + (swe_n / RHO_SNOW) / K_SNOW);
        float t0 = clamp(t_ice, 180.0f, T_FREEZE);
        float qs_i, dqs_i;
        esm_qsat_lake(t0, pres_sfc, true, T_FREEZE, EPS_W, L_S0, R_V, qs_i, dqs_i);
        float t03 = t0 * t0 * t0;
        float f_net0 = sw_abs + EMISS * lw_dn - EMISS * SIGMA_SB * t03 * t0 - CP_D * ga * (t0 - temp1) - L_S0 * ga * (qs_i - qv1) + k_eff * (T_FREEZE - t0);
        float df_dt = 4.0f * EMISS * SIGMA_SB * t03 + CP_D * ga + L_S0 * ga * dqs_i + k_eff;
        float c_skin = RHO_ICE * CP_ICE * clamp(h_ice_n, 0.01f, 0.10f);
        float t_star = t0 + (f_net0 * dt) / (c_skin + dt * df_dt);

        float e_melt = max(t_star - T_FREEZE, 0.0f) * (c_skin + dt * df_dt);
        t_ice_n = clamp(t_star, 180.0f, T_FREEZE);
        shf_out = CP_D * ga * (t_ice_n - temp1);
        float evap_raw = ga * (qs_i + dqs_i * (t_ice_n - t0) - qv1);
        float avail_si = swe_n + RHO_ICE * h_ice_n;
        evap_out = min(evap_raw, avail_si / max(dt, 1.0f));
        lhf_out = L_S0 * evap_out;

        float sub_m = evap_out * dt;
        float dm_sn_sub = (sub_m >= 0.0f) ? min(swe_n, sub_m) : sub_m;
        swe_n = max(swe_n - dm_sn_sub, 0.0f);
        float dm_ic_sub = max(sub_m - dm_sn_sub, 0.0f);
        float dh_ic_sub = min(h_ice_n, dm_ic_sub / RHO_ICE);
        h_ice_n = max(h_ice_n - dh_ic_sub, 0.0f);

        float dm_sn_mlt = min(swe_n, e_melt / L_F0);
        swe_n = max(swe_n - dm_sn_mlt, 0.0f);
        float e_rem = max(e_melt - dm_sn_mlt * L_F0, 0.0f);
        float dh_ic_mlt = min(h_ice_n, e_rem / (RHO_ICE * L_F0));
        h_ice_n = max(h_ice_n - dh_ic_mlt, 0.0f);

        float f_cond_b = k_eff * (T_FREEZE - t_ice_n);
        float r_und = (GAMMA_W / max(h_ml_c, H_MIN_ACT)) * dt;
        float fac_u = -esm_expm1(-min(max(r_und, 0.0f), 40.0f));
        float dt_ml_u = max(t_ml - T_FREEZE, 0.0f) * fac_u;
        float e_und = c_ml * dt_ml_u;
        t_ml_n = max(t_ml - dt_ml_u, T_FREEZE);
        t_bot_n = t_bot;
        float dh_base = (f_cond_b * dt - e_und) / (RHO_ICE * L_F0);
        h_ice_n = clamp(h_ice_n + dh_base, 0.0f, (RHO_W / RHO_ICE) * h_lake);

        float dm_sn_dump = 0.0f;
        if (h_ice_n <= 1.0e-4f) {
            dm_sn_dump = swe_n;
            swe_n = 0.0f;
            h_ice_n = 0.0f;
        }
        dv_m3 = ((pr_rain * dt + dm_sn_mlt + dm_sn_dump - dh_ic_sub * RHO_ICE) * area_lake) / RHO_W;
    }

    // Neumaier/Kahan compensated volume addition
    float y_k = dv_m3 + v_comp;
    float t_k = v_lake + y_k;
    float c_k = (fabs(v_lake) >= fabs(y_k)) ? ((v_lake - t_k) + y_k) : ((y_k - t_k) + v_lake);
    float v_new = max(t_k, 0.0f);

    // 5. Post-step hypsometry & 2-layer vertical mixing
    float phi2 = clamp(v_new / max(v_cap, 1.0f), 0.0f, 1.0f);
    float phi2_13 = (phi2 > 0.0f) ? pow(phi2, 1.0f / 3.0f) : 0.0f;
    float f_raw2 = f_cap * phi2_13 * phi2_13;
    float h_raw2 = h_cap * phi2_13;
    bool act2 = (h_raw2 >= H_MIN_ACT) && (f_raw2 > 1.0e-6f);
    float f_lake2 = act2 ? f_raw2 : 0.0f;
    float h_lake2 = act2 ? h_raw2 : 0.0f;

    if (!act2) {
        h_ice_n = 0.0f;
        swe_n = 0.0f;
    } else {
        h_ice_n = min(h_ice_n, (RHO_W / RHO_ICE) * h_lake2);
    }

    float h_liq2 = max(h_lake2 - (RHO_ICE / RHO_W) * h_ice_n, 0.0f);
    float h_ml_min2 = min(H_ML_MIN, h_liq2);
    float h_ml2 = clamp(h_ml_c, h_ml_min2, h_liq2);
    float h_bot2 = max(h_liq2 - h_ml2, 0.0f);

    float dm1 = t_ml_n - T_MD;
    float dm2 = t_bot_n - T_MD;
    float rho_ml = RHO_W * (1.0f - ALPHA_RHO * dm1 * dm1);
    float rho_bot = RHO_W * (1.0f - ALPHA_RHO * dm2 * dm2);

    float t_ml2 = t_ml_n;
    float t_bot2 = t_bot_n;
    bool overturn = act2 && (h_bot2 > 1.0e-3f) && (rho_ml >= rho_bot);
    if (overturn) {
        float t_mix = (h_ml2 * t_ml2 + h_bot2 * t_bot2) / max(h_liq2, 1.0e-3f);
        t_ml2 = t_mix;
        t_bot2 = t_mix;
        h_ml2 = h_liq2;
        h_bot2 = 0.0f;
    } else if (act2 && (h_liq2 > H_ML_MIN + 1.0e-3f) && (h_ice_n <= 1.0e-4f)) {
        float drho = max(rho_bot - rho_ml, 0.0f);
        float n_bv = sqrt((GRAVITY / RHO_W) * drho / max(h_liq2, 1.0f));
        float u_star_w = sqrt(max(rho1, 0.1f) / RHO_W) * sqrt(cd) * u_eff;
        float h_eq = clamp((0.25f * u_star_w) / (fabs(f_cor) + 0.15f * n_bv + 1.0e-4f), h_ml_min2, h_liq2);
        float rel_m = -esm_expm1(-dt / 86400.0f);
        float h_ml_new = clamp(h_ml2 + (h_eq - h_ml2) * rel_m, h_ml_min2, h_liq2);
        float dh = h_ml_new - h_ml2;
        float h_bot_new = max(h_liq2 - h_ml_new, 0.0f);
        if (dh > 0.0f) {
            t_ml2 = (h_ml2 * t_ml2 + dh * t_bot2) / max(h_ml_new, 1.0e-3f);
        } else if (dh < 0.0f && h_bot_new > 1.0e-3f) {
            t_bot2 = (h_bot2 * t_bot2 + (-dh) * t_ml2) / max(h_bot_new, 1.0e-3f);
        }
        h_ml2 = h_ml_new;
        h_bot2 = h_bot_new;
    }

    if (act2 && (h_ml2 > 1.0e-3f) && (h_bot2 > 1.0e-3f)) {
        float h_harm = (h_ml2 * h_bot2) / max(h_liq2, 1.0e-3f);
        float r_dif = KZ_TH / max(h_harm * 0.5f * h_liq2, 1.0e-2f);
        float f_dif = -esm_expm1(-min(max(r_dif * dt, 0.0f), 40.0f));
        float d_t = t_ml2 - t_bot2;
        t_ml2 = t_ml2 - d_t * f_dif * (h_bot2 / max(h_liq2, 1.0e-3f));
        t_bot2 = t_bot2 + d_t * f_dif * (h_ml2 / max(h_liq2, 1.0e-3f));
    } else if (act2) {
        t_bot2 = t_ml2;
    }

    t_ml2 = clamp(t_ml2, T_FREEZE, 330.0f);
    t_bot2 = clamp(t_bot2, T_FREEZE, 330.0f);
    float f_ice2 = (act2 && (h_ice_n > 1.0e-4f)) ? 1.0f : 0.0f;
    float t_rad2 = (f_ice2 > 0.5f) ? t_ice_n : t_ml2;

    state_io[0u * nc + col] = v_new;
    state_io[1u * nc + col] = c_k;
    state_io[2u * nc + col] = t_ml2;
    state_io[3u * nc + col] = t_bot2;
    state_io[4u * nc + col] = h_ml2;
    state_io[5u * nc + col] = h_ice_n;
    state_io[6u * nc + col] = swe_n;
    state_io[7u * nc + col] = t_ice_n;

    diag_out[0u * nc + col] = f_lake2;
    diag_out[1u * nc + col] = h_lake2;
    diag_out[2u * nc + col] = f_ice2;
    diag_out[3u * nc + col] = t_rad2;
    diag_out[4u * nc + col] = alb_vd;
    diag_out[5u * nc + col] = alb_vf;
    diag_out[6u * nc + col] = alb_nd;
    diag_out[7u * nc + col] = alb_nf;
    diag_out[8u * nc + col] = shf_out;
    diag_out[9u * nc + col] = lhf_out;
    diag_out[10u * nc + col] = evap_out;
    diag_out[11u * nc + col] = tau_x;
    diag_out[12u * nc + col] = tau_y;
}

// -------------------------------------------------------------------------
// 20e. SCALE-LT Explicit Bulk Lightning: Fused Charging, 3D Advection,
//      Implicit Sedimentation, Melting, & Ionic Relaxation (1 thread / subcol)
//      AOBA Formulation Eqs. (5.19)-(5.20)
// -------------------------------------------------------------------------
kernel void explicit_lightning_charge_col(
    device const float* q_chg_in    [[buffer(0)]],  // [5, nz, nc, ny, nx]
    device float* q_chg_out         [[buffer(1)]],  // [5, nz, nc, ny, nx]
    device float* rhs_phys_out      [[buffer(2)]],  // [nz, nc, ny, nx] = rho_q_tot * dz / eps0
    device float* ccd_km_out        [[buffer(3)]],  // [nc, ny, nx]
    device const float* rho_d       [[buffer(4)]],  // [nz, nc, ny, nx]
    device const float* theta_m     [[buffer(5)]],  // [nz, nc, ny, nx]
    device const float* q_v         [[buffer(6)]],  // [nz, nc, ny, nx]
    device const float* q_c         [[buffer(7)]],  // [nz, nc, ny, nx]
    device const float* q_i         [[buffer(8)]],  // [nz, nc, ny, nx]
    device const float* q_p         [[buffer(9)]],  // [nz, nc, ny, nx]
    device const float* q_r_opt     [[buffer(10)]], // [nz, nc, ny, nx] or dummy
    device const float* q_s_opt     [[buffer(11)]], // [nz, nc, ny, nx] or dummy
    device const float* q_g_opt     [[buffer(12)]], // [nz, nc, ny, nx] or dummy
    device const float* n_i_opt     [[buffer(13)]], // [nz, nc, ny, nx] or dummy
    device const float* n_s_opt     [[buffer(14)]], // [nz, nc, ny, nx] or dummy
    device const float* n_g_opt     [[buffer(15)]], // [nz, nc, ny, nx] or dummy
    device const float* u_opt       [[buffer(16)]], // [nz, nc, ny, nx] or dummy
    device const float* v_opt       [[buffer(17)]], // [nz, nc, ny, nx] or dummy
    device const float* w_stag      [[buffer(18)]], // [nz+1, nc, ny, nx]
    device const float* z_c         [[buffer(19)]], // [nz, z_cols]
    device const float* dz_c        [[buffer(20)]], // [nz, z_cols]
    constant ulong4& u_p0           [[buffer(21)]], // [nz, nc, ny, nx]
    constant ulong4& u_p1           [[buffer(22)]], // [flags, z_cols, 0, 0]
    constant float4& f_p0           [[buffer(23)]], // [dx, dy, dt, rd_over_cv]
    constant float4& f_p1           [[buffer(24)]], // [p_ref, r_d, rv_over_rd, t_freeze]
    constant float4& f_p2           [[buffer(25)]], // [rho_rain, rho_snow, rho_graupel, t0_precip]
    constant float4& f_p3           [[buffer(26)]], // [t00_precip, t0_graupel, t00_graupel, inv_eps0]
    uint col                        [[thread_position_in_grid]]
) {
    uint nz = uint(u_p0.x);
    uint nc = uint(u_p0.y);
    uint ny = uint(u_p0.z);
    uint nx = uint(u_p0.w);
    uint M = nc * ny * nx;
    if (col >= M || nz == 0u || nz > RAD_MAX_NZ) return;

    uint flags = uint(u_p1.x);
    uint z_cols = uint(u_p1.y);
    bool has_6q  = (flags & 1u) != 0u;
    bool has_qr  = (flags & 2u) != 0u;
    bool has_qs  = (flags & 4u) != 0u;
    bool has_ni  = (flags & 8u) != 0u;
    bool has_ns  = (flags & 16u) != 0u;
    bool has_ng  = (flags & 32u) != 0u;
    bool has_u   = ((flags & 64u) != 0u) && (nx > 1u);
    bool has_v   = ((flags & 128u) != 0u) && (ny > 1u);

    float dx = f_p0.x;
    float dy = f_p0.y;
    float dt = f_p0.z;
    float rd_over_cv = f_p0.w;
    float p_ref = f_p1.x;
    float r_d = f_p1.y;
    float rv_over_rd = f_p1.z;
    float t_freeze = f_p1.w;
    float rho_rain = f_p2.x;
    float rho_snow = f_p2.y;
    float rho_graupel = f_p2.z;
    float t0_precip = f_p2.w;
    float t00_precip = f_p3.x;
    float t0_graupel = f_p3.y;
    float t00_graupel = f_p3.z;
    float inv_eps0 = f_p3.w;

    uint ix = col % nx;
    uint rem = col / nx;
    uint iy = rem % ny;
    uint c = rem / ny;
    uint zc_col = (z_cols > 1u) ? c : 0u;

    uint ix_p = (ix + 1u < nx) ? (ix + 1u) : 0u;
    uint ix_m = (ix > 0u) ? (ix - 1u) : (nx - 1u);
    uint iy_p = (iy + 1u < ny) ? (iy + 1u) : 0u;
    uint iy_m = (iy > 0u) ? (iy - 1u) : (ny - 1u);

    uint col_xp = (c * ny + iy) * nx + ix_p;
    uint col_xm = (c * ny + iy) * nx + ix_m;
    uint col_yp = (c * ny + iy_p) * nx + ix;
    uint col_ym = (c * ny + iy_m) * nx + ix;

    const float PI_F = 3.141592653589793f;
    const float RHO_AIR_REF = 1.225f;
    const float RHO_CLOUD_ICE = 500.0f;
    const float DQ_MAX = 2.5e-14f;
    const float REBOUND_GI = 0.90f;
    const float Q_CLAMP = 5.0e-8f;
    float decay_ion = exp(-dt / 600.0f);
    float decay_dry = exp(-dt / 60.0f);
    float melt_time_fac = 1.0f - exp(-dt / 180.0f);
    float cfl_lim = (has_u && has_v) ? 0.225f : 0.45f;
    float c_max_x = cfl_lim * dx / max(dt, 1.0e-3f);
    float c_max_y = cfl_lim * dy / max(dt, 1.0e-3f);
    float dt_over_dx = dt / dx;
    float dt_over_dy = dt / dy;

    float temp_arr[RAD_MAX_NZ];
    float dq_i_arr[RAD_MAX_NZ];
    float dq_s_arr[RAD_MAX_NZ];
    float melt_to_r[RAD_MAX_NZ];
    float rho_sum[RAD_MAX_NZ];
    float vt_k[RAD_MAX_NZ];
    float c_p[RAD_MAX_NZ];
    float d_p[RAD_MAX_NZ];

    float z_freeze = 0.0f;
    float z_ice_top = 0.0f;

    // Pass 0: Thermodynamics, hydrometeor properties, Takahashi charging, cold-cloud depth
    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * M + col;
        float rd = rho_d[idx];
        float th = theta_m[idx];
        float qv = max(q_v[idx], 0.0f);
        float arg = max(r_d * rd * th / p_ref, 1.0e-8f);
        float exner = pow(arg, rd_over_cv);
        float temp = exner * (th / (1.0f + rv_over_rd * qv));
        temp_arr[k] = temp;
        melt_to_r[k] = 0.0f;
        rho_sum[k] = 0.0f;

        float rho_pos = max(rd, 1.0e-4f);
        float qc_p = max(q_c[idx], 0.0f);
        float qi_p = max(q_i[idx], 0.0f);
        float qs_v, qg_v;
        if (has_6q) {
            qs_v = has_qs ? max(q_s_opt[idx], 0.0f) : 0.0f;
            qg_v = max(q_g_opt[idx], 0.0f);
        } else {
            float f_ice_p = clamp((t0_precip - temp) / max(t0_precip - t00_precip, 1.0f), 0.0f, 1.0f);
            float f_graup = clamp((temp - t00_graupel) / max(t0_graupel - t00_graupel, 1.0f), 0.0f, 1.0f);
            float qp_p = max(q_p[idx], 0.0f);
            float qp_ice = f_ice_p * qp_p;
            qg_v = qp_ice * f_graup;
            qs_v = qp_ice * (1.0f - f_graup);
        }

        float z_val = z_c[k * z_cols + zc_col];
        if (temp >= t_freeze && z_val > z_freeze) z_freeze = z_val;
        if ((qi_p + qs_v + qg_v) >= 1.0e-5f && z_val > z_ice_top) z_ice_top = z_val;

        float rho_fac = sqrt(RHO_AIR_REF / rho_pos);
        float vtg = 0.0f;
        float dg = 2.0e-4f;
        float ng_v = 1.0f;
        if (qg_v > 1.0e-7f) {
            float mg = rho_pos * qg_v;
            ng_v = has_ng ? max(rho_pos * n_g_opt[idx], 1.0f)
                          : max(4.0e6f * sqrt(sqrt(max(mg / (PI_F * rho_graupel * 4.0e6f) + 1.0e-24f, 1.0e-24f))), 1.0f);
            dg = clamp(pow(6.0f * mg / (PI_F * rho_graupel * ng_v) + 1.0e-20f, 1.0f / 3.0f), 2.0e-4f, 1.5e-2f);
            vtg = clamp(94.5f * sqrt(dg) * rho_fac, 0.5f, 12.0f);
        }
        vt_k[k] = vtg;

        // Takahashi (1978) / Sato et al. (2019) polarity-reversal charging (Eq. 5.19)
        // Compute cloud-ice and snow sizes/fall-speeds strictly inside active mixed-phase riming zones
        float t_c = temp - t_freeze;
        float dq_i = 0.0f;
        float dq_s = 0.0f;
        if (t_c < 0.0f && qg_v > 1.0e-6f && qc_p > 0.0f && (qi_p > 1.0e-7f || qs_v > 1.0e-6f)) {
            float lwc = 1.0e3f * rho_pos * qc_p;
            float t_rev = -10.0f - 10.0f * tanh(0.65f * (lwc - 0.55f));
            float pol = tanh((t_c - t_rev) * 0.25f);
            float lwc_f = lwc / (lwc + 0.15f);
            float t_gate = esm_sigmoid((t_c + 40.0f) / 3.0f) * esm_sigmoid(-t_c / 1.5f);
            float dq_coll = DQ_MAX * pol * lwc_f * t_gate;
            if (qi_p > 1.0e-7f) {
                float mi = rho_pos * qi_p;
                float ni_v = has_ni ? max(rho_pos * n_i_opt[idx], 1.0f)
                                    : max(1.0e5f * sqrt(max(mi * 1.0e3f, 0.0f)), 1.0f);
                float di = clamp(pow(6.0f * mi / (PI_F * RHO_CLOUD_ICE * ni_v) + 1.0e-20f, 1.0f / 3.0f), 2.0e-5f, 5.0e-4f);
                float vti = clamp(0.45f * rho_fac, 0.05f, 1.5f);
                float dgi = dg + di;
                float r_gi = REBOUND_GI * (0.25f * PI_F) * (dgi * dgi) * fabs(vtg - vti) * ng_v * ni_v;
                dq_i = clamp(-dt * r_gi * dq_coll, -Q_CLAMP, Q_CLAMP);
            }
            if (qs_v > 1.0e-6f) {
                float ms = rho_pos * qs_v;
                float ns_v = has_ns ? max(rho_pos * n_s_opt[idx], 1.0f)
                                    : max(3.0e6f * sqrt(sqrt(max(ms / (PI_F * rho_snow * 3.0e6f) + 1.0e-24f, 1.0e-24f))), 1.0f);
                float ds = clamp(pow(6.0f * ms / (PI_F * rho_snow * ns_v) + 1.0e-20f, 1.0f / 3.0f), 1.0e-4f, 6.0e-3f);
                float vts = clamp(4.84f * sqrt(sqrt(ds)) * rho_fac, 0.2f, 3.0f);
                float dgs = dg + ds;
                float r_gs = REBOUND_GI * (0.25f * PI_F) * (dgs * dgs) * fabs(vtg - vts) * ng_v * ns_v;
                dq_s = clamp(-dt * r_gs * dq_coll, -Q_CLAMP, Q_CLAMP);
            }
        }
        dq_i_arr[k] = dq_i;
        dq_s_arr[k] = dq_s;
    }
    ccd_km_out[col] = max((z_ice_top - z_freeze) * 1.0e-3f, 0.0f);

    // Process 5 species in order s = 4 (g), 3 (s), 2 (i), 1 (r), 0 (c) so melting flows g,s -> r in 1 pass
    for (int si = 4; si >= 0; --si) {
        uint s = uint(si);
        uint s_off = s * nz * M;

        // Compute fall speed vt_k[k] for species s (s=4 already populated in Pass 0)
        if (s != 4u) {
            for (uint k = 0u; k < nz; ++k) {
                uint idx = k * M + col;
                if (s == 0u) {
                    vt_k[k] = 0.0f;
                } else {
                    float rho_pos = max(rho_d[idx], 1.0e-4f);
                    float rho_fac = sqrt(RHO_AIR_REF / rho_pos);
                    float temp = temp_arr[k];
                    if (s == 2u) {
                        float qi_p = max(q_i[idx], 0.0f);
                        vt_k[k] = (qi_p > 1.0e-7f) ? clamp(0.45f * rho_fac, 0.05f, 1.5f) : 0.0f;
                    } else if (s == 3u) {
                        float qs_v;
                        if (has_6q) {
                            qs_v = has_qs ? max(q_s_opt[idx], 0.0f) : 0.0f;
                        } else {
                            float f_ice_p = clamp((t0_precip - temp) / max(t0_precip - t00_precip, 1.0f), 0.0f, 1.0f);
                            float f_graup = clamp((temp - t00_graupel) / max(t0_graupel - t00_graupel, 1.0f), 0.0f, 1.0f);
                            qs_v = f_ice_p * max(q_p[idx], 0.0f) * (1.0f - f_graup);
                        }
                        if (qs_v > 1.0e-7f) {
                            float ms = rho_pos * qs_v;
                            float ns_v = has_ns ? max(rho_pos * n_s_opt[idx], 1.0f)
                                                : max(3.0e6f * sqrt(sqrt(max(ms / (PI_F * rho_snow * 3.0e6f) + 1.0e-24f, 1.0e-24f))), 1.0f);
                            float ds = clamp(pow(6.0f * ms / (PI_F * rho_snow * ns_v) + 1.0e-20f, 1.0f / 3.0f), 1.0e-4f, 6.0e-3f);
                            vt_k[k] = clamp(4.84f * sqrt(sqrt(ds)) * rho_fac, 0.2f, 3.0f);
                        } else {
                            vt_k[k] = 0.0f;
                        }
                    } else { // s == 1u (rain)
                        float qr_v;
                        if (has_6q) {
                            qr_v = has_qr ? max(q_r_opt[idx], 0.0f) : 0.0f;
                        } else {
                            float f_ice_p = clamp((t0_precip - temp) / max(t0_precip - t00_precip, 1.0f), 0.0f, 1.0f);
                            qr_v = (1.0f - f_ice_p) * max(q_p[idx], 0.0f);
                        }
                        if (qr_v > 1.0e-7f) {
                            float mr = rho_pos * qr_v;
                            float dr = clamp(sqrt(sqrt(mr / (PI_F * rho_rain * 8.0e6f) + 1.0e-24f)), 1.0e-4f, 5.0e-3f);
                            vt_k[k] = clamp(842.0f * pow(dr, 0.8f) * rho_fac, 0.5f, 10.0f);
                        } else {
                            vt_k[k] = 0.0f;
                        }
                    }
                }
            }
        }

        // Forward Thomas sweep combining horizontal donor-cell advection, non-inductive sep, & vertical implicit transport
        float w_pos_prev = 0.0f;
        float w_neg_prev = 0.0f;
        float v_bot = max(vt_k[0], 0.0f);

        for (uint k = 0u; k < nz; ++k) {
            uint k_off = k * M;
            uint q_idx = s_off + k_off + col;
            float q_orig = q_chg_in[q_idx];
            float q_val = q_orig;

            // Simultaneous conservative periodic horizontal donor-cell advection
            if (has_u) {
                float u_r = clamp(u_opt[k_off + col], -c_max_x, c_max_x);
                float u_l = clamp(u_opt[k_off + col_xm], -c_max_x, c_max_x);
                float q_xp = q_chg_in[s_off + k_off + col_xp];
                float q_xm = q_chg_in[s_off + k_off + col_xm];
                float fx_r = max(u_r, 0.0f) * q_orig + min(u_r, 0.0f) * q_xp;
                float fx_l = max(u_l, 0.0f) * q_xm + min(u_l, 0.0f) * q_orig;
                q_val -= dt_over_dx * (fx_r - fx_l);
            }
            if (has_v) {
                float v_u = clamp(v_opt[k_off + col], -c_max_y, c_max_y);
                float v_d = clamp(v_opt[k_off + col_ym], -c_max_y, c_max_y);
                float q_yp = q_chg_in[s_off + k_off + col_yp];
                float q_ym = q_chg_in[s_off + k_off + col_ym];
                float fy_u = max(v_u, 0.0f) * q_orig + min(v_u, 0.0f) * q_yp;
                float fy_d = max(v_d, 0.0f) * q_ym + min(v_d, 0.0f) * q_orig;
                q_val -= dt_over_dy * (fy_u - fy_d);
            }

            // Add strict zero-sum non-inductive charge separation
            if (s == 2u) q_val += dq_i_arr[k];
            else if (s == 3u) q_val += dq_s_arr[k];
            else if (s == 4u) q_val -= (dq_i_arr[k] + dq_s_arr[k]);

            float dz_k = max(dz_c[k * z_cols + zc_col], 1.0f);
            float w_pos_cur = 0.0f;
            float w_neg_cur = 0.0f;
            if (k + 1u < nz) {
                float w_int = w_stag[(k + 1u) * M + col] - 0.5f * (vt_k[k] + vt_k[k + 1u]);
                w_pos_cur = max(w_int, 0.0f);
                w_neg_cur = min(w_int, 0.0f);
            }

            float dl = (k == 0u) ? 0.0f : (-dt * w_pos_prev);
            float du = (k + 1u < nz) ? (dt * w_neg_cur) : 0.0f;
            float out_up = dt * w_pos_cur;
            float out_dn = (k == 0u) ? (dt * v_bot) : (-dt * w_neg_prev);
            float diag = dz_k + out_up + out_dn;
            float rhs = dz_k * q_val;

            if (k == 0u) {
                float inv_d = 1.0f / diag;
                c_p[0] = du * inv_d;
                d_p[0] = rhs * inv_d;
            } else {
                float inv_d = 1.0f / (diag - dl * c_p[k - 1u]);
                c_p[k] = du * inv_d;
                d_p[k] = (rhs - dl * d_p[k - 1u]) * inv_d;
            }

            w_pos_prev = w_pos_cur;
            w_neg_prev = w_neg_cur;
        }

        // Backward Thomas substitution + melting transfer + ionic relaxation + total charge accumulation
        float q_above = 0.0f;
        for (int ki = int(nz) - 1; ki >= 0; --ki) {
            uint k = uint(ki);
            float q_new = (k == nz - 1u) ? d_p[k] : (d_p[k] - c_p[k] * q_above);
            q_above = q_new;

            float temp = temp_arr[k];
            if (s == 3u || s == 4u) {
                float mf = clamp((temp - t_freeze) / 3.0f, 0.0f, 1.0f) * melt_time_fac;
                melt_to_r[k] += mf * q_new;
                q_new *= (1.0f - mf);
            } else if (s == 1u) {
                q_new += melt_to_r[k];
            }

            uint idx = k * M + col;
            float qm = 0.0f;
            if (s == 0u) qm = max(q_c[idx], 0.0f);
            else if (s == 2u) qm = max(q_i[idx], 0.0f);
            else if (has_6q) {
                if (s == 1u) qm = has_qr ? max(q_r_opt[idx], 0.0f) : 0.0f;
                else if (s == 3u) qm = has_qs ? max(q_s_opt[idx], 0.0f) : 0.0f;
                else qm = max(q_g_opt[idx], 0.0f);
            } else {
                float f_ice_p = clamp((t0_precip - temp) / max(t0_precip - t00_precip, 1.0f), 0.0f, 1.0f);
                float qp_p = max(q_p[idx], 0.0f);
                if (s == 1u) qm = (1.0f - f_ice_p) * qp_p;
                else {
                    float f_graup = clamp((temp - t00_graupel) / max(t0_graupel - t00_graupel, 1.0f), 0.0f, 1.0f);
                    qm = (s == 4u) ? (f_ice_p * qp_p * f_graup) : (f_ice_p * qp_p * (1.0f - f_graup));
                }
            }
            float decay = (qm > 1.0e-8f) ? decay_ion : decay_dry;
            q_new = clamp(q_new * decay, -Q_CLAMP, Q_CLAMP);
            q_chg_out[s_off + k * M + col] = q_new;
            rho_sum[k] += q_new;
        }
    }

    // Write pre-scaled Poisson RHS = rho_q_tot * dz / eps0 directly for rfft2
    for (uint k = 0u; k < nz; ++k) {
        float dz_k = max(dz_c[k * z_cols + zc_col], 1.0f);
        rhs_phys_out[k * M + col] = rho_sum[k] * dz_k * inv_eps0;
    }
}

// -------------------------------------------------------------------------
// 20f. SCALE-LT Explicit Bulk Lightning: Exact Batched Complex Thomas Solver
//      in Fourier Space (1 thread per (c, ky, kx) mode; zero matrix allocs)
//      AOBA Formulation Eq. (5.21)
// -------------------------------------------------------------------------
kernel void explicit_lightning_poisson_thomas(
    device const float* rhs_hat_ri  [[buffer(0)]], // [nz, nc, ny, n_kx, 2]
    device float* phi_hat_ri        [[buffer(1)]], // [nz, nc, ny, n_kx, 2]
    device const float* dz_c        [[buffer(2)]], // [nz, z_cols]
    constant ulong4& u_p0           [[buffer(3)]], // [nz, nc, ny, nx]
    constant ulong4& u_p1           [[buffer(4)]], // [n_kx, z_cols, 0, 0]
    constant float4& f_p0           [[buffer(5)]], // [dx, dy, 0, 0]
    uint tid                        [[thread_position_in_grid]]
) {
    uint nz = uint(u_p0.x);
    uint nc = uint(u_p0.y);
    uint ny = uint(u_p0.z);
    uint nx = uint(u_p0.w);
    uint n_kx = uint(u_p1.x);
    uint z_cols = uint(u_p1.y);
    uint m_tot = nc * ny * n_kx;
    if (tid >= m_tot || nz == 0u || nz > RAD_MAX_NZ) return;

    uint ix_k = tid % n_kx;
    uint rem = tid / n_kx;
    uint iy = rem % ny;
    uint c = rem / ny;
    uint zc_col = (z_cols > 1u) ? c : 0u;

    const float PI_F = 3.141592653589793f;
    float dx = f_p0.x;
    float dy = f_p0.y;
    float sx = sin(PI_F * float(ix_k) / float(max(nx, 1u)));
    float lam_x = (4.0f / (dx * dx)) * sx * sx;
    float lam_y = 0.0f;
    if (ny > 1u) {
        float sy = sin(PI_F * float(iy) / float(ny));
        lam_y = (4.0f / (dy * dy)) * sy * sy;
    }
    float lam_h = lam_x + lam_y;

    float c_star[RAD_MAX_NZ];
    float d_re[RAD_MAX_NZ];
    float d_im[RAD_MAX_NZ];

    float dz_cur = max(dz_c[0u * z_cols + zc_col], 1.0f);
    float dz_prev = dz_cur;
    for (uint k = 0u; k < nz; ++k) {
        float dz_next = (k + 1u < nz) ? max(dz_c[(k + 1u) * z_cols + zc_col], 1.0f) : dz_cur;
        float dz_dn = (k == 0u) ? (0.5f * dz_cur) : (0.5f * (dz_prev + dz_cur));
        float dz_up = (k + 1u == nz) ? (0.5f * dz_cur) : (0.5f * (dz_cur + dz_next));
        float c_dn = 1.0f / dz_dn;
        float c_up = 1.0f / dz_up;

        float a = (k == 0u) ? 0.0f : -c_dn;
        float d = (k + 1u == nz) ? 0.0f : -c_up;
        float b = c_dn + c_up + dz_cur * lam_h;

        uint idx = (k * m_tot + tid) * 2u;
        float r_re = rhs_hat_ri[idx];
        float r_im = rhs_hat_ri[idx + 1u];

        if (k == 0u) {
            float inv_b = 1.0f / b;
            c_star[0] = d * inv_b;
            d_re[0] = r_re * inv_b;
            d_im[0] = r_im * inv_b;
        } else {
            float inv_b = 1.0f / (b - a * c_star[k - 1u]);
            c_star[k] = d * inv_b;
            d_re[k] = (r_re - a * d_re[k - 1u]) * inv_b;
            d_im[k] = (r_im - a * d_im[k - 1u]) * inv_b;
        }
        dz_prev = dz_cur;
        dz_cur = dz_next;
    }

    float p_re_ab = 0.0f;
    float p_im_ab = 0.0f;
    for (int ki = int(nz) - 1; ki >= 0; --ki) {
        uint k = uint(ki);
        float p_re = (k == nz - 1u) ? d_re[k] : (d_re[k] - c_star[k] * p_re_ab);
        float p_im = (k == nz - 1u) ? d_im[k] : (d_im[k] - c_star[k] * p_im_ab);
        p_re_ab = p_re;
        p_im_ab = p_im;
        uint idx = (k * m_tot + tid) * 2u;
        phi_hat_ri[idx] = p_re;
        phi_hat_ri[idx + 1u] = p_im;
    }
}

// -------------------------------------------------------------------------
// 20g. SCALE-LT Explicit Bulk Lightning: E-Field Gradient, Breakdown Trigger,
//      Fast Untriggered Early Exit, Zero-Sum IC/CG Discharge & Flash Rate
//      1 Threadgroup (64 threads) per Host Cell c
//      AOBA Formulation Eqs. (5.22)-(5.23)
// -------------------------------------------------------------------------
kernel void explicit_lightning_discharge_grp(
    device float* q_chg             [[buffer(0)]], // [5, nz, nc, ny, nx] updated in-place
    device const float* phi_e       [[buffer(1)]], // [nz, nc, ny, nx]
    device const float* rho_d       [[buffer(2)]], // [nz, nc, ny, nx]
    device const float* z_c         [[buffer(3)]], // [nz, z_cols]
    device const float* dz_c        [[buffer(4)]], // [nz, z_cols]
    device float* e_mag_out         [[buffer(5)]], // [nz, nc, ny, nx]
    device float* e_z_out           [[buffer(6)]], // [nz, nc, ny, nx]
    device float* flash_tot_out     [[buffer(7)]], // [nc, ny, nx]
    device float* flash_cg_out      [[buffer(8)]], // [nc, ny, nx]
    device float* f_cg_out          [[buffer(9)]], // [nc, ny, nx]
    constant ulong4& u_p0           [[buffer(10)]], // [nz, nc, ny, nx]
    constant ulong4& u_p1           [[buffer(11)]], // [z_cols, 0, 0, 0]
    constant float4& f_p0           [[buffer(12)]], // [dx, dy, dt, e_crit_sl]
    constant float4& f_p1           [[buffer(13)]], // [extent_frac, neut_frac, q_per_flash, z_cg_max]
    uint ltid                       [[thread_position_in_threadgroup]],
    uint c                          [[threadgroup_position_in_grid]]
) {
    uint nz = uint(u_p0.x);
    uint nc = uint(u_p0.y);
    uint ny = uint(u_p0.z);
    uint nx = uint(u_p0.w);
    uint z_cols = uint(u_p1.x);
    if (c >= nc || nz == 0u || nz > RAD_MAX_NZ) return;

    uint n_sub = ny * nx;
    uint M = nc * n_sub;
    uint zc_col = (z_cols > 1u) ? c : 0u;

    float dx = f_p0.x;
    float dy = f_p0.y;
    float dt = f_p0.z;
    float e_crit_sl = f_p0.w;
    float ext_frac = f_p1.x;
    float neut_frac = f_p1.y;
    float q_per_flash = f_p1.z;
    float z_cg_max = f_p1.w;
    const float RHO_AIR_REF = 1.225f;
    float inv_2dx = 0.5f / dx;
    float inv_2dy = 0.5f / dy;

    threadgroup float sh_a[64];
    threadgroup float sh_b[64];

    // Pass 1: Compute 3D E-field from phi_e, write e_mag & e_z, find max |E|/E_crit
    float loc_max_ratio = 0.0f;
    for (uint sub = ltid; sub < n_sub; sub += 64u) {
        uint ix = sub % nx;
        uint iy = sub / nx;
        uint ix_p = (ix + 1u < nx) ? (ix + 1u) : 0u;
        uint ix_m = (ix > 0u) ? (ix - 1u) : (nx - 1u);
        uint iy_p = (iy + 1u < ny) ? (iy + 1u) : 0u;
        uint iy_m = (iy > 0u) ? (iy - 1u) : (ny - 1u);

        uint col = c * n_sub + sub;
        uint col_xp = (c * ny + iy) * nx + ix_p;
        uint col_xm = (c * ny + iy) * nx + ix_m;
        uint col_yp = (c * ny + iy_p) * nx + ix;
        uint col_ym = (c * ny + iy_m) * nx + ix;

        float dz_cur = max(dz_c[0u * z_cols + zc_col], 1.0f);
        float dz_prev = dz_cur;
        float phi_cur = phi_e[0u * M + col];
        float phi_prev = 0.0f;

        for (uint k = 0u; k < nz; ++k) {
            uint k_off = k * M;
            uint idx = k_off + col;
            float dz_next = (k + 1u < nz) ? max(dz_c[(k + 1u) * z_cols + zc_col], 1.0f) : dz_cur;
            float phi_next = (k + 1u < nz) ? phi_e[(k + 1u) * M + col] : 0.0f;

            float ex = (nx > 1u) ? (-(phi_e[k_off + col_xp] - phi_e[k_off + col_xm]) * inv_2dx) : 0.0f;
            float ey = (ny > 1u) ? (-(phi_e[k_off + col_yp] - phi_e[k_off + col_ym]) * inv_2dy) : 0.0f;
            float ez = 0.0f;
            if (nz > 1u) {
                float f_dn = (k == 0u) ? (phi_cur / (0.5f * dz_cur))
                                       : ((phi_cur - phi_prev) / (0.5f * (dz_prev + dz_cur)));
                float f_up = (k + 1u == nz) ? (-phi_cur / (0.5f * dz_cur))
                                            : ((phi_next - phi_cur) / (0.5f * (dz_cur + dz_next)));
                ez = -0.5f * (f_dn + f_up);
            }
            float emag = sqrt(ex * ex + ey * ey + ez * ez + 1.0e-12f);
            e_mag_out[idx] = emag;
            e_z_out[idx] = ez;

            float rho_pos = max(rho_d[idx], 1.0e-4f);
            float e_crit = e_crit_sl * max(rho_pos / RHO_AIR_REF, 0.05f);
            float er = emag / e_crit;
            if (er > loc_max_ratio) loc_max_ratio = er;

            phi_prev = phi_cur;
            phi_cur = phi_next;
            dz_prev = dz_cur;
            dz_cur = dz_next;
        }
    }

    sh_a[ltid] = loc_max_ratio;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 32u; off > 0u; off >>= 1u) {
        if (ltid < off) sh_a[ltid] = max(sh_a[ltid], sh_a[ltid + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dom_max_ratio = sh_a[0];

    // Fast uniform early exit: non-thunderstorm cells skip touching q_chg completely!
    if (dom_max_ratio < 1.0f) {
        for (uint sub = ltid; sub < n_sub; sub += 64u) {
            uint col = c * n_sub + sub;
            flash_tot_out[col] = 0.0f;
            flash_cg_out[col] = 0.0f;
            f_cg_out[col] = 0.0f;
        }
        return;
    }

    // Pass 2a (triggered domains): detect CG connection and IC channel-volume charge imbalance
    float neut_eff = min(neut_frac * clamp(dom_max_ratio, 1.0f, 3.0f), 0.80f);
    float z0 = z_c[0u * z_cols + zc_col];
    float dz0 = max(dz_c[0u * z_cols + zc_col], 1.0f);
    float area_sub = dx * dy;

    float loc_any_cg = 0.0f;
    float loc_dq_vol = 0.0f;
    float loc_chan_vol = 0.0f;

    for (uint sub = ltid; sub < n_sub; sub += 64u) {
        uint col = c * n_sub + sub;
        float rho0 = max(rho_d[0u * M + col], 1.0e-4f);
        float ec0 = max(e_crit_sl * max(rho0 / RHO_AIR_REF, 0.05f), 1.0f);
        if (fabs(e_z_out[0u * M + col]) / ec0 >= 0.25f) loc_any_cg = 1.0f;

        for (uint k = 0u; k < nz; ++k) {
            uint idx = k * M + col;
            float rho_pos = max(rho_d[idx], 1.0e-4f);
            float e_crit = e_crit_sl * max(rho_pos / RHO_AIR_REF, 0.05f);
            if (e_mag_out[idx] >= ext_frac * e_crit) {
                float dz_k = max(dz_c[k * z_cols + zc_col], 1.0f);
                float z_agl = z_c[k * z_cols + zc_col] - z0 + 0.5f * dz0;
                if (z_agl <= z_cg_max) loc_any_cg = 1.0f;

                float qsum = 0.0f;
                for (uint s = 0u; s < 5u; ++s) qsum += q_chg[s * nz * M + idx];
                float cvol = dz_k * area_sub;
                loc_dq_vol += (-neut_eff * qsum) * cvol;
                loc_chan_vol += cvol;
            }
        }
    }

    sh_a[ltid] = loc_any_cg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 32u; off > 0u; off >>= 1u) {
        if (ltid < off) sh_a[ltid] = max(sh_a[ltid], sh_a[ltid + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    bool is_cg_dom = (sh_a[0] > 0.5f);

    sh_a[ltid] = loc_dq_vol;
    sh_b[ltid] = loc_chan_vol;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 32u; off > 0u; off >>= 1u) {
        if (ltid < off) {
            sh_a[ltid] += sh_a[ltid + off];
            sh_b[ltid] += sh_b[ltid + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    bool ic_only = (!is_cg_dom) && (sh_b[0] > 1.0f);
    float imbal_mean = ic_only ? (sh_a[0] / max(sh_b[0], 1.0f)) : 0.0f;

    // Pass 2b: Apply in-place charge neutralization (+ exact IC zero-sum) and compute flash rates
    for (uint sub = ltid; sub < n_sub; sub += 64u) {
        uint col = c * n_sub + sub;
        float rho0 = max(rho_d[0u * M + col], 1.0e-4f);
        float ec0 = max(e_crit_sl * max(rho0 / RHO_AIR_REF, 0.05f), 1.0f);
        bool is_cg_sub = (fabs(e_z_out[0u * M + col]) / ec0 >= 0.25f);
        float dq_abs_col_dz = 0.0f;

        for (uint k = 0u; k < nz; ++k) {
            uint idx = k * M + col;
            float rho_pos = max(rho_d[idx], 1.0e-4f);
            float e_crit = e_crit_sl * max(rho_pos / RHO_AIR_REF, 0.05f);
            if (e_mag_out[idx] >= ext_frac * e_crit) {
                float dz_k = max(dz_c[k * z_cols + zc_col], 1.0f);
                float z_agl = z_c[k * z_cols + zc_col] - z0 + 0.5f * dz0;
                if (z_agl <= z_cg_max) is_cg_sub = true;

                float qv5[5];
                float abs_sum = 0.0f;
                for (uint s = 0u; s < 5u; ++s) {
                    float qv = q_chg[s * nz * M + idx];
                    qv5[s] = qv;
                    abs_sum += fabs(qv);
                }
                float inv_abs = 1.0f / max(abs_sum, 1.0e-18f);
                float dq_tot = 0.0f;
                for (uint s = 0u; s < 5u; ++s) {
                    float dq = -neut_eff * qv5[s];
                    if (ic_only) dq -= (fabs(qv5[s]) * inv_abs) * imbal_mean;
                    q_chg[s * nz * M + idx] = qv5[s] + dq;
                    dq_tot += dq;
                }
                dq_abs_col_dz += fabs(dq_tot) * dz_k;
            }
        }

        float flash_tot = (0.5f * dq_abs_col_dz) / (q_per_flash * dt);
        float f_cg = (flash_tot > 0.0f && is_cg_sub) ? 1.0f : 0.0f;
        flash_tot_out[col] = flash_tot;
        f_cg_out[col] = f_cg;
        flash_cg_out[col] = flash_tot * f_cg;
    }
}
"""


def _pack_rrtmgp_gpu_tables(tabs: Any, device: torch.device) -> dict[str, Any]:
    """Pre-pack official RRTMGP LW & SW tables into flat contiguous GPU buffers with CSR minors."""
    key = ("rrtmgp_gpu", str(device), id(tabs))
    cached = _ESM_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    def _build_csr_minors(specs_l: list[Any], specs_u: list[Any], n_gpt: int, n_contrib_l: int):
        per_g_l: list[list[tuple[int, int, int, int]]] = [[] for _ in range(n_gpt)]
        for sp in specs_l:
            flg = (1 if sp.scale_by_comp else 0) | (2 if sp.scale_by_dens else 0)
            for g in range(sp.gpt_start, sp.gpt_end):
                per_g_l[g].append((sp.gas_idx, sp.scaling_gas_idx, flg, sp.k_start + (g - sp.gpt_start)))
        per_g_u: list[list[tuple[int, int, int, int]]] = [[] for _ in range(n_gpt)]
        for sp in specs_u:
            flg = (1 if sp.scale_by_comp else 0) | (2 if sp.scale_by_dens else 0)
            for g in range(sp.gpt_start, sp.gpt_end):
                per_g_u[g].append((sp.gas_idx, sp.scaling_gas_idx, flg, n_contrib_l + sp.k_start + (g - sp.gpt_start)))

        off_l = [0]
        entries: list[int] = []
        for g in range(n_gpt):
            for e in per_g_l[g]:
                entries.extend(e)
            off_l.append(len(entries) // 4)
        off_u = [len(entries) // 4]
        for g in range(n_gpt):
            for e in per_g_u[g]:
                entries.extend(e)
            off_u.append(len(entries) // 4)
        if not entries:
            entries = [0, 0, 0, 0]
        return off_l, off_u, entries

    # --- LW packing (16 bands / 128 g-points) ---
    lw = tabs.lw
    n_cl_lw = int(lw.kminor_lower.shape[2])
    kminor_lw = torch.cat([lw.kminor_lower, lw.kminor_upper], dim=2).contiguous()
    off_l_lw, off_u_lw, ent_lw = _build_csr_minors(lw.minor_lower, lw.minor_upper, lw.n_gpt, n_cl_lw)

    lw_wn = lw.bnd_wavenum_mid.detach().cpu()
    k_liq_lw_b = (0.065 + 0.020 * torch.exp(-((lw_wn - 650.0) / 350.0) ** 2)).tolist()
    k_liq_lw_r = (0.28 + 0.10 * torch.exp(-((lw_wn - 950.0) / 250.0) ** 2)).tolist()
    k_ice_lw_b = (0.0025 + 0.0025 * torch.exp(-lw_wn / 600.0)).tolist()
    k_ice_lw_r = (1.05 + 0.12 * torch.exp(-((lw_wn - 900.0) / 300.0) ** 2)).tolist()
    k_dust_lw_b = (95.0 + 115.0 * torch.exp(-((lw_wn - 1030.0) / 160.0) ** 2)).tolist()

    m_air_kg_molec = 28.9647e-3 / 6.02214076e23
    scalars_lw = [
        lw.press_ref_ln0, lw.press_ref_dln, lw.press_ref_trop,
        lw.temp_ref_0, lw.temp_ref_d, lw.ref_p, lw.ref_t,
        m_air_kg_molec, C.CP_D, C.O3_G1, C.O3_G2, C.O3_G3, math.pi,
        0.0, 0.0, 0.0,
    ]
    tbl_f_lw = torch.tensor(
        scalars_lw
        + lw.diff_angle_bnd.detach().cpu().tolist()
        + k_liq_lw_b + k_liq_lw_r + k_ice_lw_b + k_ice_lw_r + k_dust_lw_b
        + lw.totplnk.detach().cpu().reshape(-1).tolist()
        + lw.vmr_ref.detach().cpu().reshape(-1).tolist(),
        dtype=torch.float32,
        device=device,
    ).contiguous()

    tbl_i_lw = torch.tensor(
        lw.gpt_to_bnd.detach().cpu().tolist()
        + lw.key_species.detach().cpu().reshape(-1).tolist()
        + off_l_lw
        + off_u_lw
        + ent_lw,
        dtype=torch.int32,
        device=device,
    ).contiguous()

    # --- SW packing (14 bands / 112 g-points) ---
    sw = tabs.sw
    n_cl_sw = int(sw.kminor_lower.shape[2])
    kminor_sw = torch.cat([sw.kminor_lower, sw.kminor_upper], dim=2).contiguous()
    rayl_sw = torch.cat([sw.rayl_lower.unsqueeze(0), sw.rayl_upper.unsqueeze(0)], dim=0).contiguous()
    off_l_sw, off_u_sw, ent_sw = _build_csr_minors(sw.minor_lower, sw.minor_upper, sw.n_gpt, n_cl_sw)

    sw_lam = 1.0e4 / torch.clamp(sw.bnd_wavenum_mid.detach().cpu(), min=500.0)
    sw_aero_sc = torch.clamp((0.55 / sw_lam) ** 1.05, 0.15, 2.0).tolist()
    sw_dust_ke = (1200.0 * torch.clamp((0.55 / sw_lam) ** 0.25, 0.5, 1.3)).tolist()
    sw_dust_ss = torch.clamp(0.92 - 0.12 * torch.exp(-((sw_lam - 0.40) / 0.18) ** 2), 0.78, 0.96).tolist()
    sw_cld_oml = torch.clamp(1.0 - 0.0015 * (sw_lam ** 2.2), 0.88, 0.9999).tolist()
    sw_cld_omi = torch.clamp(1.0 - 0.0025 * (sw_lam ** 2.1), 0.85, 0.9999).tolist()

    scalars_sw = [
        sw.press_ref_ln0, sw.press_ref_dln, sw.press_ref_trop,
        sw.temp_ref_0, sw.temp_ref_d, sw.ref_p, sw.ref_t,
        m_air_kg_molec, C.CP_D, C.O3_G1, C.O3_G2, C.O3_G3, math.pi,
        0.0, 0.0, 0.0,
    ]
    tbl_f_sw = torch.tensor(
        scalars_sw
        + sw.solar_frac_gpt.detach().cpu().tolist()
        + sw_aero_sc + sw_dust_ke + sw_dust_ss + sw_cld_oml + sw_cld_omi
        + sw.vmr_ref.detach().cpu().reshape(-1).tolist(),
        dtype=torch.float32,
        device=device,
    ).contiguous()

    tbl_i_sw = torch.tensor(
        sw.gpt_to_bnd.detach().cpu().tolist()
        + sw.is_vis_gpt.detach().cpu().to(torch.int32).tolist()
        + sw.is_par_gpt.detach().cpu().to(torch.int32).tolist()
        + sw.key_species.detach().cpu().reshape(-1).tolist()
        + off_l_sw
        + off_u_sw
        + ent_sw,
        dtype=torch.int32,
        device=device,
    ).contiguous()

    packed = {
        "kmajor_lw": lw.kmajor.contiguous(),
        "plank_lw": lw.plank_fraction.contiguous(),
        "kminor_lw": kminor_lw,
        "tbl_f_lw": tbl_f_lw,
        "tbl_i_lw": tbl_i_lw,
        "lw_dims": (lw.n_temp, lw.n_press, lw.n_eta, int(kminor_lw.shape[2])),
        "kmajor_sw": sw.kmajor.contiguous(),
        "kminor_sw": kminor_sw,
        "rayl_sw": rayl_sw,
        "tbl_f_sw": tbl_f_sw,
        "tbl_i_sw": tbl_i_sw,
        "sw_dims": (sw.n_temp, sw.n_press, sw.n_eta, int(kminor_sw.shape[2])),
    }
    _ESM_TABLE_CACHE[key] = packed
    return packed


def run_rad_rrtmgp_metal(
    lib: Any,
    tabs: Any,
    pres_c: torch.Tensor,
    temp_c: torch.Tensor,
    temp_w: torch.Tensor,
    dz_c: torch.Tensor,
    rho_d_c: torch.Tensor,
    qv_c: torch.Tensor,
    qc_c: torch.Tensor,
    qi_c: torch.Tensor,
    t_sfc: torch.Tensor,
    albedo_vis: torch.Tensor,
    albedo_nir: torch.Tensor,
    mu0: torch.Tensor,
    s_toa: torch.Tensor,
    co2_ppmv: float,
    ch4_ppmv: float,
    n2o_ppmv: float,
    emiss_sfc: float | torch.Tensor,
    re_liq_um: float | torch.Tensor,
    re_ice_um: float | torch.Tensor,
    cloud_sw_inhomog: float,
    albedo_vis_dif: torch.Tensor | None,
    albedo_nir_dif: torch.Tensor | None,
    q_dust_c: torch.Tensor | None,
    o3_vmr_c: torch.Tensor | None,
    o3_lw_coupling: bool,
    aero_opt: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute fused Metal/CUDA RRTMGP (128 LW + 112 SW g-points) kernels."""
    nz = pres_c.shape[0]
    col_shape = t_sfc.shape
    M = t_sfc.numel()
    dev = pres_c.device

    pk = _pack_rrtmgp_gpu_tables(tabs, dev)
    dummy = t_sfc.reshape(-1).contiguous()
    dz_2d = dz_c.reshape(nz, -1).contiguous()

    flags = 0
    if q_dust_c is not None:
        flags |= 1
        dust_buf = q_dust_c.reshape(nz, M).contiguous()
    else:
        dust_buf = dummy
    if o3_vmr_c is not None:
        flags |= 2
        o3_buf = o3_vmr_c.reshape(nz, M).contiguous()
    else:
        o3_buf = dummy
    if o3_lw_coupling:
        flags |= 4
    if isinstance(re_liq_um, torch.Tensor):
        flags |= 8
        re_l_buf = re_liq_um.reshape(nz, M).contiguous()
        re_l_sc = 10.0
    else:
        re_l_buf = dummy
        re_l_sc = max(min(float(re_liq_um), 40.0), 3.0)
    if isinstance(re_ice_um, torch.Tensor):
        flags |= 16
        re_i_buf = re_ice_um.reshape(nz, M).contiguous()
        re_i_sc = 48.0
    else:
        re_i_buf = dummy
        re_i_sc = max(min(float(re_ice_um), 150.0), 10.0)
    if aero_opt is not None:
        flags |= 32
        a_tsw = aero_opt[0].reshape(nz, M).contiguous()
        a_ssw = aero_opt[1].reshape(nz, M).contiguous()
        a_asw = aero_opt[2].reshape(nz, M).contiguous()
        a_tlw = aero_opt[3].reshape(nz, M).contiguous()
    else:
        a_tsw = a_ssw = a_asw = a_tlw = dummy

    vis_dif = albedo_vis_dif if albedo_vis_dif is not None else albedo_vis
    nir_dif = albedo_nir_dif if albedo_nir_dif is not None else albedo_nir
    emiss_vec = (
        emiss_sfc.reshape(M).contiguous()
        if isinstance(emiss_sfc, torch.Tensor)
        else torch.full((M,), float(emiss_sfc), dtype=torch.float32, device=dev)
    )
    sfc_in = torch.stack(
        [
            t_sfc.reshape(M),
            albedo_vis.reshape(M),
            albedo_nir.reshape(M),
            vis_dif.reshape(M),
            nir_dif.reshape(M),
            mu0.reshape(M),
            s_toa.reshape(M),
            emiss_vec,
        ],
        dim=0,
    ).contiguous()

    pc_f = pres_c.reshape(nz, M).contiguous()
    tc_f = temp_c.reshape(nz, M).contiguous()
    tw_f = temp_w.reshape(nz + 1, M).contiguous()
    rd_f = rho_d_c.reshape(nz, M).contiguous()
    qv_f = qv_c.reshape(nz, M).contiguous()
    qc_f = qc_c.reshape(nz, M).contiguous()
    qi_f = qi_c.reshape(nz, M).contiguous()

    lw_up = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    lw_dn = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    q_lw = torch.empty((nz, M), dtype=torch.float32, device=dev)
    sw_up = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    sw_dn = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    q_sw = torch.empty((nz, M), dtype=torch.float32, device=dev)
    sfc_out = torch.empty((5, M), dtype=torch.float32, device=dev)

    u_p0 = [int(nz), int(M), int(dz_2d.shape[1]), int(flags)]
    f_p0 = [
        max(float(co2_ppmv), 0.0) * 1.0e-6,
        max(float(ch4_ppmv), 0.0) * 1.0e-6,
        max(float(n2o_ppmv), 0.0) * 1.0e-6,
        float(C.O2_MIXING_RATIO),
    ]
    f_p1 = [float(re_l_sc), float(re_i_sc), 1.0e-7, 0.78084]

    lib.rad_rrtmgp_lw_grp(
        pc_f, tc_f, tw_f, dz_2d, rd_f, qv_f, qc_f, qi_f,
        o3_buf, dust_buf, re_l_buf, re_i_buf, a_tlw, sfc_in,
        lw_up, lw_dn, q_lw,
        pk["kmajor_lw"], pk["plank_lw"], pk["kminor_lw"],
        pk["tbl_f_lw"], pk["tbl_i_lw"],
        u_p0, list(pk["lw_dims"]), f_p0, f_p1,
        threads=(M * 128,),
        group_size=(128,),
    )
    lib.rad_rrtmgp_sw_grp(
        pc_f, tc_f, dz_2d, rd_f, qv_f, qc_f, qi_f,
        o3_buf, dust_buf, re_l_buf, re_i_buf, a_tsw, a_ssw, a_asw, sfc_in,
        sw_up, sw_dn, q_sw, sfc_out,
        pk["kmajor_sw"], pk["kminor_sw"], pk["rayl_sw"],
        pk["tbl_f_sw"], pk["tbl_i_sw"],
        u_p0, list(pk["sw_dims"]), f_p0, f_p1, [float(cloud_sw_inhomog), 0.0, 0.0, 0.0],
        threads=(M * 128,),
        group_size=(128,),
    )
    return (
        sw_up.reshape((nz + 1,) + col_shape),
        sw_dn.reshape((nz + 1,) + col_shape),
        lw_up.reshape((nz + 1,) + col_shape),
        lw_dn.reshape((nz + 1,) + col_shape),
        q_sw.reshape((nz,) + col_shape),
        q_lw.reshape((nz,) + col_shape),
        sfc_out.reshape((5,) + col_shape),
    )


def _pack_ckd_hd_table(device: torch.device) -> torch.Tensor:
    """Pack 111-g-point CKD-HD (72 LW + 39 SW) constants + Chou scattering into 1631 floats."""
    key = ("ckd_hd_tbl", str(device))
    cached = _ESM_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    from aoba.radiation_ckd import (
        C2_PLANCK_CM_K,
        LW_BAND_EDGES_CM,
        LW_C_FOR,
        LW_C_SELF,
        LW_CLD_ICE_A,
        LW_CLD_ICE_B,
        LW_CLD_LIQ_A,
        LW_CLD_LIQ_B,
        LW_DIFFUSIVITY,
        LW_DUST_K_ABS,
        M_AIR_G,
        M_CH4_G,
        M_CO2_G,
        M_N2O_G,
        M_O2_G,
        M_O3_G,
        SW_DUST_ASYM,
        SW_DUST_KEXT,
        SW_DUST_SSA,
        SW_ICE_A,
        SW_ICE_B,
        SW_ICE_C,
        SW_ICE_D,
        SW_ICE_E,
        SW_ICE_F,
        SW_IS_VIS_BAND,
        SW_LIQ_A,
        SW_LIQ_B,
        SW_LIQ_C,
        SW_LIQ_D,
        SW_LIQ_E,
        SW_LIQ_F,
        SW_RAY_TAU0,
        SW_SOLAR_FRAC,
        SW1_PAR_FRAC,
    )
    from aoba.radiation_hd import LW_GPTS_HD, SW_GPTS_HD

    lw_scat = (0.97, 0.96, 0.95, 0.95, 0.94, 0.91, 0.91, 0.92, 0.94, 0.95)
    aero_sw_sc = (1.35, 1.00, 0.72, 0.48, 0.30, 0.18)

    # Sort LW_GPTS_HD by band index (preserve exact ordering within each band)
    lw_sorted = sorted(LW_GPTS_HD, key=lambda r: int(r[0]))
    lw_starts = [0]
    for b in range(10):
        cnt = sum(1 for r in lw_sorted if int(r[0]) == b)
        lw_starts.append(lw_starts[-1] + cnt)

    vals: list[float] = []
    for nu1, nu2 in LW_BAND_EDGES_CM:
        vals.extend([float(nu1), float(nu2)])
    vals.extend(float(x) for x in LW_C_SELF)
    vals.extend(float(x) for x in LW_C_FOR)
    vals.extend(float(LW_CLD_LIQ_A[b] * lw_scat[b]) for b in range(10))
    vals.extend(float(LW_CLD_LIQ_B[b] * lw_scat[b]) for b in range(10))
    vals.extend(float(LW_CLD_ICE_A[b] * lw_scat[b]) for b in range(10))
    vals.extend(float(LW_CLD_ICE_B[b] * lw_scat[b]) for b in range(10))
    vals.extend(float(x) for x in LW_DUST_K_ABS)
    assert len(vals) == 90
    for row in lw_sorted:
        vals.extend(float(x) for x in row)
    assert len(vals) == 1098
    vals.extend(float(x) for x in SW_SOLAR_FRAC)
    vals.extend(float(x) for x in SW_RAY_TAU0)
    vals.extend(float(x) for x in SW_IS_VIS_BAND)
    for arr in (SW_LIQ_A, SW_LIQ_B, SW_LIQ_C, SW_LIQ_D, SW_LIQ_E, SW_LIQ_F,
                SW_ICE_A, SW_ICE_B, SW_ICE_C, SW_ICE_D, SW_ICE_E, SW_ICE_F,
                SW_DUST_KEXT, SW_DUST_SSA, SW_DUST_ASYM, aero_sw_sc):
        vals.extend(float(x) for x in arr)
    assert len(vals) == 1212
    for row in SW_GPTS_HD:
        vals.extend(float(x) for x in row)
    assert len(vals) == 1602
    vals.extend([
        float(C.STEFAN_BOLTZMANN), float(C.O3_G1), float(C.O3_G2), float(C.O3_G3),
        float(C.CP_D), float(C.CP_V), float(C.EPS_WATER), float(C.P_REF),
        float(C.GRAVITY), float(C.O2_MIXING_RATIO), float(C2_PLANCK_CM_K),
        float(LW_DIFFUSIVITY), float(SW1_PAR_FRAC),
        float(M_CO2_G / M_AIR_G), float(M_O3_G / M_AIR_G), float(M_CH4_G / M_AIR_G),
        float(M_N2O_G / M_AIR_G), float(M_O2_G / M_AIR_G),
    ])
    assert len(vals) == 1620
    vals.extend(float(x) for x in lw_starts)
    assert len(vals) == 1631

    t = torch.tensor(vals, dtype=torch.float32, device=device).contiguous()
    _ESM_TABLE_CACHE[key] = t
    return t


def run_rad_ckd_hd_metal(
    lib: Any,
    pres_c: torch.Tensor,
    pres_w: torch.Tensor,
    temp_c: torch.Tensor,
    temp_w: torch.Tensor,
    dz_c: torch.Tensor,
    rho_d_c: torch.Tensor,
    qv_c: torch.Tensor,
    qc_c: torch.Tensor,
    qi_c: torch.Tensor,
    t_sfc: torch.Tensor,
    albedo_vis: torch.Tensor,
    albedo_nir: torch.Tensor,
    mu0: torch.Tensor,
    s_toa: torch.Tensor,
    co2_ppmv: float,
    ch4_ppmv: float,
    n2o_ppmv: float,
    emiss_sfc: float | torch.Tensor,
    re_liq_um: float | torch.Tensor,
    re_ice_um: float | torch.Tensor,
    cloud_sw_inhomog: float,
    albedo_vis_dif: torch.Tensor | None,
    albedo_nir_dif: torch.Tensor | None,
    q_dust_c: torch.Tensor | None,
    o3_vmr_c: torch.Tensor | None,
    o3_lw_coupling: bool,
    aero_opt: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute fused Metal/CUDA 111-g-point CKD-HD radiation kernel."""
    nz = pres_c.shape[0]
    col_shape = t_sfc.shape
    M = t_sfc.numel()
    dev = pres_c.device

    tbl_buf = _pack_ckd_hd_table(dev)
    dummy = t_sfc.reshape(-1).contiguous()
    dz_2d = dz_c.reshape(nz, -1).contiguous()

    flags = 0
    if q_dust_c is not None:
        flags |= 1
        dust_buf = q_dust_c.reshape(nz, M).contiguous()
    else:
        dust_buf = dummy
    if o3_vmr_c is not None:
        flags |= 2
        o3_buf = o3_vmr_c.reshape(nz, M).contiguous()
    else:
        o3_buf = dummy
    if o3_lw_coupling:
        flags |= 4
    if isinstance(re_liq_um, torch.Tensor):
        flags |= 8
        re_l_buf = re_liq_um.reshape(nz, M).contiguous()
        re_l_sc = 10.0
    else:
        re_l_buf = dummy
        re_l_sc = float(re_liq_um)
    if isinstance(re_ice_um, torch.Tensor):
        flags |= 16
        re_i_buf = re_ice_um.reshape(nz, M).contiguous()
        re_i_sc = 48.0
    else:
        re_i_buf = dummy
        re_i_sc = float(re_ice_um)
    if aero_opt is not None:
        flags |= 32
        a_tsw = aero_opt[0].reshape(nz, M).contiguous()
        a_ssw = aero_opt[1].reshape(nz, M).contiguous()
        a_asw = aero_opt[2].reshape(nz, M).contiguous()
        a_tlw = aero_opt[3].reshape(nz, M).contiguous()
    else:
        a_tsw = a_ssw = a_asw = a_tlw = dummy

    vis_dif = albedo_vis_dif if albedo_vis_dif is not None else albedo_vis
    nir_dif = albedo_nir_dif if albedo_nir_dif is not None else albedo_nir
    emiss_vec = (
        emiss_sfc.reshape(M).contiguous()
        if isinstance(emiss_sfc, torch.Tensor)
        else torch.full((M,), float(emiss_sfc), dtype=torch.float32, device=dev)
    )
    sfc_in = torch.stack(
        [
            t_sfc.reshape(M),
            albedo_vis.reshape(M),
            albedo_nir.reshape(M),
            vis_dif.reshape(M),
            nir_dif.reshape(M),
            mu0.reshape(M),
            s_toa.reshape(M),
            emiss_vec,
        ],
        dim=0,
    ).contiguous()

    sw_up = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    sw_dn = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    lw_up = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    lw_dn = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    q_rad_3d = torch.empty((3, nz, M), dtype=torch.float32, device=dev)
    sfc_out = torch.empty((5, M), dtype=torch.float32, device=dev)

    lib.rad_ckd_hd_col(
        pres_c.reshape(nz, M).contiguous(),
        pres_w.reshape(nz + 1, M).contiguous(),
        temp_c.reshape(nz, M).contiguous(),
        temp_w.reshape(nz + 1, M).contiguous(),
        dz_2d,
        rho_d_c.reshape(nz, M).contiguous(),
        qv_c.reshape(nz, M).contiguous(),
        qc_c.reshape(nz, M).contiguous(),
        qi_c.reshape(nz, M).contiguous(),
        dust_buf,
        o3_buf,
        re_l_buf,
        re_i_buf,
        a_tsw,
        a_ssw,
        a_asw,
        a_tlw,
        sfc_in,
        sw_up,
        sw_dn,
        lw_up,
        lw_dn,
        q_rad_3d,
        sfc_out,
        tbl_buf,
        [int(nz), int(M), int(dz_2d.shape[1]), int(flags)],
        [float(co2_ppmv), float(ch4_ppmv), float(n2o_ppmv), float(cloud_sw_inhomog)],
        [re_l_sc, re_i_sc, 0.0, 0.0],
        threads=M,
    )
    return (
        sw_up.reshape((nz + 1,) + col_shape),
        sw_dn.reshape((nz + 1,) + col_shape),
        lw_up.reshape((nz + 1,) + col_shape),
        lw_dn.reshape((nz + 1,) + col_shape),
        q_rad_3d.reshape((3, nz) + col_shape),
        sfc_out.reshape((5,) + col_shape),
    )


def run_microphysics_6class_metal(
    lib: Any,
    rho_d: torch.Tensor,
    theta_m: torch.Tensor,
    q_v: torch.Tensor,
    q_c: torch.Tensor,
    q_i: torch.Tensor,
    q_r: torch.Tensor,
    q_s: torch.Tensor,
    q_g: torch.Tensor,
    n_c: torch.Tensor,
    n_i: torch.Tensor,
    n_r: torch.Tensor,
    n_s: torch.Tensor,
    n_g: torch.Tensor,
    dz_c: torch.Tensor,
    dt: float,
    *,
    is_ocean_sub: torch.Tensor | None = None,
    u_sfc_sub: torch.Tensor | None = None,
    q_dust_sub: torch.Tensor | None = None,
    ccn_aero_sub: torch.Tensor | None = None,
    ccn_ocean_cm3: float = 45.0,
    ccn_land_cm3: float = 160.0,
    ccn_dust_scale_cm3: float = 20.0,
    inp_dust_enhance: float = 2.0,
    kk00_auto_coef: float = 13500.0,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """Execute single-pass top-down fused 6-class 2-moment microphysics kernel."""
    nz, nc, ny, nx = rho_d.shape
    sub_per_cell = ny * nx
    M = nc * sub_per_cell
    dev = rho_d.device

    th_o = theta_m.reshape(nz, M).contiguous().clone()
    qv_o = q_v.reshape(nz, M).contiguous().clone()
    qc_o = q_c.reshape(nz, M).contiguous().clone()
    qi_o = q_i.reshape(nz, M).contiguous().clone()
    qr_o = q_r.reshape(nz, M).contiguous().clone()
    qs_o = q_s.reshape(nz, M).contiguous().clone()
    qg_o = q_g.reshape(nz, M).contiguous().clone()
    nc_o = n_c.reshape(nz, M).contiguous().clone()
    ni_o = n_i.reshape(nz, M).contiguous().clone()
    nr_o = n_r.reshape(nz, M).contiguous().clone()
    ns_o = n_s.reshape(nz, M).contiguous().clone()
    ng_o = n_g.reshape(nz, M).contiguous().clone()

    rd_f = rho_d.reshape(nz, M).contiguous()
    dummy = rd_f[0]
    flags = 0
    if is_ocean_sub is not None:
        flags |= 1
        ocn_b = is_ocean_sub.to(dtype=torch.float32).reshape(M).contiguous()
    else:
        ocn_b = dummy
    if u_sfc_sub is not None:
        flags |= 2
        usfc_b = u_sfc_sub.reshape(M).contiguous()
    else:
        usfc_b = dummy
    if q_dust_sub is not None:
        flags |= 4
        dust_b = q_dust_sub.reshape(nz, M).contiguous()
    else:
        dust_b = dummy
    if ccn_aero_sub is not None:
        flags |= 8
        ccn_b = ccn_aero_sub.reshape(M).contiguous()
    else:
        ccn_b = dummy
    if dz_c.ndim == 2:
        flags |= 16
        dz_b = dz_c.reshape(nz, nc).contiguous()
    else:
        dz_b = dz_c.reshape(nz, 1).contiguous()

    pr_rain = torch.empty((M,), dtype=torch.float32, device=dev)
    pr_snow = torch.empty((M,), dtype=torch.float32, device=dev)

    lib.microphysics_6class_col(
        th_o, qv_o, qc_o, qi_o, qr_o, qs_o, qg_o,
        nc_o, ni_o, nr_o, ns_o, ng_o,
        rd_f, dz_b, ocn_b, usfc_b, dust_b, ccn_b,
        pr_rain, pr_snow,
        [int(nz), int(M), int(sub_per_cell), int(flags)],
        [float(dt), float(C.R_D), float(C.CV_D), float(C.P_REF)],
        [float(C.RV_OVER_RD), float(C.T_FREEZE), float(C.L_F0), float(C.L_V0)],
        [float(C.CP_D), float(C.EPS_WATER), float(ccn_ocean_cm3), float(ccn_land_cm3)],
        [float(ccn_dust_scale_cm3), float(inp_dust_enhance), float(kk00_auto_coef), 0.0],
        threads=M,
    )
    sh4 = (nz, nc, ny, nx)
    sh3 = (nc, ny, nx)
    return (
        th_o.reshape(sh4), qv_o.reshape(sh4), qc_o.reshape(sh4), qi_o.reshape(sh4),
        qr_o.reshape(sh4), qs_o.reshape(sh4), qg_o.reshape(sh4),
        nc_o.reshape(sh4), ni_o.reshape(sh4), nr_o.reshape(sh4), ns_o.reshape(sh4), ng_o.reshape(sh4),
        pr_rain.reshape(sh3), pr_snow.reshape(sh3),
    )


def run_ozone_chem_metal(
    lib: Any,
    o3_vmr: torch.Tensor,
    pres_c: torch.Tensor,
    temp_c: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    q_v: torch.Tensor,
    mu0: torch.Tensor,
    surf_type: torch.Tensor,
    lai_cell: torch.Tensor,
    f_snow_cell: torch.Tensor,
    dt: float,
    *,
    n2o_ppmv: float,
    ch4_ppmv: float,
    ecl_ppbv: float,
    lnox_cell_kgn_m2s: torch.Tensor | None,
    fire_c_cell_kgc_m2s: torch.Tensor | None,
    surf_land_id: int,
    surf_ocean_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute fused 3D prognostic ozone photochemistry, dry deposition, and PBL mixing kernel."""
    nz, nc = pres_c.shape
    dev = pres_c.device

    o3_out = o3_vmr.contiguous() if o3_vmr.is_contiguous() else o3_vmr.contiguous()
    dummy = mu0.reshape(nc).contiguous()
    flags = 0
    if lnox_cell_kgn_m2s is not None:
        flags |= 1
        lnox_b = lnox_cell_kgn_m2s.reshape(nc).contiguous()
    else:
        lnox_b = dummy
    if fire_c_cell_kgc_m2s is not None:
        flags |= 2
        fire_b = fire_c_cell_kgc_m2s.reshape(nc).contiguous()
    else:
        fire_b = dummy

    diag_out = torch.empty((4, nc), dtype=torch.float32, device=dev)
    lib.ozone_chem_col(
        o3_out,
        pres_c.contiguous(),
        temp_c.contiguous(),
        rho_d.contiguous(),
        dz_c.contiguous(),
        q_v.contiguous(),
        mu0.reshape(nc).contiguous(),
        surf_type.to(dtype=torch.int32).reshape(nc).contiguous(),
        lai_cell.reshape(nc).contiguous(),
        f_snow_cell.reshape(nc).contiguous(),
        lnox_b,
        fire_b,
        diag_out,
        [int(nz), int(nc), int(flags), int(surf_land_id)],
        [float(dt), float(n2o_ppmv), float(ch4_ppmv), float(ecl_ppbv)],
        [float(C.GRAVITY), float(surf_ocean_id), 0.0, 0.0],
        threads=nc,
    )
    return o3_out, diag_out[0], diag_out[1], diag_out[2], diag_out[3]


def run_dynamic_lake_step_metal(
    lib: Any,
    lake: Any,
    u_1: torch.Tensor,
    v_1: torch.Tensor,
    temp_1: torch.Tensor,
    qv_1: torch.Tensor,
    pres_sfc: torch.Tensor,
    rho_1: torch.Tensor,
    z_ref_rel: torch.Tensor,
    sw1_dir_dn: torch.Tensor,
    sw1_dif_dn: torch.Tensor,
    sw2_dir_dn: torch.Tensor,
    sw2_dif_dn: torch.Tensor,
    lw_dn: torch.Tensor,
    precip_rain: torch.Tensor,
    precip_snow: torch.Tensor,
    mu0_cell: torch.Tensor,
    area_cell: torch.Tensor,
    f_cor_cell: torch.Tensor,
    dt: float,
    *,
    k_ext_m1: float = 0.5,
) -> None:
    """Execute single-launch fused Metal/CUDA 2-layer dynamic lake + ice + Kahan volume kernel."""
    nc = int(temp_1.numel())
    dev = temp_1.device
    geom_in = torch.stack([lake.f_cap, lake.h_cap, lake.v_cap_m3], dim=0).contiguous()
    force_in = torch.stack(
        [
            u_1, v_1, temp_1, qv_1, pres_sfc, rho_1, z_ref_rel,
            sw1_dir_dn, sw1_dif_dn, sw2_dir_dn, sw2_dif_dn, lw_dn,
            precip_rain, precip_snow, mu0_cell, area_cell, f_cor_cell,
        ],
        dim=0,
    ).contiguous()
    state_io = torch.stack(
        [
            lake.v_lake_m3, lake.v_lake_comp, lake.t_ml, lake.t_bot,
            lake.h_ml, lake.h_ice, lake.swe_ice, lake.t_ice,
        ],
        dim=0,
    ).contiguous()
    diag_out = torch.empty((13, nc), dtype=torch.float32, device=dev)

    lib.dynamic_lake_step_col(
        geom_in,
        force_in,
        state_io,
        diag_out,
        [int(nc), 0, 0, 0],
        [float(dt), float(k_ext_m1), float(C.T_FREEZE), float(C.STEFAN_BOLTZMANN)],
        [float(C.CP_D), float(C.L_V0), float(C.L_S0), float(C.L_F0)],
        [float(C.GRAVITY), float(C.VON_KARMAN), float(C.EPS_WATER), float(C.R_V)],
        threads=nc,
    )

    lake.v_lake_m3 = state_io[0].contiguous()
    lake.v_lake_comp = state_io[1].contiguous()
    lake.t_ml = state_io[2].contiguous()
    lake.t_bot = state_io[3].contiguous()
    lake.h_ml = state_io[4].contiguous()
    lake.h_ice = state_io[5].contiguous()
    lake.swe_ice = state_io[6].contiguous()
    lake.t_ice = state_io[7].contiguous()

    lake.f_lake = diag_out[0].contiguous()
    lake.h_lake = diag_out[1].contiguous()
    lake.f_ice = diag_out[2].contiguous()
    lake.t_rad = diag_out[3].contiguous()
    lake.alb_vis_dir = diag_out[4].contiguous()
    lake.alb_vis_dif = diag_out[5].contiguous()
    lake.alb_nir_dir = diag_out[6].contiguous()
    lake.alb_nir_dif = diag_out[7].contiguous()
    lake.shf = diag_out[8].contiguous()
    lake.lhf = diag_out[9].contiguous()
    lake.evap = diag_out[10].contiguous()
    lake.tau_x = diag_out[11].contiguous()
    lake.tau_y = diag_out[12].contiguous()


def run_explicit_lightning_poisson_metal(
    lib: Any,
    rhs_phys: torch.Tensor,
    dz_c: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Solve 2D/3D periodic-horizontal Dirichlet-vertical Poisson via rfft2 + fused Thomas + irfft2."""
    nz, nc, ny, nx = rhs_phys.shape
    dev = rhs_phys.device
    rhs_hat = torch.fft.rfft2(rhs_phys, dim=(-2, -1))
    n_kx = int(rhs_hat.shape[-1])
    rhs_ri = torch.view_as_real(rhs_hat).contiguous()
    phi_ri = torch.empty_like(rhs_ri)

    if dz_c.ndim == 2:
        dz_b = dz_c.to(device=dev, dtype=torch.float32).contiguous()
        z_cols = int(dz_b.shape[1])
    else:
        dz_b = dz_c.to(device=dev, dtype=torch.float32).view(nz, 1).contiguous()
        z_cols = 1

    m_tot = nc * ny * n_kx
    lib.explicit_lightning_poisson_thomas(
        rhs_ri,
        phi_ri,
        dz_b,
        [int(nz), int(nc), int(ny), int(nx)],
        [int(n_kx), int(z_cols), 0, 0],
        [float(dx), float(dy), 0.0, 0.0],
        threads=m_tot,
    )
    phi_hat = torch.view_as_complex(phi_ri)
    return torch.fft.irfft2(phi_hat, s=(ny, nx), dim=(-2, -1)).to(torch.float32).contiguous()


def run_explicit_bulk_lightning_metal(
    lib: Any,
    q_chg: torch.Tensor,
    rho_d: torch.Tensor,
    theta_m: torch.Tensor,
    q_v: torch.Tensor,
    q_c: torch.Tensor,
    q_i: torch.Tensor,
    q_p: torch.Tensor,
    w_stag: torch.Tensor,
    z_c: torch.Tensor,
    dz_c: torch.Tensor,
    *,
    dx: float,
    dy: float,
    dt: float,
    u_stag: torch.Tensor | None = None,
    v_stag: torch.Tensor | None = None,
    q_r_prog: torch.Tensor | None = None,
    q_s_prog: torch.Tensor | None = None,
    q_g_prog: torch.Tensor | None = None,
    n_i_prog: torch.Tensor | None = None,
    n_s_prog: torch.Tensor | None = None,
    n_g_prog: torch.Tensor | None = None,
    rho_rain: float,
    rho_snow: float,
    rho_graupel: float,
    t0_precip: float,
    t00_precip: float,
    t0_graupel: float,
    t00_graupel: float,
    eps0: float,
    e_crit_sl: float,
    extent_frac: float,
    neut_frac: float,
    q_per_flash: float,
    z_cg_max: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Execute 3-kernel fused Metal/CUDA SCALE-LT explicit bulk lightning & discharge step."""
    nz, nc, ny, nx = rho_d.shape
    dev = rho_d.device
    m_sub = nc * ny * nx

    q_in = q_chg.contiguous()
    q_out = torch.empty_like(q_in)
    rhs_phys = torch.empty((nz, nc, ny, nx), dtype=torch.float32, device=dev)
    ccd_km = torch.empty((nc, ny, nx), dtype=torch.float32, device=dev)

    rho_b = rho_d.contiguous()
    th_b = theta_m.contiguous()
    qv_b = q_v.contiguous()
    qc_b = q_c.contiguous()
    qi_b = q_i.contiguous()
    qp_b = q_p.contiguous()
    w_b = w_stag.contiguous()

    flags = 0
    if q_g_prog is not None:
        flags |= 1
        qg_b = q_g_prog.contiguous()
        if q_r_prog is not None:
            flags |= 2
            qr_b = q_r_prog.contiguous()
        else:
            qr_b = qc_b
        if q_s_prog is not None:
            flags |= 4
            qs_b = q_s_prog.contiguous()
        else:
            qs_b = qc_b
    else:
        qr_b = qc_b
        qs_b = qc_b
        qg_b = qc_b

    if n_i_prog is not None:
        flags |= 8
        ni_b = n_i_prog.contiguous()
    else:
        ni_b = qc_b
    if n_s_prog is not None:
        flags |= 16
        ns_b = n_s_prog.contiguous()
    else:
        ns_b = qc_b
    if n_g_prog is not None:
        flags |= 32
        ng_b = n_g_prog.contiguous()
    else:
        ng_b = qc_b

    if u_stag is not None and nx > 1:
        flags |= 64
        u_b = u_stag.contiguous()
    else:
        u_b = qc_b
    if v_stag is not None and ny > 1:
        flags |= 128
        v_b = v_stag.contiguous()
    else:
        v_b = qc_b

    if z_c.ndim == 2:
        zc_b = z_c.to(device=dev, dtype=torch.float32).contiguous()
        dz_b = dz_c.to(device=dev, dtype=torch.float32).contiguous()
        z_cols = int(zc_b.shape[1])
    else:
        zc_b = z_c.to(device=dev, dtype=torch.float32).view(nz, 1).contiguous()
        dz_b = dz_c.to(device=dev, dtype=torch.float32).view(nz, 1).contiguous()
        z_cols = 1

    # Kernel 1: Fused horizontal advection + Takahashi charging + vertical implicit Thomas + melting + decay
    lib.explicit_lightning_charge_col(
        q_in,
        q_out,
        rhs_phys,
        ccd_km,
        rho_b,
        th_b,
        qv_b,
        qc_b,
        qi_b,
        qp_b,
        qr_b,
        qs_b,
        qg_b,
        ni_b,
        ns_b,
        ng_b,
        u_b,
        v_b,
        w_b,
        zc_b,
        dz_b,
        [int(nz), int(nc), int(ny), int(nx)],
        [int(flags), int(z_cols), 0, 0],
        [float(dx), float(dy), float(dt), float(C.R_D / C.CV_D)],
        [float(C.P_REF), float(C.R_D), float(C.RV_OVER_RD), float(C.T_FREEZE)],
        [float(rho_rain), float(rho_snow), float(rho_graupel), float(t0_precip)],
        [float(t00_precip), float(t0_graupel), float(t00_graupel), float(1.0 / eps0)],
        threads=m_sub,
    )

    # Kernel 2: Exact 2D/3D periodic-horizontal / Dirichlet-vertical Poisson (rfft2 + fused Thomas + irfft2)
    phi_e = run_explicit_lightning_poisson_metal(lib, rhs_phys, dz_b, dx, dy)

    # Kernel 3: E-field gradient + early-exit breakdown trigger + zero-sum IC/CG discharge + flash rates
    e_mag = torch.empty((nz, nc, ny, nx), dtype=torch.float32, device=dev)
    e_z = torch.empty((nz, nc, ny, nx), dtype=torch.float32, device=dev)
    flash_tot_sub = torch.empty((nc, ny, nx), dtype=torch.float32, device=dev)
    flash_cg_sub = torch.empty((nc, ny, nx), dtype=torch.float32, device=dev)
    f_cg_sub = torch.empty((nc, ny, nx), dtype=torch.float32, device=dev)

    lib.explicit_lightning_discharge_grp(
        q_out,
        phi_e,
        rho_b,
        zc_b,
        dz_b,
        e_mag,
        e_z,
        flash_tot_sub,
        flash_cg_sub,
        f_cg_sub,
        [int(nz), int(nc), int(ny), int(nx)],
        [int(z_cols), 0, 0, 0],
        [float(dx), float(dy), float(dt), float(e_crit_sl)],
        [float(extent_frac), float(neut_frac), float(q_per_flash), float(z_cg_max)],
        threads=(nc * 64,),
        group_size=(64,),
    )

    return q_out, phi_e, e_mag, e_z, flash_tot_sub, flash_cg_sub, f_cg_sub, ccd_km
