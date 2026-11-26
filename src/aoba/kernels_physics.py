"""Extra Metal shaders and wrappers for AOBA host/physics acceleration (SCVT FCT, Dust, GWD, Surface, Radiation)."""

from __future__ import annotations

from typing import Any

import torch

from aoba import constants as C


_SCVT_FCT_WS_CACHE: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}


def _cache_put(cache: dict[tuple[Any, ...], Any], key: tuple[Any, ...], val: Any, limit: int = 8) -> None:
    if len(cache) >= limit:
        cache.pop(next(iter(cache)))
    cache[key] = val


PHYSICS_METAL_SHADER_SRC: str = r"""
// =========================================================================
// 12. Batched SCVT Voronoi Zalesak FCT Advection Kernels
// =========================================================================

kernel void scvt_fct_edge(
    device const int* c0_arr   [[buffer(0)]],   // [N_edges]
    device const int* c1_arr   [[buffer(1)]],   // [N_edges]
    device const float* q      [[buffer(2)]],   // [B, nz, N_cells]
    device const float* uh     [[buffer(3)]],   // [nz, N_edges]
    device float* fl_h_out     [[buffer(4)]],   // [B, nz, N_edges]
    device float* ah_out       [[buffer(5)]],   // [B, nz, N_edges]
    constant ulong4& u_dim     [[buffer(6)]],   // [B, nz, N_cells, N_edges]
    uint gid [[thread_position_in_grid]])
{
    uint B = uint(u_dim.x);
    uint nz = uint(u_dim.y);
    uint nc = uint(u_dim.z);
    uint ne = uint(u_dim.w);
    uint slice_e = nz * ne;
    if (gid >= B * slice_e) return;

    uint b = gid / slice_e;
    uint loc = gid - b * slice_e;
    uint k = loc / ne;
    uint e = loc - k * ne;

    uint cbase = b * (nz * nc) + k * nc;
    float q0 = q[cbase + uint(c0_arr[e])];
    float q1 = q[cbase + uint(c1_arr[e])];
    float u = uh[loc];
    float fl = max(u, 0.0f) * q0 + min(u, 0.0f) * q1;
    float fh = u * (0.5f * (q0 + q1));
    fl_h_out[gid] = fl;
    ah_out[gid] = fh - fl;
}

kernel void scvt_fct_cell_td(
    device const int* eoc       [[buffer(0)]],   // [N_cells, 6]
    device const float* div_w   [[buffer(1)]],   // [N_cells, 6]
    device const float* q       [[buffer(2)]],   // [B, nz, N_cells]
    device const float* m_old   [[buffer(3)]],   // [nz, N_cells]
    device const float* m_new   [[buffer(4)]],   // [nz, N_cells]
    device const float* fl_h    [[buffer(5)]],   // [B, nz, N_edges]
    device const float* om_z    [[buffer(6)]],   // [nz+1, N_cells]
    device float* q_td          [[buffer(7)]],   // [B, nz, N_cells]
    device float* az_out        [[buffer(8)]],   // [B, nz+1, N_cells]
    constant ulong4& u_dim      [[buffer(9)]],   // [B, nz, N_cells, N_edges]
    constant float& dt          [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    // Column-wise implicit-vertical donor-cell predictor + anti-diffusive fluxes:
    // Strictly conservative, constant-preserving, and unconditionally monotone for any |omega_z|*dt/m.
    uint B = uint(u_dim.x);
    uint nz = uint(u_dim.y);
    uint nc = uint(u_dim.z);
    uint ne = uint(u_dim.w);
    if (gid >= B * nc || nz < 1u || nz > 128u) return;

    uint b = gid / nc;
    uint c = gid - b * nc;
    uint c6 = c * 6u;
    uint vol = nz * nc;
    uint q_b_base = b * vol;
    uint e_b_base = b * (nz * ne);
    uint az_base = b * (nz + 1u) * nc;

    int e_idx[6];
    float dw[6];
    for (uint d = 0u; d < 6u; ++d) {
        e_idx[d] = eoc[c6 + d];
        dw[d] = div_w[c6 + d];
    }

    float q_loc[128];
    float qt_loc[128];
    float c_star[128];
    float d_star[128];

    float c_prev = 0.0f;
    float d_prev = 0.0f;
    for (uint k = 0u; k < nz; ++k) {
        uint loc = k * nc + c;
        uint qidx = q_b_base + loc;
        uint ebase = e_b_base + k * ne;

        float div_fl_h = 0.0f;
        for (uint d = 0u; d < 6u; ++d) {
            div_fl_h += dw[d] * fl_h[ebase + uint(e_idx[d])];
        }

        float q0 = q[qidx];
        q_loc[k] = q0;
        float rhs_k = max(m_old[loc] * q0 - dt * div_fl_h, 0.0f);

        float w_dn = (k > 0u) ? om_z[k * nc + c] : 0.0f;
        float w_up = (k + 1u < nz) ? om_z[(k + 1u) * nc + c] : 0.0f;
        float a_k = (k > 0u) ? (-dt * max(w_dn, 0.0f)) : 0.0f;
        float c_k = (k + 1u < nz) ? (dt * min(w_up, 0.0f)) : 0.0f;
        float b_k = max(m_new[loc], 1.0e-4f) + dt * (max(w_up, 0.0f) - min(w_dn, 0.0f));

        float den = (k == 0u) ? b_k : (b_k - a_k * c_prev);
        float inv_den = 1.0f / max(den, 1.0e-20f);
        c_prev = (k + 1u < nz) ? (c_k * inv_den) : 0.0f;
        d_prev = (k == 0u) ? (rhs_k * inv_den) : ((rhs_k - a_k * d_prev) * inv_den);
        c_star[k] = c_prev;
        d_star[k] = d_prev;
    }

    float q_next = 0.0f;
    for (int ki = int(nz) - 1; ki >= 0; --ki) {
        uint k = uint(ki);
        float sol = (ki == int(nz) - 1) ? d_star[k] : (d_star[k] - c_star[k] * q_next);
        q_next = max(sol, 0.0f);
        qt_loc[k] = q_next;
        q_td[q_b_base + k * nc + c] = q_next;
    }

    az_out[az_base + c] = 0.0f;
    az_out[az_base + nz * nc + c] = 0.0f;
    for (uint m = 1u; m < nz; ++m) {
        float w_m = om_z[m * nc + c];
        float fl_impl = max(w_m, 0.0f) * qt_loc[m - 1u] + min(w_m, 0.0f) * qt_loc[m];
        float fh_up;
        if (nz >= 4u && m >= 2u && m <= nz - 2u) {
            fh_up = w_m * ((-q_loc[m - 2u] + 7.0f * q_loc[m - 1u] + 7.0f * q_loc[m] - q_loc[m + 1u]) * (1.0f / 12.0f));
        } else {
            fh_up = w_m * (0.5f * (q_loc[m - 1u] + q_loc[m]));
        }
        az_out[az_base + m * nc + c] = fh_up - fl_impl;
    }
}

kernel void scvt_fct_bounds_ratios(
    device const int* eoc       [[buffer(0)]],   // [N_cells, 6]
    device const int* nb_idx    [[buffer(1)]],   // [N_cells, 6]
    device const float* div_w   [[buffer(2)]],   // [N_cells, 6]
    device const float* q       [[buffer(3)]],   // [B, nz, N_cells]
    device const float* q_td    [[buffer(4)]],   // [B, nz, N_cells]
    device const float* m_new   [[buffer(5)]],   // [nz, N_cells]
    device const float* ah      [[buffer(6)]],   // [B, nz, N_edges]
    device const float* az      [[buffer(7)]],   // [B, nz+1, N_cells]
    device float* r_plus        [[buffer(8)]],   // [B, nz, N_cells]
    device float* r_minus       [[buffer(9)]],   // [B, nz, N_cells]
    constant ulong4& u_dim      [[buffer(10)]],  // [B, nz, N_cells, N_edges]
    constant float& dt          [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    uint B = uint(u_dim.x);
    uint nz = uint(u_dim.y);
    uint nc = uint(u_dim.z);
    uint ne = uint(u_dim.w);
    uint vol = nz * nc;
    if (gid >= B * vol) return;

    uint b = gid / vol;
    uint loc = gid - b * vol;
    uint k = loc / nc;
    uint c = loc - k * nc;
    uint c6 = c * 6u;
    uint qbase = b * vol + k * nc;
    uint ebase = b * (nz * ne) + k * ne;

    float q0 = q[gid];
    float qtd0 = q_td[gid];
    float q_max = max(q0, qtd0);
    float q_min = min(q0, qtd0);

    float p_plus_h = 0.0f;
    float p_minus_h = 0.0f;
    for (uint d = 0u; d < 6u; ++d) {
        uint nb = uint(nb_idx[c6 + d]);
        uint nidx = qbase + nb;
        float qn = q[nidx];
        float qtn = q_td[nidx];
        q_max = max(q_max, max(qn, qtn));
        q_min = min(q_min, min(qn, qtn));

        float oe = div_w[c6 + d] * ah[ebase + uint(eoc[c6 + d])];
        p_plus_h  += max(-oe, 0.0f);
        p_minus_h += max(oe, 0.0f);
    }

    if (k > 0u) {
        q_max = max(q_max, max(q[gid - nc], q_td[gid - nc]));
        q_min = min(q_min, min(q[gid - nc], q_td[gid - nc]));
    }
    if (k + 1u < nz) {
        q_max = max(q_max, max(q[gid + nc], q_td[gid + nc]));
        q_min = min(q_min, min(q[gid + nc], q_td[gid + nc]));
    }

    uint az_base = b * (nz + 1u) * nc;
    float az_dn = az[az_base + k * nc + c];
    float az_up = az[az_base + (k + 1u) * nc + c];

    float p_plus = dt * (p_plus_h + max(az_dn, 0.0f) - min(az_up, 0.0f));
    float p_minus = dt * (p_minus_h + max(az_up, 0.0f) - min(az_dn, 0.0f));

    float m_safe = max(m_new[loc], 1.0e-4f);
    float q_plus = m_safe * max(q_max - qtd0, 0.0f);
    float q_minus = m_safe * max(qtd0 - q_min, 0.0f);
    float eps = 1.0e-20f;

    r_plus[gid]  = (p_plus  > eps) ? clamp(q_plus  / (p_plus  + eps), 0.0f, 1.0f) : 1.0f;
    r_minus[gid] = (p_minus > eps) ? clamp(q_minus / (p_minus + eps), 0.0f, 1.0f) : 1.0f;
}

kernel void scvt_fct_limit_apply(
    device const int* c0_arr    [[buffer(0)]],   // [N_edges]
    device const int* c1_arr    [[buffer(1)]],   // [N_edges]
    device const int* eoc       [[buffer(2)]],   // [N_cells, 6]
    device const float* div_w   [[buffer(3)]],   // [N_cells, 6]
    device const float* q_td    [[buffer(4)]],   // [B, nz, N_cells]
    device const float* m_new   [[buffer(5)]],   // [nz, N_cells]
    device const float* ah      [[buffer(6)]],   // [B, nz, N_edges]
    device const float* az      [[buffer(7)]],   // [B, nz+1, N_cells]
    device const float* r_plus  [[buffer(8)]],   // [B, nz, N_cells]
    device const float* r_minus [[buffer(9)]],   // [B, nz, N_cells]
    device float* q_out         [[buffer(10)]],  // [B, nz, N_cells]
    constant ulong4& u_dim      [[buffer(11)]],  // [B, nz, N_cells, N_edges]
    constant float& dt          [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    uint B = uint(u_dim.x);
    uint nz = uint(u_dim.y);
    uint nc = uint(u_dim.z);
    uint ne = uint(u_dim.w);
    uint vol = nz * nc;
    if (gid >= B * vol) return;

    uint b = gid / vol;
    uint loc = gid - b * vol;
    uint k = loc / nc;
    uint c = loc - k * nc;
    uint c6 = c * 6u;
    uint qbase = b * vol + k * nc;
    uint ebase = b * (nz * ne) + k * ne;

    float div_h_anti = 0.0f;
    for (uint d = 0u; d < 6u; ++d) {
        float w = div_w[c6 + d];
        if (w != 0.0f) {
            uint eid = uint(eoc[c6 + d]);
            float a_e = ah[ebase + eid];
            uint c0 = uint(c0_arr[eid]);
            uint c1 = uint(c1_arr[eid]);
            float lim = (a_e >= 0.0f)
                ? min(r_minus[qbase + c0], r_plus[qbase + c1])
                : min(r_plus[qbase + c0], r_minus[qbase + c1]);
            div_h_anti += w * (lim * a_e);
        }
    }

    uint az_base = b * (nz + 1u) * nc;
    float rp0 = r_plus[gid];
    float rm0 = r_minus[gid];
    float az_dn_lim = 0.0f;
    float az_up_lim = 0.0f;
    if (k > 0u) {
        float az_dn = az[az_base + k * nc + c];
        float cz_dn = (az_dn >= 0.0f) ? min(r_minus[gid - nc], rp0) : min(r_plus[gid - nc], rm0);
        az_dn_lim = cz_dn * az_dn;
    }
    if (k + 1u < nz) {
        float az_up = az[az_base + (k + 1u) * nc + c];
        float cz_up = (az_up >= 0.0f) ? min(rm0, r_plus[gid + nc]) : min(rp0, r_minus[gid + nc]);
        az_up_lim = cz_up * az_up;
    }

    float m_safe = max(m_new[loc], 1.0e-4f);
    q_out[gid] = max(q_td[gid] - (dt / m_safe) * (div_h_anti + az_up_lim - az_dn_lim), 0.0f);
}

// =========================================================================
// 13. Column-Local Mineral Dust Mixing, Gravitational Settling & Washout
// =========================================================================

kernel void dust_column_col(
    device const float* q_dust_in   [[buffer(0)]],  // [nz, N_cells]
    device const float* rho_d       [[buffer(1)]],  // [nz, N_cells]
    device const float* dz_c        [[buffer(2)]],  // [nz, N_cells]
    device const float* z_c         [[buffer(3)]],  // [nz, N_cells]
    device const float* f_emit      [[buffer(4)]],  // [N_cells]
    device const float* precip      [[buffer(5)]],  // [N_cells]
    device const float* u_star      [[buffer(6)]],  // [N_cells]
    device float* q_out             [[buffer(7)]],  // [nz, N_cells]
    constant ulong4& u_dim        [[buffer(8)]],  // [nz, N_cells, dz_cols, zc_cols]
    constant float4& f_p0           [[buffer(9)]],  // [dt, v_dep, v_settle, alpha_scav]
    uint c [[thread_position_in_grid]])
{
    uint nz = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    uint dz_cols = uint(u_dim.z);
    uint zc_cols = uint(u_dim.w);
    if (c >= nc || nz == 0u || nz > 64u) return;
    uint col_dz = (dz_cols > 1u) ? c : 0u;
    uint col_zc = (zc_cols > 1u) ? c : 0u;

    float dt = f_p0.x;
    float v_dep = f_p0.y;
    float v_settle = f_p0.z;
    float alpha_scav = f_p0.w;

    float qc[64];
    float mk[64];
    float rd[64];
    float zc[64];

    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * nc + c;
        float r = rho_d[idx];
        rd[k] = r;
        mk[k] = max(r * dz_c[k * dz_cols + col_dz], 1.0e-3f);
        zc[k] = z_c[k * zc_cols + col_zc];
        qc[k] = max(q_dust_in[idx], 0.0f);
    }

    qc[0] += dt * max(f_emit[c], 0.0f) / mk[0];

    float z0_off = -zc[0] + 0.5f * dz_c[col_dz];
    float us = max(u_star[c], 0.05f);
    float h_pbl = 1200.0f;
    uint k_pbl = (nz > 1u) ? min(6u, nz - 1u) : 0u;
    for (uint k = 0u; k < k_pbl; ++k) {
        float zr0 = max(zc[k] + z0_off, 10.0f);
        float zr1 = max(zc[k + 1u] + z0_off, 10.0f);
        float z_int = 0.5f * (zr0 + zr1);
        float kz = 0.0f;
        if (z_int < h_pbl) {
            float fac = 1.0f - z_int / h_pbl;
            kz = 0.4f * us * z_int * (fac * fac);
        }
        float dz_int = max(zc[k + 1u] - zc[k], 10.0f);
        float rho_int = 0.5f * (rd[k] + rd[k + 1u]);
        float exch = (dt * rho_int * kz) / dz_int;
        float frac = exch / (1.0f + exch * (1.0f / mk[k] + 1.0f / mk[k + 1u]));
        float dq_flux = frac * (qc[k] - qc[k + 1u]);
        qc[k]      -= dq_flux / mk[k];
        qc[k + 1u] += dq_flux / mk[k + 1u];
    }

    float wash = exp(-alpha_scav * max(precip[c], 0.0f) * dt);
    float flux_above = 0.0f;
    for (int ki = int(nz) - 1; ki >= 0; --ki) {
        uint k = uint(ki);
        float v_fall = (k == 0u) ? v_dep : v_settle;
        float sink = dt * rd[k] * v_fall;
        float qk = (mk[k] * qc[k] + dt * flux_above) / (mk[k] + sink);
        flux_above = rd[k] * v_fall * qk;
        q_out[k * nc + c] = clamp(qk * wash, 0.0f, 1.0e-3f);
    }
}

// =========================================================================
// 14. Subgrid Orographic Drag (Lott & Miller, 1997) Column Kernel
// =========================================================================

kernel void oro_gwd_col(
    device const float* rho_d      [[buffer(0)]],   // [nz, N_cells]
    device const float* u_cell     [[buffer(1)]],   // [nz, N_cells]
    device const float* v_cell     [[buffer(2)]],   // [nz, N_cells]
    device const float* theta_m    [[buffer(3)]],   // [nz, N_cells]
    device const float* z_rel_c    [[buffer(4)]],   // [nz, N_cells]
    device const float* dz_c       [[buffer(5)]],   // [nz, N_cells]
    device const float* sigma_topo [[buffer(6)]],   // [N_cells]
    device const float* slope_topo [[buffer(7)]],   // [N_cells]
    device const float* aniso_topo [[buffer(8)]],   // [N_cells]
    device const float* angle_topo [[buffer(9)]],   // [N_cells]
    device float* du_out           [[buffer(10)]],  // [nz, N_cells]
    device float* dv_out           [[buffer(11)]],  // [nz, N_cells]
    constant ulong4& u_dim         [[buffer(12)]],  // [nz, N_cells, dz_cols, z_cols]
    constant float4& f_p0          [[buffer(13)]],  // [dt, gravity, g_wave, cd_block]
    constant float& hn_crit        [[buffer(14)]],
    uint c [[thread_position_in_grid]])
{
    uint nz = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    uint dz_cols = uint(u_dim.z);
    uint z_cols = uint(u_dim.w);
    if (c >= nc || nz < 2u) return;
    uint col_dz = (dz_cols > 1u) ? c : 0u;
    uint col_zc = (z_cols > 1u) ? c : 0u;

    float sig = sigma_topo[c];
    float slp = slope_topo[c];
    if (!(sig > 10.0f && slp > 1.0e-4f)) {
        for (uint k = 0u; k < nz; ++k) {
            uint idx = k * nc + c;
            du_out[idx] = 0.0f;
            dv_out[idx] = 0.0f;
        }
        return;
    }

    float dt = f_p0.x;
    float gravity = f_p0.y;
    float g_wave = f_p0.z;
    float cd_block = f_p0.w;

    float mu = max(sig, 1.0f);
    float slope = max(slp, 0.0f);
    float gamma = clamp(aniso_topo[c], 0.0f, 1.0f);
    float psi = angle_topo[c];
    float h_eff_peak = 2.0f * mu;
    float z_env_max = max(h_eff_peak, 250.0f);

    float w_sum = 0.0f;
    float u_low = 0.0f;
    float v_low = 0.0f;
    float n_low = 0.0f;
    float rho_low = 0.0f;

    float th_prev = theta_m[c];
    float z_prev = z_rel_c[col_zc];
    float n_int_prev = 0.0f;

    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * nc + c;
        float th_cur = theta_m[idx];
        float z_cur = z_rel_c[k * z_cols + col_zc];

        float n_int_cur = n_int_prev;
        if (k + 1u < nz) {
            uint idx_p1 = (k + 1u) * nc + c;
            float th_p1 = theta_m[idx_p1];
            float z_p1 = z_rel_c[(k + 1u) * z_cols + col_zc];
            float dz_i = max(z_p1 - z_cur, 10.0f);
            float th_i = max(0.5f * (th_p1 + th_cur), 200.0f);
            float n2 = clamp((gravity / th_i) * (th_p1 - th_cur) / dz_i, 2.5e-5f, 1.0e-3f);
            n_int_cur = sqrt(n2);
        }

        float nk = (k == 0u) ? n_int_cur : ((k + 1u == nz) ? n_int_prev : 0.5f * (n_int_prev + n_int_cur));
        if (k < 2u || z_cur <= z_env_max) {
            float w = rho_d[idx] * dz_c[k * dz_cols + col_dz];
            w_sum += w;
            u_low += w * u_cell[idx];
            v_low += w * v_cell[idx];
            n_low += w * nk;
            rho_low += w * rho_d[idx];
        }
        th_prev = th_cur;
        z_prev = z_cur;
        n_int_prev = n_int_cur;
    }

    w_sum = max(w_sum, 1.0f);
    u_low /= w_sum;
    v_low /= w_sum;
    n_low /= w_sum;
    rho_low /= w_sum;

    float speed_low = sqrt(u_low * u_low + v_low * v_low + 1.0e-4f);
    float ux_hat = u_low / speed_low;
    float uy_hat = v_low / speed_low;

    float phi = atan2(v_low, u_low + 1.0e-12f) - psi;
    float cphi = cos(phi);
    float sphi = sin(phi);
    float cos2 = cphi * cphi;
    float sin2 = sphi * sphi;
    float b_fac = 1.0f - 0.18f * gamma - 0.04f * gamma * gamma;
    float c_fac = 0.48f * gamma + 0.30f * gamma * gamma;
    float t1 = b_fac * cos2 + c_fac * sin2;
    float t2 = (b_fac - c_fac);
    float d_dir = clamp(sqrt(t1 * t1 + (t2 * t2) * sin2 * cos2), 0.05f, 1.5f);

    float hn = n_low * h_eff_peak / max(speed_low, 0.5f);
    float z_block = (hn > hn_crit) ? max(h_eff_peak * (1.0f - hn_crit / max(hn, hn_crit)), 0.0f) : 0.0f;

    float h_wave = max(h_eff_peak - z_block, 0.0f);
    float slp_2mu = slope / (2.0f * mu);
    float k_wave = g_wave * (slope / (4.0f * mu)) * d_dir;
    float tau_cur = rho_low * k_wave * n_low * speed_low * (h_wave * h_wave);
    float inv_dt = 1.0f / max(dt, 1.0f);
    float blk_pref = cd_block * slp_2mu * d_dir;

    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * nc + c;
        float uk = u_cell[idx];
        float vk = v_cell[idx];
        float rk = rho_d[idx];
        float zk = z_rel_c[k * z_cols + col_zc];
        float thk = theta_m[idx];

        float geom_wake = (zk < z_block) ? sqrt(max(z_block - zk, 0.0f) / (zk + mu)) : 0.0f;
        float speed_z = sqrt(uk * uk + vk * vk + 1.0e-6f);
        float r_blk = blk_pref * geom_wake * speed_z;
        float eff_blk = r_blk / (1.0f + dt * r_blk);
        float du_blk = -eff_blk * uk;
        float dv_blk = -eff_blk * vk;

        float tau_above = 0.0f;
        if (k + 1u < nz) {
            uint idx_p1 = (k + 1u) * nc + c;
            float uk1 = u_cell[idx_p1];
            float vk1 = v_cell[idx_p1];
            float rk1 = rho_d[idx_p1];
            float zk1 = z_rel_c[(k + 1u) * z_cols + col_zc];
            float thk1 = theta_m[idx_p1];

            float dz_i = max(zk1 - zk, 10.0f);
            float th_i = max(0.5f * (thk1 + thk), 200.0f);
            float n_int_k = sqrt(clamp((gravity / th_i) * (thk1 - thk) / dz_i, 2.5e-5f, 1.0e-3f));

            float zw_app = 0.5f * (zk + zk1);
            if (zw_app <= z_block) {
                tau_above = tau_cur;
            } else {
                float rw = 0.5f * (rk + rk1);
                float uw = 0.5f * (uk + uk1);
                float vw = 0.5f * (vk + vk1);
                float u_proj = max(uw * ux_hat + vw * uy_hat, 0.0f);
                float tau_sat = rw * k_wave * (u_proj * u_proj * u_proj) / (n_int_k * (hn_crit * hn_crit));
                tau_above = min(tau_cur, tau_sat);
            }
        }

        float dtau_dz = (tau_cur - tau_above) / (rk * dz_c[k * dz_cols + col_dz]);
        float u_proj_c = max(uk * ux_hat + vk * uy_hat, 0.0f);
        float decel = min(dtau_dz, 0.8f * u_proj_c * inv_dt);

        du_out[idx] = du_blk - decel * ux_hat;
        dv_out[idx] = dv_blk - decel * uy_hat;
        tau_cur = tau_above;
    }
}

// =========================================================================
// 15. Non-Orographic Spectral Gravity-Wave Propagation Kernel
// =========================================================================

kernel void ngwd_propagate_col(
    device const float* u_proj_b     [[buffer(0)]],   // [A, nz, N_cells]
    device const float* rho_d        [[buffer(1)]],   // [nz, N_cells]
    device const float* dz_c         [[buffer(2)]],   // [nz, N_cells]
    device const float* n_int        [[buffer(3)]],   // [nz-1, N_cells]
    device const float* tau_launch_b [[buffer(4)]],   // [A, M, N_cells]
    device const float* u_src_b      [[buffer(5)]],   // [A, N_cells]
    device const int* k_launch_b     [[buffer(6)]],   // [A, N_cells]
    device const float* effgw_b      [[buffer(7)]],   // [A]
    device const float* c_bins       [[buffer(8)]],   // [M]
    device float* du_out             [[buffer(9)]],   // [A, nz, N_cells]
    constant ulong4& u_dim           [[buffer(10)]],  // [A, nz, N_cells, M]
    constant float4& f_p0            [[buffer(11)]],  // [k_h, dt, tend_max, 0]
    uint tid [[thread_position_in_grid]])
{
    uint n_axes = uint(u_dim.x);
    uint nz = uint(u_dim.y);
    uint nc = uint(u_dim.z);
    uint M = uint(u_dim.w);
    if (tid >= n_axes * nc || M > 32u) return;

    uint a = tid / nc;
    uint c = tid - a * nc;

    float sat_prefac = 0.5f * f_p0.x * effgw_b[a];
    float inv_2dt = 0.5f / max(f_p0.y, 1.0f);
    float tend_max = f_p0.z;

    float u_src = u_src_b[a * nc + c];
    int k_launch = k_launch_b[a * nc + c];
    float c_min_crit = 2.0f;

    float cb[32];
    float sgn[32];
    float tau_launch[32];
    float tau_prev[32];

    uint tl_base = a * (M * nc);
    for (uint m = 0u; m < M; ++m) {
        float c_val = c_bins[m];
        float c_rel = c_val - u_src;
        cb[m] = c_val;
        sgn[m] = (c_rel >= 0.0f) ? 1.0f : -1.0f;
        tau_launch[m] = (fabs(c_rel) > c_min_crit) ? max(tau_launch_b[tl_base + m * nc + c], 0.0f) : 0.0f;
        tau_prev[m] = 0.0f;
    }

    uint a_base = a * (nz * nc);
    for (uint k = 0u; k < nz; ++k) {
        uint cidx = k * nc + c;
        uint ac_idx = a_base + cidx;

        float rk = rho_d[cidx];
        float inv_mass = 1.0f / max(rk * dz_c[cidx], 1.0e-4f);
        float rw = (k + 1u < nz) ? (0.5f * (rk + rho_d[(k + 1u) * nc + c])) : rk;
        float nw = max((k + 1u < nz) ? n_int[cidx] : n_int[(nz - 2u) * nc + c], 5.0e-3f);
        float sat_base = sat_prefac * (rw / nw);

        float uk = u_proj_b[ac_idx];
        float uw = (k + 1u < nz) ? (0.5f * (uk + u_proj_b[ac_idx + nc])) : uk;

        int ik = int(k);
        float acc = 0.0f;
        if (ik >= k_launch) {
            bool is_l = (ik == k_launch);
            for (uint m = 0u; m < M; ++m) {
                float tin = is_l ? tau_launch[m] : tau_prev[m];
                float c_hat = cb[m] - uw;
                float sg = sgn[m];
                float tout = 0.0f;
                if (c_hat * sg > c_min_crit) {
                    float ac_hat = fabs(c_hat);
                    float tau_sat = sat_base * (ac_hat * ac_hat * ac_hat);
                    tout = min(tin, tau_sat);
                }
                float dtau = max(tin - tout, 0.0f);
                float raw_acc = (sg * dtau) * inv_mass;
                float max_acc = inv_2dt * fabs(cb[m] - uk);
                acc += clamp(raw_acc, -max_acc, max_acc);
                tau_prev[m] = tout;
            }
        }
        du_out[ac_idx] = clamp(acc, -tend_max, tend_max);
    }
}

#define RAD_MAX_NZ 64

inline float rad_lacis_o3(float x_in) {
    float x = max(x_in, 0.0f);
    float a_vis = (0.02118f * x) / (1.0f + 0.042f * x + 0.000323f * x * x);
    float t1 = 103.6f * x;
    float a_uv = (1.082f * x) / pow(1.0f + 138.6f * x, 0.805f) + (0.0658f * x) / (1.0f + t1 * t1 * t1);
    return a_uv + a_vis;
}

inline float rad_lacis_wv(float y_in) {
    float y = max(y_in, 0.0f);
    return (2.9f * y) / (pow(1.0f + 141.5f * y, 0.635f) + 5.925f * y);
}

kernel void rad_four_band_col(
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
    device const float* t_sfc        [[buffer(11)]],
    device const float* alb_vis      [[buffer(12)]],
    device const float* alb_nir      [[buffer(13)]],
    device const float* alb_vis_dif  [[buffer(14)]],
    device const float* alb_nir_dif  [[buffer(15)]],
    device const float* mu0          [[buffer(16)]],
    device const float* s_toa        [[buffer(17)]],
    device const float* emiss_buf    [[buffer(18)]],
    device float* sw_up_out          [[buffer(19)]],
    device float* sw_dn_out          [[buffer(20)]],
    device float* lw_up_out          [[buffer(21)]],
    device float* lw_dn_out          [[buffer(22)]],
    device float* q_rad_3d           [[buffer(23)]],
    device float* sfc_out            [[buffer(24)]],
    device const float* rad_consts   [[buffer(25)]],
    constant ulong4& u_p             [[buffer(26)]],
    constant float4& f_p0            [[buffer(27)]],
    constant float4& f_p1            [[buffer(28)]],
    uint col                         [[thread_position_in_grid]]
) {
    uint N = uint(u_p.x);
    uint M = uint(u_p.y);
    uint dz_cols = uint(u_p.z);
    uint flags = uint(u_p.w);
    if (col >= M || N == 0u || N > RAD_MAX_NZ) return;

    bool has_dust = (flags & 1u) != 0u;
    bool has_o3 = (flags & 2u) != 0u;
    bool o3_lw_cpl = (flags & 4u) != 0u;
    bool has_emiss_buf = (flags & 8u) != 0u;

    uint dz_col = (dz_cols == 1u) ? 0u : col;

    float SOLAR_FRAC_SW1  = rad_consts[0];
    float SOLAR_FRAC_SW2  = rad_consts[1];
    float LW_NONWIN_FRAC  = rad_consts[2];
    float LW_WINDOW_FRAC  = rad_consts[3];
    float STEFAN_B        = rad_consts[4];
    float O3_G1           = rad_consts[5];
    float O3_G2           = rad_consts[6];
    float O3_G3           = rad_consts[7];
    float CO2_PREIND      = rad_consts[8];
    float PAR_FRAC_SW1    = rad_consts[9];
    float CP_D            = rad_consts[10];
    float CP_V            = rad_consts[11];
    float EPS_WATER       = rad_consts[12];
    float P_REF           = rad_consts[13];
    float T_FREEZE        = rad_consts[14];
    float GRAVITY         = rad_consts[15];
    float DUST_K_SW1      = rad_consts[16];
    float DUST_SSA_SW1    = rad_consts[17];
    float DUST_ASYM_SW1   = rad_consts[18];
    float DUST_K_SW2      = rad_consts[19];
    float DUST_SSA_SW2    = rad_consts[20];
    float DUST_ASYM_SW2   = rad_consts[21];
    float DUST_K_LW1      = rad_consts[22];
    float DUST_K_LW2      = rad_consts[23];

    float co2_ppmv = f_p0.x;
    float re_liq_um = f_p0.y;
    float re_ice_um = f_p0.z;
    float chi_sw = f_p0.w;
    float emiss_sfc = clamp(has_emiss_buf ? emiss_buf[col] : f_p1.x, 0.0f, 1.0f);

    float c_star[2 * RAD_MAX_NZ];
    float d_star[2 * RAD_MAX_NZ];

    float mu0_in = mu0[col];
    float s_toa_col = s_toa[col];
    float f_toa_1 = SOLAR_FRAC_SW1 * s_toa_col;
    float f_toa_2 = SOLAR_FRAC_SW2 * s_toa_col;
    bool day1 = (mu0_in > 1.0e-4f) && (f_toa_1 > 1.0e-4f);
    bool day2 = (mu0_in > 1.0e-4f) && (f_toa_2 > 1.0e-4f);

    if (!day1 && !day2) {
        for (uint k = 0u; k <= N; ++k) {
            sw_up_out[k * M + col] = 0.0f;
            sw_dn_out[k * M + col] = 0.0f;
        }
        sfc_out[0u * M + col] = 0.0f;
        sfc_out[1u * M + col] = 0.0f;
        sfc_out[2u * M + col] = 0.0f;
        sfc_out[3u * M + col] = 0.0f;
        sfc_out[4u * M + col] = 0.0f;
    } else {
        float mu0_safe = max(mu0_in, 1.0e-3f);
        float mu0_ts = max(mu0_in, 1.0e-4f);
        float inv_mu0 = 1.0f / mu0_ts;
        float sqrt3 = 1.7320508075688772f;
        float mag_o3 = 35.0f * rsqrt(1224.0f * mu0_safe * mu0_safe + 1.0f);

        // Band 0 (SW1) and Band 1 (SW2) sequentially, reusing lw_up, lw_dn, q_rad_3d[0..2]
        // as temporary per-layer buffers S0..S4 of length N (freed before LW runs).
        uint nq1 = N * M;
        uint nq2 = 2u * nq1;

        for (uint band = 0u; band < 2u; ++band) {
            bool is_sw1 = (band == 0u);
            bool active = is_sw1 ? day1 : day2;
            if (!active) {
                if (is_sw1) {
                    for (uint k = 0u; k <= N; ++k) {
                        sw_up_out[k * M + col] = 0.0f;
                        sw_dn_out[k * M + col] = 0.0f;
                    }
                    sfc_out[0u * M + col] = 0.0f;
                    sfc_out[1u * M + col] = 0.0f;
                    sfc_out[4u * M + col] = 0.0f;
                } else {
                    sfc_out[2u * M + col] = 0.0f;
                    sfc_out[3u * M + col] = 0.0f;
                }
                continue;
            }

            float f_beam0 = is_sw1 ? f_toa_1 : f_toa_2;
            float r_dir = clamp(is_sw1 ? alb_vis[col] : alb_nir[col], 0.0f, 1.0f);
            float r_sfc = clamp(is_sw1 ? alb_vis_dif[col] : alb_nir_dif[col], 0.0f, 1.0f);

            float k_ext_liq = is_sw1 ? (chi_sw * (2.817e-2f + 1.305f / re_liq_um))
                                     : (chi_sw * (2.456e-2f + 1.400f / re_liq_um));
            float om_liq = is_sw1 ? min(1.0f - (-5.62e-8f + 1.63e-7f * re_liq_um), 0.999999f)
                                  : clamp(1.0f - (0.01260f + 9.01e-4f * re_liq_um), 0.0f, 0.999999f);
            float g_liq = is_sw1 ? (0.829f + 2.482e-3f * re_liq_um)
                                 : (0.783f + 5.01e-3f * re_liq_um);

            float k_ext_ice = chi_sw * (3.448e-3f + 2.431f / re_ice_um);
            float om_ice = is_sw1 ? min(1.0f - 1.0e-5f, 0.999999f)
                                  : clamp(1.0f - (0.02437f + 2.52e-4f * re_ice_um), 0.0f, 0.999999f);
            float g_ice = is_sw1 ? (0.7661f + 5.851e-4f * re_ice_um)
                                 : (0.7867f + 5.90e-4f * re_ice_um);

            float gas_cum = 0.0f;
            float gas_a_prev = 0.0f;
            if (is_sw1) {
                uint k_toa = N - 1u;
                uint idx_toa = k_toa * M + col;
                float p_toa_c = pres_c[idx_toa];
                float o3_top;
                if (has_o3) {
                    o3_top = max(o3_vmr_c[idx_toa], 0.0f);
                } else {
                    float p_hpa = max(p_toa_c * 0.01f, 1.0e-3f);
                    o3_top = O3_G1 * pow(p_hpa, O3_G2) * exp(-p_hpa / O3_G3) * 1.0e-6f;
                }
                float p_top_w = pres_w[N * M + col];
                gas_cum = (o3_top * (48.0f / 28.964f) * (p_top_w / GRAVITY)) / ((O3_G2 + 1.0f) * 2.1415e-2f);
                gas_a_prev = min(rad_lacis_o3(gas_cum * mag_o3) / SOLAR_FRAC_SW1, 0.999f);
            }

            float tau_cum = 0.0f;
            float att_prev = 1.0f;
            float e1_p = 1.0f, e2_p = 1.0f, e3_p = 0.0f, e4_p = 0.0f, cp_t_p = 0.0f, cm_t_p = 0.0f;

            for (uint n = 0u; n < N; ++n) {
                uint k = N - 1u - n;
                uint idx = k * M + col;
                float pk = pres_c[idx];
                float tk = temp_c[idx];
                float dzk = dz_c[k * dz_cols + dz_col];
                float rhok = rho_d_c[idx];
                float dm_dry = rhok * dzk;

                float tau_lay = 0.0f;
                float scat_lay = 0.0f;
                float asym_scat_lay = 0.0f;

                if (is_sw1) {
                    float o3_v;
                    if (has_o3) {
                        o3_v = max(o3_vmr_c[idx], 0.0f);
                    } else {
                        float p_hpa = max(pk * 0.01f, 1.0e-3f);
                        o3_v = O3_G1 * pow(p_hpa, O3_G2) * exp(-p_hpa / O3_G3) * 1.0e-6f;
                    }
                    float du_o3 = (o3_v * (48.0f / 28.964f) * dm_dry) / 2.1415e-2f;
                    gas_cum += du_o3;
                    float a1 = min(rad_lacis_o3(gas_cum * mag_o3) / SOLAR_FRAC_SW1, 0.999f);
                    float tr = clamp((1.0f - a1) / max(1.0f - gas_a_prev, 1.0e-5f), 1.0e-5f, 1.0f);
                    float dtau_o3 = -mu0_safe * log(tr);
                    gas_a_prev = a1;

                    float dp_pa = max(pres_w[k * M + col] - pres_w[(k + 1u) * M + col], 1.0f);
                    float dtau_ray = 0.095f * (dp_pa / P_REF);
                    tau_lay += dtau_o3 + dtau_ray;
                    scat_lay += dtau_ray;
                } else {
                    float qvk = max(qv_c[idx], 0.0f);
                    float dy_wv = 0.1f * qvk * dm_dry * (pk / P_REF) * sqrt(T_FREEZE / max(tk, 180.0f));
                    gas_cum += dy_wv;
                    float a1 = min(rad_lacis_wv(gas_cum / mu0_safe) / SOLAR_FRAC_SW2, 0.999f);
                    float tr = clamp((1.0f - a1) / max(1.0f - gas_a_prev, 1.0e-5f), 1.0e-5f, 1.0f);
                    float dtau_wv = -mu0_safe * log(tr);
                    gas_a_prev = a1;
                    tau_lay += dtau_wv;
                }

                float lwp = 1000.0f * max(qc_c[idx], 0.0f) * dm_dry;
                float iwp = 1000.0f * max(qi_c[idx], 0.0f) * dm_dry;
                float tau_l = lwp * k_ext_liq;
                float tau_i = iwp * k_ext_ice;
                float tau_c = tau_l + tau_i;
                float scat_c = tau_l * om_liq + tau_i * om_ice;
                float om_c = (tau_c > 1.0e-8f) ? (scat_c / max(tau_c, 1.0e-8f)) : 0.0f;
                float g_c = (scat_c > 1.0e-8f) ? ((tau_l * om_liq * g_liq + tau_i * om_ice * g_ice) / max(scat_c, 1.0e-8f)) : 0.0f;
                tau_lay += tau_c;
                scat_lay += tau_c * om_c;
                asym_scat_lay += tau_c * om_c * g_c;

                if (has_dust) {
                    float dm_dust = max(q_dust_c[idx], 0.0f) * dm_dry;
                    float td = (is_sw1 ? DUST_K_SW1 : DUST_K_SW2) * dm_dust;
                    float sd = td * (is_sw1 ? DUST_SSA_SW1 : DUST_SSA_SW2);
                    tau_lay += td;
                    scat_lay += sd;
                    asym_scat_lay += sd * (is_sw1 ? DUST_ASYM_SW1 : DUST_ASYM_SW2);
                }

                float om_in = scat_lay / max(tau_lay, 1.0e-8f);
                float g_in = asym_scat_lay / max(scat_lay, 1.0e-8f);
                float w0 = clamp(om_in, 0.0f, 0.999999f);
                float g0 = clamp(g_in, -0.99f, 0.99f);
                float f_del = g0 * g0;
                float denom_del = max(1.0f - w0 * f_del, 1.0e-7f);
                float tau = max(tau_lay * (1.0f - w0 * f_del), 1.0e-7f);
                float w = clamp((w0 * (1.0f - f_del)) / denom_del, 0.0f, 0.999999f);
                float g = (g0 - f_del) / max(1.0f - f_del, 1.0e-7f);

                float gamma1 = 0.5f * sqrt3 * (2.0f - w * (1.0f + g));
                float gamma2 = 0.5f * sqrt3 * w * (1.0f - g);
                float gamma3 = 0.5f * (1.0f - sqrt3 * g * mu0_ts);
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

            // Final surface row 2*N - 1
            {
                uint r_last = 2u * N - 1u;
                float direct_sfc = mu0_ts * f_beam0 * att_prev;
                float Al = e1_p - r_sfc * e3_p;
                float Bl = e2_p - r_sfc * e4_p;
                float El = r_dir * direct_sfc - cp_t_p + r_sfc * cm_t_p;
                float inv_l = 1.0f / (Bl - Al * c_star[r_last - 1u]);
                d_star[r_last] = (El - Al * d_star[r_last - 1u]) * inv_l;
            }

            // Backward Thomas sweep + flux accumulation into sw_up_out / sw_dn_out (bottom-up!)
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
                if (is_sw1) {
                    sw_up_out[out_b] = f_up_b;
                    sw_dn_out[out_b] = f_dn_b;
                    if (n == N - 1u) {
                        float sd = max(f_dir, 0.0f);
                        sfc_out[0u * M + col] = sd;
                        sfc_out[1u * M + col] = max(f_dn_b - sd, 0.0f);
                        sfc_out[4u * M + col] = PAR_FRAC_SW1 * f_dn_b;
                    }
                    if (n == 0u) {
                        uint out_t = N * M + col;
                        sw_up_out[out_t] = max(y1 * e3 - y2 * e4 + cp_c, 0.0f);
                        sw_dn_out[out_t] = mu0_ts * f_beam0;
                    }
                } else {
                    sw_up_out[out_b] += f_up_b;
                    sw_dn_out[out_b] += f_dn_b;
                    if (n == N - 1u) {
                        float sd = max(f_dir, 0.0f);
                        sfc_out[2u * M + col] = sd;
                        sfc_out[3u * M + col] = max(f_dn_b - sd, 0.0f);
                    }
                    if (n == 0u) {
                        uint out_t = N * M + col;
                        sw_up_out[out_t] += max(y1 * e3 - y2 * e4 + cp_c, 0.0f);
                        sw_dn_out[out_t] += mu0_ts * f_beam0;
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // B. Longwave Band 1 + Band 2 (Exact decoupled 1-pass exponential)
    // ------------------------------------------------------------------
    float co2_ratio = max(co2_ppmv, 1.0f) / CO2_PREIND;
    float ln_co2 = log(co2_ratio);
    float co2_w1 = max(0.02f, 0.85f + 0.22f * ln_co2);
    float co2_strat = 0.22f * max(0.20f, 1.0f + 0.35f * ln_co2);
    float co2_w2 = max(0.0f, 0.015f + 0.024f * ln_co2);
    float p_ref2 = P_REF * P_REF;

    float k_ice_nw = 0.83f * (0.00324f + 1.033f / re_ice_um);
    float k_ice_win = 0.83f * (0.00160f + 1.166f / re_ice_um);
    float k_liq_lw = 0.83f * 0.090f;

    float dn1 = 0.0f;
    float dn2 = 0.0f;
    lw_dn_out[N * M + col] = 0.0f;

    float t_w_top = max(temp_w[N * M + col], 100.0f);
    float t2 = t_w_top * t_w_top;
    float eb_top = STEFAN_B * (t2 * t2);

    for (uint n = 0u; n < N; ++n) {
        uint k = N - 1u - n;
        uint idx = k * M + col;
        float pk = pres_c[idx];
        float tk = temp_c[idx];
        float dzk = dz_c[k * dz_cols + dz_col];
        float rhok = rho_d_c[idx];
        float dm_dry = rhok * dzk;
        float qvk = max(qv_c[idx], 0.0f);

        float p_top_w = pres_w[(k + 1u) * M + col];
        float p_bot_w = pres_w[k * M + col];
        float dp_pa = max(p_bot_w - p_top_w, 1.0f);
        float dp2_norm = max(p_bot_w * p_bot_w - p_top_w * p_top_w, 0.0f) / p_ref2;

        float pr = pk / P_REF;
        float dtau_wv1 = 0.30f * pow(max(pr, 1.0e-4f), 0.55f) * qvk * dm_dry;
        float strat_core = co2_strat * exp(-pk / 10000.0f) * sqrt(max(10000.0f / max(pk, 50.0f), 1.0f));
        float dtau_co2_1 = co2_w1 * dp2_norm + (0.020f + strat_core) * (dp_pa / P_REF);

        float e_vap = (qvk * pk) / (EPS_WATER + qvk);
        float t_cont = exp(min(1800.0f * (1.0f / max(tk, 190.0f) - 1.0f / 296.0f), 8.0f));
        float dtau_wv2 = (0.45f * (e_vap / P_REF) * t_cont + 0.0025f * pr) * qvk * dm_dry;
        float dtau_co2_2 = co2_w2 * dp2_norm;

        float lwp = 1000.0f * max(qc_c[idx], 0.0f) * dm_dry;
        float iwp = 1000.0f * max(qi_c[idx], 0.0f) * dm_dry;
        float tau_lw1 = dtau_wv1 + dtau_co2_1 + k_liq_lw * lwp + k_ice_nw * iwp;
        float tau_lw2 = dtau_wv2 + dtau_co2_2 + k_liq_lw * lwp + k_ice_win * iwp;

        if (o3_lw_cpl) {
            float o3_v;
            if (has_o3) {
                o3_v = max(o3_vmr_c[idx], 0.0f);
            } else {
                float p_hpa = max(pk * 0.01f, 1.0e-3f);
                o3_v = O3_G1 * pow(p_hpa, O3_G2) * exp(-p_hpa / O3_G3) * 1.0e-6f;
            }
            tau_lw2 += 4.5f * o3_v * (48.0f / 28.964f) * dm_dry;
        }
        if (has_dust) {
            float dm_dust = max(q_dust_c[idx], 0.0f) * dm_dry;
            tau_lw1 += DUST_K_LW1 * dm_dust;
            tau_lw2 += DUST_K_LW2 * dm_dust;
        }

        float t_w_bot = max(temp_w[k * M + col], 100.0f);
        float tb2 = t_w_bot * t_w_bot;
        float eb_bot = STEFAN_B * (tb2 * tb2);

        // Band 1
        float two_t1 = min(2.0f * max(tau_lw1, 1.0e-6f), 50.0f);
        float e1 = exp(-two_t1);
        float alpha1 = (two_t1 > 1.0e-3f) ? ((1.0f - e1) / two_t1) : (1.0f - 0.5f * two_t1 + (1.0f / 6.0f) * two_t1 * two_t1);
        float w1_1 = 1.0f - alpha1;
        float w2_1 = max(alpha1 - e1, 0.0f);
        float f0_1 = LW_NONWIN_FRAC * eb_top;
        float fb_1 = LW_NONWIN_FRAC * eb_bot;
        dn1 = dn1 * e1 + w2_1 * f0_1 + w1_1 * fb_1;

        // Band 2
        float two_t2 = min(2.0f * max(tau_lw2, 1.0e-6f), 50.0f);
        float e2 = exp(-two_t2);
        float alpha2 = (two_t2 > 1.0e-3f) ? ((1.0f - e2) / two_t2) : (1.0f - 0.5f * two_t2 + (1.0f / 6.0f) * two_t2 * two_t2);
        float w1_2 = 1.0f - alpha2;
        float w2_2 = max(alpha2 - e2, 0.0f);
        float f0_2 = LW_WINDOW_FRAC * eb_top;
        float fb_2 = LW_WINDOW_FRAC * eb_bot;
        dn2 = dn2 * e2 + w2_2 * f0_2 + w1_2 * fb_2;

        lw_dn_out[k * M + col] = max(dn1, 0.0f) + max(dn2, 0.0f);

        c_star[n] = e1;
        c_star[RAD_MAX_NZ + n] = e2;
        d_star[n] = w1_1 * f0_1 + w2_1 * fb_1;
        d_star[RAD_MAX_NZ + n] = w1_2 * f0_2 + w2_2 * fb_2;

        eb_top = eb_bot;
    }

    float ts_c = max(t_sfc[col], 100.0f);
    float ts2 = ts_c * ts_c;
    float eb_sfc = STEFAN_B * (ts2 * ts2);
    float r_lw = 1.0f - emiss_sfc;
    float up1 = emiss_sfc * LW_NONWIN_FRAC * eb_sfc + r_lw * dn1;
    float up2 = emiss_sfc * LW_WINDOW_FRAC * eb_sfc + r_lw * dn2;
    lw_up_out[col] = max(up1, 0.0f) + max(up2, 0.0f);

    for (int in = int(N) - 1; in >= 0; --in) {
        uint n = uint(in);
        up1 = up1 * c_star[n] + d_star[n];
        up2 = up2 * c_star[RAD_MAX_NZ + n] + d_star[RAD_MAX_NZ + n];
        uint k_top = N - n;
        lw_up_out[k_top * M + col] = max(up1, 0.0f) + max(up2, 0.0f);
    }

    // ------------------------------------------------------------------
    // C. Bottom-up Radiative Heating Rates [3, nz, M]
    // ------------------------------------------------------------------
    uint plane = N * M;
    uint plane2 = 2u * plane;
    float fnet_sw_prev = sw_up_out[col] - sw_dn_out[col];
    float fnet_lw_prev = lw_up_out[col] - lw_dn_out[col];
    for (uint k = 0u; k < N; ++k) {
        uint idx = k * M + col;
        uint up_idx = (k + 1u) * M + col;
        float fnet_sw_cur = sw_up_out[up_idx] - sw_dn_out[up_idx];
        float fnet_lw_cur = lw_up_out[up_idx] - lw_dn_out[up_idx];
        float dzk = dz_c[k * dz_cols + dz_col];
        float cp_lay = (CP_D + CP_V * max(qv_c[idx], 0.0f)) * (rho_d_c[idx] * dzk);
        float qsw = -(fnet_sw_cur - fnet_sw_prev) / cp_lay;
        float qlw = -(fnet_lw_cur - fnet_lw_prev) / cp_lay;
        q_rad_3d[idx] = qsw;
        q_rad_3d[plane + idx] = qlw;
        q_rad_3d[plane2 + idx] = qsw + qlw;
        fnet_sw_prev = fnet_sw_cur;
        fnet_lw_prev = fnet_lw_cur;
    }
}

inline float rad_ckd_planck_tail(float nu, float inv_t, float c2, bool is_upper) {
    if (nu <= 0.0f) return is_upper ? 0.0f : 1.0f;
    float x = (c2 * nu) * inv_t;
    float acc = 0.0f;
    constexpr float inv_n4[6] = {1.0f, 0.0625f, 0.012345679f, 0.00390625f, 0.0016f, 0.00077160494f};
    for (uint i = 0u; i < 6u; ++i) {
        float nx = float(i + 1u) * x;
        float e = exp(-min(nx, 60.0f));
        acc += e * inv_n4[i] * (((nx + 3.0f) * nx + 6.0f) * nx + 6.0f);
    }
    return clamp(0.15398973382026505f * acc, 0.0f, 1.0f);
}

inline float rad_ckd_band_frac(float nu1, float nu2, float temp_k, float c2) {
    float inv_t = 1.0f / max(temp_k, 100.0f);
    float f1 = rad_ckd_planck_tail(nu1, inv_t, c2, false);
    float f2 = rad_ckd_planck_tail(nu2, inv_t, c2, true);
    return max(f1 - f2, 0.0f);
}

kernel void rad_ckd_col(
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
    device const float* sfc_in       [[buffer(13)]],
    device float* sw_up_out          [[buffer(14)]],
    device float* sw_dn_out          [[buffer(15)]],
    device float* lw_up_out          [[buffer(16)]],
    device float* lw_dn_out          [[buffer(17)]],
    device float* q_rad_3d           [[buffer(18)]],
    device float* sfc_out            [[buffer(19)]],
    device const float* tbl          [[buffer(20)]],
    constant ulong4& u_p             [[buffer(21)]],
    constant float4& f_p0            [[buffer(22)]],
    constant float4& f_p1            [[buffer(23)]],
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

    uint dz_col = (dz_cols == 1u) ? 0u : col;

    // Unpack physical constants from tbl[820..837]
    float STEFAN_B   = tbl[820];
    float O3_G1      = tbl[821];
    float O3_G2      = tbl[822];
    float O3_G3      = tbl[823];
    float CP_D       = tbl[824];
    float CP_V       = tbl[825];
    float EPS_WATER  = tbl[826];
    float P_REF      = tbl[827];
    float GRAVITY    = tbl[828];
    float O2_VMR     = tbl[829];
    float C2_PLANCK  = tbl[830];
    float LW_DIFF    = tbl[831];
    float SW1_PAR_F  = tbl[832];
    float R_CO2      = tbl[833];
    float R_O3       = tbl[834];
    float R_CH4      = tbl[835];
    float R_N2O      = tbl[836];
    float R_O2       = tbl[837];

    float co2_ppmv   = max(f_p0.x, 0.0f);
    float ch4_ppmv   = max(f_p0.y, 0.0f);
    float n2o_ppmv   = max(f_p0.z, 0.0f);
    float chi_sw     = f_p0.w;
    float re_l_sc    = clamp(f_p1.x, 3.0f, 40.0f);
    float re_i_sc    = clamp(f_p1.y, 10.0f, 150.0f);

    float co2_mass_f = co2_ppmv * 1.0e-6f * R_CO2;
    float ch4_mass_f = ch4_ppmv * 1.0e-6f * R_CH4;
    float n2o_mass_f = n2o_ppmv * 1.0e-6f * R_N2O;
    float o2_mass_f  = O2_VMR * R_O2;

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

    // Zero SW outputs up front
    for (uint k = 0u; k <= N; ++k) {
        sw_up_out[k * M + col] = 0.0f;
        sw_dn_out[k * M + col] = 0.0f;
    }
    float s_vis_dir = 0.0f, s_vis_dif = 0.0f;
    float s_nir_dir = 0.0f, s_nir_dif = 0.0f;
    float s_par_wm2 = 0.0f;

    // ------------------------------------------------------------------
    // A. Shortwave 6-Band (18 g-point) Delta-Quadrature Two-Stream
    // ------------------------------------------------------------------
    bool is_day = (mu0_in > 1.0e-4f) && (s_toa_col > 1.0e-4f);
    if (is_day) {
        float mu0_safe = max(mu0_in, 1.0e-3f);
        float mu0_ts   = max(mu0_in, 1.0e-4f);
        float inv_mu0  = 1.0f / mu0_ts;
        float sqrt3    = 1.7320508075688772f;
        uint nq1       = N * M;
        uint nq2       = 2u * nq1;

        // Overhead O3 mass above model top
        uint k_toa = N - 1u;
        uint idx_toa = k_toa * M + col;
        float o3_top;
        if (has_o3) {
            o3_top = max(o3_vmr_c[idx_toa], 0.0f);
        } else {
            float ph = max(pres_c[idx_toa] * 0.01f, 1.0e-3f);
            o3_top = O3_G1 * pow(ph, O3_G2) * exp(-ph / O3_G3) * 1.0e-6f;
        }
        float u_o3_above = (o3_top * R_O3 * (pres_w[N * M + col] / GRAVITY)) / (O3_G2 + 1.0f);

        for (uint g_idx = 0u; g_idx < 18u; ++g_idx) {
            uint goff = 640u + 10u * g_idx;
            uint b    = uint(tbl[goff] + 0.5f);
            float w_g = tbl[goff + 1u];
            float ko  = tbl[goff + 2u];
            float kh  = tbl[goff + 3u];
            float gh  = tbl[goff + 4u];
            float kc  = tbl[goff + 5u];
            float gc  = tbl[goff + 6u];
            float km  = tbl[goff + 7u];
            float gm  = tbl[goff + 8u];
            float ko2 = tbl[goff + 9u];

            float f_beam0 = (w_g * tbl[538u + b]) * s_toa_col;
            if (ko > 0.0f) {
                f_beam0 *= exp(-min(ko * u_o3_above / mu0_safe, 40.0f));
            }
            if (f_beam0 <= 1.0e-6f) continue;

            bool is_vis = (b <= 1u);
            float r_dir = is_vis ? alb_vis_d : alb_nir_d;
            float r_sfc = is_vis ? alb_vis_f : alb_nir_f;

            float ray0 = tbl[544u + b];
            float la = tbl[550u + b], lb = tbl[556u + b], lc = tbl[562u + b], ld = tbl[568u + b], le = tbl[574u + b], lf = tbl[580u + b];
            float ia = tbl[586u + b], ib = tbl[592u + b], ic = tbl[598u + b], id = tbl[604u + b], ie = tbl[610u + b], i_f = tbl[616u + b];
            float dk = tbl[622u + b], ds = tbl[628u + b], da = tbl[634u + b];

            float tau_cum = 0.0f;
            float att_prev = 1.0f;
            float e1_p = 1.0f, e2_p = 1.0f, e3_p = 0.0f, e4_p = 0.0f, cp_t_p = 0.0f, cm_t_p = 0.0f;

            for (uint n = 0u; n < N; ++n) {
                uint k = N - 1u - n;
                uint idx = k * M + col;
                float pk = pres_c[idx];
                float dzk = dz_c[k * dz_cols + dz_col];
                float dm_dry = rho_d_c[idx] * dzk;
                float dp_frac = max(pres_w[k * M + col] - pres_w[(k + 1u) * M + col], 1.0f) / P_REF;

                float tau_ray = ray0 * dp_frac;
                float tau_lay = tau_ray;
                float scat_lay = tau_ray;
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
                    float qvk = max(qv_c[idx], 0.0f);
                    tau_lay += kh * pow(max((pk + 80.0f) / P_REF, 1.0e-4f), gh) * (qvk * dm_dry);
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
                float g = (g0 - f_del) / max(1.0f - f_del, 1.0e-7f);

                float gamma1 = 0.5f * sqrt3 * (2.0f - w * (1.0f + g));
                float gamma2 = 0.5f * sqrt3 * w * (1.0f - g);
                float gamma3 = 0.5f * (1.0f - sqrt3 * g * mu0_ts);
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

            // Bottom surface row
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

    // ------------------------------------------------------------------
    // B. Longwave 10-Band (32 g-point) Linear-in-tau Absorption Solver
    // ------------------------------------------------------------------
    for (uint k = 0u; k <= N; ++k) {
        lw_up_out[k * M + col] = 0.0f;
        lw_dn_out[k * M + col] = 0.0f;
    }

    float ts2 = ts_c * ts_c;
    float eb_sfc = STEFAN_B * (ts2 * ts2);
    float r_lw = 1.0f - emiss_sfc;

    constexpr uint lw_g_start[10] = {0u, 4u, 7u, 11u, 15u, 19u, 21u, 24u, 26u, 29u};
    constexpr uint lw_g_end[10]   = {4u, 7u, 11u, 15u, 19u, 21u, 24u, 26u, 29u, 32u};

    for (uint b = 0u; b < 10u; ++b) {
        float nu1 = tbl[2u * b];
        float nu2 = tbl[2u * b + 1u];
        float c_self = tbl[20u + b];
        float c_for  = tbl[30u + b];
        float cla = tbl[40u + b], clb = tbl[50u + b];
        float cia = tbl[60u + b], cib = tbl[70u + b];
        float kd_lw = tbl[80u + b];

        float bsfc_b = rad_ckd_band_frac(nu1, nu2, ts_c, C2_PLANCK) * eb_sfc;

        // Precompute band Planck flux at all N+1 interfaces into q_rad_3d[0..N]
        // and band base optical depth (continuum + cloud + dust) into d_star[RAD_MAX_NZ + n]
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
                float qdk = max(q_dust_c[idx], 0.0f);
                tb += kd_lw * (qdk * dm_dry);
            }
            d_star[RAD_MAX_NZ + n] = tb;
        }

        // Loop over g-points in band b
        for (uint g_idx = lw_g_start[b]; g_idx < lw_g_end[b]; ++g_idx) {
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
            float fb0 = w_g * q_rad_3d[col];
            for (uint n = 0u; n < N; ++n) {
                uint k = N - 1u - n;
                uint idx = k * M + col;
                float pk = pres_c[idx];
                float tk = temp_c[idx];
                float dzk = dz_c[k * dz_cols + dz_col];
                float dm_dry = rho_d_c[idx] * dzk;

                float tau = d_star[RAD_MAX_NZ + n];
                if (kh > 0.0f) {
                    float qvk = max(qv_c[idx], 0.0f);
                    tau += kh * pow(max((pk + 80.0f) / P_REF, 1.0e-4f), gh) * (qvk * dm_dry);
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
                    tau += ko * pow(max((pk + 120.0f) / P_REF, 1.0e-4f), go) * (o3v * R_O3 * dm_dry);
                }
                if (km > 0.0f || kn > 0.0f) {
                    float sf = pow(min(pk / 8000.0f, 1.0f), 0.38f);
                    float pn = max((pk + 100.0f) / P_REF, 1.0e-4f);
                    if (km > 0.0f) tau += km * pow(pn, gm) * (ch4_mass_f * sf * dm_dry);
                    if (kn > 0.0f) tau += kn * pow(pn, gn) * (n2o_mass_f * sf * dm_dry);
                }

                float dtau = clamp(LW_DIFF * tau, 1.0e-6f, 50.0f);
                float exp_lt = exp(-dtau);
                float alpha = (dtau > 1.0e-3f) ? ((1.0f - exp_lt) / dtau) : (1.0f - 0.5f * dtau + (1.0f / 6.0f) * dtau * dtau);
                float w1 = 1.0f - alpha;
                float w2 = max(alpha - exp_lt, 0.0f);

                float fb1 = w_g * q_rad_3d[(n + 1u) * M + col];
                cur_dn = cur_dn * exp_lt + (w2 * fb0 + w1 * fb1);
                c_star[n] = exp_lt;
                d_star[n] = w1 * fb0 + w2 * fb1;
                lw_dn_out[k * M + col] += cur_dn;
                fb0 = fb1;
            }

            float cur_up = emiss_sfc * (w_g * bsfc_b) + r_lw * cur_dn;
            lw_up_out[col] += cur_up;
            for (int in = int(N) - 1; in >= 0; --in) {
                uint n = uint(in);
                cur_up = cur_up * c_star[n] + d_star[n];
                uint k_top = N - n;
                lw_up_out[k_top * M + col] += cur_up;
            }
        }
    }

    // ------------------------------------------------------------------
    // C. Bottom-up Radiative Heating Rates [3, nz, M]
    // ------------------------------------------------------------------
    uint plane = N * M;
    uint plane2 = 2u * plane;
    float fnet_sw_prev = sw_up_out[col] - sw_dn_out[col];
    float fnet_lw_prev = lw_up_out[col] - lw_dn_out[col];
    for (uint k = 0u; k < N; ++k) {
        uint idx = k * M + col;
        uint up_idx = (k + 1u) * M + col;
        float fnet_sw_cur = sw_up_out[up_idx] - sw_dn_out[up_idx];
        float fnet_lw_cur = lw_up_out[up_idx] - lw_dn_out[up_idx];
        float dzk = dz_c[k * dz_cols + dz_col];
        float cp_lay = (CP_D + CP_V * max(qv_c[idx], 0.0f)) * (rho_d_c[idx] * dzk);
        float qsw = -(fnet_sw_cur - fnet_sw_prev) / cp_lay;
        float qlw = -(fnet_lw_cur - fnet_lw_prev) / cp_lay;
        q_rad_3d[idx] = qsw;
        q_rad_3d[plane + idx] = qlw;
        q_rad_3d[plane2 + idx] = qsw + qlw;
        fnet_sw_prev = fnet_sw_cur;
        fnet_lw_prev = fnet_lw_cur;
    }
}

inline float2 sfc_dai_band(float lc, float mu_s, float omega, float alb_g_dir, float alb_g_dif) {
    float k_opt = 0.5f / mu_s;
    float tmp_as = 0.5f * omega * (0.5f / (0.5f + mu_s)) * (
        1.0f - (mu_s / (0.5f + mu_s)) * log((0.5f + mu_s) / mu_s)
    );
    float beta_0 = clamp(((1.0f + k_opt) * tmp_as) / (omega * k_opt), 0.05f, 0.95f);

    float b = 1.0f - 0.5f * omega;
    float c = 0.5f * omega;
    float h = sqrt(b * b - c * c);
    float p1 = b + h;
    float p2 = b - h;

    float s1 = exp(-min(h * lc, 40.0f));
    float s2 = exp(-min(k_opt * lc, 40.0f));

    // Diffuse
    float d1 = p1 * (1.0f - alb_g_dif * p2 / c) / s1 - p2 * (1.0f - alb_g_dif * p1 / c) * s1;
    float d1_safe = (fabs(d1) < 1.0e-7f) ? 1.0e-7f : d1;
    float h7 = c * (1.0f - alb_g_dif * p2 / c) / (d1_safe * s1);
    float h8 = -c * (1.0f - alb_g_dif * p1 / c) * s1 / d1_safe;
    float alb_dif = clamp(h7 + h8, 0.0f, 1.0f);

    // Direct
    float sigma_raw = k_opt * k_opt - (b * b - c * c);
    float k_reg = (fabs(sigma_raw) < 1.0e-4f) ? (k_opt + 2.0e-4f) : k_opt;
    float sigma = k_reg * k_reg - (b * b - c * c);
    float d_coef = omega * k_reg * beta_0;
    float f_coef = omega * k_reg * (1.0f - beta_0);
    float u1 = b - c / max(alb_g_dir, 1.0e-4f);
    float h1 = -d_coef * (b - k_reg) - c * f_coef;
    float d2 = (u1 + h) / s1 - (u1 - h) * s1;
    float d2_safe = (fabs(d2) < 1.0e-7f) ? 1.0e-7f : d2;

    float t_b = d_coef - h1 * (b + k_reg) / sigma;
    float t_c = d_coef - c - (h1 / sigma) * (u1 + k_reg);
    float h2 = (t_b * (u1 - h) * s1 - p2 * t_c * s2) / d2_safe;
    float h3 = (-t_b * (u1 + h) / s1 + p1 * t_c * s2) / d2_safe;
    float alb_dir = clamp(h1 / sigma + h2 + h3, 0.0f, 1.0f);
    return float2(alb_dir, alb_dif);
}

kernel void surface_albedos_col(
    device const int* surf_type      [[buffer(0)]],
    device const float* swe_land     [[buffer(1)]],
    device const float* swe_max      [[buffer(2)]],
    device const float* sigma_topo   [[buffer(3)]],
    device const float* snow_age     [[buffer(4)]],
    device const float* w_liq_top    [[buffer(5)]],
    device const float* h_canopy     [[buffer(6)]],
    device const float* lai          [[buffer(7)]],
    device const float* m_seaice     [[buffer(8)]],
    device const float* swe_seaice   [[buffer(9)]],
    device const float* mu0          [[buffer(10)]],
    device float* out_8              [[buffer(11)]],
    device const float* sfc_consts   [[buffer(12)]],
    constant uint& M                 [[buffer(13)]],
    uint col                         [[thread_position_in_grid]]
) {
    if (col >= M) return;

    float ALB_SN_VN = sfc_consts[0];
    float ALB_SN_NN = sfc_consts[1];
    float ALB_SN_VO = sfc_consts[2];
    float ALB_SN_NO = sfc_consts[3];
    float RHO_SNOW  = sfc_consts[4];
    float SOIL_POR  = sfc_consts[5];
    float MC_ICE_CR = sfc_consts[6];
    float ALB_SI_V  = sfc_consts[7];
    float ALB_SI_N  = sfc_consts[8];
    float L_SCAT_V  = sfc_consts[9];
    float L_SCAT_N  = sfc_consts[10];

    int st = surf_type[col];
    bool is_ocean = (st == 0);
    bool is_ice = (st == 2);

    float mu_in = mu0[col];
    float sn = max(swe_land[col], 0.0f);
    float sn_max = max(swe_max[col], sn + 1.0e-4f);
    float n_melt = 200.0f / clamp(sigma_topo[col], 10.0f, 1000.0f);
    float ratio = clamp(sn / max(sn_max, 1.0f), 0.0f, 1.0f);
    float f_acc = tanh(sn / 25.0f);
    float theta_acos = acos(clamp(2.0f * ratio - 1.0f, -1.0f + 1.0e-6f, 1.0f - 1.0e-6f)) / 3.141592653589793f;
    float f_abl = 1.0f - pow(clamp(theta_acos, 0.0f, 1.0f), n_melt);
    float a_sn = (sn <= 1.0e-3f) ? 0.0f : clamp(min(f_acc, f_abl), 0.0f, 1.0f);
    if (is_ice) a_sn = 1.0f;

    float snow_depth = swe_land[col] / RHO_SNOW;
    float h0 = h_canopy[col];
    float h_buried = max(h0 - snow_depth, 0.0f);
    float lai_0 = lai[col];
    float lai_buried = lai_0 * (h_buried / max(h0, 0.05f));
    float h_exp = (1.0f - a_sn) * h0 + a_sn * h_buried;
    float lai_exp = (1.0f - a_sn) * lai_0 + a_sn * lai_buried;

    float alb_vd, alb_vf, alb_nd, alb_nf, f_ice;
    if (is_ocean) {
        f_ice = clamp(sqrt(max(m_seaice[col], 0.0f) / MC_ICE_CR), 0.0f, 1.0f);
        float mu = clamp(mu_in, 0.01f, 1.0f);
        float ocn_dir = clamp(0.026f / (pow(mu, 1.7f) + 0.065f) + 0.15f * (mu - 0.1f) * (mu - 0.5f) * (mu - 1.0f), 0.02f, 0.40f);
        float ocn_dif = 0.06f;
        float f_sn_ice = tanh(max(swe_seaice[col], 0.0f) / 15.0f);
        float ice_vis = (1.0f - f_sn_ice) * ALB_SI_V + f_sn_ice * ALB_SN_VN;
        float ice_nir = (1.0f - f_sn_ice) * ALB_SI_N + f_sn_ice * ALB_SN_NN;
        alb_vd = (1.0f - f_ice) * ocn_dir + f_ice * ice_vis;
        alb_vf = (1.0f - f_ice) * ocn_dif + f_ice * ice_vis;
        alb_nd = (1.0f - f_ice) * ocn_dir + f_ice * ice_nir;
        alb_nf = (1.0f - f_ice) * ocn_dif + f_ice * ice_nir;
    } else {
        f_ice = 0.0f;
        float sage = snow_age[col];
        float f_alb = clamp(sage / (1.0f + max(sage, 0.0f)), 0.0f, 0.999f);
        float vis0 = ALB_SN_VN + f_alb * (ALB_SN_VO - ALB_SN_VN);
        float nir0 = ALB_SN_NN + f_alb * (ALB_SN_NO - ALB_SN_NN);
        float mu_c = clamp(mu_in, 0.0f, 1.0f);
        float om_mu = 1.0f - mu_c;
        float om_mu2 = om_mu * om_mu;
        float zen_dir = 0.4f * om_mu2 * om_mu2 * om_mu;
        float zen_dif = 0.4f * 0.0058176653f;  // (1 - cos(50 deg))^5
        float sn_vd = clamp(vis0 + (1.0f - vis0) * zen_dir, 0.05f, 0.98f);
        float sn_vf = clamp(vis0 + (1.0f - vis0) * zen_dif, 0.05f, 0.98f);
        float sn_nd = clamp(nir0 + (1.0f - nir0) * zen_dir, 0.05f, 0.95f);
        float sn_nf = clamp(nir0 + (1.0f - nir0) * zen_dif, 0.05f, 0.95f);

        float wet = clamp(w_liq_top[col] / SOIL_POR, 0.0f, 1.0f);
        float dry_pow = (1.0f - wet) * sqrt(max(1.0f - wet, 0.0f));
        float soil_vis = 0.14f + 0.14f * dry_pow;
        float soil_nir = 0.28f + 0.22f * dry_pow;

        float gv_dir = (1.0f - a_sn) * soil_vis + a_sn * sn_vd;
        float gv_dif = (1.0f - a_sn) * soil_vis + a_sn * sn_vf;
        float gn_dir = (1.0f - a_sn) * soil_nir + a_sn * sn_nd;
        float gn_dif = (1.0f - a_sn) * soil_nir + a_sn * sn_nf;

        float lc = max(lai_exp, 0.0f);
        if (lc < 1.0e-3f) {
            alb_vd = gv_dir; alb_vf = gv_dif; alb_nd = gn_dir; alb_nf = gn_dif;
        } else {
            float mu_s = max(mu_in, 0.02f);
            float2 v = sfc_dai_band(lc, mu_s, L_SCAT_V, gv_dir, gv_dif);
            float2 n = sfc_dai_band(lc, mu_s, L_SCAT_N, gn_dir, gn_dif);
            alb_vd = v.x; alb_vf = v.y; alb_nd = n.x; alb_nf = n.y;
        }
    }

    out_8[0u * M + col] = alb_vd;
    out_8[1u * M + col] = alb_vf;
    out_8[2u * M + col] = alb_nd;
    out_8[3u * M + col] = alb_nf;
    out_8[4u * M + col] = h_exp;
    out_8[5u * M + col] = lai_exp;
    out_8[6u * M + col] = a_sn;
    out_8[7u * M + col] = f_ice;
}

inline float turb_dyer_phi_m(float zeta) {
    if (zeta < 0.0f) return pow(1.0f - 16.0f * zeta, -0.25f);
    return 1.0f + 5.0f * min(zeta, 2.0f);
}

inline float turb_dyer_phi_h(float zeta) {
    if (zeta < 0.0f) return rsqrt(1.0f - 16.0f * zeta);
    return 1.0f + 5.0f * min(zeta, 2.0f);
}

inline float turb_psi_m(float zeta) {
    if (zeta < 0.0f) {
        float x = pow(1.0f - 16.0f * zeta, 0.25f);
        return 2.0f * log(0.5f * (1.0f + x)) + log(0.5f * (1.0f + x * x)) - 2.0f * atan(x) + 1.5707963267948966f;
    }
    return -5.0f * min(zeta, 2.0f);
}

inline float turb_psi_h(float zeta) {
    if (zeta < 0.0f) {
        float x2 = sqrt(1.0f - 16.0f * zeta);
        return 2.0f * log(0.5f * (1.0f + x2));
    }
    return -5.0f * min(zeta, 2.0f);
}

kernel void surface_transfer_col(
    device const float* u_1          [[buffer(0)]],
    device const float* v_1          [[buffer(1)]],
    device const float* theta_v1     [[buffer(2)]],
    device const float* theta_vsfc   [[buffer(3)]],
    device const float* z_ref        [[buffer(4)]],
    device const float* z0m          [[buffer(5)]],
    device const float* z0h          [[buffer(6)]],
    device const float* z0q          [[buffer(7)]],
    device const float* rsl_depth    [[buffer(8)]],
    device float* out_6              [[buffer(9)]],
    constant ulong4& u_p             [[buffer(10)]],
    constant float4& f_p0            [[buffer(11)]],
    constant float4& f_p1            [[buffer(12)]],
    uint col                         [[thread_position_in_grid]]
) {
    uint M = uint(u_p.x);
    uint n_iter = uint(u_p.y);
    bool has_rsl = (u_p.z != 0ul);
    if (col >= M) return;

    float GRAVITY    = f_p0.x;
    float VON_KARMAN = f_p0.y;
    float RSL_MU_M   = f_p0.z;
    float RSL_MU_H   = f_p0.w;
    float RSL_NU     = f_p1.x;
    float RSL_LAMBDA = f_p1.y;

    float z_safe = max(z_ref[col], 2.0f);
    float z0m_s = min(max(z0m[col], 1.0e-6f), 0.45f * z_safe);
    float z0h_s = min(max(z0h[col], 1.0e-7f), 0.45f * z_safe);
    float z0q_s = min(max(z0q[col], 1.0e-7f), 0.45f * z_safe);

    float u = u_1[col];
    float v = v_1[col];
    float wind = sqrt(u * u + v * v + 0.25f);
    float thv1 = theta_v1[col];
    float thvs = theta_vsfc[col];
    float theta_bar = 0.5f * (thv1 + thvs);
    float ri_b = clamp(
        GRAVITY * z_safe * (thv1 - thvs) / (max(theta_bar, 200.0f) * wind * wind),
        -5.0f, 1.5f
    );

    float ln_zm = log(z_safe / z0m_s);
    float ln_zh = log(z_safe / z0h_s);
    float ln_zq = log(z_safe / z0q_s);

    bool active_rsl = false;
    float s_zm = 1.0f, s_0m = 1.0f, e_zm = 0.0f, e_0m = 0.0f;
    float s_zh = 1.0f, s_0h = 1.0f, e_zh = 0.0f, e_0h = 0.0f;
    float s_zq = 1.0f, s_0q = 1.0f, e_zq = 0.0f, e_0q = 0.0f;
    if (has_rsl) {
        float zs = max(rsl_depth[col], 0.0f);
        active_rsl = (zs > 0.05f);
        if (active_rsl) {
            float inv_zs = 1.0f / zs;
            float inv_lam = 1.0f / RSL_LAMBDA;
            float mux;

            mux = max(RSL_MU_M * (z_safe * inv_zs), 0.05f);
            s_zm = 1.0f + RSL_NU / mux;
            e_zm = inv_lam * log(1.0f + RSL_LAMBDA / mux) * exp(-min(mux, 40.0f));
            mux = max(RSL_MU_M * (z0m_s * inv_zs), 0.05f);
            s_0m = 1.0f + RSL_NU / mux;
            e_0m = inv_lam * log(1.0f + RSL_LAMBDA / mux) * exp(-min(mux, 40.0f));

            mux = max(RSL_MU_H * (z_safe * inv_zs), 0.05f);
            s_zh = 1.0f + RSL_NU / mux;
            e_zh = inv_lam * log(1.0f + RSL_LAMBDA / mux) * exp(-min(mux, 40.0f));
            mux = max(RSL_MU_H * (z0h_s * inv_zs), 0.05f);
            s_0h = 1.0f + RSL_NU / mux;
            e_0h = inv_lam * log(1.0f + RSL_LAMBDA / mux) * exp(-min(mux, 40.0f));

            mux = max(RSL_MU_H * (z0q_s * inv_zs), 0.05f);
            s_zq = s_zh; e_zq = e_zh;
            s_0q = 1.0f + RSL_NU / mux;
            e_0q = inv_lam * log(1.0f + RSL_LAMBDA / mux) * exp(-min(mux, 40.0f));
        }
    }

    float rz_m = z0m_s / z_safe;
    float rz_h = z0h_s / z_safe;
    float rz_q = z0q_s / z_safe;
    float zeta = clamp(ri_b * ln_zm, -5.0f, 2.0f);

    for (uint it = 0u; it < n_iter; ++it) {
        float zeta_0m = zeta * rz_m;
        float zeta_0h = zeta * rz_h;
        float fm = ln_zm - turb_psi_m(zeta) + turb_psi_m(zeta_0m);
        float fh = ln_zh - turb_psi_h(zeta) + turb_psi_h(zeta_0h);
        if (active_rsl) {
            float rsl_m = turb_dyer_phi_m(s_0m * zeta_0m) * e_0m - turb_dyer_phi_m(s_zm * zeta) * e_zm;
            float rsl_h = turb_dyer_phi_h(s_0h * zeta_0h) * e_0h - turb_dyer_phi_h(s_zh * zeta) * e_zh;
            fm -= rsl_m;
            fh -= rsl_h;
        }
        fm = max(fm, 0.8f);
        fh = max(fh, 0.8f);
        zeta = clamp(ri_b * fm * fm / fh, -5.0f, 2.0f);
    }

    float zeta_0m = zeta * rz_m;
    float zeta_0h = zeta * rz_h;
    float zeta_0q = zeta * rz_q;
    float fm = ln_zm - turb_psi_m(zeta) + turb_psi_m(zeta_0m);
    float fh = ln_zh - turb_psi_h(zeta) + turb_psi_h(zeta_0h);
    float fq = ln_zq - turb_psi_h(zeta) + turb_psi_h(zeta_0q);
    if (active_rsl) {
        fm -= (turb_dyer_phi_m(s_0m * zeta_0m) * e_0m - turb_dyer_phi_m(s_zm * zeta) * e_zm);
        fh -= (turb_dyer_phi_h(s_0h * zeta_0h) * e_0h - turb_dyer_phi_h(s_zh * zeta) * e_zh);
        fq -= (turb_dyer_phi_h(s_0q * zeta_0q) * e_0q - turb_dyer_phi_h(s_zq * zeta) * e_zq);
    }
    fm = max(fm, 0.8f);
    fh = max(fh, 0.8f);
    fq = max(fq, 0.8f);

    float vk2 = VON_KARMAN * VON_KARMAN;
    float c_m = vk2 / (fm * fm);
    float c_h = vk2 / (fm * fh);
    float c_q = vk2 / (fm * fq);
    float u_star = sqrt(c_m) * wind;

    out_6[0u * M + col] = c_m;
    out_6[1u * M + col] = c_h;
    out_6[2u * M + col] = c_q;
    out_6[3u * M + col] = u_star;
    out_6[4u * M + col] = zeta;
    out_6[5u * M + col] = wind;
}

inline float bio_arrhenius(float val25, float dha, float t_k, float r_gas) {
    float t_s = clamp(t_k, 240.0f, 335.0f);
    return val25 * exp(dha * (t_s - 298.15f) / (298.15f * r_gas * t_s));
}

inline float bio_peaked_arrh_ds(float t_k, float ha, float ds, float hd, float r_gas) {
    float t0 = 298.15f;
    float t_s = clamp(t_k, 250.0f, 330.0f);
    float num = 1.0f + exp((t0 * ds - hd) / (r_gas * t0));
    float den = 1.0f + exp((t_s * ds - hd) / (r_gas * t_s));
    return exp(ha * (t_s - t0) / (r_gas * t0 * t_s)) * (num / den);
}

kernel void two_leaf_gas_exchange_col(
    device const float* i_sun        [[buffer(0)]],
    device const float* i_sh         [[buffer(1)]],
    device const float* l_sun        [[buffer(2)]],
    device const float* l_sh         [[buffer(3)]],
    device const float* sun_frac     [[buffer(4)]],
    device const float* vc25_c3      [[buffer(5)]],
    device const float* jm25_c3      [[buffer(6)]],
    device const float* r25_c3       [[buffer(7)]],
    device const float* vc25_c4      [[buffer(8)]],
    device const float* jm25_c4      [[buffer(9)]],
    device const float* vp25_c4      [[buffer(10)]],
    device const float* r25_c4       [[buffer(11)]],
    device const float* tg_mid       [[buffer(12)]],
    device const float* t_home       [[buffer(13)]],
    device const float* f_c4         [[buffer(14)]],
    device const float* t_sun        [[buffer(15)]],
    device const float* t_sh         [[buffer(16)]],
    device const float* vpd_sun      [[buffer(17)]],
    device const float* vpd_sh       [[buffer(18)]],
    device const float* pres_sfc     [[buffer(19)]],
    device const float* lai_exp      [[buffer(20)]],
    device const float* ws_beta      [[buffer(21)]],
    device const float* gbw_sun      [[buffer(22)]],
    device const float* gbw_sh       [[buffer(23)]],
    device float* out_5              [[buffer(24)]],
    device const float* gas_consts   [[buffer(25)]],
    constant ulong4& u_p             [[buffer(26)]],
    constant float4& f_p0            [[buffer(27)]],
    uint col                         [[thread_position_in_grid]]
) {
    uint M = uint(u_p.x);
    uint flags = uint(u_p.y);
    if (col >= M) return;

    float lc = lai_exp[col];
    if (lc < 0.01f) {
        out_5[0u * M + col] = 0.0f;
        out_5[1u * M + col] = 0.0f;
        out_5[2u * M + col] = 0.0f;
        out_5[3u * M + col] = 0.0f;
        out_5[4u * M + col] = 0.0f;
        return;
    }

    float R_GAS          = gas_consts[0];
    float T_FREEZE       = gas_consts[1];
    float O2_RATIO       = gas_consts[2];
    float THETA_NRH      = gas_consts[3];
    float PHI_PSII_REF   = gas_consts[4];
    float BETA_C3        = gas_consts[5];
    float BETA_C4        = gas_consts[6];
    float GAMMA25_C3     = gas_consts[7];
    float KC25_C3        = gas_consts[8];
    float KO25_C3        = gas_consts[9];
    float DHA_GAMMA      = gas_consts[10];
    float DHA_KC_C3      = gas_consts[11];
    float DHA_KO_C3      = gas_consts[12];
    float KP25_UMOL_C4   = gas_consts[13];
    float DHA_KP_C4      = gas_consts[14];
    float PHI_LEAK_C4    = gas_consts[15];
    float HA_VCMAX_BASE  = gas_consts[16];
    float HA_JMAX        = gas_consts[17];
    float HA_VPMAX       = gas_consts[18];
    float HD_ENZYME      = gas_consts[19];
    float HA_VCMAX_SLOPE = gas_consts[20];
    float co2_ppmv       = f_p0.x;

    float p_s = pres_sfc[col];
    float ca_pa = (co2_ppmv * 1.0e-6f) * p_s;
    float oi_pa = O2_RATIO * p_s;
    float tg = tg_mid[col];
    float th = t_home[col];
    float ws = clamp(ws_beta[col], 0.0f, 1.0f);
    float f4 = f_c4[col];
    float sf = sun_frac[col];

    float ls_sun = l_sun[col];
    float ls_sh = l_sh[col];
    float g0_mol = (ls_sun + ls_sh > 0.01f) ? (0.002f * ws) : 0.0f;
    float eta_25 = exp(-3.719f + 580.0f / (298.15f - 138.0f));
    float phi_20 = 0.656f;

    // Acclimated thermal kinetics, quantum yield phi(T_growth), and stomatal sensitivity xi'(T_growth)
    // (Kumarathunge et al., 2019; Schneider et al., 2025, Table 1 & Eqs. 7, 11-12; Gan et al., 2026, Eq. 18)
    float tg_c = clamp(tg - T_FREEZE, 0.0f, 40.0f);
    float th_c = clamp(th - T_FREEZE, 0.0f, 40.0f);
    float ha_v_eff = HA_VCMAX_BASE + HA_VCMAX_SLOPE * tg_c;
    float ds_v = 645.13f - 0.38f * tg_c;
    float ds_j = 658.77f - 0.84f * th_c - 0.52f * (tg_c - th_c);

    float tg_safe = clamp(tg, 253.15f, 333.15f);
    float eta_g = exp(-3.719f + 580.0f / (tg_safe - 138.0f)) / eta_25;
    float tc_pg = clamp(tg - T_FREEZE, -5.0f, 50.0f);
    float phi_g = PHI_PSII_REF * (max(0.352f + 0.022f * tc_pg - 3.4e-4f * tc_pg * tc_pg, 0.02f) / phi_20);
    float gamma3_g = bio_arrhenius(GAMMA25_C3, DHA_GAMMA, tg, R_GAS) * (p_s / 101325.0f);
    float km3_g = bio_arrhenius(KC25_C3, DHA_KC_C3, tg, R_GAS) * (1.0f + oi_pa / bio_arrhenius(KO25_C3, DHA_KO_C3, tg, R_GAS));
    float xi3 = sqrt(BETA_C3 * (km3_g + gamma3_g) / (1.6f * eta_g));
    float kp4_g = bio_arrhenius(KP25_UMOL_C4 * 1.0e-6f * 101325.0f, DHA_KP_C4, tg, R_GAS);
    float xi4 = sqrt(BETA_C4 * kp4_g / (1.6f * eta_g));

    float gpp_tot = 0.0f;
    float rd_tot = 0.0f;
    float gsw_sun_r = 0.0f;
    float gsw_sh_r = 0.0f;

    for (uint side = 0u; side < 2u; ++side) {
        bool is_sun = (side == 0u);
        float i_par = is_sun ? i_sun[col] : i_sh[col];
        float tl = is_sun ? t_sun[col] : t_sh[col];
        float d_af = max(is_sun ? vpd_sun[col] : vpd_sh[col], 20.0f);
        bool has_gbw = is_sun ? ((flags & 1u) != 0u) : ((flags & 2u) != 0u);

        float v3t = vc25_c3[col], v3s = v3t * sf;
        float j3t = jm25_c3[col], j3s = j3t * sf;
        float r3t = r25_c3[col],  r3s = r3t * sf;
        float v4t = vc25_c4[col], v4s = v4t * sf;
        float j4t = jm25_c4[col], j4s = j4t * sf;
        float p4t = vp25_c4[col], p4s = p4t * sf;
        float r4t = r25_c4[col],  r4s = r4t * sf;

        float vc25_3 = is_sun ? v3s : max(v3t - v3s, 1.0e-6f);
        float jm25_3 = is_sun ? j3s : max(j3t - j3s, 1.0e-6f);
        float r25_3  = is_sun ? r3s : max(r3t - r3s, 1.0e-6f);
        float vc25_4 = is_sun ? v4s : max(v4t - v4s, 1.0e-6f);
        float jm25_4 = is_sun ? j4s : max(j4t - j4s, 1.0e-6f);
        float vp25_4 = is_sun ? p4s : max(p4t - p4s, 1.0e-6f);
        float r25_4  = is_sun ? r4s : max(r4t - r4s, 1.0e-6f);

        float fv = bio_peaked_arrh_ds(tl, ha_v_eff, ds_v, HD_ENZYME, R_GAS);
        float fj = bio_peaked_arrh_ds(tl, HA_JMAX,  ds_j, HD_ENZYME, R_GAS);
        float fp = bio_peaked_arrh_ds(tl, HA_VPMAX, ds_v, HD_ENZYME, R_GAS);

        float tc_r = clamp(tl - T_FREEZE, -20.0f, 55.0f);
        float fr = exp((0.1012f * tc_r - 0.0005f * tc_r * tc_r) - 2.2175f);

        float gamma3 = bio_arrhenius(GAMMA25_C3, DHA_GAMMA, tl, R_GAS) * (p_s / 101325.0f);
        float km3 = bio_arrhenius(KC25_C3, DHA_KC_C3, tl, R_GAS) * (1.0f + oi_pa / bio_arrhenius(KO25_C3, DHA_KO_C3, tl, R_GAS));
        float vc3 = ws * vc25_3 * fv;
        float jm3 = ws * jm25_3 * fj;
        float ip3 = phi_g * i_par;
        float b3 = ip3 + jm3;
        float j_3 = (b3 - sqrt(max(b3 * b3 - 4.0f * THETA_NRH * ip3 * jm3, 0.0f))) / (2.0f * THETA_NRH);
        float rd3 = r25_3 * fr;

        float kp4 = bio_arrhenius(KP25_UMOL_C4 * 1.0e-6f * 101325.0f, DHA_KP_C4, tl, R_GAS);
        float vc4 = ws * vc25_4 * fv;
        float jm4 = ws * jm25_4 * fj;
        float vp4 = ws * vp25_4 * fp;
        float b4 = ip3 + jm4;
        float j_4 = (b4 - sqrt(max(b4 * b4 - 4.0f * THETA_NRH * ip3 * jm4, 0.0f))) / (2.0f * THETA_NRH);
        float sig4 = 1.0f / (1.0f + exp(-0.35f * ((tl - T_FREEZE) - 12.0f)));
        float al4 = 0.20f * (1.0f - PHI_LEAK_C4) * j_4 * sig4;
        float acj4 = min(vc4, al4);
        float rd4 = r25_4 * fr;

        float molar_vol = R_GAS * tl / p_s;
        float gbw_mol = 0.0f, gbc_mol = 0.0f;
        uint n_bl = 1u;
        if (has_gbw) {
            float g_ms = is_sun ? gbw_sun[col] : gbw_sh[col];
            gbw_mol = max(g_ms / molar_vol, 1.0e-4f);
            gbc_mol = gbw_mol / 1.37f;
            n_bl = 2u;
        }

        float cs_pa = ca_pa;
        float d_s = max(d_af, 50.0f);
        float gpp_mix = 0.0f;
        float rd_mix = (1.0f - f4) * rd3 + f4 * rd4;
        float gsw_mol = g0_mol;

        for (uint bl = 0u; bl < n_bl; ++bl) {
            float cs_mol = max(cs_pa / p_s, 1.0e-5f);
            float sqrt_ds = sqrt(d_s);

            float chi3 = clamp(gamma3 / cs_pa + (1.0f - gamma3 / cs_pa) * (xi3 / (xi3 + sqrt_ds)), 0.05f, 0.95f);
            float ci3 = chi3 * cs_pa;
            float ac3 = vc3 * max((ci3 - gamma3) / (ci3 + km3), 0.0f);
            float aj3 = 0.25f * j_3 * max((ci3 - gamma3) / (ci3 + 2.0f * gamma3), 0.0f);
            float gpp3 = min(ac3, aj3);
            float an3_pos = max(gpp3 - rd3, 0.0f);
            float gsw3_mol = 1.6f * (an3_pos * 1.0e-6f) / max(cs_mol * (1.0f - chi3), 1.0e-6f);

            float chi4 = clamp(xi4 / (xi4 + sqrt_ds), 0.05f, 0.92f);
            float cm4 = chi4 * cs_pa;
            float ap4 = (1.0f - PHI_LEAK_C4) * vp4 * cm4 / max(cm4 + kp4, 0.5f);
            float gpp4 = min(acj4, ap4);
            float an4_pos = max(gpp4 - rd4, 0.0f);
            float gsw4_mol = 1.6f * (an4_pos * 1.0e-6f) / max(cs_mol * (1.0f - chi4), 1.0e-6f);

            gpp_mix = (1.0f - f4) * gpp3 + f4 * gpp4;
            gsw_mol = (1.0f - f4) * gsw3_mol + f4 * gsw4_mol + g0_mol;

            if (has_gbw && bl + 1u < n_bl) {
                float an_mix_mol = max(gpp_mix - rd_mix, 0.0f) * 1.0e-6f;
                cs_pa = clamp(ca_pa - (an_mix_mol / gbc_mol) * p_s, 0.35f * ca_pa, ca_pa);
                d_s = max((gbw_mol / (gbw_mol + gsw_mol)) * d_af, 20.0f);
            }
        }

        gpp_tot += gpp_mix;
        rd_tot += rd_mix;
        float gsw_ms = gsw_mol * molar_vol;
        if (is_sun) {
            gsw_sun_r = (ls_sun > 1.0e-4f) ? gsw_ms : 0.0f;
        } else {
            gsw_sh_r = (ls_sh > 1.0e-4f) ? gsw_ms : 0.0f;
        }
    }

    out_5[0u * M + col] = gpp_tot;
    out_5[1u * M + col] = rd_tot;
    out_5[2u * M + col] = gpp_tot - rd_tot;
    out_5[3u * M + col] = gsw_sun_r;
    out_5[4u * M + col] = gsw_sh_r;
}

kernel void scvt_reconstruct_winds(
    device const float* u_edge     [[buffer(0)]],
    device const int* eoc          [[buffer(1)]],
    device const float* rec_ew     [[buffer(2)]],
    device const float* rec_nw     [[buffer(3)]],
    device float* u_zonal          [[buffer(4)]],
    device float* v_merid          [[buffer(5)]],
    constant ulong4& u_p           [[buffer(6)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint nz = uint(u_p.x);
    uint n_cells = uint(u_p.y);
    uint n_edges = uint(u_p.z);
    if (gid >= nz * n_cells) return;
    uint k = gid / n_cells;
    uint cell = gid - k * n_cells;
    uint cbase = cell * 6u;
    uint kbase = k * n_edges;
    float ue = 0.0f;
    float vn = 0.0f;
    for (uint m = 0u; m < 6u; ++m) {
        uint e = uint(eoc[cbase + m]);
        float val = u_edge[kbase + e];
        ue = fma(val, rec_ew[cbase + m], ue);
        vn = fma(val, rec_nw[cbase + m], vn);
    }
    u_zonal[gid] = ue;
    v_merid[gid] = vn;
}

kernel void scvt_project_winds(
    device const float* vec_east   [[buffer(0)]],
    device const float* vec_north  [[buffer(1)]],
    device const int* c0           [[buffer(2)]],
    device const int* c1           [[buffer(3)]],
    device const float* east_ne    [[buffer(4)]],
    device const float* north_ne   [[buffer(5)]],
    device float* u_edge_out       [[buffer(6)]],
    constant ulong4& u_p           [[buffer(7)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint nz = uint(u_p.x);
    uint n_edges = uint(u_p.y);
    uint n_cells = uint(u_p.z);
    if (gid >= nz * n_edges) return;
    uint k = gid / n_edges;
    uint e = gid - k * n_edges;
    uint kbase = k * n_cells;
    uint i0 = kbase + uint(c0[e]);
    uint i1 = kbase + uint(c1[e]);
    float ve = 0.5f * (vec_east[i0] + vec_east[i1]);
    float vn = 0.5f * (vec_north[i0] + vec_north[i1]);
    u_edge_out[gid] = ve * east_ne[e] + vn * north_ne[e];
}

// =========================================================================
// 19. Batched 4-Species Natural Aerosol Column (Settling, Dry/Wet Removal, AOD)
// =========================================================================

inline float _aoba_neg_expm1_neg(float x) {
    // Accurate 1 - exp(-x) for x >= 0 without catastrophic cancellation as x -> 0.
    float xc = clamp(x, 0.0f, 40.0f);
    if (xc < 1.0e-3f) {
        return xc * (1.0f - 0.5f * xc * (1.0f - (1.0f / 3.0f) * xc * (1.0f - 0.25f * xc)));
    }
    return 1.0f - exp(-xc);
}

inline float _aoba_lambert_w0_exp(float y) {
    // Overflow-free principal Lambert W0(exp(y)) solving w + ln(w) = y (Iacono & Boyd, 2017).
    float w;
    if (y > 1.0f) {
        float ly = log(y);
        w = y - ly + ly / y;
    } else {
        float ey = exp(clamp(y, -30.0f, 1.0f));
        w = ey / (1.0f + ey);
    }
    w = max(w, 1.0e-12f);
    for (int it = 0; it < 3; ++it) {
        w = max(w * (1.0f + y - log(w)) / (1.0f + w), 1.0e-12f);
    }
    return w;
}

kernel void natural_aerosol_4sp_col(
    device const float* q_in        [[buffer(0)]],  // [4, nz, nc]
    device const float* rho_d       [[buffer(1)]],  // [nz, nc]
    device const float* dz_c        [[buffer(2)]],  // [nz, dz_cols]
    device const float* emis_4      [[buffer(3)]],  // [4, nc]
    device const float* precip      [[buffer(4)]],  // [nc]
    device const float* u_star_in   [[buffer(5)]],  // [nc]
    device float* q_out             [[buffer(6)]],  // [4, nz, nc]
    device float* aod_out           [[buffer(7)]],  // [4, nc]
    constant ulong4& u_dim          [[buffer(8)]],  // [n_sp, nz, nc, dz_cols]
    constant float4& v_settle_4     [[buffer(9)]],  // [v_ss, v_so4, v_bc, v_oc]
    constant float4& v_dry_4        [[buffer(10)]], // [vd_ss, vd_so4, vd_bc, vd_oc]
    constant float4& alpha_wet_4    [[buffer(11)]], // [aw_ss, aw_so4, aw_bc, aw_oc]
    constant float4& mee_4          [[buffer(12)]], // [mee_ss, mee_so4, mee_bc, mee_oc]
    constant float& dt              [[buffer(13)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint n_sp = uint(u_dim.x);
    uint nz = uint(u_dim.y);
    uint nc = uint(u_dim.z);
    uint dz_cols = uint(u_dim.w);
    if (gid >= n_sp * nc || nz == 0u) return;

    uint s = gid / nc;
    uint c = gid - s * nc;
    uint col_dz = (dz_cols > 1u) ? c : 0u;

    float vs = (s == 0u) ? v_settle_4.x : ((s == 1u) ? v_settle_4.y : ((s == 2u) ? v_settle_4.z : v_settle_4.w));
    float vd = (s == 0u) ? v_dry_4.x    : ((s == 1u) ? v_dry_4.y    : ((s == 2u) ? v_dry_4.z    : v_dry_4.w));
    float aw = (s == 0u) ? alpha_wet_4.x: ((s == 1u) ? alpha_wet_4.y: ((s == 2u) ? alpha_wet_4.z: alpha_wet_4.w));
    float mee = (s == 0u)? mee_4.x      : ((s == 1u) ? mee_4.y      : ((s == 2u) ? mee_4.z      : mee_4.w));

    float pr = max(precip[c], 0.0f);
    float wash = exp(-min(aw * pr * dt, 20.0f));
    float emis = max(emis_4[s * nc + c], 0.0f);
    float u_st = max(u_star_in[c], 0.05f);

    uint sp_base = s * nz * nc;

    // 1. Register-resident emission injection + mass-conserving PBL turbulent mixing (k = 0..n_mix)
    uint n_mix = (nz > 1u) ? min(6u, nz - 1u) : 0u;
    float q_pbl[7];
    float m_pbl[7];
    float dz_pbl[7];
    float rho_pbl[7];
    float z_rel[7];
    for (uint k = 0u; k <= n_mix; ++k) {
        uint idx2 = k * nc + c;
        float rd = rho_d[idx2];
        float dz = dz_c[k * dz_cols + col_dz];
        rho_pbl[k] = rd;
        dz_pbl[k] = dz;
        m_pbl[k] = max(rd * dz, 1.0e-5f);
        q_pbl[k] = max(q_in[sp_base + idx2], 0.0f);
        z_rel[k] = (k == 0u) ? (0.5f * dz) : (z_rel[k - 1u] + 0.5f * (dz_pbl[k - 1u] + dz));
    }
    q_pbl[0] += (dt * emis) / m_pbl[0];
    for (uint k = 0u; k < n_mix; ++k) {
        float z_int = 0.5f * (z_rel[k] + z_rel[k + 1u]);
        if (z_int < 1200.0f) {
            float fz = 1.0f - z_int / 1200.0f;
            float kz = 0.4f * u_st * z_int * fz * fz;
            float dz_int = max(0.5f * (dz_pbl[k] + dz_pbl[k + 1u]), 10.0f);
            float rho_int = 0.5f * (rho_pbl[k] + rho_pbl[k + 1u]);
            float exch = (dt * rho_int * kz) / dz_int;
            float frac = exch / (1.0f + exch * (1.0f / m_pbl[k] + 1.0f / m_pbl[k + 1u]));
            float dq_flux = frac * (q_pbl[k] - q_pbl[k + 1u]);
            q_pbl[k] -= dq_flux / m_pbl[k];
            q_pbl[k + 1u] += dq_flux / m_pbl[k + 1u];
        }
    }

    // 2. Top-down implicit gravitational settling + surface dry deposition + wet scavenging
    float flux_above = 0.0f;
    float col_mass = 0.0f;

    for (int ki = int(nz) - 1; ki >= 0; --ki) {
        uint k = uint(ki);
        uint idx2 = k * nc + c;
        uint idx3 = sp_base + idx2;
        float rd = rho_d[idx2];
        float dz = dz_c[k * dz_cols + col_dz];
        float mk_raw = rd * dz;
        float mk = max(mk_raw, 1.0e-5f);
        float q_cur = (k <= n_mix) ? q_pbl[k] : max(q_in[idx3], 0.0f);
        float v_bot = (k == 0u) ? (vs + vd) : vs;
        float sink = dt * rd * v_bot;
        float q_k = (mk * q_cur + dt * flux_above) / (mk + sink);
        flux_above = rd * v_bot * q_k;
        float q_fin = clamp(q_k * wash, 0.0f, 1.0e-3f);
        q_out[idx3] = q_fin;
        col_mass += mk_raw * q_fin;
    }
    aod_out[s * nc + c] = mee * col_mass;
}

// =========================================================================
// 20. 5-Category Sea-Ice ITD Sync/Rebinning/Ridging & EVP Subcycle Kernels
// =========================================================================

kernel void seaice_itd_sync_col(
    device const float* a_in        [[buffer(0)]],  // [5, nc]
    device const float* v_in        [[buffer(1)]],  // [5, nc]
    device const float* m_target_in [[buffer(2)]],  // [nc]
    device const float* m_prev_in   [[buffer(3)]],  // [nc]
    device const float* is_ocn_in   [[buffer(4)]],  // [nc]
    device float* a_out             [[buffer(5)]],  // [5, nc]
    device float* v_out             [[buffer(6)]],  // [5, nc]
    constant ulong4& u_dim          [[buffer(7)]],  // [n_cat=5, nc, 0, 0]
    constant float4& f_p0           [[buffer(8)]],  // [inv_rho_seaice, 0, 0, 0]
    uint c                          [[thread_position_in_grid]])
{
    uint nc = uint(u_dim.y);
    if (c >= nc) return;

    float ocn = is_ocn_in[c];
    if (ocn < 0.5f) {
        for (uint n = 0u; n < 5u; ++n) {
            a_out[n * nc + c] = 0.0f;
            v_out[n * nc + c] = 0.0f;
        }
        return;
    }

    float inv_rho = f_p0.x;
    float v_target = max(m_target_in[c] * inv_rho, 0.0f);
    float a_c[5];
    float v_c[5];
    float a_tot = 0.0f;
    float v_cur = 0.0f;
    for (uint n = 0u; n < 5u; ++n) {
        float an = max(a_in[n * nc + c], 0.0f);
        float vn = max(v_in[n * nc + c], 0.0f);
        a_c[n] = an;
        v_c[n] = vn;
        a_tot += an;
        v_cur += vn;
    }

    float dv = v_target - v_cur;
    float a_tot_c = clamp(a_tot, 0.0f, 1.0f);
    bool has_ice = (v_cur > 1.0e-6f);
    float inv_at = 1.0f / max(a_tot_c, 1.0e-6f);
    float inv_vc = 1.0f / max(v_cur, 1.0e-6f);
    float dv_pos = max(dv, 0.0f);
    float dv_neg = min(dv, 0.0f);

    float a_n[5];
    float v_n[5];
    for (uint n = 0u; n < 5u; ++n) {
        float wg = has_ice ? (a_c[n] * inv_at) : ((n == 0u) ? 1.0f : 0.0f);
        float wm = has_ice ? (v_c[n] * inv_vc) : 0.0f;
        float vn = max(v_c[n] + dv_pos * wg + dv_neg * wm, 0.0f);
        float an = a_c[n];
        if (n == 0u && (!has_ice) && (v_target > 1.0e-6f)) {
            an = clamp(vn / 0.25f, 0.0f, 1.0f);
        }
        if (dv < 0.0f) {
            float mr = (v_c[n] > 1.0e-7f) ? clamp(vn / max(v_c[n], 1.0e-7f), 0.0f, 1.0f) : 1.0f;
            an *= mr;
        }
        if (vn <= 1.0e-8f) {
            an = 0.0f;
        }
        a_n[n] = an;
        v_n[n] = vn;
    }

    // Rebin across ITD_BOUNDS_M = {0.0, 0.60, 1.40, 2.40, 3.60}
    float ao[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float vo[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint n = 0u; n < 5u; ++n) {
        if (a_n[n] > 1.0e-7f) {
            float hn = v_n[n] / max(a_n[n], 1.0e-6f);
            uint tgt = 0u;
            if (hn >= 0.60f) tgt = 1u;
            if (hn >= 1.40f) tgt = 2u;
            if (hn >= 2.40f) tgt = 3u;
            if (hn >= 3.60f) tgt = 4u;
            ao[tgt] += a_n[n];
            vo[tgt] += v_n[n];
        }
    }

    // Mechanical ridging enforcing sum(ao) <= 1.0 while conserving volume
    float a_sum = ao[0] + ao[1] + ao[2] + ao[3] + ao[4];
    if (a_sum > 1.000001f) {
        for (uint n = 0u; n < 4u; ++n) {
            float rem = max(ao[0] + ao[1] + ao[2] + ao[3] + ao[4] - 1.0f, 0.0f);
            float take_a = min(ao[n], rem);
            float frac = take_a / max(ao[n], 1.0e-7f);
            float move_v = vo[n] * frac;
            ao[n] -= take_a;
            vo[n] -= move_v;
            ao[n + 1u] += 0.5f * take_a;
            vo[n + 1u] += move_v;
        }
        float a_sum2 = ao[0] + ao[1] + ao[2] + ao[3] + ao[4];
        if (a_sum2 > 1.0f) {
            float inv_s = 1.0f / a_sum2;
            for (uint n = 0u; n < 5u; ++n) ao[n] *= inv_s;
        }
    }

    float v_sum = vo[0] + vo[1] + vo[2] + vo[3] + vo[4];
    float sv = (v_sum > 1.0e-8f) ? (v_target / max(v_sum, 1.0e-8f)) : 1.0f;
    for (uint n = 0u; n < 5u; ++n) {
        a_out[n * nc + c] = clamp(ao[n], 0.0f, 1.0f);
        v_out[n * nc + c] = vo[n] * sv;
    }
}

kernel void seaice_evp_stress_cell(
    device const float* u_cur       [[buffer(0)]],  // [ne]
    device const float* p_strength  [[buffer(1)]],  // [nc]
    device const float* is_ocn_f    [[buffer(2)]],  // [nc]
    device const int* eoc           [[buffer(3)]],  // [nc, max_eoc]
    device const float* div_w       [[buffer(4)]],  // [nc, max_eoc]
    device const float* inv_dc_e    [[buffer(5)]],  // [ne]
    device float* s1_io             [[buffer(6)]],  // [nc]
    device float* ss_io             [[buffer(7)]],  // [nc]
    constant ulong4& u_dim          [[buffer(8)]],  // [nc, max_eoc, 0, 0]
    constant float4& f_p0           [[buffer(9)]],  // [alpha_e, inv_1pa, e2_inv, delta_min]
    uint c                          [[thread_position_in_grid]])
{
    uint nc = uint(u_dim.x);
    uint max_eoc = uint(u_dim.y);
    if (c >= nc) return;

    float ocn = is_ocn_f[c];
    if (ocn < 0.5f) {
        s1_io[c] = 0.0f;
        ss_io[c] = 0.0f;
        return;
    }

    uint base = c * max_eoc;
    float div_u = 0.0f;
    float u_abs[8];
    float sum_u = 0.0f;
    float sum_inv_dc = 0.0f;
    uint m_lim = min(max_eoc, 8u);
    for (uint m = 0u; m < m_lim; ++m) {
        uint e = uint(eoc[base + m]);
        float ue = u_cur[e];
        div_u = fma(ue, div_w[base + m], div_u);
        float ua = fabs(ue);
        u_abs[m] = ua;
        sum_u += ua;
        sum_inv_dc += inv_dc_e[e];
    }
    float inv_m = 1.0f / float(m_lim);
    float u_mean = sum_u * inv_m;
    float dev_sum = 0.0f;
    for (uint m = 0u; m < m_lim; ++m) {
        dev_sum += fabs(u_abs[m] - u_mean);
    }
    float shear_est = (dev_sum * inv_m) * (sum_inv_dc * inv_m);

    float alpha_e = f_p0.x;
    float inv_1pa = f_p0.y;
    float e2_inv = f_p0.z;
    float dmin = f_p0.w;
    float delta = sqrt(div_u * div_u + e2_inv * shear_est * shear_est + dmin * dmin);

    float pstr = p_strength[c];
    float zeta = pstr / (2.0f * delta);
    float eta_v = zeta * e2_inv;

    float s1_raw = (s1_io[c] + alpha_e * (zeta * (div_u - delta))) * inv_1pa;
    float ss_raw = (ss_io[c] + alpha_e * (eta_v * shear_est)) * inv_1pa;
    s1_io[c] = min(max(s1_raw, -2.0f * pstr - 1.0f), 0.0f);
    ss_io[c] = max(min(ss_raw, pstr + 1.0f), 0.0f);
}

kernel void seaice_evp_vel_s1_edge(
    device const float* u_cur       [[buffer(0)]],  // [ne]
    device const float* s1          [[buffer(1)]],  // [nc]
    device const float* m_ice_e     [[buffer(2)]],  // [ne]
    device const float* a_ice_e     [[buffer(3)]],  // [ne]
    device const float* tau_air_e   [[buffer(4)]],  // [ne]
    device const float* u_ocn_e     [[buffer(5)]],  // [ne]
    device const float* grad_eta_e  [[buffer(6)]],  // [ne]
    device const float* has_ice_e   [[buffer(7)]],  // [ne]
    device const float* inv_dc_e    [[buffer(8)]],  // [ne]
    device const int* c0            [[buffer(9)]],  // [ne]
    device const int* c1            [[buffer(10)]], // [ne]
    device const int* ep_idx        [[buffer(11)]], // [ne, max_trisk]
    device const float* cor_w       [[buffer(12)]], // [ne, max_trisk]
    device float* u_half_out        [[buffer(13)]], // [ne]
    device float* f_ext_out         [[buffer(14)]], // [ne]
    constant ulong4& u_dim          [[buffer(15)]], // [ne, max_trisk, 0, 0]
    constant float4& f_p0           [[buffer(16)]], // [dt_e, u_cap, rho_sw_cw, gravity]
    uint e                          [[thread_position_in_grid]])
{
    uint ne = uint(u_dim.x);
    uint max_tr = uint(u_dim.y);
    if (e >= ne) return;

    float hi = has_ice_e[e];
    if (hi < 0.5f) {
        u_half_out[e] = 0.0f;
        f_ext_out[e] = 0.0f;
        return;
    }

    float dt_e = f_p0.x;
    float u_cap = f_p0.y;
    float rho_cw = f_p0.z;
    float grav = f_p0.w;

    float mi = m_ice_e[e];
    float m_safe = max(mi, 10.0f);
    float ai = a_ice_e[e];
    float uo = u_ocn_e[e];
    float uc = u_cur[e];

    float f_int = 0.5f * ((s1[uint(c1[e])] - s1[uint(c0[e])]) * inv_dc_e[e]);
    float f_ext = ai * tau_air_e[e] + f_int - mi * grav * grad_eta_e[e];
    f_ext_out[e] = f_ext;

    float c_drag = ai * rho_cw * (fabs(uc - uo) + 0.05f);
    uint base = e * max_tr;
    float cor_sum = 0.0f;
    for (uint m = 0u; m < max_tr; ++m) {
        cor_sum = fma(cor_w[base + m], u_cur[uint(ep_idx[base + m])], cor_sum);
    }
    float cor_0 = cor_sum * mi;
    float hdt = 0.5f * dt_e;
    float uh = (m_safe * uc + hdt * (f_ext + cor_0 + c_drag * uo)) / (m_safe + hdt * c_drag);
    u_half_out[e] = clamp(uh, -u_cap, u_cap);
}

kernel void seaice_evp_vel_s2_edge(
    device float* u_cur_io          [[buffer(0)]],  // [ne]
    device const float* u_half      [[buffer(1)]],  // [ne]
    device const float* f_ext_in    [[buffer(2)]],  // [ne]
    device const float* m_ice_e     [[buffer(3)]],  // [ne]
    device const float* a_ice_e     [[buffer(4)]],  // [ne]
    device const float* u_ocn_e     [[buffer(5)]],  // [ne]
    device const float* has_ice_e   [[buffer(6)]],  // [ne]
    device const int* ep_idx        [[buffer(7)]],  // [ne, max_trisk]
    device const float* cor_w       [[buffer(8)]],  // [ne, max_trisk]
    constant ulong4& u_dim          [[buffer(9)]],  // [ne, max_trisk, 0, 0]
    constant float4& f_p0           [[buffer(10)]], // [dt_e, u_cap, rho_sw_cw, 0]
    uint e                          [[thread_position_in_grid]])
{
    uint ne = uint(u_dim.x);
    uint max_tr = uint(u_dim.y);
    if (e >= ne) return;

    if (has_ice_e[e] < 0.5f) {
        u_cur_io[e] = 0.0f;
        return;
    }

    float dt_e = f_p0.x;
    float u_cap = f_p0.y;
    float rho_cw = f_p0.z;

    float mi = m_ice_e[e];
    float m_safe = max(mi, 10.0f);
    float ai = a_ice_e[e];
    float uo = u_ocn_e[e];
    float uc = u_cur_io[e];
    float uh = u_half[e];

    float c_drag = ai * rho_cw * (fabs(uh - uo) + 0.05f);
    uint base = e * max_tr;
    float cor_sum = 0.0f;
    for (uint m = 0u; m < max_tr; ++m) {
        cor_sum = fma(cor_w[base + m], u_half[uint(ep_idx[base + m])], cor_sum);
    }
    float cor_h = cor_sum * mi;
    float rhs = m_safe * uc + dt_e * (f_ext_in[e] + cor_h + c_drag * uo);
    u_cur_io[e] = clamp(rhs / (m_safe + dt_e * c_drag), -u_cap, u_cap);
}

// =========================================================================
// 21. OGCM 2D Barotropic RK3 Subcycle Kernels
// =========================================================================

kernel void ogcm_bt_vel_edge(
    device const float* u0_bt       [[buffer(0)]],  // [ne]
    device const float* eta_bt      [[buffer(1)]],  // [nc]
    device const float* f_baro      [[buffer(2)]],  // [ne]
    device const float* g_inv_dc    [[buffer(3)]],  // [ne]
    device const float* h_col_e     [[buffer(4)]],  // [ne]
    device const float* ocn_e0f     [[buffer(5)]],  // [ne]
    device const int* c0            [[buffer(6)]],  // [ne]
    device const int* c1            [[buffer(7)]],  // [ne]
    device float* u_out             [[buffer(8)]],  // [ne]
    device float* flux_out          [[buffer(9)]],  // [ne]
    constant ulong& ne_u            [[buffer(10)]],
    constant float4& f_dt           [[buffer(11)]],  // [rk_dt, 1 / (1 + rk_dt * r_bt_pert), 0, 0]
    uint e                          [[thread_position_in_grid]])
{
    if (e >= uint(ne_u)) return;
    float m = ocn_e0f[e];
    float pgf = g_inv_dc[e] * (eta_bt[uint(c1[e])] - eta_bt[uint(c0[e])]);
    float u_st = clamp(((u0_bt[e] + f_dt.x * (pgf + f_baro[e])) * f_dt.y) * m, -2.5f, 2.5f);
    u_out[e] = u_st;
    flux_out[e] = h_col_e[e] * u_st;
}

kernel void ogcm_bt_eta_cell(
    device const float* eta0_bt     [[buffer(0)]],  // [nc]
    device const float* flux_bt     [[buffer(1)]],  // [ne]
    device const float* is_ocn_f    [[buffer(2)]],  // [nc]
    device const int* eoc           [[buffer(3)]],  // [nc, max_eoc]
    device const float* div_w       [[buffer(4)]],  // [nc, max_eoc]
    device float* eta_out           [[buffer(5)]],  // [nc]
    constant ulong4& u_dim          [[buffer(6)]],  // [nc, max_eoc, 0, 0]
    constant float& rk_dt           [[buffer(7)]],
    uint c                          [[thread_position_in_grid]])
{
    uint nc = uint(u_dim.x);
    uint max_eoc = uint(u_dim.y);
    if (c >= nc) return;
    float ocn = is_ocn_f[c];
    if (ocn < 0.5f) {
        eta_out[c] = 0.0f;
        return;
    }
    uint base = c * max_eoc;
    float div_f = 0.0f;
    for (uint m = 0u; m < max_eoc; ++m) {
        div_f = fma(flux_bt[uint(eoc[base + m])], div_w[base + m], div_f);
    }
    eta_out[c] = clamp(eta0_bt[c] - rk_dt * div_f, -5.0f, 5.0f);
}

// =========================================================================
// 21b. OGCM 3D Batched Zalesak FCT + z* + Full Redi + Kahan Kernels
// =========================================================================

kernel void ogcm_fct_edge(
    device const int* c0_arr        [[buffer(0)]],   // [ne]
    device const int* c1_arr        [[buffer(1)]],   // [ne]
    device const float* inv_dc      [[buffer(2)]],   // [ne]
    device const float* w_vor_edge  [[buffer(3)]],   // [ne]
    device const float* dz_c_int    [[buffer(4)]],   // [nzo-1, nc]
    device const float* qp          [[buffer(5)]],   // [B, nzo, nc]
    device const float* h_u_edge    [[buffer(6)]],   // [nzo, ne]
    device const float* h_e         [[buffer(7)]],   // [nzo, ne]
    device const float* wet_ef      [[buffer(8)]],   // [nzo, ne]
    device const float* wet_int_cf  [[buffer(9)]],   // [nzo-1, nc]
    device const float* wet_int_ef  [[buffer(10)]],  // [nzo-1, ne]
    device const float* sx_int      [[buffer(11)]],  // [nzo-1, ne]
    device float* fl_h_out          [[buffer(12)]],  // [B, nzo, ne]
    device float* ah_e_out          [[buffer(13)]],  // [B, nzo, ne]
    device float* f_dif_out         [[buffer(14)]],  // [B, nzo, ne]
    device float* s_dqdx_ew_out     [[buffer(15)]],  // [B, nzo-1, ne]
    constant ulong4& u_dim          [[buffer(16)]],  // [B, nzo, nc, ne]
    constant float4& f_p            [[buffer(17)]],  // [kappa_redi, enable_redi, 0, 0]
    uint gid [[thread_position_in_grid]])
{
    uint B = uint(u_dim.x);
    uint nzo = uint(u_dim.y);
    uint nc = uint(u_dim.z);
    uint ne = uint(u_dim.w);
    uint slice_e = nzo * ne;
    if (gid >= B * slice_e) return;

    uint b = gid / slice_e;
    uint loc = gid - b * slice_e;
    uint k = loc / ne;
    uint e = loc - k * ne;

    float we = wet_ef[loc];
    bool full_redi = (f_p.y > 0.5f);
    if (we < 0.5f) {
        fl_h_out[gid] = 0.0f;
        ah_e_out[gid] = 0.0f;
        f_dif_out[gid] = 0.0f;
        if (full_redi && k + 1u < nzo) {
            s_dqdx_ew_out[b * ((nzo - 1u) * ne) + k * ne + e] = 0.0f;
        }
        return;
    }

    uint c0 = uint(c0_arr[e]);
    uint c1 = uint(c1_arr[e]);
    uint q_b = b * (nzo * nc);
    uint q_k = q_b + k * nc;
    float q0 = qp[q_k + c0];
    float q1 = qp[q_k + c1];

    float hu = h_u_edge[loc];
    float fl = (max(hu, 0.0f) * q0 + min(hu, 0.0f) * q1) * we;
    float fh = (0.5f * hu * (q0 + q1)) * we;
    fl_h_out[gid] = fl;
    ah_e_out[gid] = (fh - fl) * we;

    float inv_dc_e = inv_dc[e];
    float dq_dx = ((q1 - q0) * inv_dc_e) * we;
    float kappa = f_p.x;

    if (full_redi) {
        float sx_dqdz_lo = 0.0f;
        float sx_dqdz_hi = 0.0f;
        if (k + 1u < nzo) {
            uint k_int = k * ne + e;
            float wie = wet_int_ef[k_int];
            uint q_kp1 = q_b + (k + 1u) * nc;
            uint wc_int = k * nc;
            float dqdz0 = ((qp[q_kp1 + c0] - q0) / dz_c_int[wc_int + c0]) * wet_int_cf[wc_int + c0];
            float dqdz1 = ((qp[q_kp1 + c1] - q1) / dz_c_int[wc_int + c1]) * wet_int_cf[wc_int + c1];
            float dqdz_e = 0.5f * (dqdz0 + dqdz1) * wie;
            float sx_k = sx_int[k_int];
            sx_dqdz_hi = sx_k * dqdz_e;

            // Cross-slope vertical flux term on interface k+1/2
            float we_p1 = wet_ef[(k + 1u) * ne + e];
            float dq_dx_p1 = ((qp[q_kp1 + c1] - qp[q_kp1 + c0]) * inv_dc_e) * we_p1;
            float dq_dx_int = 0.5f * (dq_dx + dq_dx_p1) * wie;
            s_dqdx_ew_out[b * ((nzo - 1u) * ne) + k_int] = (sx_k * dq_dx_int) * w_vor_edge[e];
        }
        if (k > 0u) {
            uint km1 = k - 1u;
            uint km1_int = km1 * ne + e;
            float wie = wet_int_ef[km1_int];
            uint q_km1 = q_b + km1 * nc;
            uint wc_int = km1 * nc;
            float dqdz0 = ((q0 - qp[q_km1 + c0]) / dz_c_int[wc_int + c0]) * wet_int_cf[wc_int + c0];
            float dqdz1 = ((q1 - qp[q_km1 + c1]) / dz_c_int[wc_int + c1]) * wet_int_cf[wc_int + c1];
            float dqdz_e = 0.5f * (dqdz0 + dqdz1) * wie;
            sx_dqdz_lo = sx_int[km1_int] * dqdz_e;
        }
        float sx_dqdz_lay = 0.5f * (sx_dqdz_hi + sx_dqdz_lo);
        f_dif_out[gid] = (-kappa * h_e[loc] * (dq_dx + sx_dqdz_lay)) * we;
    } else {
        f_dif_out[gid] = (-kappa * h_e[loc] * dq_dx) * we;
    }
}

kernel void ogcm_fct_cell_td(
    device const int* eoc           [[buffer(0)]],   // [nc, 6]
    device const float* div_w       [[buffer(1)]],   // [nc, 6]
    device const float* qp          [[buffer(2)]],   // [B, nzo, nc]
    device const float* h_old_z     [[buffer(3)]],   // [nzo, nc]
    device const float* h_new_z     [[buffer(4)]],   // [nzo, nc]
    device const float* fl_h        [[buffer(5)]],   // [B, nzo, ne]
    device const float* w_star      [[buffer(6)]],   // [nzo+1, nc]
    device const float* wet_cf      [[buffer(7)]],   // [nzo, nc]
    device const float* wet_int_cf  [[buffer(8)]],   // [nzo-1, nc]
    device float* qp_td_out         [[buffer(9)]],   // [B, nzo, nc]
    device float* fl_v_out          [[buffer(10)]],  // [B, nzo-1, nc]
    device float* av_int_out        [[buffer(11)]],  // [B, nzo-1, nc]
    constant ulong4& u_dim          [[buffer(12)]],  // [B, nzo, nc, ne]
    constant float& dt              [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    uint B = uint(u_dim.x);
    uint nzo = uint(u_dim.y);
    uint nc = uint(u_dim.z);
    uint ne = uint(u_dim.w);
    uint vol = nzo * nc;
    if (gid >= B * vol) return;

    uint b = gid / vol;
    uint loc = gid - b * vol;
    uint k = loc / nc;
    uint c = loc - k * nc;
    float wc = wet_cf[loc];
    if (wc < 0.5f) {
        qp_td_out[gid] = 0.0f;
        if (k + 1u < nzo) {
            uint vi = b * ((nzo - 1u) * nc) + k * nc + c;
            fl_v_out[vi] = 0.0f;
            av_int_out[vi] = 0.0f;
        }
        return;
    }

    uint c6 = c * 6u;
    uint ebase = b * (nzo * ne) + k * ne;

    float div_fl_h = 0.0f;
    for (uint d = 0u; d < 6u; ++d) {
        div_fl_h = fma(div_w[c6 + d], fl_h[ebase + uint(eoc[c6 + d])], div_fl_h);
    }
    div_fl_h *= wc;

    float q0 = qp[gid];
    float fl_below = 0.0f;
    float fl_above = 0.0f;
    if (k + 1u < nzo) {
        uint wi_idx = k * nc + c;
        float w_b = w_star[(k + 1u) * nc + c];
        float wic = wet_int_cf[wi_idx];
        float q_p1 = qp[gid + nc];
        fl_below = ((w_b >= 0.0f) ? (w_b * q_p1) : (w_b * q0)) * wic;
        float fh_below = (0.5f * w_b * (q0 + q_p1)) * wic;
        uint vi = b * ((nzo - 1u) * nc) + wi_idx;
        fl_v_out[vi] = fl_below;
        av_int_out[vi] = (fh_below - fl_below) * wic;
    }
    if (k > 0u) {
        uint km1 = k - 1u;
        float w_a = w_star[k * nc + c];
        float wic = wet_int_cf[km1 * nc + c];
        float q_m1 = qp[gid - nc];
        fl_above = ((w_a >= 0.0f) ? (w_a * q0) : (w_a * q_m1)) * wic;
    }

    float h_old_s = max(h_old_z[loc], 1.0f);
    float h_new_s = max(h_new_z[loc], 1.0f);
    float vin_fl = fl_below - fl_above;
    qp_td_out[gid] = ((h_old_s * q0 + dt * (-div_fl_h + vin_fl)) / h_new_s) * wc;
}

kernel void ogcm_fct_bounds_ratios(
    device const int* c0_arr        [[buffer(0)]],   // [ne]
    device const int* c1_arr        [[buffer(1)]],   // [ne]
    device const int* eoc           [[buffer(2)]],   // [nc, 6]
    device const float* div_w       [[buffer(3)]],   // [nc, 6]
    device const float* qp          [[buffer(4)]],   // [B, nzo, nc]
    device const float* qp_td       [[buffer(5)]],   // [B, nzo, nc]
    device const float* h_new_z     [[buffer(6)]],   // [nzo, nc]
    device const float* ah_e        [[buffer(7)]],   // [B, nzo, ne]
    device const float* av_int      [[buffer(8)]],   // [B, nzo-1, nc]
    device const float* wet_cf      [[buffer(9)]],   // [nzo, nc]
    device const float* wet_ef      [[buffer(10)]],  // [nzo, ne]
    device const float* wet_int_cf  [[buffer(11)]],  // [nzo-1, nc]
    device float* r_plus_out        [[buffer(12)]],  // [B, nzo, nc]
    device float* r_minus_out       [[buffer(13)]],  // [B, nzo, nc]
    constant ulong4& u_dim          [[buffer(14)]],  // [B, nzo, nc, ne]
    constant float& dt              [[buffer(15)]],
    uint gid [[thread_position_in_grid]])
{
    uint B = uint(u_dim.x);
    uint nzo = uint(u_dim.y);
    uint nc = uint(u_dim.z);
    uint ne = uint(u_dim.w);
    uint vol = nzo * nc;
    if (gid >= B * vol) return;

    uint b = gid / vol;
    uint loc = gid - b * vol;
    uint k = loc / nc;
    uint c = loc - k * nc;
    float wc = wet_cf[loc];
    if (wc < 0.5f) {
        r_plus_out[gid] = 1.0f;
        r_minus_out[gid] = 1.0f;
        return;
    }

    uint c6 = c * 6u;
    uint qbase = b * vol + k * nc;
    uint ebase = b * (nzo * ne) + k * ne;

    float q0 = qp[gid];
    float qtd0 = qp_td[gid];
    float q_max = max(q0, qtd0);
    float q_min = min(q0, qtd0);

    float p_plus_h = 0.0f;
    float p_minus_h = 0.0f;
    for (uint d = 0u; d < 6u; ++d) {
        uint eid = uint(eoc[c6 + d]);
        if (wet_ef[k * ne + eid] > 0.5f) {
            uint c0i = uint(c0_arr[eid]);
            uint c1i = uint(c1_arr[eid]);
            uint nb = (c0i == c) ? c1i : c0i;
            float qn = qp[qbase + nb];
            float qtn = qp_td[qbase + nb];
            q_max = max(q_max, max(qn, qtn));
            q_min = min(q_min, min(qn, qtn));
        }
        float a_in = -(div_w[c6 + d] * ah_e[ebase + eid]);
        p_plus_h  += max(a_in, 0.0f);
        p_minus_h += max(-a_in, 0.0f);
    }

    float p_plus_v = 0.0f;
    float p_minus_v = 0.0f;
    uint vi_base = b * ((nzo - 1u) * nc);
    if (k + 1u < nzo) {
        if (wet_int_cf[k * nc + c] > 0.5f) {
            float q_dn = qp[gid + nc];
            float qt_dn = qp_td[gid + nc];
            q_max = max(q_max, max(q_dn, qt_dn));
            q_min = min(q_min, min(q_dn, qt_dn));
        }
        float av_b = av_int[vi_base + k * nc + c];
        p_plus_v  += max(av_b, 0.0f);
        p_minus_v += max(-av_b, 0.0f);
    }
    if (k > 0u) {
        uint km1 = k - 1u;
        if (wet_int_cf[km1 * nc + c] > 0.5f) {
            float q_up = qp[gid - nc];
            float qt_up = qp_td[gid - nc];
            q_max = max(q_max, max(q_up, qt_up));
            q_min = min(q_min, min(q_up, qt_up));
        }
        float av_a = av_int[vi_base + km1 * nc + c];
        p_plus_v  += max(-av_a, 0.0f);
        p_minus_v += max(av_a, 0.0f);
    }

    float p_plus  = dt * (p_plus_h  + p_plus_v)  * wc;
    float p_minus = dt * (p_minus_h + p_minus_v) * wc;
    float h_new_s = max(h_new_z[loc], 1.0f);
    float q_plus  = h_new_s * max(q_max - qtd0, 0.0f);
    float q_minus = h_new_s * max(qtd0 - q_min, 0.0f);
    float eps = 1.0e-12f;

    r_plus_out[gid]  = (p_plus  > eps) ? clamp(q_plus  / (p_plus  + eps), 0.0f, 1.0f) : 1.0f;
    r_minus_out[gid] = (p_minus > eps) ? clamp(q_minus / (p_minus + eps), 0.0f, 1.0f) : 1.0f;
}

kernel void ogcm_fct_limit_kahan(
    device const int* c0_arr        [[buffer(0)]],   // [ne]
    device const int* c1_arr        [[buffer(1)]],   // [ne]
    device const int* eoc           [[buffer(2)]],   // [nc, 6]
    device const float* div_w       [[buffer(3)]],   // [nc, 6]
    device const float* active_w    [[buffer(4)]],   // [nc, 6] active_eoc * inv_area_c
    device const float* q_in        [[buffer(5)]],   // [B, nzo, nc]
    device const float* q_comp_in   [[buffer(6)]],   // [B, nzo, nc]
    device const float* qp          [[buffer(7)]],   // [B, nzo, nc]
    device const float* f_nl_q      [[buffer(8)]],   // [B, nzo-1, nc]
    device const float* h_old_z     [[buffer(9)]],   // [nzo, nc]
    device const float* h_new_z     [[buffer(10)]],  // [nzo, nc]
    device const float* fl_h        [[buffer(11)]],  // [B, nzo, ne]
    device const float* ah_e        [[buffer(12)]],  // [B, nzo, ne]
    device const float* f_dif       [[buffer(13)]],  // [B, nzo, ne]
    device const float* s_dqdx_ew   [[buffer(14)]],  // [B, nzo-1, ne]
    device const float* fl_v        [[buffer(15)]],  // [B, nzo-1, nc]
    device const float* av_int      [[buffer(16)]],  // [B, nzo-1, nc]
    device const float* r_plus      [[buffer(17)]],  // [B, nzo, nc]
    device const float* r_minus     [[buffer(18)]],  // [B, nzo, nc]
    device const float* wet_cf      [[buffer(19)]],  // [nzo, nc]
    device const float* wet_ef      [[buffer(20)]],  // [nzo, ne]
    device const float* wet_int_cf  [[buffer(21)]],  // [nzo-1, nc]
    device float* q_out             [[buffer(22)]],  // [B, nzo, nc]
    device float* q_comp_out        [[buffer(23)]],  // [B, nzo, nc]
    constant ulong4& u_dim          [[buffer(24)]],  // [B, nzo, nc, ne]
    constant float4& f_p            [[buffer(25)]],  // [dt, kappa_redi, enable_redi, 0]
    uint gid [[thread_position_in_grid]])
{
    uint B = uint(u_dim.x);
    uint nzo = uint(u_dim.y);
    uint nc = uint(u_dim.z);
    uint ne = uint(u_dim.w);
    uint vol = nzo * nc;
    if (gid >= B * vol) return;

    uint b = gid / vol;
    uint loc = gid - b * vol;
    uint k = loc / nc;
    uint c = loc - k * nc;
    float wc = wet_cf[loc];
    if (wc < 0.5f) {
        float q_old = q_in[gid];
        float y = q_comp_in[gid];
        float t = q_old + y;
        q_out[gid] = t;
        q_comp_out[gid] = (fabs(q_old) >= fabs(y)) ? ((q_old - t) + y) : ((y - t) + q_old);
        return;
    }

    uint c6 = c * 6u;
    uint qbase = b * vol + k * nc;
    uint ebase = b * (nzo * ne) + k * ne;

    float dt = f_p.x;
    float kappa = f_p.y;
    bool full_redi = (f_p.z > 0.5f);
    float inv_dt_safe = 1.0f / max(dt, 1.0f);

    float div_h = 0.0f;
    for (uint d = 0u; d < 6u; ++d) {
        float w = div_w[c6 + d];
        if (w != 0.0f) {
            uint eid = uint(eoc[c6 + d]);
            uint eidx = ebase + eid;
            float a_e = ah_e[eidx];
            uint c0i = uint(c0_arr[eid]);
            uint c1i = uint(c1_arr[eid]);
            float c_lim = ((a_e >= 0.0f)
                ? min(r_minus[qbase + c0i], r_plus[qbase + c1i])
                : min(r_plus[qbase + c0i], r_minus[qbase + c1i])) * wet_ef[k * ne + eid];
            float f_tot = fl_h[eidx] + c_lim * a_e + f_dif[eidx];
            div_h = fma(w, f_tot, div_h);
        }
    }
    div_h *= wc;

    float rp0 = r_plus[gid];
    float rm0 = r_minus[gid];
    float qp0 = qp[gid];
    float h_new_0 = max(h_new_z[loc], 1.0f);
    uint vi_base = b * ((nzo - 1u) * nc);
    uint se_base = b * ((nzo - 1u) * ne);

    float fv_below = 0.0f;
    if (k + 1u < nzo) {
        uint wi_idx = k * nc + c;
        float wic = wet_int_cf[wi_idx];
        uint vi = vi_base + wi_idx;
        float av_b = av_int[vi];
        float cv_b = ((av_b >= 0.0f) ? min(r_minus[gid + nc], rp0) : min(r_plus[gid + nc], rm0)) * wic;
        float f_adv_v = fl_v[vi] + cv_b * av_b;

        float f_redi_b = 0.0f;
        if (full_redi && wic > 0.5f) {
            uint se_k = se_base + k * ne;
            float sdg = 0.0f;
            for (uint d = 0u; d < 6u; ++d) {
                sdg = fma(active_w[c6 + d], s_dqdx_ew[se_k + uint(eoc[c6 + d])], sdg);
            }
            f_redi_b = kappa * sdg * wic;
        }
        float h_min_b = min(h_new_0, max(h_new_z[(k + 1u) * nc + c], 1.0f));
        float q_span_b = fabs(qp[gid + nc] - qp0) + 0.5f;
        float cap_b = 0.35f * h_min_b * q_span_b * inv_dt_safe;
        float fnl_b = clamp(f_nl_q[vi], -cap_b, cap_b) * wic;
        fv_below = f_adv_v + f_redi_b + fnl_b;
    }

    float fv_above = 0.0f;
    if (k > 0u) {
        uint km1 = k - 1u;
        uint wi_idx = km1 * nc + c;
        float wic = wet_int_cf[wi_idx];
        uint vi = vi_base + wi_idx;
        float av_a = av_int[vi];
        float cv_a = ((av_a >= 0.0f) ? min(rm0, r_plus[gid - nc]) : min(rp0, r_minus[gid - nc])) * wic;
        float f_adv_v = fl_v[vi] + cv_a * av_a;

        float f_redi_a = 0.0f;
        if (full_redi && wic > 0.5f) {
            uint se_km1 = se_base + km1 * ne;
            float sdg = 0.0f;
            for (uint d = 0u; d < 6u; ++d) {
                sdg = fma(active_w[c6 + d], s_dqdx_ew[se_km1 + uint(eoc[c6 + d])], sdg);
            }
            f_redi_a = kappa * sdg * wic;
        }
        float h_min_a = min(max(h_new_z[km1 * nc + c], 1.0f), h_new_0);
        float q_span_a = fabs(qp0 - qp[gid - nc]) + 0.5f;
        float cap_a = 0.35f * h_min_a * q_span_a * inv_dt_safe;
        float fnl_a = clamp(f_nl_q[vi], -cap_a, cap_a) * wic;
        fv_above = f_adv_v + f_redi_a + fnl_a;
    }

    float dh_z = h_new_z[loc] - h_old_z[loc];
    float v_inflow = fv_below - fv_above;
    float dq_fct = ((dt * (-div_h + v_inflow) - dh_z * qp0) / h_new_0) * wc;

    // Exact Neumaier / Kahan compensated summation
    float q_old = q_in[gid];
    float y = dq_fct + q_comp_in[gid];
    float t = q_old + y;
    float new_comp = (fabs(q_old) >= fabs(y)) ? ((q_old - t) + y) : ((y - t) + q_old);
    q_out[gid] = t;
    q_comp_out[gid] = new_comp;
}

// =========================================================================
// 22. Fused FlexPFT Marine BGC Column + Sinking + OMIP/mocsy 2.0 Carbonate
// =========================================================================

kernel void ocean_bgc_column_col(
    device float* no3_io            [[buffer(0)]],  // [nzo, nc]
    device float* phy_io            [[buffer(1)]],  // [nzo, nc]
    device float* zoo_io            [[buffer(2)]],  // [nzo, nc]
    device float* det_n_io          [[buffer(3)]],  // [nzo, nc]
    device float* det_c_io          [[buffer(4)]],  // [nzo, nc]
    device float* dic_io            [[buffer(5)]],  // [nzo, nc]
    device float* alk_io            [[buffer(6)]],  // [nzo, nc]
    device float* no3_comp_io       [[buffer(7)]],  // [nzo, nc]
    device float* dic_comp_io       [[buffer(8)]],  // [nzo, nc]
    device float* alk_comp_io       [[buffer(9)]],  // [nzo, nc]
    device const float* temp_k      [[buffer(10)]], // [nzo, nc]
    device const float* sal_psu     [[buffer(11)]], // [nzo, nc]
    device const float* h_c         [[buffer(12)]], // [nzo, nc]
    device const float* wet_cf      [[buffer(13)]], // [nzo, nc]
    device const float* sfc_in      [[buffer(14)]], // [4, nc]: h0_new, sw_par_sfc, u10_ocn, f_seaice
    device float* diag_3d           [[buffer(15)]], // [3, nzo, nc]: chl, npp_c, c_to_n
    device float* diag_2d           [[buffer(16)]], // [5, nc]: fgco2, spco2_uatm, ph_sfc, intpp, epc100
    constant ulong4& u_dim          [[buffer(17)]], // [nzo, nc, 0, 0]
    constant float4& f_p0           [[buffer(18)]], // [dt, co2_ppmv, rho_sw, 0]
    uint c                          [[thread_position_in_grid]])
{
    uint nzo = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    if (c >= nc || nzo == 0u || nzo > 64u) return;

    float wet0 = wet_cf[c];
    if (wet0 < 0.5f) {
        for (uint k = 0u; k < nzo; ++k) {
            uint idx = k * nc + c;
            diag_3d[idx] = 0.0f;
            diag_3d[nzo * nc + idx] = 0.0f;
            diag_3d[2u * nzo * nc + idx] = 0.0f;
        }
        for (uint d = 0u; d < 5u; ++d) {
            diag_2d[d * nc + c] = 0.0f;
        }
        return;
    }

    float dt = f_p0.x;
    float co2_ppmv = f_p0.y;
    float rho_sw = f_p0.z;
    float dt_day = dt / 86400.0f;
    float inv_dt = 1.0f / max(dt, 1.0f);
    float w_dt = (15.0f / 86400.0f) * dt;

    float h0_new = sfc_in[c];
    float sw_par = sfc_in[nc + c];
    float u10 = sfc_in[2u * nc + c];
    float f_ice = clamp(sfc_in[3u * nc + c], 0.0f, 1.0f);
    float i_sfc = max(sw_par, 0.0f) * (1.0f - f_ice);

    float tau_top = 0.0f;
    float f_in_n = 0.0f;
    float f_in_c = 0.0f;
    float intpp_acc = 0.0f;
    float epc_100 = 0.0f;

    float dic0_val = 0.0f;
    float dic0_cmp = 0.0f;
    float alk0_val = 0.0f;

    // Constants (Sasai et al., 2022; Aumont et al., 2015; Orr et al., 2017)
    const float i_thresh = (0.80f * 0.05f) / 2.0f;
    const float ea_over_r = 4.8e4f / 8.314462618f;
    const float inv_tref = 1.0f / 293.15f;
    const float r_cn_zoo = 106.0f / 16.0f;

    for (uint k = 0u; k < nzo; ++k) {
        uint idx = k * nc + c;
        float wk = wet_cf[idx];
        if (wk < 0.5f) {
            diag_3d[idx] = 0.0f;
            diag_3d[nzo * nc + idx] = 0.0f;
            diag_3d[2u * nzo * nc + idx] = 0.0f;
            continue;
        }
        bool is_bot = (k + 1u >= nzo) || (wet_cf[(k + 1u) * nc + c] < 0.5f);

        float hc_k = h_c[idx];
        float hk = (k == 0u) ? max(h0_new, 1.0f) : max(hc_k, 1.0f);

        float no3_v = no3_io[idx];
        float phy_v = phy_io[idx];
        float zoo_v = zoo_io[idx];
        float dn_v  = det_n_io[idx];
        float dc_v  = det_c_io[idx];
        float dic_v = dic_io[idx];
        float alk_v = alk_io[idx];
        float no3_c = no3_comp_io[idx];
        float dic_c = dic_comp_io[idx];
        float alk_c = alk_comp_io[idx];

        // 1. PAR attenuation with chlorophyll self-shading
        float chl_est = clamp(phy_v * 1.59f, 0.0f, 30.0f);
        float kd = 0.04f + 0.03f * chl_est;
        float tau_par = kd * hc_k;
        float i_top = i_sfc * exp(-min(tau_top, 40.0f));
        tau_top += tau_par;
        float att = (tau_par > 1.0e-4f)
            ? (_aoba_neg_expm1_neg(tau_par) / max(tau_par, 1.0e-4f))
            : (1.0f - 0.5f * tau_par + (1.0f / 6.0f) * tau_par * tau_par);
        float par_k = i_top * att;

        // 2. FlexPFT optimal quota Q, theta_hat, mu_fl
        float t_safe = clamp(temp_k[idx], 270.0f, 315.0f);
        float f_temp = exp(-ea_over_r * (1.0f / t_safe - inv_tref));
        float v0 = 1.4f * f_temp;

        float theta_hat = 0.0f;
        if (par_k > i_thresh) {
            float i_safe = max(par_k, i_thresh + 1.0e-6f);
            float x_light = (2.0f * i_safe) / (0.80f * v0);
            float r_maint = (0.05f * f_temp) / v0;
            float y_arg = log(1.0f + r_maint) + 1.0f + x_light;
            float w_val = _aoba_lambert_w0_exp(y_arg);
            float th_raw = (1.0f / 0.80f) + (v0 / (2.0f * i_safe)) * (1.0f - w_val);
            theta_hat = clamp(th_raw, 0.0f, 1.0f);
        }
        float arg_s = clamp((2.0f * theta_hat * par_k) / v0, 0.0f, 40.0f);
        float v_light = v0 * _aoba_neg_expm1_neg(arg_s);

        float n_pos = max(no3_v, 0.0f);
        float v0_n = 1.0f * f_temp;
        float a0_n = 1.0f * f_temp;
        float ratio_va = v0_n / max(a0_n, 1.0e-6f);
        float v_hat_n = (v0_n * n_pos) / (n_pos + ratio_va + 2.0f * sqrt(ratio_va * n_pos + 1.0e-12f) + 1.0e-12f);

        float v_n_safe = max(v_hat_n, 1.0e-9f);
        float b_cost = (v_light / v_n_safe) + 0.60f;
        float gamma_q = max(0.0195f * b_cost, 1.0e-5f);
        float q_quota = clamp(0.0195f * (1.0f + sqrt(1.0f + 1.0f / gamma_q)), 0.0195f * 1.05f, 0.25f);

        float f_v = 0.0f;
        if (v_light > 1.0e-9f && n_pos > 1.0e-9f) {
            float fv_raw = (1.0f - 0.0195f / q_quota) / (1.0f + (v_hat_n / max(v_light, 1.0e-9f)) * (0.60f + 1.0f / q_quota));
            f_v = clamp(fv_raw, 0.0f, 0.99f);
        }
        float mu_fl = (f_v * v_hat_n) / q_quota;
        float c_to_n = 1.0f / q_quota;

        // 3. Strictly conservative NPZD + flexible carbon reactions
        float p_pos = max(phy_v, 0.0f);
        float z_pos = max(zoo_v, 0.0f);
        float dn_pos = max(dn_v, 0.0f);
        float dc_pos = max(dc_v, 0.0f);

        float dn_up = min(mu_fl * p_pos * dt_day, 0.5f * n_pos);
        float dc_up = dn_up * c_to_n;

        float p2 = p_pos * p_pos;
        float g_rate = (1.0f * f_temp) * (p2 / (p2 + 0.25f)) * z_pos;
        float m_p_rate = (0.03f * f_temp) * p_pos + 0.05f * p2;
        float tot_p = (g_rate + m_p_rate) * dt_day;
        float sc_p = (tot_p > 0.8f * p_pos) ? ((0.8f * p_pos) / max(tot_p, 1.0e-12f)) : 1.0f;
        float dn_graze = (g_rate * dt_day) * sc_p;
        float dn_pmort = (m_p_rate * dt_day) * sc_p;

        float z_excr_r = (0.05f * f_temp) * z_pos;
        float z_mort_r = 0.15f * z_pos * z_pos;
        float tot_z = (z_excr_r + z_mort_r) * dt_day;
        float sc_z = (tot_z > 0.8f * z_pos) ? ((0.8f * z_pos) / max(tot_z, 1.0e-12f)) : 1.0f;
        float dn_zexcr = (z_excr_r * dt_day) * sc_z;
        float dn_zmort = (z_mort_r * dt_day) * sc_z;

        float f_remin = _aoba_neg_expm1_neg(min(0.12f * f_temp * dt_day, 20.0f));
        float dn_remin = f_remin * dn_pos;
        float dc_remin = f_remin * dc_pos;

        float d_no3 = -dn_up + dn_zexcr + dn_remin;
        float phy_new = max(phy_v + (dn_up - dn_graze - dn_pmort), 0.0f);
        float zoo_new = max(zoo_v + (0.70f * dn_graze - dn_zexcr - dn_zmort), 0.0f);
        float dn_mid = max(dn_v + (dn_pmort + 0.30f * dn_graze + dn_zmort - dn_remin), 0.0f);

        float dc_pmort = dn_pmort * c_to_n;
        float dc_fecal = 0.30f * dn_graze * c_to_n;
        float dc_zresp = 0.70f * dn_graze * (c_to_n - r_cn_zoo) + dn_zexcr * r_cn_zoo;
        float dc_zmort = dn_zmort * r_cn_zoo;
        float dc_mid = max(dc_v + (dc_pmort + dc_fecal + dc_zmort - dc_remin), 0.0f);

        float dc_caco3_p = 0.07f * (dc_pmort + dc_fecal);
        float dc_caco3_d = 0.07f * dc_remin;
        float d_dic = -dc_up + dc_zresp + dc_remin - dc_caco3_p + dc_caco3_d;
        float d_alk = -d_no3 - 2.0f * (dc_caco3_p - dc_caco3_d);

        // 4. Implicit top-down detritus sinking + immediate benthic remineralization
        float denom_s = hk + w_dt;
        float dn_k = (hk * dn_mid + f_in_n) / denom_s;
        float dc_k = (hk * dc_mid + f_in_c) / denom_s;
        float f_out_n = w_dt * dn_k;
        float f_out_c = w_dt * dc_k;

        if (k == 1u) {
            epc_100 = (f_out_c * 1.0e-3f) * inv_dt;
        }
        if (is_bot) {
            float bn = f_out_n / hk;
            float bc = f_out_c / hk;
            d_no3 += bn;
            d_dic += bc;
            d_alk -= bn;
            f_in_n = 0.0f;
            f_in_c = 0.0f;
        } else {
            f_in_n = f_out_n;
            f_in_c = f_out_c;
        }

        // Neumaier-Kahan compensated summation for NO3, DIC, Alk
        {
            float d_eff = d_no3 + no3_c;
            float t = no3_v + d_eff;
            no3_c = (fabs(no3_v) >= fabs(d_eff)) ? ((no3_v - t) + d_eff) : ((d_eff - t) + no3_v);
            no3_v = t;
        }
        {
            float d_eff = d_dic + dic_c;
            float t = dic_v + d_eff;
            dic_c = (fabs(dic_v) >= fabs(d_eff)) ? ((dic_v - t) + d_eff) : ((d_eff - t) + dic_v);
            dic_v = t;
        }
        {
            float d_eff = d_alk + alk_c;
            float t = alk_v + d_eff;
            alk_c = (fabs(alk_v) >= fabs(d_eff)) ? ((alk_v - t) + d_eff) : ((d_eff - t) + alk_v);
            alk_v = t;
        }

        no3_io[idx] = no3_v;
        no3_comp_io[idx] = no3_c;
        phy_io[idx] = phy_new;
        zoo_io[idx] = zoo_new;
        det_n_io[idx] = dn_k;
        det_c_io[idx] = dc_k;
        alk_io[idx] = alk_v;
        alk_comp_io[idx] = alk_c;

        if (k == 0u) {
            dic0_val = dic_v;
            dic0_cmp = dic_c;
            alk0_val = alk_v;
        } else {
            dic_io[idx] = dic_v;
            dic_comp_io[idx] = dic_c;
        }

        float npp_k = dc_up * inv_dt;
        diag_3d[idx] = phy_new * (theta_hat / q_quota);
        diag_3d[nzo * nc + idx] = npp_k;
        diag_3d[2u * nzo * nc + idx] = c_to_n;
        intpp_acc += npp_k * hk;
    }

    // 5. Surface OMIP / mocsy 2.0 Carbonate Equilibrium & Wanninkhof (2014) Air-Sea CO2 Flux
    float t_k = clamp(temp_k[c], 271.15f, 315.0f);
    float s = clamp(sal_psu[c], 5.0f, 45.0f);
    float inv_t = 1.0f / t_k;
    float ln_t = log(t_k);
    float sqrt_s = sqrt(s);
    float inv_rho_1000 = 1.0f / (1000.0f * rho_sw);
    float ct = clamp(dic0_val * inv_rho_1000, 1.0e-5f, 5.0e-3f);
    float at = clamp(alk0_val * inv_rho_1000, 1.0e-5f, 5.0e-3f);
    float bt = 4.157e-4f * (s / 35.0f);

    float pk1 = 3633.86f * inv_t - 61.2172f + 9.67770f * ln_t - 0.011555f * s + 1.152e-4f * s * s;
    float pk2 = 471.78f * inv_t + 25.9290f - 3.16967f * ln_t - 0.01781f * s + 1.122e-4f * s * s;
    float ln_kb = (-8966.90f - 2890.53f * sqrt_s - 77.942f * s + 1.728f * s * sqrt_s - 0.0996f * s * s) * inv_t
        + (148.0248f + 137.1942f * sqrt_s + 1.62142f * s)
        + (-24.4344f - 25.085f * sqrt_s - 0.2474f * s) * ln_t
        + 0.053105f * sqrt_s * t_k;
    float ln_kw = 148.9652f - 13847.26f * inv_t - 23.6521f * ln_t
        + (-5.977f + 118.67f * inv_t + 1.0495f * ln_t) * sqrt_s
        - 0.01615f * s;

    const float ln10 = 2.302585093f;
    float k1s = exp((8.0f - pk1) * ln10);
    float k2s = exp((8.0f - pk2) * ln10);
    float kbs = exp(ln_kb + 8.0f * ln10);
    float kws = exp(ln_kw + 16.0f * ln10);
    float k12s = k1s * k2s;

    float x = 0.794328f;
    for (int it = 0; it < 7; ++it) {
        float denom_c = x * x + k1s * x + k12s;
        float inv_dc = 1.0f / max(denom_c, 1.0e-12f);
        float carb_alk = ct * (k1s * x + 2.0f * k12s) * inv_dc;
        float bor_alk = bt * kbs / (x + kbs);
        float wat_alk = (kws / x - x) * 1.0e-8f;
        float f_val = carb_alk + bor_alk + wat_alk - at;

        float dcarb = ct * (k1s * denom_c - (k1s * x + 2.0f * k12s) * (2.0f * x + k1s)) * (inv_dc * inv_dc);
        float xkb = x + kbs;
        float dbor = -bt * kbs / (xkb * xkb);
        float dwat = (-kws / (x * x) - 1.0f) * 1.0e-8f;
        float df_dx = min(dcarb + dbor + dwat, -1.0e-12f);

        float st = clamp(f_val / df_dx, -0.5f * x, 0.5f * x);
        x = clamp(x - st, 1.0e-3f, 1.0e4f);
    }

    float denom_f = max(x * x + k1s * x + k12s, 1.0e-12f);
    float co2_star_kg = ct * (x * x) / denom_f;
    float co2_star_m3 = co2_star_kg * rho_sw;

    float t100 = t_k * 0.01f;
    float ln_k0 = -60.2409f + 93.4517f / t100 + 23.3585f * log(t100)
        + s * (0.023517f - 0.023656f * t100 + 0.0047036f * t100 * t100);
    float k0 = max(exp(ln_k0), 1.0e-6f);
    float pco2 = clamp((co2_star_kg / k0) * 1.0e6f, 20.0f, 4000.0f);
    float ph_tot = 8.0f - log(x) * 0.4342944819032518f;

    float ln_phi0 = -160.7333f + 215.4152f / t100 + 89.8920f * log(t100) - 1.47759f * t100 * t100
        + s * (0.029941f - 0.027455f * t100 + 0.0053407f * t100 * t100);
    float co2_sat_m3 = 1000.0f * exp(ln_phi0) * (co2_ppmv * 1.0e-6f);

    float tc_c = clamp(t_k - 273.15f, -2.0f, 40.0f);
    float sc_co2 = 2116.8f + tc_c * (-136.25f + tc_c * (4.7353f + tc_c * (-0.092307f + tc_c * 0.0007555f)));
    float kw_ms = (1.0f - f_ice) * 6.97e-7f * (u10 * u10) * sqrt(660.0f / max(sc_co2, 50.0f));
    float fgco2 = kw_ms * (co2_sat_m3 - co2_star_m3);

    float ddic_gas = (fgco2 * 1000.0f * dt) / max(h0_new, 10.0f);
    {
        float d_eff = ddic_gas + dic0_cmp;
        float t = dic0_val + d_eff;
        dic0_cmp = (fabs(dic0_val) >= fabs(d_eff)) ? ((dic0_val - t) + d_eff) : ((d_eff - t) + dic0_val);
        dic0_val = t;
    }
    dic_io[c] = dic0_val;
    dic_comp_io[c] = dic0_cmp;

    diag_2d[c] = fgco2;
    diag_2d[nc + c] = pco2;
    diag_2d[2u * nc + c] = ph_tot;
    diag_2d[3u * nc + c] = intpp_acc * 1.0e-3f;
    diag_2d[4u * nc + c] = epc_100;
}

// =========================================================================
// 24. Host SCVT RK3-HEVI Stage Prep (Cell + Edge Fused Kernels)
// =========================================================================

kernel void scvt_rk3_prep_cell(
    device const int* eoc           [[buffer(0)]],   // [nc * 6]
    device const float* esgn        [[buffer(1)]],   // [nc * 6] (non-zero => valid edge)
    device const float* area_e      [[buffer(2)]],   // [ne]
    device const float* area_c      [[buffer(3)]],   // [nc]
    device const float* m_s         [[buffer(4)]],   // [nz, nc]
    device const float* Thm_s       [[buffer(5)]],   // [nz, nc]
    device const float* Thm0        [[buffer(6)]],   // [nz, nc]
    device const float* u_s         [[buffer(7)]],   // [nz, ne]
    device const float* dz_c        [[buffer(8)]],   // [nz, nc]
    device const float* dz_w_int    [[buffer(9)]],   // [nz-1, nc]
    device const float* q_load      [[buffer(10)]],  // [nz, nc]
    device const float* zw_slope_e  [[buffer(11)]],  // [nz-1, ne]
    device const float* sponge_w    [[buffer(12)]],  // [nz-1]
    device const float* k_slope_top [[buffer(13)]],  // [nc]
    device float* pres_s            [[buffer(14)]],  // [nz, nc]
    device float* dp_dThm_s         [[buffer(15)]],  // [nz, nc]
    device float* q_odz_out         [[buffer(16)]],  // [nz, nc]
    device float* thm_s             [[buffer(17)]],  // [nz, nc]
    device float* rtot_s            [[buffer(18)]],  // [nz, nc]
    device float* ke_cell           [[buffer(19)]],  // [nz, nc]
    device float* u_dot_gradz_s     [[buffer(20)]],  // [nz-1, nc]
    device float* hevi_out          [[buffer(21)]],  // [7, nz-1, nc]: rdw, rdw/rtw, thw, dl, inv_piv, cp, 1/dz_w
    device float* p_cell_out        [[buffer(22)]],  // [nz, nc]
    constant ulong4& u_dim          [[buffer(23)]],  // [nz, nc, ne, same_thm0]
    constant float4& f_p0           [[buffer(24)]],  // [p_ref, r_d, gamma_d, gravity]
    constant float4& f_p1           [[buffer(25)]],  // [dt_ac, beta_hevi, 0, 0]
    uint c                          [[thread_position_in_grid]]
) {
    uint nz = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    uint ne = uint(u_dim.z);
    bool same_thm0 = (uint(u_dim.w) != 0u);
    if (c >= nc || nz < 2u) return;

    float p_ref = f_p0.x;
    float r_d = f_p0.y;
    float gamma = f_p0.z;
    float gravity = f_p0.w;
    float inv_pref_rd = r_d / p_ref;
    float dt_ac = f_p1.x;
    float beta = f_p1.y;
    float bdt2 = (beta * dt_ac) * (beta * dt_ac);
    float inv_ac = 1.0f / area_c[c];
    uint k_stop = uint(k_slope_top[c] + 0.5f);

    uint n_int = nz - 1u;
    uint int_stride = n_int * nc;
    device float* o_rdw = hevi_out;
    device float* o_rr  = hevi_out + int_stride;
    device float* o_thw = hevi_out + 2u * int_stride;
    device float* o_dl  = hevi_out + 3u * int_stride;
    device float* o_inv = hevi_out + 4u * int_stride;
    device float* o_cp  = hevi_out + 5u * int_stride;
    device float* o_idw = hevi_out + 6u * int_stride;

    uint c6 = c * 6u;
    int e_idx[6];
    float ae_w[6];
    for (uint s = 0u; s < 6u; ++s) {
        int ei = eoc[c6 + s];
        e_idx[s] = ei;
        ae_w[s] = (esgn[c6 + s] != 0.0f) ? (area_e[uint(ei)] * inv_ac) : 0.0f;
    }

    float rho_prev = 0.0f;
    float thm_prev = 0.0f;
    float rtot_prev = 0.0f;
    float dp_prev = 0.0f;
    float q_over_dz_prev = 0.0f;
    float u_edge_prev[6];

    float thm_w_prev = 0.0f;
    float fac_p_prev = 0.0f;
    float fac_g_prev = 0.0f;
    float c_up_prev = 0.0f;
    float q_odz_up_prev = 0.0f;
    float inv_piv_prev = 1.0f;

    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * nc + c;
        float m_v = m_s[idx];
        float T_v = Thm_s[idx];
        float dz_v = dz_c[idx];
        float q_v = q_load[idx];

        float rho_v = m_v / dz_v;
        float thm_v = T_v / max(m_v, 1.0e-4f);
        float arg = max((T_v / dz_v) * inv_pref_rd, 1.0e-7f);
        float p_v = p_ref * exp2(gamma * log2(arg));
        float dp_v = gamma * p_v / max(T_v, 1.0e-3f);
        float rtot_v = rho_v * q_v;
        float q_odz = q_v / dz_v;

        pres_s[idx] = p_v;
        dp_dThm_s[idx] = dp_v;
        q_odz_out[idx] = q_odz;
        thm_s[idx] = thm_v;
        rtot_s[idx] = rtot_v;
        p_cell_out[idx] = same_thm0 ? p_v : fma(dp_v, Thm0[idx] - T_v, p_v);

        uint ke_base = k * ne;
        float ke_sum = 0.0f;
        float uz_sum = 0.0f;
        bool do_slope = (k > 0u && (k - 1u) < k_stop);
        uint m_base_e = do_slope ? ((k - 1u) * ne) : 0u;
        for (uint s = 0u; s < 6u; ++s) {
            float w = ae_w[s];
            float u_cur = (w != 0.0f) ? u_s[ke_base + uint(e_idx[s])] : 0.0f;
            ke_sum = fma(w, u_cur * u_cur, ke_sum);
            if (do_slope && w != 0.0f) {
                float u_int = 0.5f * (u_edge_prev[s] + u_cur);
                uz_sum = fma(w, u_int * zw_slope_e[m_base_e + uint(e_idx[s])], uz_sum);
            }
            u_edge_prev[s] = u_cur;
        }
        ke_cell[idx] = 0.5f * ke_sum;

        if (k > 0u) {
            uint m = k - 1u;
            uint midx = m * nc + c;
            u_dot_gradz_s[midx] = uz_sum;

            float rdw = 0.5f * (rho_prev + rho_v);
            float rtw = max(0.5f * (rtot_prev + rtot_v), 1.0e-5f);
            float thw = 0.5f * (thm_prev + thm_v);
            float inv_dz_w = 1.0f / dz_w_int[midx];
            float rdw_ov_rtw = rdw / rtw;
            o_rdw[midx] = rdw;
            o_rr[midx]  = rdw_ov_rtw;
            o_thw[midx] = thw;
            o_idw[midx] = inv_dz_w;

            float fac_p = bdt2 * (rdw_ov_rtw * inv_dz_w);
            float fac_g = bdt2 * (0.5f * gravity * rdw_ov_rtw);
            float c_dn = dp_prev;
            float c_up = dp_v;

            float diag_m = max(
                1.0f + dt_ac * sponge_w[m] + fac_p * thw * (c_up + c_dn) + fac_g * (q_odz - q_over_dz_prev),
                0.5f
            );
            if (m == 0u) {
                o_dl[midx] = 0.0f;
                float inv0 = 1.0f / diag_m;
                o_inv[midx] = inv0;
                inv_piv_prev = inv0;
            } else {
                float dl_m = -fac_p * c_dn * thm_w_prev + fac_g * q_over_dz_prev;
                float du_prev = -fac_p_prev * c_up_prev * thw - fac_g_prev * q_odz_up_prev;
                float cp_prev = du_prev * inv_piv_prev;
                o_cp[(m - 1u) * nc + c] = cp_prev;
                o_dl[midx] = dl_m;
                float inv_m = 1.0f / (diag_m - dl_m * cp_prev);
                o_inv[midx] = inv_m;
                inv_piv_prev = inv_m;
            }
            thm_w_prev = thw;
            fac_p_prev = fac_p;
            fac_g_prev = fac_g;
            c_up_prev = c_up;
            q_odz_up_prev = q_odz;
        }
        rho_prev = rho_v;
        thm_prev = thm_v;
        rtot_prev = rtot_v;
        dp_prev = dp_v;
        q_over_dz_prev = q_odz;
    }
    o_cp[(nz - 2u) * nc + c] = 0.0f;
}

kernel void scvt_rk3_prep_edge(
    device const int* c0            [[buffer(0)]],   // [ne]
    device const int* c1            [[buffer(1)]],   // [ne]
    device const float* inv_dc      [[buffer(2)]],   // [ne]
    device const float* m_s         [[buffer(3)]],   // [nz, nc]
    device const float* thm_s       [[buffer(4)]],   // [nz, nc]
    device const float* rtot_s      [[buffer(5)]],   // [nz, nc]
    device const float* pres_s      [[buffer(6)]],   // [nz, nc]
    device const float* z_c         [[buffer(7)]],   // [nz, nc]
    device const float* ke_cell     [[buffer(8)]],   // [nz, nc]
    device const float* u_s         [[buffer(9)]],   // [nz, ne]
    device const float* w_s         [[buffer(10)]],  // [nz+1, nc]
    device const float* u_dot_gradz [[buffer(11)]],  // [nz-1, nc]
    device const float* dz_w_e      [[buffer(12)]],  // [nz-1, ne]
    device const float* rot_acc     [[buffer(13)]],  // [nz, ne]
    device const float* visc_u      [[buffer(14)]],  // [nz, ne]
    device const float* du_ext      [[buffer(15)]],  // [nz, ne]
    device float4* edge_c4          [[buffer(16)]],  // [nz, ne]: (fz0/(rtot*dc), fz1/(rtot*dc), m_edge, thm_e)
    device float* tend_u_slow       [[buffer(17)]],  // [nz, ne]
    constant ulong4& u_dim          [[buffer(18)]],  // [nz, nc, ne, 0]
    constant float4& f_par          [[buffer(19)]],  // [gravity, dtau_stage, 0, 0]
    uint gid                        [[thread_position_in_grid]]
) {
    uint nz = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    uint ne = uint(u_dim.z);
    if (gid >= nz * ne) return;

    float gravity = f_par.x;
    float dtau_v = f_par.y;

    uint k = gid / ne;
    uint e = gid - k * ne;
    uint cell0 = uint(c0[e]);
    uint cell1 = uint(c1[e]);
    uint i0 = k * nc + cell0;
    uint i1 = k * nc + cell1;

    float u_v = u_s[gid];
    float sgn_u = (u_v > 0.0f) ? 1.0f : ((u_v < 0.0f) ? -1.0f : 0.0f);

    float th0 = thm_s[i0];
    float th1 = thm_s[i1];
    float rt0 = rtot_s[i0];
    float rt1 = rtot_s[i1];
    float p0 = max(pres_s[i0], 0.1f);
    float p1 = max(pres_s[i1], 0.1f);
    float dz10 = z_c[i1] - z_c[i0];
    float idc = inv_dc[e];

    float m_edge_v = 0.5f * (m_s[i0] + m_s[i1]);
    float thm_e_v = 0.5f * (th0 + th1) - 0.15f * sgn_u * (th1 - th0);
    float inv_rtot_dc = idc / max(0.5f * (rt0 + rt1), 1.0e-5f);
    float fz0_sc = exp(-0.5f * (rt0 * gravity / p0) * dz10) * inv_rtot_dc;
    float fz1_sc = exp( 0.5f * (rt1 * gravity / p1) * dz10) * inv_rtot_dc;
    edge_c4[gid] = float4(fz0_sc, fz1_sc, m_edge_v, thm_e_v);

    float grad_ke = (ke_cell[i1] - ke_cell[i0]) * idc;

    float vadv = 0.0f;
    if (k + 1u < nz) {
        uint midx0 = k * nc + cell0;
        uint midx1 = k * nc + cell1;
        uint widx0 = (k + 1u) * nc + cell0;
        uint widx1 = (k + 1u) * nc + cell1;
        float wt0 = w_s[widx0] - u_dot_gradz[midx0];
        float wt1 = w_s[widx1] - u_dot_gradz[midx1];
        float wt_e = 0.5f * (wt0 + wt1);
        float dz_up = dz_w_e[k * ne + e];
        float cfl_z = fabs(wt_e) * (dtau_v / max(dz_up, 1.0f));
        float wt_eff = wt_e / max(1.0f, cfl_z / 0.40f);
        float beta_upw = min(1.0f, 0.35f + 1.5f * cfl_z);
        float du_dz = (u_s[(k + 1u) * ne + e] - u_v) / dz_up;
        vadv += -0.5f * (wt_eff - beta_upw * fabs(wt_eff)) * du_dz;
    }
    if (k > 0u) {
        uint m = k - 1u;
        uint midx0 = m * nc + cell0;
        uint midx1 = m * nc + cell1;
        uint widx0 = k * nc + cell0;
        uint widx1 = k * nc + cell1;
        float wt0 = w_s[widx0] - u_dot_gradz[midx0];
        float wt1 = w_s[widx1] - u_dot_gradz[midx1];
        float wt_e = 0.5f * (wt0 + wt1);
        float dz_dn = dz_w_e[m * ne + e];
        float cfl_z = fabs(wt_e) * (dtau_v / max(dz_dn, 1.0f));
        float wt_eff = wt_e / max(1.0f, cfl_z / 0.40f);
        float beta_upw = min(1.0f, 0.35f + 1.5f * cfl_z);
        float du_dz = (u_v - u_s[(k - 1u) * ne + e]) / dz_dn;
        vadv += -0.5f * (wt_eff + beta_upw * fabs(wt_eff)) * du_dz;
    }

    tend_u_slow[gid] = rot_acc[gid] - grad_ke + vadv + visc_u[gid] + du_ext[gid];
}

// =========================================================================
// 25. 3D OGCM Fused Baroclinic Momentum, Vertical Diffusion, & KPP Kernels
// =========================================================================

kernel void ogcm_baroclinic_f_fixed(
    device const int* c0            [[buffer(0)]],   // [ne]
    device const int* c1            [[buffer(1)]],   // [ne]
    device const float* inv_dc      [[buffer(2)]],   // [ne]
    device const float* p_baroc     [[buffer(3)]],   // [nzo, nc]
    device const float* lap_u       [[buffer(4)]],   // [nzo, ne]
    device const float* tau_n_e     [[buffer(5)]],   // [ne]
    device const float* h_e0_safe   [[buffer(6)]],   // [ne]
    device const float* wet_ef      [[buffer(7)]],   // [nzo, ne]
    device float* f_fixed           [[buffer(8)]],   // [nzo, ne]
    constant ulong4& u_dim          [[buffer(9)]],   // [nzo, nc, ne, 0]
    constant float4& f_p            [[buffer(10)]],  // [inv_rho_sw, a_h, 0, 0]
    uint gid                        [[thread_position_in_grid]]
) {
    uint nzo = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    uint ne = uint(u_dim.z);
    if (gid >= nzo * ne) return;
    float w = wet_ef[gid];
    if (w <= 0.0f) {
        f_fixed[gid] = 0.0f;
        return;
    }
    uint k = gid / ne;
    uint e = gid - k * ne;
    float inv_rho = f_p.x;
    float a_h = f_p.y;
    float pgf = -inv_rho * (p_baroc[k * nc + uint(c1[e])] - p_baroc[k * nc + uint(c0[e])]) * inv_dc[e];
    float ff = (pgf + a_h * lap_u[gid]) * w;
    if (k == 0u) {
        ff += (tau_n_e[e] * inv_rho / h_e0_safe[e]) * w;
    }
    f_fixed[gid] = ff;
}

kernel void ogcm_baroclinic_rk3_update(
    device const float* u0          [[buffer(0)]],
    device const float* u_s         [[buffer(1)]],
    device const float* cor_3d      [[buffer(2)]],
    device const float* f_fixed     [[buffer(3)]],
    device const float* is_bot_f    [[buffer(4)]],
    device const float* wet_ef      [[buffer(5)]],
    device float* u_out             [[buffer(6)]],
    constant uint& n_tot            [[buffer(7)]],
    constant float4& f_p            [[buffer(8)]],   // [rk_dt, r_rayleigh, 0, 0]
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= n_tot) return;
    float w = wet_ef[gid];
    if (w <= 0.0f) {
        u_out[gid] = 0.0f;
        return;
    }
    float rk_dt = f_p.x;
    float r_drag = f_p.y + is_bot_f[gid] * fabs(u_s[gid]);
    float u_new = ((u0[gid] + rk_dt * (cor_3d[gid] + f_fixed[gid])) / (1.0f + rk_dt * r_drag)) * w;
    u_out[gid] = clamp(u_new, -3.0f, 3.0f);
}

kernel void ogcm_mom_vdiff_col(
    device const int* c0            [[buffer(0)]],   // [ne]
    device const int* c1            [[buffer(1)]],   // [ne]
    device const float* u_star      [[buffer(2)]],   // [nzo, ne]
    device const float* av_diag     [[buffer(3)]],   // [nzo-1, nc]
    device const float* h_e_safe    [[buffer(4)]],   // [nzo, ne]
    device const float* dz_int_e    [[buffer(5)]],   // [nzo-1, ne]
    device const float* wet_ef      [[buffer(6)]],   // [nzo, ne]
    device const float* wet_int_ef  [[buffer(7)]],   // [nzo-1, ne]
    device float* u_out             [[buffer(8)]],   // [nzo, ne]
    constant ulong4& u_dim          [[buffer(9)]],   // [nzo, nc, ne, 0]
    constant float& dt              [[buffer(10)]],
    uint e                          [[thread_position_in_grid]]
) {
    uint nzo = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    uint ne = uint(u_dim.z);
    if (e >= ne || nzo == 0u || nzo > 64u) return;

    if (wet_ef[e] <= 0.0f) {
        for (uint k = 0u; k < nzo; ++k) u_out[k * ne + e] = 0.0f;
        return;
    }
    if (nzo == 1u) {
        u_out[e] = u_star[e];
        return;
    }

    uint cell0 = uint(c0[e]);
    uint cell1 = uint(c1[e]);
    float c_st[64];
    float d_st[64];
    float c_mom_prev = 0.0f;
    float c_prev = 0.0f;
    float d_prev = 0.0f;

    for (uint k = 0u; k < nzo; ++k) {
        uint idx = k * ne + e;
        float h_k = h_e_safe[idx];
        float a_k = (k > 0u) ? (-c_mom_prev / h_k) : 0.0f;
        float c_mom_k = 0.0f;
        if (k + 1u < nzo) {
            float wie = wet_int_ef[idx];
            if (wie > 0.0f) {
                float av_e = 0.5f * (av_diag[k * nc + cell0] + av_diag[k * nc + cell1]);
                c_mom_k = (dt * av_e / dz_int_e[idx]) * wie;
            }
        }
        float d_k = (k + 1u < nzo) ? (-c_mom_k / h_k) : 0.0f;
        float b_k = 1.0f - a_k - d_k;
        float rhs_k = u_star[idx];

        float inv_den = (k == 0u) ? (1.0f / b_k) : (1.0f / (b_k - a_k * c_prev));
        c_prev = (k + 1u < nzo) ? (d_k * inv_den) : 0.0f;
        d_prev = (k == 0u) ? (rhs_k * inv_den) : ((rhs_k - a_k * d_prev) * inv_den);
        c_st[k] = c_prev;
        d_st[k] = d_prev;
        c_mom_prev = c_mom_k;
    }

    float x_next = d_st[nzo - 1u];
    u_out[(nzo - 1u) * ne + e] = x_next;
    for (int k = int(nzo) - 2; k >= 0; --k) {
        uint uk = uint(k);
        x_next = d_st[uk] - c_st[uk] * x_next;
        u_out[uk * ne + e] = x_next;
    }
}

kernel void ogcm_tracer_vdiff_freeze_col(
    device const float* T_in        [[buffer(0)]],   // [nzo, nc]
    device const float* t_comp_in   [[buffer(1)]],   // [nzo, nc]
    device const float* S_in        [[buffer(2)]],   // [nzo, nc]
    device const float* s_comp_in   [[buffer(3)]],   // [nzo, nc]
    device const float* kv_diag     [[buffer(4)]],   // [nzo-1, nc]
    device const float* kv_redi     [[buffer(5)]],   // [nzo-1, nc]
    device const float* z_c         [[buffer(6)]],   // [nzo]
    device const float* dz_c_int    [[buffer(7)]],   // [nzo-1, nc]
    device const float* h_c_safe    [[buffer(8)]],   // [nzo, nc]
    device const float* h0_new      [[buffer(9)]],   // [nc]
    device const float* wet_cf      [[buffer(10)]],  // [nzo, nc]
    device const float* wet_int_cf  [[buffer(11)]],  // [nzo-1, nc]
    device const float* sfc_m_ice   [[buffer(12)]],  // [nc]
    device const float* sfc_t_ml    [[buffer(13)]],  // [nc]
    device float* T_out             [[buffer(14)]],  // [nzo, nc]
    device float* t_comp_out        [[buffer(15)]],  // [nzo, nc]
    device float* S_out             [[buffer(16)]],  // [nzo, nc]
    device float* s_comp_out        [[buffer(17)]],  // [nzo, nc]
    device float* kv_out            [[buffer(18)]],  // [nzo-1, nc]
    device float* t_ml_out          [[buffer(19)]],  // [nc]
    device float* m_ice_out         [[buffer(20)]],  // [nc]
    device const float* h0_frz      [[buffer(21)]],  // [nc] static h_c_safe[0] for freezing
    constant ulong4& u_dim          [[buffer(22)]],  // [nzo, nc, 0, 0]
    constant float4& f_p0           [[buffer(23)]],  // [dt, rho_sw, cp_sw, l_f0]
    constant float4& f_p1           [[buffer(24)]],  // [t_frz_ocn, t_ref_eos, s_ref_psu, s_seaice_psu]
    constant float4& f_p2           [[buffer(25)]],  // [eos_a0, eos_at, eos_az, eos_as]
    constant float4& f_p3           [[buffer(26)]],  // [eos_b0, kappa_v_conv, 0, 0]
    uint c                          [[thread_position_in_grid]]
) {
    uint nzo = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    if (c >= nc || nzo < 2u || nzo > 64u) return;

    float dt = f_p0.x;
    float rho_sw = f_p0.y;
    float cp_sw = f_p0.z;
    float l_f0 = f_p0.w;
    float t_frz_ocn = f_p1.x;
    float t_ref_eos = f_p1.y;
    float s_ref_psu = f_p1.z;
    float s_seaice = f_p1.w;
    float eos_a0 = f_p2.x;
    float eos_at = f_p2.y;
    float eos_az = f_p2.z;
    float eos_as = f_p2.w;
    float eos_b0 = f_p3.x;
    float kappa_conv = f_p3.y;

    bool is_ocn = (wet_cf[c] > 0.5f);
    if (!is_ocn) {
        for (uint k = 0u; k < nzo; ++k) {
            uint idx = k * nc + c;
            T_out[idx] = T_in[idx];
            t_comp_out[idx] = t_comp_in[idx];
            S_out[idx] = S_in[idx];
            s_comp_out[idx] = s_comp_in[idx];
            if (k + 1u < nzo) kv_out[idx] = 0.0f;
        }
        t_ml_out[c] = sfc_t_ml[c];
        m_ice_out[c] = sfc_m_ice[c];
        return;
    }

    float c_st[64];
    float dt_st[64];
    float ds_st[64];

    // Forward pass: EOS convective stability, flux-increment tridiagonal assembly & forward Thomas
    float T_cur = T_in[c];
    float S_cur = S_in[c];
    float dT_e = T_cur - t_ref_eos;
    float dS_e = S_cur - s_ref_psu;
    float rho_cur = rho_sw * (-(eos_a0 + eos_at * dT_e + eos_az * z_c[0] + eos_as * dS_e) * dT_e + eos_b0 * dS_e) * wet_cf[c];

    float c_dif_prev = 0.0f;
    float f_exp_t_prev = 0.0f;
    float f_exp_s_prev = 0.0f;
    float c_prev = 0.0f;
    float dt_prev = 0.0f;
    float ds_prev = 0.0f;

    for (uint k = 0u; k < nzo; ++k) {
        uint idx = k * nc + c;
        float h_k = (k == 0u) ? h0_new[c] : h_c_safe[idx];
        float c_dif_k = 0.0f;
        float f_exp_t_k = 0.0f;
        float f_exp_s_k = 0.0f;

        float T_next = T_cur;
        float S_next = S_cur;
        float rho_next = 0.0f;
        if (k + 1u < nzo) {
            uint nidx = (k + 1u) * nc + c;
            T_next = T_in[nidx];
            S_next = S_in[nidx];
            float dTn = T_next - t_ref_eos;
            float dSn = S_next - s_ref_psu;
            rho_next = rho_sw * (-(eos_a0 + eos_at * dTn + eos_az * z_c[k + 1u] + eos_as * dSn) * dTn + eos_b0 * dSn) * wet_cf[nidx];
            float wic = wet_int_cf[idx];
            float kv_k = 0.0f;
            if (wic > 0.5f) {
                float dz_i = dz_c_int[idx];
                float drho_dz = (rho_next - rho_cur) / dz_i;
                kv_k = kv_diag[idx] + kv_redi[idx];
                if (drho_dz < 0.0f) kv_k = max(kv_k, kappa_conv);
                kv_k *= wic;
                c_dif_k = (dt * kv_k / dz_i) * wic;
                f_exp_t_k = c_dif_k * (T_next - T_cur);
                f_exp_s_k = c_dif_k * (S_next - S_cur);
            }
            kv_out[idx] = kv_k;
        }

        float a_k = (k > 0u) ? (-c_dif_prev / h_k) : 0.0f;
        float d_k = (k + 1u < nzo) ? (-c_dif_k / h_k) : 0.0f;
        float b_k = 1.0f - a_k - d_k;
        float rhs_t = (f_exp_t_k - f_exp_t_prev) / h_k;
        float rhs_s = (f_exp_s_k - f_exp_s_prev) / h_k;

        float inv_den = (k == 0u) ? (1.0f / b_k) : (1.0f / (b_k - a_k * c_prev));
        c_prev = (k + 1u < nzo) ? (d_k * inv_den) : 0.0f;
        dt_prev = (k == 0u) ? (rhs_t * inv_den) : ((rhs_t - a_k * dt_prev) * inv_den);
        ds_prev = (k == 0u) ? (rhs_s * inv_den) : ((rhs_s - a_k * ds_prev) * inv_den);
        c_st[k] = c_prev;
        dt_st[k] = dt_prev;
        ds_st[k] = ds_prev;

        c_dif_prev = c_dif_k;
        f_exp_t_prev = f_exp_t_k;
        f_exp_s_prev = f_exp_s_k;
        T_cur = T_next;
        S_cur = S_next;
        rho_cur = rho_next;
    }

    // Back-substitution in-place into dt_st[], ds_st[]
    float xt = dt_st[nzo - 1u];
    float xs = ds_st[nzo - 1u];
    for (int k = int(nzo) - 2; k >= 0; --k) {
        uint uk = uint(k);
        xt = dt_st[uk] - c_st[uk] * xt;
        xs = ds_st[uk] - c_st[uk] * xs;
        dt_st[uk] = xt;
        ds_st[uk] = xs;
    }

    // Telescoping implicit flux update + exact Neumaier/Kahan summation + Layer-0 freezing
    float f_imp_t_prev = 0.0f;
    float f_imp_s_prev = 0.0f;
    for (uint k = 0u; k < nzo; ++k) {
        uint idx = k * nc + c;
        float wcf = wet_cf[idx];
        float h_k = (k == 0u) ? h0_new[c] : h_c_safe[idx];
        float Tk = T_in[idx];
        float Sk = S_in[idx];

        float f_imp_t_k = 0.0f;
        float f_imp_s_k = 0.0f;
        if (k + 1u < nzo) {
            float wic = wet_int_cf[idx];
            if (wic > 0.5f) {
                float c_dif_k = (dt * kv_out[idx] / dz_c_int[idx]) * wic;
                uint nidx = (k + 1u) * nc + c;
                f_imp_t_k = c_dif_k * ((T_in[nidx] - Tk) + (dt_st[k + 1u] - dt_st[k]));
                f_imp_s_k = c_dif_k * ((S_in[nidx] - Sk) + (ds_st[k + 1u] - ds_st[k]));
            }
        }
        float dt_cons = ((f_imp_t_k - f_imp_t_prev) / h_k) * wcf;
        float ds_cons = ((f_imp_s_k - f_imp_s_prev) / h_k) * wcf;

        // Neumaier/Kahan compensated addition for T and S
        float tc_old = t_comp_in[idx];
        float dt_eff = dt_cons + tc_old;
        float T_new = Tk + dt_eff;
        float tc_new = (fabs(Tk) >= fabs(dt_eff)) ? ((Tk - T_new) + dt_eff) : ((dt_eff - T_new) + Tk);

        float sc_old = s_comp_in[idx];
        float ds_eff = ds_cons + sc_old;
        float S_new = Sk + ds_eff;
        float sc_new = (fabs(Sk) >= fabs(ds_eff)) ? ((Sk - S_new) + ds_eff) : ((ds_eff - S_new) + Sk);
        if (wcf > 0.5f) S_new = clamp(S_new, 5.0f, 45.0f);

        if (k == 0u) {
            float freeze_exc = max(t_frz_ocn - T_new, 0.0f);
            float h0_s = h0_frz[c];
            float dm_freeze = (cp_sw * rho_sw * h0_s * freeze_exc) / l_f0;
            T_new = T_new + freeze_exc;
            float ds_brine = ((s_ref_psu - s_seaice) * dm_freeze) / (rho_sw * h0_s);
            float dsb_eff = ds_brine + sc_new;
            float S_br = S_new + dsb_eff;
            sc_new = (fabs(S_new) >= fabs(dsb_eff)) ? ((S_new - S_br) + dsb_eff) : ((dsb_eff - S_br) + S_new);
            S_new = clamp(S_br, 5.0f, 45.0f);

            float T0_cl = max(T_new, t_frz_ocn);
            T_out[c] = T0_cl;
            t_comp_out[c] = tc_new;
            S_out[c] = S_new;
            s_comp_out[c] = sc_new;
            t_ml_out[c] = T0_cl;
            m_ice_out[c] = max(sfc_m_ice[c] + dm_freeze, 0.0f);
        } else {
            T_out[idx] = (wcf > 0.5f) ? max(T_new, t_frz_ocn) : T_new;
            t_comp_out[idx] = tc_new;
            S_out[idx] = S_new;
            s_comp_out[idx] = sc_new;
        }
        f_imp_t_prev = f_imp_t_k;
        f_imp_s_prev = f_imp_s_k;
    }
}

kernel void ogcm_kpp_vdiff_col(
    device const float* rho_anom    [[buffer(0)]],   // [nzo, nc]
    device const float* u_cell      [[buffer(1)]],   // [nzo, nc]
    device const float* v_cell      [[buffer(2)]],   // [nzo, nc]
    device const float* u_star_o    [[buffer(3)]],   // [nc]
    device const float* b0_sfc      [[buffer(4)]],   // [nc]
    device const float* wt_sfc      [[buffer(5)]],   // [nc]
    device const float* ws_sfc      [[buffer(6)]],   // [nc]
    device const float* wet_cf      [[buffer(7)]],   // [nzo, nc]
    device const float* wet_int_cf  [[buffer(8)]],   // [nzo-1, nc]
    device const float* h_c         [[buffer(9)]],   // [nzo, nc]
    device const float* dz_c_int    [[buffer(10)]],  // [nzo-1, nc]
    device const float* z_c         [[buffer(11)]],  // [nzo]
    device const float* z_int_m     [[buffer(12)]],  // [nzo-1]
    device const float* h_col_kpp   [[buffer(13)]],  // [nc]
    device const float* h_col_safe  [[buffer(14)]],  // [nc]
    device const float* f_abs       [[buffer(15)]],  // [nc]
    device const float* kv_bg       [[buffer(16)]],  // [nzo-1, nc]
    device const float* kv_tide_num [[buffer(17)]],  // [nzo-1, nc]
    device float* kv_out            [[buffer(18)]],  // [nzo-1, nc]
    device float* av_out            [[buffer(19)]],  // [nzo-1, nc]
    device float* h_bl_out          [[buffer(20)]],  // [nc]
    device float* f_nl_t_out        [[buffer(21)]],  // [nzo-1, nc]
    device float* f_nl_s_out        [[buffer(22)]],  // [nzo-1, nc]
    constant ulong4& u_dim          [[buffer(23)]],  // [nzo, nc, 0, 0]
    constant float4& f_p0           [[buffer(24)]],  // [gravity, rho_sw, von_karman, kpp_ri_crit]
    constant float4& f_p1           [[buffer(25)]],  // [kpp_kv_max, kpp_cs_nl, shear_ri_0, shear_nu_0]
    constant float4& f_p2           [[buffer(26)]],  // [tidal_kv_max, kappa_v_conv, a_v_mom, 0]
    uint c                          [[thread_position_in_grid]]
) {
    uint nzo = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    if (c >= nc || nzo < 2u || nzo > 64u) return;

    if (wet_cf[c] <= 0.5f) {
        h_bl_out[c] = 0.0f;
        for (uint m = 0u; m < nzo - 1u; ++m) {
            uint midx = m * nc + c;
            kv_out[midx] = 0.0f;
            av_out[midx] = 0.0f;
            f_nl_t_out[midx] = 0.0f;
            f_nl_s_out[midx] = 0.0f;
        }
        return;
    }

    float g_ov_rho = f_p0.x / f_p0.y;
    float vk = f_p0.z;
    float ri_crit = f_p0.w;
    float kv_max = f_p1.x;
    float cs_nl = f_p1.y;
    float sh_ri0 = f_p1.z;
    float sh_nu0 = f_p1.w;
    float tide_max = f_p2.x;
    float kappa_conv = f_p2.y;
    float av_mom = f_p2.z;

    float ustar = u_star_o[c];
    float b0 = b0_sfc[c];
    float wt0 = wt_sfc[c];
    float ws0 = ws_sfc[c];
    float ustar3 = ustar * ustar * ustar;
    float w_star_est = pow(max(-b0, 0.0f) * h_col_kpp[c], 1.0f / 3.0f);
    float w_s0 = vk * pow(ustar3 + 2.0f * (w_star_est * w_star_est * w_star_est) + 1.0e-12f, 1.0f / 3.0f);

    float rho0 = rho_anom[c];
    float u0 = u_cell[c];
    float v0 = v_cell[c];
    float h0 = h_c[c];

    // Pass 1: N^2 on interfaces + contiguous bulk-Ri mixed-layer depth h_bl
    float n2_r[64];
    float s2_r[64];
    float drho_r[64];
    float rho_prev = rho0;
    float u_prev = u0;
    float v_prev = v0;
    for (uint m = 0u; m < nzo - 1u; ++m) {
        uint midx = m * nc + c;
        uint nidx = (m + 1u) * nc + c;
        float wic = wet_int_cf[midx];
        float rho_n = rho_anom[nidx];
        float u_n = u_cell[nidx];
        float v_n = v_cell[nidx];
        float inv_dz = 1.0f / dz_c_int[midx];
        float dr = ((rho_n - rho_prev) * inv_dz) * wic;
        float du = (u_n - u_prev) * inv_dz;
        float dv = (v_n - v_prev) * inv_dz;
        drho_r[m] = dr;
        n2_r[m] = g_ov_rho * max(dr, 0.0f);
        s2_r[m] = (du * du + dv * dv) * wic;
        rho_prev = rho_n;
        u_prev = u_n;
        v_prev = v_n;
    }

    float h_bl_ri = h0;
    bool contig = true;
    for (uint k = 1u; k < nzo; ++k) {
        if (!contig) break;
        uint idx = k * nc + c;
        if (wet_cf[idx] <= 0.5f) {
            contig = false;
            break;
        }
        float db_k = g_ov_rho * (rho_anom[idx] - rho0);
        float du_k = u_cell[idx] - u0;
        float dv_k = v_cell[idx] - v0;
        float dv2_k = du_k * du_k + dv_k * dv_k;
        float n2_c = (k + 1u < nzo) ? (0.5f * (n2_r[k - 1u] + n2_r[k])) : n2_r[nzo - 2u];
        float vt2_k = 0.08f * sqrt(max(n2_c, 0.0f)) * z_c[k] * w_s0;
        float ri_b = (db_k * z_c[k]) / (dv2_k + vt2_k + 1.0e-6f);
        if (ri_b < ri_crit || db_k <= 0.0f) {
            h_bl_ri += h_c[idx];
        } else {
            contig = false;
        }
    }

    float h_bl = h_bl_ri;
    if (b0 > 1.0e-11f) {
        float h_ek = 0.7f * ustar / f_abs[c];
        float h_mo = ustar3 / (vk * max(b0, 1.0e-12f));
        float h_stab_cap = min(h_ek, h_mo);
        h_bl = min(h_bl_ri, max(h_stab_cap, h0));
    }
    h_bl = clamp(h_bl, h0, h_col_safe[c]);
    h_bl_out[c] = h_bl;

    float inv_hbl = 1.0f / max(h_bl, 1.0f);
    float neg_b0 = max(-b0, 0.0f);
    bool is_conv_sfc = (b0 < -1.0e-11f);

    // Pass 2: KPP diffusivity, non-local fluxes, shear, tidal, and convective adjustment
    for (uint m = 0u; m < nzo - 1u; ++m) {
        uint midx = m * nc + c;
        float wic = wet_int_cf[midx];
        if (wic <= 0.5f) {
            kv_out[midx] = 0.0f;
            av_out[midx] = 0.0f;
            f_nl_t_out[midx] = 0.0f;
            f_nl_s_out[midx] = 0.0f;
            continue;
        }
        float sigma = z_int_m[m] * inv_hbl;
        float kv_kpp = 0.0f;
        float fnl_t = 0.0f;
        float fnl_s = 0.0f;
        if (sigma > 0.0f && sigma < 1.0f) {
            float one_m_s = 1.0f - sigma;
            float g_shape = sigma * one_m_s * one_m_s;
            float conv_scale = 16.0f * vk * sigma * neg_b0 * h_bl;
            float w_s_int = vk * pow(ustar3 + conv_scale + 1.0e-12f, 1.0f / 3.0f);
            kv_kpp = clamp(h_bl * w_s_int * g_shape, 0.0f, kv_max);
            if (is_conv_sfc) {
                float nlw = cs_nl * g_shape;
                fnl_t = -nlw * wt0;
                fnl_s = -nlw * ws0;
            }
        }

        float n2_m = n2_r[m];
        float ri_g = n2_m / (s2_r[m] + 1.0e-10f);
        float ri_rat = clamp(ri_g / sh_ri0, 0.0f, 1.0f);
        float omr2 = 1.0f - ri_rat * ri_rat;
        float kv_shear = sh_nu0 * (omr2 * omr2 * omr2);

        float kv_tide = clamp(kv_tide_num[midx] / max(n2_m, 1.0e-7f), 0.0f, tide_max);
        float kv_tot = kv_bg[m] + kv_tide + kv_kpp + kv_shear;
        if (drho_r[m] < 0.0f) kv_tot = max(kv_tot, kappa_conv);

        kv_out[midx] = kv_tot;
        av_out[midx] = av_mom + kv_kpp + kv_shear;
        f_nl_t_out[midx] = fnl_t;
        f_nl_s_out[midx] = fnl_s;
    }
}

kernel void ogcm_pre_kpp_col(
    device const float* T_in        [[buffer(0)]],   // [nzo, nc]
    device const float* S_in        [[buffer(1)]],   // [nzo, nc]
    device const float* t_comp_in   [[buffer(2)]],   // [nzo, nc]
    device const float* s_comp_in   [[buffer(3)]],   // [nzo, nc]
    device const float* sfc_t_ml    [[buffer(4)]],   // [nc]
    device const float* sfc_m_ice   [[buffer(5)]],   // [nc]
    device const float* m_ice_prev  [[buffer(6)]],   // [nc]
    device const float* r_coast     [[buffer(7)]],   // [nc]
    device const float* q_calve     [[buffer(8)]],   // [nc]
    device const float* evap_cell   [[buffer(9)]],   // [nc]
    device const float* precip_cell [[buffer(10)]],  // [nc]
    device const float* sw_net_ocn  [[buffer(11)]],  // [nc]
    device const float* wet_cf      [[buffer(12)]],  // [nzo, nc]
    device const float* h_c_safe    [[buffer(13)]],  // [nzo, nc]
    device const float* dz_nom      [[buffer(14)]],  // [nzo]
    device const float* z_c         [[buffer(15)]],  // [nzo]
    device const float* frac_out_0  [[buffer(16)]],  // [nc]
    device const float* frac_sub    [[buffer(17)]],  // [nzo, nc]
    device float* T_out             [[buffer(18)]],  // [nzo, nc]
    device float* S_out             [[buffer(19)]],  // [nzo, nc]
    device float* t_comp_out        [[buffer(20)]],  // [nzo, nc]
    device float* s_comp_out        [[buffer(21)]],  // [nzo, nc]
    device float* rho_anom_out      [[buffer(22)]],  // [nzo, nc]
    device float* p_baroc_out       [[buffer(23)]],  // [nzo, nc]
    device float* sfc_flux3_out     [[buffer(24)]],  // [3, nc]: wt_sfc, ws_sfc, b0_sfc
    constant ulong4& u_dim          [[buffer(25)]],  // [nzo, nc, apply_sw, 0]
    constant float4& f_p0           [[buffer(26)]],  // [dt, rho_sw, cp_sw, gravity]
    constant float4& f_p1           [[buffer(27)]],  // [t_ref_eos, s_ref_psu, s_seaice_psu, eos_b0]
    constant float4& f_p2           [[buffer(28)]],  // [eos_a0, eos_at, eos_az, eos_as]
    uint c                          [[thread_position_in_grid]]
) {
    uint nzo = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    uint apply_sw = uint(u_dim.z);
    if (c >= nc || nzo == 0u) return;

    float dt = f_p0.x;
    float rho_sw = f_p0.y;
    float cp_sw = f_p0.z;
    float gravity = f_p0.w;
    float t_ref = f_p1.x;
    float s_ref = f_p1.y;
    float s_ice = f_p1.z;
    float eos_b0 = f_p1.w;
    float eos_a0 = f_p2.x;
    float eos_at = f_p2.y;
    float eos_az = f_p2.z;
    float eos_as = f_p2.w;

    if (wet_cf[c] <= 0.5f) {
        for (uint k = 0u; k < nzo; ++k) {
            uint idx = k * nc + c;
            T_out[idx] = T_in[idx];
            S_out[idx] = S_in[idx];
            t_comp_out[idx] = t_comp_in[idx];
            s_comp_out[idx] = s_comp_in[idx];
            rho_anom_out[idx] = 0.0f;
            p_baroc_out[idx] = 0.0f;
        }
        sfc_flux3_out[c] = 0.0f;
        sfc_flux3_out[nc + c] = 0.0f;
        sfc_flux3_out[2u * nc + c] = 0.0f;
        return;
    }

    float t0_prev = T_in[c];
    float s0_prev = S_in[c];
    float h0 = h_c_safe[c];

    // Layer 0 calving + freshwater + sea-ice salt budget with Kahan summation
    float t0_in = sfc_t_ml[c];
    float dt_calve = -(dt * q_calve[c]) / (cp_sw * rho_sw * h0);
    float tc0 = t_comp_in[c];
    float d_eff_c = dt_calve + tc0;
    float t0_cur = t0_in + d_eff_c;
    tc0 = (fabs(t0_in) >= fabs(d_eff_c)) ? ((t0_in - t0_cur) + d_eff_c) : ((d_eff_c - t0_cur) + t0_in);

    float dm_ice = sfc_m_ice[c] - m_ice_prev[c];
    float fw = evap_cell[c] - precip_cell[c] - r_coast[c];
    float ds_sfc = (s_ref * (fw * dt) + (s_ref - s_ice) * dm_ice) / (rho_sw * h0);
    float sc0 = s_comp_in[c];
    float d_eff_s = ds_sfc + sc0;
    float s0_cur = s0_prev + d_eff_s;
    sc0 = (fabs(s0_prev) >= fabs(d_eff_s)) ? ((s0_prev - s0_cur) + d_eff_s) : ((d_eff_s - s0_cur) + s0_prev);
    s0_cur = clamp(s0_cur, 5.0f, 45.0f);

    float q_to_dt = (apply_sw != 0u) ? ((dt / (cp_sw * rho_sw)) * max(sw_net_ocn[c], 0.0f)) : 0.0f;
    float p_cum = 0.0f;

    for (uint k = 0u; k < nzo; ++k) {
        uint idx = k * nc + c;
        float wcf = wet_cf[idx];
        float Tk = (k == 0u) ? t0_cur : T_in[idx];
        float Sk = (k == 0u) ? s0_cur : S_in[idx];
        float tck = (k == 0u) ? tc0 : t_comp_in[idx];
        float sck = (k == 0u) ? sc0 : s_comp_in[idx];

        if (apply_sw != 0u && wcf > 0.5f) {
            float dt_sw = (k == 0u) ? (-(q_to_dt * frac_out_0[c]) / h0) : ((q_to_dt * frac_sub[idx]) / h_c_safe[idx]);
            float d_eff = dt_sw + tck;
            float Tn = Tk + d_eff;
            tck = (fabs(Tk) >= fabs(d_eff)) ? ((Tk - Tn) + d_eff) : ((d_eff - Tn) + Tk);
            Tk = Tn;
        }

        T_out[idx] = Tk;
        S_out[idx] = Sk;
        t_comp_out[idx] = tck;
        s_comp_out[idx] = sck;

        float dTe = Tk - t_ref;
        float dSe = Sk - s_ref;
        float rho_a = rho_sw * (-(eos_a0 + eos_at * dTe + eos_az * z_c[k] + eos_as * dSe) * dTe + eos_b0 * dSe) * wcf;
        rho_anom_out[idx] = rho_a;

        float g_rdz = gravity * rho_a * dz_nom[k];
        p_cum += g_rdz;
        p_baroc_out[idx] = p_cum - 0.5f * g_rdz;

        if (k == 0u) {
            float inv_dt = 1.0f / max(dt, 1.0f);
            float wt = (Tk - t0_prev) * h0 * inv_dt;
            float ws = (Sk - s0_prev) * h0 * inv_dt;
            float dTp = t0_prev - t_ref;
            float dSp = s0_prev - s_ref;
            float rho0_p = rho_sw * (-(eos_a0 + eos_at * dTp + eos_az * z_c[0] + eos_as * dSp) * dTp + eos_b0 * dSp);
            float b0 = -(gravity / rho_sw) * ((rho_a - rho0_p) * h0 * inv_dt);
            sfc_flux3_out[c] = wt;
            sfc_flux3_out[nc + c] = ws;
            sfc_flux3_out[2u * nc + c] = b0;
        }
    }
}

kernel void ogcm_gm_slope_psi_edge(
    device const int* c0            [[buffer(0)]],   // [ne]
    device const int* c1            [[buffer(1)]],   // [ne]
    device const float* inv_dc      [[buffer(2)]],   // [ne]
    device const float* w_vor_edge  [[buffer(3)]],   // [ne]
    device const float* rho_anom    [[buffer(4)]],   // [nzo, nc]
    device const float* dz_c_int    [[buffer(5)]],   // [nzo-1, nc]
    device const float* wet_int_cf  [[buffer(6)]],   // [nzo-1, nc]
    device const float* h_bl        [[buffer(7)]],   // [nc]
    device const float* h_e_safe    [[buffer(8)]],   // [nzo, ne]
    device const float* z_int_m     [[buffer(9)]],   // [nzo-1]
    device const float* wet_ef      [[buffer(10)]],  // [nzo, ne]
    device const float* wet_int_ef  [[buffer(11)]],  // [nzo-1, ne]
    device const float* gm_scale_e  [[buffer(12)]],  // [ne]
    device const float* fk_coef_e   [[buffer(13)]],  // [ne]
    device float* sx_int_out        [[buffer(14)]],  // [nzo-1, ne]
    device float* psi_int_out       [[buffer(15)]],  // [nzo-1, ne]
    device float* s2_edge_w_out     [[buffer(16)]],  // [nzo-1, ne] or [1]
    constant ulong4& u_dim          [[buffer(17)]],  // [nzo, nc, ne, flags]
    constant float4& f_p0           [[buffer(18)]],  // [kappa_gm, gm_slope_max, g_ov_rho, 0]
    uint gid                        [[thread_position_in_grid]]
) {
    uint nzo = uint(u_dim.x);
    uint nc = uint(u_dim.y);
    uint ne = uint(u_dim.z);
    uint flags = uint(u_dim.w);
    if (nzo < 2u || gid >= (nzo - 1u) * ne) return;

    bool use_scale_gm = ((flags & 4u) != 0u);
    bool use_submeso  = ((flags & 2u) != 0u);
    bool use_redi     = ((flags & 1u) != 0u);

    float wie = wet_int_ef[gid];
    if (wie <= 0.0f) {
        sx_int_out[gid] = 0.0f;
        psi_int_out[gid] = 0.0f;
        if (use_redi) s2_edge_w_out[gid] = 0.0f;
        return;
    }

    uint k = gid / ne;
    uint e = gid - k * ne;
    uint cell0 = uint(c0[e]);
    uint cell1 = uint(c1[e]);
    float idc = inv_dc[e];

    uint c0_k = k * nc + cell0;
    uint c1_k = k * nc + cell1;
    float rho_c0_k = rho_anom[c0_k];
    float rho_c1_k = rho_anom[c1_k];
    float rho_c0_n = rho_anom[c0_k + nc];
    float rho_c1_n = rho_anom[c1_k + nc];

    float drho_dx_cur  = ((rho_c1_k - rho_c0_k) * idc) * wet_ef[gid];
    float drho_dx_next = ((rho_c1_n - rho_c0_n) * idc) * wet_ef[gid + ne];
    float dz0 = ((rho_c0_n - rho_c0_k) / dz_c_int[c0_k]) * wet_int_cf[c0_k];
    float dz1 = ((rho_c1_n - rho_c1_k) / dz_c_int[c1_k]) * wet_int_cf[c1_k];
    float drho_dz_e = 0.5f * (dz0 + dz1);
    float drho_dx_int = 0.5f * (drho_dx_cur + drho_dx_next);

    float hbl_e = max(0.5f * (h_bl[cell0] + h_bl[cell1]), 1.0f);
    float zk = z_int_m[k];
    float smax = f_p0.y;
    float slope_raw = clamp(-drho_dx_int / max(drho_dz_e, 1.0e-5f), -smax, smax) * wie;
    float sx = slope_raw * clamp(zk / hbl_e, 0.0f, 1.0f);
    float kgm = use_scale_gm ? (f_p0.x * gm_scale_e[e]) : f_p0.x;
    float psi_k = kgm * sx;

    if (use_submeso) {
        float inv_hbl_fk = 1.0f / max(hbl_e, 10.0f);
        float z_norm = zk * inv_hbl_fk;
        if (z_norm > 0.0f && z_norm < 1.0f) {
            float drho0 = ((rho_anom[cell1] - rho_anom[cell0]) * idc) * wet_ef[e];
            float db_dx_ml = -f_p0.z * drho0;
            float fk_c = fk_coef_e[e] * hbl_e * hbl_e;
            float xi = 2.0f * z_norm - 1.0f;
            float xi2 = xi * xi;
            float mu_fk = (1.0f - xi2) * (1.0f + (5.0f / 21.0f) * xi2);
            psi_k += clamp(fk_c * db_dx_ml * mu_fk, -2.5f, 2.5f) * wie;
        }
    }
    float psi_cap = 0.15f * min(h_e_safe[gid], h_e_safe[gid + ne]);
    psi_k = clamp(psi_k, -psi_cap, psi_cap) * wie;

    sx_int_out[gid] = sx;
    psi_int_out[gid] = psi_k;
    if (use_redi) s2_edge_w_out[gid] = (sx * sx) * w_vor_edge[e];
}

kernel void ogcm_gm_apply_edge(
    device const float* u_star      [[buffer(0)]],   // [nzo, ne]
    device const float* u_bt_mean   [[buffer(1)]],   // [ne]
    device const float* u_star_baro [[buffer(2)]],   // [ne]
    device const float* h_e         [[buffer(3)]],   // [nzo, ne]
    device const float* h_e_safe    [[buffer(4)]],   // [nzo, ne]
    device const float* wet_ef      [[buffer(5)]],   // [nzo, ne]
    device const float* psi_int     [[buffer(6)]],   // [nzo-1, ne]
    device float* u_new_out         [[buffer(7)]],   // [nzo, ne]
    device float* h_u_edge_out      [[buffer(8)]],   // [nzo, ne]
    constant ulong4& u_dim          [[buffer(9)]],   // [nzo, ne, 0, 0]
    uint gid                        [[thread_position_in_grid]]
) {
    uint nzo = uint(u_dim.x);
    uint ne = uint(u_dim.y);
    if (gid >= nzo * ne) return;

    float wef = wet_ef[gid];
    if (wef <= 0.0f) {
        u_new_out[gid] = 0.0f;
        h_u_edge_out[gid] = 0.0f;
        return;
    }

    uint k = gid / ne;
    uint e = gid - k * ne;
    float du_bt = u_bt_mean[e] - u_star_baro[e];
    float u_n = clamp((u_star[gid] + du_bt) * wef, -4.0f, 4.0f);
    float psi_k = (k + 1u < nzo) ? psi_int[gid] : 0.0f;
    float psi_prev = (k > 0u) ? psi_int[gid - ne] : 0.0f;
    float u_gm = (-(psi_k - psi_prev) / h_e_safe[gid]) * wef;

    u_new_out[gid] = u_n;
    h_u_edge_out[gid] = h_e[gid] * ((u_n + u_gm) * wef);
}

// =========================================================================
// 23. Fused Student Subgrid Tile Reconstruction + Surface Microclimate + Update
// =========================================================================

kernel void crm_student_tile_update_col(
    device const float* atmos_rho   [[buffer(0)]],   // [nz, nc]
    device const float* stu_thm     [[buffer(1)]],   // [nz, nc]
    device const float* stu_qv      [[buffer(2)]],   // [nz, nc]
    device const float* stu_qc      [[buffer(3)]],   // [nz, nc]
    device const float* stu_qi      [[buffer(4)]],   // [nz, nc]
    device const float* stu_qp      [[buffer(5)]],   // [nz, nc]
    device const float* stu_fcld    [[buffer(6)]],   // [nz, nc]
    device const float* stu_u       [[buffer(7)]],   // [nz, nc]
    device const float* stu_v       [[buffer(8)]],   // [nz, nc]
    device const float* stu_1d      [[buffer(9)]],   // [6, nc]: rain, snow, f_prec, rho_0, dz_0, exner_0
    device const float* score_cld   [[buffer(10)]],  // [nc, n_sub] or [1]
    device const float* score_prec  [[buffer(11)]],  // [nc, n_sub] or [1]
    device const float* shf_tile    [[buffer(12)]],  // [nc, n_sub]
    device const float* evap_tile   [[buffer(13)]],  // [nc, n_sub]
    device const float* dthm_cp     [[buffer(14)]],  // [nc, n_sub] or [1]
    device const float* dqv_cp      [[buffer(15)]],  // [nc, n_sub] or [1]
    device const float* u_gust      [[buffer(16)]],  // [nc, n_sub] or [1]
    device float* crm_rho           [[buffer(17)]],  // [nz, nc, n_sub] (out)
    device float* crm_thm           [[buffer(18)]],  // [nz, nc, n_sub] (in/out)
    device float* crm_qn            [[buffer(19)]],  // [nz, nc, n_sub] (out)
    device float* crm_qp            [[buffer(20)]],  // [nz, nc, n_sub] (out)
    device float* crm_qv            [[buffer(21)]],  // [nz, nc, n_sub] (out)
    device float* crm_qc            [[buffer(22)]],  // [nz, nc, n_sub] (out)
    device float* crm_qi            [[buffer(23)]],  // [nz, nc, n_sub] (out)
    device float* crm_u             [[buffer(24)]],  // [nz, nc, n_sub] (in/out)
    device float* crm_v             [[buffer(25)]],  // [nz, nc, n_sub] (in/out)
    device float* rain_3d           [[buffer(26)]],  // [nc, n_sub] (out)
    device float* snow_3d           [[buffer(27)]],  // [nc, n_sub] (out)
    constant ulong4& u_pack         [[buffer(28)]],  // [nz, nc, (ny<<16)|nx, (step_shift<<3)|(has_cp<<2)|(has_sp<<1)|has_sc]
    constant float4& f_p            [[buffer(29)]],  // [dt, cp_d, damp_anom, 0]
    uint c                          [[threadgroup_position_in_grid]],
    uint s                          [[thread_position_in_threadgroup]])
{
    uint nz = uint(u_pack.x);
    uint nc = uint(u_pack.y);
    uint ny_t = uint(u_pack.z >> 16u);
    uint nx_t = uint(u_pack.z & 0xFFFFu);
    uint n_sub = ny_t * nx_t;
    uint flags = uint(u_pack.w);
    uint has_sc = flags & 1u;
    uint has_sp = (flags >> 1u) & 1u;
    uint has_cp = (flags >> 2u) & 1u;
    uint step_shift = flags >> 3u;
    if (c >= nc || s >= n_sub || n_sub <= 1u || n_sub > 64u) return;

    threadgroup float sm_a[64];
    threadgroup float sm_b[64];
    threadgroup float sm_c[64];
    threadgroup float sm_d[64];
    threadgroup float sm_red[4];

    float fn_sub = float(n_sub);
    float inv_nsub = 1.0f / fn_sub;
    uint shift = (c + step_shift) % n_sub;
    uint base_c = c * n_sub;
    float rfb_s = float((s + shift) % n_sub);
    float tb_s = (1.0f - rfb_s * inv_nsub) * 1.0e-4f;

    // Parallel cloud rank across all n_sub tiles in shared memory (O(n_sub) per thread)
    float rank_cld_s = rfb_s;
    if (has_sc != 0u) {
        float ks = score_cld[base_c + s] + tb_s;
        sm_a[s] = ks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float r = 0.0f;
        for (uint j = 0u; j < n_sub; ++j) {
            float kj = sm_a[j];
            if (kj > ks || (kj == ks && j < s)) r += 1.0f;
        }
        rank_cld_s = r;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Parallel precipitation subcolumn multipliers -> rain_3d, snow_3d (coalesced)
    float rain_c = stu_1d[c];
    float snow_c = stu_1d[nc + c];
    float f_p_eff = clamp(stu_1d[2u * nc + c], inv_nsub, 1.0f);
    float fp_n = f_p_eff * fn_sub;
    float rank_prec_s = rank_cld_s;
    if (has_sp != 0u) {
        float ks = score_prec[base_c + s] + tb_s;
        sm_a[s] = ks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rp = 0.0f;
        for (uint j = 0u; j < n_sub; ++j) {
            float kj = sm_a[j];
            if (kj > ks || (kj == ks && j < s)) rp += 1.0f;
        }
        rank_prec_s = rp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float wp_s = clamp(fp_n - rank_prec_s, 0.0f, 1.0f);
    sm_a[s] = wp_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0u) {
        float sw = 0.0f;
        for (uint j = 0u; j < n_sub; ++j) sw += sm_a[j];
        sm_red[0] = 1.0f / max(sw * inv_nsub, 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m_p_s = wp_s * sm_red[0];
    rain_3d[base_c + s] = rain_c * m_p_s;
    snow_3d[base_c + s] = snow_c * m_p_s;

    // Surface microclimate + cold-pool (k == 0) precomputation in shared memory
    float dt = f_p.x;
    float cp_d = f_p.y;
    float damp = f_p.z;
    float dm_0 = max(stu_1d[3u * nc + c] * stu_1d[4u * nc + c], 1.0f);
    float ex_0 = max(stu_1d[5u * nc + c], 0.1f);
    float inv_cp_dm_ex = dt / (cp_d * dm_0 * ex_0);
    float inv_dm = dt / dm_0;

    float shf_s = shf_tile[base_c + s];
    float evap_s = evap_tile[base_c + s];
    sm_a[s] = shf_s;
    sm_b[s] = evap_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0u) {
        float s_shf = 0.0f;
        float s_evap = 0.0f;
        for (uint j = 0u; j < n_sub; ++j) {
            s_shf += sm_a[j];
            s_evap += sm_b[j];
        }
        sm_red[0] = s_shf * inv_nsub;
        sm_red[1] = s_evap * inv_nsub;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float shf_mean = sm_red[0];
    float evap_mean = sm_red[1];

    float dt_v_s;
    float dq_v_s;
    float dthm_cp_s = 0.0f;
    if (has_cp != 0u) {
        dthm_cp_s = dthm_cp[base_c + s];
        dt_v_s = clamp(dthm_cp_s, -8.0f, 8.0f);
        dq_v_s = clamp(dqv_cp[base_c + s], -6.0e-3f, 6.0e-3f);
    } else {
        dt_v_s = clamp((shf_s - shf_mean) * inv_cp_dm_ex, -3.0f, 3.0f);
        dq_v_s = clamp((evap_s - evap_mean) * inv_dm, -2.0e-3f, 2.0e-3f);
    }
    sm_a[s] = dt_v_s;
    sm_b[s] = dq_v_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0u) {
        float s_dt = 0.0f;
        float s_dq = 0.0f;
        for (uint j = 0u; j < n_sub; ++j) {
            s_dt += sm_a[j];
            s_dq += sm_b[j];
        }
        sm_red[0] = s_dt * inv_nsub;
        sm_red[1] = s_dq * inv_nsub;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dthm_anom_s = dt_v_s - sm_red[0];
    float dqv_anom_s = dq_v_s - sm_red[1];

    float qv_0_c = stu_qv[c];
    float qv_cand_s = max(qv_0_c + dqv_anom_s, 0.0f);
    sm_a[s] = qv_cand_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0u) {
        float s_qvc = 0.0f;
        for (uint j = 0u; j < n_sub; ++j) s_qvc += sm_a[j];
        float qcm = s_qvc * inv_nsub;
        sm_red[0] = (qcm > 1.0e-10f) ? (qv_0_c / max(qcm, 1.0e-10f)) : -1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qv_scale_0 = sm_red[0];
    float qv_0_mc_s = (qv_scale_0 >= 0.0f) ? (qv_cand_s * qv_scale_0) : qv_0_c;

    // Cold-pool radial density-current outflow at k == 0 (single shared-mem neighbor stencil)
    float u_cp_anom_s = 0.0f;
    float v_cp_anom_s = 0.0f;
    if (has_cp != 0u) {
        sm_d[s] = dthm_cp_s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint iy = s / nx_t;
        uint ix = s - iy * nx_t;
        float gx = 0.0f;
        float gy = 0.0f;
        if (nx_t > 1u) {
            uint ixp = (ix + 1u < nx_t) ? (ix + 1u) : 0u;
            uint ixm = (ix > 0u) ? (ix - 1u) : (nx_t - 1u);
            gx = 0.5f * (sm_d[iy * nx_t + ixp] - sm_d[iy * nx_t + ixm]);
        }
        if (ny_t > 1u) {
            uint iyp = (iy + 1u < ny_t) ? (iy + 1u) : 0u;
            uint iym = (iy > 0u) ? (iy - 1u) : (ny_t - 1u);
            gy = 0.5f * (sm_d[iyp * nx_t + ix] - sm_d[iym * nx_t + ix]);
        }
        float gn = sqrt(gx * gx + gy * gy);
        float ang = (6.28318530718f * inv_nsub) * float(s + shift);
        float ex = (gn > 1.0e-4f) ? (gx / max(gn, 1.0e-4f)) : cos(ang);
        float ey = (gn > 1.0e-4f) ? (gy / max(gn, 1.0e-4f)) : sin(ang);
        float ug = u_gust[base_c + s];
        float ur_s = ug * ex;
        float vr_s = ug * ey;
        sm_a[s] = ur_s;
        sm_b[s] = vr_s;
        sm_c[s] = ug * ug;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s == 0u) {
            float s_ur = 0.0f, s_vr = 0.0f, s_ug2 = 0.0f;
            for (uint j = 0u; j < n_sub; ++j) {
                s_ur += sm_a[j];
                s_vr += sm_b[j];
                s_ug2 += sm_c[j];
            }
            sm_red[0] = s_ur * inv_nsub;
            sm_red[1] = s_vr * inv_nsub;
            sm_red[2] = s_ug2 * inv_nsub;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float up_s = ur_s - sm_red[0];
        float vp_s = vr_s - sm_red[1];
        float eke_tgt = sm_red[2];
        sm_a[s] = up_s * up_s + vp_s * vp_s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s == 0u) {
            float s_eke = 0.0f;
            for (uint j = 0u; j < n_sub; ++j) s_eke += sm_a[j];
            float eke_cur = s_eke * inv_nsub;
            sm_red[3] = (eke_cur > 1.0e-8f) ? min(sqrt(eke_tgt / max(eke_cur, 1.0e-8f)), 2.0f) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        u_cp_anom_s = up_s * sm_red[3];
        v_cp_anom_s = vp_s * sm_red[3];
    }

    // Vertical column update (k = 0..nz-1): 100% coalesced across tile index s
    for (uint k = 0u; k < nz; ++k) {
        uint kc = k * nc + c;
        uint idx = kc * n_sub + s;
        float rho_c = atmos_rho[kc];
        float thm_c = stu_thm[kc];
        float qv_c  = stu_qv[kc];
        float qc_c  = stu_qc[kc];
        float qi_c  = stu_qi[kc];
        float qp_c  = stu_qp[kc];
        float u_c   = stu_u[kc];
        float v_c   = stu_v[kc];
        float fc_phys = max(stu_fcld[kc], (qc_c + qi_c) / 0.006f);
        float fc_n  = clamp(fc_phys, inv_nsub, 1.0f) * fn_sub;

        float thm_old = crm_thm[idx];
        float u_old   = crm_u[idx];
        float v_old   = crm_v[idx];
        float wc_s    = clamp(fc_n - rank_cld_s, 0.0f, 1.0f);
        sm_a[s] = thm_old;
        sm_b[s] = u_old;
        sm_c[s] = v_old;
        sm_d[s] = wc_s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s == 0u) {
            float s_thm = 0.0f, s_u = 0.0f, s_v = 0.0f, s_wc = 0.0f;
            for (uint j = 0u; j < n_sub; ++j) {
                s_thm += sm_a[j];
                s_u   += sm_b[j];
                s_v   += sm_c[j];
                s_wc  += sm_d[j];
            }
            sm_red[0] = s_thm * inv_nsub;
            sm_red[1] = s_u * inv_nsub;
            sm_red[2] = s_v * inv_nsub;
            sm_red[3] = 1.0f / max(s_wc * inv_nsub, 1.0e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_c = wc_s * sm_red[3];
        float qc_s = qc_c * m_c;
        float qi_s = qi_c * m_c;
        float qp_s = qp_c * m_c;
        float thm_s = thm_c + damp * (thm_old - sm_red[0]);
        float qv_s = qv_c;
        float u_s = u_c + damp * (u_old - sm_red[1]);
        float v_s = v_c + damp * (v_old - sm_red[2]);
        if (k == 0u) {
            thm_s = thm_c + dthm_anom_s;
            qv_s = qv_0_mc_s;
            if (has_cp != 0u) {
                u_s = u_c + u_cp_anom_s;
                v_s = v_c + v_cp_anom_s;
            }
        }
        crm_rho[idx] = rho_c;
        crm_thm[idx] = thm_s;
        crm_qv[idx]  = qv_s;
        crm_qc[idx]  = qc_s;
        crm_qi[idx]  = qi_s;
        crm_qp[idx]  = qp_s;
        crm_qn[idx]  = qv_s + qc_s + qi_s;
        crm_u[idx]   = u_s;
        crm_v[idx]   = v_s;
    }
}

// =========================================================================
// 24. Fused Surface 3-Source Land Newton & Ocean/Sea-Ice Post Kernels
// =========================================================================

inline void aoba_qsat_mixed_d(
    float temp, float pres, float eps_w, float e_triple,
    float t_frz, float t_ice_all, float t_liq_all,
    thread float& qsat_out, thread float& dq_out)
{
    float omega = clamp((temp - t_ice_all) / (t_liq_all - t_ice_all), 0.0f, 1.0f);
    float tw = max(temp, 160.0f);
    float ti = max(temp, 120.0f);
    float tc = tw - t_frz;
    float ew = e_triple * exp(17.67f * tc / (tc + 243.50f));
    float ei = e_triple * exp(21.8745584f * (ti - t_frz) / (ti - 7.66f));
    float dw = max(pres - ew, 0.05f * pres);
    float di = max(pres - ei, 0.05f * pres);
    float qsw = eps_w * ew / dw;
    float qsi = eps_w * ei / di;
    qsat_out = omega * qsw + (1.0f - omega) * qsi;

    float d_tw = tc + 243.50f;
    float d_ti = ti - 7.66f;
    float dew_dT = ew * (17.67f * 243.50f) / (d_tw * d_tw);
    float dei_dT = ei * (21.8745584f * (t_frz - 7.66f)) / (d_ti * d_ti);
    float dqsw = eps_w * pres * dew_dT / (dw * dw);
    float dqsi = eps_w * pres * dei_dT / (di * di);
    float domega = (temp > t_ice_all && temp < t_liq_all) ? (1.0f / (t_liq_all - t_ice_all)) : 0.0f;
    dq_out = omega * dqsw + (1.0f - omega) * dqsi + domega * (qsw - qsi);
}

kernel void surface_land_pre_bio_col(
    device const float* t_sun_cur     [[buffer(0)]],
    device const float* t_sh_cur      [[buffer(1)]],
    device const float* t_gnd_cur     [[buffer(2)]],
    device const float* q_af_in       [[buffer(3)]],
    device const float* c_h_land      [[buffer(4)]],
    device const float* wind_eff_land [[buffer(5)]],
    device const float* u_star_land   [[buffer(6)]],
    device const float* pb_const      [[buffer(7)]],   // [11, N]
    device float* pb_out              [[buffer(8)]],   // [13, N]
    constant uint& n_tot              [[buffer(9)]],
    constant float4& f_c0             [[buffer(10)]],  // [gravity, von_karman, eps_water, e_triple]
    constant float4& f_c1             [[buffer(11)]],  // [t_freeze, t_ice_all, t_liq_all, 0]
    uint i                            [[thread_position_in_grid]])
{
    if (i >= n_tot) return;
    uint N = n_tot;
    float pres_sfc     = pb_const[i];
    float beta_rsl     = pb_const[N + i];
    float alpha_u_half = pb_const[2u * N + i];
    float lai_exp      = pb_const[3u * N + i];
    float k_db_inst    = pb_const[4u * N + i];
    float h_exp        = pb_const[5u * N + i];
    float z0m_terr     = pb_const[6u * N + i];
    float f_veg_rad    = pb_const[7u * N + i];
    float f_sun_area   = pb_const[8u * N + i];
    float f_sh_area    = pb_const[9u * N + i];
    uint flags         = uint(pb_const[10u * N + i] + 0.5f);
    bool has_veg = (flags & 1u) != 0u;
    bool has_sun = (flags & 2u) != 0u;
    bool has_sh  = (flags & 4u) != 0u;

    float ts = t_sun_cur[i];
    float tsh = t_sh_cur[i];
    float tg = t_gnd_cur[i];
    float ch = c_h_land[i];
    float weff = wind_eff_land[i];
    float ust = u_star_land[i];

    float ga_land = ch * weff;
    float u_can_top = min(max(ust / beta_rsl, 0.10f), weff);
    float gb_top = 0.01f * sqrt(u_can_top / 0.04f);
    float gb_tot_can = gb_top * (1.0f - exp(-min(alpha_u_half * lai_exp, 30.0f))) / alpha_u_half;
    float k_sum = k_db_inst + alpha_u_half;
    float gb_sun = has_sun ? max(gb_top * (1.0f - exp(-min(k_sum * lai_exp, 30.0f))) / k_sum, 1.0e-4f) : 0.0f;
    float gb_sh = has_sh ? max(gb_tot_can - gb_sun, 1.0e-4f) : 0.0f;

    float t_fol_avg = has_veg ? (f_sun_area * ts + f_sh_area * tsh) : tg;
    float stab_uc = f_c0.x * h_exp * (t_fol_avg - tg) / (max(tg, 180.0f) * (ust * ust + 1.0e-3f));
    float c_dense = (stab_uc > 0.0f) ? (0.004f / (1.0f + 0.5f * min(stab_uc, 10.0f))) : 0.004f;
    float c_bare = f_c0.y / (0.13f * pow(max(ust * z0m_terr / 1.5e-5f, 1.0f), 0.45f));
    float c_uv = (1.0f - f_veg_rad) * c_bare + f_veg_rad * c_dense;
    float r_d_uc = 1.0f / max(c_uv * ust, 1.0e-3f);

    float qs_sun, dq_sun, qs_sh, dq_sh, qs_gnd, dq_gnd;
    aoba_qsat_mixed_d(ts,  pres_sfc, f_c0.z, f_c0.w, f_c1.x, f_c1.y, f_c1.z, qs_sun, dq_sun);
    aoba_qsat_mixed_d(tsh, pres_sfc, f_c0.z, f_c0.w, f_c1.x, f_c1.y, f_c1.z, qs_sh,  dq_sh);
    aoba_qsat_mixed_d(tg,  pres_sfc, f_c0.z, f_c0.w, f_c1.x, f_c1.y, f_c1.z, qs_gnd, dq_gnd);

    float q_af_vpd = max(q_af_in[i], 1.0e-7f);
    float e_af_pa = q_af_vpd * pres_sfc / (f_c0.z + (1.0f - f_c0.z) * q_af_vpd);
    float tc_sun = max(ts, 160.0f) - f_c1.x;
    float tc_sh  = max(tsh, 160.0f) - f_c1.x;
    float es_sun = f_c0.w * exp(17.67f * tc_sun / (tc_sun + 243.50f));
    float es_sh  = f_c0.w * exp(17.67f * tc_sh / (tc_sh + 243.50f));
    float vpd_sun_af = max(es_sun - e_af_pa, 20.0f);
    float vpd_sh_af  = max(es_sh  - e_af_pa, 20.0f);

    pb_out[i]           = ga_land;
    pb_out[N + i]       = gb_sun;
    pb_out[2u * N + i]  = gb_sh;
    pb_out[3u * N + i]  = r_d_uc;
    pb_out[4u * N + i]  = qs_sun;
    pb_out[5u * N + i]  = dq_sun;
    pb_out[6u * N + i]  = qs_sh;
    pb_out[7u * N + i]  = dq_sh;
    pb_out[8u * N + i]  = qs_gnd;
    pb_out[9u * N + i]  = dq_gnd;
    pb_out[10u * N + i] = q_af_vpd;
    pb_out[11u * N + i] = vpd_sun_af;
    pb_out[12u * N + i] = vpd_sh_af;
}

kernel void surface_land_newton_jac_col(
    device const float* jac_const   [[buffer(0)]],   // [21, N]
    device const float* pb_out      [[buffer(1)]],   // [13, N]
    device const float* t_sun_cur   [[buffer(2)]],
    device const float* t_sh_cur    [[buffer(3)]],
    device const float* t_gnd_cur   [[buffer(4)]],
    device const float* gsw_sun_in  [[buffer(5)]],
    device const float* gsw_sh_in   [[buffer(6)]],
    device const float* prev_dt_sun [[buffer(7)]],
    device const float* prev_dt_sh  [[buffer(8)]],
    device const float* prev_dt_gnd [[buffer(9)]],
    device float* jac_out           [[buffer(10)]],  // [33, N]
    constant ulong4& u_pack         [[buffer(11)]],  // [N, apply_damping, write_full_diag, 0]
    constant float4& f_p0           [[buffer(12)]],  // [dt, eps_g, sig, t_freeze]
    constant float4& f_p1           [[buffer(13)]],  // [l_v0, l_s0, 0, 0]
    uint i                          [[thread_position_in_grid]])
{
    uint N = uint(u_pack.x);
    if (i >= N) return;
    bool apply_damping = (u_pack.y != 0u);
    bool write_full = (u_pack.z != 0u);

    float dt       = f_p0.x;
    float eps_g    = f_p0.y;
    float sig      = f_p0.z;
    float t_freeze = f_p0.w;
    float l_v0     = f_p1.x;
    float l_s0     = f_p1.y;

    float t_soil0        = jac_const[i];
    float t_air_sfc      = jac_const[N + i];
    float qv_1           = jac_const[2u * N + i];
    float r_soil         = jac_const[3u * N + i];
    float h_soil_base    = jac_const[4u * N + i];
    float a_sn           = jac_const[5u * N + i];
    float f_wet          = jac_const[6u * N + i];
    float w_canopy       = jac_const[7u * N + i];
    float rho_sfc        = jac_const[8u * N + i];
    float cp_air         = jac_const[9u * N + i];
    float f_veg_rad      = jac_const[10u * N + i];
    float f_sun_area     = jac_const[11u * N + i];
    float f_sh_area      = jac_const[12u * N + i];
    float lw_refl_c      = jac_const[13u * N + i];
    float lw_emit_c_coef = jac_const[14u * N + i];
    float sw_abs_sun     = jac_const[15u * N + i];
    float sw_abs_sh      = jac_const[16u * N + i];
    float sw_abs_gnd     = jac_const[17u * N + i];
    float lw_down        = jac_const[18u * N + i];
    float g_cond_coeff   = jac_const[19u * N + i];
    uint flags           = uint(jac_const[20u * N + i] + 0.5f);
    float t_sun_t0       = jac_const[21u * N + i];
    float t_sh_t0        = jac_const[22u * N + i];
    float c_sun_dt       = jac_const[23u * N + i];
    float c_sh_dt        = jac_const[24u * N + i];
    bool has_veg  = (flags & 1u) != 0u;
    bool has_sun  = (flags & 2u) != 0u;
    bool has_sh   = (flags & 4u) != 0u;
    bool has_snow = (flags & 8u) != 0u;

    float ga_land  = pb_out[i];
    float gb_sun   = pb_out[N + i];
    float gb_sh    = pb_out[2u * N + i];
    float r_d_uc   = pb_out[3u * N + i];
    float qsat_sun = pb_out[4u * N + i];
    float dq_sun   = pb_out[5u * N + i];
    float qsat_sh  = pb_out[6u * N + i];
    float dq_sh    = pb_out[7u * N + i];
    float qsat_gnd = pb_out[8u * N + i];
    float dq_gnd   = pb_out[9u * N + i];
    float q_af_vpd = pb_out[10u * N + i];

    float t_sun = t_sun_cur[i];
    float t_sh  = t_sh_cur[i];
    float t_gnd = t_gnd_cur[i];
    float gsw_sun = gsw_sun_in[i];
    float gsw_sh  = gsw_sh_in[i];

    float e_pot_wet = rho_sfc * (gb_sun * (qsat_sun - q_af_vpd) + gb_sh * (qsat_sh - q_af_vpd));
    bool dew_on_leaf = (e_pot_wet <= 0.0f) && has_veg;
    float f_wet_eff = (e_pot_wet > 1.0e-12f) ? min(f_wet, max(w_canopy, 0.0f) / (dt * max(e_pot_wet, 1.0e-12f))) : f_wet;
    if (dew_on_leaf) f_wet_eff = 1.0f;

    float g_trans_sun = (dew_on_leaf || !has_sun) ? 0.0f : ((1.0f - f_wet) * (gb_sun * gsw_sun) / max(gb_sun + gsw_sun, 1.0e-6f));
    float g_trans_sh  = (dew_on_leaf || !has_sh)  ? 0.0f : ((1.0f - f_wet) * (gb_sh * gsw_sh)   / max(gb_sh + gsw_sh,   1.0e-6f));
    float g_inter_sun = has_sun ? (f_wet_eff * gb_sun) : 0.0f;
    float g_inter_sh  = has_sh  ? (f_wet_eff * gb_sh)  : 0.0f;
    float gw_0 = g_trans_sun + g_inter_sun;
    float gw_1 = g_trans_sh + g_inter_sh;

    float q_soil_ref = has_veg ? q_af_vpd : qv_1;
    bool dry_soil_cap = (qsat_gnd > q_soil_ref) && (q_soil_ref > h_soil_base * qsat_gnd);
    float h_soil = dry_soil_cap ? (q_soil_ref / max(qsat_gnd, 1.0e-8f)) : h_soil_base;
    float g_soil_sf = (1.0f - a_sn) / max(r_d_uc + r_soil, 1.0e-4f);
    float g_snow_sb = a_sn / max(r_d_uc, 1.0e-4f);
    float gw_2 = g_soil_sf + g_snow_sb;
    float h_gnd = (gw_2 > 1.0e-8f) ? ((g_snow_sb + g_soil_sf * h_soil) / max(gw_2, 1.0e-8f)) : h_soil;
    float dhq_num = g_snow_sb + (dry_soil_cap ? 0.0f : (g_soil_sf * h_soil));
    float dhq_gnd_dT = ((gw_2 > 1.0e-8f) ? (dhq_num / max(gw_2, 1.0e-8f)) : (dry_soil_cap ? 0.0f : h_soil)) * dq_gnd;

    float gh_0 = 2.0f * gb_sun;
    float gh_1 = 2.0f * gb_sh;
    float gh_2 = 1.0f / r_d_uc;
    float gh_sum = ga_land + gh_0 + gh_1 + gh_2;
    float t_af = has_veg ? ((ga_land * t_air_sfc + gh_0 * t_sun + gh_1 * t_sh + gh_2 * t_gnd) / gh_sum) : t_gnd;
    float gw_sum = ga_land + gw_0 + gw_1 + gw_2;
    float q_af = (ga_land * qv_1 + gw_0 * qsat_sun + gw_1 * qsat_sh + gw_2 * h_gnd * qsat_gnd) / gw_sum;

    float wh_0 = has_veg ? (gh_0 / gh_sum) : 0.0f;
    float wh_1 = has_veg ? (gh_1 / gh_sum) : 0.0f;
    float wh_2 = has_veg ? (gh_2 / gh_sum) : 1.0f;
    float ww_0 = gw_0 / gw_sum;
    float ww_1 = gw_1 / gw_sum;
    float ww_2 = gw_2 / gw_sum;

    float l_vap_sun = (t_sun < t_freeze) ? l_s0 : l_v0;
    float l_vap_sh  = (t_sh  < t_freeze) ? l_s0 : l_v0;
    float l_vap_gnd = (a_sn > 0.5f || t_gnd < t_freeze) ? l_s0 : l_v0;

    float t0_3 = t_sun * t_sun * t_sun;
    float t1_3 = t_sh  * t_sh  * t_sh;
    float t2_3 = t_gnd * t_gnd * t_gnd;
    float t0_4 = t0_3 * t_sun;
    float t1_4 = t1_3 * t_sh;
    float t2_4 = t2_3 * t_gnd;

    float dc_sun = f_veg_rad * f_sun_area;
    float dc_sh  = f_veg_rad * f_sh_area;
    float r_lw_0 = dc_sun * (lw_refl_c * lw_down + eps_g * sig * t2_4 - lw_emit_c_coef * sig * t0_4);
    float r_lw_1 = dc_sh  * (lw_refl_c * lw_down + eps_g * sig * t2_4 - lw_emit_c_coef * sig * t1_4);
    float r_lw_2 = eps_g * (1.0f - f_veg_rad) * lw_down + eps_g * sig * (dc_sun * t0_4 + dc_sh * t1_4) - eps_g * sig * t2_4;

    float rho_cp = rho_sfc * cp_air;
    float h_0 = rho_cp * gh_0 * (t_sun - t_af);
    float h_1 = rho_cp * gh_1 * (t_sh  - t_af);
    float h_2 = has_veg ? (rho_cp * gh_2 * (t_gnd - t_af)) : (rho_cp * ga_land * (t_gnd - t_air_sfc));

    float e_0 = rho_sfc * gw_0 * (qsat_sun - q_af);
    float e_1 = rho_sfc * gw_1 * (qsat_sh  - q_af);
    float e_2 = max(rho_sfc * gw_2 * (h_gnd * qsat_gnd - q_af), -rho_sfc * ga_land * qv_1);

    float g0_down = g_cond_coeff * (t_gnd - t_soil0);
    float b0 = sw_abs_sun + r_lw_0 - h_0 - l_vap_sun * e_0 - c_sun_dt * (t_sun - t_sun_t0);
    float b1 = sw_abs_sh  + r_lw_1 - h_1 - l_vap_sh  * e_1 - c_sh_dt  * (t_sh  - t_sh_t0);
    float b2 = sw_abs_gnd + r_lw_2 - h_2 - l_vap_gnd * e_2 - g0_down;

    float k_lw_0 = 4.0f * dc_sun * sig;
    float k_lw_1 = 4.0f * dc_sh  * sig;
    float lw_g_3 = 4.0f * eps_g * sig * t2_3;

    float j00 = rho_cp * gh_0 * (1.0f - wh_0) + l_vap_sun * rho_sfc * gw_0 * dq_sun * (1.0f - ww_0) + k_lw_0 * lw_emit_c_coef * t0_3 + c_sun_dt;
    float j01 = -rho_cp * gh_0 * wh_1 - l_vap_sun * rho_sfc * gw_0 * ww_1 * dq_sh;
    float j02 = -rho_cp * gh_0 * wh_2 - l_vap_sun * rho_sfc * gw_0 * ww_2 * dhq_gnd_dT - dc_sun * lw_g_3;

    float j10 = -rho_cp * gh_1 * wh_0 - l_vap_sh * rho_sfc * gw_1 * ww_0 * dq_sun;
    float j11 = rho_cp * gh_1 * (1.0f - wh_1) + l_vap_sh * rho_sfc * gw_1 * dq_sh * (1.0f - ww_1) + k_lw_1 * lw_emit_c_coef * t1_3 + c_sh_dt;
    float j12 = -rho_cp * gh_1 * wh_2 - l_vap_sh * rho_sfc * gw_1 * ww_2 * dhq_gnd_dT - dc_sh * lw_g_3;

    float dh2_dt2 = has_veg ? (rho_cp * gh_2 * (1.0f - wh_2)) : (rho_cp * ga_land);
    float j20 = has_veg ? (-rho_cp * gh_2 * wh_0 - l_vap_gnd * rho_sfc * gw_2 * ww_0 * dq_sun - eps_g * k_lw_0 * t0_3) : 0.0f;
    float j21 = has_veg ? (-rho_cp * gh_2 * wh_1 - l_vap_gnd * rho_sfc * gw_2 * ww_1 * dq_sh  - eps_g * k_lw_1 * t1_3) : 0.0f;
    float j22 = dh2_dt2 + l_vap_gnd * rho_sfc * gw_2 * dhq_gnd_dT * (1.0f - ww_2) + g_cond_coeff + lw_g_3;

    float t_pin_sun = has_sh  ? t_sh  : t_gnd;
    float t_pin_sh  = has_sun ? t_sun : t_gnd;

    float a00 = has_sun ? max(j00, 1.0f) : 1.0f;
    float a01 = has_sun ? j01 : 0.0f;
    float a02 = has_sun ? j02 : 0.0f;
    float b0s = has_sun ? b0  : (t_pin_sun - t_sun);

    float a10 = has_sh ? j10 : 0.0f;
    float a11 = has_sh ? max(j11, 1.0f) : 1.0f;
    float a12 = has_sh ? j12 : 0.0f;
    float b1s = has_sh ? b1  : (t_pin_sh - t_sh);

    float a22 = max(j22, 1.0f);

    float c00 = a11 * a22 - a12 * j21;
    float c01 = a12 * j20 - a10 * a22;
    float c02 = a10 * j21 - a11 * j20;
    float det = a00 * c00 + a01 * c01 + a02 * c02;
    float inv_det = 1.0f / ((fabs(det) < 1.0e-12f) ? 1.0e-12f : det);

    float c10 = a02 * j21 - a01 * a22;
    float c11 = a00 * a22 - a02 * j20;
    float c12 = a01 * j20 - a00 * j21;
    float c20 = a01 * a12 - a02 * a11;
    float c21 = a02 * a10 - a00 * a12;
    float c22 = a00 * a11 - a01 * a10;

    float dt0 = clamp((c00 * b0s + c10 * b1s + c20 * b2) * inv_det, -10.0f, 10.0f);
    float dt1 = clamp((c01 * b0s + c11 * b1s + c21 * b2) * inv_det, -10.0f, 10.0f);
    float dt2 = clamp((c02 * b0s + c12 * b1s + c22 * b2) * inv_det, -10.0f, 10.0f);
    if (apply_damping) {
        if (dt0 * prev_dt_sun[i] < 0.0f) dt0 *= 0.5f;
        if (dt1 * prev_dt_sh[i]  < 0.0f) dt1 *= 0.5f;
        if (dt2 * prev_dt_gnd[i] < 0.0f) dt2 *= 0.5f;
    }

    float t_sun_new = clamp(t_sun + dt0, 150.0f, 370.0f);
    float t_sh_new  = clamp(t_sh  + dt1, 150.0f, 370.0f);
    float t_gnd_new = clamp(t_gnd + dt2, 150.0f, 370.0f);

    float melt_excess_t = (has_snow && t_gnd_new > t_freeze) ? (t_gnd_new - t_freeze) : 0.0f;
    float t_gnd_final = t_gnd_new - melt_excess_t;

    jac_out[i]           = t_sun_new;
    jac_out[N + i]       = t_sh_new;
    jac_out[2u * N + i]  = t_gnd_new;
    jac_out[3u * N + i]  = t_gnd_final;
    jac_out[4u * N + i]  = dt0;
    jac_out[5u * N + i]  = dt1;
    jac_out[6u * N + i]  = dt2;
    jac_out[7u * N + i]  = t_gnd_final - t_gnd;
    jac_out[8u * N + i]  = melt_excess_t * a22;
    jac_out[9u * N + i]  = q_af;
    if (write_full) {
        jac_out[10u * N + i] = wh_0;
        jac_out[11u * N + i] = wh_1;
        jac_out[12u * N + i] = wh_2;
        jac_out[13u * N + i] = ww_0;
        jac_out[14u * N + i] = ww_1;
        jac_out[15u * N + i] = ww_2;
        jac_out[16u * N + i] = gh_0;
        jac_out[17u * N + i] = gh_1;
        jac_out[18u * N + i] = gh_2;
        jac_out[19u * N + i] = rho_cp;
        jac_out[20u * N + i] = h_0;
        jac_out[21u * N + i] = h_1;
        jac_out[22u * N + i] = h_2;
        jac_out[23u * N + i] = g_trans_sun;
        jac_out[24u * N + i] = g_trans_sh;
        jac_out[25u * N + i] = g_inter_sun;
        jac_out[26u * N + i] = g_inter_sh;
        jac_out[27u * N + i] = g_soil_sf;
        jac_out[28u * N + i] = g_snow_sb;
        jac_out[29u * N + i] = h_soil;
        jac_out[30u * N + i] = dry_soil_cap ? 1.0f : 0.0f;
        jac_out[31u * N + i] = dhq_gnd_dT;
        jac_out[32u * N + i] = a22;
    }
}

kernel void surface_ocean_seaice_post_col(
    device const float* t_ocean_ml      [[buffer(0)]],
    device const float* t_ocean_ml_comp [[buffer(1)]],
    device const float* h_ocean_ml      [[buffer(2)]],
    device const float* m_seaice        [[buffer(3)]],
    device const float* t_seaice        [[buffer(4)]],
    device const float* swe_seaice      [[buffer(5)]],
    device const float* f_ice_in        [[buffer(6)]],
    device const float* qsat_ow_in      [[buffer(7)]],
    device const float* qsat_si_in      [[buffer(8)]],
    device const float* ga_h_o_in       [[buffer(9)]],
    device const float* ga_q_o_in       [[buffer(10)]],
    device const float* rho_sfc_in      [[buffer(11)]],
    device const float* cp_air_in       [[buffer(12)]],
    device const float* t_air_sfc_in    [[buffer(13)]],
    device const float* qv_1_in         [[buffer(14)]],
    device const float* sw1_dir_in      [[buffer(15)]],
    device const float* sw1_dif_in      [[buffer(16)]],
    device const float* sw2_dir_in      [[buffer(17)]],
    device const float* sw2_dif_in      [[buffer(18)]],
    device const float* lw_down_in      [[buffer(19)]],
    device const float* precip_snow_in  [[buffer(20)]],
    device const float* mu0_in          [[buffer(21)]],
    device float* ocn_out               [[buffer(22)]],  // [9, N]
    constant uint& n_tot                [[buffer(23)]],
    constant float4& f_p0               [[buffer(24)]],  // [dt, emiss_ocn, sig, t_freeze]
    constant float4& f_p1               [[buffer(25)]],  // [t_freeze_ocn, l_v0, l_s0, l_f0]
    constant float4& f_p2               [[buffer(26)]],  // [rho_sw, cp_sw, rho_si, k_si]
    constant float4& f_p3               [[buffer(27)]],  // [rho_snow, k_snow, alb_si_v, alb_si_n]
    constant float4& f_p4               [[buffer(28)]],  // [alb_sn_v, alb_sn_n, 0, 0]
    uint i                              [[thread_position_in_grid]])
{
    if (i >= n_tot) return;
    uint N = n_tot;
    float dt        = f_p0.x;
    float emiss_ocn = f_p0.y;
    float sig       = f_p0.z;
    float t_frz     = f_p0.w;
    float t_frz_ocn = f_p1.x;
    float l_v0      = f_p1.y;
    float l_s0      = f_p1.z;
    float l_f0      = f_p1.w;
    float rho_sw    = f_p2.x;
    float cp_sw     = f_p2.y;
    float rho_si    = f_p2.z;
    float k_si      = f_p2.w;
    float rho_snow  = f_p3.x;
    float k_snow    = f_p3.y;
    float alb_si_v  = f_p3.z;
    float alb_si_n  = f_p3.w;
    float alb_sn_v  = f_p4.x;
    float alb_sn_n  = f_p4.y;

    float t_ml    = t_ocean_ml[i];
    float comp0   = t_ocean_ml_comp[i];
    float h_ml    = h_ocean_ml[i];
    float m_si    = m_seaice[i];
    float t_si    = t_seaice[i];
    float swe_si  = swe_seaice[i];
    float f_ice   = f_ice_in[i];
    float qsat_ow = qsat_ow_in[i];
    float qsat_si = qsat_si_in[i];
    float ga_h    = ga_h_o_in[i];
    float ga_q    = ga_q_o_in[i];
    float rho_sfc = rho_sfc_in[i];
    float cp_air  = cp_air_in[i];
    float t_air   = t_air_sfc_in[i];
    float qv_1    = qv_1_in[i];
    float sw1_d   = sw1_dir_in[i];
    float sw1_f   = sw1_dif_in[i];
    float sw2_d   = sw2_dir_in[i];
    float sw2_f   = sw2_dif_in[i];
    float lw_dn   = lw_down_in[i];
    float p_sn    = precip_snow_in[i];
    float mu      = clamp(mu0_in[i], 0.01f, 1.0f);

    float rho_cp_ga = rho_sfc * cp_air * ga_h;
    float shf_ow0 = rho_cp_ga * (t_ml - t_air);
    float evap_ow0 = rho_sfc * ga_q * (qsat_ow - qv_1);
    float lhf_ow0 = l_v0 * evap_ow0;
    float t_ml_2 = t_ml * t_ml;
    float t_ml_3 = t_ml_2 * t_ml;
    float t_ml_4 = t_ml_2 * t_ml_2;
    float lw_emit_ow0 = emiss_ocn * sig * t_ml_4;
    float lw_up_ow0 = lw_emit_ow0 + (1.0f - emiss_ocn) * lw_dn;
    float alb_ocn_dir = clamp(
        0.026f / (pow(mu, 1.7f) + 0.065f) + 0.15f * (mu - 0.1f) * (mu - 0.5f) * (mu - 1.0f),
        0.02f, 0.40f
    );
    float sw_abs_ow = (sw1_d + sw2_d) * (1.0f - alb_ocn_dir) + (sw1_f + sw2_f) * 0.94f;
    float g_down_ow0 = sw_abs_ow + lw_dn - lw_up_ow0 - shf_ow0 - lhf_ow0 - l_f0 * p_sn;
    // AOBA Formulation Eq. (9.1b): Bolton (1980, Eq. 10) open-water flux sensitivity [W m^-2 K^-1]
    float d_to = max(t_ml, 250.0f) - 29.65f;
    float dqow_dT = qsat_ow * 4302.645f / (d_to * d_to);
    float devap_dT = rho_sfc * ga_q * dqow_dT;
    float dlw_dT = 4.0f * emiss_ocn * sig * t_ml_3;
    float lambda_ow = rho_cp_ga + l_v0 * devap_dT + dlw_dT;

    float h_ice_eff = max(m_si / (rho_si * max(f_ice, 0.05f)), 0.05f);
    float h_sn_ice = swe_si / rho_snow;
    float cond_ice = 1.0f / (0.5f * h_ice_eff / k_si + h_sn_ice / k_snow);

    float shf_si0 = rho_cp_ga * (t_si - t_air);
    float subl_si0 = rho_sfc * ga_q * (qsat_si - qv_1);
    float t_si_3 = t_si * t_si * t_si;
    float lw_up_si0 = emiss_ocn * sig * (t_si_3 * t_si) + (1.0f - emiss_ocn) * lw_dn;
    float g_cond_si0 = cond_ice * (t_si - t_frz_ocn);
    float d_ti = max(t_si, 180.0f) - 7.66f;
    float dqsi_dT = qsat_si * (21.8745584f * (t_frz - 7.66f)) / (d_ti * d_ti);
    float denom_si = rho_cp_ga + l_s0 * rho_sfc * ga_q * dqsi_dT + cond_ice + 4.0f * emiss_ocn * sig * t_si_3;

    float f_sn_ice = tanh(max(swe_si, 0.0f) / 15.0f);
    float ice_vis = (1.0f - f_sn_ice) * alb_si_v + f_sn_ice * alb_sn_v;
    float ice_nir = (1.0f - f_sn_ice) * alb_si_n + f_sn_ice * alb_sn_n;
    float sw_abs_si = (sw1_d + sw1_f) * (1.0f - ice_vis) + (sw2_d + sw2_f) * (1.0f - ice_nir);

    float res_si = sw_abs_si + lw_dn - lw_up_si0 - shf_si0 - l_s0 * subl_si0 - g_cond_si0;
    float t_si_raw = clamp(t_si + clamp(res_si / denom_si, -10.0f, 10.0f), 170.0f, 310.0f);
    float si_melt_ex = max(t_si_raw - t_frz, 0.0f);
    float t_si_new = (f_ice > 1.0e-4f) ? (t_si_raw - si_melt_ex) : t_frz_ocn;
    float dt_si = t_si_new - t_si;

    float shf_si = shf_si0 + rho_cp_ga * dt_si;
    float subl_si = subl_si0 + rho_sfc * ga_q * dqsi_dT * dt_si;
    float lhf_si = l_s0 * subl_si;
    float g_oi_up = cond_ice * (t_frz_ocn - t_si_new);

    float swe_si_mid = max(swe_si + f_ice * dt * (p_sn - max(subl_si, 0.0f)), 0.0f);
    float q_top_melt = f_ice * si_melt_ex * denom_si;
    float sn_si_melt = min(q_top_melt * dt / l_f0, swe_si_mid);
    float swe_si_next = swe_si_mid - sn_si_melt;
    float q_rem_melt = max(q_top_melt * dt - l_f0 * sn_si_melt, 0.0f);

    float cw_ml = rho_sw * cp_sw * h_ml;
    float dt_ml_flux = (dt * ((1.0f - f_ice) * g_down_ow0 - f_ice * g_oi_up))
        / (cw_ml + dt * (1.0f - f_ice) * lambda_ow);
    // Neumaier compensated summation #1
    float y1 = dt_ml_flux + comp0;
    float t_ml_prov = t_ml + y1;
    float comp1 = (fabs(t_ml) >= fabs(y1)) ? ((t_ml - t_ml_prov) + y1) : ((y1 - t_ml_prov) + t_ml);

    float e_freeze = cw_ml * max(t_frz_ocn - t_ml_prov, 0.0f);
    float e_basal_melt = cw_ml * max(t_ml_prov - t_frz_ocn, 0.0f);
    float m_freeze = e_freeze / l_f0;
    float m_melt = min((e_basal_melt + q_rem_melt) / l_f0, m_si + m_freeze);
    float m_si_next = max(m_si + m_freeze - m_melt, 0.0f);
    float dt_ml_lat = (l_f0 * m_freeze - l_f0 * min(e_basal_melt / l_f0, m_si)) / cw_ml;
    // Neumaier compensated summation #2
    float y2 = dt_ml_lat + comp1;
    float t_ml_next = t_ml_prov + y2;
    float comp2 = (fabs(t_ml_prov) >= fabs(y2)) ? ((t_ml_prov - t_ml_next) + y2) : ((y2 - t_ml_next) + t_ml_prov);
    float comp_final = (e_freeze > 0.0f || m_melt > 0.0f) ? 0.0f : comp2;

    float shf_ow = shf_ow0 + rho_cp_ga * dt_ml_flux;
    float evap_ow = evap_ow0 + devap_dT * dt_ml_flux;
    float lhf_ow = l_v0 * evap_ow;
    float lw_emit_ow = max(lw_emit_ow0 + dlw_dT * dt_ml_flux, 0.0f);

    float shf_o = (1.0f - f_ice) * shf_ow + f_ice * shf_si;
    float evap_o = (1.0f - f_ice) * evap_ow + f_ice * subl_si;
    float lhf_o = (1.0f - f_ice) * lhf_ow + f_ice * lhf_si;
    float t_si_2 = t_si_new * t_si_new;
    float lw_emit_o = (1.0f - f_ice) * lw_emit_ow + f_ice * (emiss_ocn * sig * (t_si_2 * t_si_2));

    ocn_out[i]          = t_ml_next;
    ocn_out[N + i]      = comp_final;
    ocn_out[2u * N + i] = m_si_next;
    ocn_out[3u * N + i] = t_si_new;
    ocn_out[4u * N + i] = swe_si_next;
    ocn_out[5u * N + i] = shf_o;
    ocn_out[6u * N + i] = evap_o;
    ocn_out[7u * N + i] = lhf_o;
    ocn_out[8u * N + i] = lw_emit_o;
}

// =========================================================================
// 25. Fused ML Feature Extraction & Exact Conservation Closure Kernels
// =========================================================================

inline float aoba_log1p(float x) {
    float u = 1.0f + x;
    return (u == 1.0f) ? x : (log(u) * (x / (u - 1.0f)));
}

inline float aoba_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float aoba_softplus(float x) {
    return (x > 20.0f) ? x : aoba_log1p(exp(x));
}

kernel void crm_distill_extract_col(
    device const float* rho_d     [[buffer(0)]],   // [nz, N]
    device const float* theta_m   [[buffer(1)]],   // [nz, N]
    device const float* q_v       [[buffer(2)]],   // [nz, N]
    device const float* q_c       [[buffer(3)]],   // [nz, N]
    device const float* q_i       [[buffer(4)]],   // [nz, N]
    device const float* q_p       [[buffer(5)]],   // [nz, N]
    device const float* u_in      [[buffer(6)]],   // [nz, N]
    device const float* v_in      [[buffer(7)]],   // [nz, N]
    device const float* dz_c      [[buffer(8)]],   // [nz, N]
    device const float* dthm_ls   [[buffer(9)]],   // [nz, N]
    device const float* dqn_ls    [[buffer(10)]],  // [nz, N]
    device const float* q_rad     [[buffer(11)]],  // [nz, N]
    device const float* sfc_7     [[buffer(12)]],  // [7, N]: shf, lhf, u_star, tau_x, tau_y, is_ocean, f_cor
    device const float* struct_m  [[buffer(13)]],  // [N, 16]
    device float* x_out           [[buffer(14)]],  // [N, 14*nz + 23]
    device float* exner_out       [[buffer(15)]],  // [nz, N]
    device float* temp_out        [[buffer(16)]],  // [nz, N]
    constant ulong4& u_pack       [[buffer(17)]],  // [nz, N, d_in, 0]
    constant float4& f_c0         [[buffer(18)]],  // [dt, p_ref, r_d, gamma_d]
    constant float4& f_c1         [[buffer(19)]],  // [kappa_d, rv_over_rd, eps_water, e_triple]
    constant float4& f_c2         [[buffer(20)]],  // [t_freeze, t_ice_all, t_liq_all, 0]
    uint grp                      [[threadgroup_position_in_grid]],
    uint lane                     [[thread_position_in_threadgroup]])
{
    uint nz = uint(u_pack.x);
    uint N = uint(u_pack.y);
    uint d_in = uint(u_pack.z);
    uint c0 = grp * 32u;
    if (c0 >= N) return;
    uint n_valid = (c0 + 32u <= N) ? 32u : (N - c0);
    uint c = c0 + lane;
    bool active = (lane < n_valid);

    float dt = f_c0.x;
    float p_ref = f_c0.y;
    float r_d = f_c0.z;
    float gamma_d = f_c0.w;
    float kappa_d = f_c1.x;
    float rv_rd = f_c1.y;
    float eps_w = f_c1.z;
    float e_tr = f_c1.w;
    float t_frz = f_c2.x;
    float t_ice = f_c2.y;
    float t_liq = f_c2.z;
    float gk = gamma_d * kappa_d;
    float inv_pref_rd = r_d / p_ref;
    float inv_t_span = 1.0f / (t_liq - t_ice);

    // 5 x 32x33 bank-conflict-free shared-memory transpose slab (21.1 KB):
    // 100% coalesced reads from [nz, N] and 100% coalesced writes to [N, d_in].
    threadgroup float sm_t[5u * 32u * 33u];
    const uint s_str = 32u * 33u;

    for (uint k0 = 0u; k0 < nz; k0 += 32u) {
        uint klen = (k0 + 32u <= nz) ? 32u : (nz - k0);

        // Group A (5 thermo features: 0=t_li, 1=log_p, 2=th_l, 3=qn_tot, 7=rh)
        if (active) {
            for (uint dk = 0u; dk < klen; ++dk) {
                uint idx = (k0 + dk) * N + c;
                float r = rho_d[idx];
                float thm = theta_m[idx];
                float qv = q_v[idx];
                float arg = max(r * thm * inv_pref_rd, 1.0e-7f);
                float log2_arg = log2(arg);
                float pres = p_ref * exp2(gamma_d * log2_arg);
                float ex = exp2(gk * log2_arg);
                float temp = ex * thm / (1.0f + rv_rd * qv);
                exner_out[idx] = ex;
                temp_out[idx] = temp;

                float qc_p = max(q_c[idx], 0.0f);
                float qi_p = max(q_i[idx], 0.0f);
                float qn_tot = max(max(qv, 0.0f) + qc_p + qi_p, 0.0f);
                float t_li = clamp(temp - (2489.449f * qc_p + 2820.911f * qi_p), 150.0f, 400.0f);
                float th_l = t_li / max(ex, 0.05f);

                float omega = clamp((t_li - t_ice) * inv_t_span, 0.0f, 1.0f);
                float tc = max(t_li, 160.0f) - t_frz;
                float ti = max(t_li, 120.0f);
                float ew = e_tr * exp(17.67f * tc / (tc + 243.50f));
                float ei = e_tr * exp(21.8745584f * (ti - t_frz) / (ti - 7.66f));
                float qsw = eps_w * ew / max(pres - ew, 0.05f * pres);
                float qsi = eps_w * ei / max(pres - ei, 0.05f * pres);
                float qs = clamp(omega * qsw + (1.0f - omega) * qsi, 1.0e-7f, 0.2f);
                float rh = clamp(qn_tot / qs, 0.0f, 2.0f);

                uint s_off = lane * 33u + dk;
                sm_t[0u * s_str + s_off] = (t_li - 250.0f) * 0.02f;
                sm_t[1u * s_str + s_off] = (log(max(pres, 1.0f)) - 10.0f) * 0.4f;
                sm_t[2u * s_str + s_off] = (th_l - 320.0f) * 0.0125f;
                sm_t[3u * s_str + s_off] = aoba_log1p(qn_tot * 1.0e3f);
                sm_t[4u * s_str + s_off] = rh;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < klen) {
            uint k_col = k0 + lane;
            for (uint r = 0u; r < n_valid; ++r) {
                uint xb = (c0 + r) * d_in + k_col;
                uint rs = r * 33u + lane;
                x_out[xb + 0u * nz] = sm_t[0u * s_str + rs];
                x_out[xb + 1u * nz] = sm_t[1u * s_str + rs];
                x_out[xb + 2u * nz] = sm_t[2u * s_str + rs];
                x_out[xb + 3u * nz] = sm_t[3u * s_str + rs];
                x_out[xb + 4u * nz] = 0.0f;
                x_out[xb + 5u * nz] = 0.0f;
                x_out[xb + 6u * nz] = 0.0f;
                x_out[xb + 7u * nz] = sm_t[4u * s_str + rs];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Group B (5 kinematics/forcing features: 8=u, 9=v, 10=dthm_ls, 11=dqn_ls, 12=q_rad)
        if (active) {
            for (uint dk = 0u; dk < klen; ++dk) {
                uint idx = (k0 + dk) * N + c;
                uint s_off = lane * 33u + dk;
                sm_t[0u * s_str + s_off] = clamp(u_in[idx] * (1.0f / 30.0f), -4.0f, 4.0f);
                sm_t[1u * s_str + s_off] = clamp(v_in[idx] * (1.0f / 30.0f), -4.0f, 4.0f);
                sm_t[2u * s_str + s_off] = clamp(dthm_ls[idx] * dt, -10.0f, 10.0f);
                sm_t[3u * s_str + s_off] = clamp(dqn_ls[idx] * dt * 1.0e3f, -10.0f, 10.0f);
                sm_t[4u * s_str + s_off] = clamp(q_rad[idx] * dt * 2.0f, -10.0f, 10.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < klen) {
            uint k_col = k0 + lane;
            for (uint r = 0u; r < n_valid; ++r) {
                uint xb = (c0 + r) * d_in + k_col;
                uint rs = r * 33u + lane;
                x_out[xb + 8u * nz]  = sm_t[0u * s_str + rs];
                x_out[xb + 9u * nz]  = sm_t[1u * s_str + rs];
                x_out[xb + 10u * nz] = sm_t[2u * s_str + rs];
                x_out[xb + 11u * nz] = sm_t[3u * s_str + rs];
                x_out[xb + 12u * nz] = sm_t[4u * s_str + rs];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Group C (1 feature: 13=log(dz_c))
        if (active) {
            for (uint dk = 0u; dk < klen; ++dk) {
                sm_t[lane * 33u + dk] = log(max(dz_c[(k0 + dk) * N + c], 10.0f) * 2.0e-3f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < klen) {
            uint fcol = 13u * nz + k0 + lane;
            for (uint r = 0u; r < n_valid; ++r) {
                x_out[(c0 + r) * d_in + fcol] = sm_t[r * 33u + lane];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Surface scalars (7) -> transposed via sm_t for coalesced writes
    if (active) {
        uint s_off = lane * 33u;
        sm_t[s_off + 0u] = clamp(sfc_7[0u * N + c] * 5.0e-3f, -5.0f, 5.0f);
        sm_t[s_off + 1u] = clamp(sfc_7[1u * N + c] * 5.0e-3f, -5.0f, 5.0f);
        sm_t[s_off + 2u] = clamp(sfc_7[2u * N + c] * 2.0f, 0.0f, 5.0f);
        sm_t[s_off + 3u] = clamp(sfc_7[3u * N + c] * 5.0f, -5.0f, 5.0f);
        sm_t[s_off + 4u] = clamp(sfc_7[4u * N + c] * 5.0f, -5.0f, 5.0f);
        sm_t[s_off + 5u] = sfc_7[5u * N + c];
        sm_t[s_off + 6u] = clamp(fabs(sfc_7[6u * N + c]) * 1.0e4f, 0.0f, 5.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 7u) {
        uint fcol = 14u * nz + lane;
        for (uint r = 0u; r < n_valid; ++r) {
            x_out[(c0 + r) * d_in + fcol] = sm_t[r * 33u + lane];
        }
    }
    // Structural memory [N, 16]: copy coalesced across lanes 0..15 for each row r
    if (lane < 16u) {
        uint fcol = 14u * nz + 7u + lane;
        for (uint r = 0u; r < n_valid; ++r) {
            x_out[(c0 + r) * d_in + fcol] = struct_m[(c0 + r) * 16u + lane];
        }
    }
}

kernel void crm_distill_closure_col(
    device const float* raw_prof    [[buffer(0)]],   // [N, 8, nz]
    device const float* raw_scal    [[buffer(1)]],   // [N, 3]
    device const float* rho_d       [[buffer(2)]],   // [nz, N]
    device const float* dz_c        [[buffer(3)]],   // [nz, N]
    device const float* exner       [[buffer(4)]],   // [nz, N]
    device const float* thm_post_ls [[buffer(5)]],   // [nz, N]
    device const float* qn_post_ls  [[buffer(6)]],   // [nz, N]
    device const float* qp_pre      [[buffer(7)]],   // [nz, N]
    device const float* qc_pre      [[buffer(8)]],   // [nz, N]
    device const float* qi_pre      [[buffer(9)]],   // [nz, N]
    device const float* u_pre       [[buffer(10)]],  // [nz, N]
    device const float* v_pre       [[buffer(11)]],  // [nz, N]
    device const float* sfc_5       [[buffer(12)]],  // [5, N]: temp_sfc, shf, evap, tau_x, tau_y
    device float* out_prof          [[buffer(13)]],  // [9, nz, N]: thm, qn, qp, qv, qc, qi, u, v, f_cld
    device float* out_scal          [[buffer(14)]],  // [3, N]: precip_rain, precip_snow, f_prec
    constant ulong4& u_pack         [[buffer(15)]],  // [nz, N, 0, 0]
    constant float4& f_p            [[buffer(16)]],  // [dt, cp_d, l_v0, l_f0]
    constant float4& f_p1           [[buffer(17)]],  // [rv_over_rd, p_ref, t_freeze, eps_water]
    uint c                          [[thread_position_in_grid]])
{
    uint nz = uint(u_pack.x);
    uint N = uint(u_pack.y);
    if (c >= N) return;

    float dt = f_p.x;
    float cp_d = f_p.y;
    float l_v0 = f_p.z;
    float l_f0 = f_p.w;
    float rv_rd = f_p1.x;
    float p_ref = f_p1.y;
    float t_frz = f_p1.z;
    float eps_w = f_p1.w;

    float temp_sfc = sfc_5[c];
    float shf      = sfc_5[N + c];
    float evap     = sfc_5[2u * N + c];
    float tau_x    = sfc_5[3u * N + c];
    float tau_y    = sfc_5[4u * N + c];

    uint rp_base = c * 8u * nz;
    uint rs_base = c * 3u;
    uint plane = nz * N;

    // Pass 1: column integrals for convective water, PBL SHF weight, enthalpy, momentum & flux-form cache
    float dm_sum_m = 0.0f;
    float dm_pbl_sum = 0.0f;
    float w_pre_conv = 0.0f;
    float rh_pre_max = 0.0f;
    float dh_raw = 0.0f;
    float denom_h = 0.0f;
    float du_imp = 0.0f;
    float dv_imp = 0.0f;

    // Low-register column buffers (7x[64] = 1.75 KB vs 19x[128] = 9.5 KB) with buffer reuse across passes
    float qsat_col[64];
    float qnb_col[64];
    float dm_c_col[64];
    float sconv_col[64];
    float qp_p_col[64];
    float rp0_s[64];
    float rp1_s[64];

    float dm0 = max(rho_d[c] * dz_c[c], 1.0e-6f);
    float dq_evap0 = (evap * dt) / max(dm0, 1.0e-4f);
    float ex0 = max(exner[c], 0.10f);
    float inv_pbl_ex = 1.0f / (0.015f * ex0);
    uint nz_c = min(nz, 64u);

    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * N + c;
        float dm = max(rho_d[idx] * dz_c[idx], 1.0e-6f);
        float ex = exner[idx];
        float ex_safe = max(ex, 0.05f);
        float s_lin = clamp((ex - 0.50f) / 0.14f, 0.0f, 1.0f);
        float s_conv = s_lin * s_lin;
        float pbl_sh = clamp(1.0f - (ex0 - ex) * inv_pbl_ex, 0.0f, 1.0f) * s_conv;

        dm_sum_m += dm * s_conv;
        dm_pbl_sum += dm * pbl_sh;

        if (k < 64u) {
            dm_c_col[k] = dm;
            sconv_col[k] = s_conv;
        }
        if (s_conv > 0.0f) {
            float qn0 = max(qn_post_ls[idx], 0.0f);
            float qp0 = max(qp_pre[idx], 0.0f);
            float qc0 = max(qc_pre[idx], 0.0f);
            float qi0 = max(qi_pre[idx], 0.0f);
            float qci0 = qc0 + qi0;
            if (qci0 > qn0) {
                float sc = qn0 / max(qci0, 1.0e-12f);
                qc0 *= sc;
                qi0 *= sc;
            }
            float qv0 = max(qn0 - qc0 - qi0, 0.0f);
            float t_post_ls = (thm_post_ls[idx] * ex) / (1.0f + rv_rd * qv0);
            float t_env = clamp(t_post_ls, 180.0f, 340.0f);
            float ex2 = ex_safe * ex_safe;
            float p_env = p_ref * ex2 * ex_safe * sqrt(ex_safe);
            float es_w = 611.2f * exp(clamp(17.67f * (t_env - t_frz) / max(t_env - 29.65f, 20.0f), -25.0f, 12.0f));
            float es_i = 611.2f * exp(clamp(21.875f * (t_env - t_frz) / max(t_env - 7.66f, 20.0f), -25.0f, 12.0f));
            float om_liq = clamp((t_env - 253.15f) / 20.0f, 0.0f, 1.0f);
            float es_env = om_liq * es_w + (1.0f - om_liq) * es_i;
            float qsat_env = clamp(eps_w * es_env / max(p_env - es_env, 100.0f), 1.0e-7f, 0.10f);
            rh_pre_max = max(rh_pre_max, qn0 / max(qsat_env, 1.0e-7f));

            w_pre_conv += dm * (qn0 + qp0);
            float qnb = (k == 0u) ? max(qn0 + dq_evap0, 0.0f) : qn0;
            float fp_sig = aoba_sigmoid(raw_prof[rp_base + 2u * nz + k] - 2.0f);
            float qp_p = s_conv * min(qnb, 2.0e-3f) * (0.25f * (fp_sig * fp_sig));
            if (k < 64u) {
                qsat_col[k] = qsat_env;
                qnb_col[k] = qnb;
                qp_p_col[k] = qp_p;
            }

            denom_h += cp_d * dm * s_conv;
        } else if (k < 64u) {
            qsat_col[k] = 1.0e-7f;
            qnb_col[k] = 0.0f;
            qp_p_col[k] = 0.0f;
        }
    }

    float dm_sum_safe = max(dm_sum_m, 1.0e-3f);
    float w_avail_conv = max(w_pre_conv + evap * dt, 1.0e-8f);
    float col_rh_gate = clamp((rh_pre_max - 0.72f) / 0.25f, 0.0f, 1.0f);
    float pf_sig = aoba_sigmoid(raw_scal[rs_base + 0u] - 2.0f);
    float p_frac = 0.25f * col_rh_gate * (pf_sig * pf_sig);
    float p_tot_mass = (w_avail_conv > 0.05f) ? max(p_frac * w_avail_conv, 0.0f) : 0.0f;

    // Pass 1b (in-place, zero scratch array): 2-pass [0.25, 0.50, 0.25] vertical binomial smoothing of rp[0], rp[1]
    for (uint k = 0u; k < nz_c; ++k) {
        float sc_k = sconv_col[k];
        rp0_s[k] = (sc_k > 0.0f) ? (raw_prof[rp_base + 0u * nz + k] * sc_k) : 0.0f;
        rp1_s[k] = (sc_k > 0.0f) ? (raw_prof[rp_base + 1u * nz + k] * sc_k) : 0.0f;
    }
    if (nz_c >= 3u) {
        for (int pass = 0; pass < 2; ++pass) {
            float prev0 = rp0_s[0];
            rp0_s[0] = 0.75f * prev0 + 0.25f * rp0_s[1];
            float prev1 = rp1_s[0];
            rp1_s[0] = 0.75f * prev1 + 0.25f * rp1_s[1];
            for (uint k = 1u; k + 1u < nz_c; ++k) {
                float cur0 = rp0_s[k];
                rp0_s[k] = 0.25f * prev0 + 0.50f * cur0 + 0.25f * rp0_s[k + 1u];
                prev0 = cur0;
                float cur1 = rp1_s[k];
                rp1_s[k] = 0.25f * prev1 + 0.50f * cur1 + 0.25f * rp1_s[k + 1u];
                prev1 = cur1;
            }
            rp0_s[nz_c - 1u] = 0.25f * prev0 + 0.75f * rp0_s[nz_c - 1u];
            rp1_s[nz_c - 1u] = 0.25f * prev1 + 0.75f * rp1_s[nz_c - 1u];
        }
    }

    float dm_dqn_sum = 0.0f;
    float dm_wbal_sum = 0.0f;
    float w_qp0_sum = 0.0f;
    float w_qp_prov_sum = 0.0f;
    for (uint k = 0u; k < nz_c; ++k) {
        float sc_k = sconv_col[k];
        if (sc_k > 0.0f) {
            float ex_k = exner[k * N + c];
            float pbl_g = clamp((ex_k - 0.90f) / 0.05f, 0.0f, 1.0f);
            float l_gate = max(col_rh_gate, pbl_g);
            float d_temp_r = sc_k * l_gate * tanh(rp0_s[k] * 0.2f) * 0.15f;
            dh_raw += cp_d * dm_c_col[k] * d_temp_r;
            float du_r = sc_k * l_gate * tanh(raw_prof[rp_base + 5u * nz + k] * 0.2f) * 4.0f;
            float dv_r = sc_k * l_gate * tanh(raw_prof[rp_base + 6u * nz + k] * 0.2f) * 4.0f;
            du_imp += dm_c_col[k] * du_r;
            dv_imp += dm_c_col[k] * dv_r;
            float dq_lim = l_gate * min(0.015f * qnb_col[k], 1.0e-4f);
            float dqn_r = sc_k * tanh(rp1_s[k] * 0.2f) * dq_lim;
            rp1_s[k] = dqn_r;
            dm_dqn_sum += dm_c_col[k] * dqn_r;
            dm_wbal_sum += dm_c_col[k] * dq_lim;
            w_qp0_sum += dm_c_col[k] * max(qp_pre[k * N + c], 0.0f);
            qp_p_col[k] *= col_rh_gate;
            w_qp_prov_sum += dm_c_col[k] * qp_p_col[k];
        } else {
            rp1_s[k] = 0.0f;
        }
    }
    float dqn_corr = dm_dqn_sum / max(dm_wbal_sum, 1.0e-8f);
    float w_qn_tr_sum = 0.0f;
    float w_p_sum = 0.0f;
    for (uint k = 0u; k < nz_c; ++k) {
        if (sconv_col[k] > 0.0f) {
            uint idx = k * N + c;
            float qn0_k = max(qn_post_ls[idx], 0.0f);
            float qc0_k = max(qc_pre[idx], 0.0f);
            float qi0_k = max(qi_pre[idx], 0.0f);
            float qci0_k = qc0_k + qi0_k;
            if (qci0_k > qn0_k) {
                float sc = qn0_k / max(qci0_k, 1.0e-12f);
                qc0_k *= sc;
                qi0_k *= sc;
            }
            float l_gate = max(col_rh_gate, clamp((exner[idx] - 0.90f) / 0.05f, 0.0f, 1.0f));
            float wb = l_gate * min(0.015f * qnb_col[k], 1.0e-4f);
            float dqn_bal = rp1_s[k] - dqn_corr * wb;
            float qn_tr = max(qnb_col[k] + dqn_bal, 0.0f);
            float rh_tr = clamp(qn_tr / max(qsat_col[k], 1.0e-7f), 0.0f, 2.0f);
            float rh_p = clamp((rh_tr - 0.60f) / 0.35f, 0.0f, 1.0f);
            float wp_l = qn_tr * (rh_p * rh_p + 0.01f) + 2.0f * qc0_k + 0.25f * qi0_k;
            qnb_col[k] = qn_tr;
            rp1_s[k] = wp_l;
            w_qn_tr_sum += dm_c_col[k] * qn_tr;
            w_p_sum += dm_c_col[k] * wp_l;
        } else {
            qnb_col[k] = 0.0f;
            rp1_s[k] = 0.0f;
        }
    }
    float qp_cap_col = w_qp0_sum + 0.15f * w_qn_tr_sum;
    float qp_sc = (w_qp_prov_sum > qp_cap_col) ? (qp_cap_col / max(w_qp_prov_sum, 1.0e-6f)) : 1.0f;
    float w_qp_conv_sum = w_qp_prov_sum * qp_sc;
    float p_need_qn = max(p_tot_mass + (w_qp_conv_sum - w_qp0_sum), 0.0f);
    float inv_wp_sum = p_need_qn / max(w_p_sum, 1.0e-6f);
    float p_sink_mass = 0.0f;
    for (uint k = 0u; k < nz_c; ++k) {
        if (sconv_col[k] > 0.0f) {
            float qn_tr = qnb_col[k];
            float dq_psink = min(inv_wp_sum * rp1_s[k], 0.9f * qn_tr);
            qnb_col[k] = max(qn_tr - dq_psink, 0.0f);
            rp1_s[k] = dq_psink;
            p_sink_mass += dm_c_col[k] * dq_psink;
        } else {
            rp1_s[k] = 0.0f;
        }
    }

    // Pass 2: convective water, condensate partition, autoconversion fallout & precip integral
    float w_post_conv = 0.0f;
    float p_fall_mass = 0.0f;
    float d_m_qp = 0.0f;

    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * N + c;
        float dm = (k < 64u) ? dm_c_col[k] : max(rho_d[idx] * dz_c[idx], 1.0e-6f);
        float ex = exner[idx];
        float s_conv = (k < 64u) ? sconv_col[k] : 0.0f;

        float qn0 = max(qn_post_ls[idx], 0.0f);
        float qp0 = max(qp_pre[idx], 0.0f);
        float qc0 = max(qc_pre[idx], 0.0f);
        float qi0 = max(qi_pre[idx], 0.0f);
        float qci0 = qc0 + qi0;
        if (qci0 > qn0) {
            float sc = qn0 / max(qci0, 1.0e-12f);
            qc0 *= sc;
            qi0 *= sc;
        }
        float qv0 = max(qn0 - qc0 - qi0, 0.0f);

        float qn_n = qn0;
        float qp_n = qp0;
        float qc_n = qc0;
        float qi_n = qi0;
        float qv_n = qv0;

        if (s_conv > 0.0f && k < 64u) {
            float t_post_ls = (thm_post_ls[idx] * ex) / (1.0f + rv_rd * qv0);
            float t_env = clamp(t_post_ls, 180.0f, 340.0f);
            float qsat_env = qsat_col[k];

            qn_n = qnb_col[k];
            qp_n = qp_p_col[k] * qp_sc;
            w_post_conv += dm * (qn_n + qp_n);

            float rh_lin = clamp((qn_n / max(qsat_env, 1.0e-7f) - 0.55f) / 0.40f, 0.0f, 1.0f);
            float rh_gate = rh_lin * rh_lin;
            float fc_sig = aoba_sigmoid(raw_prof[rp_base + 3u * nz + k] - 2.2f);
            float f_cond = s_conv * rh_gate * 0.95f * (fc_sig * fc_sig);
            float om_liq_p = clamp((t_env - 253.15f) / 20.0f, 0.0f, 1.0f);
            float f_ice_nn = aoba_sigmoid(raw_prof[rp_base + 4u * nz + k]);
            float f_ice = (t_env <= 253.15f) ? 1.0f : ((1.0f - om_liq_p) * f_ice_nn);
            float q_cond = max(qn_n * f_cond, max(qn_n - qsat_env, 0.0f));
            float qc_pf = q_cond * (1.0f - f_ice);
            float qi_pf = q_cond * f_ice;
            qv_n = max(qn_n - qc_pf - qi_pf, 0.0f);

            float qc_fall = max(qc_pf - 5.0e-4f, 0.0f);
            float qi_cap = min(qsat_env, 1.0e-4f);
            float qi_fall = max(qi_pf - qi_cap, 0.0f);
            qc_n = qc_pf - qc_fall;
            qi_n = qi_pf - qi_fall;
            qn_n = qv_n + qc_n + qi_n;
            p_fall_mass += dm * (qc_fall + qi_fall);
            // Accumulate qf_k into rp1_s[k] (which already holds dqps_k)
            rp1_s[k] += (qc_fall + qi_fall);

            d_m_qp += dm * (qp_n - qp0);
            out_prof[8u * plane + idx] = s_conv * rh_gate * aoba_sigmoid(raw_prof[rp_base + 7u * nz + k] - 2.0f);
        } else {
            out_prof[8u * plane + idx] = 0.0f;
        }

        out_prof[1u * plane + idx] = qn_n;
        out_prof[2u * plane + idx] = qp_n;
        out_prof[3u * plane + idx] = qv_n;
        out_prof[4u * plane + idx] = qc_n;
        out_prof[5u * plane + idx] = qi_n;
    }

    float p_tot_orig = max(w_avail_conv - w_post_conv, 0.0f);
    float p_tot_fin = p_tot_orig + p_fall_mass;
    float f_snow = aoba_sigmoid(raw_scal[rs_base + 1u]) * clamp((278.0f - temp_sfc) / 6.0f, 0.0f, 1.0f);
    out_scal[c]          = (p_tot_fin * (1.0f - f_snow)) / dt;
    out_scal[N + c]      = (p_tot_fin * f_snow) / dt;
    out_scal[2u * N + c] = col_rh_gate * aoba_sigmoid(raw_scal[rs_base + 2u] - 1.0f);

    float dh_precip = l_v0 * (p_tot_orig - p_sink_mass + d_m_qp) + l_f0 * (p_tot_fin * f_snow);
    float h_corr = (dh_precip - dh_raw) / max(denom_h, 1.0f);
    float dT_shf_scale = (shf * dt) / (cp_d * max(dm_pbl_sum, 1.0e-3f));
    float u_corr = (tau_x * dt - du_imp) / dm_sum_safe;
    float v_corr = (tau_y * dt - dv_imp) / dm_sum_safe;

    // Pass 3: reuse qnb_col=t_col, qp_p_col=t_pls, qsat_col=ex_col (zero new stack arrays!)
    bool use_adj = (nz <= 64u);

    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * N + c;
        float ex = exner[idx];
        float ex_safe = max(ex, 0.05f);
        float s_conv = (k < 64u) ? sconv_col[k] : 0.0f;
        float qn0 = max(qn_post_ls[idx], 0.0f);
        float qc0 = max(qc_pre[idx], 0.0f);
        float qi0 = max(qi_pre[idx], 0.0f);
        float qci0 = qc0 + qi0;
        if (qci0 > qn0) {
            float sc = qn0 / max(qci0, 1.0e-12f);
            qc0 *= sc;
            qi0 *= sc;
        }
        float qv0 = max(qn0 - qc0 - qi0, 0.0f);
        float t_post_ls = (thm_post_ls[idx] * ex) / (1.0f + rv_rd * qv0);
        float d_temp_cons = 0.0f;
        float l_gate = max(col_rh_gate, clamp((ex - 0.90f) / 0.05f, 0.0f, 1.0f));
        if (s_conv > 0.0f) {
            float pbl_sh = clamp(1.0f - (ex0 - ex) * inv_pbl_ex, 0.0f, 1.0f) * s_conv;
            float qc_n = out_prof[4u * plane + idx];
            float qi_n = out_prof[5u * plane + idx];
            float qf_plus_dqps = (k < 64u) ? rp1_s[k] : 0.0f;
            float d_temp_lat = (l_v0 * ((qc_n + qi_n) - (qc0 + qi0) + qf_plus_dqps) + l_f0 * (qi_n - qi0)) / cp_d;
            float rp0_k = (k < 64u) ? rp0_s[k] : (raw_prof[rp_base + 0u * nz + k] * s_conv);
            float d_temp_r = s_conv * l_gate * tanh(rp0_k * 0.2f) * 0.15f;
            d_temp_cons = dT_shf_scale * pbl_sh + d_temp_lat + d_temp_r + h_corr * s_conv;
        }
        if (use_adj) {
            qp_p_col[k] = t_post_ls;
            qnb_col[k] = t_post_ls + d_temp_cons;
            qsat_col[k] = ex_safe;
        } else if (s_conv > 0.0f) {
            float qv_n = out_prof[3u * plane + idx];
            float d_thm_exact = (d_temp_cons * (1.0f + rv_rd * qv_n) + t_post_ls * rv_rd * (qv_n - qv0)) / ex_safe;
            out_prof[idx] = clamp(thm_post_ls[idx] + d_thm_exact, 150.0f, 10000.0f);
        } else {
            out_prof[idx] = clamp(thm_post_ls[idx], 150.0f, 10000.0f);
        }
        if (s_conv > 0.0f) {
            float du_r = s_conv * l_gate * tanh(raw_prof[rp_base + 5u * nz + k] * 0.2f) * 4.0f;
            float dv_r = s_conv * l_gate * tanh(raw_prof[rp_base + 6u * nz + k] * 0.2f) * 4.0f;
            out_prof[6u * plane + idx] = u_pre[idx] + du_r + u_corr * s_conv;
            out_prof[7u * plane + idx] = v_pre[idx] + dv_r + v_corr * s_conv;
        } else {
            out_prof[6u * plane + idx] = u_pre[idx];
            out_prof[7u * plane + idx] = v_pre[idx];
        }
    }

    if (use_adj) {
        for (uint start = 0u; start < 2u; ++start) {
            for (uint k = start; k + 1u < nz; k += 2u) {
                if (sconv_col[k] > 0.0f && sconv_col[k + 1u] > 0.0f) {
                    float th0 = qnb_col[k] / qsat_col[k];
                    float th1 = qnb_col[k + 1u] / qsat_col[k + 1u];
                    if (th0 > th1) {
                        float dm0_p = dm_c_col[k];
                        float dm1_p = dm_c_col[k + 1u];
                        float th_eq = (dm0_p * qnb_col[k] + dm1_p * qnb_col[k + 1u])
                                      / max(dm0_p * qsat_col[k] + dm1_p * qsat_col[k + 1u], 1.0e-6f);
                        qnb_col[k]      = th_eq * qsat_col[k];
                        qnb_col[k + 1u] = th_eq * qsat_col[k + 1u];
                    }
                }
            }
        }
        for (uint k = 0u; k < nz; ++k) {
            uint idx = k * N + c;
            if (sconv_col[k] > 0.0f) {
                float qn0 = max(qn_post_ls[idx], 0.0f);
                float qc0 = max(qc_pre[idx], 0.0f);
                float qi0 = max(qi_pre[idx], 0.0f);
                float qci0 = qc0 + qi0;
                if (qci0 > qn0) {
                    float sc = qn0 / max(qci0, 1.0e-12f);
                    qc0 *= sc;
                    qi0 *= sc;
                }
                float qv0 = max(qn0 - qc0 - qi0, 0.0f);
                float qv_n = out_prof[3u * plane + idx];
                float t_p = qp_p_col[k];
                float d_temp_adj = qnb_col[k] - t_p;
                float d_thm_exact = (d_temp_adj * (1.0f + rv_rd * qv_n) + t_p * rv_rd * (qv_n - qv0)) / qsat_col[k];
                out_prof[idx] = clamp(thm_post_ls[idx] + d_thm_exact, 150.0f, 10000.0f);
            } else {
                out_prof[idx] = clamp(thm_post_ls[idx], 150.0f, 10000.0f);
            }
        }
    }
}

kernel void rad_distill_extract_col(
    device const float* pres_c     [[buffer(0)]],   // [nz, M]
    device const float* temp_c     [[buffer(1)]],   // [nz, M]
    device const float* dz_c       [[buffer(2)]],   // [nz, M]
    device const float* rho_d_c    [[buffer(3)]],   // [nz, M]
    device const float* qv_c       [[buffer(4)]],   // [nz, M]
    device const float* qc_c       [[buffer(5)]],   // [nz, M]
    device const float* qi_c       [[buffer(6)]],   // [nz, M]
    device const float* o3_c       [[buffer(7)]],   // [nz, M] or [1]
    device const float* q_dust_c   [[buffer(8)]],   // [nz, M] or [1]
    device const float* tau_aero_c [[buffer(9)]],   // [nz, M] or [1]
    device const float* sfc_8      [[buffer(10)]],  // [8, M]: t_sfc, alb_vd, alb_vf, alb_nd, alb_nf, emiss, mu0, s_toa
    device float* x_out            [[buffer(11)]],  // [M, 9*nz + 13]
    device float* cp_layer_out     [[buffer(12)]],  // [nz, M]
    device float* s_inc_out        [[buffer(13)]],  // [M]
    constant ulong4& u_pack        [[buffer(14)]],  // [nz, M, d_in, (has_aero<<2)|(has_dust<<1)|has_o3]
    constant float4& f_p0          [[buffer(15)]],  // [cp_d, cp_v, co2_f, ch4_f]
    constant float4& f_p1          [[buffer(16)]],  // [n2o_f, 0, 0, 0]
    uint grp                       [[threadgroup_position_in_grid]],
    uint lane                      [[thread_position_in_threadgroup]])
{
    uint nz = uint(u_pack.x);
    uint M = uint(u_pack.y);
    uint d_in = uint(u_pack.z);
    uint flags = uint(u_pack.w);
    bool has_o3   = (flags & 1u) != 0u;
    bool has_dust = (flags & 2u) != 0u;
    bool has_aero = (flags & 4u) != 0u;
    uint c0 = grp * 32u;
    if (c0 >= M) return;
    uint n_valid = (c0 + 32u <= M) ? 32u : (M - c0);
    uint c = c0 + lane;
    bool active = (lane < n_valid);

    // 5 x 32x33 bank-conflict-free shared-memory transpose slab (21.1 KB):
    // reads qv/qc/qi ONCE (no duplicate DRAM pass) and cuts 36 barriers -> 8 barriers.
    threadgroup float sm_t[5u * 32u * 33u];
    const uint s_str = 32u * 33u;

    float cp_d = f_p0.x;
    float cp_v = f_p0.y;
    float cwp_tot = 0.0f;

    for (uint k0 = 0u; k0 < nz; k0 += 32u) {
        uint klen = (k0 + 32u <= nz) ? 32u : (nz - k0);

        // Pass A: features 0..4 (T, log p, qv, qc, qi) + cp_layer_out + cwp_tot in ONE single read pass
        if (active) {
            for (uint dk = 0u; dk < klen; ++dk) {
                uint idx = (k0 + dk) * M + c;
                float qv = max(qv_c[idx], 0.0f);
                float qc = max(qc_c[idx], 0.0f);
                float qi = max(qi_c[idx], 0.0f);
                float dm = max(rho_d_c[idx] * dz_c[idx], 1.0e-6f);
                cp_layer_out[idx] = (cp_d + cp_v * qv) * dm;
                cwp_tot += (qc + qi) * dm;

                uint s_off = lane * 33u + dk;
                sm_t[0u * s_str + s_off] = (temp_c[idx] - 250.0f) * 0.02f;
                sm_t[1u * s_str + s_off] = (log(max(pres_c[idx], 1.0f)) - 10.0f) * 0.4f;
                sm_t[2u * s_str + s_off] = aoba_log1p(qv * 1.0e3f);
                sm_t[3u * s_str + s_off] = aoba_log1p(qc * 1.0e4f);
                sm_t[4u * s_str + s_off] = aoba_log1p(qi * 1.0e4f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < klen) {
            uint k_col = k0 + lane;
            for (uint r = 0u; r < n_valid; ++r) {
                uint xb = (c0 + r) * d_in + k_col;
                uint rs = r * 33u + lane;
                x_out[xb + 0u * nz] = sm_t[0u * s_str + rs];
                x_out[xb + 1u * nz] = sm_t[1u * s_str + rs];
                x_out[xb + 2u * nz] = sm_t[2u * s_str + rs];
                x_out[xb + 3u * nz] = sm_t[3u * s_str + rs];
                x_out[xb + 4u * nz] = sm_t[4u * s_str + rs];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pass B: features 5..8 (dm, o3, dust, aerosol)
        if (active) {
            for (uint dk = 0u; dk < klen; ++dk) {
                uint idx = (k0 + dk) * M + c;
                uint s_off = lane * 33u + dk;
                float dm = max(rho_d_c[idx] * dz_c[idx], 1.0e-6f);
                sm_t[0u * s_str + s_off] = aoba_log1p(dm * 1.0e-3f);
                if (has_o3)   sm_t[1u * s_str + s_off] = aoba_log1p(max(o3_c[idx], 0.0f) * 1.0e6f);
                if (has_dust) sm_t[2u * s_str + s_off] = aoba_log1p(max(q_dust_c[idx], 0.0f) * 1.0e6f);
                if (has_aero) sm_t[3u * s_str + s_off] = aoba_log1p(max(tau_aero_c[idx], 0.0f) * 10.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < klen) {
            uint k_col = k0 + lane;
            for (uint r = 0u; r < n_valid; ++r) {
                uint xb = (c0 + r) * d_in + k_col;
                uint rs = r * 33u + lane;
                x_out[xb + 5u * nz] = sm_t[0u * s_str + rs];
                x_out[xb + 6u * nz] = has_o3   ? sm_t[1u * s_str + rs] : 0.0f;
                x_out[xb + 7u * nz] = has_dust ? sm_t[2u * s_str + rs] : 0.0f;
                x_out[xb + 8u * nz] = has_aero ? sm_t[3u * s_str + rs] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 13 trailing surface & greenhouse-gas scalars (transposed via sm_t)
    if (active) {
        float t_sfc = sfc_8[0u * M + c];
        float mu0   = sfc_8[6u * M + c];
        float s_toa = sfc_8[7u * M + c];
        float mu_c  = clamp(mu0, 0.0f, 1.0f);
        float s_inc = mu_c * max(s_toa, 0.0f);
        s_inc_out[c] = s_inc;
        uint row_s = lane * 33u;
        sm_t[row_s + 0u]  = mu_c;
        sm_t[row_s + 1u]  = clamp(s_inc * (1.0f / 1361.0f), 0.0f, 4.0f);
        sm_t[row_s + 2u]  = (mu0 > 1.0e-4f) ? 1.0f : 0.0f;
        sm_t[row_s + 3u]  = clamp(sfc_8[1u * M + c], 0.0f, 1.0f);
        sm_t[row_s + 4u]  = clamp(sfc_8[2u * M + c], 0.0f, 1.0f);
        sm_t[row_s + 5u]  = clamp(sfc_8[3u * M + c], 0.0f, 1.0f);
        sm_t[row_s + 6u]  = clamp(sfc_8[4u * M + c], 0.0f, 1.0f);
        sm_t[row_s + 7u]  = clamp(sfc_8[5u * M + c], 0.0f, 1.0f);
        sm_t[row_s + 8u]  = clamp((t_sfc - 280.0f) * 0.03f, -6.0f, 6.0f);
        sm_t[row_s + 9u]  = f_p0.z;
        sm_t[row_s + 10u] = f_p0.w;
        sm_t[row_s + 11u] = f_p1.x;
        sm_t[row_s + 12u] = aoba_log1p(max(cwp_tot, 0.0f) * 10.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 13u) {
        uint fcol = 9u * nz + lane;
        for (uint r = 0u; r < n_valid; ++r) x_out[(c0 + r) * d_in + fcol] = sm_t[r * 33u + lane];
    }
}

kernel void rad_distill_closure_col(
    device const float* raw_qrad  [[buffer(0)]],   // [M, nz]
    device const float* raw_scal  [[buffer(1)]],   // [M, 7]
    device const float* cp_layer  [[buffer(2)]],   // [nz, M]
    device const float* ctx_7     [[buffer(3)]],   // [7, M]: s_inc, alb_vd, alb_vf, alb_nd, alb_nf, emiss, t_sfc
    device float* q_rad_out       [[buffer(4)]],   // [nz, M]
    device float* scal_out        [[buffer(5)]],   // [7, M]: sw1_dir, sw1_dif, sw2_dir, sw2_dif, lw_dn, olr, asr
    constant ulong4& u_pack       [[buffer(6)]],   // [nz, M, 0, 0]
    constant float4& f_p          [[buffer(7)]],   // [sig, 120.0 / sec_per_day, 0, 0]
    uint c                        [[thread_position_in_grid]])
{
    uint nz = uint(u_pack.x);
    uint M = uint(u_pack.y);
    if (c >= M) return;

    float sig = f_p.x;
    float qrad_scale = f_p.y;
    uint rs = c * 7u;
    uint rq = c * nz;

    float s_inc   = ctx_7[0u * M + c];
    float alb_vd  = ctx_7[1u * M + c];
    float alb_vf  = ctx_7[2u * M + c];
    float alb_nd  = ctx_7[3u * M + c];
    float alb_nf  = ctx_7[4u * M + c];
    float emiss   = ctx_7[5u * M + c];
    float t_sfc   = ctx_7[6u * M + c];

    float s1d_f = aoba_sigmoid(raw_scal[rs + 0u] - 1.5f);
    float s1f_f = aoba_sigmoid(raw_scal[rs + 1u] - 2.0f);
    float s2d_f = aoba_sigmoid(raw_scal[rs + 2u] - 1.5f);
    float s2f_f = aoba_sigmoid(raw_scal[rs + 3u] - 2.0f);
    float s_norm = 1.0f / max(s1d_f + s1f_f + s2d_f + s2f_f, 1.0f);

    float sw1_dir = s_inc * (s1d_f * s_norm);
    float sw1_dif = s_inc * (s1f_f * s_norm);
    float sw2_dir = s_inc * (s2d_f * s_norm);
    float sw2_dif = s_inc * (s2f_f * s_norm);

    float f_sw_sfc_net =
        (1.0f - alb_vd) * sw1_dir + (1.0f - alb_vf) * sw1_dif
        + (1.0f - alb_nd) * sw2_dir + (1.0f - alb_nf) * sw2_dif;
    float f_atm_sw = aoba_sigmoid(raw_scal[rs + 4u] - 1.5f);
    float asr = f_sw_sfc_net + f_atm_sw * max(s_inc - f_sw_sfc_net, 0.0f);

    float ts = max(t_sfc, 120.0f);
    float ts2 = ts * ts;
    float eb_sfc = sig * (ts2 * ts2);
    float lw_dn = clamp(aoba_softplus(raw_scal[rs + 5u] + 0.8f) * 0.65f * eb_sfc, 5.0f, 850.0f);
    float olr   = clamp(aoba_softplus(raw_scal[rs + 6u] + 0.5f) * 0.65f * eb_sfc, 15.0f, 750.0f);
    float f_lw_sfc_net = emiss * (lw_dn - eb_sfc);

    scal_out[0u * M + c] = sw1_dir;
    scal_out[1u * M + c] = sw1_dif;
    scal_out[2u * M + c] = sw2_dir;
    scal_out[3u * M + c] = sw2_dif;
    scal_out[4u * M + c] = lw_dn;
    scal_out[5u * M + c] = olr;
    scal_out[6u * M + c] = asr;

    float f_atm_raw = 0.0f;
    float cp_tot = 0.0f;
    for (uint k = 0u; k < nz; ++k) {
        uint idx = k * M + c;
        float cp_k = cp_layer[idx];
        float qr_k = tanh(raw_qrad[rq + k] * 0.05f) * qrad_scale;
        f_atm_raw += cp_k * qr_k;
        cp_tot += cp_k;
        q_rad_out[idx] = qr_k;
    }
    float f_atm_exact = (asr - olr) - (f_sw_sfc_net + f_lw_sfc_net);
    float corr = (f_atm_exact - f_atm_raw) / max(cp_tot, 1.0f);
    for (uint k = 0u; k < nz; ++k) {
        q_rad_out[k * M + c] += corr;
    }
}
"""


def run_scvt_reconstruct_winds_metal(
    lib: Any,
    mc: dict[str, Any],
    u_edge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    is_1d = u_edge.ndim == 1
    u_2d = u_edge.unsqueeze(0) if is_1d else u_edge
    nz = u_2d.shape[0]
    nc = mc["n_cells"]
    ne = mc["n_edges"]
    u_zonal = torch.empty((nz, nc), dtype=torch.float32, device=u_edge.device)
    v_merid = torch.empty((nz, nc), dtype=torch.float32, device=u_edge.device)
    lib.scvt_reconstruct_winds(
        u_2d.contiguous(),
        mc["eoc"],
        mc["rec_ew"],
        mc["rec_nw"],
        u_zonal,
        v_merid,
        [int(nz), int(nc), int(ne), 0],
        threads=nz * nc,
    )
    if is_1d:
        return u_zonal[0], v_merid[0]
    return u_zonal, v_merid


def run_scvt_project_winds_metal(
    lib: Any,
    mc: dict[str, Any],
    vec_east_cell: torch.Tensor,
    vec_north_cell: torch.Tensor,
) -> torch.Tensor:
    is_1d = vec_east_cell.ndim == 1
    ve_2d = vec_east_cell.unsqueeze(0) if is_1d else vec_east_cell
    vn_2d = vec_north_cell.unsqueeze(0) if is_1d else vec_north_cell
    nz = ve_2d.shape[0]
    nc = mc["n_cells"]
    ne = mc["n_edges"]
    out = torch.empty((nz, ne), dtype=torch.float32, device=vec_east_cell.device)
    lib.scvt_project_winds(
        ve_2d.contiguous(),
        vn_2d.contiguous(),
        mc["c0"],
        mc["c1"],
        mc["east_ne"],
        mc["north_ne"],
        out,
        [int(nz), int(ne), int(nc), 0],
        threads=nz * ne,
    )
    return out[0] if is_1d else out


_SFC_CONST_CACHE: dict[tuple[float, ...], torch.Tensor] = {}
_GAS_CONST_CACHE: dict[tuple[float, ...], torch.Tensor] = {}


def run_surface_albedos_metal(
    lib: Any,
    surf_type_i32: torch.Tensor,
    swe_land: torch.Tensor,
    swe_max: torch.Tensor,
    sigma_topo: torch.Tensor,
    snow_age: torch.Tensor,
    w_liq_top: torch.Tensor,
    h_canopy: torch.Tensor,
    lai: torch.Tensor,
    m_seaice: torch.Tensor,
    swe_seaice: torch.Tensor,
    mu0: torch.Tensor,
    consts: tuple[float, ...],
) -> torch.Tensor:
    shape = mu0.shape
    M = mu0.numel()
    dev = mu0.device
    c_buf = _SFC_CONST_CACHE.get(consts)
    if c_buf is None or c_buf.device != dev:
        c_buf = torch.tensor(consts, dtype=torch.float32, device=dev)
        _SFC_CONST_CACHE[consts] = c_buf
    out_8 = torch.empty((8, M), dtype=torch.float32, device=dev)
    lib.surface_albedos_col(
        surf_type_i32.view(M).contiguous(),
        swe_land.view(M).contiguous(),
        swe_max.view(M).contiguous(),
        sigma_topo.view(M).contiguous(),
        snow_age.view(M).contiguous(),
        w_liq_top.view(M).contiguous(),
        h_canopy.view(M).contiguous(),
        lai.view(M).contiguous(),
        m_seaice.view(M).contiguous(),
        swe_seaice.view(M).contiguous(),
        mu0.view(M).contiguous(),
        out_8,
        c_buf,
        int(M),
        threads=M,
    )
    return out_8.view((8,) + shape)


def run_surface_transfer_metal(
    lib: Any,
    u_1: torch.Tensor,
    v_1: torch.Tensor,
    theta_v1: torch.Tensor,
    theta_vsfc: torch.Tensor,
    z_ref: torch.Tensor,
    z0m: torch.Tensor,
    z0h: torch.Tensor,
    z0q: torch.Tensor,
    rsl_depth: torch.Tensor | None,
    n_iter: int,
    consts: tuple[float, ...],
) -> torch.Tensor:
    shape = u_1.shape
    M = u_1.numel()
    dev = u_1.device
    out_6 = torch.empty((6, M), dtype=torch.float32, device=dev)
    has_rsl = 1 if rsl_depth is not None else 0
    rsl_buf = rsl_depth.view(M).contiguous() if rsl_depth is not None else u_1.view(M)
    lib.surface_transfer_col(
        u_1.view(M).contiguous(),
        v_1.view(M).contiguous(),
        theta_v1.view(M).contiguous(),
        theta_vsfc.view(M).contiguous(),
        z_ref.view(M).contiguous(),
        z0m.view(M).contiguous(),
        z0h.view(M).contiguous(),
        z0q.view(M).contiguous(),
        rsl_buf,
        out_6,
        [int(M), int(n_iter), has_rsl, 0],
        [float(consts[0]), float(consts[1]), float(consts[2]), float(consts[3])],
        [float(consts[4]), float(consts[5]), 0.0, 0.0],
        threads=M,
    )
    return out_6.view((6,) + shape)


def run_two_leaf_gas_exchange_metal(
    lib: Any,
    i_sun: torch.Tensor,
    i_sh: torch.Tensor,
    l_sun: torch.Tensor,
    l_sh: torch.Tensor,
    sun_frac: torch.Tensor,
    vc25_c3: torch.Tensor,
    jm25_c3: torch.Tensor,
    r25_c3: torch.Tensor,
    vc25_c4: torch.Tensor,
    jm25_c4: torch.Tensor,
    vp25_c4: torch.Tensor,
    r25_c4: torch.Tensor,
    tg_mid: torch.Tensor,
    t_home: torch.Tensor,
    f_c4: torch.Tensor,
    t_sun: torch.Tensor,
    t_sh: torch.Tensor,
    vpd_sun: torch.Tensor,
    vpd_sh: torch.Tensor,
    pres_sfc: torch.Tensor,
    lai_exp: torch.Tensor,
    ws_beta: torch.Tensor,
    gbw_sun: torch.Tensor | None,
    gbw_sh: torch.Tensor | None,
    co2_ppmv: float,
    consts: tuple[float, ...],
) -> torch.Tensor:
    shape = lai_exp.shape
    M = lai_exp.numel()
    dev = lai_exp.device
    c_buf = _GAS_CONST_CACHE.get(consts)
    if c_buf is None or c_buf.device != dev:
        c_buf = torch.tensor(consts, dtype=torch.float32, device=dev)
        _GAS_CONST_CACHE[consts] = c_buf
    dummy = lai_exp.view(M)
    flags = 0
    if gbw_sun is not None:
        flags |= 1
        gbw_sun_b = gbw_sun.view(M).contiguous()
    else:
        gbw_sun_b = dummy
    if gbw_sh is not None:
        flags |= 2
        gbw_sh_b = gbw_sh.view(M).contiguous()
    else:
        gbw_sh_b = dummy

    out_5 = torch.empty((5, M), dtype=torch.float32, device=dev)
    lib.two_leaf_gas_exchange_col(
        i_sun.view(M).contiguous(),
        i_sh.view(M).contiguous(),
        l_sun.view(M).contiguous(),
        l_sh.view(M).contiguous(),
        sun_frac.view(M).contiguous(),
        vc25_c3.view(M).contiguous(),
        jm25_c3.view(M).contiguous(),
        r25_c3.view(M).contiguous(),
        vc25_c4.view(M).contiguous(),
        jm25_c4.view(M).contiguous(),
        vp25_c4.view(M).contiguous(),
        r25_c4.view(M).contiguous(),
        tg_mid.view(M).contiguous(),
        t_home.view(M).contiguous(),
        f_c4.view(M).contiguous(),
        t_sun.view(M).contiguous(),
        t_sh.view(M).contiguous(),
        vpd_sun.view(M).contiguous(),
        vpd_sh.view(M).contiguous(),
        pres_sfc.view(M).contiguous(),
        lai_exp.view(M).contiguous(),
        ws_beta.view(M).contiguous(),
        gbw_sun_b,
        gbw_sh_b,
        out_5,
        c_buf,
        [int(M), int(flags), 0, 0],
        [float(co2_ppmv), 0.0, 0.0, 0.0],
        threads=M,
    )
    return out_5.view((5,) + shape)


_RAD_CONST_CACHE: dict[tuple[float, ...], torch.Tensor] = {}


def run_rad_four_band_metal(
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
    emiss_sfc: float | torch.Tensor,
    re_liq_um: float,
    re_ice_um: float,
    cloud_sw_inhomog: float,
    albedo_vis_dif: torch.Tensor | None,
    albedo_nir_dif: torch.Tensor | None,
    q_dust_c: torch.Tensor | None,
    o3_vmr_c: torch.Tensor | None,
    o3_lw_coupling: bool,
    consts: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nz = pres_c.shape[0]
    col_shape = t_sfc.shape
    M = t_sfc.numel()
    dev = pres_c.device

    rc_buf = _RAD_CONST_CACHE.get(consts)
    if rc_buf is None or rc_buf.device != dev:
        rc_buf = torch.tensor(consts, dtype=torch.float32, device=dev)
        _RAD_CONST_CACHE[consts] = rc_buf

    dummy = t_sfc.view(-1)
    dz_2d = dz_c.view(nz, -1).contiguous()

    flags = 0
    if q_dust_c is not None:
        flags |= 1
        dust_buf = q_dust_c.view(nz, M).contiguous()
    else:
        dust_buf = dummy
    if o3_vmr_c is not None:
        flags |= 2
        o3_buf = o3_vmr_c.view(nz, M).contiguous()
    else:
        o3_buf = dummy
    if o3_lw_coupling:
        flags |= 4
    if isinstance(emiss_sfc, torch.Tensor):
        flags |= 8
        emiss_buf = emiss_sfc.view(M).contiguous()
        emiss_scalar = 0.0
    else:
        emiss_buf = dummy
        emiss_scalar = float(emiss_sfc)

    vis_dif = albedo_vis_dif if albedo_vis_dif is not None else albedo_vis
    nir_dif = albedo_nir_dif if albedo_nir_dif is not None else albedo_nir

    sw_up = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    sw_dn = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    lw_up = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    lw_dn = torch.empty((nz + 1, M), dtype=torch.float32, device=dev)
    q_rad_3d = torch.empty((3, nz, M), dtype=torch.float32, device=dev)
    sfc_out = torch.empty((5, M), dtype=torch.float32, device=dev)

    lib.rad_four_band_col(
        pres_c.view(nz, M).contiguous(),
        pres_w.view(nz + 1, M).contiguous(),
        temp_c.view(nz, M).contiguous(),
        temp_w.view(nz + 1, M).contiguous(),
        dz_2d,
        rho_d_c.view(nz, M).contiguous(),
        qv_c.view(nz, M).contiguous(),
        qc_c.view(nz, M).contiguous(),
        qi_c.view(nz, M).contiguous(),
        dust_buf,
        o3_buf,
        t_sfc.view(M).contiguous(),
        albedo_vis.view(M).contiguous(),
        albedo_nir.view(M).contiguous(),
        vis_dif.view(M).contiguous(),
        nir_dif.view(M).contiguous(),
        mu0.view(M).contiguous(),
        s_toa.view(M).contiguous(),
        emiss_buf,
        sw_up,
        sw_dn,
        lw_up,
        lw_dn,
        q_rad_3d,
        sfc_out,
        rc_buf,
        [int(nz), int(M), int(dz_2d.shape[1]), int(flags)],
        [float(co2_ppmv), float(re_liq_um), float(re_ice_um), float(cloud_sw_inhomog)],
        [emiss_scalar, 0.0, 0.0, 0.0],
        threads=M,
    )
    return (
        sw_up.view((nz + 1,) + col_shape),
        sw_dn.view((nz + 1,) + col_shape),
        lw_up.view((nz + 1,) + col_shape),
        lw_dn.view((nz + 1,) + col_shape),
        q_rad_3d.view((3, nz) + col_shape),
        sfc_out.view((5,) + col_shape),
    )


def run_rad_ckd_metal(
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
    ckd_tables: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nz = pres_c.shape[0]
    col_shape = t_sfc.shape
    M = t_sfc.numel()
    dev = pres_c.device

    tbl_buf = _RAD_CONST_CACHE.get(ckd_tables)
    if tbl_buf is None or tbl_buf.device != dev:
        tbl_buf = torch.tensor(ckd_tables, dtype=torch.float32, device=dev)
        _RAD_CONST_CACHE[ckd_tables] = tbl_buf

    dummy = t_sfc.view(-1)
    dz_2d = dz_c.view(nz, -1).contiguous()

    flags = 0
    if q_dust_c is not None:
        flags |= 1
        dust_buf = q_dust_c.view(nz, M).contiguous()
    else:
        dust_buf = dummy
    if o3_vmr_c is not None:
        flags |= 2
        o3_buf = o3_vmr_c.view(nz, M).contiguous()
    else:
        o3_buf = dummy
    if o3_lw_coupling:
        flags |= 4
    if isinstance(re_liq_um, torch.Tensor):
        flags |= 8
        re_l_buf = re_liq_um.view(nz, M).contiguous()
        re_l_scalar = 10.0
    else:
        re_l_buf = dummy
        re_l_scalar = float(re_liq_um)
    if isinstance(re_ice_um, torch.Tensor):
        flags |= 16
        re_i_buf = re_ice_um.view(nz, M).contiguous()
        re_i_scalar = 48.0
    else:
        re_i_buf = dummy
        re_i_scalar = float(re_ice_um)

    vis_dif = albedo_vis_dif if albedo_vis_dif is not None else albedo_vis
    nir_dif = albedo_nir_dif if albedo_nir_dif is not None else albedo_nir
    emiss_vec = (
        emiss_sfc.view(M).contiguous()
        if isinstance(emiss_sfc, torch.Tensor)
        else torch.full((M,), float(emiss_sfc), dtype=torch.float32, device=dev)
    )
    sfc_in = torch.stack(
        [
            t_sfc.view(M),
            albedo_vis.view(M),
            albedo_nir.view(M),
            vis_dif.view(M),
            nir_dif.view(M),
            mu0.view(M),
            s_toa.view(M),
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

    lib.rad_ckd_col(
        pres_c.view(nz, M).contiguous(),
        pres_w.view(nz + 1, M).contiguous(),
        temp_c.view(nz, M).contiguous(),
        temp_w.view(nz + 1, M).contiguous(),
        dz_2d,
        rho_d_c.view(nz, M).contiguous(),
        qv_c.view(nz, M).contiguous(),
        qc_c.view(nz, M).contiguous(),
        qi_c.view(nz, M).contiguous(),
        dust_buf,
        o3_buf,
        re_l_buf,
        re_i_buf,
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
        [re_l_scalar, re_i_scalar, 0.0, 0.0],
        threads=M,
    )
    return (
        sw_up.view((nz + 1,) + col_shape),
        sw_dn.view((nz + 1,) + col_shape),
        lw_up.view((nz + 1,) + col_shape),
        lw_dn.view((nz + 1,) + col_shape),
        q_rad_3d.view((3, nz) + col_shape),
        sfc_out.view((5,) + col_shape),
    )


def run_scvt_fct_advect_metal(
    lib: Any,
    mesh: Any,
    mc: dict[str, Any],
    q_batch: torch.Tensor,
    mass_old: torch.Tensor,
    mass_new: torch.Tensor,
    uh_edge: torch.Tensor,
    omega_z: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Execute batched SCVT Voronoi Zalesak FCT over `[B, nz, N_cells]` in 4 fused Metal kernels."""
    B, nz, nc = q_batch.shape
    ne = mc["n_edges"]
    dev = q_batch.device

    nb_idx = mc.get("nb_idx")
    if nb_idx is None:
        e_cell = mesh.edges_on_cell
        nb_c0 = mesh.cells_on_edge[e_cell, 0]
        nb_c1 = mesh.cells_on_edge[e_cell, 1]
        cell_id = torch.arange(nc, device=dev).unsqueeze(1)
        raw_nb = torch.where(nb_c0 == cell_id, nb_c1, nb_c0)
        valid_nb = mesh.edge_sign_on_cell != 0.0
        nb_idx = torch.where(valid_nb, raw_nb, cell_id.expand_as(raw_nb)).to(torch.int32).contiguous()
        mc["nb_idx"] = nb_idx

    key = (dev.type, dev.index or 0, B, nz, nc, ne)
    ws = _SCVT_FCT_WS_CACHE.get(key)
    if ws is None:
        ws = {
            "fl_h": torch.empty((B, nz, ne), device=dev, dtype=torch.float32),
            "ah": torch.empty((B, nz, ne), device=dev, dtype=torch.float32),
            "az": torch.empty((B, nz + 1, nc), device=dev, dtype=torch.float32),
            "q_td": torch.empty((B, nz, nc), device=dev, dtype=torch.float32),
            "r_plus": torch.empty((B, nz, nc), device=dev, dtype=torch.float32),
            "r_minus": torch.empty((B, nz, nc), device=dev, dtype=torch.float32),
        }
        _cache_put(_SCVT_FCT_WS_CACHE, key, ws)

    q_c = q_batch.contiguous()
    mo_c = mass_old.contiguous()
    mn_c = mass_new.contiguous()
    uh_c = uh_edge.contiguous()
    om_c = omega_z.contiguous()
    q_out = torch.empty_like(q_c)
    u_dim = [B, nz, nc, ne]
    dt_f = float(dt)

    lib.scvt_fct_edge(
        mc["c0"], mc["c1"], q_c, uh_c, ws["fl_h"], ws["ah"], u_dim,
        threads=B * nz * ne,
    )
    lib.scvt_fct_cell_td(
        mc["eoc"], mc["div_w"], q_c, mo_c, mn_c, ws["fl_h"], om_c,
        ws["q_td"], ws["az"], u_dim, dt_f,
        threads=B * nc,
    )
    lib.scvt_fct_bounds_ratios(
        mc["eoc"], nb_idx, mc["div_w"], q_c, ws["q_td"], mn_c, ws["ah"], ws["az"],
        ws["r_plus"], ws["r_minus"], u_dim, dt_f,
        threads=B * nz * nc,
    )
    lib.scvt_fct_limit_apply(
        mc["c0"], mc["c1"], mc["eoc"], mc["div_w"], ws["q_td"], mn_c,
        ws["ah"], ws["az"], ws["r_plus"], ws["r_minus"], q_out, u_dim, dt_f,
        threads=B * nz * nc,
    )
    return q_out


def run_dust_column_metal(
    lib: Any,
    q_dust: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    z_c: torch.Tensor,
    f_dust_emit: torch.Tensor,
    precip_rate: torch.Tensor,
    u_star_cell: torch.Tensor,
    dt: float,
    v_dep: float,
    v_settle: float,
    alpha_scav: float,
) -> torch.Tensor:
    nz, nc = q_dust.shape
    dz_2d = dz_c.view(nz, -1).contiguous()
    zc_2d = z_c.view(nz, -1).contiguous()
    q_out = torch.empty_like(q_dust)
    lib.dust_column_col(
        q_dust.contiguous(), rho_d.contiguous(), dz_2d, zc_2d,
        f_dust_emit.contiguous(), precip_rate.contiguous(), u_star_cell.contiguous(),
        q_out, [nz, nc, dz_2d.shape[1], zc_2d.shape[1]], [float(dt), float(v_dep), float(v_settle), float(alpha_scav)],
        threads=nc,
    )
    return q_out


def run_oro_gwd_metal(
    lib: Any,
    rho_d: torch.Tensor,
    u_cell: torch.Tensor,
    v_cell: torch.Tensor,
    theta_m: torch.Tensor,
    z_rel_c: torch.Tensor,
    dz_c: torch.Tensor,
    sigma_topo: torch.Tensor,
    slope_topo: torch.Tensor,
    aniso_topo: torch.Tensor,
    angle_topo: torch.Tensor,
    dt: float,
    gravity: float,
    g_wave: float,
    cd_block: float,
    hn_crit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    nz, nc = rho_d.shape
    dz_2d = dz_c.view(nz, -1).contiguous()
    zc_2d = z_rel_c.view(nz, -1).contiguous()
    du_out = torch.empty_like(u_cell)
    dv_out = torch.empty_like(v_cell)
    lib.oro_gwd_col(
        rho_d.contiguous(), u_cell.contiguous(), v_cell.contiguous(),
        theta_m.contiguous(), zc_2d, dz_2d,
        sigma_topo.contiguous(), slope_topo.contiguous(),
        aniso_topo.contiguous(), angle_topo.contiguous(),
        du_out, dv_out, [nz, nc, dz_2d.shape[1], zc_2d.shape[1]],
        [float(dt), float(gravity), float(g_wave), float(cd_block)],
        float(hn_crit),
        threads=nc,
    )
    return du_out, dv_out


def run_ngwd_propagate_metal(
    lib: Any,
    u_proj_b: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    n_int: torch.Tensor,
    tau_launch_b: torch.Tensor,
    u_src_b: torch.Tensor,
    k_launch_b: torch.Tensor,
    effgw_b: torch.Tensor,
    c_bins: torch.Tensor,
    k_h: float,
    dt: float,
    tend_max: float,
) -> torch.Tensor:
    n_axes, nz, nc = u_proj_b.shape
    n_bins = c_bins.numel()
    du_out = torch.empty_like(u_proj_b)
    lib.ngwd_propagate_col(
        u_proj_b.contiguous(), rho_d.contiguous(), dz_c.contiguous(), n_int.contiguous(),
        tau_launch_b.contiguous(), u_src_b.contiguous(),
        k_launch_b.to(torch.int32).contiguous(),
        effgw_b.contiguous(), c_bins.contiguous(),
        du_out, [n_axes, nz, nc, n_bins],
        [float(k_h), float(dt), float(tend_max), 0.0],
        threads=n_axes * nc,
    )
    return du_out


_EVP_WS_CACHE: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_BT_WS_CACHE: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def run_natural_aerosol_4sp_metal(
    lib: Any,
    q_in: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    emis_4: torch.Tensor,
    precip_rate: torch.Tensor,
    u_star_cell: torch.Tensor,
    v_settle_4: tuple[float, float, float, float],
    v_dry_4: tuple[float, float, float, float],
    alpha_wet_4: tuple[float, float, float, float],
    mee_4: tuple[float, float, float, float],
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_sp, nz, nc = q_in.shape
    dz_2d = dz_c.view(nz, -1).contiguous()
    q_out = torch.empty_like(q_in)
    aod_out = torch.empty((n_sp, nc), dtype=q_in.dtype, device=q_in.device)
    lib.natural_aerosol_4sp_col(
        q_in.contiguous(),
        rho_d.contiguous(),
        dz_2d,
        emis_4.contiguous(),
        precip_rate.contiguous(),
        u_star_cell.contiguous(),
        q_out,
        aod_out,
        [n_sp, nz, nc, dz_2d.shape[1]],
        [float(x) for x in v_settle_4],
        [float(x) for x in v_dry_4],
        [float(x) for x in alpha_wet_4],
        [float(x) for x in mee_4],
        float(dt),
        threads=n_sp * nc,
    )
    return q_out, aod_out


def run_seaice_itd_sync_metal(
    lib: Any,
    a_cat: torch.Tensor,
    v_cat: torch.Tensor,
    m_target: torch.Tensor,
    m_prev: torch.Tensor,
    is_ocn_f: torch.Tensor,
    inv_rho_seaice: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_cat, nc = a_cat.shape
    a_out = torch.empty_like(a_cat)
    v_out = torch.empty_like(v_cat)
    lib.seaice_itd_sync_col(
        a_cat.contiguous(),
        v_cat.contiguous(),
        m_target.contiguous(),
        m_prev.contiguous(),
        is_ocn_f.contiguous(),
        a_out,
        v_out,
        [n_cat, nc, 0, 0],
        [float(inv_rho_seaice), 0.0, 0.0, 0.0],
        threads=nc,
    )
    return a_out, v_out


def run_seaice_evp_subcycle_metal(
    lib: Any,
    u_ice_e: torch.Tensor,
    sigma_1: torch.Tensor,
    sigma_s: torch.Tensor,
    p_strength: torch.Tensor,
    m_ice_e: torch.Tensor,
    a_ice_e: torch.Tensor,
    tau_air_e: torch.Tensor,
    u_ocn_e: torch.Tensor,
    grad_eta_e: torch.Tensor,
    cor_w: torch.Tensor,
    ep_idx_i32: torch.Tensor,
    c0_i32: torch.Tensor,
    c1_i32: torch.Tensor,
    eoc_i32: torch.Tensor,
    div_w: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    has_ice_e: torch.Tensor,
    is_ocn_f: torch.Tensor,
    alpha_e: float,
    inv_1pa: float,
    e2_inv: float,
    delta_min: float,
    dt_e: float,
    u_cap: float,
    rho_sw_cw: float,
    gravity: float,
    n_sub: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = u_ice_e.device
    nc = sigma_1.shape[0]
    ne = u_ice_e.shape[0]
    max_eoc = eoc_i32.shape[1]
    max_tr = ep_idx_i32.shape[1]

    u_cur = (u_ice_e * has_ice_e).contiguous()
    s1 = sigma_1.contiguous().clone()
    ss = sigma_s.contiguous().clone()

    key = (dev.type, dev.index or 0, ne)
    ws = _EVP_WS_CACHE.get(key)
    if ws is None:
        ws = (
            torch.empty(ne, dtype=u_cur.dtype, device=dev),
            torch.empty(ne, dtype=u_cur.dtype, device=dev),
        )
        _EVP_WS_CACHE[key] = ws
    u_half, f_ext = ws

    p_str_c = p_strength.contiguous()
    is_ocn_c = is_ocn_f.contiguous()
    div_w_c = div_w.contiguous()
    inv_dc_c = inv_dc_edge.contiguous()
    m_e_c = m_ice_e.contiguous()
    a_e_c = a_ice_e.contiguous()
    tau_c = tau_air_e.contiguous()
    uo_c = u_ocn_e.contiguous()
    geta_c = grad_eta_e.contiguous()
    hi_c = has_ice_e.contiguous()
    cor_c = cor_w.contiguous()

    u_cell_dim = [nc, max_eoc, 0, 0]
    f_cell_p = [float(alpha_e), float(inv_1pa), float(e2_inv), float(delta_min)]
    u_edge_dim = [ne, max_tr, 0, 0]
    f_s1_p = [float(dt_e), float(u_cap), float(rho_sw_cw), float(gravity)]
    f_s2_p = [float(dt_e), float(u_cap), float(rho_sw_cw), 0.0]

    for _ in range(n_sub):
        lib.seaice_evp_stress_cell(
            u_cur, p_str_c, is_ocn_c, eoc_i32, div_w_c, inv_dc_c,
            s1, ss, u_cell_dim, f_cell_p, threads=nc,
        )
        lib.seaice_evp_vel_s1_edge(
            u_cur, s1, m_e_c, a_e_c, tau_c, uo_c, geta_c, hi_c, inv_dc_c,
            c0_i32, c1_i32, ep_idx_i32, cor_c, u_half, f_ext,
            u_edge_dim, f_s1_p, threads=ne,
        )
        lib.seaice_evp_vel_s2_edge(
            u_cur, u_half, f_ext, m_e_c, a_e_c, uo_c, hi_c,
            ep_idx_i32, cor_c, u_edge_dim, f_s2_p, threads=ne,
        )
    return u_cur, s1, ss


def run_ogcm_barotropic_rk3_metal(
    lib: Any,
    u_bt_init: torch.Tensor,
    eta_init: torch.Tensor,
    f_baro: torch.Tensor,
    g_inv_dc: torch.Tensor,
    h_col_e: torch.Tensor,
    ocn_e0f: torch.Tensor,
    is_ocn_f: torch.Tensor,
    c0_i32: torch.Tensor,
    c1_i32: torch.Tensor,
    eoc_i32: torch.Tensor,
    div_w: torch.Tensor,
    n_bt: int,
    dt_bt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = u_bt_init.device
    ne = u_bt_init.shape[0]
    nc = eta_init.shape[0]
    max_eoc = eoc_i32.shape[1]

    key = (dev.type, dev.index or 0, ne, nc)
    ws = _BT_WS_CACHE.get(key)
    if ws is None:
        ws = (
            torch.empty(ne, dtype=u_bt_init.dtype, device=dev),
            torch.empty(ne, dtype=u_bt_init.dtype, device=dev),
            torch.empty(ne, dtype=u_bt_init.dtype, device=dev),
            torch.empty(nc, dtype=eta_init.dtype, device=dev),
            torch.empty(nc, dtype=eta_init.dtype, device=dev),
        )
        _BT_WS_CACHE[key] = ws
    u_bt, u_st, flux_bt, eta_bt, eta_st = ws

    u_bt.copy_(u_bt_init)
    eta_bt.copy_(eta_init)
    u_bt_acc = torch.zeros_like(u_bt)

    f_b = f_baro.contiguous()
    g_i = g_inv_dc.contiguous()
    hc_e = h_col_e.contiguous()
    oe = ocn_e0f.contiguous()
    io = is_ocn_f.contiguous()
    dw = div_w.contiguous()
    u_cell_dim = [nc, max_eoc, 0, 0]

    dt_total = max(float(n_bt) * float(dt_bt), 1.0)
    r_bt_pert = 1.0 / dt_total
    dt1 = float(dt_bt) / 3.0
    dt2 = 0.5 * float(dt_bt)
    dt3 = float(dt_bt)
    p_dt1 = [dt1, 1.0 / (1.0 + dt1 * r_bt_pert), 0.0, 0.0]
    p_dt2 = [dt2, 1.0 / (1.0 + dt2 * r_bt_pert), 0.0, 0.0]
    p_dt3 = [dt3, 1.0 / (1.0 + dt3 * r_bt_pert), 0.0, 0.0]

    for _ in range(n_bt):
        # Stage 1: (u_bt, eta_bt) -> (u_st, eta_st)
        lib.ogcm_bt_vel_edge(u_bt, eta_bt, f_b, g_i, hc_e, oe, c0_i32, c1_i32, u_st, flux_bt, ne, p_dt1, threads=ne)
        lib.ogcm_bt_eta_cell(eta_bt, flux_bt, io, eoc_i32, dw, eta_st, u_cell_dim, dt1, threads=nc)
        # Stage 2: (u_bt, eta_st) -> (u_st, eta_st)
        lib.ogcm_bt_vel_edge(u_bt, eta_st, f_b, g_i, hc_e, oe, c0_i32, c1_i32, u_st, flux_bt, ne, p_dt2, threads=ne)
        lib.ogcm_bt_eta_cell(eta_bt, flux_bt, io, eoc_i32, dw, eta_st, u_cell_dim, dt2, threads=nc)
        # Stage 3: (u_bt, eta_st) -> (u_st, eta_st), then commit to (u_bt, eta_bt)
        lib.ogcm_bt_vel_edge(u_bt, eta_st, f_b, g_i, hc_e, oe, c0_i32, c1_i32, u_st, flux_bt, ne, p_dt3, threads=ne)
        lib.ogcm_bt_eta_cell(eta_bt, flux_bt, io, eoc_i32, dw, eta_st, u_cell_dim, dt3, threads=nc)
        u_bt.copy_(u_st)
        eta_bt.copy_(eta_st)
        u_bt_acc.add_(u_bt)

    return u_bt.clone(), u_bt_acc * (1.0 / float(n_bt))


_OGCM_FCT_WS_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}


def run_ogcm_fct_zstar_metal(
    lib: Any,
    q: torch.Tensor,
    q_comp: torch.Tensor,
    f_nl_q: torch.Tensor,
    h_u_edge: torch.Tensor,
    sx_int: torch.Tensor,
    w_star: torch.Tensor,
    h_e: torch.Tensor,
    h_old_z: torch.Tensor,
    h_new_z: torch.Tensor,
    dz_c_int: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    wet_cf: torch.Tensor,
    wet_ef: torch.Tensor,
    wet_int_cf: torch.Tensor,
    wet_int_ef: torch.Tensor,
    c0_i32: torch.Tensor,
    c1_i32: torch.Tensor,
    eoc_i32: torch.Tensor,
    div_w: torch.Tensor,
    w_vor_edge: torch.Tensor,
    active_w: torch.Tensor,
    q_ref: torch.Tensor,
    dt: float,
    kappa_redi: float,
    enable_full_redi: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute batched `[B, nzo, nc]` OGCM Zalesak FCT + z* + Redi + Kahan on GPU (Metal/CUDA)."""
    B, nzo, nc = q.shape
    ne = int(h_u_edge.shape[-1])
    dev = q.device
    dtype = q.dtype

    key = (dev.type, dev.index or 0, B, nzo, nc, ne, dtype)
    ws = _OGCM_FCT_WS_CACHE.get(key)
    if ws is None:
        if len(_OGCM_FCT_WS_CACHE) >= 4:
            _OGCM_FCT_WS_CACHE.pop(next(iter(_OGCM_FCT_WS_CACHE)))
        ws = (
            torch.empty((B, nzo, ne), device=dev, dtype=dtype),       # fl_h
            torch.empty((B, nzo, ne), device=dev, dtype=dtype),       # ah_e
            torch.empty((B, nzo, ne), device=dev, dtype=dtype),       # f_dif
            torch.empty((B, max(nzo - 1, 1), ne), device=dev, dtype=dtype),  # s_dqdx_ew
            torch.empty((B, nzo, nc), device=dev, dtype=dtype),       # qp_td
            torch.empty((B, max(nzo - 1, 1), nc), device=dev, dtype=dtype),  # fl_v
            torch.empty((B, max(nzo - 1, 1), nc), device=dev, dtype=dtype),  # av_int
            torch.empty((B, nzo, nc), device=dev, dtype=dtype),       # r_plus
            torch.empty((B, nzo, nc), device=dev, dtype=dtype),       # r_minus
            torch.empty((B, nzo, nc), device=dev, dtype=dtype),       # qp
        )
        _OGCM_FCT_WS_CACHE[key] = ws
    fl_h, ah_e, f_dif, s_dqdx_ew, qp_td, fl_v, av_int, r_plus, r_minus, qp = ws

    torch.sub(q, q_ref, out=qp)
    qp.mul_(wet_cf.unsqueeze(0))
    q_c = q.contiguous()
    qc_c = q_comp.contiguous()
    fnl_c = f_nl_q.contiguous()
    hu_c = h_u_edge.contiguous()
    he_c = h_e.contiguous()
    ho_c = h_old_z.contiguous()
    hn_c = h_new_z.contiguous()
    ws_c = w_star.contiguous()
    wcf_c = wet_cf.contiguous()
    wef_c = wet_ef.contiguous()
    wic_c = wet_int_cf.contiguous()
    wie_c = wet_int_ef.contiguous()
    sx_c = sx_int.contiguous()
    dz_c = dz_c_int.contiguous()
    idc_c = inv_dc_edge.contiguous()
    wve_c = w_vor_edge.contiguous()
    dw_c = div_w.contiguous()
    aw_c = active_w.contiguous()

    u_dim = [B, nzo, nc, ne]
    redi_f = 1.0 if enable_full_redi else 0.0
    n_edge_tot = B * nzo * ne
    n_cell_tot = B * nzo * nc

    lib.ogcm_fct_edge(
        c0_i32, c1_i32, idc_c, wve_c, dz_c, qp, hu_c, he_c, wef_c, wic_c, wie_c, sx_c,
        fl_h, ah_e, f_dif, s_dqdx_ew,
        u_dim, [float(kappa_redi), redi_f, 0.0, 0.0],
        threads=n_edge_tot,
    )
    lib.ogcm_fct_cell_td(
        eoc_i32, dw_c, qp, ho_c, hn_c, fl_h, ws_c, wcf_c, wic_c,
        qp_td, fl_v, av_int,
        u_dim, float(dt),
        threads=n_cell_tot,
    )
    lib.ogcm_fct_bounds_ratios(
        c0_i32, c1_i32, eoc_i32, dw_c, qp, qp_td, hn_c, ah_e, av_int, wcf_c, wef_c, wic_c,
        r_plus, r_minus,
        u_dim, float(dt),
        threads=n_cell_tot,
    )
    q_out = torch.empty_like(q_c)
    q_comp_out = torch.empty_like(qc_c)
    lib.ogcm_fct_limit_kahan(
        c0_i32, c1_i32, eoc_i32, dw_c, aw_c,
        q_c, qc_c, qp, fnl_c, ho_c, hn_c, fl_h, ah_e, f_dif, s_dqdx_ew,
        fl_v, av_int, r_plus, r_minus, wcf_c, wef_c, wic_c,
        q_out, q_comp_out,
        u_dim, [float(dt), float(kappa_redi), redi_f, 0.0],
        threads=n_cell_tot,
    )
    return q_out, q_comp_out


def run_ocean_bgc_column_metal(
    lib: Any,
    no3: torch.Tensor,
    phy: torch.Tensor,
    zoo: torch.Tensor,
    det_n: torch.Tensor,
    det_c: torch.Tensor,
    dic: torch.Tensor,
    alk: torch.Tensor,
    no3_comp: torch.Tensor,
    dic_comp: torch.Tensor,
    alk_comp: torch.Tensor,
    temp_k: torch.Tensor,
    sal_psu: torch.Tensor,
    h_c: torch.Tensor,
    wet_cf: torch.Tensor,
    h0_new: torch.Tensor,
    sw_par_sfc: torch.Tensor,
    u10_ocn: torch.Tensor,
    f_seaice: torch.Tensor,
    co2_ppmv: float,
    dt: float,
    rho_sw: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    nzo, nc = no3.shape
    dev = no3.device
    dtype = no3.dtype

    no3_io = no3.contiguous().clone()
    phy_io = phy.contiguous().clone()
    zoo_io = zoo.contiguous().clone()
    dn_io = det_n.contiguous().clone()
    dc_io = det_c.contiguous().clone()
    dic_io = dic.contiguous().clone()
    alk_io = alk.contiguous().clone()
    no3_c = no3_comp.contiguous().clone()
    dic_c = dic_comp.contiguous().clone()
    alk_c = alk_comp.contiguous().clone()

    sfc_in = torch.stack([h0_new, sw_par_sfc, u10_ocn, f_seaice], dim=0).contiguous()
    diag_3d = torch.empty((3, nzo, nc), dtype=dtype, device=dev)
    diag_2d = torch.empty((5, nc), dtype=dtype, device=dev)

    lib.ocean_bgc_column_col(
        no3_io, phy_io, zoo_io, dn_io, dc_io, dic_io, alk_io,
        no3_c, dic_c, alk_c,
        temp_k.contiguous(), sal_psu.contiguous(), h_c.contiguous(), wet_cf.contiguous(),
        sfc_in, diag_3d, diag_2d,
        [nzo, nc, 0, 0],
        [float(dt), float(co2_ppmv), float(rho_sw), 0.0],
        threads=nc,
    )
    return (
        no3_io, phy_io, zoo_io, dn_io, dc_io, dic_io, alk_io,
        no3_c, dic_c, alk_c,
        diag_3d[0], diag_3d[1], diag_3d[2],
        diag_2d[0], diag_2d[1], diag_2d[2], diag_2d[3], diag_2d[4],
    )


_SCVT_PREP_WS_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def run_scvt_rk3_prep_metal(
    lib: Any,
    mc: dict[str, Any],
    m_s: torch.Tensor,
    Thm_s: torch.Tensor,
    Thm0: torch.Tensor,
    u_s: torch.Tensor,
    w_s: torch.Tensor,
    dz_c: torch.Tensor,
    dz_w_int: torch.Tensor,
    dz_w_e: torch.Tensor,
    z_c: torch.Tensor,
    q_load: torch.Tensor,
    area_e: torch.Tensor,
    zw_slope_e: torch.Tensor,
    rot_acc: torch.Tensor,
    visc_u: torch.Tensor,
    du_ext: torch.Tensor,
    sponge_w: torch.Tensor,
    dt_ac: float,
    beta_hevi: float,
    dtau_stage: float = 0.0,
) -> tuple[torch.Tensor, ...]:
    """Fused GPU SCVT RK3 stage prep (`scvt_rk3_prep_cell` SoA + `scvt_rk3_prep_edge` float4)."""
    nz, nc = m_s.shape
    ne = int(u_s.shape[1])
    dev = m_s.device
    dtype = m_s.dtype

    ws_key = (dev.type, dev.index or 0, nz, nc, ne, dtype)
    ws = _SCVT_PREP_WS_CACHE.get(ws_key)
    if ws is None:
        if len(_SCVT_PREP_WS_CACHE) >= 4:
            _SCVT_PREP_WS_CACHE.pop(next(iter(_SCVT_PREP_WS_CACHE)))
        ws = {
            "pres_s": torch.empty((nz, nc), device=dev, dtype=dtype),
            "dp_dThm_s": torch.empty((nz, nc), device=dev, dtype=dtype),
            "q_odz": torch.empty((nz, nc), device=dev, dtype=dtype),
            "thm_s": torch.empty((nz, nc), device=dev, dtype=dtype),
            "rtot_s": torch.empty((nz, nc), device=dev, dtype=dtype),
            "ke_cell": torch.empty((nz, nc), device=dev, dtype=dtype),
            "u_dot_gradz_s": torch.empty((nz - 1, nc), device=dev, dtype=dtype),
            "hevi_c": torch.empty((7, nz - 1, nc), device=dev, dtype=dtype),
            "p_cell": torch.empty((nz, nc), device=dev, dtype=dtype),
            "k_slope_top": torch.empty((nc,), device=dev, dtype=torch.float32),
            "k_slope_ptr": -1,
            "edge_c4": torch.empty((nz, ne, 4), device=dev, dtype=dtype),
            "tend_u_slow": torch.empty((nz, ne), device=dev, dtype=dtype),
        }
        _SCVT_PREP_WS_CACHE[ws_key] = ws

    zw_c = zw_slope_e.contiguous()
    zw_ptr = zw_c.data_ptr()
    if ws["k_slope_ptr"] != zw_ptr:
        lib.scvt_slope_top_cell(
            mc["eoc"], mc["esgn"], zw_c, ws["k_slope_top"], [nz, nc, ne, 0], threads=nc,
        )
        ws["k_slope_ptr"] = zw_ptr
    k_slope_top = ws["k_slope_top"]

    pres_s = ws["pres_s"]
    dp_dThm_s = ws["dp_dThm_s"]
    q_odz = ws["q_odz"]
    thm_s = ws["thm_s"]
    rtot_s = ws["rtot_s"]
    ke_cell = ws["ke_cell"]
    u_dot_gradz_s = ws["u_dot_gradz_s"]
    hevi_c = ws["hevi_c"]
    p_cell = ws["p_cell"]

    Thm_s_c = Thm_s.contiguous()
    Thm0_c = Thm0.contiguous()
    same_thm0 = 1 if (Thm0_c.data_ptr() == Thm_s_c.data_ptr()) else 0
    u_dim = [nz, nc, ne, same_thm0]
    lib.scvt_rk3_prep_cell(
        mc["eoc"], mc["esgn"], area_e.reshape(-1).contiguous(), mc["area_cell"],
        m_s.contiguous(), Thm_s_c, Thm0_c, u_s.contiguous(),
        dz_c.contiguous(), dz_w_int.contiguous(), q_load.contiguous(),
        zw_c, sponge_w.reshape(-1).contiguous(), k_slope_top,
        pres_s, dp_dThm_s, q_odz, thm_s, rtot_s, ke_cell, u_dot_gradz_s,
        hevi_c, p_cell,
        u_dim,
        [float(C.P_REF), float(C.R_D), float(C.GAMMA_D), float(C.GRAVITY)],
        [float(dt_ac), float(beta_hevi), 0.0, 0.0],
        threads=nc,
    )

    edge_c4 = ws["edge_c4"]
    tend_u_slow = ws["tend_u_slow"]

    dtau_v = float(dtau_stage) if float(dtau_stage) > 0.0 else float(dt_ac)
    lib.scvt_rk3_prep_edge(
        mc["c0"], mc["c1"], mc["inv_dc"],
        m_s.contiguous(), thm_s, rtot_s, pres_s, z_c.contiguous(), ke_cell,
        u_s.contiguous(), w_s.contiguous(), u_dot_gradz_s, dz_w_e.contiguous(),
        rot_acc.contiguous(), visc_u.contiguous(), du_ext.contiguous(),
        edge_c4, tend_u_slow,
        u_dim, [float(C.GRAVITY), dtau_v, 0.0, 0.0],
        threads=nz * ne,
    )
    return (
        edge_c4, pres_s, dp_dThm_s, q_odz, hevi_c, p_cell, k_slope_top, u_dot_gradz_s, tend_u_slow,
    )


def run_ogcm_baroclinic_f_fixed_metal(
    lib: Any,
    mc: dict[str, Any],
    p_baroc: torch.Tensor,
    lap_u: torch.Tensor,
    tau_n_e: torch.Tensor,
    h_e0_safe: torch.Tensor,
    wet_ef: torch.Tensor,
    a_h: float,
) -> torch.Tensor:
    nzo, ne = lap_u.shape
    nc = int(p_baroc.shape[1])
    f_fixed = torch.empty_like(lap_u)
    lib.ogcm_baroclinic_f_fixed(
        mc["c0"], mc["c1"], mc["inv_dc"],
        p_baroc.contiguous(), lap_u.contiguous(),
        tau_n_e.contiguous(), h_e0_safe.contiguous(), wet_ef.contiguous(),
        f_fixed,
        [nzo, nc, ne, 0],
        [float(1.0 / C.RHO_SEAWATER), float(a_h), 0.0, 0.0],
        threads=nzo * ne,
    )
    return f_fixed


def run_ogcm_baroclinic_rk3_update_metal(
    lib: Any,
    u0: torch.Tensor,
    u_s: torch.Tensor,
    cor_3d: torch.Tensor,
    f_fixed: torch.Tensor,
    is_bot_f: torch.Tensor,
    wet_ef: torch.Tensor,
    rk_dt: float,
    r_rayleigh: float,
) -> torch.Tensor:
    u_out = torch.empty_like(u0)
    n_tot = int(u0.numel())
    lib.ogcm_baroclinic_rk3_update(
        u0.contiguous(), u_s.contiguous(), cor_3d.contiguous(), f_fixed.contiguous(),
        is_bot_f.contiguous(), wet_ef.contiguous(), u_out,
        n_tot, [float(rk_dt), float(r_rayleigh), 0.0, 0.0],
        threads=n_tot,
    )
    return u_out


def run_ogcm_mom_vdiff_metal(
    lib: Any,
    mc: dict[str, Any],
    u_star: torch.Tensor,
    av_diag: torch.Tensor,
    h_e_safe: torch.Tensor,
    dz_int_e: torch.Tensor,
    wet_ef: torch.Tensor,
    wet_int_ef: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    nzo, ne = u_star.shape
    nc = int(av_diag.shape[1])
    u_out = torch.empty_like(u_star)
    lib.ogcm_mom_vdiff_col(
        mc["c0"], mc["c1"],
        u_star.contiguous(), av_diag.contiguous(), h_e_safe.contiguous(),
        dz_int_e.contiguous(), wet_ef.contiguous(), wet_int_ef.contiguous(),
        u_out,
        [nzo, nc, ne, 0], float(dt),
        threads=ne,
    )
    return u_out


def run_ogcm_tracer_vdiff_freeze_metal(
    lib: Any,
    T: torch.Tensor,
    t_comp: torch.Tensor,
    S: torch.Tensor,
    s_comp: torch.Tensor,
    kv_diag: torch.Tensor,
    kv_redi: torch.Tensor,
    z_c: torch.Tensor,
    dz_c_int: torch.Tensor,
    h_c_safe: torch.Tensor,
    h0_new: torch.Tensor,
    h0_frz: torch.Tensor,
    wet_cf: torch.Tensor,
    wet_int_cf: torch.Tensor,
    sfc_m_seaice: torch.Tensor,
    sfc_t_ocean_ml: torch.Tensor,
    dt: float,
    t_ref_eos: float,
    s_ref_psu: float,
    s_seaice_psu: float,
    eos_a0: float,
    eos_at: float,
    eos_az: float,
    eos_as: float,
    eos_b0: float,
    kappa_v_conv: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nzo, nc = T.shape
    dev = T.device
    dtype = T.dtype
    T_out = torch.empty_like(T)
    t_comp_out = torch.empty_like(t_comp)
    S_out = torch.empty_like(S)
    s_comp_out = torch.empty_like(s_comp)
    kv_out = torch.empty((nzo - 1, nc), device=dev, dtype=dtype)
    t_ml_out = torch.empty_like(sfc_t_ocean_ml)
    m_ice_out = torch.empty_like(sfc_m_seaice)

    lib.ogcm_tracer_vdiff_freeze_col(
        T.contiguous(), t_comp.contiguous(), S.contiguous(), s_comp.contiguous(),
        kv_diag.contiguous(), kv_redi.contiguous(), z_c.reshape(-1).contiguous(),
        dz_c_int.contiguous(), h_c_safe.contiguous(), h0_new.contiguous(),
        wet_cf.contiguous(), wet_int_cf.contiguous(),
        sfc_m_seaice.contiguous(), sfc_t_ocean_ml.contiguous(),
        T_out, t_comp_out, S_out, s_comp_out, kv_out, t_ml_out, m_ice_out,
        h0_frz.contiguous(),
        [nzo, nc, 0, 0],
        [float(dt), float(C.RHO_SEAWATER), float(C.CP_SEAWATER), float(C.L_F0)],
        [float(C.T_FREEZE_OCEAN), float(t_ref_eos), float(s_ref_psu), float(s_seaice_psu)],
        [float(eos_a0), float(eos_at), float(eos_az), float(eos_as)],
        [float(eos_b0), float(kappa_v_conv), 0.0, 0.0],
        threads=nc,
    )
    return T_out, t_comp_out, S_out, s_comp_out, kv_out, t_ml_out, m_ice_out


def run_ogcm_kpp_vdiff_metal(
    lib: Any,
    rho_anom: torch.Tensor,
    u_cell: torch.Tensor,
    v_cell: torch.Tensor,
    u_star_o: torch.Tensor,
    b0_sfc: torch.Tensor,
    wt_sfc: torch.Tensor,
    ws_sfc: torch.Tensor,
    wet_cf: torch.Tensor,
    wet_int_cf: torch.Tensor,
    h_c: torch.Tensor,
    dz_c_int: torch.Tensor,
    z_c: torch.Tensor,
    z_int_m: torch.Tensor,
    h_col_kpp: torch.Tensor,
    h_col_safe: torch.Tensor,
    f_abs: torch.Tensor,
    kv_bg: torch.Tensor,
    kv_tide_num: torch.Tensor,
    kpp_ri_crit: float,
    kpp_kv_max: float,
    kpp_cs_nl: float,
    shear_ri_0: float,
    shear_nu_0: float,
    tidal_kv_max: float,
    kappa_v_conv: float,
    a_v_mom: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nzo, nc = rho_anom.shape
    dev = rho_anom.device
    dtype = rho_anom.dtype
    kv_out = torch.empty((nzo - 1, nc), device=dev, dtype=dtype)
    av_out = torch.empty((nzo - 1, nc), device=dev, dtype=dtype)
    h_bl_out = torch.empty(nc, device=dev, dtype=dtype)
    f_nl_t = torch.empty((nzo - 1, nc), device=dev, dtype=dtype)
    f_nl_s = torch.empty((nzo - 1, nc), device=dev, dtype=dtype)

    lib.ogcm_kpp_vdiff_col(
        rho_anom.contiguous(), u_cell.contiguous(), v_cell.contiguous(),
        u_star_o.contiguous(), b0_sfc.contiguous(), wt_sfc.contiguous(), ws_sfc.contiguous(),
        wet_cf.contiguous(), wet_int_cf.contiguous(), h_c.contiguous(), dz_c_int.contiguous(),
        z_c.reshape(-1).contiguous(), z_int_m.reshape(-1).contiguous(),
        h_col_kpp.contiguous(), h_col_safe.contiguous(), f_abs.contiguous(),
        kv_bg.reshape(-1).contiguous(), kv_tide_num.contiguous(),
        kv_out, av_out, h_bl_out, f_nl_t, f_nl_s,
        [nzo, nc, 0, 0],
        [float(C.GRAVITY), float(C.RHO_SEAWATER), float(C.VON_KARMAN), float(kpp_ri_crit)],
        [float(kpp_kv_max), float(kpp_cs_nl), float(shear_ri_0), float(shear_nu_0)],
        [float(tidal_kv_max), float(kappa_v_conv), float(a_v_mom), 0.0],
        threads=nc,
    )
    return kv_out, av_out, h_bl_out, f_nl_t, f_nl_s


def run_ogcm_pre_kpp_metal(
    lib: Any,
    T: torch.Tensor,
    S: torch.Tensor,
    t_comp: torch.Tensor,
    s_comp: torch.Tensor,
    sfc_t_ocean_ml: torch.Tensor,
    sfc_m_seaice: torch.Tensor,
    m_seaice_prev: torch.Tensor,
    r_coast: torch.Tensor,
    q_calve: torch.Tensor,
    evap_cell: torch.Tensor,
    precip_cell: torch.Tensor,
    sw_net_ocn: torch.Tensor,
    wet_cf: torch.Tensor,
    h_c_safe: torch.Tensor,
    dz_nom: torch.Tensor,
    z_c: torch.Tensor,
    frac_out_0: torch.Tensor,
    frac_sub: torch.Tensor,
    dt: float,
    apply_sw: bool,
    t_ref_eos: float,
    s_ref_psu: float,
    s_seaice_psu: float,
    eos_a0: float,
    eos_at: float,
    eos_az: float,
    eos_as: float,
    eos_b0: float,
) -> tuple[torch.Tensor, ...]:
    nzo, nc = T.shape
    dev = T.device
    dtype = T.dtype
    T_out = torch.empty_like(T)
    S_out = torch.empty_like(S)
    t_comp_out = torch.empty_like(t_comp)
    s_comp_out = torch.empty_like(s_comp)
    rho_anom = torch.empty_like(T)
    p_baroc = torch.empty_like(T)
    sfc_flux3 = torch.empty((3, nc), device=dev, dtype=dtype)

    lib.ogcm_pre_kpp_col(
        T.contiguous(), S.contiguous(), t_comp.contiguous(), s_comp.contiguous(),
        sfc_t_ocean_ml.contiguous(), sfc_m_seaice.contiguous(), m_seaice_prev.contiguous(),
        r_coast.contiguous(), q_calve.contiguous(), evap_cell.contiguous(),
        precip_cell.contiguous(), sw_net_ocn.contiguous(),
        wet_cf.contiguous(), h_c_safe.contiguous(), dz_nom.reshape(-1).contiguous(),
        z_c.reshape(-1).contiguous(), frac_out_0.contiguous(), frac_sub.contiguous(),
        T_out, S_out, t_comp_out, s_comp_out, rho_anom, p_baroc, sfc_flux3,
        [nzo, nc, 1 if apply_sw else 0, 0],
        [float(dt), float(C.RHO_SEAWATER), float(C.CP_SEAWATER), float(C.GRAVITY)],
        [float(t_ref_eos), float(s_ref_psu), float(s_seaice_psu), float(eos_b0)],
        [float(eos_a0), float(eos_at), float(eos_az), float(eos_as)],
        threads=nc,
    )
    return T_out, S_out, t_comp_out, s_comp_out, sfc_flux3[0], sfc_flux3[1], sfc_flux3[2], rho_anom, p_baroc


_OGCM_GM_WS_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def run_ogcm_gm_bolus_edge_metal(
    lib: Any,
    mc: dict[str, Any],
    w_vor_edge: torch.Tensor,
    u_star: torch.Tensor,
    u_bt_mean: torch.Tensor,
    u_star_baro: torch.Tensor,
    rho_anom: torch.Tensor,
    dz_c_int: torch.Tensor,
    wet_int_cf: torch.Tensor,
    h_bl: torch.Tensor,
    h_e: torch.Tensor,
    h_e_safe: torch.Tensor,
    z_int_m: torch.Tensor,
    wet_ef: torch.Tensor,
    wet_int_ef: torch.Tensor,
    gm_scale_e: torch.Tensor | None,
    fk_coef_e: torch.Tensor | None,
    kappa_gm: float,
    gm_slope_max: float,
    enable_scale_aware_gm: bool,
    enable_submeso: bool,
    enable_full_redi: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nzo, ne = u_star.shape
    nc = int(rho_anom.shape[1])
    dev = u_star.device
    dtype = u_star.dtype
    u_new = torch.empty_like(u_star)
    sx_int = torch.empty((max(nzo - 1, 1), ne), device=dev, dtype=dtype)
    h_u_edge = torch.empty_like(u_star)
    s2_edge_w = torch.empty((max(nzo - 1, 1), ne), device=dev, dtype=dtype) if enable_full_redi else torch.empty(1, device=dev, dtype=dtype)
    psi_key = (dev.type, dev.index or 0, max(nzo - 1, 1), ne, dtype)
    psi_int = _OGCM_GM_WS_CACHE.get(psi_key)
    if psi_int is None:
        if len(_OGCM_GM_WS_CACHE) >= 4:
            _OGCM_GM_WS_CACHE.pop(next(iter(_OGCM_GM_WS_CACHE)))
        psi_int = torch.empty((max(nzo - 1, 1), ne), device=dev, dtype=dtype)
        _OGCM_GM_WS_CACHE[psi_key] = psi_int
    dummy1 = torch.zeros(1, device=dev, dtype=dtype)
    gm_s = gm_scale_e.contiguous() if (enable_scale_aware_gm and gm_scale_e is not None) else dummy1
    fk_c = fk_coef_e.contiguous() if (enable_submeso and fk_coef_e is not None) else dummy1
    flags = ((1 if (enable_scale_aware_gm and gm_scale_e is not None) else 0) << 2) | (
        (1 if (enable_submeso and fk_coef_e is not None) else 0) << 1
    ) | (1 if enable_full_redi else 0)

    hes_c = h_e_safe.contiguous()
    wef_c = wet_ef.contiguous()
    if nzo >= 2:
        lib.ogcm_gm_slope_psi_edge(
            mc["c0"], mc["c1"], mc["inv_dc"], w_vor_edge.contiguous(),
            rho_anom.contiguous(), dz_c_int.contiguous(), wet_int_cf.contiguous(),
            h_bl.contiguous(), hes_c, z_int_m.reshape(-1).contiguous(),
            wef_c, wet_int_ef.contiguous(), gm_s, fk_c,
            sx_int, psi_int, s2_edge_w,
            [nzo, nc, ne, flags],
            [float(kappa_gm), float(gm_slope_max), float(C.GRAVITY / C.RHO_SEAWATER), 0.0],
            threads=(nzo - 1) * ne,
        )
    lib.ogcm_gm_apply_edge(
        u_star.contiguous(), u_bt_mean.contiguous(), u_star_baro.contiguous(),
        h_e.contiguous(), hes_c, wef_c, psi_int,
        u_new, h_u_edge,
        [nzo, ne, 0, 0],
        threads=nzo * ne,
    )
    return u_new, sx_int, h_u_edge, s2_edge_w


def run_crm_student_tile_update_metal(
    lib: Any,
    atmos_rho: torch.Tensor,
    stu_thm: torch.Tensor,
    stu_qv: torch.Tensor,
    stu_qc: torch.Tensor,
    stu_qi: torch.Tensor,
    stu_qp: torch.Tensor,
    stu_fcld: torch.Tensor,
    stu_u: torch.Tensor,
    stu_v: torch.Tensor,
    stu_rain: torch.Tensor,
    stu_snow: torch.Tensor,
    stu_fprec: torch.Tensor,
    score_cld: torch.Tensor | None,
    score_prec: torch.Tensor | None,
    rho_pre_0: torch.Tensor,
    dz_0: torch.Tensor,
    exner_0: torch.Tensor,
    shf_tile: torch.Tensor,
    evap_tile: torch.Tensor,
    dthm_cp: torch.Tensor | None,
    dqv_cp: torch.Tensor | None,
    u_gust: torch.Tensor | None,
    crm_rho: torch.Tensor,
    crm_thm: torch.Tensor,
    crm_qn: torch.Tensor,
    crm_qp: torch.Tensor,
    crm_qv: torch.Tensor,
    crm_qc: torch.Tensor,
    crm_qi: torch.Tensor,
    crm_u: torch.Tensor,
    crm_v: torch.Tensor,
    rain_3d: torch.Tensor,
    snow_3d: torch.Tensor,
    ny_t: int,
    nx_t: int,
    step_count: int,
    dt: float,
) -> None:
    nz, nc = atmos_rho.shape
    n_sub = int(ny_t * nx_t)
    dev = atmos_rho.device
    dtype = atmos_rho.dtype
    stu_1d = torch.stack([stu_rain, stu_snow, stu_fprec, rho_pre_0, dz_0, exner_0], dim=0).contiguous()
    dummy1 = torch.zeros(1, device=dev, dtype=dtype)
    has_sc = 1 if score_cld is not None else 0
    has_sp = 1 if score_prec is not None else 0
    has_cp = 1 if (dthm_cp is not None and dqv_cp is not None and u_gust is not None) else 0
    sc_buf = score_cld.reshape(nc, n_sub).contiguous() if score_cld is not None else dummy1
    sp_buf = score_prec.reshape(nc, n_sub).contiguous() if score_prec is not None else dummy1
    dt_cp_buf = dthm_cp.reshape(nc, n_sub).contiguous() if has_cp else dummy1
    dq_cp_buf = dqv_cp.reshape(nc, n_sub).contiguous() if has_cp else dummy1
    ug_buf = u_gust.reshape(nc, n_sub).contiguous() if has_cp else dummy1
    step_shift = int(step_count) // 12
    flags = (step_shift << 3) | (has_cp << 2) | (has_sp << 1) | has_sc
    lib.crm_student_tile_update_col(
        atmos_rho.contiguous(), stu_thm.contiguous(), stu_qv.contiguous(),
        stu_qc.contiguous(), stu_qi.contiguous(), stu_qp.contiguous(),
        stu_fcld.contiguous(), stu_u.contiguous(), stu_v.contiguous(),
        stu_1d, sc_buf, sp_buf,
        shf_tile.reshape(nc, n_sub).contiguous(),
        evap_tile.reshape(nc, n_sub).contiguous(),
        dt_cp_buf, dq_cp_buf, ug_buf,
        crm_rho, crm_thm, crm_qn, crm_qp, crm_qv, crm_qc, crm_qi, crm_u, crm_v,
        rain_3d, snow_3d,
        [nz, nc, (int(ny_t) << 16) | int(nx_t), flags],
        [float(dt), float(C.CP_D), 0.85, 0.0],
        threads=(nc * n_sub,),
        group_size=(n_sub,),
    )


def run_surface_land_pre_bio_metal(
    lib: Any,
    t_sun_cur: torch.Tensor,
    t_sh_cur: torch.Tensor,
    t_gnd_cur: torch.Tensor,
    q_af_in: torch.Tensor,
    c_h_land: torch.Tensor,
    wind_eff_land: torch.Tensor,
    u_star_land: torch.Tensor,
    pb_const: torch.Tensor,
    pb_out: torch.Tensor,
) -> None:
    n_tot = int(t_sun_cur.numel())
    lib.surface_land_pre_bio_col(
        t_sun_cur.contiguous(), t_sh_cur.contiguous(), t_gnd_cur.contiguous(),
        q_af_in.contiguous(), c_h_land.contiguous(), wind_eff_land.contiguous(),
        u_star_land.contiguous(), pb_const, pb_out,
        n_tot,
        [float(C.GRAVITY), float(C.VON_KARMAN), float(C.EPS_WATER), float(C.E_TRIPLE)],
        [float(C.T_FREEZE), float(C.T_ICE_ALL), float(C.T_LIQ_ALL), 0.0],
        threads=n_tot,
    )


def run_surface_land_newton_jac_metal(
    lib: Any,
    jac_const: torch.Tensor,
    pb_out: torch.Tensor,
    t_sun_cur: torch.Tensor,
    t_sh_cur: torch.Tensor,
    t_gnd_cur: torch.Tensor,
    gsw_sun: torch.Tensor,
    gsw_sh: torch.Tensor,
    prev_dt_sun: torch.Tensor,
    prev_dt_sh: torch.Tensor,
    prev_dt_gnd: torch.Tensor,
    jac_out: torch.Tensor,
    dt: float,
    eps_g: float,
    apply_damping: bool,
    write_full_diag: bool,
) -> None:
    n_tot = int(t_sun_cur.numel())
    lib.surface_land_newton_jac_col(
        jac_const, pb_out,
        t_sun_cur.contiguous(), t_sh_cur.contiguous(), t_gnd_cur.contiguous(),
        gsw_sun.contiguous(), gsw_sh.contiguous(),
        prev_dt_sun.contiguous(), prev_dt_sh.contiguous(), prev_dt_gnd.contiguous(),
        jac_out,
        [n_tot, 1 if apply_damping else 0, 1 if write_full_diag else 0, 0],
        [float(dt), float(eps_g), float(C.STEFAN_BOLTZMANN), float(C.T_FREEZE)],
        [float(C.L_V0), float(C.L_S0), 0.0, 0.0],
        threads=n_tot,
    )


def run_surface_ocean_seaice_post_metal(
    lib: Any,
    t_ocean_ml: torch.Tensor,
    t_ocean_ml_comp: torch.Tensor,
    h_ocean_ml: torch.Tensor,
    m_seaice: torch.Tensor,
    t_seaice: torch.Tensor,
    swe_seaice: torch.Tensor,
    f_ice: torch.Tensor,
    qsat_ow: torch.Tensor,
    qsat_si: torch.Tensor,
    ga_h_o: torch.Tensor,
    ga_q_o: torch.Tensor,
    rho_sfc: torch.Tensor,
    cp_air: torch.Tensor,
    t_air_sfc: torch.Tensor,
    qv_1: torch.Tensor,
    sw1_dir_down: torch.Tensor,
    sw1_dif_down: torch.Tensor,
    sw2_dir_down: torch.Tensor,
    sw2_dif_down: torch.Tensor,
    lw_down: torch.Tensor,
    precip_snow: torch.Tensor,
    mu0: torch.Tensor,
    dt: float,
    emiss_ocn: float,
    rho_snow: float,
    alb_si_v: float,
    alb_si_n: float,
    alb_sn_v: float,
    alb_sn_n: float,
) -> tuple[torch.Tensor, ...]:
    orig_shape = t_ocean_ml.shape
    n_tot = int(t_ocean_ml.numel())
    ocn_out = torch.empty((9, n_tot), device=t_ocean_ml.device, dtype=t_ocean_ml.dtype)
    lib.surface_ocean_seaice_post_col(
        t_ocean_ml.contiguous(), t_ocean_ml_comp.contiguous(), h_ocean_ml.contiguous(),
        m_seaice.contiguous(), t_seaice.contiguous(), swe_seaice.contiguous(),
        f_ice.contiguous(), qsat_ow.contiguous(), qsat_si.contiguous(),
        ga_h_o.contiguous(), ga_q_o.contiguous(), rho_sfc.contiguous(),
        cp_air.contiguous(), t_air_sfc.contiguous(), qv_1.contiguous(),
        sw1_dir_down.contiguous(), sw1_dif_down.contiguous(),
        sw2_dir_down.contiguous(), sw2_dif_down.contiguous(),
        lw_down.contiguous(), precip_snow.contiguous(), mu0.contiguous(),
        ocn_out, n_tot,
        [float(dt), float(emiss_ocn), float(C.STEFAN_BOLTZMANN), float(C.T_FREEZE)],
        [float(C.T_FREEZE_OCEAN), float(C.L_V0), float(C.L_S0), float(C.L_F0)],
        [float(C.RHO_SEAWATER), float(C.CP_SEAWATER), float(C.RHO_SEAICE), float(C.K_THERM_SEAICE)],
        [float(rho_snow), float(C.K_THERM_SNOW), float(alb_si_v), float(alb_si_n)],
        [float(alb_sn_v), float(alb_sn_n), 0.0, 0.0],
        threads=n_tot,
    )
    return tuple(ocn_out[k].view(orig_shape) for k in range(9))


def run_crm_distill_extract_metal(
    lib: Any,
    rho_d: torch.Tensor,
    theta_m: torch.Tensor,
    q_v: torch.Tensor,
    q_c: torch.Tensor,
    q_i: torch.Tensor,
    q_p: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dz_c: torch.Tensor,
    dthm_ls: torch.Tensor,
    dqn_ls: torch.Tensor,
    q_rad: torch.Tensor,
    shf: torch.Tensor,
    lhf: torch.Tensor,
    u_star: torch.Tensor,
    tau_x: torch.Tensor,
    tau_y: torch.Tensor,
    is_ocean: torch.Tensor,
    f_cor: torch.Tensor,
    struct_mat: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nz, n_col = rho_d.shape
    dev = rho_d.device
    dtype = rho_d.dtype
    d_in = 14 * nz + 23
    sfc_7 = torch.stack([shf, lhf, u_star, tau_x, tau_y, is_ocean.to(dtype), f_cor], dim=0).contiguous()
    x_out = torch.empty((n_col, d_in), device=dev, dtype=dtype)
    exner_out = torch.empty((nz, n_col), device=dev, dtype=dtype)
    temp_out = torch.empty((nz, n_col), device=dev, dtype=dtype)
    n_grp = (n_col + 31) // 32
    lib.crm_distill_extract_col(
        rho_d.contiguous(), theta_m.contiguous(), q_v.contiguous(),
        q_c.contiguous(), q_i.contiguous(), q_p.contiguous(),
        u.contiguous(), v.contiguous(), dz_c.contiguous(),
        dthm_ls.contiguous(), dqn_ls.contiguous(), q_rad.contiguous(),
        sfc_7, struct_mat.contiguous(),
        x_out, exner_out, temp_out,
        [nz, n_col, d_in, 0],
        [float(dt), float(C.P_REF), float(C.R_D), float(C.GAMMA_D)],
        [float(C.KAPPA_D), float(C.RV_OVER_RD), float(C.EPS_WATER), float(C.E_TRIPLE)],
        [float(C.T_FREEZE), float(C.T_ICE_ALL), float(C.T_LIQ_ALL), 0.0],
        threads=(n_grp * 32,),
        group_size=(32,),
    )
    return x_out, exner_out, temp_out


def run_crm_distill_closure_metal(
    lib: Any,
    raw_prof: torch.Tensor,
    raw_scal: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    exner: torch.Tensor,
    temp_sfc: torch.Tensor,
    thm_post_ls: torch.Tensor,
    qn_post_ls: torch.Tensor,
    qp_pre: torch.Tensor,
    qc_pre: torch.Tensor,
    qi_pre: torch.Tensor,
    u_pre: torch.Tensor,
    v_pre: torch.Tensor,
    shf: torch.Tensor,
    evap: torch.Tensor,
    tau_x: torch.Tensor,
    tau_y: torch.Tensor,
    dt: float,
) -> dict[str, torch.Tensor]:
    nz, n_col = rho_d.shape
    dev = rho_d.device
    dtype = rho_d.dtype
    sfc_5 = torch.stack([temp_sfc, shf, evap, tau_x, tau_y], dim=0).contiguous()
    out_prof = torch.empty((9, nz, n_col), device=dev, dtype=dtype)
    out_scal = torch.empty((3, n_col), device=dev, dtype=dtype)
    lib.crm_distill_closure_col(
        raw_prof.contiguous(), raw_scal.contiguous(),
        rho_d.contiguous(), dz_c.contiguous(), exner.contiguous(),
        thm_post_ls.contiguous(), qn_post_ls.contiguous(),
        qp_pre.contiguous(), qc_pre.contiguous(), qi_pre.contiguous(),
        u_pre.contiguous(), v_pre.contiguous(), sfc_5,
        out_prof, out_scal,
        [nz, n_col, 0, 0],
        [float(dt), float(C.CP_D), float(C.L_V0), float(C.L_F0)],
        [float(C.RV_OVER_RD), float(C.P_REF), float(C.T_FREEZE), float(C.EPS_WATER)],
        threads=n_col,
    )
    return {
        "theta_m": out_prof[0],
        "q_n": out_prof[1],
        "q_p": out_prof[2],
        "q_v": out_prof[3],
        "q_c": out_prof[4],
        "q_i": out_prof[5],
        "u": out_prof[6],
        "v": out_prof[7],
        "precip_rain": out_scal[0],
        "precip_snow": out_scal[1],
        "f_cld": out_prof[8],
        "f_prec": out_scal[2],
    }


def run_rad_distill_extract_metal(
    lib: Any,
    pres_c: torch.Tensor,
    temp_c: torch.Tensor,
    dz_c: torch.Tensor,
    rho_d_c: torch.Tensor,
    qv_c: torch.Tensor,
    qc_c: torch.Tensor,
    qi_c: torch.Tensor,
    t_sfc: torch.Tensor,
    alb_vd: torch.Tensor,
    alb_vf: torch.Tensor,
    alb_nd: torch.Tensor,
    alb_nf: torch.Tensor,
    emiss_sfc: torch.Tensor,
    mu0: torch.Tensor,
    s_toa: torch.Tensor,
    co2_f: float,
    ch4_f: float,
    n2o_f: float,
    o3_vmr_c: torch.Tensor | None = None,
    q_dust_c: torch.Tensor | None = None,
    tau_aero_sw_c: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nz, m_col = pres_c.shape
    dev = pres_c.device
    dtype = pres_c.dtype
    d_in = 9 * nz + 13
    dummy1 = torch.zeros(1, device=dev, dtype=dtype)
    has_o3 = 1 if o3_vmr_c is not None else 0
    has_dust = 1 if q_dust_c is not None else 0
    has_aero = 1 if tau_aero_sw_c is not None else 0
    flags = (has_aero << 2) | (has_dust << 1) | has_o3
    sfc_8 = torch.stack([t_sfc, alb_vd, alb_vf, alb_nd, alb_nf, emiss_sfc, mu0, s_toa], dim=0).contiguous()
    x_out = torch.empty((m_col, d_in), device=dev, dtype=dtype)
    cp_layer = torch.empty((nz, m_col), device=dev, dtype=dtype)
    s_inc = torch.empty((m_col,), device=dev, dtype=dtype)
    n_thr = ((m_col + 31) // 32) * 32
    lib.rad_distill_extract_col(
        pres_c.contiguous(), temp_c.contiguous(), dz_c.contiguous(),
        rho_d_c.contiguous(), qv_c.contiguous(), qc_c.contiguous(), qi_c.contiguous(),
        o3_vmr_c.contiguous() if o3_vmr_c is not None else dummy1,
        q_dust_c.contiguous() if q_dust_c is not None else dummy1,
        tau_aero_sw_c.contiguous() if tau_aero_sw_c is not None else dummy1,
        sfc_8, x_out, cp_layer, s_inc,
        [nz, m_col, d_in, flags],
        [float(C.CP_D), float(C.CP_V), float(co2_f), float(ch4_f)],
        [float(n2o_f), 0.0, 0.0, 0.0],
        threads=(n_thr,),
        group_size=(32,),
    )
    return x_out, cp_layer, s_inc


def run_rad_distill_closure_metal(
    lib: Any,
    raw_qrad: torch.Tensor,
    raw_scal: torch.Tensor,
    cp_layer: torch.Tensor,
    s_inc: torch.Tensor,
    alb_vd: torch.Tensor,
    alb_vf: torch.Tensor,
    alb_nd: torch.Tensor,
    alb_nf: torch.Tensor,
    emiss_sfc: torch.Tensor,
    t_sfc: torch.Tensor,
) -> dict[str, torch.Tensor]:
    nz, m_col = cp_layer.shape
    dev = cp_layer.device
    dtype = cp_layer.dtype
    ctx_7 = torch.stack([s_inc, alb_vd, alb_vf, alb_nd, alb_nf, emiss_sfc, t_sfc], dim=0).contiguous()
    q_rad_out = torch.empty((nz, m_col), device=dev, dtype=dtype)
    scal_out = torch.empty((7, m_col), device=dev, dtype=dtype)
    lib.rad_distill_closure_col(
        raw_qrad.contiguous(), raw_scal.contiguous(), cp_layer.contiguous(), ctx_7,
        q_rad_out, scal_out,
        [nz, m_col, 0, 0],
        [float(C.STEFAN_BOLTZMANN), float(120.0 / C.SECONDS_PER_DAY), 0.0, 0.0],
        threads=m_col,
    )
    return {
        "q_rad": q_rad_out,
        "sw1_dir": scal_out[0],
        "sw1_dif": scal_out[1],
        "sw2_dir": scal_out[2],
        "sw2_dif": scal_out[3],
        "lw_dn": scal_out[4],
        "olr": scal_out[5],
        "asr": scal_out[6],
    }
