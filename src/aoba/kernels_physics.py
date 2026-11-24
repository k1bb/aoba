"""Extra Metal shaders and wrappers for AOBA host/physics acceleration (SCVT FCT, Dust, GWD, Surface, Radiation)."""

from __future__ import annotations

from typing import Any

import torch


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
    uint ebase = b * (nz * ne) + k * ne;

    float div_fl_h = 0.0f;
    for (uint d = 0u; d < 6u; ++d) {
        div_fl_h += div_w[c6 + d] * fl_h[ebase + uint(eoc[c6 + d])];
    }

    uint az_base = b * (nz + 1u) * nc;
    if (k == 0u) {
        az_out[az_base + c] = 0.0f;
    }
    if (k + 1u == nz) {
        az_out[az_base + nz * nc + c] = 0.0f;
    }

    float q0 = q[gid];
    float fl_dn = 0.0f;
    float fl_up = 0.0f;
    if (k > 0u) {
        float w_dn = om_z[k * nc + c];
        float q_km1 = q[gid - nc];
        fl_dn = max(w_dn, 0.0f) * q_km1 + min(w_dn, 0.0f) * q0;
    }
    if (k + 1u < nz) {
        uint m = k + 1u;
        float w_up = om_z[m * nc + c];
        float q_kp1 = q[gid + nc];
        fl_up = max(w_up, 0.0f) * q0 + min(w_up, 0.0f) * q_kp1;
        float fh_up;
        if (nz >= 4u && m >= 2u && m <= nz - 2u) {
            float q_km1 = q[gid - nc];
            float q_kp2 = q[gid + 2u * nc];
            fh_up = w_up * ((-q_km1 + 7.0f * q0 + 7.0f * q_kp1 - q_kp2) * (1.0f / 12.0f));
        } else {
            fh_up = w_up * (0.5f * (q0 + q_kp1));
        }
        az_out[az_base + m * nc + c] = fh_up - fl_up;
    }

    float m_safe = max(m_new[loc], 1.0e-4f);
    q_td[gid] = max((m_old[loc] * q0 - dt * (div_fl_h + fl_up - fl_dn)) / m_safe, 0.0f);
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
        float al4 = 0.20f * j_4 * sig4;
        float b_cj4 = vc4 + al4;
        float acj4 = (b_cj4 - sqrt(max(b_cj4 * b_cj4 - 4.0f * 0.85f * vc4 * al4, 0.0f))) / (2.0f * 0.85f);
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
            float b_cj3 = ac3 + aj3;
            float gpp3 = (b_cj3 - sqrt(max(b_cj3 * b_cj3 - 4.0f * 0.98f * ac3 * aj3, 0.0f))) / (2.0f * 0.98f);
            float an3_pos = max(gpp3 - rd3, 0.0f);
            float gsw3_mol = 1.6f * (an3_pos * 1.0e-6f) / max(cs_mol * (1.0f - chi3), 1.0e-6f);

            float chi4 = clamp(xi4 / (xi4 + sqrt_ds), 0.05f, 0.92f);
            float cm4 = chi4 * cs_pa;
            float ap4 = (1.0f - PHI_LEAK_C4) * vp4 * cm4 / max(cm4 + kp4, 0.5f);
            float b_p4 = acj4 + ap4;
            float gpp4 = (b_p4 - sqrt(max(b_p4 * b_p4 - 4.0f * 0.85f * acj4 * ap4, 0.0f))) / (2.0f * 0.85f);
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
    device float* q_out             [[buffer(5)]],  // [4, nz, nc]
    device float* aod_out           [[buffer(6)]],  // [4, nc]
    constant ulong4& u_dim          [[buffer(7)]],  // [n_sp, nz, nc, dz_cols]
    constant float4& v_settle_4     [[buffer(8)]],  // [v_ss, v_so4, v_bc, v_oc]
    constant float4& v_dry_4        [[buffer(9)]],  // [vd_ss, vd_so4, vd_bc, vd_oc]
    constant float4& alpha_wet_4    [[buffer(10)]], // [aw_ss, aw_so4, aw_bc, aw_oc]
    constant float4& mee_4          [[buffer(11)]], // [mee_ss, mee_so4, mee_bc, mee_oc]
    constant float& dt              [[buffer(12)]],
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

    uint sp_base = s * nz * nc;
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
        float q_cur = max(q_in[idx3], 0.0f);
        if (k == 0u) {
            q_cur += (dt * emis) / mk;
        }
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
    constant float& rk_dt           [[buffer(11)]],
    uint e                          [[thread_position_in_grid]])
{
    if (e >= uint(ne_u)) return;
    float m = ocn_e0f[e];
    float pgf = g_inv_dc[e] * (eta_bt[uint(c1[e])] - eta_bt[uint(c0[e])]);
    float u_st = clamp((u0_bt[e] + rk_dt * (pgf + f_baro[e])) * m, -2.5f, 2.5f);
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
        float v0 = 2.2f * f_temp;

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
        float g_rate = (1.0f * f_temp) * (p2 / (p2 + 0.64f)) * z_pos;
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

        float f_remin = _aoba_neg_expm1_neg(min(0.05f * f_temp * dt_day, 20.0f));
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
        threads=B * nz * nc,
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

    dt1 = float(dt_bt) / 3.0
    dt2 = 0.5 * float(dt_bt)
    dt3 = float(dt_bt)

    for _ in range(n_bt):
        # Stage 1: (u_bt, eta_bt) -> (u_st, eta_st)
        lib.ogcm_bt_vel_edge(u_bt, eta_bt, f_b, g_i, hc_e, oe, c0_i32, c1_i32, u_st, flux_bt, ne, dt1, threads=ne)
        lib.ogcm_bt_eta_cell(eta_bt, flux_bt, io, eoc_i32, dw, eta_st, u_cell_dim, dt1, threads=nc)
        # Stage 2: (u_bt, eta_st) -> (u_st, eta_st)
        lib.ogcm_bt_vel_edge(u_bt, eta_st, f_b, g_i, hc_e, oe, c0_i32, c1_i32, u_st, flux_bt, ne, dt2, threads=ne)
        lib.ogcm_bt_eta_cell(eta_bt, flux_bt, io, eoc_i32, dw, eta_st, u_cell_dim, dt2, threads=nc)
        # Stage 3: (u_bt, eta_st) -> (u_st, eta_st), then commit to (u_bt, eta_bt)
        lib.ogcm_bt_vel_edge(u_bt, eta_st, f_b, g_i, hc_e, oe, c0_i32, c1_i32, u_st, flux_bt, ne, dt3, threads=ne)
        lib.ogcm_bt_eta_cell(eta_bt, flux_bt, io, eoc_i32, dw, eta_st, u_cell_dim, dt3, threads=nc)
        u_bt.copy_(u_st)
        eta_bt.copy_(eta_st)
        u_bt_acc.add_(u_bt)

    return u_bt.clone(), u_bt_acc * (1.0 / float(n_bt))


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
