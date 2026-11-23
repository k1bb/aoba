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
"""


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

    key = (dev.index or 0, B, nz, nc, ne)
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
