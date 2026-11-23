"""Verification tests for AOBA's 2-moment cloud microphysics, natural CCN/INP activation,
prognostic effective radii, and direct optical coupling to CKD radiation.
"""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.microphysics import (
    compute_natural_ccn_inp,
    diagnose_cloud_effective_radii,
    step_microphysics_two_moment,
)
from aoba.thermodynamics import rcemip_analytic_sounding
from aoba.vertical_grid import build_vertical_grid, compute_stf_heights


def test_2m_water_conservation_and_albrecht_twomey_physics() -> None:
    """Verify exact column water conservation and marine vs continental/dusty droplet physics."""
    vgrid = build_vertical_grid(nz=32, device="cpu")
    z_w, z_c, dz_c = compute_stf_heights(vgrid, torch.zeros(2))
    snd = rcemip_analytic_sounding(z_c, z_w, sst=300.0)

    rho_d = snd["rho_d_c"].clone()
    theta_m = snd["theta_m_c"].clone()
    qv0 = snd["qv_c"].clone()
    qn = qv0.clone()
    qp = torch.zeros_like(qn)
    # Inject identical warm cloud liquid anomaly into col 0 (clean ocean) and col 1 (dusty land)
    qn[6:10, :] = qn[6:10, :] + 1.8e-3

    is_ocean = torch.tensor([True, False])
    u_sfc = torch.tensor([4.0, 4.0])
    q_dust = torch.zeros_like(qn)
    q_dust[:, 1] = 2.0e-7  # dusty continental air mass
    w_eff = torch.full_like(qn, 0.8)

    nc = torch.zeros_like(qn)
    ni = torch.zeros_like(qn)
    init_water = (rho_d * (qn + qp) * dz_c).sum(dim=0)
    tot_precip = torch.zeros(2)
    qv_cur = qv0.clone()

    for _ in range(15):
        res = step_microphysics_two_moment(
            rho_d, theta_m, qn, qp, qv_cur, nc, ni, dz_c, dt=30.0,
            w_eff=w_eff, is_ocean=is_ocean, u_sfc=u_sfc, q_dust=q_dust,
        )
        theta_m, qn, qp, qv_cur = res.theta_m_c, res.q_n, res.q_p, res.q_v
        nc, ni = res.n_c, res.n_i
        tot_precip = tot_precip + (res.precip_rain + res.precip_snow) * 30.0

    final_water = (rho_d * (qn + qp) * dz_c).sum(dim=0) + tot_precip
    rel_err = torch.abs(final_water - init_water) / init_water
    assert float(rel_err.max().item()) < 1.0e-5, f"2M water conservation err={rel_err}"

    # Twomey + Albrecht check: dusty continental col 1 has higher N_c, smaller r_e, less warm rain
    re_l, re_i = diagnose_cloud_effective_radii(rho_d, res.q_c, res.q_i, nc, ni)
    assert float(nc[7, 1].item()) > float(nc[7, 0].item()) * 1.5
    assert float(re_l[7, 1].item()) < float(re_l[7, 0].item()) - 1.5
    assert float(tot_precip[0].item()) > float(tot_precip[1].item())


def test_natural_ccn_inp_dust_and_temperature_response() -> None:
    rho = torch.ones(4)
    temp = torch.tensor([260.0, 243.0, 238.0, 225.0])
    _, n_clean = compute_natural_ccn_inp(rho, temp, q_dust=torch.zeros(4))
    _, n_dusty = compute_natural_ccn_inp(rho, temp, q_dust=torch.full((4,), 2.0e-7))
    # Secondary-ice floor holds in warm mixed-phase (260 K); colder T increases INP monotonically
    assert float(n_clean[1]) > float(n_clean[0])
    assert float(n_clean[2]) > float(n_clean[1])
    # Dust enhances heterogeneous INP in mixed-phase zone (243 K)
    assert float(n_dusty[1]) > float(n_clean[1]) * 1.5


def test_coupled_all_three_pillars_cpu_and_mps() -> None:
    """Verify full coupled AOBA with CKD radiation + 3D OGCM + 2M microphysics on CPU & MPS."""
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devs:
        cfg = AOBAConfig(
            glevel=1, nz=16, z_top=36000.0, crm_nx=4, crm_ny=1,
            use_earth2014=False, rad_scheme="ckd", ocean_mode="ogcm_3d",
            ocean_nz=8, microphysics_scheme="two_moment", enable_nonoro_gwd=True,
        )
        m = init_aoba_model(cfg, device=dev)
        for _ in range(4):
            d = step_aoba_model(m)
        assert torch.isfinite(torch.tensor(d.global_mean_t_sfc_k))
        assert torch.isfinite(torch.tensor(d.global_mean_olr_wm2))
        assert m.crm.n_c is not None and m.crm.n_i is not None
        assert torch.isfinite(m.crm.n_c).all() and torch.isfinite(m.crm.n_i).all()
        assert m.ogcm is not None and torch.isfinite(m.ogcm.temp).all() and torch.isfinite(m.ogcm.sal).all()
