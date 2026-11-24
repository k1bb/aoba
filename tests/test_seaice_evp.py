#!/usr/bin/env python3
"""Unit and integration tests for A3 Voronoi Sea-Ice EVP Rheology & 5-Category ITD."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba import constants as C
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.seaice_evp import step_seaice_evp_itd_voronoi


def test_evp_itd_conservation_and_ridging() -> None:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = AOBAConfig(
        glevel=3, nz=18, dt_host=720.0, crm_nx=1, crm_ny=1,
        ocean_mode="ogcm_3d", enable_seaice_evp_itd=True,
    )
    model = init_aoba_model(cfg, device=dev)
    assert model.ogcm is not None and model.ogcm.seaice_evp is not None
    evp = model.ogcm.seaice_evp
    sfc, mesh, ogcm = model.sfc, model.mesh, model.ogcm

    area64 = mesh.area_cell.detach().cpu().double()
    m0_tot = float((sfc.m_seaice.detach().cpu().double() * area64).sum().item())
    assert m0_tot > 0.0

    # Apply strong convergent polar wind stress for 5 days -> verify exact ice mass conservation,
    # area bounded <= 1.0, internal compressive stress sigma_1 < 0, and finite velocity
    lat = mesh.lat_cell
    tau_x = -0.25 * torch.sin(2.0 * lat)
    tau_y = 0.20 * torch.sign(lat)  # Pushes sea ice poleward -> convergence & ridging!
    u_ocn = torch.zeros(mesh.n_edges, device=dev)
    eta = torch.zeros(mesh.n_cells, device=dev)
    for _ in range(5 * 24):
        step_seaice_evp_itd_voronoi(evp, sfc, mesh, tau_x, tau_y, u_ocn, eta, dt=3600.0)

    m1_tot = float((sfc.m_seaice.detach().cpu().double() * area64).sum().item())
    v_itd_mass = float(((evp.v_cat.sum(dim=0).detach().cpu().double() * C.RHO_SEAICE) * area64).sum().item())
    rel_err = abs(m1_tot - m0_tot) / max(m0_tot, 1.0)
    sync_err = abs(v_itd_mass - m1_tot) / max(m1_tot, 1.0)
    a_max = float(evp.a_cat.sum(dim=0).max().item())
    s1_min = float(evp.sigma_1.min().item())
    u_max = float(evp.u_ice_e.abs().max().item())

    assert rel_err < 1.0e-5, f"Sea-ice EVP mass conservation drift: {rel_err:.3e}"
    assert sync_err < 1.0e-5, f"ITD volume vs m_seaice mismatch: {sync_err:.3e}"
    assert a_max <= 1.0001, f"Total ice concentration exceeded 1.0: {a_max}"
    assert s1_min < -1.0, f"Expected compressive internal stress sigma_1 < 0, got {s1_min}"
    assert u_max > 1.0e-4 and u_max <= 0.61
    print(
        f"  [PASS] 5-day A3 EVP+ITD on {dev}: mass rel_err={rel_err:.2e}, max(A)={a_max:.3f}, "
        f"min(sigma_1)={s1_min:.1f} N/m, max|u_ice|={u_max:.3f} m/s"
    )

    # Coupled step_aoba_model smoke test
    step_aoba_model(model)
    assert torch.isfinite(evp.u_ice_e).all() and torch.isfinite(evp.v_cat).all()
    print(f"  [PASS] Coupled step_aoba_model with enable_seaice_evp_itd=True on {dev}")


if __name__ == "__main__":
    print("Running A3 Sea-Ice EVP Rheology & 5-Category ITD tests...")
    test_evp_itd_conservation_and_ridging()
    print("ALL A3 SEA-ICE EVP+ITD TESTS PASSED!")
