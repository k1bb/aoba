"""Phase 1 Milestone 3 Tests: Zalesak 3D FCT, 3D Nonhydrostatic RK3-HEVI CRM, & Coupled RCE."""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba.crm import (
    build_discrete_hydrostatic_rcemip_crm,
    fct_advect_3d,
    step_coupled_crm_rce,
    step_crm_rk3_hevi,
)
from aoba.surface import SURF_LAND, SURF_OCEAN, init_surface_state
from aoba.vertical_grid import build_vertical_grid


def _available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def test_zalesak_fct_3d_positivity_and_conservation(device: str) -> None:
    """Verify exact mass conservation, strict non-negativity, and shape preservation of 3D FCT."""
    nz, ncol, ny, nx = 16, 2, 8, 8
    dx = dy = 4000.0
    dz_c = torch.linspace(80.0, 600.0, nz, device=device)
    dt = 10.0

    rho0 = torch.exp(-torch.linspace(0.0, 1.5, nz, device=device)).view(nz, 1, 1, 1).expand(nz, ncol, ny, nx).clone()
    # Non-uniform horizontal + vertical mass fluxes satisfying CFL < 0.4
    fx = 3.5 * torch.sin(torch.linspace(0.0, 6.28, nx, device=device)).view(1, 1, 1, nx).expand_as(rho0)
    fy = -2.5 * torch.cos(torch.linspace(0.0, 6.28, ny, device=device)).view(1, 1, ny, 1).expand_as(rho0)
    fz = torch.zeros((nz + 1, ncol, ny, nx), device=device)
    fz[1:-1] = 0.25 * torch.sin(torch.linspace(0.2, 2.9, nz - 1, device=device)).view(nz - 1, 1, 1, 1)

    div_f = (
        (fx - torch.roll(fx, 1, -1)) / dx
        + (fy - torch.roll(fy, 1, -2)) / dy
        + (fz[1:] - fz[:-1]) / dz_c.view(nz, 1, 1, 1)
    )
    rho1 = rho0 - dt * div_f
    assert (rho1 > 0.1).all()

    # Sharp discontinuous square cloud-water cube surrounded by strict zeros
    q = torch.zeros_like(rho0)
    q[3:8, :, 2:5, 2:5] = 4.0e-3
    m0 = (rho0 * q * dz_c.view(nz, 1, 1, 1)).sum(dim=(0, 2, 3))

    q_next = fct_advect_3d(q, rho0, rho1, fx, fy, fz, dx, dy, dz_c, dt)
    m1 = (rho1 * q_next * dz_c.view(nz, 1, 1, 1)).sum(dim=(0, 2, 3))

    assert (q_next >= 0.0).all()
    assert float(q_next.max().item()) <= 4.0e-3 + 1.0e-7
    assert torch.allclose(m1, m0, rtol=2.0e-5, atol=1.0e-7)


def test_crm_hydrostatic_rest_and_warm_bubble(device: str) -> None:
    """Verify resting hydrostatic balance, dry-mass conservation, and rising warm thermal bubble."""
    vgrid = build_vertical_grid(device=device, dtype=torch.float32)
    dz = vgrid.dzeta_c.view(vgrid.nz, 1, 1, 1)

    # 1. Resting hydrostatic atmosphere (zero perturbation)
    crm_rest = build_discrete_hydrostatic_rcemip_crm(vgrid, n_col=1, pert_amplitude_k=0.0)
    mass_0 = (crm_rest.rho_d * dz).sum()
    for _ in range(15):
        step_crm_rk3_hevi(crm_rest, vgrid, dt_dyn=10.0)
    mass_1 = (crm_rest.rho_d * dz).sum()

    assert torch.allclose(mass_1, mass_0, rtol=2.0e-5)
    # Spurious resting vertical velocity in float32 must remain tiny (< 0.02 m s^-1)
    assert float(crm_rest.w.abs().max().item()) < 0.02

    # 2. Warm buoyant bubble (+2.5 K centered near z ~ 1.5 km, iy = 4, ix = 4)
    crm_bub = build_discrete_hydrostatic_rcemip_crm(vgrid, n_col=1, pert_amplitude_k=0.0)
    y_idx = torch.arange(8, device=device, dtype=torch.float32).view(1, 1, 8, 1)
    x_idx = torch.arange(8, device=device, dtype=torch.float32).view(1, 1, 1, 8)
    z_col = vgrid.zeta_c.view(vgrid.nz, 1, 1, 1)
    r2 = ((x_idx - 3.5) / 1.8) ** 2 + ((y_idx - 3.5) / 1.8) ** 2 + ((z_col - 1500.0) / 900.0) ** 2
    bubble = 2.5 * torch.exp(-r2)
    crm_bub.theta_m = crm_bub.theta_m + bubble

    mass_b0 = (crm_bub.rho_d * dz).sum()
    Thm_b0 = (crm_bub.rho_d * crm_bub.theta_m * dz).sum()
    for _ in range(25):
        step_crm_rk3_hevi(crm_bub, vgrid, dt_dyn=8.0)
    mass_b1 = (crm_bub.rho_d * dz).sum()
    Thm_b1 = (crm_bub.rho_d * crm_bub.theta_m * dz).sum()

    assert torch.allclose(mass_b1, mass_b0, rtol=2.0e-5)
    assert torch.allclose(Thm_b1, Thm_b0, rtol=5.0e-5)
    # Bubble core must develop strong positive updraft > 0.5 m s^-1 while domain-mean w ~ 0
    assert float(crm_bub.w[:, 0, 3:5, 3:5].max().item()) > 0.5
    assert abs(float(crm_bub.w.mean().item())) < 0.02


def test_coupled_rce_ocean_and_vegetated_land(device: str) -> None:
    """Verify full coupled 3D CRM + Radiation + Surface + Two-Leaf Biosphere RCE integration."""
    vgrid = build_vertical_grid(device=device, dtype=torch.float32)
    crm = build_discrete_hydrostatic_rcemip_crm(vgrid, n_col=2, pert_amplitude_k=0.10)
    stypes = torch.tensor([SURF_OCEAN, SURF_LAND], device=device)
    sfc = init_surface_state(stypes, t_init=300.0)
    sfc.bio.lai[1] = 3.5
    sfc.bio.h_canopy[1] = 18.0

    mu0 = torch.full((2,), 0.7426, device=device)  # RCEMIP solar zenith cosine
    rad_cache: dict[str, torch.Tensor] = {}

    for step in range(12):
        diag = step_coupled_crm_rce(
            crm=crm,
            sfc=sfc,
            vgrid=vgrid,
            dt_crm=20.0,
            step_idx=step,
            cos_zenith=mu0,
            co2_ppmv=400.0,
            n_dyn_substeps=2,
            rad_stride=4,
            cached_rad_cache=rad_cache,
        )

    assert not torch.isnan(crm.theta_m).any()
    assert not torch.isnan(crm.q_n).any()
    assert (crm.q_n >= 0.0).all()
    assert (crm.q_p >= 0.0).all()
    assert (diag.olr > 150.0).all() and (diag.olr < 380.0).all()
    assert (diag.asr > 150.0).all()
    # Vegetated land column (col 1) must actively assimilate carbon via Two-Leaf EEO biosphere!
    assert float(diag.gpp[1].item()) > 2.0
    assert float(diag.gpp[0].item()) == 0.0


if __name__ == "__main__":
    for dev in _available_devices():
        print(f"Running Phase 1 Milestone 3 CRM/FCT/RCE tests on device={dev}...")
        test_zalesak_fct_3d_positivity_and_conservation(dev)
        test_crm_hydrostatic_rest_and_warm_bubble(dev)
        test_coupled_rce_ocean_and_vegetated_land(dev)
    print("ALL PHASE 1 MILESTONE 3 CRM, FCT & COUPLED RCE TESTS PASSED!")
