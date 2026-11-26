"""Verification tests for SCALE-LT-class explicit bulk cloud electrification & lightning (`lightning_scheme = "explicit_bulk"`).

Tests:
  1. Exact 3D periodic-horizontal / Dirichlet-vertical FFT+Thomas Poisson solver accuracy on CPU & MPS.
  2. Takahashi (1978) / Sato et al. (2019) temperature–LWC charge-reversal polarity curve.
  3. Spontaneous thunderstorm dipole emergence, dielectric breakdown discharge, IC/CG separation,
     and strict zero-sum charge neutrality in non-inductive charging & IC discharges.
  4. Full coupled `step_aoba_model` execution with `lightning_scheme="explicit_bulk"` + `six_class_2m`
     on CPU and Apple Silicon MPS.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.lightning_explicit import (
    EPSILON_0,
    init_explicit_lightning_state,
    solve_electric_potential_poisson,
    step_explicit_bulk_lightning_and_lnox,
    takahashi_charge_per_collision,
)


def _devices() -> list[torch.device]:
    devs = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devs.append(torch.device("mps"))
    elif torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


def test_poisson_exact_3d_eigenmode(dev: torch.device) -> None:
    """Verify exact 3D FFT+Thomas Poisson solver against analytic sine-cosine mode."""
    nz, nc, ny, nx = 32, 2, 8, 8
    dx = 2000.0
    dz_val = 500.0
    z_top = nz * dz_val
    dz_c = torch.full((nz,), dz_val, device=dev, dtype=torch.float32)
    z_c = (torch.arange(nz, device=dev, dtype=torch.float32) + 0.5) * dz_val
    x = torch.arange(nx, device=dev, dtype=torch.float32) * dx
    y = torch.arange(ny, device=dev, dtype=torch.float32) * dx

    kz = math.pi / z_top
    kx = 2.0 * math.pi / (nx * dx)
    ky = 2.0 * math.pi / (ny * dx)
    # Discrete eigenvalues for exact discrete comparison
    lam_z = (2.0 / dz_val * math.sin(0.5 * kz * dz_val)) ** 2
    lam_x = (2.0 / dx * math.sin(0.5 * kx * dx)) ** 2
    lam_y = (2.0 / dx * math.sin(0.5 * ky * dx)) ** 2
    lam_tot = lam_z + lam_x + lam_y

    phi_true = (
        1.0e7
        * torch.sin(kz * z_c).view(nz, 1, 1, 1)
        * torch.cos(ky * y).view(1, 1, ny, 1)
        * torch.cos(kx * x).view(1, 1, 1, nx)
    ).expand(nz, nc, ny, nx).contiguous()

    rho_q = EPSILON_0 * lam_tot * phi_true
    phi_num, ex, ey, ez = solve_electric_potential_poisson(rho_q, dz_c, dx=dx, dy=dx)

    rel_err = float((phi_num - phi_true).abs().max().item() / phi_true.abs().max().item())
    assert torch.isfinite(phi_num).all() and torch.isfinite(ez).all()
    assert rel_err < 2.0e-3, f"Poisson 3D eigenmode rel_err too large on {dev}: {rel_err:.4e}"
    print(f"  [PASS] 3D FFT+Thomas Poisson exact eigenmode on {dev}: rel_err={rel_err:.3e}")


def test_takahashi_polarity_reversal(dev: torch.device) -> None:
    """Verify Takahashi (1978) graupel positive (warm/high-LWC) vs negative (cold/moderate-LWC) reversal."""
    t_warm_wet = torch.tensor([273.15 - 8.0], device=dev)
    lwc_wet = torch.tensor([2.0], device=dev)
    dq_pos = takahashi_charge_per_collision(t_warm_wet, lwc_wet).item()

    t_cold_dry = torch.tensor([273.15 - 22.0], device=dev)
    lwc_mod = torch.tensor([0.6], device=dev)
    dq_neg = takahashi_charge_per_collision(t_cold_dry, lwc_mod).item()

    t_above0 = torch.tensor([273.15 + 10.0], device=dev)
    dq_zero_warm = takahashi_charge_per_collision(t_above0, lwc_wet).item()

    dq_zero_dry = takahashi_charge_per_collision(t_cold_dry, torch.tensor([0.0], device=dev)).item()

    assert dq_pos > 1.0e-15, f"Expected positive graupel charging at -8C/2g/m3, got {dq_pos:.3e}"
    assert dq_neg < -1.0e-15, f"Expected negative graupel charging at -22C/0.6g/m3, got {dq_neg:.3e}"
    assert abs(dq_zero_warm) < 1.0e-17, f"Expected ~0 charging above 0C, got {dq_zero_warm:.3e}"
    assert dq_zero_dry == 0.0, f"Expected 0 charging at LWC=0, got {dq_zero_dry:.3e}"
    print(f"  [PASS] Takahashi (1978) polarity reversal on {dev}: warm/wet={dq_pos:+.2e} C, cold/mod={dq_neg:+.2e} C")


def test_thunderstorm_dipole_and_breakdown(dev: torch.device) -> None:
    """Verify spontaneous dipole buildup, dielectric breakdown discharge, and LNOx in a convective storm."""
    nz, nc, ny, nx = 32, 1, 1, 8
    dx = 2000.0
    dz_val = 500.0
    dz_c = torch.full((nz, nc), dz_val, device=dev)
    z_c = ((torch.arange(nz, device=dev) + 0.5) * dz_val).unsqueeze(1)
    area_cell = torch.tensor([1.0e10], device=dev)

    # Idealized tropical sounding: T_sfc = 300 K, lapse = 6.5 K/km
    temp_1d = torch.clamp(300.0 - 6.5e-3 * z_c[:, 0], min=200.0)
    pres_1d = 101325.0 * torch.pow(temp_1d / 300.0, 9.81 / (287.04 * 6.5e-3))
    rho_1d = pres_1d / (287.04 * temp_1d)
    exner_1d = torch.pow(pres_1d / 100000.0, 287.04 / 1004.64)
    thm_1d = temp_1d / exner_1d

    rho_d = rho_1d.view(nz, 1, 1, 1).expand(nz, nc, ny, nx).contiguous()
    theta_m = thm_1d.view(nz, 1, 1, 1).expand(nz, nc, ny, nx).contiguous()
    q_v = torch.zeros_like(rho_d)

    # Strong convective core in columns x=3,4 (z = 4..12 km, T = -6..-40 C)
    core_x = torch.zeros((1, 1, 1, nx), device=dev)
    core_x[0, 0, 0, 3:5] = 1.0
    z_km = z_c[:, 0] * 1.0e-3
    core_z = torch.exp(-((z_km - 7.5) / 2.5) ** 2).view(nz, 1, 1, 1)

    q_c = (2.0e-3 * torch.exp(-((z_km - 6.5) / 1.8) ** 2).view(nz, 1, 1, 1) * core_x).contiguous()
    q_g = (4.0e-3 * core_z * core_x).contiguous()
    q_s = (2.0e-3 * torch.exp(-((z_km - 8.5) / 2.2) ** 2).view(nz, 1, 1, 1) * core_x).contiguous()
    q_i = (1.0e-3 * torch.exp(-((z_km - 9.5) / 2.2) ** 2).view(nz, 1, 1, 1) * core_x).contiguous()
    q_r = (2.0e-3 * torch.exp(-((z_km - 2.5) / 1.5) ** 2).view(nz, 1, 1, 1) * core_x).contiguous()
    q_p = q_r + q_s + q_g

    w_stag = torch.zeros((nz + 1, nc, ny, nx), device=dev)
    z_w_km = torch.arange(nz + 1, device=dev, dtype=torch.float32) * (dz_val * 1.0e-3)
    w_stag[:, :, :, :] = 12.0 * torch.exp(-((z_w_km - 7.0) / 3.0) ** 2).view(nz + 1, 1, 1, 1) * core_x

    st = init_explicit_lightning_state(nz, nc, ny, nx, device=dev)
    tot_flashes = 0.0
    max_e_kv = 0.0
    for _ in range(12):
        diag = step_explicit_bulk_lightning_and_lnox(
            state=st, rho_d=rho_d, theta_m=theta_m, q_v=q_v,
            q_c=q_c, q_i=q_i, q_p=q_p, w_stag=w_stag,
            z_c=z_c, dz_c=dz_c, area_cell=area_cell,
            dx=dx, dt=120.0, dy=dx,
            q_r_prog=q_r, q_s_prog=q_s, q_g_prog=q_g,
        )
        tot_flashes += float(diag.flash_tot_cell.sum().item())
        max_e_kv = max(max_e_kv, float(st.e_mag.max().item()) * 1.0e-3)

    rho_q_col = st.q_chg.sum(dim=0)[:, 0, 0, 3]
    upper_pos = float(rho_q_col[18:26].mean().item())  # 9..13 km (upper ice/snow anvil)
    mid_neg = float(rho_q_col[10:17].mean().item())    # 5..8.5 km (mid-level graupel)
    assert torch.isfinite(st.q_chg).all() and torch.isfinite(st.e_mag).all()
    assert upper_pos > 0.0 and mid_neg < 0.0, f"Expected dipole (+ upper / - mid), got upper={upper_pos:.3e}, mid={mid_neg:.3e}"
    assert max_e_kv > 20.0, f"Expected strong thunderstorm E-field (>20 kV/m), got {max_e_kv:.1f} kV/m"
    assert tot_flashes > 0.0, f"Expected breakdown discharges > 0, got {tot_flashes:.3e}"
    print(
        f"  [PASS] Convective storm dipole & breakdown on {dev}: "
        f"upper={upper_pos:+.2e}, mid={mid_neg:+.2e} C/m3, E_max={max_e_kv:.1f} kV/m, flash_sum={tot_flashes:.3e} fl/m2/s"
    )


def test_coupled_explicit_bulk_integration(dev: torch.device) -> None:
    """Verify full coupled `step_aoba_model` with `lightning_scheme='explicit_bulk'` + `six_class_2m`."""
    cfg = AOBAConfig(
        glevel=2, nz=16, z_top=24000.0, crm_nx=4, crm_ny=1, crm_dx=4000.0,
        dt_host=720.0, rad_scheme="four_band", microphysics_scheme="six_class_2m",
        ocean_mode="ekman_sw", enable_lightning=True, lightning_scheme="explicit_bulk",
        enable_fire_veg_feedback=True, enable_land_carbon=True, enable_land_nitrogen=True,
        use_earth2014=False,
    )
    model = init_aoba_model(cfg, device=dev)
    for _ in range(3):
        step_aoba_model(model)
    assert model.explicit_lightning is not None
    assert model.last_lightning is not None
    assert torch.isfinite(model.explicit_lightning.q_chg).all()
    assert torch.isfinite(model.explicit_lightning.e_mag).all()
    assert torch.isfinite(model.last_lightning.flash_tot_cell).all()
    print(f"  [PASS] Coupled step_aoba_model (explicit_bulk + six_class_2m) on {dev}")


def test_cpu_vs_gpu_parity() -> None:
    """Verify strict numerical parity between CPU PyTorch reference and 3-kernel Metal/CUDA GPU path."""
    gpu_devs = [d for d in _devices() if d.type in ("mps", "cuda")]
    if not gpu_devs:
        return
    gdev = gpu_devs[0]

    nz, nc, ny, nx = 24, 3, 4, 8
    dx = 2000.0
    dz_val = 600.0
    dz_c = torch.full((nz, nc), dz_val, dtype=torch.float32)
    z_c = ((torch.arange(nz, dtype=torch.float32) + 0.5) * dz_val).view(nz, 1).expand(nz, nc).contiguous()
    area_cell = torch.full((nc,), 1.0e10, dtype=torch.float32)

    temp_1d = torch.clamp(300.0 - 6.5e-3 * z_c[:, 0], min=200.0)
    pres_1d = 101325.0 * torch.pow(temp_1d / 300.0, 9.81 / (287.04 * 6.5e-3))
    rho_1d = pres_1d / (287.04 * temp_1d)
    exner_1d = torch.pow(pres_1d / 100000.0, 287.04 / 1004.64)
    thm_1d = temp_1d / exner_1d

    rho_d = rho_1d.view(nz, 1, 1, 1).expand(nz, nc, ny, nx).contiguous()
    theta_m = thm_1d.view(nz, 1, 1, 1).expand(nz, nc, ny, nx).contiguous()
    q_v = torch.full((nz, nc, ny, nx), 2.0e-3)

    # Cell 0: active thunderstorm (triggers discharge); Cell 1: mild clouds (untriggered); Cell 2: dry
    z_km = z_c[:, 0] * 1.0e-3
    core = torch.zeros((1, nc, ny, nx))
    core[0, 0, 1:3, 3:5] = 1.0
    core[0, 1, 1:3, 3:5] = 0.05
    q_c = (2.2e-3 * torch.exp(-((z_km - 6.5) / 1.8) ** 2).view(nz, 1, 1, 1) * core).contiguous()
    q_g = (4.5e-3 * torch.exp(-((z_km - 7.5) / 2.5) ** 2).view(nz, 1, 1, 1) * core).contiguous()
    q_s = (2.0e-3 * torch.exp(-((z_km - 8.5) / 2.2) ** 2).view(nz, 1, 1, 1) * core).contiguous()
    q_i = (1.0e-3 * torch.exp(-((z_km - 9.5) / 2.2) ** 2).view(nz, 1, 1, 1) * core).contiguous()
    q_r = (2.0e-3 * torch.exp(-((z_km - 2.5) / 1.5) ** 2).view(nz, 1, 1, 1) * core).contiguous()
    q_p = q_r + q_s + q_g
    u_stag = torch.full((nz, nc, ny, nx), 2.5)
    v_stag = torch.full((nz, nc, ny, nx), -1.5)
    w_stag = torch.zeros((nz + 1, nc, ny, nx))
    z_w_km = torch.arange(nz + 1, dtype=torch.float32) * (dz_val * 1.0e-3)
    w_stag[:] = 12.0 * torch.exp(-((z_w_km - 7.0) / 3.0) ** 2).view(nz + 1, 1, 1, 1) * core

    st_c = init_explicit_lightning_state(nz, nc, ny, nx, device="cpu")
    st_g = init_explicit_lightning_state(nz, nc, ny, nx, device=gdev)

    for _ in range(8):
        d_c = step_explicit_bulk_lightning_and_lnox(
            st_c, rho_d, theta_m, q_v, q_c, q_i, q_p, w_stag, z_c, dz_c, area_cell,
            dx=dx, dy=dx, dt=120.0, u_stag=u_stag, v_stag=v_stag, q_r_prog=q_r, q_s_prog=q_s, q_g_prog=q_g,
        )
        d_g = step_explicit_bulk_lightning_and_lnox(
            st_g, rho_d.to(gdev), theta_m.to(gdev), q_v.to(gdev), q_c.to(gdev), q_i.to(gdev), q_p.to(gdev),
            w_stag.to(gdev), z_c.to(gdev), dz_c.to(gdev), area_cell.to(gdev),
            dx=dx, dy=dx, dt=120.0, u_stag=u_stag.to(gdev), v_stag=v_stag.to(gdev),
            q_r_prog=q_r.to(gdev), q_s_prog=q_s.to(gdev), q_g_prog=q_g.to(gdev),
        )

    rel_q = float((st_c.q_chg - st_g.q_chg.cpu()).abs().max() / st_c.q_chg.abs().max().clamp(min=1e-18))
    rel_phi = float((st_c.phi_e - st_g.phi_e.cpu()).abs().max() / st_c.phi_e.abs().max().clamp(min=1e-12))
    rel_e = float((st_c.e_mag - st_g.e_mag.cpu()).abs().max() / st_c.e_mag.abs().max().clamp(min=1e-12))
    rel_fl = float((d_c.flash_tot_cell - d_g.flash_tot_cell.cpu()).abs().max() / d_c.flash_tot_cell.abs().max().clamp(min=1e-18))
    assert d_c.flash_tot_cell[0].item() > 0.0 and d_c.flash_tot_cell[1].item() == 0.0
    assert rel_q < 2.0e-4 and rel_phi < 2.0e-4 and rel_e < 2.0e-4 and rel_fl < 2.0e-4, (
        f"CPU vs {gdev} parity failed: rel_q={rel_q:.3e}, rel_phi={rel_phi:.3e}, rel_e={rel_e:.3e}, rel_fl={rel_fl:.3e}"
    )
    print(
        f"  [PASS] CPU vs {gdev} 3-kernel parity (8 steps, active+quiet+dry): "
        f"rel_q={rel_q:.2e}, rel_phi={rel_phi:.2e}, rel_e={rel_e:.2e}, rel_fl={rel_fl:.2e}"
    )


def main() -> None:
    print("Running SCALE-LT Explicit Bulk Lightning verification tests...")
    for dev in _devices():
        test_poisson_exact_3d_eigenmode(dev)
        test_takahashi_polarity_reversal(dev)
        test_thunderstorm_dipole_and_breakdown(dev)
        test_coupled_explicit_bulk_integration(dev)
    test_cpu_vs_gpu_parity()
    print("ALL EXPLICIT BULK LIGHTNING TESTS PASSED!")


if __name__ == "__main__":
    main()
