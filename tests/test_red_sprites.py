"""Verification suite for AOBA's Quasi-Electrostatic Red Sprite & Sprite-Halo Model (`src/aoba/red_sprites.py`).

Tests:
  1. Day vs. Night D-region ionospheric dielectric screening (Pasko et al., 1997): same 600 C km
     +CG flash triggers a bright crimson sprite at night (~70-82 km -> ~44-56 km) and is completely
     short-circuited during daytime.
  2. Luque & Gordillo-Vázquez (2012) sub-breakdown delayed sprite ignition via O- + N2 detachment
     (0.7 E_k <= E_peak < E_k at night where tau_sigma >= 5 ms).
  3. Optical quenching red-to-blue altitude shift (Pasko et al., 1997): N2 1st Positive (red)
     dominates at ~72 km while collisional N2 quenching shifts color toward N2 2nd Positive (blue)
     in the lower tendrils (~46 km).
  4. Positive vs. negative streamer stopping-altitude asymmetry (Ebert et al., 2010).
  5. Full global coupled `step_aoba_model` integration with `enable_red_sprites = True` on CPU & MPS.
"""

from __future__ import annotations

from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.red_sprites import compute_red_sprites, compute_sprite_vertical_profile


def _devices() -> list[torch.device]:
    devs = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devs.append(torch.device("mps"))
    elif torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


def test_day_night_screening_and_color_shift(dev: torch.device) -> None:
    """Verify nighttime red sprite ignition, daytime D-region suppression, and red-head/blue-tendril shift."""
    night = compute_sprite_vertical_profile(cmc_ckm=1100.0, mu0=-0.4, z_q_km=10.0, positive_cg=True, device=dev)
    day = compute_sprite_vertical_profile(cmc_ckm=1100.0, mu0=0.8, z_q_km=10.0, positive_cg=True, device=dev)

    assert bool(night["is_prompt_sprite"].item())
    assert bool(night["is_halo"].item())
    assert not bool(day["is_prompt_sprite"].item()) and not bool(day["is_delayed_sprite"].item())
    assert float(night["e_over_ek"].max().item()) > 4.0 * float(day["e_over_ek"].max().item())

    z_init = float(night["z_init_km"].item())
    z_stop = float(night["z_stop_km"].item())
    assert 68.0 <= z_init <= 86.0, f"Unexpected sprite initiation altitude: {z_init:.1f} km"
    assert 38.0 <= z_stop < z_init - 14.0, f"Expected deep downward streamer propagation: {z_init:.1f} -> {z_stop:.1f} km"

    # Optical quenching: red/blue ratio at upper head (~76 km) >> ratio in lower tendrils (~52 km)
    z = night["z_km"]
    k_head = int((z - 76.0).abs().argmin().item())
    k_tend = int((z - 52.0).abs().argmin().item())
    ratio_head = float(night["red_1pn2_kr"][k_head].item() / max(night["blue_2pn2_kr"][k_head].item(), 1e-9))
    ratio_tend = float(night["red_1pn2_kr"][k_tend].item() / max(night["blue_2pn2_kr"][k_tend].item(), 1e-9))
    assert float(night["red_1pn2_kr"][k_head].item()) > 100.0, "Expected bright >100 kR red sprite head"
    assert ratio_head > 1.5 * ratio_tend, f"Expected red->blue quenching shift: head={ratio_head:.2f}, tendril={ratio_tend:.2f}"

    # Positive vs negative streamer polarity asymmetry (Ebert et al., 2010)
    neg = compute_sprite_vertical_profile(cmc_ckm=1100.0, mu0=-0.4, z_q_km=10.0, positive_cg=False, device=dev)
    assert float(neg["z_stop_km"].item()) > z_stop, (
        f"Expected +CG streamer ({z_stop:.1f} km) to penetrate deeper than -CG ({float(neg['z_stop_km'].item()):.1f} km)"
    )

    print(
        f"  [PASS] Day/Night screening & red->blue quenching on {dev}: "
        f"night z={z_init:.0f}->{z_stop:.0f} km (red={float(night['red_1pn2_kr'][k_head].item()):.0f} kR), "
        f"day E/Ek_max={float(day['e_over_ek'].max().item()):.2f}"
    )


def test_luque_gv_delayed_detachment_regime(dev: torch.device) -> None:
    """Verify Luque & Gordillo-Vázquez (2012) sub-breakdown delayed sprite ignition (0.7 E_k <= E < E_k)."""
    found_delayed = False
    for cmc in (550.0, 620.0, 700.0, 760.0):
        prof = compute_sprite_vertical_profile(cmc_ckm=cmc, mu0=-0.35, z_q_km=10.0, positive_cg=True, device=dev)
        if bool(prof["is_delayed_sprite"].item()):
            found_delayed = True
            e_max = float(prof["e_over_ek"].max().item())
            assert 0.70 <= e_max < 1.0, f"Delayed sprite outside [0.7, 1.0) E_k: {e_max:.3f}"
            print(f"  [PASS] Luque & GV (2012) delayed detachment sprite on {dev}: CMC={cmc:.0f} C km, E/Ek={e_max:.2f}")
            break
    assert found_delayed, "Failed to trigger sub-breakdown delayed detachment sprite regime"


def test_global_batch_and_coupled_integration(dev: torch.device) -> None:
    """Verify vectorized global batch & full coupled `step_aoba_model(enable_red_sprites=True)`."""
    nz, nc = 32, 16
    z_c = torch.linspace(500.0, 56000.0, nz, device=dev).view(nz, 1).expand(nz, nc).contiguous()
    dz_c = torch.full((nz, nc), 1750.0, device=dev)
    rho_h = (1.225 * torch.exp(-z_c / 7200.0)).contiguous()
    thm_h = (300.0 * torch.exp(z_c / 24000.0)).contiguous()
    qv_h = torch.zeros_like(rho_h)
    qi_h = torch.full_like(rho_h, 2.0e-4)
    qp_h = torch.full_like(rho_h, 5.0e-4)
    area = torch.full((nc,), 3.0e13, device=dev)
    # Half night (mu0=-0.4), half day (mu0=+0.7) with identical strong CG lightning
    mu0 = torch.cat([torch.full((nc // 2,), -0.4, device=dev), torch.full((nc // 2,), 0.7, device=dev)])
    fl_cg = torch.full((nc,), 1.0e-10, device=dev)
    ccd = torch.full((nc,), 11.0, device=dev)

    sp = compute_red_sprites(
        flash_cg_cell=fl_cg, cold_cloud_km_cell=ccd, mu0_cell=mu0,
        rho_host=rho_h, theta_m_host=thm_h, q_v_host=qv_h, q_i_host=qi_h, q_p_host=qp_h,
        z_c_host=z_c, dz_c_host=dz_c, area_cell=area,
    )
    night_rate = float(sp.sprite_rate_cell[: nc // 2].mean().item())
    day_rate = float(sp.sprite_rate_cell[nc // 2 :].mean().item())
    assert night_rate > 100.0 * max(day_rate, 1e-25), f"Expected strong night dominance: night={night_rate:.2e}, day={day_rate:.2e}"
    assert float(sp.sprite_nox_mol_m2s[: nc // 2].mean().item()) > 0.0

    # Full coupled step_aoba_model with enable_red_sprites=True
    cfg = AOBAConfig(
        glevel=2, nz=16, z_top=48000.0, z_sponge_base=36000.0, crm_nx=4, crm_ny=1, crm_dx=4000.0,
        dt_host=720.0, rad_scheme="four_band", microphysics_scheme="six_class_2m",
        ocean_mode="ekman_sw", enable_lightning=True, lightning_scheme="explicit_bulk",
        enable_red_sprites=True, use_earth2014=False,
    )
    model = init_aoba_model(cfg, device=dev)
    for _ in range(3):
        step_aoba_model(model)
    assert model.last_red_sprites is not None
    assert torch.isfinite(model.last_red_sprites.sprite_rate_cell).all()
    assert torch.isfinite(model.last_red_sprites.red_1pn2_kr_cell).all()
    print(f"  [PASS] Global batch (night={night_rate:.2e} vs day={day_rate:.2e} m^-2 s^-1) & coupled step on {dev}")


def main() -> None:
    print("Running Quasi-Electrostatic Red Sprite & Sprite-Halo verification tests...")
    for dev in _devices():
        test_day_night_screening_and_color_shift(dev)
        test_luque_gv_delayed_detachment_regime(dev)
        test_global_batch_and_coupled_integration(dev)
    print("ALL RED SPRITE TESTS PASSED!")


if __name__ == "__main__":
    main()
