#!/usr/bin/env python3
"""Unit and integration tests for A1 Map-Free Priority-Flood Voronoi River Routing."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.river_routing import init_river_routing_state, step_river_routing_voronoi
from aoba.surface import SURF_OCEAN


def test_priority_flood_acyclic_and_mass_conserving() -> None:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = AOBAConfig(glevel=3, nz=18, dt_host=720.0, crm_nx=1, crm_ny=1, ocean_mode="ogcm_3d", enable_river_routing=True)
    model = init_aoba_model(cfg, device=dev)
    assert model.river is not None
    river = model.river
    mesh = model.mesh

    # 1. Verify EVERY land cell reaches an ocean cell in finite hops (zero cycles / zero dead-end pits)
    ds = river.downstream_idx.cpu().tolist()
    st = mesh.surf_type.cpu().tolist()
    nc = mesh.n_cells
    max_hops = 0
    for i in range(nc):
        if st[i] == SURF_OCEAN:
            assert ds[i] == i
            continue
        cur = i
        hops = 0
        while st[cur] != SURF_OCEAN:
            nxt = ds[cur]
            assert nxt != cur, f"Self-loop on land cell {cur}"
            cur = nxt
            hops += 1
            assert hops <= nc, f"Cycle detected starting from land cell {i}"
        max_hops = max(max_hops, hops)
    assert max_hops > 0
    print(f"  [PASS] Priority-Flood network acyclic on G3 (max river chain = {max_hops} Voronoi hops)")

    # 2. Exact global liquid water conservation over 60 simulated days (pulsed runoff -> storage + ocean delivery)
    area = mesh.area_cell
    is_land = (mesh.surf_type != SURF_OCEAN).to(area.dtype)
    runoff = 2.0e-5 * is_land  # ~1.73 mm/day uniform land runoff
    dt = 3600.0
    total_in_kg = 0.0
    total_out_kg = 0.0
    area64 = area.detach().cpu().double()
    for step in range(60 * 24):
        r_now = runoff if step < 30 * 24 else torch.zeros_like(runoff)
        total_in_kg += float((r_now.detach().cpu().double() * area64).sum().item() * dt)
        mouth = step_river_routing_voronoi(river, r_now, area, dt)
        total_out_kg += float((mouth.detach().cpu().double() * area64).sum().item() * dt)

    stor_end_kg = float((river.storage_kg.detach().cpu().double() + river.storage_comp.detach().cpu().double()).sum().item())
    rel_err = abs((total_out_kg + stor_end_kg) - total_in_kg) / max(total_in_kg, 1.0)
    assert rel_err < 1.0e-6, f"River routing mass conservation error: {rel_err:.3e}"
    assert total_out_kg > 0.5 * total_in_kg, "Runoff did not reach ocean river mouths"
    print(f"  [PASS] 60-day Voronoi river routing conservation on {dev}: rel_err={rel_err:.2e}, ocean_delivery_frac={total_out_kg/total_in_kg:.3f}")

    # 3. Coupled step_aoba_model smoke test
    step_aoba_model(model)
    assert torch.isfinite(model.river.storage_kg).all() and torch.isfinite(model.river.mouth_flux_kgm2s).all()
    print(f"  [PASS] Coupled step_aoba_model with enable_river_routing=True on {dev}")


if __name__ == "__main__":
    print("Running A1 Priority-Flood Voronoi River Routing tests...")
    test_priority_flood_acyclic_and_mass_conserving()
    print("ALL A1 RIVER ROUTING TESTS PASSED!")
