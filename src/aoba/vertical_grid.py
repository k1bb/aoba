"""Charney-Phillips vertical grid and Klemp (2011) smoothed terrain-following coordinate.

References
----------
- Gassmann (2013, Q. J. R. Meteorol. Soc., 139, 152-175, Sec. 3):
  Charney-Phillips vertical staggering (w, theta_m at N_z + 1 interfaces;
  rho_d, p, u_h, q_n, q_p at N_z full-level centers).
- Klemp (2011, Mon. Wea. Rev., 139, 2163-2169, Eqs. 4-7):
  Smoothed Terrain-Following (STF) height coordinate z = zeta + A(zeta) * h_s(x, zeta)
  with progressive multi-pass Laplacian smoothing of coordinate surfaces aloft,
  anchored at the effective aerodynamic datum h(x) = z_topo(x) + d(x).
- Khairoutdinov & Randall (2003, J. Atmos. Sci., 60, Sec. 2) & Skamarock et al. (2012):
  Upper-domain Rayleigh wave-absorbing sponge profile above z_sponge.

AOBA Formulation Reference: Sec. 2.4, Eqs. (2.9)-(2.11) (Hybrid Stretched Charney-Phillips Levels, Klemp STF & Sponge).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class VerticalGrid:
    """Reference 1D stretched Charney-Phillips vertical coordinate (zeta in [0, z_top]).

    Attributes
    ----------
    nz : int
        Number of full vertical layers (default 32).
    z_top : float
        Domain top height [m] (default 24,000 m).
    zeta_w : torch.Tensor
        Nominal interface (half-level) heights [nz + 1] in [0, z_top] [m].
    zeta_c : torch.Tensor
        Nominal full-level center heights [nz] [m].
    dzeta_c : torch.Tensor
        Nominal layer thicknesses zeta_w[k+1] - zeta_w[k] [nz] [m].
    dzeta_w : torch.Tensor
        Nominal center-to-center spacings zeta_c[k] - zeta_c[k-1] [nz + 1] [m].
    terrain_atten_w : torch.Tensor
        Klemp (2011, Eq. 2/4) vertical terrain attenuation A(zeta_w) in [0, 1] [nz + 1].
    sponge_w : torch.Tensor
        Rayleigh damping rate tau_sponge(zeta_w)^-1 [nz + 1] [s^-1].
    sponge_c : torch.Tensor
        Rayleigh damping rate tau_sponge(zeta_c)^-1 [nz] [s^-1].
    """

    nz: int
    z_top: float
    zeta_w: torch.Tensor
    zeta_c: torch.Tensor
    dzeta_c: torch.Tensor
    dzeta_w: torch.Tensor
    terrain_atten_w: torch.Tensor
    sponge_w: torch.Tensor
    sponge_c: torch.Tensor
    crm_sponge_w: torch.Tensor
    crm_sponge_c: torch.Tensor

    def to(self, device: torch.device) -> "VerticalGrid":
        """Move grid tensors to the target compute device."""
        return VerticalGrid(
            nz=self.nz,
            z_top=self.z_top,
            zeta_w=self.zeta_w.to(device),
            zeta_c=self.zeta_c.to(device),
            dzeta_c=self.dzeta_c.to(device),
            dzeta_w=self.dzeta_w.to(device),
            terrain_atten_w=self.terrain_atten_w.to(device),
            sponge_w=self.sponge_w.to(device),
            sponge_c=self.sponge_c.to(device),
            crm_sponge_w=self.crm_sponge_w.to(device),
            crm_sponge_c=self.crm_sponge_c.to(device),
        )


# AOBA Formulation Eqs. (2.9), (2.11)
def build_vertical_grid(
    nz: int = 32,
    z_top: float = 24000.0,
    dz_min: float = 60.0,
    z_sponge_base: float = 17500.0,
    tau_sponge_min: float = 300.0,
    terrain_scale_exp: float = 1.5,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> VerticalGrid:
    """Construct smooth stretched N_z-layer Charney-Phillips vertical coordinate.

    Uses smooth geometric-hyperbolic stretching from exact lowest layer thickness
    `dz_min = 60 m` (so lowest full-level center sits at `tilde_z_1 = 30 m > h_c - d > z_0m`)
    through the boundary layer (~13 layers below 2 km), free troposphere (~14 layers across
    2-16 km), and 5 coarse lower-stratospheric sponge layers up to `z_top = 24 km`.

    References
    ----------
    - Klemp (2011, Mon. Wea. Rev., 139, Eq. 2 & Eq. 4): A(zeta) = (1 - zeta / z_top)^p.
    - Skamarock et al. (2012) & Klemp et al. (2008): sin^2 Rayleigh damping above z_sponge_base.
    """
    dev = torch.device(device)

    # Solve monotonic polynomial-geometric layer thickness profile dz_k = dz_min + c * s_k^q
    # so that dz[0] == dz_min exactly and sum_{k=0}^{nz-1} dz[k] == z_top exactly.
    idx = torch.arange(nz, dtype=torch.float64)
    s = idx / max(nz - 1, 1)
    # Blend gentle low-level stretching (s^1.35) with upper-troposphere/sponge stretching (s^4.2)
    shape = 0.38 * (s ** 1.35) + 0.62 * (s ** 4.2)
    rem = z_top - nz * dz_min
    if rem < 0.0:
        raise ValueError(f"z_top={z_top} must exceed nz * dz_min = {nz * dz_min}")
    scale = rem / float(shape.sum().item())
    dz = dz_min + scale * shape

    zeta_w64 = torch.zeros(nz + 1, dtype=torch.float64)
    zeta_w64[1:] = torch.cumsum(dz, dim=0)
    zeta_w64[-1] = z_top  # exact top enforcement
    zeta_c64 = 0.5 * (zeta_w64[:-1] + zeta_w64[1:])
    dzeta_c64 = zeta_w64[1:] - zeta_w64[:-1]

    dzeta_w64 = torch.empty(nz + 1, dtype=torch.float64)
    dzeta_w64[0] = zeta_c64[0] - zeta_w64[0]
    dzeta_w64[1:-1] = zeta_c64[1:] - zeta_c64[:-1]
    dzeta_w64[-1] = zeta_w64[-1] - zeta_c64[-1]

    # Klemp (2011, Eq. 2 & Eq. 4) hybrid attenuation A(zeta) = cos^2(pi/2 * zeta / z_atten_top)^p
    # satisfies A(0) = 1, A(z >= z_atten_top) = 0, dA/dzeta = 0 smoothly. Cap z_atten_top at 24 km
    # so high-top stratospheric/mesospheric coordinate surfaces (z >= 24 km) remain strictly horizontal.
    z_atten_top = min(float(z_top), 24000.0)
    xi_atten = torch.clamp(zeta_w64 / max(z_atten_top, 1.0), 0.0, 1.0)
    atten_w64 = torch.cos(0.5 * math.pi * xi_atten) ** (2.0 * terrain_scale_exp)
    atten_w64[zeta_w64 >= z_atten_top] = 0.0

    # Rayleigh wave-absorbing sponge (Klemp et al., 2008; Skamarock et al., 2012):
    # gamma_r(zeta) = (1 / tau_min) * sin^2(0.5 * pi * (zeta - z_s) / (z_top - z_s)) for zeta > z_s
    def _sponge(z: torch.Tensor) -> torch.Tensor:
        xi = torch.clamp((z - z_sponge_base) / max(z_top - z_sponge_base, 1.0), 0.0, 1.0)
        return (1.0 / tau_sponge_min) * torch.sin(0.5 * math.pi * xi) ** 2

    sponge_w64 = _sponge(zeta_w64)
    sponge_c64 = _sponge(zeta_c64)

    # Perturbation-only upper wave-absorbing layer for embedded periodic CRMs in high-top
    # configurations (z_top > 32 km; Klemp et al., 2008): absorbs trapped convective gravity waves
    # above 26 km before 1/sqrt(rho) mesospheric amplification while preserving host-mean fields.
    if z_top > 32000.0:
        z_crm_sp = min(z_sponge_base, 26000.0)
        xi_cw = torch.clamp((zeta_w64 - z_crm_sp) / max(z_top - z_crm_sp, 1.0), 0.0, 1.0)
        xi_cc = torch.clamp((zeta_c64 - z_crm_sp) / max(z_top - z_crm_sp, 1.0), 0.0, 1.0)
        crm_sponge_w64 = torch.maximum(sponge_w64, (1.0 / 20.0) * torch.sin(0.5 * math.pi * xi_cw) ** 2)
        crm_sponge_c64 = torch.maximum(sponge_c64, (1.0 / 90.0) * torch.sin(0.5 * math.pi * xi_cc) ** 2)
    else:
        crm_sponge_w64 = sponge_w64
        crm_sponge_c64 = sponge_c64

    return VerticalGrid(
        nz=nz,
        z_top=z_top,
        zeta_w=zeta_w64.to(device=dev, dtype=dtype),
        zeta_c=zeta_c64.to(device=dev, dtype=dtype),
        dzeta_c=dzeta_c64.to(device=dev, dtype=dtype),
        dzeta_w=dzeta_w64.to(device=dev, dtype=dtype),
        terrain_atten_w=atten_w64.to(device=dev, dtype=dtype),
        sponge_w=sponge_w64.to(device=dev, dtype=dtype),
        sponge_c=sponge_c64.to(device=dev, dtype=dtype),
        crm_sponge_w=crm_sponge_w64.to(device=dev, dtype=dtype),
        crm_sponge_c=crm_sponge_c64.to(device=dev, dtype=dtype),
    )


# AOBA Formulation Eq. (2.10)
def compute_stf_heights(
    vgrid: VerticalGrid,
    z_datum: torch.Tensor,
    smoothed_datum_levels: torch.Tensor | None = None,
    gamma_min: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute physical interface heights z_w, center heights z_c, and Jacobian dz_c.

    Implements Klemp (2011, Mon. Wea. Rev., 139, Eq. 4 & Eq. 7):
        z_w[k, ...] = zeta_w[k] + A(zeta_w[k]) * h_s[k, ...]
    anchored at the effective aerodynamic datum `z_datum = z_topo + d` at k = 0,
    enforcing minimum fractional grid spacing `gamma_min` (Klemp 2011, Eq. 7).

    Parameters
    ----------
    vgrid : VerticalGrid
        1D reference vertical coordinate.
    z_datum : torch.Tensor
        Effective lower aerodynamic datum `z_topo + d` [m] of shape `[...]`.
    smoothed_datum_levels : torch.Tensor | None
        Optional progressively Laplacian-smoothed terrain surfaces `h_s[nz + 1, ...]`
        (Klemp 2011, Eqs. 5-6). When omitted (e.g., single-column or uniform CRM domain),
        uses basic hybrid terrain-following `h_s[k] = z_datum`.
    gamma_min : float
        Minimum allowed ratio `(z_w[k] - z_w[k-1]) / dzeta_c[k-1]` (Klemp 2011, Eq. 7).

    Returns
    -------
    z_w : torch.Tensor
        Physical interface heights `[nz + 1, ...]` [m].
    z_c : torch.Tensor
        Physical layer-center heights `[nz, ...]` [m].
    dz_c : torch.Tensor
        Physical layer thicknesses (= metric Jacobian `sqrt(G)`) `[nz, ...]` [m].
    """
    nz = vgrid.nz
    col_shape = z_datum.shape
    expand_shape = (nz + 1,) + (1,) * len(col_shape)

    zeta_w = vgrid.zeta_w.view(expand_shape)
    atten_w = vgrid.terrain_atten_w.view(expand_shape)

    if smoothed_datum_levels is None:
        h_s = z_datum.unsqueeze(0).expand((nz + 1,) + col_shape)
    else:
        h_s = smoothed_datum_levels

    raw_z_w = zeta_w + atten_w * h_s

    # Enforce Klemp (2011, Eq. 7) positivity / minimum spacing constraint bottom-up
    z_levels: list[torch.Tensor] = [z_datum]
    for k in range(1, nz):
        min_zk = z_levels[k - 1] + gamma_min * vgrid.dzeta_c[k - 1]
        z_levels.append(torch.maximum(raw_z_w[k], min_zk))
    z_levels.append(torch.full_like(z_datum, vgrid.z_top))

    z_w = torch.stack(z_levels, dim=0)
    z_c = 0.5 * (z_w[:-1] + z_w[1:])
    dz_c = z_w[1:] - z_w[:-1]
    return z_w, z_c, dz_c
