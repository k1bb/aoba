"""Exact 30-Class Köppen-Geiger Climate Classification (Beck et al., 2018, 2023, Table 1).

Diagnoses emergent global climate & biome zones solely from simulated monthly mean near-surface
air temperature `T_m` [degC] and monthly total precipitation `P_m` [mm month^-1], following the
strict priority order (`E` before `B`, then `A`, `C`, `D`) of Beck et al. (2018, Table 1).

References
----------
- Beck, Zimmermann, McVicar, Vergopolan, Berg, & Wood (2018, Scientific Data, 5, 180214, Table 1).
- Beck, McVicar, Vergopolan, Berg, Luthi, Dufour, Zeng, Jiang, van Dijk, & Miralles
  (2023, Scientific Data, 10, 724).

AOBA Formulation Reference: Sec. 12.2 (30-Class Köppen-Geiger Climate Classification).
"""

from __future__ import annotations

import torch


# Official Beck et al. (2018, Table 1) 1..30 integer legend
KOPPEN_NAMES: dict[int, str] = {
    1: "Af",  2: "Am",  3: "Aw",
    4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk",
    8: "Csa", 9: "Csb", 10: "Csc",
    11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc",
    17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd",
    21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd",
    25: "Dfa", 26: "Dfb", 27: "Dfc", 28: "Dfd",
    29: "ET",  30: "EF",
}


def classify_koppen_geiger(
    temp_monthly_c: torch.Tensor,
    precip_monthly_mm: torch.Tensor,
    lat_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify 12-month `[12, ...]` temperature [degC] & precipitation [mm] into `(code_30, major_5)`.

    Parameters
    ----------
    temp_monthly_c : torch.Tensor
        Monthly mean air temperature `[12, ...]` in degrees Celsius (`month 0 = Jan .. 11 = Dec`).
    precip_monthly_mm : torch.Tensor
        Monthly total precipitation `[12, ...]` in `mm month^-1` (`kg m^-2 month^-1`).
    lat_rad : torch.Tensor
        Latitude `[...]` in radians (`>= 0` Northern Hemisphere: summer = Apr-Sep `[3:9]`;
        `< 0` Southern Hemisphere: summer = Oct-Mar `[0,1,2,9,10,11]`).

    Returns
    -------
    code_30 : torch.Tensor
        `int64` tensor `[...]` in `1..30` matching `KOPPEN_NAMES` (Beck et al., 2018, Table 1).
    major_5 : torch.Tensor
        `int64` tensor `[...]` in `1..5` (`1=A Tropical, 2=B Arid, 3=C Temperate, 4=D Cold, 5=E Polar`).
    """
    assert temp_monthly_c.shape[0] == 12 and precip_monthly_mm.shape[0] == 12
    p_m = torch.clamp(precip_monthly_mm, min=0.0)
    t_m = temp_monthly_c

    t_ann = t_m.mean(dim=0)
    t_hot = t_m.max(dim=0).values
    t_cold = t_m.min(dim=0).values
    t_mon10 = (t_m > 10.0).sum(dim=0)

    p_ann = p_m.sum(dim=0)
    p_dry = p_m.min(dim=0).values

    # Hemisphere-aware high-sun summer (Apr-Sep for NH, Oct-Mar for SH) vs winter half-years
    nh = lat_rad >= 0.0
    apr_sep = p_m[3:9]                                     # [6, ...]
    oct_mar = torch.cat([p_m[0:3], p_m[9:12]], dim=0)      # [6, ...]

    nh_exp = nh.unsqueeze(0).expand_as(apr_sep)
    p_sum_6 = torch.where(nh_exp, apr_sep, oct_mar)
    p_win_6 = torch.where(nh_exp, oct_mar, apr_sep)

    p_sum_tot = p_sum_6.sum(dim=0)
    p_win_tot = p_win_6.sum(dim=0)

    p_sdry = p_sum_6.min(dim=0).values
    p_swet = p_sum_6.max(dim=0).values
    p_wdry = p_win_6.min(dim=0).values
    p_wwet = p_win_6.max(dim=0).values

    # Aridity threshold P_th (Beck et al., 2018, Table 1)
    win_70 = p_win_tot >= 0.70 * p_ann
    sum_70 = p_sum_tot >= 0.70 * p_ann
    p_th = torch.where(
        win_70,
        2.0 * t_ann,
        torch.where(sum_70, 2.0 * t_ann + 28.0, 2.0 * t_ann + 14.0),
    )
    # Note: when P_ann == 0, 0 < 10 * P_th can fail if T_ann < -14 C unless handled via P_ann <= 0 -> BWk
    is_arid_b = (p_ann < 10.0 * p_th) | (p_ann < 1.0e-3)

    # Priority 1: Polar E (T_hot < 10 degC — evaluated BEFORE B per Beck et al., 2018, Table 1!)
    is_e = t_hot < 10.0
    code_e = torch.where(t_hot < 0.0, 30, 29)  # 30=EF, 29=ET

    # Priority 2: Arid B (not E, and P_ann < 10 * P_th)
    is_b = (~is_e) & is_arid_b
    is_bw = (p_ann < 5.0 * p_th) | (p_ann < 1.0e-3)
    hot_b = t_ann >= 18.0
    code_b = torch.where(
        is_bw,
        torch.where(hot_b, 4, 5),   # 4=BWh, 5=BWk
        torch.where(hot_b, 6, 7),   # 6=BSh, 7=BSk
    )

    # Priority 3: Tropical A (not E, not B, and T_cold >= 18 degC)
    rest = (~is_e) & (~is_b)
    is_a = rest & (t_cold >= 18.0)
    is_af = p_dry >= 60.0
    is_am = (~is_af) & (p_dry >= (100.0 - p_ann / 25.0))
    code_a = torch.where(is_af, 1, torch.where(is_am, 2, 3))  # 1=Af, 2=Am, 3=Aw

    # Seasonal dryness sub-criteria s / w / f for C and D (Beck et al., 2018, Table 1):
    # When both s and w criteria match, compare summer vs winter total precipitation
    cond_s = (p_sdry < 40.0) & (p_sdry < (p_wwet / 3.0))
    cond_w = p_wdry < (p_swet / 10.0)
    use_s = cond_s & ((~cond_w) | (p_win_tot > p_sum_tot))
    use_w = cond_w & (~use_s)

    # Temperature summer sub-criteria a / b / c / d
    sub_a = t_hot >= 22.0
    sub_b = (~sub_a) & (t_mon10 >= 4)
    sub_d = (~sub_a) & (~sub_b) & (t_cold < -38.0)

    # Priority 4: Temperate C (rest & T_hot >= 10 & 0 < T_cold < 18)
    is_c = rest & (t_cold > 0.0) & (t_cold < 18.0)
    c_off = torch.where(sub_a, 0, torch.where(sub_b, 1, 2))  # 0=a, 1=b, 2=c
    code_c = torch.where(use_s, 8 + c_off, torch.where(use_w, 11 + c_off, 14 + c_off))

    # Priority 5: Cold / Continental D (rest & T_hot >= 10 & T_cold <= 0)
    d_off = torch.where(sub_a, 0, torch.where(sub_b, 1, torch.where(sub_d, 3, 2)))  # 0=a, 1=b, 2=c, 3=d
    code_d = torch.where(use_s, 17 + d_off, torch.where(use_w, 21 + d_off, 25 + d_off))

    code_30 = torch.where(
        is_e,
        code_e,
        torch.where(is_b, code_b, torch.where(is_a, code_a, torch.where(is_c, code_c, code_d))),
    ).to(torch.int64)

    major_5 = torch.where(
        is_e,
        5,
        torch.where(is_b, 2, torch.where(is_a, 1, torch.where(is_c, 3, 4))),
    ).to(torch.int64)

    return code_30, major_5
