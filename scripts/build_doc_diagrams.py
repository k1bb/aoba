#!/usr/bin/env python3
"""Generate conceptual SVG diagrams for AOBA documentation (EN/JA)."""

from __future__ import annotations

from html import escape as _xml_esc
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parents[1] / "docs" / "assets"


def _esc(s: str) -> str:
    """Escape XML special characters safely without double-escaping existing entities."""
    return _xml_esc(s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"), quote=True)


def _svg_wrap(width: int, height: int, title: str, subtitle: str, body: str) -> str:
    t_e = _esc(title)
    s_e = _esc(subtitle)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{t_e}">
  <defs>
    <style>
      .bg {{ fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1.5; rx: 14; }}
      .title {{ font: 700 17px "IBM Plex Sans", "IBM Plex Sans JP", "Hiragino Sans", sans-serif; fill: #0f172a; }}
      .sub {{ font: 400 12px "IBM Plex Sans", "IBM Plex Sans JP", "Hiragino Sans", sans-serif; fill: #475569; }}
      .box-t {{ font: 700 13px "IBM Plex Sans", "IBM Plex Sans JP", "Hiragino Sans", sans-serif; fill: #0f172a; }}
      .box-s {{ font: 400 11px "IBM Plex Sans", "IBM Plex Sans JP", "Hiragino Sans", sans-serif; fill: #334155; }}
      .mono {{ font: 600 11px "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; fill: #0f172a; }}
      .lbl {{ font: 600 11px "IBM Plex Sans", "IBM Plex Sans JP", "Hiragino Sans", sans-serif; fill: #1e293b; }}
      .badge {{ font: 700 10px "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; fill: #ffffff; }}
    </style>
    <marker id="arr" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#334155"/>
    </marker>
    <marker id="arr-blue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#0284c7"/>
    </marker>
    <marker id="arr-red" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626"/>
    </marker>
    <marker id="arr-green" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#15803d"/>
    </marker>
    <marker id="arr-amber" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#d97706"/>
    </marker>
  </defs>
  <rect class="bg" x="2" y="2" width="{width - 4}" height="{height - 4}" rx="14"/>
  <text class="title" x="24" y="32">{t_e}</text>
  <text class="sub" x="24" y="51">{s_e}</text>
  <line x1="24" y1="62" x2="{width - 24}" y2="62" stroke="#e2e8f0" stroke-width="1.2"/>
{body}
</svg>
'''


def _card(
    x: int, y: int, w: int, h: int, fill: str, stroke: str, title: str, lines: list[str], badge: str = "", badge_fill: str = "#0284c7"
) -> str:
    out = [f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>']
    if badge:
        bw = max(56, len(badge) * 7 + 14)
        out.append(f'  <rect x="{x + w - bw - 8}" y="{y + 7}" width="{bw}" height="17" rx="8" fill="{badge_fill}"/>')
        out.append(f'  <text class="badge" x="{x + w - bw / 2 - 8:.1f}" y="{y + 19}" text-anchor="middle">{_esc(badge)}</text>')
    out.append(f'  <text class="box-t" x="{x + 12}" y="{y + 22}">{_esc(title)}</text>')
    for i, ln in enumerate(lines):
        out.append(f'  <text class="box-s" x="{x + 12}" y="{y + 40 + i * 16}">{_esc(ln)}</text>')
    return "\n".join(out)


def build_fig01_overview() -> str:
    body = "\n".join([
        # Top: Global Host Atmosphere
        _card(24, 78, 812, 82, "#e0f2fe", "#0284c7", "1. Global Host Atmosphere (SCVT Icosahedral Voronoi C-Grid, Nonhydrostatic RK3-HEVI)", [
            "Synoptic Rossby waves, Hadley/Walker cells, stratospheric Brewer-Dobson circulation, Zalesak FCT tracer advection",
            "Orographic & non-orographic GWD  |  Prognostic O3, CO2 (3D), Dust, Sea-Salt, DMS/SO4, BC/OC, Volcanic H2SO4, H2^18O/HDO",
        ], "Sec. 1-3, 9", "#0369a1"),
        # Middle Left: Embedded CRM
        _card(24, 194, 260, 134, "#eff6ff", "#2563eb", "2. Embedded Cloud-Resolving Model", [
            "1:1 Superparameterization (2D/3D CRM)",
            "Cumulus-free explicit convection (crm_dx)",
            "Selectable Microphysics: 1M / 2M / 6-Class",
            "MYNN 2.5 PBL + 3D Smagorinsky SGS",
            "CRM Lightning Flash & LNOx Emission",
            "Optional: Standalone Global CRM mode",
        ], "Sec. 5, 6, 9", "#1d4ed8"),
        # Middle Center: Radiation & Chemistry
        _card(304, 194, 252, 134, "#fef9c3", "#ca8a04", "3. Radiation, Ozone & Aerosols", [
            "Selectable rad_scheme:",
            "  four_band | ckd | ckd_hd | rrtmgp",
            "Optional 3D TICA cloud-side scattering",
            "Prognostic 3D O3 (Chapman/PSC/dep)",
            "Natural Aerosols (Dust/SS/DMS/SOA)",
            "Volcanic SO2->H2SO4 & 11-yr Solar",
        ], "Sec. 4, 10", "#a16207"),
        # Middle Right: Carbon & Isotopes
        _card(576, 194, 260, 134, "#f3e8ff", "#7e22ce", "4. Closed Biogeochemical Loops", [
            "Prognostic CO2 (emissions-driven + 3D)",
            "Prognostic CH4 & N2O (OH/UV sinks)",
            "Silicate Weathering CO2 Thermostat",
            "Stable Water Isotopes (H2^18O, HDO)",
            "Milankovitch Orbit & Planetary Presets",
            "Aquaplanet / Snowball / TRAPPIST-1e",
        ], "Sec. 10, 11", "#6b21a8"),
        # Bottom Left: MAML Land + EEO
        _card(24, 362, 260, 132, "#dcfce7", "#15803d", "5. MAML Land, EEO Biosphere & C/N", [
            "1:1 Subgrid Land under each CRM col",
            "Emergent C3/C4 EEO Optimal Vegetation",
            "Subgrid Orography & Dynamic Wetlands",
            "5-Layer Freeze-Thaw Soil & Snowpack",
            "Prognostic C/N Pools & SPITFIRE Fire",
            "D8 River Routing -> Ocean Salinity/Nut",
        ], "Sec. 7, 8", "#15803d"),
        # Bottom Center: Ocean + Sea Ice + Waves
        _card(304, 362, 252, 132, "#cffafe", "#0891b2", "6. Ocean, Sea Ice, Waves & BGC", [
            "Selectable: 3D OGCM | 1.5-Layer Ekman",
            "Optional Fine Ocean Grid (ocean_glevel)",
            "EVP-ITD Sea Ice & SNICAR BC Albedo",
            "Waves Hs/Tp -> Langmuir KPP & Spray",
            "FlexPFT Marine BGC + O2 / OMZ / N2O",
            "Current-Relative Wind Air-Sea Flux",
        ], "Sec. 9", "#0e7490"),
        # Bottom Right: Dynamic Ice Sheet
        _card(576, 362, 260, 132, "#f1f5f9", "#475569", "7. Dynamic Ice Sheet & Isostasy", [
            "Glen Flow Law (n=3) SCVT SIA Solver",
            "EMA Surface Mass Balance (SMB)",
            "Optional Acceleration (10x - 100x)",
            "Bedrock Glacial Isostatic Adjustment",
            "Dynamic Land <-> IceSheet Mask Flip",
            "Conservative Barometric Vertical Remap",
        ], "Sec. 11", "#334155"),
        # Arrows connecting layers
        '  <line x1="154" y1="160" x2="154" y2="192" stroke="#0284c7" stroke-width="2" marker-end="url(#arr-blue)" marker-start="url(#arr-blue)"/>',
        '  <line x1="430" y1="160" x2="430" y2="192" stroke="#d97706" stroke-width="2" marker-end="url(#arr-amber)" marker-start="url(#arr-amber)"/>',
        '  <line x1="706" y1="160" x2="706" y2="192" stroke="#7e22ce" stroke-width="2" marker-end="url(#arr)" marker-start="url(#arr)"/>',
        '  <line x1="284" y1="260" x2="302" y2="260" stroke="#334155" stroke-width="1.8" marker-end="url(#arr)" marker-start="url(#arr)"/>',
        '  <line x1="556" y1="260" x2="574" y2="260" stroke="#334155" stroke-width="1.8" marker-end="url(#arr)" marker-start="url(#arr)"/>',
        '  <line x1="154" y1="328" x2="154" y2="360" stroke="#15803d" stroke-width="2" marker-end="url(#arr-green)" marker-start="url(#arr-green)"/>',
        '  <line x1="430" y1="328" x2="430" y2="360" stroke="#0284c7" stroke-width="2" marker-end="url(#arr-blue)" marker-start="url(#arr-blue)"/>',
        '  <line x1="284" y1="428" x2="302" y2="428" stroke="#0284c7" stroke-width="1.8" marker-end="url(#arr-blue)"/>',
        '  <text class="lbl" x="293" y="420" text-anchor="middle" font-size="9">River</text>',
        '  <line x1="556" y1="428" x2="574" y2="428" stroke="#475569" stroke-width="1.8" marker-end="url(#arr)" marker-start="url(#arr)"/>',
        '  <line x1="706" y1="328" x2="706" y2="360" stroke="#475569" stroke-width="1.8" marker-end="url(#arr)" marker-start="url(#arr)"/>',
        # Footer note
        '  <text class="sub" x="24" y="516">Design Principle: 100% map-free universal physics (only Earth2014 topography/bathymetry). Single float32 codebase: Laptop (MPS) to HPC (CUDA).</text>',
    ])
    return _svg_wrap(860, 532, "AOBA Earth System Architecture & Multi-Scale Coupling",
                     "How large-scale dynamics, cloud-resolving convection, radiation, land biosphere, ocean, cryosphere, and biogeochemistry couple without prescribed maps", body)


def build_fig02_grid() -> str:
    body = "\n".join([
        # Left panel: SCVT Voronoi Host + Embedded CRM
        '  <rect x="24" y="76" width="390" height="318" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4"/>',
        '  <text class="box-t" x="38" y="98">A. Horizontal SCVT Voronoi Cell &amp; Embedded CRM (MAML)</text>',
        # Hexagon Voronoi cell
        '  <polygon points="150,128 228,128 266,192 228,256 150,256 112,192" fill="#e0f2fe" stroke="#0284c7" stroke-width="2.2"/>',
        '  <circle cx="189" cy="192" r="4" fill="#0369a1"/>',
        '  <text class="mono" x="189" y="118" text-anchor="middle">Host SCVT Cell i (dx ~ 220 km at G5)</text>',
        '  <text class="lbl" x="274" y="196">Normal wind u_e</text>',
        '  <line x1="247" y1="192" x2="270" y2="192" stroke="#dc2626" stroke-width="2" marker-end="url(#arr-red)"/>',
        '  <text class="lbl" x="189" y="211" text-anchor="middle">Scalar center (rho, theta_m, q)</text>',
        # Zoom arrow into CRM curtain
        '  <line x1="189" y1="258" x2="189" y2="284" stroke="#2563eb" stroke-width="2" marker-end="url(#arr-blue)"/>',
        '  <text class="lbl" x="198" y="275">Embedded CRM (nx x ny cols, configurable crm_dx)</text>',
        # CRM columns + MAML 1:1 surface boxes
        '  <rect x="46" y="290" width="346" height="44" fill="#dbeafe" stroke="#2563eb" stroke-width="1.4"/>',
        *[f'  <line x1="{46 + i * 43.25:.1f}" y1="290" x2="{46 + i * 43.25:.1f}" y2="334" stroke="#60a5fa" stroke-width="1"/>' for i in range(1, 8)],
        '  <text class="mono" x="219" y="316" text-anchor="middle">CRM Columns (c = 1 .. nx*ny) — Explicit Updrafts &amp; Clouds</text>',
        '  <rect x="46" y="338" width="346" height="34" fill="#dcfce7" stroke="#15803d" stroke-width="1.4"/>',
        *[f'  <line x1="{46 + i * 43.25:.1f}" y1="338" x2="{46 + i * 43.25:.1f}" y2="372" stroke="#4ade80" stroke-width="1"/>' for i in range(1, 8)],
        '  <text class="mono" x="219" y="359" text-anchor="middle">1:1 MAML Subgrid Surface (Soil, Canopy, Snow, Lakes, Topo)</text>',

        # Right panel: Charney-Phillips vertical staggering + Klemp STF
        '  <rect x="434" y="76" width="402" height="318" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4"/>',
        '  <text class="box-t" x="448" y="98">B. Vertical Charney-Phillips Staggering &amp; Klemp STF</text>',
        # Sponge zone
        '  <rect x="456" y="116" width="220" height="52" fill="#fee2e2" stroke="#ef4444" stroke-dasharray="4 3"/>',
        '  <text class="lbl" x="566" y="138" text-anchor="middle">Rayleigh Sponge Layer (z &gt; z_sp)</text>',
        '  <text class="box-s" x="566" y="154" text-anchor="middle">48 km top (~0.85 hPa), absorbs gravity waves</text>',
        # Stretched interior levels
        '  <rect x="456" y="168" width="220" height="150" fill="#f8fafc" stroke="#94a3b8"/>',
        # Half levels (w) & full levels (rho, theta_m, q, u)
        '  <line x1="456" y1="200" x2="676" y2="200" stroke="#0284c7" stroke-width="1.6"/>',
        '  <line x1="456" y1="242" x2="676" y2="242" stroke="#0284c7" stroke-width="1.6"/>',
        '  <line x1="456" y1="284" x2="676" y2="284" stroke="#0284c7" stroke-width="1.6"/>',
        '  <circle cx="566" cy="221" r="4.5" fill="#15803d"/>',
        '  <circle cx="566" cy="263" r="4.5" fill="#15803d"/>',
        '  <circle cx="566" cy="301" r="4.5" fill="#15803d"/>',
        '  <rect x="561" y="196" width="10" height="8" fill="#dc2626"/>',
        '  <rect x="561" y="238" width="10" height="8" fill="#dc2626"/>',
        '  <rect x="561" y="280" width="10" height="8" fill="#dc2626"/>',
        # Terrain mountain wave at bottom
        '  <path d="M 456 352 Q 515 352 555 322 Q 595 292 635 338 L 676 352 L 676 376 L 456 376 Z" fill="#a8a29e" stroke="#57534e" stroke-width="1.5"/>',
        '  <text class="lbl" x="566" y="366" text-anchor="middle">Terrain z_topo(x) — STF smooths aloft</text>',
        # Legend on right of B
        '  <rect x="694" y="182" width="10" height="8" fill="#dc2626"/>',
        '  <text class="lbl" x="710" y="190">Half-level z_w (k+1/2)</text>',
        '  <text class="box-s" x="710" y="206">Vertical velocity w,</text>',
        '  <text class="box-s" x="710" y="220">radiative &amp; FCT fluxes</text>',
        '  <circle cx="699" cy="248" r="4.5" fill="#15803d"/>',
        '  <text class="lbl" x="710" y="252">Full-level z_c (k)</text>',
        '  <text class="box-s" x="710" y="268">rho_d, theta_m, p, T,</text>',
        '  <text class="box-s" x="710" y="282">q_v, q_c, q_i, u, v, q^2</text>',
        '  <text class="mono" x="694" y="316">dz_min = 60 m</text>',
        '  <text class="box-s" x="694" y="332">Geometric stretch</text>',
        '  <text class="box-s" x="694" y="346">up to z_top = 48 km</text>',
    ])
    return _svg_wrap(860, 412, "Horizontal SCVT-CRM Multi-Scale Grid & Vertical Charney-Phillips Staggering",
                     "How AOBA pairs an icosahedral Voronoi global grid with embedded cloud-resolving columns and terrain-following vertical levels", body)


def build_fig03_radiation() -> str:
    body = "\n".join([
        # Left: Physical schematic of SW/LW streams
        '  <rect x="24" y="76" width="330" height="334" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4"/>',
        '  <text class="box-t" x="38" y="98">A. Two-Stream Radiative Transfer Physics</text>',
        '  <rect x="44" y="114" width="290" height="34" rx="6" fill="#fef3c7" stroke="#f59e0b"/>',
        '  <text class="lbl" x="189" y="135" text-anchor="middle">Top of Atmosphere (TOA): S0 * mu0, OLR, ASR</text>',
        # Atmosphere layers
        '  <rect x="44" y="160" width="290" height="62" rx="6" fill="#e0e7ff" stroke="#6366f1"/>',
        '  <text class="lbl" x="189" y="180" text-anchor="middle">Stratosphere: O3 UV/9.6um, CO2 15um, Volcanic H2SO4</text>',
        '  <text class="box-s" x="189" y="198" text-anchor="middle">11-yr Solar Cycle TSI/UV + Pinatubo/SAI Aerosols</text>',
        '  <rect x="44" y="232" width="290" height="92" rx="6" fill="#e0f2fe" stroke="#0284c7"/>',
        '  <text class="lbl" x="189" y="252" text-anchor="middle">Troposphere: H2O, CO2, CH4, N2O, Clouds, Aerosols</text>',
        '  <text class="box-s" x="189" y="270" text-anchor="middle">Liquid &amp; Ice Cloud Optics (r_e, chi_SW inhomogeneity)</text>',
        '  <text class="box-s" x="189" y="286" text-anchor="middle">5-Species Aerosols (Dust, SeaSalt, SO4, BC, OC/SOA)</text>',
        '  <text class="box-s" x="189" y="302" text-anchor="middle">Optional 3D TICA: horizontal tilted-beam cloud shift</text>',
        '  <rect x="44" y="336" width="290" height="58" rx="6" fill="#dcfce7" stroke="#15803d"/>',
        '  <text class="lbl" x="189" y="356" text-anchor="middle">Surface: Land (VIS/NIR Canopy) / Ocean / Snow-Ice</text>',
        '  <text class="box-s" x="189" y="374" text-anchor="middle">SNICAR BC/Dust Snow Albedo + PAR -> EEO Plants</text>',
        # SW down / LW up arrows
        '  <line x1="76" y1="148" x2="76" y2="334" stroke="#d97706" stroke-width="2.4" marker-end="url(#arr-amber)"/>',
        '  <text class="lbl" x="62" y="236" fill="#b45309">SW</text>',
        '  <line x1="302" y1="334" x2="302" y2="148" stroke="#dc2626" stroke-width="2.4" marker-end="url(#arr-red)"/>',
        '  <text class="lbl" x="308" y="236" fill="#b91c1c">LW</text>',

        # Right: 4 Selectable Schemes Ladder
        '  <rect x="374" y="76" width="462" height="334" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4"/>',
        '  <text class="box-t" x="388" y="98">B. Selectable Radiation Schemes (forcing.rad_scheme)</text>',
        _card(390, 112, 430, 62, "#f1f5f9", "#64748b", '1. "four_band" — Fast Analytical 4-Band (2 SW + 2 LW)', [
            "Zero lookup tables; ideal for teaching, aquaplanets, & rapid prototyping",
            "Gases: H2O, CO2, O3 (UV/9.6um)  |  Fastest throughput",
        ], "FAST", "#475569"),
        _card(390, 184, 430, 62, "#e0f2fe", "#0284c7", '2. "ckd" — 16-Band / 50 g-pt Correlated-k (Default)', [
            "Self-contained deterministic k-distributions; exact H2O/CO2/CH4/N2O/O3",
            "Fused Metal/CUDA kernel; balanced accuracy & multi-decadal speed",
        ], "DEFAULT", "#0284c7"),
        _card(390, 256, 430, 62, "#fef3c7", "#d97706", '3. "ckd_hd" — 29-Band / 111 g-pt MSTRN11-Class CKD', [
            "Chou LW cloud scattering + 5-species aerosols + fine H2O/CO2 overlap",
            "Fused GPU kernel (15x speedup); zero external files required",
        ], "HIGH-RES", "#d97706"),
        _card(390, 328, 430, 66, "#ede9fe", "#7c3aed", '4. "rrtmgp" — Official HITRAN / MT_CKD 4.1 (240 g-pts)', [
            "Auto-fetches BSD-3 RRTMGP tables (128 LW + 112 SW g-pts); 2-gas eta",
            "Benchmark CMIP-grade accuracy; fused GPU kernel (110-168x speedup)",
        ], "HITRAN", "#6d28d9"),
    ])
    return _svg_wrap(860, 426, "Atmospheric Radiation Transfer & Selectable Scheme Hierarchy",
                     "Choose one of four physically consistent radiation schemes via [forcing] rad_scheme, plus optional 3D TICA cloud scattering", body)


def build_fig04_microphysics() -> str:
    body = "\n".join([
        # Top row: 3 selectable scheme cards
        _card(24, 76, 256, 108, "#f1f5f9", "#64748b", '1. "one_moment" (SAM 1M)', [
            "Prognostic: q_n (total non-precip),",
            "  q_p (total precip). Fixed r_e.",
            "Diagnostic T-partitioning of",
            "  liquid/ice & rain/snow/graupel.",
            "Lowest memory & compute cost.",
        ], "1-MOMENT", "#475569"),
        _card(302, 76, 256, 108, "#e0f2fe", "#0284c7", '2. "two_moment" (Default KK00)', [
            "Adds prognostic N_c & N_i.",
            "Aerosol-aware CCN/INP activation",
            "  (Dust, SeaSalt, SO4, SOA).",
            "Khairoutdinov-Kogan autoconversion",
            "  & prognostic r_e -> Radiation.",
        ], "DEFAULT", "#0284c7"),
        _card(580, 76, 256, 108, "#ede9fe", "#7c3aed", '3. "six_class_2m" (Morrison)', [
            "Full 6-class 2-moment hydrometeors:",
            "  Cloud (q_c,N_c), Ice (q_i,N_i),",
            "  Rain (q_r,N_r), Snow (q_s,N_s),",
            "  Rimed Graupel (q_g,N_g).",
            "Fused GPU kernel + true q_g lightning.",
        ], "6-CLASS 2M", "#6d28d9"),

        # Bottom diagram: Microphysical process pathway
        '  <rect x="24" y="206" width="812" height="198" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4"/>',
        '  <text class="box-t" x="38" y="228">Hydrometeor Phase-Change, Precipitation, &amp; Electrification Pathways inside Explicit CRM Columns</text>',
        _card(44, 246, 150, 62, "#f0f9ff", "#0284c7", "Water Vapor (q_v)", [
            "Supersaturation adjustment",
            "Condensation / Deposition",
        ]),
        _card(244, 246, 175, 62, "#dbeafe", "#2563eb", "Cloud Water & Ice", [
            "q_c, N_c (liquid) | q_i, N_i (ice)",
            "Activated by CCN & Dust INP",
        ]),
        _card(470, 246, 170, 62, "#e0e7ff", "#4f46e5", "Rain, Snow, Graupel", [
            "q_r, q_s, q_g (+ N_r, N_s, N_g)",
            "Autoconv, Accretion, Riming",
        ]),
        _card(684, 246, 134, 62, "#fef3c7", "#d97706", "Surface & LNOx", [
            "Rain/Snow -> MAML",
            "Graupel-Ice -> LNOx",
        ]),
        '  <line x1="194" y1="277" x2="242" y2="277" stroke="#0284c7" stroke-width="2" marker-end="url(#arr-blue)" marker-start="url(#arr-blue)"/>',
        '  <line x1="419" y1="277" x2="468" y2="277" stroke="#2563eb" stroke-width="2" marker-end="url(#arr-blue)"/>',
        '  <line x1="640" y1="277" x2="682" y2="277" stroke="#d97706" stroke-width="2" marker-end="url(#arr-amber)"/>',
        '  <path d="M 555 308 L 555 352 L 119 352 L 119 310" fill="none" stroke="#64748b" stroke-width="1.6" stroke-dasharray="5 3" marker-end="url(#arr)"/>',
        '  <text class="lbl" x="337" y="346" text-anchor="middle">Sub-cloud Evaporation / Sublimation (Cold Pools &amp; Convective Downdrafts)</text>',
        '  <text class="box-s" x="44" y="384">Sedimentation: Subcycled CFL-safe / exact exponential vertical fall speed (v_r &gt; v_g &gt; v_s &gt; v_i) with exact water &amp; enthalpy conservation.</text>',
    ])
    return _svg_wrap(860, 422, "Cloud Microphysics Schemes (forcing.microphysics_scheme) & Hydrometeor Pathways",
                     "Select 1-moment, 2-moment aerosol-cloud, or full 6-class 2-moment microphysics inside cumulus-free explicit CRM columns", body)


def build_fig05_land() -> str:
    body = "\n".join([
        _card(24, 78, 258, 128, "#dcfce7", "#15803d", "1. Two-Leaf EEO Biosphere", [
            "Zero prescribed PFT maps!",
            "Optimal chi, Vcmax25, Jmax25, C3/C4",
            "Lambert-W0 emergent LAI & canopy h_c",
            "Evergreen vs Deciduous phenology",
            "Emits BVOC -> Secondary Organic Aero",
        ], "Sec. 7", "#15803d"),
        _card(301, 78, 258, 128, "#fef3c7", "#d97706", "2. 3-Source Canopy & MAML Subgrid", [
            "Sunlit leaf + Shaded leaf + Ground EB",
            "RSL Monin-Obukhov (De Ridder, 2010)",
            "Optional Subgrid Topo lapse/aspect",
            "Optional Dynamic Wetlands & Lakes",
            "Snow aging + SNICAR BC/Dust albedo",
        ], "Sec. 6, 8", "#b45309"),
        _card(578, 78, 258, 128, "#fee2e2", "#dc2626", "3. Terrestrial C/N & Wildfire", [
            "6 C pools (Leaf, Wood, Root, 3 Soil)",
            "Prognostic Mineral N & BNF",
            "N-limitation feedback on Vcmax/LAI",
            "SPITFIRE Fire Ignition (CRM Lightning)",
            "Fire emits CO2, BC/OC, CH4, NOx",
        ], "Sec. 8", "#b91c1c"),
        _card(24, 240, 396, 134, "#e0f2fe", "#0284c7", "4. 5-Layer Freeze-Thaw Soil Hydrology & Heat", [
            "Implicit tridiagonal Richards water + Clapp-Hornberger retention",
            "Dynamic root-zone water access linked to canopy height",
            "5-layer soil heat diffusion with latent-heat zero-curtain freeze/thaw",
            "Surface + subsurface runoff -> D8 River Routing (Manning velocity)",
            "Coastal freshwater & dissolved nutrient discharge -> Ocean OGCM",
        ], "Sec. 8", "#0369a1"),
        _card(440, 240, 396, 134, "#f3e8ff", "#7e22ce", "5. Land-Atmosphere Biogeochemical & Dust Coupling", [
            "Wetland inundation fraction f_wet -> CH4 & open-water evaporation",
            "Soil moisture + T + N -> Natural soil N2O emission",
            "Bare-soil wind friction u* -> Mineral dust uplift (to Radiation/INP/Fe)",
            "Land NBP = GPP - Ra - Rh - Fire -> Prognostic atmospheric CO2",
            "Runoff & T -> Silicate weathering long-term CO2 drawdown",
        ], "Sec. 8, 10", "#6b21a8"),
        '  <line x1="282" y1="142" x2="299" y2="142" stroke="#334155" stroke-width="1.8" marker-end="url(#arr)" marker-start="url(#arr)"/>',
        '  <line x1="559" y1="142" x2="576" y2="142" stroke="#334155" stroke-width="1.8" marker-end="url(#arr)" marker-start="url(#arr)"/>',
        '  <line x1="220" y1="206" x2="220" y2="238" stroke="#0284c7" stroke-width="2" marker-end="url(#arr-blue)" marker-start="url(#arr-blue)"/>',
        '  <line x1="638" y1="206" x2="638" y2="238" stroke="#7e22ce" stroke-width="2" marker-end="url(#arr)" marker-start="url(#arr)"/>',
        '  <line x1="420" y1="307" x2="438" y2="307" stroke="#334155" stroke-width="1.8" marker-end="url(#arr)"/>',
    ])
    return _svg_wrap(860, 394, "MAML Subgrid Land, Autonomous EEO Biosphere, Hydrology & Terrestrial C/N",
                     "Every embedded CRM column owns its own land patch where vegetation, soil water, snow, wetlands, fire, and carbon emerge organically", body)


def build_fig06_ocean() -> str:
    body = "\n".join([
        # Left: Ocean mode choice
        _card(24, 78, 250, 122, "#f1f5f9", "#64748b", 'Option A: "ekman_sw"', [
            "1.5-Layer Wind-Driven Ekman",
            "+ Shallow-Water Rossby/Kelvin",
            "Zero prescribed Q-fluxes.",
            "Fast conceptual & aquaplanet",
            "  climate experiments.",
        ], "1.5-LAYER", "#475569"),
        _card(24, 214, 250, 166, "#cffafe", "#0891b2", 'Option B: "ogcm_3d" (Default)', [
            "Full 3D Multi-Layer SCVT OGCM",
            "Earth2014 bathymetry, 18+ layers",
            "Zalesak FCT T/S + z* free surface",
            "Scale-aware GM90 + Redi + Submeso",
            "KPP mixed layer + tidal mixing",
            "Independent Fine Grid (ocean_glevel)",
            "Current-relative wind (u_a - u_o)",
        ], "DEFAULT", "#0e7490"),

        # Center: Surface Coupling & Cryosphere
        _card(298, 78, 260, 140, "#dbeafe", "#2563eb", "Surface Waves & Sea Spray", [
            "enable_wave_coupling:",
            "Prognostic Hs & peak period Tp",
            "Stokes drift -> Langmuir KPP boost",
            "Wave-age Charnock roughness z0",
            "High-wind sea-spray enthalpy flux",
            "  (Fairall/Andreas for cyclones)",
        ], "WAVES", "#1d4ed8"),
        _card(298, 232, 260, 148, "#f0f9ff", "#0284c7", "EVP-ITD Dynamic Sea Ice", [
            "enable_seaice_evp_itd:",
            "5-category thickness distribution",
            "Elastic-Viscous-Plastic rheology",
            "Fused GPU 60-subcycle EVP kernel",
            "Snow cover + SNICAR BC darkening",
            "Brine rejection -> deep convection",
        ], "SEA ICE", "#0369a1"),

        # Right: Marine BGC + O2/OMZ
        _card(582, 78, 254, 302, "#dcfce7", "#15803d", "FlexPFT Marine BGC & O2/OMZ", [
            "enable_ocean_bgc (+ enable_ocean_o2_omz):",
            "",
            "• 3D Prognostic Pools:",
            "  NO3, PO4, dFe, DIC, Alk, O2, POC",
            "• Emergent FlexPFT Phytoplankton:",
            "  Dynamic Chl:C & optimal size/light",
            "• Dust Fe & LNOx + River nutrients",
            "• Carbonate Solver (float64 pCO2):",
            "  Air-sea fgCO2 <-> Prognostic CO2",
            "• Dissolved O2 & Suboxic OMZ:",
            "  Garcia-Gordon solubility,",
            "  O2 < 20 mmol/m3 -> Denitrification",
            "  & Marine N2O emission",
            "• DMS production -> Sulfate CCN",
        ], "MARINE BGC", "#15803d"),

        '  <line x1="274" y1="268" x2="296" y2="148" stroke="#2563eb" stroke-width="1.8" marker-end="url(#arr-blue)" marker-start="url(#arr-blue)"/>',
        '  <line x1="274" y1="306" x2="296" y2="306" stroke="#0284c7" stroke-width="1.8" marker-end="url(#arr-blue)" marker-start="url(#arr-blue)"/>',
        '  <line x1="558" y1="148" x2="580" y2="148" stroke="#15803d" stroke-width="1.8" marker-end="url(#arr-green)"/>',
        '  <line x1="558" y1="306" x2="580" y2="306" stroke="#15803d" stroke-width="1.8" marker-end="url(#arr-green)" marker-start="url(#arr-green)"/>',
    ])
    return _svg_wrap(860, 400, "Ocean Circulation (forcing.ocean_mode), Waves, Sea Ice, & Marine Biogeochemistry",
                     "Choose 3D multi-layer OGCM (with optional fine ocean_glevel) or fast 1.5-layer Ekman ocean, coupled to waves, EVP-ITD sea ice, and FlexPFT BGC", body)


def build_fig07_biogeochem_planetary() -> str:
    body = "\n".join([
        _card(24, 78, 400, 148, "#fef3c7", "#d97706", "1. Closed Carbon, Methane, N2O & Ozone Cycles", [
            "co2_mode: constant | scenario | csv | prognostic (+ 3D FCT)",
            "  dCO2/dt = (E_anthro + E_volc - NBP_land - F_ocn - F_weath) / 2.124",
            "CH4 & N2O: Wetlands + Fire + Ocean OMZ <-> OH & Strat UV sinks",
            "o3_mode: rcemip | latitude_seasonal | climatology | prognostic",
            "  3D Chapman UV + Catalytic NOx/HOx + PSC + EEO dry deposition",
            "Volcanic SO2->H2SO4 (Bateman) & 11-yr Schwabe Solar Cycle",
        ], "Sec. 10", "#b45309"),
        _card(440, 78, 396, 148, "#e0f2fe", "#0284c7", "2. Stable Water Isotopes (H2^18O, HDO) & Ice Cores", [
            "enable_water_isotopes: 3D VSMOW-normalized q_v_18o, q_v_hdo",
            "Ocean evaporation: Craig-Gordon (1965) kinetic fractionation",
            "Condensation: Majoube (1971) & Merlivat-Nief (1967) equilibrium",
            "Ice supersaturation: Jouzel-Merlivat (1984) -> realistic d-excess",
            "Exact Rayleigh rainout distillation -> Snowpack d18O archive",
            "Direct proxy comparison with Greenland & Antarctic ice cores!",
        ], "Sec. 10.5", "#0369a1"),
        _card(24, 246, 400, 134, "#f1f5f9", "#475569", "3. Dynamic Ice Sheet & Milankovitch Paleoclimate", [
            "enable_dynamic_icesheet: Glen n=3 SIA + Bedrock Isostasy",
            "Optional 10x-100x ice-sheet acceleration + vertical grid remap",
            "enable_milankovitch_orbit: Berger (1978) / Laskar (2004)",
            "  continuous eccentricity, obliquity, & precession evolution",
            "enable_silicate_weathering: Walker (1981) geologic thermostat",
        ], "Sec. 11", "#334155"),
        _card(440, 246, 396, 134, "#f3e8ff", "#7e22ce", "4. Universal Planetary & Exoplanet Presets", [
            "planet_preset (100% map-free universal equations):",
            '  "earth"           : Modern / Paleo Earth (Earth2014)',
            '  "aquaplanet"      : 100% Ocean, zero topography (ITCZ/MJO)',
            '  "snowball_earth"  : Faint young Sun + glaciated + weathering',
            '  "tidally_locked" / "trappist1e": Fixed substellar eyeball world',
            '  "mars_like"       : Cold arid bare-land terrestrial planet',
        ], "Sec. 11.4", "#6b21a8"),
        '  <line x1="424" y1="152" x2="438" y2="152" stroke="#334155" stroke-width="1.6" marker-end="url(#arr)" marker-start="url(#arr)"/>',
        '  <line x1="224" y1="226" x2="224" y2="244" stroke="#334155" stroke-width="1.6" marker-end="url(#arr)" marker-start="url(#arr)"/>',
        '  <line x1="638" y1="226" x2="638" y2="244" stroke="#7e22ce" stroke-width="1.6" marker-end="url(#arr)" marker-start="url(#arr)"/>',
    ])
    return _svg_wrap(860, 400, "Biogeochemical Loops, Paleoclimate Water Isotopes, Ice Sheets & Planetary Regimes",
                     "From modern climate change to Ice-Age d18O cores, Snowball Earth, and tidally locked exoplanets within one equation set", body)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    figs = {
        "fig_01_aoba_overview.svg": build_fig01_overview(),
        "fig_02_grid_staggering.svg": build_fig02_grid(),
        "fig_03_radiation_schemes.svg": build_fig03_radiation(),
        "fig_04_microphysics_convection.svg": build_fig04_microphysics(),
        "fig_05_land_maml_eeo.svg": build_fig05_land(),
        "fig_06_ocean_ice_bgc.svg": build_fig06_ocean(),
        "fig_07_biogeochem_planetary.svg": build_fig07_biogeochem_planetary(),
    }
    for name, svg in figs.items():
        p = ASSETS_DIR / name
        p.write_text(svg, encoding="utf-8")
        print(f"Wrote {p} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
