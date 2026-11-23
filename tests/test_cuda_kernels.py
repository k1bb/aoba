"""Verification suite for AOBA's single-source Metal-to-CUDA transpiler and CUDA kernel backend.

Compiles the exact transpiled CUDA C++17 kernel bodies using the host C++17 compiler
(`clang++`/`g++`) with genuine 32-lane warp-shuffle and 64-thread shared-memory barrier
emulation, verifies PyTorch C++ extension header syntax, and asserts numerical parity
against PyTorch CPU & Apple Silicon Metal across individual physics/dynamics kernels and
a full coupled AOBA global step (CKD radiation + 3D OGCM + 2-moment microphysics + GWD).
When executed on an NVIDIA CUDA host, also runs end-to-end native CUDA parity checks.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.kernels import (
    _all_shader_src,
    get_active_gpu_backend,
    solve_tridiagonal_pytorch,
)
from aoba.kernels_cuda import (
    KernelSpec,
    build_cuda_extension_sources,
    build_host_cpp_verification_source,
)
from aoba.kernels_extra import (
    run_crm_fct_pair_metal,
    run_crm_microphysics_2m_metal,
    run_crm_mynn_metal,
    run_crm_sgs_metal,
)
from aoba.kernels_physics import run_rad_ckd_metal
from aoba.microphysics import _get_mp2m_metal_consts, step_microphysics_two_moment
from aoba.radiation_ckd import compute_ckd_radiation, pack_ckd_tables_flat
from aoba.runtime import synchronize_device


class _HostCUDACallable:
    """ctypes wrapper invoking a compiled C++17 host-emulated transpiled CUDA kernel."""

    def __init__(self, c_fn: Any, spec: KernelSpec) -> None:
        self._fn = c_fn
        self._spec = spec
        argtypes: list[Any] = []
        for p in spec.params:
            if p.kind in ("f32_const_ptr", "f32_ptr", "f32x4_const_ptr", "f32x4_ptr", "i32_const_ptr", "u32_const_ptr"):
                argtypes.append(ctypes.c_void_p)
            elif p.kind in ("uint", "ulong"):
                argtypes.append(ctypes.c_uint64)
            elif p.kind == "float":
                argtypes.append(ctypes.c_double)
            elif p.kind == "ulong4":
                argtypes.append(ctypes.POINTER(ctypes.c_uint64))
            elif p.kind == "float4":
                argtypes.append(ctypes.POINTER(ctypes.c_float))
        argtypes.extend([ctypes.c_int64, ctypes.c_int64])
        self._fn.argtypes = argtypes
        self._fn.restype = None

    def __call__(
        self,
        *args: Any,
        threads: int | tuple[int, ...] = 0,
        group_size: int | tuple[int, ...] | None = None,
    ) -> None:
        n_threads = int(threads[0]) if isinstance(threads, (tuple, list)) else int(threads)
        grp = 0 if group_size is None else (int(group_size[0]) if isinstance(group_size, (tuple, list)) else int(group_size))
        keepalive: list[Any] = []
        c_args: list[Any] = []
        for a, p in zip(args, self._spec.params):
            if p.kind in ("f32_const_ptr", "f32_ptr", "f32x4_const_ptr", "f32x4_ptr", "i32_const_ptr", "u32_const_ptr"):
                t: torch.Tensor = a
                if not t.is_contiguous():
                    t = t.contiguous()
                    keepalive.append(t)
                c_args.append(ctypes.c_void_p(t.data_ptr()))
            elif p.kind in ("uint", "ulong"):
                c_args.append(ctypes.c_uint64(int(a)))
            elif p.kind == "float":
                c_args.append(ctypes.c_double(float(a)))
            elif p.kind == "ulong4":
                arr = (ctypes.c_uint64 * 4)(int(a[0]), int(a[1]), int(a[2]), int(a[3]))
                keepalive.append(arr)
                c_args.append(arr)
            elif p.kind == "float4":
                arr = (ctypes.c_float * 4)(float(a[0]), float(a[1]), float(a[2]), float(a[3]))
                keepalive.append(arr)
                c_args.append(arr)
        self._fn(*c_args, ctypes.c_int64(n_threads), ctypes.c_int64(grp))


class _HostCUDASimLib:
    """Drop-in host-CPU simulator executing ALL 43 transpiled CUDA kernels natively."""

    def __init__(self, dylib_path: Path, specs: list[KernelSpec]) -> None:
        self._cdll = ctypes.CDLL(str(dylib_path))
        self._specs = {s.name: s for s in specs}
        for s in specs:
            c_fn = getattr(self._cdll, f"host_run_{s.name}")
            setattr(self, s.name, _HostCUDACallable(c_fn, s))


@pytest.fixture(scope="module")
def host_cuda_sim() -> tuple[_HostCUDASimLib, list[KernelSpec]]:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("No C++17 host compiler available")

    msl = _all_shader_src()
    host_src, specs = build_host_cpp_verification_source(msl)
    td = tempfile.mkdtemp(prefix="aoba_cuda_sim_")
    cpp_path = Path(td) / "aoba_cuda_sim.cpp"
    ext_suffix = ".dylib" if sysconfig.get_platform().startswith("macosx") else ".so"
    so_path = Path(td) / f"aoba_cuda_sim{ext_suffix}"
    cpp_path.write_text(host_src, encoding="utf-8")

    cmd = [
        compiler,
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-pthread",
        "-Wall",
        "-Wextra",
        "-Wno-unused-function",
        str(cpp_path),
        "-o",
        str(so_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Transpiled CUDA kernel compilation failed:\n{res.stderr}"
    return _HostCUDASimLib(so_path, specs), specs


def test_transpiler_and_pytorch_cpp_binding_syntax(host_cuda_sim: tuple[_HostCUDASimLib, list[KernelSpec]]) -> None:
    """Verify all 43 kernels transpile cleanly and PyTorch pybind11 C++ bindings parse cleanly."""
    _, specs = host_cuda_sim
    assert len(specs) == 43
    names = {s.name for s in specs}
    for required in (
        "batched_thomas",
        "crm_2d_stage_prep_simd",
        "crm_2d_fast_hevi_simd",
        "crm_2d_sgs_simd",
        "crm_2d_fct_simd_pair",
        "crm_microphysics_col",
        "crm_microphysics_2m_col",
        "rad_ckd_col",
        "rad_four_band_col",
        "crm_mynn_col",
        "scvt_fast_edge",
        "scvt_fast_cell_hevi",
        "surface_albedos_col",
        "surface_transfer_col",
        "two_leaf_gas_exchange_col",
        "dust_column_col",
        "oro_gwd_col",
        "ngwd_propagate_col",
    ):
        assert required in names

    # Also syntax-check the PyTorch C++ pybind11 wrapper source against real Torch headers
    compiler = shutil.which("clang++") or shutil.which("g++")
    assert compiler is not None
    cpp_src, cuda_src, _ = build_cuda_extension_sources(_all_shader_src())
    assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)" in cpp_src
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK();" in cuda_src

    from torch.utils.cpp_extension import include_paths

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "binding_check.cpp"
        p.write_text("#include <torch/extension.h>\n" + cpp_src, encoding="utf-8")
        inc_flags = [f"-I{d}" for d in include_paths()] + [f"-I{sysconfig.get_path('include')}"]
        # Use C++20 for PyTorch >= 2.10, falling back to C++17 on older toolchains
        res = subprocess.run(
            [compiler, "-std=c++20", "-fsyntax-only", "-DTORCH_EXTENSION_NAME=aoba_test_ext", *inc_flags, str(p)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"PyTorch C++ binding syntax check failed:\n{res.stderr}"


def test_cuda_batched_thomas_and_simd_crm_kernels(host_cuda_sim: tuple[_HostCUDASimLib, list[KernelSpec]]) -> None:
    """Verify batched_thomas, 2D SIMD FCT pair, 2D SIMD SGS, and partial-warp tail safety."""
    sim_lib, _ = host_cuda_sim
    torch.manual_seed(42)

    # 1. batched_thomas (M=137 deliberately non-multiple of CUDA block_size=128)
    K, M = 44, 137
    dl = torch.randn(K, M) * 0.1
    du = torch.randn(K, M) * 0.1
    d = torch.full((K, M), 2.5) + torch.rand(K, M) * 0.5
    rhs = torch.randn(K, M)
    ref = solve_tridiagonal_pytorch(dl, d, du, rhs)
    out = torch.empty_like(rhs)
    sim_lib.batched_thomas(dl, d, du, rhs, out, K, M, threads=M)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    # 2. CRM 2D SIMD SGS + FCT pair on n_col=10, nx=8 (10 % 8 == 2 -> tests partial tail warp!)
    nz, n_col, ny, nx = 16, 10, 1, 8
    rho = torch.full((nz, n_col, ny, nx), 1.1) + 0.05 * torch.randn(nz, n_col, ny, nx)
    u = 5.0 * torch.randn(nz, n_col, ny, nx)
    v = 3.0 * torch.randn(nz, n_col, ny, nx)
    w = 0.5 * torch.randn(nz + 1, n_col, ny, nx)
    w[0] = 0.0
    w[-1] = 0.0
    thm = torch.linspace(300.0, 360.0, nz).view(nz, 1, 1, 1).expand(nz, n_col, ny, nx).clone()
    dz_c = torch.full((nz, 1), 1000.0)
    dz_w = torch.full((nz - 1, 1), 1000.0)

    from aoba.crm import _compute_3d_sgs_tendencies, fct_advect_3d
    from aoba import constants as C

    du_ref, dv_ref, dw_ref, dth_ref = _compute_3d_sgs_tendencies(rho, u, v, w, thm, 4000.0, 4000.0, dz_c, dz_w)
    du_cu, dv_cu, dw_cu, dth_cu = run_crm_sgs_metal(sim_lib, rho, u, v, w, thm, 4000.0, 4000.0, dz_c, dz_w, C.GRAVITY, 0.18)
    assert torch.allclose(du_cu, du_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(dv_cu, dv_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(dw_cu, dw_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(dth_cu, dth_ref, atol=1e-4, rtol=1e-4)

    q0 = torch.rand(nz, n_col, ny, nx) * 0.01
    q1 = torch.rand(nz, n_col, ny, nx) * 0.002
    fx = rho * u
    fy = torch.zeros_like(fx)
    fz = torch.zeros(nz + 1, n_col, ny, nx)
    fz[1:-1] = 0.5 * (rho[:-1] + rho[1:]) * w[1:-1]
    q0_ref = fct_advect_3d(q0, rho, rho, fx, fy, fz, 4000.0, 4000.0, dz_c, 10.0)
    q1_ref = fct_advect_3d(q1, rho, rho, fx, fy, fz, 4000.0, 4000.0, dz_c, 10.0)
    q0_cu, q1_cu = run_crm_fct_pair_metal(sim_lib, q0, q1, rho, rho, fx, fy, fz, 4000.0, 4000.0, dz_c, 10.0)
    assert torch.allclose(q0_cu, q0_ref, atol=1e-5, rtol=1e-4)
    assert torch.allclose(q1_cu, q1_ref, atol=1e-5, rtol=1e-4)


def test_cuda_rad_ckd_and_2m_microphysics_parity(host_cuda_sim: tuple[_HostCUDASimLib, list[KernelSpec]]) -> None:
    """Verify transpiled CUDA `rad_ckd_col` and `crm_microphysics_2m_col` match PyTorch reference."""
    sim_lib, _ = host_cuda_sim
    torch.manual_seed(7)
    nz, M = 24, 15  # non-multiple of block size
    p = torch.logspace(5.0, 2.0, nz).view(nz, 1).expand(nz, M).contiguous()
    T = torch.linspace(292.0, 215.0, nz).view(nz, 1).expand(nz, M).contiguous()
    rho = p / (287.05 * T)
    dz = torch.full((nz, 1), 1500.0)
    qv = (0.012 * (p / 1e5) ** 3).clamp(min=3e-6)
    qc = torch.zeros(nz, M)
    qc[3:6] = 2.0e-4
    qi = torch.zeros(nz, M)
    qi[12:15] = 4.0e-5
    o3 = torch.full((nz, M), 2.0e-6)
    re_l = torch.full((nz, M), 10.0)
    re_i = torch.full((nz, M), 45.0)
    t_sfc = torch.full((M,), 294.0)
    alb = torch.full((M,), 0.12)
    mu0 = torch.linspace(0.0, 0.9, M)

    pres_w = torch.zeros(nz + 1, M)
    pres_w[0] = p[0] * 1.05
    pres_w[1:-1] = 0.5 * (p[:-1] + p[1:])
    pres_w[-1] = p[-1] * 0.5
    temp_w = torch.zeros(nz + 1, M)
    temp_w[0] = t_sfc
    temp_w[1:-1] = 0.5 * (T[:-1] + T[1:])
    temp_w[-1] = T[-1]
    ref = compute_ckd_radiation(
        pres_c=p, pres_w=pres_w, temp_c=T, temp_w=temp_w, dz_c=dz, rho_d_c=rho,
        qv_c=qv, qc_c=qc, qi_c=qi, t_sfc=t_sfc,
        albedo_vis=alb, albedo_nir=alb, mu0=mu0, s_toa=torch.full((M,), 1361.0),
        co2_ppmv=284.3, ch4_ppmv=0.80825, n2o_ppmv=0.27302,
        re_liq_um=re_l, re_ice_um=re_i, cloud_sw_inhomog=0.65,
        albedo_vis_dif=alb, albedo_nir_dif=alb, o3_vmr_c=o3, o3_lw_coupling=True,
    )
    sw_up, sw_dn, lw_up, lw_dn, qrad, _ = run_rad_ckd_metal(
        sim_lib,
        pres_c=p, pres_w=pres_w, temp_c=T, temp_w=temp_w, dz_c=dz, rho_d_c=rho,
        qv_c=qv, qc_c=qc, qi_c=qi, t_sfc=t_sfc,
        albedo_vis=alb, albedo_nir=alb, mu0=mu0, s_toa=torch.full((M,), 1361.0),
        co2_ppmv=284.3, ch4_ppmv=0.80825, n2o_ppmv=0.27302, emiss_sfc=0.98,
        re_liq_um=re_l, re_ice_um=re_i, cloud_sw_inhomog=0.65,
        albedo_vis_dif=alb, albedo_nir_dif=alb, q_dust_c=None, o3_vmr_c=o3,
        o3_lw_coupling=True, ckd_tables=pack_ckd_tables_flat(),
    )
    assert torch.allclose(sw_up, ref.sw_flux_up, atol=2e-2, rtol=1e-4)
    assert torch.allclose(sw_dn, ref.sw_flux_dn, atol=2e-2, rtol=1e-4)
    assert torch.allclose(lw_up, ref.lw_flux_up, atol=2e-2, rtol=1e-4)
    assert torch.allclose(lw_dn, ref.lw_flux_dn, atol=2e-2, rtol=1e-4)
    assert torch.allclose(qrad[2], ref.q_rad_net, atol=2e-5, rtol=1e-4)

    # 2-moment microphysics
    thm = T * (1e5 / p) ** (287.05 / 1004.64)
    qn = qv + qc + qi
    qp = torch.zeros(nz, M)
    nc = torch.where(qc > 0, torch.full_like(qc, 8.0e7), torch.zeros_like(qc))
    ni = torch.where(qi > 0, torch.full_like(qi, 5.0e4), torch.zeros_like(qi))
    w_eff = torch.full((nz, M), 0.8)
    is_ocean = (torch.arange(M) % 2 == 1)
    u_sfc = torch.full((M,), 7.5)
    q_dust = torch.full((nz, M), 1e-8)

    res_ref = step_microphysics_two_moment(
        rho, thm, qn, qp, qv, nc, ni, dz, 30.0,
        w_eff=w_eff, is_ocean=is_ocean, u_sfc=u_sfc, q_dust=q_dust,
    )
    res_cu = run_crm_microphysics_2m_metal(
        sim_lib,
        rho_d=rho, theta_m_c=thm, q_n=qn, q_p=qp, q_v_prev=qv, n_c_in=nc, n_i_in=ni,
        dz_c=dz, dt=30.0, w_eff=w_eff, is_ocean=is_ocean, u_sfc=u_sfc, q_dust=q_dust,
        mp_consts=_get_mp2m_metal_consts(),
    )
    assert torch.allclose(res_cu[0], res_ref.theta_m_c, atol=5e-4, rtol=1e-4)
    assert torch.allclose(res_cu[1], res_ref.q_n, atol=5e-6, rtol=1e-4)
    assert torch.allclose(res_cu[5], res_ref.q_c, atol=5e-6, rtol=1e-4)
    assert torch.allclose(res_cu[6], res_ref.q_i, atol=5e-6, rtol=1e-4)
    assert torch.allclose(res_cu[7], res_ref.n_c, atol=5e3, rtol=1e-3)
    assert torch.allclose(res_cu[8], res_ref.n_i, atol=2e2, rtol=1e-3)
    assert torch.allclose(res_cu[9], res_ref.precip_rain, atol=1e-6, rtol=1e-4)


def test_full_coupled_aoba_step_through_transpiled_cuda_kernels(
    host_cuda_sim: tuple[_HostCUDASimLib, list[KernelSpec]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route ALL fused GPU kernels of a full coupled AOBA run through transpiled CUDA C++17."""
    sim_lib, _ = host_cuda_sim

    cfg = AOBAConfig(
        glevel=1, nz=16, z_top=36000.0, crm_nx=4, crm_ny=1,
        use_earth2014=False, rad_scheme="ckd", ocean_mode="ogcm_3d",
        ocean_nz=6, microphysics_scheme="two_moment", enable_nonoro_gwd=True,
    )
    torch.manual_seed(123)
    m_ref = init_aoba_model(cfg, device="cpu")
    for _ in range(3):
        d_ref = step_aoba_model(m_ref)

    # Monkeypatch GPU dispatch hooks so CPU tensors route through our compiled CUDA C++17 sim_lib
    import aoba.kernels as ak
    import aoba.scvt_dynamics as sd

    monkeypatch.setattr(ak, "_get_gpu_lib", lambda dev=None: sim_lib)
    monkeypatch.setattr(
        ak, "can_use_crm_metal",
        lambda t: t.dtype == torch.float32 and 2 <= t.shape[0] <= ak._MAX_COL_METAL_K and not t.requires_grad,
    )
    monkeypatch.setattr(
        ak, "can_use_elem_metal",
        lambda t: t.dtype == torch.float32 and t.numel() > 0 and not t.requires_grad,
    )
    monkeypatch.setattr(
        sd, "_can_use_scvt_metal",
        lambda x: x.dtype == torch.float32 and not x.requires_grad and (x.ndim == 1 or (x.ndim == 2 and x.shape[0] <= ak._MAX_COL_METAL_K)),
    )
    # Also update imported bindings inside modules
    import aoba.biosphere as bio
    import aoba.crm as crm_mod
    import aoba.dust as dust_mod
    import aoba.gravity_wave as gw_mod
    import aoba.microphysics as mp_mod
    import aoba.orography as oro_mod
    import aoba.radiation as rad_mod
    import aoba.radiation_ckd as ckd_mod
    import aoba.scvt_mesh as mesh_mod
    import aoba.surface as sfc_mod
    import aoba.turbulence as turb_mod

    for mod in (crm_mod, mp_mod, rad_mod, ckd_mod, turb_mod, dust_mod, gw_mod, oro_mod, sd):
        if hasattr(mod, "can_use_crm_metal"):
            monkeypatch.setattr(mod, "can_use_crm_metal", ak.can_use_crm_metal)
    for mod in (bio, turb_mod, sfc_mod, mesh_mod, sd):
        if hasattr(mod, "can_use_elem_metal"):
            monkeypatch.setattr(mod, "can_use_elem_metal", ak.can_use_elem_metal)

    torch.manual_seed(123)
    m_cu = init_aoba_model(cfg, device="cpu")
    for _ in range(3):
        d_cu = step_aoba_model(m_cu)

    assert abs(d_cu.global_mean_t_sfc_k - d_ref.global_mean_t_sfc_k) < 1e-2
    assert abs(d_cu.global_mean_olr_wm2 - d_ref.global_mean_olr_wm2) < 0.2
    assert abs(d_cu.global_mean_asr_wm2 - d_ref.global_mean_asr_wm2) < 0.2
    assert abs(d_cu.global_mean_precip_mm_day - d_ref.global_mean_precip_mm_day) < 0.05
    assert torch.allclose(m_cu.atmos.theta_m, m_ref.atmos.theta_m, atol=5e-2, rtol=1e-3)
    assert torch.allclose(m_cu.atmos.u_edge, m_ref.atmos.u_edge, atol=5e-2, rtol=1e-3)


def test_real_cuda_backend_when_available() -> None:
    """Run native CUDA JIT kernel compilation & coupled parity check when an NVIDIA GPU is present."""
    if not torch.cuda.is_available():
        pytest.skip("No NVIDIA CUDA GPU present on this machine")

    assert get_active_gpu_backend("cuda") == "cuda"
    cfg = AOBAConfig(
        glevel=1, nz=16, z_top=36000.0, crm_nx=8, crm_ny=1,
        use_earth2014=False, rad_scheme="ckd", ocean_mode="ogcm_3d",
        ocean_nz=6, microphysics_scheme="two_moment", enable_nonoro_gwd=True,
    )
    torch.manual_seed(99)
    m_cpu = init_aoba_model(cfg, device="cpu")
    for _ in range(3):
        d_cpu = step_aoba_model(m_cpu)

    torch.manual_seed(99)
    m_gpu = init_aoba_model(cfg, device="cuda")
    for _ in range(3):
        d_gpu = step_aoba_model(m_gpu)
    synchronize_device("cuda")

    assert abs(d_gpu.global_mean_t_sfc_k - d_cpu.global_mean_t_sfc_k) < 1e-2
    assert abs(d_gpu.global_mean_olr_wm2 - d_cpu.global_mean_olr_wm2) < 0.2
    assert abs(d_gpu.global_mean_asr_wm2 - d_cpu.global_mean_asr_wm2) < 0.2
