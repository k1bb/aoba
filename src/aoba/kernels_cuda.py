"""Automatic single-source Metal-to-CUDA transpiler and PyTorch CUDA kernel backend for AOBA.

Translates AOBA's 51 fused Metal compute shaders (`kernels.py`, `kernels_extra.py`,
`kernels_physics.py`) directly into native NVIDIA CUDA C++17 kernels compiled on demand
via `torch.utils.cpp_extension.load_inline`, exposing an identical `lib.<kernel>(...,
threads=..., group_size=...)` dispatch interface so every physics and dynamics module
automatically accelerates on both Apple Silicon (`mps`) and NVIDIA GPUs (`cuda`) from a
single source of truth.

References
----------
- NVIDIA CUDA C++ Programming Guide (warp shuffle primitives `__shfl_sync`,
  `__shfl_xor_sync`, shared memory synchronization `__syncthreads`, IEEE-754 float32 math).
- PyTorch C++/CUDA Extension API (`torch.utils.cpp_extension.load_inline`,
  `at::cuda::getCurrentCUDAStream`, `c10::cuda::CUDAGuard`).
- AOBA Metal shader specification (`src/aoba/kernels.py`, `src/aoba/kernels_extra.py`,
  `src/aoba/kernels_physics.py`).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any
import warnings
import torch


_CUDA_LIB_CACHE: dict[str, Any] = {}
_CUDA_LIB_FAILED: set[str] = set()

# Register-heavy column kernels achieve higher SM occupancy at block_size=128 on A100/H100
_HEAVY_COL_KERNELS: frozenset[str] = frozenset({
    "rad_ckd_col",
    "rad_four_band_col",
    "crm_microphysics_col",
    "crm_microphysics_2m_col",
    "crm_mynn_col",
    "crm_fast_hevi_col",
    "scvt_fast_cell_hevi",
    "oro_gwd_col",
    "ngwd_propagate_col",
    "dust_column_col",
    "natural_aerosol_4sp_col",
    "ocean_bgc_column_col",
    "two_leaf_gas_exchange_col",
    "surface_transfer_col",
})


@dataclass(frozen=True)
class KernelParam:
    """Single parsed parameter of a Metal/CUDA compute kernel."""

    name: str
    kind: str  # 'f32_const_ptr', 'f32_ptr', 'f32x4_const_ptr', 'f32x4_ptr', 'i32_const_ptr', 'uint', 'ulong', 'float', 'ulong4', 'float4'
    buffer_index: int


@dataclass(frozen=True)
class KernelSpec:
    """Metadata for a transpiled compute kernel."""

    name: str
    params: tuple[KernelParam, ...]
    uses_simd: bool
    uses_shared: bool


def _strip_comments(text: str) -> str:
    """Remove C/C++ line and block comments from parameter lists."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _find_matching_brace(src: str, open_idx: int) -> int:
    """Return index of '}' matching '{' at `open_idx`, ignoring comments and strings."""
    depth = 0
    i = open_idx
    n = len(src)
    while i < n:
        ch = src[i]
        if ch == "/" and i + 1 < n:
            nxt = src[i + 1]
            if nxt == "/":
                nl = src.find("\n", i + 2)
                i = n if nl < 0 else nl + 1
                continue
            if nxt == "*":
                end = src.find("*/", i + 2)
                i = n if end < 0 else end + 2
                continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"Unmatched '{{' at offset {open_idx}")


def _parse_buffer_param(decl_no_attr: str, buf_idx: int) -> KernelParam:
    """Parse a normalized Metal `[[buffer(i)]]` declaration into a `KernelParam`."""
    s = " ".join(decl_no_attr.split())
    m = re.match(r"^(.*?)\s+([A-Za-z_]\w*)$", s)
    if not m:
        raise ValueError(f"Cannot parse Metal parameter declaration: {decl_no_attr!r}")
    type_part = re.sub(r"\s*([*&])\s*", r"\1", m.group(1).strip())
    name = m.group(2)

    mapping = {
        "device const float*": "f32_const_ptr",
        "device float*": "f32_ptr",
        "device const float4*": "f32x4_const_ptr",
        "device float4*": "f32x4_ptr",
        "device const int*": "i32_const_ptr",
        "device const uint*": "u32_const_ptr",
        "constant uint&": "uint",
        "constant ulong&": "ulong",
        "constant float&": "float",
        "constant ulong4&": "ulong4",
        "constant float4&": "float4",
    }
    if type_part not in mapping:
        raise ValueError(f"Unsupported Metal parameter type {type_part!r} in {decl_no_attr!r}")
    return KernelParam(name=name, kind=mapping[type_part], buffer_index=buf_idx)


def _cuda_param_decl(p: KernelParam) -> str:
    """Return CUDA `__global__` parameter declaration for `p` (`__restrict__` on read-only ptrs)."""
    decls = {
        "f32_const_ptr": f"const float* __restrict__ {p.name}",
        "f32_ptr": f"float* {p.name}",
        "f32x4_const_ptr": f"const float4* __restrict__ {p.name}",
        "f32x4_ptr": f"float4* {p.name}",
        "i32_const_ptr": f"const int* __restrict__ {p.name}",
        "u32_const_ptr": f"const uint* __restrict__ {p.name}",
        "uint": f"uint {p.name}",
        "ulong": f"ulong {p.name}",
        "float": f"float {p.name}",
        "ulong4": f"aoba_ulong4 {p.name}",
        "float4": f"float4 {p.name}",
    }
    return decls[p.kind]


def _transform_body_code(code: str) -> str:
    """Translate MSL expressions/qualifiers inside helpers and kernel bodies to CUDA C++17."""
    # Address-space qualifiers on local pointers/references and shared arrays
    code = re.sub(r"\bthreadgroup_barrier\s*\(\s*mem_flags::mem_threadgroup\s*\)", "AOBA_SYNCTHREADS()", code)
    code = re.sub(r"\bthreadgroup\b", "AOBA_SHARED", code)
    code = re.sub(r"\bdevice\s+const\b", "const", code)
    code = re.sub(r"\bdevice\b\s+", "", code)
    code = re.sub(r"\bthread\s+const\b", "const", code)
    code = re.sub(r"\bthread\b\s+", "", code)

    # Warp shuffle intrinsics
    code = re.sub(r"\bsimd_shuffle_xor\s*\(", "aoba_shfl_xor(", code)
    code = re.sub(r"\bsimd_shuffle\s*\(", "aoba_shfl(", code)

    # Vector constructors (handles both 1-arg splats like float4(0.0f) and N-arg constructors)
    code = re.sub(r"\bfloat4\s*\(", "aoba_float4(", code)
    code = re.sub(r"\bfloat2\s*\(", "aoba_float2(", code)
    return code


_CUDA_PREAMBLE = r"""
#include <cmath>
#include <cstdint>

namespace aoba_k {

using uint = unsigned int;
using ushort = unsigned short;
using uchar = unsigned char;
using ulong = unsigned long long;

struct aoba_ulong4 {
    unsigned long long x;
    unsigned long long y;
    unsigned long long z;
    unsigned long long w;
};

#ifdef __CUDACC__
#define AOBA_GLOBAL __global__
#define AOBA_DEVICE __device__ __forceinline__
#define AOBA_SHARED __shared__
#define AOBA_SYNCTHREADS() __syncthreads()

AOBA_DEVICE float2 aoba_float2(float x, float y) { return make_float2(x, y); }
AOBA_DEVICE float2 aoba_float2(float v) { return make_float2(v, v); }
AOBA_DEVICE float4 aoba_float4(float x, float y, float z, float w) { return make_float4(x, y, z, w); }
AOBA_DEVICE float4 aoba_float4(float v) { return make_float4(v, v, v, v); }

AOBA_DEVICE float aoba_shfl(float val, uint src_lane) {
    return __shfl_sync(0xffffffffu, val, static_cast<int>(src_lane));
}
AOBA_DEVICE float aoba_shfl_xor(float val, uint lane_mask) {
    return __shfl_xor_sync(0xffffffffu, val, static_cast<int>(lane_mask));
}
AOBA_DEVICE float aoba_shfl_xor(float val, ushort lane_mask) {
    return __shfl_xor_sync(0xffffffffu, val, static_cast<int>(lane_mask));
}

AOBA_DEVICE float max(float a, float b) { return fmaxf(a, b); }
AOBA_DEVICE float min(float a, float b) { return fminf(a, b); }
AOBA_DEVICE int max(int a, int b) { return a > b ? a : b; }
AOBA_DEVICE int min(int a, int b) { return a < b ? a : b; }
AOBA_DEVICE uint max(uint a, uint b) { return a > b ? a : b; }
AOBA_DEVICE uint min(uint a, uint b) { return a < b ? a : b; }
AOBA_DEVICE float clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
AOBA_DEVICE int clamp(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
AOBA_DEVICE uint clamp(uint x, uint lo, uint hi) { return x < lo ? lo : (x > hi ? hi : x); }
AOBA_DEVICE float fabs(float x) { return fabsf(x); }
AOBA_DEVICE float exp(float x) { return expf(x); }
AOBA_DEVICE float log(float x) { return logf(x); }
AOBA_DEVICE float pow(float a, float b) { return powf(a, b); }
AOBA_DEVICE float sqrt(float x) { return sqrtf(x); }
AOBA_DEVICE float rsqrt(float x) { return rsqrtf(x); }
AOBA_DEVICE float sin(float x) { return sinf(x); }
AOBA_DEVICE float cos(float x) { return cosf(x); }
AOBA_DEVICE float tanh(float x) { return tanhf(x); }
AOBA_DEVICE float acos(float x) { return acosf(x); }
AOBA_DEVICE float atan(float x) { return atanf(x); }
AOBA_DEVICE float atan2(float y, float x) { return atan2f(y, x); }
AOBA_DEVICE float fma(float a, float b, float c) { return fmaf(a, b, c); }

#define AOBA_GRID_GID() (blockIdx.x * blockDim.x + threadIdx.x)
#define AOBA_LOCAL_LID() (threadIdx.x)
#define AOBA_GROUP_ID() (blockIdx.x)
#define AOBA_SIMD_LANE() (threadIdx.x & 31u)
#endif
"""


def transpile_metal_to_cuda(msl_src: str) -> tuple[str, list[KernelSpec]]:
    """Transpile AOBA's Metal shader source into CUDA C++17 kernel bodies + metadata."""
    cleaned = re.sub(r"^\s*#include\s*<metal_stdlib>\s*$", "", msl_src, flags=re.M)
    cleaned = re.sub(r"^\s*using\s+namespace\s+metal\s*;\s*$", "", cleaned, flags=re.M)

    out_chunks: list[str] = [_CUDA_PREAMBLE]
    specs: list[KernelSpec] = []

    pos = 0
    n = len(cleaned)
    kernel_pat = re.compile(r"\bkernel\s+void\s+(\w+)\s*\(")

    while pos < n:
        m = kernel_pat.search(cleaned, pos)
        if m is None:
            tail = cleaned[pos:]
            out_chunks.append(_transform_non_kernel_chunk(tail))
            break

        # Emit everything before this kernel (helper functions, #define, comments)
        prefix = cleaned[pos:m.start()]
        out_chunks.append(_transform_non_kernel_chunk(prefix))

        kname = m.group(1)
        # Find matching ')' for parameter list
        p_open = m.end() - 1
        depth = 0
        p_close = p_open
        while p_close < n:
            if cleaned[p_close] == "(":
                depth += 1
            elif cleaned[p_close] == ")":
                depth -= 1
                if depth == 0:
                    break
            p_close += 1
        params_raw = cleaned[p_open + 1:p_close]

        brace_open = cleaned.find("{", p_close)
        brace_close = _find_matching_brace(cleaned, brace_open)
        body_raw = cleaned[brace_open + 1:brace_close]

        # Parse kernel parameters
        params_nc = _strip_comments(params_raw)
        buf_params: list[KernelParam] = []
        id_inits: list[str] = []

        for raw_item in params_nc.split(","):
            item = " ".join(raw_item.split())
            if not item:
                continue
            attr_m = re.search(r"\[\[\s*([^\]]+?)\s*\]\]", item)
            if not attr_m:
                raise ValueError(f"Kernel {kname} parameter missing [[...]] attribute: {item!r}")
            attr = attr_m.group(1).strip()
            decl_no_attr = item[:attr_m.start()].strip()

            buf_m = re.match(r"^buffer\s*\(\s*(\d+)\s*\)$", attr)
            if buf_m:
                buf_params.append(_parse_buffer_param(decl_no_attr, int(buf_m.group(1))))
            else:
                var_m = re.match(r"^(?:const\s+)?(?:uint|int|ushort)\s+([A-Za-z_]\w*)$", decl_no_attr)
                if not var_m:
                    raise ValueError(f"Kernel {kname} unexpected thread-id param: {item!r}")
                vname = var_m.group(1)
                if attr == "thread_position_in_grid":
                    id_inits.append(f"    const uint {vname} = AOBA_GRID_GID();")
                elif attr == "thread_position_in_threadgroup":
                    id_inits.append(f"    const uint {vname} = AOBA_LOCAL_LID();")
                elif attr == "threadgroup_position_in_grid":
                    id_inits.append(f"    const uint {vname} = AOBA_GROUP_ID();")
                elif attr == "thread_index_in_simdgroup":
                    id_inits.append(f"    const uint {vname} = AOBA_SIMD_LANE();")
                else:
                    raise ValueError(f"Kernel {kname} unsupported attribute [[{attr}]]")

        # Validate buffer indices are contiguous 0..K-1 in declaration order
        expected_idxs = list(range(len(buf_params)))
        actual_idxs = [p.buffer_index for p in buf_params]
        if actual_idxs != expected_idxs:
            buf_params.sort(key=lambda p: p.buffer_index)
            if [p.buffer_index for p in buf_params] != expected_idxs:
                raise ValueError(f"Kernel {kname} non-contiguous buffer indices: {actual_idxs}")

        body_cuda = _transform_body_code(body_raw)
        body_nc = _strip_comments(body_cuda)
        uses_simd = ("aoba_shfl(" in body_nc) or ("aoba_shfl_xor(" in body_nc)
        uses_shared = ("AOBA_SHARED" in body_nc) or ("AOBA_SYNCTHREADS()" in body_nc)

        param_sig = ",\n    ".join(_cuda_param_decl(p) for p in buf_params)
        id_block = "\n".join(id_inits) + ("\n" if id_inits else "")
        out_chunks.append(
            f"\nAOBA_GLOBAL void {kname}(\n    {param_sig}\n) {{\n{id_block}{body_cuda}\n}}\n"
        )
        specs.append(
            KernelSpec(
                name=kname,
                params=tuple(buf_params),
                uses_simd=uses_simd,
                uses_shared=uses_shared,
            )
        )
        pos = brace_close + 1

    out_chunks.append("\n} // namespace aoba_k\n")
    return "".join(out_chunks), specs


def _transform_non_kernel_chunk(chunk: str) -> str:
    """Transform inline helper functions and non-kernel code outside kernel definitions."""
    if not chunk.strip():
        return chunk
    # Convert MSL `inline <ret> <name>(` to `AOBA_DEVICE <ret> <name>(`
    chunk = re.sub(r"\binline\s+((?:void|float|float2|float4|int|uint|bool)\s+[A-Za-z_]\w*\s*\()", r"AOBA_DEVICE \1", chunk)
    return _transform_body_code(chunk)


def _cpp_wrapper_param(p: KernelParam) -> str:
    """Return C++ pybind11 launcher parameter type for `p`."""
    if p.kind in ("f32_const_ptr", "f32_ptr", "f32x4_const_ptr", "f32x4_ptr", "i32_const_ptr", "u32_const_ptr"):
        return f"const torch::Tensor& {p.name}"
    if p.kind in ("uint", "ulong"):
        return f"int64_t {p.name}"
    if p.kind == "float":
        return f"double {p.name}"
    if p.kind == "ulong4":
        return f"const std::array<int64_t, 4>& {p.name}"
    if p.kind == "float4":
        return f"const std::array<double, 4>& {p.name}"
    raise ValueError(f"Unknown param kind: {p.kind}")


def _cuda_call_arg(p: KernelParam) -> str:
    """Return expression passing `p` from C++ launcher into `__global__` CUDA kernel."""
    if p.kind == "f32_const_ptr":
        return f"{p.name}.data_ptr<float>()"
    if p.kind == "f32_ptr":
        return f"{p.name}.data_ptr<float>()"
    if p.kind == "f32x4_const_ptr":
        return f"reinterpret_cast<const float4*>({p.name}.data_ptr<float>())"
    if p.kind == "f32x4_ptr":
        return f"reinterpret_cast<float4*>({p.name}.data_ptr<float>())"
    if p.kind == "i32_const_ptr":
        return f"{p.name}.data_ptr<int32_t>()"
    if p.kind == "u32_const_ptr":
        return f"reinterpret_cast<const aoba_k::uint*>({p.name}.data_ptr())"
    if p.kind == "uint":
        return f"static_cast<aoba_k::uint>({p.name})"
    if p.kind == "ulong":
        return f"static_cast<aoba_k::ulong>({p.name})"
    if p.kind == "float":
        return f"static_cast<float>({p.name})"
    if p.kind == "ulong4":
        return (
            f"aoba_k::aoba_ulong4{{static_cast<unsigned long long>({p.name}[0]), "
            f"static_cast<unsigned long long>({p.name}[1]), "
            f"static_cast<unsigned long long>({p.name}[2]), "
            f"static_cast<unsigned long long>({p.name}[3])}}"
        )
    if p.kind == "float4":
        return (
            f"make_float4(static_cast<float>({p.name}[0]), static_cast<float>({p.name}[1]), "
            f"static_cast<float>({p.name}[2]), static_cast<float>({p.name}[3]))"
        )
    raise ValueError(f"Unknown param kind: {p.kind}")


def build_cuda_extension_sources(msl_src: str) -> tuple[str, str, list[KernelSpec]]:
    """Generate `(cpp_sources, cuda_sources, specs)` ready for `load_inline`."""
    cuda_kernels_src, specs = transpile_metal_to_cuda(msl_src)

    cuda_launchers: list[str] = [
        "#include <array>\n",
        "#include <ATen/cuda/CUDAContext.h>\n",
        "#include <c10/cuda/CUDAGuard.h>\n",
        "#include <c10/cuda/CUDAException.h>\n",
        cuda_kernels_src,
        "\n// --- PyTorch CUDA Kernel Launchers ---\n",
    ]
    cpp_decls: list[str] = [
        "#include <array>\n",
        "#include <pybind11/stl.h>\n\n",
    ]
    pybind_defs: list[str] = []

    for spec in specs:
        sig_params = [_cpp_wrapper_param(p) for p in spec.params] + [
            "int64_t total_threads",
            "int64_t group_size",
        ]
        sig_str = ", ".join(sig_params)
        cpp_decls.append(f"void launch_{spec.name}({sig_str});\n")
        pybind_defs.append(f'    m.def("{spec.name}", &launch_{spec.name});\n')

        first_tensor = next(
            p.name for p in spec.params
            if p.kind in ("f32_const_ptr", "f32_ptr", "f32x4_const_ptr", "f32x4_ptr", "i32_const_ptr", "u32_const_ptr")
        )
        checks: list[str] = []
        for p in spec.params:
            if p.kind in ("f32_const_ptr", "f32_ptr", "f32x4_const_ptr", "f32x4_ptr"):
                checks.append(
                    f'    TORCH_CHECK({p.name}.is_cuda() && {p.name}.is_contiguous() '
                    f'&& {p.name}.scalar_type() == torch::kFloat32, '
                    f'"{spec.name}: {p.name} must be contiguous CUDA float32");'
                )
            elif p.kind == "i32_const_ptr":
                checks.append(
                    f'    TORCH_CHECK({p.name}.is_cuda() && {p.name}.is_contiguous() '
                    f'&& {p.name}.scalar_type() == torch::kInt32, '
                    f'"{spec.name}: {p.name} must be contiguous CUDA int32");'
                )
            elif p.kind == "u32_const_ptr":
                checks.append(
                    f'    TORCH_CHECK({p.name}.is_cuda() && {p.name}.is_contiguous() '
                    f'&& {p.name}.element_size() == 4, '
                    f'"{spec.name}: {p.name} must be contiguous 32-bit CUDA uint/int tensor");'
                )

        default_block = 128 if spec.name in _HEAVY_COL_KERNELS else 256
        call_args = ",\n        ".join(_cuda_call_arg(p) for p in spec.params)
        launch_body = f"""
    aoba_k::{spec.name}<<<static_cast<unsigned int>(grid), static_cast<unsigned int>(block), 0, stream>>>(
        {call_args}
    );"""

        cuda_launchers.append(
            f"""
void launch_{spec.name}({sig_str}) {{
    if (total_threads <= 0) return;
{chr(10).join(checks)}
    const c10::cuda::CUDAGuard device_guard({first_tensor}.device());
    const int64_t block = (group_size > 0) ? group_size : {default_block};
    const int64_t grid = (total_threads + block - 1) / block;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();{launch_body}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
"""
        )

    cpp_src = (
        "".join(cpp_decls)
        + "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        + "".join(pybind_defs)
        + "}\n"
    )
    cuda_src = "".join(cuda_launchers)
    return cpp_src, cuda_src, specs


def _conv_f32(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def _conv_i32(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.int32:
        t = t.to(dtype=torch.int32)
    return t if t.is_contiguous() else t.contiguous()


def _conv_u32(t: torch.Tensor) -> torch.Tensor:
    if t.element_size() != 4:
        t = t.to(dtype=torch.int32)
    return t if t.is_contiguous() else t.contiguous()


def _conv_ulong4(val: Any) -> list[int]:
    return [int(val[0]), int(val[1]), int(val[2]), int(val[3])]


def _conv_float4(val: Any) -> list[float]:
    return [float(val[0]), float(val[1]), float(val[2]), float(val[3])]


_CONVERTER_BY_KIND = {
    "f32_const_ptr": _conv_f32,
    "f32_ptr": _conv_f32,
    "f32x4_const_ptr": _conv_f32,
    "f32x4_ptr": _conv_f32,
    "i32_const_ptr": _conv_i32,
    "u32_const_ptr": _conv_u32,
    "uint": int,
    "ulong": int,
    "float": float,
    "ulong4": _conv_ulong4,
    "float4": _conv_float4,
}


class _CUDAKernelCallable:
    """Callable matching `torch.mps.compile_shader` kernel invocation signature."""

    __slots__ = ("_fn", "_name", "_convs", "_n_params")

    def __init__(self, fn: Any, spec: KernelSpec) -> None:
        self._fn = fn
        self._name = spec.name
        self._convs = tuple(_CONVERTER_BY_KIND[p.kind] for p in spec.params)
        self._n_params = len(spec.params)

    def __call__(
        self,
        *args: Any,
        threads: int | tuple[int, ...] = 0,
        group_size: int | tuple[int, ...] | None = None,
    ) -> None:
        n_threads = int(threads[0]) if isinstance(threads, (tuple, list)) else int(threads)
        if group_size is None:
            grp = 0
        elif isinstance(group_size, (tuple, list)):
            grp = int(group_size[0])
        else:
            grp = int(group_size)
        if len(args) != self._n_params:
            raise TypeError(
                f"CUDA kernel {self._name} expected {self._n_params} args, got {len(args)}"
            )
        norm_args = [c(a) for c, a in zip(self._convs, args)]
        self._fn(*norm_args, n_threads, grp)


class _CUDALibProxy:
    """Drop-in CUDA counterpart to `torch.mps.compile_shader`'s library object."""

    def __init__(self, ext_module: Any, specs: list[KernelSpec]) -> None:
        self._ext = ext_module
        self._specs = {s.name: s for s in specs}
        for s in specs:
            setattr(self, s.name, _CUDAKernelCallable(getattr(ext_module, s.name), s))

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return tuple(self._specs.keys())


def get_cuda_cache_dir(key_hash: str) -> Path:
    """Return persistent directory for compiled CUDA extension artifacts."""
    base_env = os.environ.get("AOBA_CUDA_CACHE_DIR", "").strip()
    if base_env:
        base = Path(base_env).expanduser()
    else:
        base = Path.home() / ".cache" / "aoba_cuda_ext"
    target = base / f"aoba_cuda_{key_hash}"
    target.mkdir(parents=True, exist_ok=True)
    return target


_CUDA_BACKEND_REV = "v3_restrict_l1"


def get_cuda_lib(msl_src: str) -> _CUDALibProxy | None:
    """Compile (or load cached) AOBA CUDA kernel extension when running on NVIDIA CUDA."""
    if not torch.cuda.is_available():
        return None
    if os.environ.get("AOBA_DISABLE_CUDA_KERNELS") == "1":
        return None

    fast_math = os.environ.get("AOBA_CUDA_FAST_MATH") == "1"
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    hasher = hashlib.sha256()
    hasher.update(msl_src.encode("utf-8"))
    hasher.update(
        f"|rev={_CUDA_BACKEND_REV}|pt={torch.__version__}|cu={torch.version.cuda}"
        f"|sm={cc_major}{cc_minor}|fm={int(fast_math)}".encode()
    )
    key = hasher.hexdigest()[:12]

    if key in _CUDA_LIB_CACHE:
        return _CUDA_LIB_CACHE[key]
    if key in _CUDA_LIB_FAILED:
        return None

    try:
        from torch.utils.cpp_extension import load_inline

        cpp_src, cuda_src, specs = build_cuda_extension_sources(msl_src)
        build_dir = get_cuda_cache_dir(key)
        mod_name = f"aoba_cuda_kernels_{key}"
        # Do not force -std=c++17: PyTorch >= 2.10 requires C++20 and load_inline supplies the right -std automatically.
        extra_cuda = ["-O3"]
        if fast_math:
            extra_cuda.append("--use_fast_math")
        verbose = os.environ.get("AOBA_CUDA_VERBOSE") == "1"
        if verbose or not any(build_dir.glob(f"{mod_name}*.so")):
            print(
                f"[AOBA CUDA] Compiling {len(specs)} fused CUDA kernels for "
                f"{torch.cuda.get_device_name(0)} (sm_{cc_major}{cc_minor}) into {build_dir} ...",
                flush=True,
            )
        ext = load_inline(
            name=mod_name,
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=None,
            extra_cflags=["-O3"],
            extra_cuda_cflags=extra_cuda,
            build_directory=str(build_dir),
            verbose=verbose,
            with_cuda=True,
        )
        proxy = _CUDALibProxy(ext, specs)
        _CUDA_LIB_CACHE[key] = proxy
        return proxy
    except Exception as exc:
        _CUDA_LIB_FAILED.add(key)
        warnings.warn(
            f"AOBA CUDA fused kernel compilation unavailable ({exc}); falling back to PyTorch CUDA ops.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# Host C++17 Verification Harness (allows strict compiler + numerical parity
# verification of ALL 43 transpiled CUDA kernels even on machines without nvcc)
# ---------------------------------------------------------------------------
_HOST_CPP_SHIM = r"""
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
inline float2 make_float2(float x, float y) { return float2{x, y}; }
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }

namespace aoba_sim {
struct SpinBarrier {
    const int n_threads;
    std::atomic<int> arrived{0};
    std::atomic<int> phase{0};
    explicit SpinBarrier(int n) : n_threads(n) {}
    void wait() {
        const int p = phase.load(std::memory_order_relaxed);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == n_threads) {
            arrived.store(0, std::memory_order_relaxed);
            phase.store(p + 1, std::memory_order_release);
        } else {
            while (phase.load(std::memory_order_acquire) == p) {
                std::this_thread::yield();
            }
        }
    }
};

struct BlockCtx {
    uint32_t block_idx;
    uint32_t block_dim;
    SpinBarrier* block_barrier;
    SpinBarrier* warp_barriers[2];
    float warp_buf[2][32];
};

thread_local uint32_t tl_tid = 0;
thread_local BlockCtx* tl_ctx = nullptr;
} // namespace aoba_sim

#define AOBA_GLOBAL inline
#define AOBA_DEVICE inline
#define AOBA_SHARED static thread_local
// NOTE: for kernels using shared memory, we override AOBA_SHARED via block-scoped macro below.
"""


def build_host_cpp_verification_source(msl_src: str) -> tuple[str, list[KernelSpec]]:
    """Build a standalone C++17 source file that compiles & executes ALL transpiled CUDA kernels on CPU.

    This executes the exact same transpiled kernel bodies produced by `transpile_metal_to_cuda`
    (with genuine 32-lane warp shuffle emulation and 64-thread block barrier emulation) so every
    syntax rule, type, indexing offset, warp shuffle, and physics formula can be verified on macOS
    with `clang++ -std=c++17 -O2 -Wall -Wextra -Werror` before deploying to NVIDIA GPUs.
    """
    cuda_kernels_src, specs = transpile_metal_to_cuda(msl_src)

    # Replace `#ifdef __CUDACC__ ... #endif` block in preamble with host-CPU C++17 emulation
    host_math_block = r"""
#ifndef __CUDACC__
#define AOBA_GLOBAL inline
#define AOBA_DEVICE inline
#define AOBA_SHARED static
#define AOBA_SYNCTHREADS() aoba_sim::tl_ctx->block_barrier->wait()

AOBA_DEVICE float2 aoba_float2(float x, float y) { return make_float2(x, y); }
AOBA_DEVICE float2 aoba_float2(float v) { return make_float2(v, v); }
AOBA_DEVICE float4 aoba_float4(float x, float y, float z, float w) { return make_float4(x, y, z, w); }
AOBA_DEVICE float4 aoba_float4(float v) { return make_float4(v, v, v, v); }

AOBA_DEVICE float aoba_shfl(float val, uint src_lane) {
    const uint w = aoba_sim::tl_tid >> 5u;
    const uint lane = aoba_sim::tl_tid & 31u;
    aoba_sim::tl_ctx->warp_buf[w][lane] = val;
    aoba_sim::tl_ctx->warp_barriers[w]->wait();
    const float out = aoba_sim::tl_ctx->warp_buf[w][src_lane & 31u];
    aoba_sim::tl_ctx->warp_barriers[w]->wait();
    return out;
}
AOBA_DEVICE float aoba_shfl_xor(float val, uint lane_mask) {
    const uint lane = aoba_sim::tl_tid & 31u;
    return aoba_shfl(val, lane ^ lane_mask);
}
AOBA_DEVICE float aoba_shfl_xor(float val, ushort lane_mask) {
    const uint lane = aoba_sim::tl_tid & 31u;
    return aoba_shfl(val, lane ^ static_cast<uint>(lane_mask));
}

AOBA_DEVICE float max(float a, float b) { return std::fmax(a, b); }
AOBA_DEVICE float min(float a, float b) { return std::fmin(a, b); }
AOBA_DEVICE int max(int a, int b) { return a > b ? a : b; }
AOBA_DEVICE int min(int a, int b) { return a < b ? a : b; }
AOBA_DEVICE uint max(uint a, uint b) { return a > b ? a : b; }
AOBA_DEVICE uint min(uint a, uint b) { return a < b ? a : b; }
AOBA_DEVICE float clamp(float x, float lo, float hi) { return std::fmin(std::fmax(x, lo), hi); }
AOBA_DEVICE int clamp(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
AOBA_DEVICE uint clamp(uint x, uint lo, uint hi) { return x < lo ? lo : (x > hi ? hi : x); }
AOBA_DEVICE float fabs(float x) { return std::fabs(x); }
AOBA_DEVICE float exp(float x) { return std::exp(x); }
AOBA_DEVICE float log(float x) { return std::log(x); }
AOBA_DEVICE float pow(float a, float b) { return std::pow(a, b); }
AOBA_DEVICE float sqrt(float x) { return std::sqrt(x); }
AOBA_DEVICE float rsqrt(float x) { return 1.0f / std::sqrt(x); }
AOBA_DEVICE float sin(float x) { return std::sin(x); }
AOBA_DEVICE float cos(float x) { return std::cos(x); }
AOBA_DEVICE float tanh(float x) { return std::tanh(x); }
AOBA_DEVICE float acos(float x) { return std::acos(x); }
AOBA_DEVICE float atan(float x) { return std::atan(x); }
AOBA_DEVICE float atan2(float y, float x) { return std::atan2(y, x); }
AOBA_DEVICE float fma(float a, float b, float c) { return std::fma(a, b, c); }

#define AOBA_GRID_GID() (aoba_sim::tl_ctx->block_idx * aoba_sim::tl_ctx->block_dim + aoba_sim::tl_tid)
#define AOBA_LOCAL_LID() (aoba_sim::tl_tid)
#define AOBA_GROUP_ID() (aoba_sim::tl_ctx->block_idx)
#define AOBA_SIMD_LANE() (aoba_sim::tl_tid & 31u)
#endif
"""

    # Build C-ABI extern "C" entrypoints callable via ctypes for every kernel
    c_wrappers: list[str] = ['\nextern "C" {\n']
    for spec in specs:
        c_params: list[str] = []
        call_args: list[str] = []
        for p in spec.params:
            if p.kind == "f32_const_ptr":
                c_params.append(f"const float* {p.name}")
                call_args.append(p.name)
            elif p.kind == "f32_ptr":
                c_params.append(f"float* {p.name}")
                call_args.append(p.name)
            elif p.kind == "f32x4_const_ptr":
                c_params.append(f"const float* {p.name}")
                call_args.append(f"reinterpret_cast<const float4*>({p.name})")
            elif p.kind == "f32x4_ptr":
                c_params.append(f"float* {p.name}")
                call_args.append(f"reinterpret_cast<float4*>({p.name})")
            elif p.kind == "i32_const_ptr":
                c_params.append(f"const int32_t* {p.name}")
                call_args.append(p.name)
            elif p.kind == "u32_const_ptr":
                c_params.append(f"const uint32_t* {p.name}")
                call_args.append(p.name)
            elif p.kind == "uint":
                c_params.append(f"uint64_t {p.name}")
                call_args.append(f"static_cast<aoba_k::uint>({p.name})")
            elif p.kind == "ulong":
                c_params.append(f"uint64_t {p.name}")
                call_args.append(f"static_cast<aoba_k::ulong>({p.name})")
            elif p.kind == "float":
                c_params.append(f"double {p.name}")
                call_args.append(f"static_cast<float>({p.name})")
            elif p.kind == "ulong4":
                c_params.append(f"const uint64_t* {p.name}")
                call_args.append(
                    f"aoba_k::aoba_ulong4{{{p.name}[0], {p.name}[1], {p.name}[2], {p.name}[3]}}"
                )
            elif p.kind == "float4":
                c_params.append(f"const float* {p.name}")
                call_args.append(f"make_float4({p.name}[0], {p.name}[1], {p.name}[2], {p.name}[3])")

        c_params.extend(["int64_t total_threads", "int64_t group_size"])
        c_sig = ", ".join(c_params)
        arg_str = ", ".join(call_args)

        if spec.uses_simd or spec.uses_shared:
            c_wrappers.append(
                f"""
void host_run_{spec.name}({c_sig}) {{
    const uint32_t bdim = (group_size > 0) ? static_cast<uint32_t>(group_size) : 64u;
    const uint32_t n_blocks = static_cast<uint32_t>((total_threads + bdim - 1) / bdim);
    for (uint32_t blk = 0u; blk < n_blocks; ++blk) {{
        aoba_sim::SpinBarrier b_bar(static_cast<int>(bdim));
        aoba_sim::SpinBarrier w_bar0(std::min<int>(32, static_cast<int>(bdim)));
        aoba_sim::SpinBarrier w_bar1(std::max<int>(1, static_cast<int>(bdim) - 32));
        aoba_sim::BlockCtx ctx{{blk, bdim, &b_bar, {{&w_bar0, &w_bar1}}, {{}}}};
        std::vector<std::thread> workers;
        workers.reserve(bdim);
        for (uint32_t tid = 0u; tid < bdim; ++tid) {{
            workers.emplace_back([&, tid]() {{
                aoba_sim::tl_tid = tid;
                aoba_sim::tl_ctx = &ctx;
                aoba_k::{spec.name}({arg_str});
            }});
        }}
        for (auto& th : workers) th.join();
    }}
}}
"""
            )
        else:
            c_wrappers.append(
                f"""
void host_run_{spec.name}({c_sig}) {{
    (void)group_size;
    // Deliberately round up to block=128 just like CUDA to test every kernel's gid bounds guard!
    const uint32_t bdim = 128u;
    const uint32_t n_blocks = static_cast<uint32_t>((total_threads + bdim - 1) / bdim);
    aoba_sim::BlockCtx ctx{{0u, bdim, nullptr, {{nullptr, nullptr}}, {{}}}};
    aoba_sim::tl_ctx = &ctx;
    for (uint32_t blk = 0u; blk < n_blocks; ++blk) {{
        ctx.block_idx = blk;
        for (uint32_t tid = 0u; tid < bdim; ++tid) {{
            aoba_sim::tl_tid = tid;
            aoba_k::{spec.name}({arg_str});
        }}
    }}
}}
"""
            )
    c_wrappers.append('} // extern "C"\n')

    # Inject host_math_block right after `#endif` in cuda_kernels_src
    anchor = "#define AOBA_SIMD_LANE() (threadIdx.x & 31u)\n#endif"
    assert anchor in cuda_kernels_src
    patched_kernels = cuda_kernels_src.replace(anchor, anchor + "\n" + host_math_block, 1)

    full_host_src = (
        _HOST_CPP_SHIM.replace("#define AOBA_GLOBAL inline\n#define AOBA_DEVICE inline\n#define AOBA_SHARED static thread_local\n", "")
        + "\n"
        + patched_kernels
        + "\n"
        + "".join(c_wrappers)
    )
    return full_host_src, specs
