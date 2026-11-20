"""AOBA: Autonomous Optimality Biosphere-Atmosphere model."""

from aoba import constants
from aoba.io_zarr import append_zarr, load_zarr, save_zarr
from aoba.kernels import solve_tridiagonal_batched
from aoba.radiation import RadiationOutput, compute_four_band_radiation, compute_solar_zenith
from aoba.runtime import get_data_dir, get_device, get_output_dir
from aoba.thermodynamics import (
    diagnose_pressure_temperature,
    esat_ice,
    esat_liquid,
    liquid_fraction_condensate,
    qsat_mixed,
    rcemip_analytic_sounding,
    saturation_adjustment,
)
from aoba.vertical_grid import VerticalGrid, build_vertical_grid, compute_stf_heights

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "constants",
    "get_device",
    "get_output_dir",
    "get_data_dir",
    "VerticalGrid",
    "build_vertical_grid",
    "compute_stf_heights",
    "solve_tridiagonal_batched",
    "liquid_fraction_condensate",
    "esat_liquid",
    "esat_ice",
    "qsat_mixed",
    "diagnose_pressure_temperature",
    "saturation_adjustment",
    "rcemip_analytic_sounding",
    "RadiationOutput",
    "compute_solar_zenith",
    "compute_four_band_radiation",
    "save_zarr",
    "append_zarr",
    "load_zarr",
]
