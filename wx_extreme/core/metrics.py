"""Core metrics for evaluating weather forecasts."""

import numpy as np
import xarray as xr
from scipy import stats

def get_panguweather_t2_forecasts() -> xr.Dataset:
    """Retrieve an xarray handle to a prepared data cube of Pangu-Weather forecasts."""
    return xr.open_dataset(
        "gcs://brightband-share/heatwave/pangu-weather-forecasts.t2.zarr",
        backend_kwargs=dict(
            storage_options=dict(
                token='anon'
            )
        ),
        engine="zarr"
    )

def load_era5_data():
    """Load ERA5 analysis data."""
    return xr.open_dataset(
        "gcs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        storage_options=dict(token='anon'),
        engine='zarr'
    )[['2m_temperature']].sel(time=slice("2021-06-21", "2021-07-10"))
