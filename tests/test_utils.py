"""Tests for utility functions."""

import numpy as np
import xarray as xr
import pytest
from wx_extreme.utils import spatial_utils, met_utils, stat_utils


def test_grid_spacing():
    """Test grid spacing calculations."""
    lat = np.array([30.0, 31.0])
    lon = np.array([0.0, 1.0])
    
    dx, dy = spatial_utils.get_grid_spacing(lat, lon)
    
    assert isinstance(dx, float)
    assert isinstance(dy, float)
    assert dx > 0
    assert dy > 0


def test_potential_temperature():
    """Test potential temperature calculation."""
    temp = xr.DataArray(
        np.array([273.15, 283.15]),
        dims=['level'],
        coords={'level': [0, 1]}
    )
    pressure = xr.DataArray(
        np.array([1000.0, 900.0]),
        dims=['level'],
        coords={'level': [0, 1]}
    )
    
    theta = met_utils.potential_temperature(temp, pressure)
    
    assert isinstance(theta, xr.DataArray)
    assert theta.dims == temp.dims
    assert np.all(theta.values > temp.values)  # Potential temp should be higher


def test_calculate_percentile():
    """Test percentile calculation."""
    data = xr.DataArray(
        np.random.randn(100),
        dims=['time'],
        coords={'time': np.arange(100)}
    )
    
    p95 = stat_utils.calculate_percentile(data, q=95)
    
    assert isinstance(p95, (float, np.ndarray, xr.DataArray))
    assert np.all(data.values <= p95.values)
    assert np.mean(data > p95) < 0.1  # Roughly 5% should exceed 95th percentile


def test_exceedance_probability():
    """Test exceedance probability calculation."""
    data = xr.DataArray(
        np.random.randn(1000),
        dims=['time'],
        coords={'time': np.arange(1000)}
    )
    threshold = 0.0
    
    prob = stat_utils.exceedance_probability(data, threshold)
    
    assert isinstance(prob, (float, np.ndarray, xr.DataArray))
    assert 0 <= prob <= 1
    np.testing.assert_allclose(
        prob,
        np.mean(data > threshold),
        rtol=1e-10
    )
