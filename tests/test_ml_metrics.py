"""Tests for ML model metrics."""

import numpy as np
import xarray as xr
import pandas as pd
import pytest

from wx_extreme.core.metrics import MLModelMetrics

def create_synthetic_data():
    """Create synthetic data for testing."""
    # Create grid
    times = pd.date_range('2020-01-01', '2020-01-10')
    lats = np.linspace(30, 40, 10)
    lons = np.linspace(-100, -90, 10)
    
    # Create temperature field
    temp = np.random.normal(25, 5, (len(times), len(lats), len(lons)))
    
    # Create DataArray
    data = xr.DataArray(
        temp,
        dims=['time', 'latitude', 'longitude'],
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons
        }
    )
    
    return data, data.copy()

def test_extreme_value_skill_score():
    """Test extreme value skill score calculation."""
    forecast, obs = create_synthetic_data()
    
    score = MLModelMetrics.extreme_value_skill_score(
        forecast,
        obs,
        threshold=30
    )
    
    assert isinstance(score, float)
    assert 0 <= score <= 1
    
    # Perfect forecast should have score 1
    perfect_score = MLModelMetrics.extreme_value_skill_score(
        obs,
        obs,
        threshold=30
    )
    assert perfect_score == 1.0

def test_pattern_prediction_score():
    """Test pattern prediction score calculation."""
    forecast, obs = create_synthetic_data()
    
    score = MLModelMetrics.pattern_prediction_score(
        forecast,
        obs,
        spatial_scale=1.0
    )
    
    assert isinstance(score, float)
    assert 0 <= score <= 1
    
    # Perfect forecast should have score 1
    perfect_score = MLModelMetrics.pattern_prediction_score(obs, obs)
    assert perfect_score > 0.99
    
    # Random forecast should have low score
    random_forecast = xr.DataArray(
        np.random.normal(0, 1, obs.shape),
        dims=obs.dims,
        coords=obs.coords
    )
    random_score = MLModelMetrics.pattern_prediction_score(random_forecast, obs)
    assert random_score < 0.5

def test_physical_consistency_score():
    """Test physical consistency score calculation."""
    forecast, obs = create_synthetic_data()
    
    # Create additional variables needed for physical consistency
    pressure = xr.DataArray(
        np.tile(1000 - np.arange(10).reshape(-1, 1), (1, 10)) * 100,  # Pressure levels
        dims=['latitude', 'longitude'],
        coords={
            'latitude': forecast.latitude,
            'longitude': forecast.longitude
        }
    ).expand_dims(time=forecast.time)
    
    # Test physical consistency score
    score = MLModelMetrics.physical_consistency_score(
        forecast,
        pressure,
        obs
    )
    
    assert isinstance(score, float)
    assert 0 <= score <= 1
    
    # Perfect forecast should have high consistency
    perfect_score = MLModelMetrics.physical_consistency_score(
        obs,
        pressure,
        obs
    )
    assert perfect_score > 0.95 