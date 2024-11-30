"""Tests for extreme event detector."""

import numpy as np
import xarray as xr
import pandas as pd
import pytest
from wx_extreme.core.detector import ExtremeEventDetector

def create_test_data():
    """Create synthetic data for testing."""
    times = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    lats = np.linspace(30, 40, 10)
    lons = np.linspace(-100, -90, 10)
    
    # Create temperature field with a heat wave
    temp = np.random.normal(25, 5, size=(len(times), len(lats), len(lons)))
    temp[180:190] += 15  # Add heat wave
    
    return xr.DataArray(
        temp,
        dims=['time', 'latitude', 'longitude'],
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons
        }
    )

def test_percentile_threshold():
    """Test event detection using percentile threshold."""
    detector = ExtremeEventDetector(
        threshold_method="percentile",
        threshold_value=95,
        min_duration=3
    )
    
    data = create_test_data()
    events = detector.detect_events(data)
    
    assert isinstance(events, xr.DataArray)
    assert events.dims == data.dims
    assert events.dtype == bool
    assert events.sum() > 0

def test_absolute_threshold():
    """Test event detection using absolute threshold."""
    detector = ExtremeEventDetector(
        threshold_method="absolute",
        threshold_value=35,
        min_duration=3
    )
    
    data = create_test_data()
    events = detector.detect_events(data)
    
    assert isinstance(events, xr.DataArray)
    assert events.sum() > 0

def test_spatial_coherence():
    """Test spatial coherence filtering."""
    detector = ExtremeEventDetector(
        threshold_method="percentile",
        threshold_value=95,
        min_duration=3,
        spatial_scale=2.0
    )
    
    data = create_test_data()
    events = detector.detect_events(data)
    
    # Events should be spatially coherent
    assert events.sum() <= data.where(data > data.quantile(0.95)).count()

def test_duration_requirement():
    """Test minimum duration requirement."""
    detector = ExtremeEventDetector(
        threshold_method="percentile",
        threshold_value=95,
        min_duration=5
    )
    
    data = create_test_data()
    events = detector.detect_events(data)
    
    # Check that events meet minimum duration
    assert events.sum() > 0  # Should detect some events
    
    # Count consecutive True values along time dimension
    def count_consecutive(x):
        return np.max(np.diff(np.where(np.concatenate(([True], x, [True])))[0])-1)
    
    # Check each grid point
    for lat in events.latitude:
        for lon in events.longitude:
            event_series = events.sel(latitude=lat, longitude=lon)
            if event_series.sum() > 0:  # If there are any events
                max_duration = count_consecutive(event_series.values)
                assert max_duration >= 5

