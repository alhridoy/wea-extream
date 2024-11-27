"""Tests for time utility functions."""

import numpy as np
import xarray as xr
import pandas as pd
import pytest
from datetime import datetime, timedelta
from wx_extreme.utils import time_utils


def test_get_time_resolution():
    """Test time resolution calculation."""
    times = xr.DataArray(
        pd.date_range('2020-01-01', '2020-01-10', freq='6H'),
        dims=['time']
    )
    
    resolution = time_utils.get_time_resolution(times)
    assert isinstance(resolution, timedelta)
    assert resolution == timedelta(hours=6)


def test_get_time_window():
    """Test time window creation."""
    times = xr.DataArray(
        pd.date_range('2020-01-01', '2020-01-10', freq='1D'),
        dims=['time']
    )
    window = timedelta(days=3)
    
    time_slice = time_utils.get_time_window(times, window)
    assert isinstance(time_slice, slice)
    assert time_slice.start == times[0]
    assert time_slice.stop == times[0] + window


def test_aggregate_by_time():
    """Test time aggregation."""
    times = pd.date_range('2020-01-01', '2020-01-10', freq='6H')
    data = xr.DataArray(
        np.random.randn(len(times)),
        dims=['time'],
        coords={'time': times}
    )
    
    # Test mean aggregation
    daily_mean = time_utils.aggregate_by_time(data, '1D', method='mean')
    assert len(daily_mean) < len(data)
    assert daily_mean.time.dt.freq == '1D'
    
    # Test other methods
    daily_max = time_utils.aggregate_by_time(data, '1D', method='max')
    assert len(daily_max) == len(daily_mean)
    assert np.all(daily_max >= daily_mean)
    
    with pytest.raises(ValueError):
        time_utils.aggregate_by_time(data, '1D', method='invalid')


def test_get_season():
    """Test season determination."""
    times = xr.DataArray(
        pd.date_range('2020-01-01', '2020-12-31', freq='1M'),
        dims=['time']
    )
    
    seasons = time_utils.get_season(times)
    assert len(seasons) == len(times)
    assert all(s in ['DJF', 'MAM', 'JJA', 'SON'] for s in seasons)
    assert seasons[0] == 'DJF'  # January
    assert seasons[6] == 'JJA'  # July
