"""Tests for plotting utilities."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest
import pandas as pd

from wx_extreme.utils import plot_utils

@pytest.fixture
def sample_data():
    """Create sample data for testing plots."""
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
    
    return data

def test_model_comparison_heatmap(sample_data):
    """Test model comparison heatmap plotting."""
    # Create forecast and reference data
    forecast = sample_data + 2  # Add bias
    reference = sample_data
    
    # Test bias plot
    fig, ax = plt.subplots()
    plot_utils.plot_model_comparison_heatmap(
        forecast,
        reference,
        metric='bias',
        title='Test Bias Plot',
        ax=ax
    )
    
    # Check plot elements
    assert ax.get_title() == 'Test Bias Plot'
    assert ax.get_xlabel() == 'Longitude'
    assert ax.get_ylabel() == 'Latitude'
    
    # Test RMSE plot
    fig, ax = plt.subplots()
    plot_utils.plot_model_comparison_heatmap(
        forecast,
        reference,
        metric='rmse',
        title='Test RMSE Plot',
        ax=ax
    )
    
    plt.close('all')

def test_extreme_event_heatmap(sample_data):
    """Test extreme event heatmap plotting."""
    # Create event mask
    events = sample_data > np.percentile(sample_data, 95)
    
    # Test plot
    fig, ax = plt.subplots()
    plot_utils.plot_extreme_event_heatmap(
        events,
        sample_data,
        ax=ax
    )
    
    # Check plot elements
    assert ax.get_title() == 'Temperature with Extreme Events'
    assert ax.get_xlabel() == 'Longitude'
    assert ax.get_ylabel() == 'Latitude'
    
    plt.close('all')

def test_plot_with_invalid_data():
    """Test plotting with invalid data."""
    # Create invalid data with time dimension
    times = pd.date_range('2020-01-01', '2020-01-10')
    invalid_data = xr.DataArray(
        np.nan * np.ones((len(times), 10, 10)),
        dims=['time', 'x', 'y'],
        coords={'time': times}
    )
    
    # Test handling of invalid data
    with pytest.warns(UserWarning):
        fig, ax = plt.subplots()
        plot_utils.plot_model_comparison_heatmap(
            invalid_data,
            invalid_data,
            metric='bias',
            ax=ax
        )
    
    plt.close('all')

def test_plot_dimensions():
    """Test plotting with different dimensions."""
    # Create 3D data
    data_3d = xr.DataArray(
        np.random.normal(0, 1, (5, 10, 10)),
        dims=['time', 'latitude', 'longitude'],
        coords={
            'time': pd.date_range('2020-01-01', '2020-01-05'),
            'latitude': np.linspace(30, 40, 10),
            'longitude': np.linspace(-100, -90, 10)
        }
    )
    
    # Test plotting with time dimension
    fig, ax = plt.subplots()
    plot_utils.plot_model_comparison_heatmap(
        data_3d,
        data_3d,
        metric='correlation',
        ax=ax
    )
    
    plt.close('all') 