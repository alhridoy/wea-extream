import pytest
import numpy as np
import pandas as pd
import xarray as xr

@pytest.fixture
def sample_temperature_data():
    """Create sample temperature data for testing."""
    times = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    lats = np.linspace(30, 40, 10)
    lons = np.linspace(-100, -90, 10)
    
    temp = np.random.normal(25, 5, size=(len(times), len(lats), len(lons)))
    
    return xr.DataArray(
        temp,
        dims=['time', 'latitude', 'longitude'],
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons
        }
    )

@pytest.fixture
def sample_events_data():
    """Create sample events data for testing."""
    times = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    events = np.zeros(len(times), dtype=bool)
    events[180:190] = True  # 10-day event
    events[250:255] = True  # 5-day event
    
    return xr.DataArray(
        events,
        dims=['time'],
        coords={'time': times}
    ) 