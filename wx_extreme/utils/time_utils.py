"""Time-related utility functions."""

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from scipy import ndimage


def get_time_resolution(time_coords):
    """Calculate the time resolution of the dataset.
    
    Args:
        time_coords: xarray time coordinate array
        
    Returns:
        timedelta: The time resolution
    """
    if len(time_coords) < 2:
        raise ValueError("Need at least 2 time points to calculate resolution")
    
    return time_coords[1] - time_coords[0]


def get_time_window(time_coords, window_size):
    """Get a rolling time window for analysis.
    
    Args:
        time_coords: xarray time coordinate array
        window_size: Size of the window in same units as time_coords
        
    Returns:
        slice: Time window slice
    """
    start_time = time_coords[0]
    end_time = start_time + window_size
    
    return slice(start_time, end_time)


def aggregate_by_time(data, freq, method='mean'):
    """Aggregate data by time frequency.
    
    Args:
        data: xarray DataArray
        freq: Frequency string (e.g., '1D' for daily)
        method: Aggregation method ('mean', 'sum', 'max', 'min')
        
    Returns:
        xarray.DataArray: Aggregated data
    """
    if method == 'mean':
        return data.resample(time=freq).mean()
    elif method == 'sum':
        return data.resample(time=freq).sum()
    elif method == 'max':
        return data.resample(time=freq).max()
    elif method == 'min':
        return data.resample(time=freq).min()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def get_season(time_coords):
    """Get meteorological season for given times.
    
    Args:
        time_coords: xarray time coordinate array
        
    Returns:
        list: Season for each time point ('DJF', 'MAM', 'JJA', 'SON')
    """
    seasons = {
        1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
        6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON',
        11: 'SON', 12: 'DJF'
    }
    
    return [seasons[t.dt.month.item()] for t in time_coords]


def label_events(event_mask: xr.DataArray) -> tuple[xr.DataArray, int]:
    """Label connected components in time to identify distinct events.
    
    Args:
        event_mask: Boolean mask indicating extreme events
        
    Returns:
        tuple: (labeled_array, num_events)
            - labeled_array: Array with integer labels for each event
            - num_events: Number of distinct events found
    """
    # Convert to numpy array for labeling
    mask_values = event_mask.values
    
    # Label connected components along time axis
    structure = np.ones(3)  # Consider adjacent timesteps as connected
    labeled_array, num_events = ndimage.label(mask_values, structure=structure)
    
    # Convert back to xarray with same coordinates
    labeled_events = xr.DataArray(
        labeled_array,
        dims=event_mask.dims,
        coords=event_mask.coords
    )
    
    return labeled_events, num_events
