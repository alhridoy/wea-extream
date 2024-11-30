"""Extreme weather event detection module."""

from typing import Dict, Optional, Union, List, Tuple
import numpy as np
import xarray as xr
from scipy import ndimage
import pandas as pd

class ExtremeEventDetector:
    """Detector for extreme weather events."""
    
    def __init__(
        self,
        threshold_method: str = "percentile",
        threshold_value: float = 95,
        min_duration: int = 3,
        spatial_scale: float = 0.0,  # degrees for spatial coherence
    ):
        """Initialize extreme event detector.
        
        Args:
            threshold_method: Method for threshold ('percentile' or 'absolute')
            threshold_value: Threshold value (percentile 0-100 or absolute value)
            min_duration: Minimum duration in time steps for event
            spatial_scale: Minimum spatial scale in degrees
        """
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.min_duration = min_duration
        self.spatial_scale = spatial_scale
    
    def detect_events(
        self,
        data: xr.DataArray,
        climatology: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """Detect extreme events in the data.
        
        Args:
            data: Input data array (must have 'time' dimension)
            climatology: Optional baseline climatology
            
        Returns:
            Boolean mask of detected events
        """
        # Ensure data has expected dimensions
        expected_dims = ['time', 'latitude', 'longitude']
        if not all(dim in data.dims for dim in expected_dims):
            raise ValueError(f"Data must have dimensions {expected_dims}")
        
        # Calculate threshold
        if self.threshold_method == "percentile":
            if climatology is not None:
                threshold = climatology.quantile(
                    self.threshold_value/100, dim="time"
                )
            else:
                threshold = data.quantile(
                    self.threshold_value/100, dim="time"
                )
        else:
            threshold = self.threshold_value
        
        # Detect exceedances
        events = data > threshold
        
        # Apply spatial coherence filter if requested
        if self.spatial_scale > 0:
            events = self._apply_spatial_filter(events)
        
        # Apply duration requirement
        if self.min_duration > 1:
            events = self._apply_duration_filter(events)
        
        # Transpose to expected dimension order
        events = events.transpose('time', 'latitude', 'longitude')
        
        return events
    
    def _apply_spatial_filter(self, events: xr.DataArray) -> xr.DataArray:
        """Apply spatial coherence filtering."""
        if not hasattr(events, 'latitude') or not hasattr(events, 'longitude'):
            return events
        
        # Convert spatial scale to grid points
        lat_res = abs(events.latitude[1] - events.latitude[0])
        scale_px = max(1, int(self.spatial_scale / lat_res))
        
        # Create circular kernel for spatial filtering
        y, x = np.ogrid[-scale_px:scale_px+1, -scale_px:scale_px+1]
        kernel = x*x + y*y <= scale_px*scale_px
        
        # Apply opening operation for each time step
        filtered = xr.apply_ufunc(
            lambda x: ndimage.binary_opening(x, kernel),
            events,
            input_core_dims=[['latitude', 'longitude']],
            output_core_dims=[['latitude', 'longitude']],
            vectorize=True,
        )
        
        return filtered
    
    def _apply_duration_filter(self, events: xr.DataArray) -> xr.DataArray:
        """Apply minimum duration requirement."""
        # Create time window kernel
        kernel = np.ones(self.min_duration)
        
        # Apply opening operation along time dimension
        filtered = xr.apply_ufunc(
            lambda x: ndimage.binary_opening(x, kernel),
            events,
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True,
        )
        
        return filtered
