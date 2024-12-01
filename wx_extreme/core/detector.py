"""Module for detecting extreme weather events."""

import numpy as np
import xarray as xr
from typing import Optional

class ExtremeEventDetector:
    """Detector for extreme weather events."""
    
    def __init__(
        self,
        threshold_method: str = "percentile",
        threshold_value: float = 95,
        min_duration: int = 3,
        spatial_scale: Optional[float] = None
    ):
        """Initialize detector.
        
        Args:
            threshold_method: Method for threshold ("percentile" or "absolute")
            threshold_value: Threshold value (percentile or absolute value)
            min_duration: Minimum duration in time steps
            spatial_scale: Minimum spatial scale in grid points
        """
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.min_duration = min_duration
        self.spatial_scale = spatial_scale
        
    def _compute_threshold(self, data: xr.DataArray) -> xr.DataArray:
        """Compute threshold for extreme event detection."""
        if self.threshold_method == "percentile":
            # Ensure data is properly chunked for quantile computation
            if data.chunks is not None:
                # Create safe chunks that respect data dimensions
                chunks = {}
                for dim in data.dims:
                    if dim == 'time':
                        chunks[dim] = data.sizes[dim]  # Keep time dimension whole
                    else:
                        # Use smaller chunks for spatial dimensions
                        chunks[dim] = min(data.sizes[dim], 20)
                data = data.chunk(chunks)
            
            # Compute quantile along time dimension
            q = float(self.threshold_value) / 100.0
            return data.quantile(q=q, dim='time')
        elif self.threshold_method == "absolute":
            return xr.full_like(data.isel(time=0), self.threshold_value)
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
    
    def _apply_duration_filter(self, events: xr.DataArray) -> xr.DataArray:
        """Filter events by minimum duration."""
        if self.min_duration <= 1:
            return events
            
        # Count consecutive True values along time dimension
        events_int = events.astype(int)
        rolled = events_int.rolling(time=self.min_duration, center=True)
        duration = rolled.sum()
        
        # Events must persist for minimum duration
        return duration >= self.min_duration
    
    def _apply_spatial_filter(self, events: xr.DataArray) -> xr.DataArray:
        """Filter events by spatial scale."""
        if not self.spatial_scale:
            return events
            
        # Convert spatial scale to grid points (rounded to nearest integer)
        scale_points = int(round(self.spatial_scale))
        
        # Simple spatial coherence check
        rolled_lat = events.rolling(latitude=scale_points, center=True)
        rolled_lon = events.rolling(longitude=scale_points, center=True)
        
        spatial_sum_lat = rolled_lat.sum()
        spatial_sum_lon = rolled_lon.sum()
        
        # Events must have minimum spatial extent
        return (spatial_sum_lat >= scale_points) & (spatial_sum_lon >= scale_points)
    
    def detect_events(self, data: xr.DataArray) -> xr.DataArray:
        """Detect extreme events in the data.
        
        Args:
            data: Input data array with dimensions (time, latitude, longitude)
            
        Returns:
            Boolean array marking extreme events
        """
        # Validate dimensions
        required_dims = {'time', 'latitude', 'longitude'}
        if not required_dims.issubset(data.dims):
            raise ValueError(
                f"Data must have dimensions {required_dims}, "
                f"got {data.dims}"
            )
        
        # Compute threshold
        threshold = self._compute_threshold(data)
        
        # Detect exceedances
        events = data > threshold
        
        # Apply duration filter
        events = self._apply_duration_filter(events)
        
        # Apply spatial filter if specified
        if self.spatial_scale:
            events = self._apply_spatial_filter(events)
        
        return events
