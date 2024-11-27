"""Extreme weather event detection module."""

import dataclasses
from typing import Dict, Optional, Union, List, Tuple
import numpy as np
import xarray as xr
from scipy import ndimage

from wx_extreme.utils import time_utils, spatial_utils


@dataclasses.dataclass
class EventDefinition:
    """Definition of an extreme weather event.
    
    Attributes:
        threshold: Threshold value for the variable
        window: Time window for event persistence (in hours)
        spatial_scale: Minimum spatial scale of the event (in km)
        relative_to_climatology: Whether threshold is relative to climatology
        percentile: If using percentile-based threshold
        combine_vars: List of other variables and conditions for compound events
    """
    
    threshold: float
    window: str = "24h"
    spatial_scale: float = 100.0  # km
    relative_to_climatology: bool = False
    percentile: Optional[float] = None
    combine_vars: Optional[List[Dict[str, float]]] = None


class ExtremeEventDetector:
    """Detector for extreme weather events.
    
    This class provides methods to detect various types of extreme weather events
    in meteorological data, including:
    1. Single-variable threshold exceedance
    2. Compound events involving multiple variables
    3. Spatially coherent extremes
    4. Persistent extreme events
    """
    
    def __init__(
        self,
        thresholds: Dict[str, Union[float, Dict]],
        climatology: Optional[xr.Dataset] = None,
    ):
        """Initialize the detector.
        
        Args:
            thresholds: Dictionary mapping variable names to their extreme event
                definitions. Can be simple threshold values or dictionaries with
                full event definitions.
            climatology: Optional climatology dataset for relative thresholds.
        """
        self.event_defs = {}
        for var, thresh in thresholds.items():
            if isinstance(thresh, (int, float)):
                self.event_defs[var] = EventDefinition(threshold=float(thresh))
            else:
                self.event_defs[var] = EventDefinition(**thresh)
        
        self.climatology = climatology

    def detect_events(
        self,
        dataset: xr.Dataset,
        return_severity: bool = False,
    ) -> xr.Dataset:
        """Detect extreme events in the dataset.
        
        Args:
            dataset: Input dataset to analyze
            return_severity: If True, return severity scores instead of binary mask
            
        Returns:
            Dataset with boolean masks (or severity scores) for detected events
        """
        events = {}
        
        for var, event_def in self.event_defs.items():
            if var not in dataset:
                continue
            
            # Get base threshold
            if event_def.relative_to_climatology and self.climatology is not None:
                clim_mean = self.climatology[var].mean()
                clim_std = self.climatology[var].std()
                threshold = clim_mean + event_def.threshold * clim_std
            elif event_def.percentile is not None:
                threshold = np.percentile(dataset[var], event_def.percentile)
            else:
                threshold = event_def.threshold
            
            # Detect threshold exceedance
            if return_severity:
                events[f"{var}_extreme"] = (dataset[var] - threshold) / threshold
                events[f"{var}_extreme"] = events[f"{var}_extreme"].where(
                    events[f"{var}_extreme"] > 0, 0
                )
            else:
                events[f"{var}_extreme"] = dataset[var] > threshold
            
            # Apply spatial coherence check
            if event_def.spatial_scale > 0:
                events[f"{var}_extreme"] = self._apply_spatial_filter(
                    events[f"{var}_extreme"],
                    event_def.spatial_scale,
                    dataset.latitude,
                    dataset.longitude,
                )
            
            # Check temporal persistence
            if event_def.window:
                window_size = time_utils.parse_time_window(event_def.window)
                events[f"{var}_extreme"] = self._check_persistence(
                    events[f"{var}_extreme"], window_size
                )
            
            # Handle compound events
            if event_def.combine_vars:
                for other_var in event_def.combine_vars:
                    if other_var["var"] not in dataset:
                        continue
                    other_cond = (
                        dataset[other_var["var"]] > other_var["threshold"]
                    )
                    events[f"{var}_extreme"] &= other_cond
        
        return xr.Dataset(events)

    def _apply_spatial_filter(
        self,
        event_mask: xr.DataArray,
        min_scale: float,
        latitude: xr.DataArray,
        longitude: xr.DataArray,
    ) -> xr.DataArray:
        """Apply spatial coherence filtering.
        
        Args:
            event_mask: Boolean mask of events
            min_scale: Minimum spatial scale in km
            latitude: Latitude coordinates
            longitude: Longitude coordinates
            
        Returns:
            Filtered event mask
        """
        # Convert spatial scale to grid points
        dx, dy = spatial_utils.get_grid_spacing(latitude, longitude)
        kernel_size = int(min_scale / min(dx, dy))
        
        # Apply morphological operations to enforce spatial scale
        struct = ndimage.generate_binary_structure(2, 2)
        struct = ndimage.iterate_structure(struct, kernel_size)
        
        filtered = xr.apply_ufunc(
            lambda x: ndimage.binary_opening(x, structure=struct),
            event_mask,
            input_core_dims=[["latitude", "longitude"]],
            output_core_dims=[["latitude", "longitude"]],
            vectorize=True,
        )
        
        return filtered

    def _check_persistence(
        self,
        event_mask: xr.DataArray,
        window_size: int,
    ) -> xr.DataArray:
        """Check temporal persistence of events.
        
        Args:
            event_mask: Boolean mask of events
            window_size: Size of rolling window in time steps
            
        Returns:
            Mask of persistent events
        """
        # Count number of event occurrences in rolling window
        counts = event_mask.rolling(time=window_size).sum()
        
        # Event must occur in at least 75% of the window
        threshold = 0.75 * window_size
        persistent = counts >= threshold
        
        # Align with original time axis
        persistent = persistent.shift(time=-(window_size // 2))
        
        return persistent

    def get_event_statistics(
        self,
        dataset: xr.Dataset,
        events: xr.Dataset,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for detected events.
        
        Args:
            dataset: Original data
            events: Detected events dataset
            
        Returns:
            Dictionary of event statistics
        """
        stats = {}
        
        for var in self.event_defs:
            if f"{var}_extreme" not in events:
                continue
                
            event_mask = events[f"{var}_extreme"]
            
            # Basic statistics
            stats[var] = {
                "frequency": float(event_mask.mean()),
                "spatial_coverage": float(
                    event_mask.mean(["latitude", "longitude"])
                ),
                "max_duration": int(
                    event_mask.astype(int)
                    .sum("time")
                    .max(["latitude", "longitude"])
                ),
            }
            
            # Intensity statistics if we have the original variable
            if var in dataset:
                extreme_values = dataset[var].where(event_mask)
                stats[var].update({
                    "mean_intensity": float(extreme_values.mean()),
                    "max_intensity": float(extreme_values.max()),
                })
        
        return stats
