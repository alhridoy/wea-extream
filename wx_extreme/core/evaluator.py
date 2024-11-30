"""Module for evaluating extreme weather events."""

import numpy as np
import xarray as xr
from typing import Dict, List, Union, Optional, Tuple
from scipy import ndimage

def evaluate_extremes(
    data: xr.DataArray,
    events: xr.DataArray,
    reference: Optional[xr.DataArray] = None,
) -> Dict[str, float]:
    """Evaluate detected extreme events.
    
    Args:
        data: Input data array
        events: Boolean mask of detected events
        reference: Optional reference dataset
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    # Basic metrics
    results['frequency'] = float(events.mean())
    
    if events.sum() > 0:
        # Intensity metrics
        anomalies = data.where(events)
        results['mean_intensity'] = float(anomalies.mean())
        results['max_intensity'] = float(anomalies.max())
        
        # Duration metrics
        labeled, num = label_events(events)
        durations = []
        for i in range(1, num + 1):
            duration = (labeled == i).sum('time')
            durations.append(float(duration))
        
        results['mean_duration'] = float(np.mean(durations))
        results['max_duration'] = float(np.max(durations))
    else:
        results['mean_intensity'] = 0
        results['max_intensity'] = 0
        results['mean_duration'] = 0
        results['max_duration'] = 0
    
    # Compare with reference if provided
    if reference is not None:
        results['bias'] = float(events.sum() - reference.sum())
    
    return results

def label_events(events: xr.DataArray) -> Tuple[xr.DataArray, int]:
    """Label connected events in space and time.
    
    Args:
        events: Boolean mask of events
        
    Returns:
        Labeled array and number of events
    """
    # Create structure for 3D connectivity
    struct = ndimage.generate_binary_structure(events.ndim, 1)
    
    # Label events
    labeled, num = ndimage.label(events, structure=struct)
    
    # Convert to DataArray
    labeled = xr.DataArray(
        labeled,
        dims=events.dims,
        coords=events.coords
    )
    
    return labeled, num
