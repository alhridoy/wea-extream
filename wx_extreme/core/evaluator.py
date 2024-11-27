"""Module for evaluating extreme weather events."""

import numpy as np
import xarray as xr
from typing import Dict, List, Union, Optional

from wx_extreme.utils import time_utils, spatial_utils
from wx_extreme.core.detector import ExtremeEventDetector


def evaluate_extremes(
    data: xr.DataArray,
    detector: ExtremeEventDetector,
    reference: Optional[xr.DataArray] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate extreme events in the data.
    
    Args:
        data: Input data array
        detector: Extreme event detector instance
        reference: Optional reference data for comparison
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of evaluation metrics
    """
    if metrics is None:
        metrics = ['frequency', 'intensity', 'duration']
        
    results = {}
    
    # Create dataset with the variable name matching detector's thresholds
    var_name = list(detector.event_defs.keys())[0]  # Get first variable name
    ds = xr.Dataset({var_name: data})
    
    # Detect extreme events
    events = detector.detect_events(ds)
    event_mask = events[f"{var_name}_extreme"]
    
    # Count events (connected components in time)
    labeled_events, num_events = time_utils.label_events(event_mask)
    
    # Compute metrics
    if 'frequency' in metrics:
        results['frequency'] = num_events
        
    if 'intensity' in metrics:
        if num_events > 0:
            # Calculate intensities where events occur
            intensities = data.where(event_mask)
            results['mean_intensity'] = float(intensities.mean().values)
            results['max_intensity'] = float(intensities.max().values)
        else:
            results['mean_intensity'] = 0
            results['max_intensity'] = 0
            
    if 'duration' in metrics:
        if num_events > 0:
            # Calculate durations for each event
            durations = []
            for i in range(1, num_events + 1):
                event_duration = (labeled_events == i).sum()
                durations.append(event_duration)
            results['mean_duration'] = float(np.mean(durations))
            results['max_duration'] = float(np.max(durations))
        else:
            results['mean_duration'] = 0
            results['max_duration'] = 0
            
    # Compare with reference if provided
    if reference is not None:
        ref_ds = xr.Dataset({var_name: reference})
        ref_events = detector.detect_events(ref_ds)
        ref_mask = ref_events[f"{var_name}_extreme"]
        ref_labeled, ref_num = time_utils.label_events(ref_mask)
        
        results['bias'] = num_events - ref_num
        
        if num_events > 0 and ref_num > 0:
            # Calculate intensity bias
            ref_intensities = reference.where(ref_mask)
            results['intensity_bias'] = float(
                intensities.mean().values - ref_intensities.mean().values
            )
        else:
            results['intensity_bias'] = 0
            
    return results
