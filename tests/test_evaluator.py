"""Tests for extreme event evaluator."""

import numpy as np
import xarray as xr
import pandas as pd
import pytest
from datetime import datetime, timedelta

from wx_extreme.core.evaluator import evaluate_extremes
from wx_extreme.core.detector import ExtremeEventDetector


def test_evaluate_extremes():
    """Test extreme event evaluation."""
    # Create test data
    times = pd.date_range('2020-01-01', '2020-01-10', freq='1h')
    data = xr.DataArray(
        np.random.randn(len(times)),
        dims=['time'],
        coords={'time': times}
    )
    
    # Create an extreme event (high values)
    data[24:48] = 3.0  # One day of extreme values
    
    # Create detector
    detector = ExtremeEventDetector(
        thresholds={'temperature': 2.0}
    )
    
    # Evaluate without reference
    metrics = evaluate_extremes(data, detector)
    
    assert isinstance(metrics, dict)
    assert metrics['frequency'] > 0
    assert metrics['mean_intensity'] > 0
    assert metrics['max_intensity'] > 0
    assert metrics['mean_duration'] > 0
    assert metrics['max_duration'] > 0
    
    # Test with reference data
    ref_data = data.copy()
    ref_data[24:36] = 3.0  # Shorter event in reference
    
    metrics_with_ref = evaluate_extremes(data, detector, reference=ref_data)
    
    assert 'bias' in metrics_with_ref
    assert 'intensity_bias' in metrics_with_ref
    assert metrics_with_ref['bias'] >= 0  # Should detect more/longer events in data


def test_evaluate_extremes_no_events():
    """Test evaluation when no events are found."""
    times = pd.date_range('2020-01-01', '2020-01-10', freq='1h')
    data = xr.DataArray(
        np.zeros(len(times)),  # All zeros, no extremes
        dims=['time'],
        coords={'time': times}
    )
    
    detector = ExtremeEventDetector(
        thresholds={'temperature': 2.0}
    )
    
    metrics = evaluate_extremes(data, detector)
    
    assert metrics['frequency'] == 0
    assert metrics['mean_intensity'] == 0
    assert metrics['max_intensity'] == 0
    assert metrics['mean_duration'] == 0
    assert metrics['max_duration'] == 0
