"""Tests for extreme event evaluator."""

import numpy as np
import xarray as xr
import pandas as pd
import pytest
from wx_extreme.core.evaluator import evaluate_extremes, label_events


def create_test_events():
    """Create synthetic events for testing."""
    times = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    data = np.random.normal(25, 5, size=len(times))
    events = np.zeros_like(data, dtype=bool)
    
    # Add two events
    events[180:190] = True  # 10-day event
    events[250:255] = True  # 5-day event
    
    data = xr.DataArray(data, dims=['time'], coords={'time': times})
    events = xr.DataArray(events, dims=['time'], coords={'time': times})
    
    return data, events


def test_evaluate_basic():
    """Test basic evaluation metrics."""
    data, events = create_test_events()
    metrics = evaluate_extremes(data, events)
    
    assert 'frequency' in metrics
    assert 'mean_intensity' in metrics
    assert 'max_intensity' in metrics
    assert 'mean_duration' in metrics
    assert 'max_duration' in metrics
    
    assert metrics['frequency'] > 0
    assert metrics['mean_duration'] > 0
    assert metrics['max_duration'] >= metrics['mean_duration']


def test_evaluate_no_events():
    """Test evaluation with no events."""
    data, events = create_test_events()
    events = events * False  # Set all to False
    
    metrics = evaluate_extremes(data, events)
    
    assert metrics['frequency'] == 0
    assert metrics['mean_intensity'] == 0
    assert metrics['max_intensity'] == 0
    assert metrics['mean_duration'] == 0
    assert metrics['max_duration'] == 0


def test_evaluate_with_reference():
    """Test evaluation against reference dataset."""
    data, events = create_test_events()
    reference = events.copy()
    reference[100:105] = True  # Add different event in reference
    
    metrics = evaluate_extremes(data, events, reference)
    
    assert 'bias' in metrics
    assert isinstance(metrics['bias'], float)


def test_label_events():
    """Test event labeling."""
    _, events = create_test_events()
    labeled, num = label_events(events)
    
    assert isinstance(labeled, xr.DataArray)
    assert num == 2  # Should find two events
    assert labeled.max() == 2
    assert (labeled > 0).sum() == events.sum()
