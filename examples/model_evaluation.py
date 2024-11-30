"""
WX-Extreme Tutorial

This script demonstrates how to use the WX-Extreme package for detecting and evaluating 
extreme weather events.

Dataset Requirements:
- Dimensions: time, latitude, longitude
- Data Variables: Temperature, pressure levels (for physical consistency evaluation)
- Format: NetCDF/xarray compatible
- Coordinates: 
  - Latitude: degrees North (-90 to 90)
  - Longitude: degrees East (-180 to 180)
  - Time: datetime format

Common sources:
- ERA5 reanalysis
- CMIP6 model outputs
- Weather model forecasts (e.g., GFS, ECMWF)
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from wx_extreme.core.detector import ExtremeEventDetector
from wx_extreme.core.evaluator import evaluate_extremes
from wx_extreme.core.metrics import MLModelMetrics
from wx_extreme.utils import plot_utils

def create_sample_data():
    """Create synthetic dataset that mimics real weather data."""
    times = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    lats = np.linspace(30, 40, 20)
    lons = np.linspace(-100, -90, 20)
    
    # Create temperature field with seasonal cycle and heat waves
    time_grid, lat_grid, lon_grid = np.meshgrid(
        np.arange(len(times)), 
        lats, 
        lons, 
        indexing='ij'
    )
    
    # Base temperature with seasonal cycle
    seasonal_cycle = 25 + 15 * np.sin(2 * np.pi * time_grid / 365)
    
    # Add latitude dependence
    temp = seasonal_cycle - 0.5 * (lat_grid - 35)
    
    # Add random variations
    temp += np.random.normal(0, 2, temp.shape)
    
    # Add heat waves
    temp[180:190] += 10  # Summer heat wave
    temp[250:255] += 8   # Fall heat wave
    
    # Create pressure levels
    pressure = 1000 - lat_grid/10
    pressure = pressure * 100  # Convert to Pa
    
    # Create DataArrays
    temp_da = xr.DataArray(
        temp,
        dims=['time', 'latitude', 'longitude'],
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons
        },
        attrs={'units': 'celsius', 'long_name': 'Temperature'}
    )
    
    pressure_da = xr.DataArray(
        pressure,
        dims=['time', 'latitude', 'longitude'],
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons
        },
        attrs={'units': 'Pa', 'long_name': 'Pressure'}
    )
    
    return temp_da, pressure_da

def main():
    # Create sample data
    print("Creating sample data...")
    temperature, pressure = create_sample_data()
    
    # Initialize detector
    print("\nDetecting extreme events...")
    detector = ExtremeEventDetector(
        threshold_method="percentile",  # Use percentile-based threshold
        threshold_value=95,             # 95th percentile
        min_duration=3,                 # Minimum 3 days
        spatial_scale=2.0              # Require spatial coherence
    )
    
    # Detect events
    events = detector.detect_events(temperature)
    
    # Plot events
    plt.figure(figsize=(12, 6))
    plot_utils.plot_extreme_event_heatmap(
        events.isel(time=180),  # Show events on day 180
        temperature.isel(time=180)
    )
    plt.title('Detected Heat Events - Day 180')
    plt.savefig('heat_events.png')
    plt.close()
    
    # Evaluate events
    print("\nEvaluating detected events...")
    metrics = evaluate_extremes(temperature, events)
    print("\nEvent Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    # Model comparison
    print("\nComparing model forecasts...")
    forecast = temperature + np.random.normal(1, 1, temperature.shape)
    
    # Calculate scores
    pattern_score = MLModelMetrics.pattern_prediction_score(
        forecast,
        temperature,
        spatial_scale=1.0
    )
    
    phys_score = MLModelMetrics.physical_consistency_score(
        forecast,
        pressure,
        temperature
    )
    
    print(f"\nModel Evaluation Scores:")
    print(f"Pattern Prediction Score: {pattern_score:.2f}")
    print(f"Physical Consistency Score: {phys_score:.2f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plot_utils.plot_model_comparison_heatmap(
        forecast.isel(time=180),
        temperature.isel(time=180),
        metric='bias',
        title='Model Bias - Day 180'
    )
    plt.savefig('model_comparison.png')
    plt.close()

if __name__ == "__main__":
    main() 