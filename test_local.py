"""Quick local test of WX-Extreme package."""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from wx_extreme.core.detector import ExtremeEventDetector

def test_with_synthetic_data():
    """Test package with synthetic temperature data."""
    # Create synthetic data
    times = pd.date_range('2020-01-01', '2020-01-31', freq='D')
    lats = np.linspace(30, 35, 10)
    lons = np.linspace(-100, -95, 10)
    
    # Create temperature field with a heat wave
    temp = np.random.normal(25, 5, size=(len(times), len(lats), len(lons)))
    temp[10:15] += 10  # Add heat wave
    
    # Create xarray DataArray
    data = xr.DataArray(
        temp,
        dims=['time', 'latitude', 'longitude'],
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons
        }
    )
    
    # Initialize detector
    detector = ExtremeEventDetector(
        threshold_method="percentile",
        threshold_value=95,
        min_duration=3
    )
    
    # Detect events
    print("Detecting extreme events...")
    events = detector.detect_events(data)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Plot temperature at peak of heat wave
    plt.subplot(121)
    data[12].plot()
    plt.title('Temperature (Day 12)')
    
    # Plot detected events
    plt.subplot(122)
    events[12].plot()
    plt.title('Detected Events (Day 12)')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()
    
    print(f"Total events detected: {events.sum().item()}")
    print("Results saved to test_results.png")

if __name__ == "__main__":
    test_with_synthetic_data() 