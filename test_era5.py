"""Test WX-Extreme with ERA5 data."""

import cdsapi
import xarray as xr
from wx_extreme.core.detector import ExtremeEventDetector

def download_era5():
    """Download ERA5 temperature data."""
    c = cdsapi.Client()
    
    print("Downloading ERA5 data...")
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': '2m_temperature',
            'year': '2020',
            'month': '07',  # July
            'day': list(map(str, range(1, 32))),
            'time': ['00:00', '12:00'],  # Twice daily
            'area': [40, -100, 30, -90],  # Central US
            'format': 'netcdf'
        },
        'era5_temp.nc'
    )
    print("Download complete!")

def analyze_era5():
    """Analyze ERA5 data for extreme events."""
    # Load data
    print("Loading data...")
    ds = xr.open_dataset('era5_temp.nc')
    temp = ds['t2m'] - 273.15  # Convert to Celsius
    
    # Initialize detector
    detector = ExtremeEventDetector(
        threshold_method="percentile",
        threshold_value=95,
        min_duration=2  # At least 1 day (2 time steps)
    )
    
    # Detect events
    print("Detecting extreme events...")
    events = detector.detect_events(temp)
    
    # Print summary
    total_events = events.sum().item()
    event_days = events.any(dim=['latitude', 'longitude']).sum().item()
    
    print(f"\nResults:")
    print(f"Total event grid points: {total_events}")
    print(f"Days with events: {event_days}")
    
    # Save results
    events.to_netcdf('era5_events.nc')
    print("\nResults saved to era5_events.nc")

if __name__ == "__main__":
    try:
        download_era5()
        analyze_era5()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nNote: ERA5 data requires CDS API key. Set up following:")
        print("https://cds.climate.copernicus.eu/api-how-to") 