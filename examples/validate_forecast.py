"""Validate Pangu-Weather forecasts against ERA5 data."""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import dask
from wx_extreme.core.detector import ExtremeEventDetector

def get_optimal_chunks(ds, is_forecast=False):
    """Get optimal chunk sizes based on dataset."""
    chunks = {}
    for dim, size in ds.dims.items():
        if dim in ['time', 'init_time', 'fcst_hour']:
            chunks[dim] = min(size, 50)  # Max 50 time steps per chunk
        elif dim in ['latitude', 'longitude']:
            chunks[dim] = min(max(size // 4, 10), 50)  # Between 10 and 50
        else:
            chunks[dim] = size
    return chunks

def load_forecast_data():
    """Load Pangu-Weather forecast data."""
    return xr.open_dataset(
        "gcs://brightband-share/heatwave/pangu-weather-forecasts.t2.zarr",
        backend_kwargs=dict(storage_options=dict(token='anon')),
        engine="zarr"
    )

def load_era5_data():
    """Load ERA5 analysis data."""
    return xr.open_dataset(
        "gcs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        storage_options=dict(token='anon'),
        engine='zarr'
    )[['2m_temperature']].sel(time=slice("2021-06-21", "2021-07-10"))

def preprocess_data(forecast_ds, era5_ds):
    """Preprocess and align forecast and analysis data."""
    print("\nPreprocessing data...")
    
    # Convert temperatures to Celsius
    if 't2' in forecast_ds:
        # Handle Pangu-Weather forecast data structure
        print("Processing Pangu-Weather forecast...")
        
        # Select the first initialization time
        forecast = forecast_ds['t2'].isel(init_time=0)
        
        # Create time index from forecast hours
        base_time = pd.Timestamp('2021-06-21')
        times = [base_time + pd.Timedelta(hours=float(h)) for h in forecast.fcst_hour.values]
        
        # Create new DataArray with time dimension
        forecast = xr.DataArray(
            forecast.values,
            dims=['time', 'latitude', 'longitude'],
            coords={
                'time': times,
                'latitude': forecast.latitude,
                'longitude': forecast.longitude
            }
        )
        
        # Convert to Celsius
        forecast = forecast - 273.15
    else:
        forecast = forecast_ds['2m_temperature'] - 273.15
    
    # Process ERA5 data
    print("Processing ERA5 analysis...")
    analysis = era5_ds['2m_temperature'] - 273.15
    
    # Print coordinate ranges
    print("\nCoordinate ranges:")
    print("Forecast latitudes:", forecast.latitude.values.min(), "to", forecast.latitude.values.max())
    print("Analysis latitudes:", analysis.latitude.values.min(), "to", analysis.latitude.values.max())
    print("Forecast longitudes:", forecast.longitude.values.min(), "to", forecast.longitude.values.max())
    print("Analysis longitudes:", analysis.longitude.values.min(), "to", analysis.longitude.values.max())
    
    # Ensure matching coordinates
    print("\nAligning spatial coordinates...")
    # Find common coordinates with a small tolerance for floating point differences
    common_lats = np.intersect1d(
        np.round(forecast.latitude.values, 4),
        np.round(analysis.latitude.values, 4)
    )
    common_lons = np.intersect1d(
        np.round(forecast.longitude.values, 4),
        np.round(analysis.longitude.values, 4)
    )
    
    if len(common_lats) == 0 or len(common_lons) == 0:
        raise ValueError(
            "No common coordinates found between forecast and analysis data.\n"
            "This might be due to different coordinate systems or region selection.\n"
            "Try adjusting the region boundaries or coordinate systems."
        )
    
    print(f"Found {len(common_lats)} common latitudes and {len(common_lons)} common longitudes")
    
    # Select nearest coordinates
    forecast = forecast.sel(
        latitude=common_lats,
        longitude=common_lons,
        method='nearest'
    )
    analysis = analysis.sel(
        latitude=common_lats,
        longitude=common_lons,
        method='nearest'
    )
    
    # Convert to daily means
    print("Computing daily means...")
    forecast = forecast.resample(time='1D').mean()
    analysis = analysis.resample(time='1D').mean()
    
    # Ensure matching time periods
    print("Aligning temporal coordinates...")
    common_times = pd.DatetimeIndex(np.intersect1d(forecast.time, analysis.time))
    if len(common_times) == 0:
        raise ValueError("No common time periods found between forecast and analysis data")
    
    print(f"Found {len(common_times)} common time periods")
    forecast = forecast.sel(time=common_times)
    analysis = analysis.sel(time=common_times)
    
    # Use consistent chunks for both datasets
    chunks = {
        'time': len(common_times),  # Keep time dimension whole
        'latitude': max(min(len(common_lats), 20), 1),  # Ensure at least size 1
        'longitude': max(min(len(common_lons), 20), 1)  # Ensure at least size 1
    }
    
    print(f"Using chunks: {chunks}")
    forecast = forecast.chunk(chunks)
    analysis = analysis.chunk(chunks)
    
    print("Computing final arrays...")
    return forecast.compute(), analysis.compute()

def calculate_metrics(forecast, analysis):
    """Calculate forecast verification metrics."""
    print("\nCalculating verification metrics...")
    
    # Compute metrics
    metrics = {}
    
    # Basic error metrics
    diff = forecast - analysis
    metrics['bias'] = float(diff.mean().values)
    metrics['mae'] = float(abs(diff).mean().values)
    metrics['rmse'] = float(np.sqrt((diff ** 2).mean().values))
    
    # Pattern correlation
    metrics['pattern_corr'] = float(
        xr.corr(forecast, analysis, dim=['latitude', 'longitude']).mean().values
    )
    
    print("\nForecast Metrics:")
    print(f"Bias: {metrics['bias']:.2f}°C")
    print(f"MAE: {metrics['mae']:.2f}°C")
    print(f"RMSE: {metrics['rmse']:.2f}°C")
    print(f"Pattern Correlation: {metrics['pattern_corr']:.3f}")
    
    return metrics

def detect_extremes(forecast, analysis, threshold=95):
    """Detect extreme events in both forecast and analysis."""
    print(f"\nDetecting extreme events (threshold: {threshold}th percentile)...")
    
    # Ensure data is properly chunked for event detection
    chunks = {
        'time': -1,  # Keep time dimension whole for quantile calculation
        'latitude': min(forecast.sizes['latitude'], 20),
        'longitude': min(forecast.sizes['longitude'], 20)
    }
    
    forecast = forecast.chunk(chunks)
    analysis = analysis.chunk(chunks)
    
    detector = ExtremeEventDetector(
        threshold_method="percentile",
        threshold_value=threshold,
        min_duration=3
    )
    
    # Process in chunks to save memory
    print("Processing forecast events...")
    forecast_events = detector.detect_events(forecast)
    
    print("Processing analysis events...")
    analysis_events = detector.detect_events(analysis)
    
    # Calculate metrics
    print("Computing event detection metrics...")
    hits = int((forecast_events & analysis_events).sum().values)
    misses = int((~forecast_events & analysis_events).sum().values)
    false_alarms = int((forecast_events & ~analysis_events).sum().values)
    
    # Calculate scores with safety checks
    total = hits + misses
    pod = hits / total if total > 0 else 0
    
    total = hits + false_alarms
    far = false_alarms / total if total > 0 else 0
    
    total = hits + misses + false_alarms
    csi = hits / total if total > 0 else 0
    
    print("\nExtreme Event Detection Metrics:")
    print(f"Hits: {hits}")
    print(f"Misses: {misses}")
    print(f"False Alarms: {false_alarms}")
    print(f"Probability of Detection: {pod:.2f}")
    print(f"False Alarm Ratio: {far:.2f}")
    print(f"Critical Success Index: {csi:.2f}")
    
    return forecast_events, analysis_events

def plot_comparison(forecast, analysis, forecast_events, analysis_events, time_idx):
    """Create comparison plots."""
    print("\nCreating comparison plots...")
    
    fig = plt.figure(figsize=(15, 10))
    proj = ccrs.PlateCarree()
    
    # Temperature comparison
    ax1 = plt.subplot(221, projection=proj)
    forecast.isel(time=time_idx).plot(
        ax=ax1,
        transform=proj,
        cmap='RdYlBu_r',
        vmin=15,
        vmax=45,
        cbar_kwargs={'label': 'Temperature (°C)'}
    )
    ax1.coastlines()
    ax1.set_title('Forecast Temperature')
    
    ax2 = plt.subplot(222, projection=proj)
    analysis.isel(time=time_idx).plot(
        ax=ax2,
        transform=proj,
        cmap='RdYlBu_r',
        vmin=15,
        vmax=45,
        cbar_kwargs={'label': 'Temperature (°C)'}
    )
    ax2.coastlines()
    ax2.set_title('Analysis Temperature')
    
    # Extreme event comparison
    ax3 = plt.subplot(223, projection=proj)
    forecast_events.isel(time=time_idx).plot(
        ax=ax3,
        transform=proj,
        cmap='Reds',
        cbar_kwargs={'label': 'Extreme Event'}
    )
    ax3.coastlines()
    ax3.set_title('Forecast Extreme Events')
    
    ax4 = plt.subplot(224, projection=proj)
    analysis_events.isel(time=time_idx).plot(
        ax=ax4,
        transform=proj,
        cmap='Reds',
        cbar_kwargs={'label': 'Extreme Event'}
    )
    ax4.coastlines()
    ax4.set_title('Analysis Extreme Events')
    
    plt.tight_layout()
    return fig

def evaluate_forecast_skill(forecast_ds, era5_ds, lead_times=None):
    """Evaluate forecast skill at different lead times."""
    if lead_times is None:
        lead_times = range(24, 241, 24)  # 1-10 days in 24h steps
    
    results = []
    for init_time in forecast_ds.init_time:
        print(f"\nProcessing forecast initialized at {init_time.values}")
        forecast = forecast_ds.sel(init_time=init_time)
        
        for lead_time in lead_times:
            # Select data at this lead time
            fcst_time = pd.Timestamp(init_time.values) + pd.Timedelta(hours=lead_time)
            if fcst_time not in era5_ds.time:
                continue
                
            f = forecast.sel(fcst_hour=lead_time)
            a = era5_ds.sel(time=fcst_time)
            
            # Normalize coordinates
            f, a = normalize_coords(f, a)
            
            # Calculate metrics
            bias = float((f - a).mean().values)
            rmse = float(np.sqrt(((f - a) ** 2).mean().values))
            corr = float(xr.corr(f, a, dim=['latitude', 'longitude']).values)
            
            results.append({
                'init_time': pd.Timestamp(init_time.values),
                'lead_time': lead_time,
                'valid_time': fcst_time,
                'bias': bias,
                'rmse': rmse,
                'correlation': corr
            })
    
    return pd.DataFrame(results)

def plot_forecast_skill(skill_df):
    """Plot forecast skill metrics by lead time."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    metrics = ['bias', 'rmse', 'correlation']
    titles = ['Bias (°C)', 'RMSE (°C)', 'Pattern Correlation']
    
    for ax, metric, title in zip(axes, metrics, titles):
        for init_time in skill_df.init_time.unique():
            df = skill_df[skill_df.init_time == init_time]
            ax.plot(df.lead_time, df[metric], 
                   alpha=0.5, label=init_time.strftime('%Y-%m-%d %H:%M'))
        
        ax.set_xlabel('Lead Time (hours)')
        ax.set_ylabel(title)
        ax.grid(True)
    
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def main():
    """Main function."""
    try:
        print("Loading and processing data...")
        forecast_ds = load_forecast_data()
        era5_ds = load_era5_data()
        
        # Evaluate forecast skill by lead time
        skill_df = evaluate_forecast_skill(forecast_ds, era5_ds)
        
        # Plot skill metrics
        fig_skill = plot_forecast_skill(skill_df)
        fig_skill.savefig('forecast_skill.png', dpi=300, bbox_inches='tight')
        
        # Process single forecast for event detection
        init_time = forecast_ds.init_time[0]
        forecast_ds = forecast_ds.sel(init_time=init_time)
        
        # Preprocess data
        forecast, analysis = preprocess_data(forecast_ds, era5_ds)
        
        # Calculate metrics
        metrics = calculate_metrics(forecast, analysis)
        
        # Detect and evaluate extreme events
        forecast_events, analysis_events = detect_extremes(forecast, analysis)
        
        # Create visualization
        event_time = int(analysis_events.sum(dim=['latitude', 'longitude']).argmax())
        fig = plot_comparison(forecast, analysis, forecast_events, analysis_events, event_time)
        
        print("Saving plots...")
        fig.savefig('forecast_validation.png', dpi=300, bbox_inches='tight')
        print("Complete! Results saved to forecast_validation.png and forecast_skill.png")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 