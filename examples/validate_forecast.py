"""Validate Pangu-Weather forecasts against ERA5 data."""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import dask
from wx_extreme.core.detector import ExtremeEventDetector
from wx_extreme.core.metrics import (
    get_panguweather_t2_forecasts, 
    load_era5_data,
    calculate_forecast_metrics
)

def evaluate_forecast_skill():
    """Evaluate forecast skill at different lead times."""
    print("Loading Pangu-Weather forecasts...")
    forecast_ds = get_panguweather_t2_forecasts()
    
    print("Loading ERA5 data...")
    era5_ds = load_era5_data()
    
    results = []
    for init_time in forecast_ds.init_time:
        print(f"\nProcessing forecast initialized at {init_time.values}")
        forecast = forecast_ds.sel(init_time=init_time)
        
        for fcst_hour in forecast.fcst_hour:
            # Get valid time
            valid_time = pd.Timestamp(init_time.values) + pd.Timedelta(hours=float(fcst_hour))
            if valid_time not in era5_ds.time:
                continue
            
            # Get forecast and analysis data
            f = forecast.sel(fcst_hour=fcst_hour)
            a = era5_ds.sel(time=valid_time)
            
            # Ensure matching coordinates
            common_lats = np.intersect1d(
                np.round(f.latitude.values, 4),
                np.round(a.latitude.values, 4)
            )
            common_lons = np.intersect1d(
                np.round(f.longitude.values, 4),
                np.round(a.longitude.values, 4)
            )
            
            # Select common coordinates
            f = f.sel(
                latitude=common_lats,
                longitude=common_lons,
                method='nearest'
            )
            a = a.sel(
                latitude=common_lats,
                longitude=common_lons,
                method='nearest'
            )
            
            # Convert temperatures to Celsius
            f = f.t2 - 273.15
            a = a['2m_temperature'] - 273.15
            
            # Add threshold for extreme events
            f.attrs['threshold'] = np.percentile(a, 95)
            
            # Calculate comprehensive metrics
            metrics = calculate_forecast_metrics(f, a)
            metrics.update({
                'init_time': pd.Timestamp(init_time.values),
                'fcst_hour': float(fcst_hour),
                'valid_time': valid_time,
            })
            
            results.append(metrics)
    
    return pd.DataFrame(results)

def plot_forecast_skill(skill_df):
    """Plot forecast skill metrics by forecast hour."""
    metric_groups = {
        'Basic Error Metrics': ['bias', 'rmse', 'mae'],
        'Pattern Metrics': ['correlation', 'acc'],
        'Skill Scores': ['mse_skill_score', 'murphy_score'],
        'Extreme Event Metrics': ['pod', 'far', 'csi', 'hss', 'ets']
    }
    
    n_groups = len(metric_groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 4*n_groups))
    
    for ax, (group_name, metrics) in zip(axes, metric_groups.items()):
        for metric in metrics:
            if metric in skill_df.columns:
                ax.plot(skill_df.fcst_hour, skill_df[metric], 
                       label=metric, alpha=0.7)
        
        ax.set_xlabel('Forecast Hour')
        ax.set_ylabel('Score')
        ax.set_title(group_name)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    return fig

def main():
    """Main function."""
    try:
        print("Evaluating forecast skill...")
        skill_df = evaluate_forecast_skill()
        
        print("\nPlotting results...")
        fig = plot_forecast_skill(skill_df)
        fig.savefig('forecast_skill.png', dpi=300, bbox_inches='tight')
        
        print("\nResults saved to forecast_skill.png")
        print("\nSkill metrics summary:")
        
        # Print summary statistics for key metrics
        key_metrics = ['bias', 'rmse', 'correlation', 'acc', 'pod', 'far', 'csi', 'hss']
        summary = skill_df.groupby('fcst_hour')[key_metrics].mean()
        print(summary.round(4))
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 