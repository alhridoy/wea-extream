"""Core metrics for evaluating weather forecasts."""

import numpy as np
import xarray as xr
from scipy import stats

def calculate_forecast_metrics(forecast: xr.DataArray, 
                             observation: xr.DataArray,
                             dim: list = ['latitude', 'longitude']) -> dict:
    """Calculate comprehensive forecast verification metrics.
    
    Args:
        forecast: Forecast data
        observation: Observation/analysis data
        dim: Dimensions to calculate metrics over
        
    Returns:
        Dictionary of metrics
    """
    # Basic error metrics
    error = forecast - observation
    abs_error = np.abs(error)
    squared_error = error ** 2
    
    metrics = {
        # Basic metrics
        'bias': float(error.mean().values),
        'mae': float(abs_error.mean().values),
        'rmse': float(np.sqrt(squared_error.mean().values)),
        'mse': float(squared_error.mean().values),
        
        # Pattern metrics
        'correlation': float(xr.corr(forecast, observation, dim=dim).values),
        'pattern_correlation': float(xr.corr(forecast, observation, dim=dim).values),
        
        # Skill scores
        'mse_skill_score': 1 - float(squared_error.mean().values) / float(observation.var().values),
        'murphy_score': float(1 - squared_error.mean().values / observation.var().values),
        
        # Distribution metrics
        'mean_error': float(error.mean().values),
        'std_error': float(error.std().values),
        'max_error': float(abs_error.max().values),
        'min_error': float(abs_error.min().values),
        
        # Percentile metrics
        'p90_error': float(abs_error.quantile(0.9).values),
        'p95_error': float(abs_error.quantile(0.95).values),
        'p99_error': float(abs_error.quantile(0.99).values),
        
        # Anomaly correlation
        'acc': float(xr.corr(
            forecast - forecast.mean(dim=dim),
            observation - observation.mean(dim=dim),
            dim=dim
        ).values),
    }
    
    # Add categorical metrics if threshold is provided
    if 'threshold' in forecast.attrs:
        threshold = forecast.attrs['threshold']
        f_binary = forecast > threshold
        o_binary = observation > threshold
        
        hits = float(((f_binary == 1) & (o_binary == 1)).sum().values)
        misses = float(((f_binary == 0) & (o_binary == 1)).sum().values)
        false_alarms = float(((f_binary == 1) & (o_binary == 0)).sum().values)
        correct_negatives = float(((f_binary == 0) & (o_binary == 0)).sum().values)
        
        total = hits + misses + false_alarms + correct_negatives
        
        metrics.update({
            # Contingency table metrics
            'hits': hits,
            'misses': misses,
            'false_alarms': false_alarms,
            'correct_negatives': correct_negatives,
            
            # Skill scores
            'pod': hits / (hits + misses) if (hits + misses) > 0 else np.nan,
            'far': false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan,
            'csi': hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan,
            'accuracy': (hits + correct_negatives) / total if total > 0 else np.nan,
            'bias_score': (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else np.nan,
            'hss': calculate_heidke_skill_score(hits, misses, false_alarms, correct_negatives),
            'ets': calculate_equitable_threat_score(hits, misses, false_alarms, correct_negatives)
        })
    
    return metrics

def calculate_heidke_skill_score(hits, misses, false_alarms, correct_negatives):
    """Calculate Heidke Skill Score."""
    total = hits + misses + false_alarms + correct_negatives
    if total == 0:
        return np.nan
        
    expected_hits = ((hits + misses) * (hits + false_alarms)) / total
    expected_correct = (expected_hits + 
                       ((misses + correct_negatives) * 
                        (false_alarms + correct_negatives)) / total)
    
    if (total - expected_correct) == 0:
        return np.nan
        
    return (hits + correct_negatives - expected_correct) / (total - expected_correct)

def calculate_equitable_threat_score(hits, misses, false_alarms, correct_negatives):
    """Calculate Equitable Threat Score."""
    total = hits + misses + false_alarms + correct_negatives
    if total == 0:
        return np.nan
        
    hits_random = ((hits + misses) * (hits + false_alarms)) / total
    
    if (hits + misses + false_alarms - hits_random) == 0:
        return np.nan
        
    return (hits - hits_random) / (hits + misses + false_alarms - hits_random)

def get_panguweather_t2_forecasts() -> xr.Dataset:
    """Retrieve an xarray handle to a prepared data cube of Pangu-Weather forecasts."""
    return xr.open_dataset(
        "gcs://brightband-share/heatwave/pangu-weather-forecasts.t2.zarr",
        backend_kwargs=dict(
            storage_options=dict(
                token='anon'
            )
        ),
        engine="zarr"
    )

def load_era5_data():
    """Load ERA5 analysis data."""
    return xr.open_dataset(
        "gcs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        storage_options=dict(token='anon'),
        engine='zarr'
    )[['2m_temperature']].sel(time=slice("2021-06-21", "2021-07-10"))
