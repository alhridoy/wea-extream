"""Metrics for evaluating extreme weather events and ML models."""

from typing import Dict, List, Optional, Union
import numpy as np
import xarray as xr
from scipy import stats


class ExtremeMetrics:
    """Metrics for evaluating extreme weather events."""
    
    @staticmethod
    def frequency_bias(
        forecast: xr.DataArray,
        observed: xr.DataArray,
        threshold: float,
        method: str = "absolute"
    ) -> float:
        """Calculate frequency bias in extreme event detection."""
        if method == "percentile":
            f_thresh = np.percentile(forecast, threshold)
            o_thresh = np.percentile(observed, threshold)
        else:
            f_thresh = o_thresh = threshold
            
        forecast_events = (forecast > f_thresh).sum()
        observed_events = (observed > o_thresh).sum()
        
        return float(forecast_events / observed_events) if observed_events > 0 else np.nan
    
    @staticmethod
    def intensity_bias(
        forecast: xr.DataArray,
        observed: xr.DataArray,
        threshold: float,
        method: str = "absolute"
    ) -> float:
        """Calculate intensity bias in extreme events."""
        if method == "percentile":
            f_thresh = np.percentile(forecast, threshold)
            o_thresh = np.percentile(observed, threshold)
        else:
            f_thresh = o_thresh = threshold
            
        f_extremes = forecast.where(forecast > f_thresh)
        o_extremes = observed.where(observed > o_thresh)
        
        f_mean = float(f_extremes.mean()) if not np.isnan(f_extremes).all() else np.nan
        o_mean = float(o_extremes.mean()) if not np.isnan(o_extremes).all() else np.nan
        
        return f_mean / o_mean if o_mean != 0 else np.nan


class PhysicalConsistencyMetrics:
    """Metrics for evaluating physical consistency."""
    
    @staticmethod
    def hydrostatic_balance_error(
        temperature: xr.DataArray,
        pressure: xr.DataArray,
        height: xr.DataArray
    ) -> float:
        """Calculate error in hydrostatic balance."""
        g = 9.81  # gravitational acceleration
        R = 287.0  # gas constant for dry air
        
        # Calculate pressure gradient
        dp_dz = pressure.differentiate('level') / height.differentiate('level')
        
        # Calculate density
        rho = pressure / (R * temperature)
        
        # Calculate hydrostatic balance error
        error = dp_dz + rho * g
        
        return float(np.abs(error).mean())
    
    @staticmethod
    def mass_conservation_error(
        wind_u: xr.DataArray,
        wind_v: xr.DataArray,
        density: xr.DataArray
    ) -> float:
        """Calculate mass conservation error."""
        # Calculate divergence
        div = wind_u.differentiate('longitude') + wind_v.differentiate('latitude')
        
        # Calculate mass conservation error
        error = div * density
        
        return float(np.abs(error).mean())


class PatternMetrics:
    """Metrics for evaluating spatial and temporal patterns."""
    
    @staticmethod
    def spatial_correlation(
        forecast: xr.DataArray,
        observed: xr.DataArray,
        threshold: float,
        method: str = "absolute"
    ) -> float:
        """Calculate spatial correlation of extreme events."""
        if method == "percentile":
            f_thresh = np.percentile(forecast, threshold)
            o_thresh = np.percentile(observed, threshold)
        else:
            f_thresh = o_thresh = threshold
            
        f_binary = (forecast > f_thresh).astype(float)
        o_binary = (observed > o_thresh).astype(float)
        
        return float(stats.pearsonr(f_binary.values.flatten(), o_binary.values.flatten())[0])
    
    @staticmethod
    def pattern_similarity(
        forecast: xr.DataArray,
        observed: xr.DataArray,
        threshold: float,
        method: str = "absolute"
    ) -> float:
        """Calculate pattern similarity score."""
        if method == "percentile":
            f_thresh = np.percentile(forecast, threshold)
            o_thresh = np.percentile(observed, threshold)
        else:
            f_thresh = o_thresh = threshold
            
        f_binary = (forecast > f_thresh).astype(float)
        o_binary = (observed > o_thresh).astype(float)
        
        intersection = np.logical_and(f_binary, o_binary).sum()
        union = np.logical_or(f_binary, o_binary).sum()
        
        return float(intersection / union) if union > 0 else np.nan


class ImpactMetrics:
    """Metrics for evaluating impact of extreme events."""
    
    @staticmethod
    def critical_success_index(
        forecast: xr.DataArray,
        observed: xr.DataArray,
        threshold: float,
        method: str = "absolute"
    ) -> float:
        """Calculate Critical Success Index for extreme events."""
        if method == "percentile":
            f_thresh = np.percentile(forecast, threshold)
            o_thresh = np.percentile(observed, threshold)
        else:
            f_thresh = o_thresh = threshold
            
        f_binary = forecast > f_thresh
        o_binary = observed > o_thresh
        
        hits = np.logical_and(f_binary, o_binary).sum()
        misses = np.logical_and(~f_binary, o_binary).sum()
        false_alarms = np.logical_and(f_binary, ~o_binary).sum()
        
        return float(hits / (hits + misses + false_alarms)) if (hits + misses + false_alarms) > 0 else np.nan
    
    @staticmethod
    def extreme_dependency_score(
        forecast: xr.DataArray,
        observed: xr.DataArray,
        threshold: float,
        method: str = "absolute"
    ) -> float:
        """Calculate Extreme Dependency Score."""
        if method == "percentile":
            f_thresh = np.percentile(forecast, threshold)
            o_thresh = np.percentile(observed, threshold)
        else:
            f_thresh = o_thresh = threshold
            
        f_binary = forecast > f_thresh
        o_binary = observed > o_thresh
        
        hits = np.logical_and(f_binary, o_binary).sum()
        total = len(f_binary)
        
        if hits == 0 or total == 0:
            return np.nan
            
        hit_rate = hits / total
        return 2 * np.log(hit_rate) / np.log(hit_rate + np.finfo(float).eps) - 1


class DurationMetrics:
    """Metrics for evaluating duration of extreme events."""
    
    @staticmethod
    def duration_bias(
        forecast: xr.DataArray,
        observed: xr.DataArray,
        threshold: float,
        method: str = "absolute",
        min_duration: int = 1
    ) -> float:
        """Calculate bias in extreme event duration."""
        if method == "percentile":
            f_thresh = np.percentile(forecast, threshold)
            o_thresh = np.percentile(observed, threshold)
        else:
            f_thresh = o_thresh = threshold
            
        f_binary = forecast > f_thresh
        o_binary = observed > o_thresh
        
        # Find runs of True values
        def get_runs(x):
            runs = np.split(np.arange(len(x)), np.where(np.diff(x))[0] + 1)
            return [len(run) for run in runs if len(run) >= min_duration and x[run[0]]]
        
        f_durations = get_runs(f_binary)
        o_durations = get_runs(o_binary)
        
        f_mean = np.mean(f_durations) if f_durations else np.nan
        o_mean = np.mean(o_durations) if o_durations else np.nan
        
        return f_mean / o_mean if o_mean != 0 else np.nan


class MLModelMetrics:
    """Metrics specific to ML model evaluation."""
    
    @staticmethod
    def extreme_value_skill_score(
        forecast: xr.DataArray,
        observed: xr.DataArray,
        threshold: float,
        method: str = "absolute"
    ) -> float:
        """Calculate skill score for extreme value prediction.
        
        This metric focuses specifically on model performance for extreme events,
        giving higher weight to accurate prediction of extremes.
        """
        if method == "percentile":
            f_thresh = np.percentile(forecast, threshold)
            o_thresh = np.percentile(observed, threshold)
        else:
            f_thresh = o_thresh = threshold
            
        # Calculate errors only for extreme events
        f_extremes = forecast.where(observed > o_thresh)
        o_extremes = observed.where(observed > o_thresh)
        
        # Calculate weighted RMSE
        weights = (observed - o_thresh).where(observed > o_thresh)
        weights = weights / weights.sum()
        
        mse = ((f_extremes - o_extremes)**2 * weights).sum()
        rmse = float(np.sqrt(mse))
        
        return 1 / (1 + rmse)  # Convert to skill score (0-1)
    
    @staticmethod
    def pattern_prediction_score(
        forecast: xr.DataArray,
        observed: xr.DataArray,
        spatial_scale: float = 1.0,
        temporal_scale: str = "1D"
    ) -> float:
        """Evaluate model's ability to predict spatial-temporal patterns."""
        # Ensure data has expected dimensions
        if not all(dim in forecast.dims for dim in ['time', 'latitude', 'longitude']):
            raise ValueError("Data must have dimensions ['time', 'latitude', 'longitude']")
        
        # Calculate spatial correlation over time
        spatial_corr = []
        for t in forecast.time:
            f_slice = forecast.sel(time=t)
            o_slice = observed.sel(time=t)
            if not np.all(np.isnan(f_slice)) and not np.all(np.isnan(o_slice)):
                corr = stats.pearsonr(
                    f_slice.values.flatten(),
                    o_slice.values.flatten()
                )[0]
                if not np.isnan(corr):
                    spatial_corr.append(corr)
        
        # Calculate temporal correlation at each point
        temporal_corr = []
        for lat in forecast.latitude:
            for lon in forecast.longitude:
                f_slice = forecast.sel(latitude=lat, longitude=lon)
                o_slice = observed.sel(latitude=lat, longitude=lon)
                if not np.all(np.isnan(f_slice)) and not np.all(np.isnan(o_slice)):
                    corr = stats.pearsonr(f_slice.values, o_slice.values)[0]
                    if not np.isnan(corr):
                        temporal_corr.append(corr)
        
        # Combine scores (geometric mean)
        if not spatial_corr or not temporal_corr:
            return 0.0
        
        spatial_mean = np.mean(spatial_corr)
        temporal_mean = np.mean(temporal_corr)
        
        # Handle negative correlations
        if spatial_mean < 0 or temporal_mean < 0:
            return 0.0
        
        return float(np.sqrt(spatial_mean * temporal_mean))
    
    @staticmethod
    def physical_consistency_score(
        forecast: xr.DataArray,
        pressure: xr.DataArray,
        observed: xr.DataArray
    ) -> float:
        """Evaluate physical consistency of model predictions.
        
        Args:
            forecast: Model forecast data
            pressure: Pressure levels
            observed: Observed data
            
        Returns:
            Physical consistency score (0-1)
        """
        # Check for perfect forecast
        if np.allclose(forecast, observed):
            return 1.0
        
        # Calculate potential temperature
        from wx_extreme.utils.met_utils import potential_temperature
        
        # Ensure pressure has same shape as temperature
        if pressure.shape != forecast.shape:
            pressure = pressure.broadcast_like(forecast)
        
        theta_f = potential_temperature(forecast, pressure)
        theta_o = potential_temperature(observed, pressure)
        
        # Calculate stability metrics
        stability_f = theta_f.diff('latitude').mean()
        stability_o = theta_o.diff('latitude').mean()
        
        # Calculate consistency score
        score = 1.0 - abs(stability_f - stability_o) / (abs(stability_f) + abs(stability_o) + 1e-6)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, float(score)))
        
        return score

def load_forecast_data():
    """Load Pangu-Weather forecast from AWS S3."""
    return xr.open_dataset(
        "s3://pangu-weather-public/forecasts/latest/t2m.zarr",
        engine="zarr",
        backend_kwargs={
            "storage_options": {
                "anon": True
            }
        }
    )
