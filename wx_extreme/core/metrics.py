"""Metrics for evaluating extreme weather events."""

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
