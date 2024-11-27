"""Statistical utility functions."""

import numpy as np
import xarray as xr
from scipy import stats
from typing import Optional, Union, Tuple


def calculate_percentile(
    data: Union[np.ndarray, xr.DataArray],
    q: Union[float, np.ndarray],
    dim: Optional[Union[str, list]] = None,
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate percentile(s) along specified dimension(s).
    
    Args:
        data: Input data
        q: Percentile(s) to compute (0-100)
        dim: Dimension(s) to calculate percentile over
        
    Returns:
        Percentile values
    """
    if isinstance(data, xr.DataArray):
        return data.quantile(q/100, dim=dim)
    return np.percentile(data, q, axis=dim)


def exceedance_probability(
    data: Union[np.ndarray, xr.DataArray],
    threshold: Union[float, np.ndarray],
    dim: Optional[Union[str, list]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate probability of exceeding threshold.
    
    Args:
        data: Input data
        threshold: Threshold value(s)
        dim: Dimension(s) to calculate probability over
        
    Returns:
        Exceedance probability (0-1)
    """
    if isinstance(data, xr.DataArray):
        return (data > threshold).mean(dim=dim)
    return np.mean(data > threshold, axis=dim)


def return_period(
    data: Union[np.ndarray, xr.DataArray],
    value: Union[float, np.ndarray],
    time_unit: str = 'year',
) -> Union[float, np.ndarray]:
    """Calculate return period for given value(s).
    
    Args:
        data: Input data
        value: Value(s) to calculate return period for
        time_unit: Time unit for return period
        
    Returns:
        Return period in specified time units
    """
    if isinstance(data, xr.DataArray):
        data = data.values
        
    # Calculate empirical exceedance probability
    sorted_data = np.sort(data)
    ranks = stats.rankdata(sorted_data, method='average')
    n = len(data)
    
    # Convert to return periods
    exceedance_prob = 1 - (ranks - 0.5) / n
    periods = 1 / exceedance_prob
    
    # Interpolate to get return periods for specific values
    return np.interp(value, sorted_data, periods)


def extreme_value_fit(
    data: Union[np.ndarray, xr.DataArray],
    distribution: str = 'gev',
) -> Tuple[np.ndarray, dict]:
    """Fit extreme value distribution to data.
    
    Args:
        data: Input data
        distribution: Distribution type ('gev' or 'gpd')
        
    Returns:
        Tuple of (parameters, fit_info)
    """
    if isinstance(data, xr.DataArray):
        data = data.values
        
    if distribution.lower() == 'gev':
        # Fit Generalized Extreme Value distribution
        shape, loc, scale = stats.genextreme.fit(data)
        params = np.array([shape, loc, scale])
        info = {
            'distribution': 'GEV',
            'log_likelihood': stats.genextreme.logpdf(data, shape, loc, scale).sum()
        }
    elif distribution.lower() == 'gpd':
        # Fit Generalized Pareto distribution
        shape, loc, scale = stats.genpareto.fit(data)
        params = np.array([shape, loc, scale])
        info = {
            'distribution': 'GPD',
            'log_likelihood': stats.genpareto.logpdf(data, shape, loc, scale).sum()
        }
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
        
    return params, info


def block_maxima(
    data: Union[np.ndarray, xr.DataArray],
    block_size: int,
    dim: Optional[str] = None,
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate block maxima.
    
    Args:
        data: Input data
        block_size: Size of blocks
        dim: Dimension to calculate blocks over (required for xarray)
        
    Returns:
        Block maxima values
    """
    if isinstance(data, xr.DataArray):
        if dim is None:
            raise ValueError("dim must be specified for xarray input")
        return data.coarsen({dim: block_size}).max()
    
    # Reshape array to get blocks
    n_blocks = len(data) // block_size
    blocks = data[:n_blocks*block_size].reshape(n_blocks, block_size)
    return np.max(blocks, axis=1)


def peaks_over_threshold(
    data: Union[np.ndarray, xr.DataArray],
    threshold: Union[float, np.ndarray],
    separation: int = 1,
) -> Union[np.ndarray, xr.DataArray]:
    """Extract peaks over threshold with minimum separation.
    
    Args:
        data: Input data
        threshold: Threshold value(s)
        separation: Minimum separation between peaks
        
    Returns:
        Peak values
    """
    if isinstance(data, xr.DataArray):
        data = data.values
        
    # Find values over threshold
    over_thresh = data > threshold
    
    # Find peaks with minimum separation
    peaks = []
    last_peak = -np.inf
    
    for i in range(len(data)):
        if over_thresh[i] and i > last_peak + separation:
            if i == 0 or data[i] > data[i-1]:
                if i == len(data)-1 or data[i] > data[i+1]:
                    peaks.append(i)
                    last_peak = i
                    
    return data[peaks]
