"""Thermodynamic utility functions."""

import numpy as np
import xarray as xr
from typing import Tuple, Optional


def potential_temperature(temp: xr.DataArray, pressure: xr.DataArray) -> xr.DataArray:
    """Calculate potential temperature.
    
    Args:
        temp: Temperature in Kelvin
        pressure: Pressure in hPa
        
    Returns:
        Potential temperature in Kelvin
    """
    return temp * (1000.0 / pressure) ** 0.286


def relative_humidity(
    temp: xr.DataArray,
    pressure: xr.DataArray,
    specific_humidity: xr.DataArray
) -> xr.DataArray:
    """Calculate relative humidity.
    
    Args:
        temp: Temperature in Kelvin
        pressure: Pressure in hPa
        specific_humidity: Specific humidity in kg/kg
        
    Returns:
        Relative humidity (0-1)
    """
    # Constants
    epsilon = 0.622  # Ratio of gas constants
    
    # Saturation vapor pressure (Bolton's formula)
    es = 6.112 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65))
    
    # Actual vapor pressure
    e = pressure * specific_humidity / (epsilon + (1 - epsilon) * specific_humidity)
    
    return e / es


def equivalent_potential_temperature(
    temp: xr.DataArray,
    pressure: xr.DataArray,
    specific_humidity: xr.DataArray
) -> xr.DataArray:
    """Calculate equivalent potential temperature.
    
    Args:
        temp: Temperature in Kelvin
        pressure: Pressure in hPa
        specific_humidity: Specific humidity in kg/kg
        
    Returns:
        Equivalent potential temperature in Kelvin
    """
    # Constants
    Lv = 2.5e6  # Latent heat of vaporization
    cp = 1004.0  # Specific heat at constant pressure
    
    # Calculate potential temperature
    theta = potential_temperature(temp, pressure)
    
    # Calculate equivalent potential temperature
    return theta * np.exp(Lv * specific_humidity / (cp * temp))


def static_stability(
    temp: xr.DataArray,
    pressure: xr.DataArray,
    height: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """Calculate static stability.
    
    Args:
        temp: Temperature in Kelvin
        pressure: Pressure in hPa
        height: Optional height levels in meters
        
    Returns:
        Static stability in K/km or K/Pa depending on input
    """
    theta = potential_temperature(temp, pressure)
    
    if height is not None:
        # Calculate with respect to height
        return theta.differentiate('level') / (height.differentiate('level') / 1000)
    else:
        # Calculate with respect to pressure
        return -theta.differentiate('level') / pressure.differentiate('level')
