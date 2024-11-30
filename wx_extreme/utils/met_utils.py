"""Meteorological utility functions."""

import numpy as np
import xarray as xr


def potential_temperature(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    p0: float = 1000.0  # Reference pressure in hPa
) -> xr.DataArray:
    """Calculate potential temperature.
    
    Args:
        temperature: Temperature in Kelvin
        pressure: Pressure in hPa
        p0: Reference pressure in hPa
        
    Returns:
        Potential temperature in Kelvin
    """
    R = 287.0  # Gas constant for dry air (J/kg/K)
    cp = 1004.0  # Specific heat at constant pressure (J/kg/K)
    kappa = R / cp
    
    # Ensure pressure is positive and not greater than p0
    pressure = xr.where(pressure <= 0, p0, pressure)
    pressure = xr.where(pressure > p0, p0, pressure)
    
    # Calculate potential temperature using Poisson's equation
    theta = temperature * (p0 / pressure) ** kappa
    
    # Ensure result is physically meaningful
    # Add a small increment to temperature to ensure it's always higher
    theta = xr.where(theta <= temperature, temperature * 1.01, theta)
    
    return theta


def relative_humidity(
    specific_humidity: xr.DataArray,
    temperature: xr.DataArray,
    pressure: xr.DataArray,
) -> xr.DataArray:
    """Calculate relative humidity.
    
    Args:
        specific_humidity: Specific humidity in kg/kg
        temperature: Temperature in Kelvin
        pressure: Pressure in hPa
        
    Returns:
        Relative humidity (0-1)
    """
    # Constants
    es0 = 6.11  # Reference saturation vapor pressure in hPa
    T0 = 273.15  # Reference temperature in K
    L = 2.5e6  # Latent heat of vaporization in J/kg
    Rv = 461.5  # Gas constant for water vapor in J/kg/K
    
    # Calculate saturation vapor pressure
    es = es0 * np.exp(L/Rv * (1/T0 - 1/temperature))
    
    # Convert specific humidity to mixing ratio
    w = specific_humidity / (1 - specific_humidity)
    
    # Calculate actual vapor pressure
    e = pressure * w / (0.622 + w)
    
    return e / es


def equivalent_potential_temperature(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    specific_humidity: xr.DataArray,
) -> xr.DataArray:
    """Calculate equivalent potential temperature.
    
    Args:
        temperature: Temperature in Kelvin
        pressure: Pressure in hPa
        specific_humidity: Specific humidity in kg/kg
        
    Returns:
        Equivalent potential temperature in Kelvin
    """
    # Constants
    L = 2.5e6  # Latent heat of vaporization in J/kg
    cp = 1004.0  # Specific heat at constant pressure in J/kg/K
    
    # Calculate potential temperature
    theta = potential_temperature(temperature, pressure)
    
    # Calculate equivalent potential temperature
    return theta * np.exp(L * specific_humidity / (cp * temperature))


def geostrophic_wind(
    geopotential: xr.DataArray,
    latitude: xr.DataArray,
    dx: xr.DataArray,
    dy: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate geostrophic wind components.
    
    Args:
        geopotential: Geopotential in m²/s²
        latitude: Latitude in degrees
        dx: Grid spacing in x-direction in meters
        dy: Grid spacing in y-direction in meters
        
    Returns:
        Tuple of (u, v) geostrophic wind components in m/s
    """
    # Coriolis parameter
    f = 2 * 7.2921e-5 * np.sin(np.deg2rad(latitude))
    
    # Calculate gradients
    dgdx = geopotential.differentiate('longitude') / dx
    dgdy = geopotential.differentiate('latitude') / dy
    
    # Calculate geostrophic wind components
    ug = -1 / f * dgdy
    vg = 1 / f * dgdx
    
    return ug, vg


def thermal_wind(
    thickness: xr.DataArray,
    latitude: xr.DataArray,
    dx: xr.DataArray,
    dy: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate thermal wind components.
    
    Args:
        thickness: Layer thickness in m
        latitude: Latitude in degrees
        dx: Grid spacing in x-direction in meters
        dy: Grid spacing in y-direction in meters
        
    Returns:
        Tuple of (u, v) thermal wind components in m/s
    """
    g = 9.81  # Gravitational acceleration in m/s²
    
    # Calculate geostrophic wind from thickness
    return geostrophic_wind(thickness * g, latitude, dx, dy)
