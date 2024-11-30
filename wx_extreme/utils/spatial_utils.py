"""Spatial utility functions."""

import numpy as np
import xarray as xr
from typing import Tuple


def get_grid_spacing(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> Tuple[float, float]:
    """Calculate grid spacing in kilometers.
    
    Args:
        latitude: Latitude coordinates
        longitude: Longitude coordinates
        
    Returns:
        Tuple of (dx, dy) grid spacing in kilometers
    """
    R = 6371.0  # Earth radius in km
    
    # Convert to radians
    lat = np.deg2rad(latitude.mean())  # Use mean latitude for dx calculation
    dlat = np.deg2rad(latitude[1] - latitude[0])
    dlon = np.deg2rad(longitude[1] - longitude[0])
    
    # Calculate spacing
    dx = R * np.cos(lat) * dlon
    dy = R * dlat
    
    return abs(float(dx)), abs(float(dy))


def get_dx(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> xr.DataArray:
    """Calculate grid spacing in x-direction (meters).
    
    Args:
        latitude: Latitude coordinates
        longitude: Longitude coordinates
        
    Returns:
        Grid spacing in x-direction
    """
    R = 6371000.0  # Earth radius in meters
    
    # Convert to radians
    lat = np.deg2rad(latitude)
    dlon = np.deg2rad(longitude[1] - longitude[0])
    
    # Calculate spacing
    dx = R * np.cos(lat) * dlon
    
    return abs(dx)


def get_dy(
    latitude: xr.DataArray,
) -> xr.DataArray:
    """Calculate grid spacing in y-direction (meters).
    
    Args:
        latitude: Latitude coordinates
        
    Returns:
        Grid spacing in y-direction
    """
    R = 6371000.0  # Earth radius in meters
    
    # Convert to radians
    dlat = np.deg2rad(latitude[1] - latitude[0])
    
    # Calculate spacing
    dy = R * dlat
    
    return abs(dy)


def haversine_distance(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    """Calculate haversine distance between two points.
    
    Args:
        lon1: Longitude of first point (degrees)
        lat1: Latitude of first point (degrees)
        lon2: Longitude of second point (degrees)
        lat2: Latitude of second point (degrees)
        
    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in km
    
    # Convert to radians
    lat1, lon1 = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2, lon2 = np.deg2rad(lat2), np.deg2rad(lon2)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def great_circle_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate great circle distance between two points.
    
    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)
        
    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in km
    
    # Convert to radians
    lat1, lon1 = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2, lon2 = np.deg2rad(lat2), np.deg2rad(lon2)
    
    # Great circle formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def calculate_area(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> xr.DataArray:
    """Calculate grid cell areas in square kilometers.
    
    Args:
        latitude: Latitude coordinates
        longitude: Longitude coordinates
        
    Returns:
        Grid cell areas in square kilometers
    """
    dx, dy = get_grid_spacing(latitude.values, longitude.values)
    return dx * dy
