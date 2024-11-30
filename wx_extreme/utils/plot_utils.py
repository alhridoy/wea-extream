"""Visualization utility functions."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from typing import Optional, Union, Tuple, List, Dict
import warnings


def setup_map(
    projection: ccrs.Projection = ccrs.PlateCarree(),
    central_longitude: float = 0.0,
    extent: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Set up a map for plotting.
    
    Args:
        projection: Map projection
        central_longitude: Central longitude for projection
        extent: Map extent [lon_min, lon_max, lat_min, lat_max]
        figsize: Figure size (width, height)
        
    Returns:
        Tuple of (figure, axes)
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines()
    
    if extent is not None:
        ax.set_extent(extent)
        
    return fig, ax


def plot_field(
    data: xr.DataArray,
    ax: Optional[plt.Axes] = None,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot a 2D field on a map.
    
    Args:
        data: Data to plot (must have latitude and longitude coordinates)
        ax: Axes to plot on (if None, creates new figure)
        projection: Map projection
        cmap: Colormap
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        colorbar: Whether to add colorbar
        colorbar_label: Label for colorbar
        title: Plot title
        
    Returns:
        Plot axes
    """
    if ax is None:
        _, ax = setup_map(projection=projection)
        
    # Plot data
    im = data.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=colorbar,
    )
    
    if colorbar and colorbar_label:
        im.colorbar.set_label(colorbar_label)
        
    if title:
        ax.set_title(title)
        
    return ax


def plot_contours(
    data: xr.DataArray,
    ax: Optional[plt.Axes] = None,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    levels: Union[int, List[float]] = 10,
    colors: str = 'k',
    linewidths: float = 1.0,
    labels: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot contours on a map.
    
    Args:
        data: Data to plot contours of
        ax: Axes to plot on (if None, creates new figure)
        projection: Map projection
        levels: Contour levels
        colors: Contour colors
        linewidths: Contour line widths
        labels: Whether to add contour labels
        title: Plot title
        
    Returns:
        Plot axes
    """
    if ax is None:
        _, ax = setup_map(projection=projection)
        
    # Plot contours
    cs = data.plot.contour(
        ax=ax,
        transform=ccrs.PlateCarree(),
        levels=levels,
        colors=colors,
        linewidths=linewidths,
    )
    
    if labels:
        ax.clabel(cs, inline=True, fontsize=8)
        
    if title:
        ax.set_title(title)
        
    return ax


def plot_wind_barbs(
    u: xr.DataArray,
    v: xr.DataArray,
    ax: Optional[plt.Axes] = None,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    skip: int = 1,
    length: float = 4,
    color: str = 'k',
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot wind barbs on a map.
    
    Args:
        u: U-component of wind
        v: V-component of wind
        ax: Axes to plot on (if None, creates new figure)
        projection: Map projection
        skip: Plot every nth wind barb
        length: Length of wind barbs
        color: Color of wind barbs
        title: Plot title
        
    Returns:
        Plot axes
    """
    if ax is None:
        _, ax = setup_map(projection=projection)
        
    # Get coordinates
    lons = u.longitude.values[::skip]
    lats = u.latitude.values[::skip]
    
    # Create coordinate meshgrid
    lon, lat = np.meshgrid(lons, lats)
    
    # Plot wind barbs
    ax.barbs(
        lon, lat,
        u.values[::skip, ::skip],
        v.values[::skip, ::skip],
        transform=ccrs.PlateCarree(),
        length=length,
        color=color,
    )
    
    if title:
        ax.set_title(title)
        
    return ax


def plot_streamlines(
    u: xr.DataArray,
    v: xr.DataArray,
    ax: Optional[plt.Axes] = None,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    density: float = 1,
    color: str = 'k',
    linewidth: float = 1,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot streamlines on a map.
    
    Args:
        u: U-component of wind
        v: V-component of wind
        ax: Axes to plot on (if None, creates new figure)
        projection: Map projection
        density: Density of streamlines
        color: Color of streamlines
        linewidth: Width of streamlines
        title: Plot title
        
    Returns:
        Plot axes
    """
    if ax is None:
        _, ax = setup_map(projection=projection)
        
    # Get coordinates
    lons = u.longitude.values
    lats = u.latitude.values
    
    # Create coordinate meshgrid
    lon, lat = np.meshgrid(lons, lats)
    
    # Plot streamlines
    ax.streamplot(
        lon, lat,
        u.values,
        v.values,
        transform=ccrs.PlateCarree(),
        density=density,
        color=color,
        linewidth=linewidth,
    )
    
    if title:
        ax.set_title(title)
        
    return ax


def plot_return_period(
    data: Union[np.ndarray, xr.DataArray],
    values: Union[float, np.ndarray],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    marker: str = 'o',
    line_style: str = '-',
    color: str = 'b',
    xlabel: str = 'Return Period (years)',
    ylabel: str = 'Value',
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot return period curve.
    
    Args:
        data: Input data
        values: Values to plot return periods for
        ax: Axes to plot on (if None, creates new figure)
        figsize: Figure size if creating new figure
        marker: Marker style for points
        line_style: Line style
        color: Color for plot
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        
    Returns:
        Plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    # Calculate return periods
    periods = return_period(data, values)
    
    # Plot
    ax.plot(periods, values, marker=marker, linestyle=line_style, color=color)
    ax.set_xscale('log')
    
    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
        
    return ax


def plot_model_comparison_heatmap(
    model_data: xr.DataArray,
    reference_data: xr.DataArray,
    metric: str = 'bias',
    cmap: str = 'RdYlBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot heatmap comparing model predictions to reference data."""
    if ax is None:
        ax = plt.gca()
    
    # Check for required dimensions
    if 'time' not in model_data.dims or 'time' not in reference_data.dims:
        warnings.warn("Data missing time dimension")
        return ax
    
    # Calculate metric
    if metric == 'bias':
        diff = model_data - reference_data
        data = diff.mean(dim='time')
        label = 'Mean Bias'
    elif metric == 'rmse':
        diff = (model_data - reference_data)**2
        data = np.sqrt(diff.mean(dim='time'))
        label = 'RMSE'
    else:
        # Calculate correlation at each point
        data = xr.corr(model_data, reference_data, dim='time')
        label = 'Correlation'
    
    # Create plot
    try:
        im = ax.pcolormesh(
            data.longitude,
            data.latitude,
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(im, ax=ax, label=label)
    except (AttributeError, ValueError) as e:
        warnings.warn(f"Error creating plot: {str(e)}")
        return ax
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    return ax


def plot_extreme_event_heatmap(
    events: xr.DataArray,
    temperature: xr.DataArray,
    time_slice: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot heatmap of extreme temperature events."""
    if ax is None:
        ax = plt.gca()
    
    # Select time slice if specified
    if time_slice:
        events = events.sel(time=time_slice)
        temperature = temperature.sel(time=time_slice)
    else:
        # Use first time step if no slice specified
        events = events.isel(time=0)
        temperature = temperature.isel(time=0)
    
    # Plot temperature heatmap
    try:
        temp_plot = ax.pcolormesh(
            temperature.longitude,
            temperature.latitude,
            temperature,
            cmap='RdYlBu_r'
        )
        plt.colorbar(temp_plot, ax=ax, label='Temperature (°C)')
        
        # Overlay event contours if any events exist
        if events.any():
            ax.contour(
                events.longitude,
                events.latitude,
                events,
                colors='k',
                linewidths=2,
                levels=[0.5]
            )
    except (AttributeError, ValueError) as e:
        warnings.warn(f"Error creating plot: {str(e)}")
        return ax
    
    ax.set_title('Temperature with Extreme Events')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    return ax
