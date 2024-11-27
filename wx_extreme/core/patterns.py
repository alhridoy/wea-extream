"""Pattern analysis module for extreme weather events."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import xarray as xr
from scipy import signal, ndimage


class PatternAnalyzer:
    """Analyzer for spatial and temporal patterns in weather data."""
    
    def __init__(
        self,
        min_size: int = 3,
        min_duration: int = 6,
        connectivity: int = 2,
    ):
        """Initialize pattern analyzer.
        
        Args:
            min_size: Minimum spatial size (grid points) for pattern detection
            min_duration: Minimum temporal duration for pattern detection
            connectivity: Connectivity for spatial pattern detection (1 or 2)
        """
        self.min_size = min_size
        self.min_duration = min_duration
        self.connectivity = connectivity
    
    def find_spatial_patterns(
        self,
        data: xr.DataArray,
        threshold: float,
        method: str = "absolute"
    ) -> Tuple[xr.DataArray, List[Dict]]:
        """Find spatial patterns in data.
        
        Args:
            data: Input data array
            threshold: Detection threshold
            method: Threshold method ('absolute' or 'percentile')
            
        Returns:
            Labeled array and pattern properties
        """
        # Apply threshold
        if method == "percentile":
            thresh_value = np.percentile(data, threshold)
        else:
            thresh_value = threshold
            
        binary = data > thresh_value
        
        # Label connected components
        labels, num = ndimage.label(binary, structure=np.ones((3, 3)))
        
        # Get properties
        props = []
        for i in range(1, num + 1):
            mask = labels == i
            if mask.sum() >= self.min_size:
                props.append({
                    "label": i,
                    "size": mask.sum(),
                    "centroid": ndimage.center_of_mass(mask),
                    "max_value": data.where(mask).max().item(),
                    "mean_value": data.where(mask).mean().item()
                })
        
        return xr.DataArray(labels, dims=data.dims, coords=data.coords), props
    
    def find_temporal_patterns(
        self,
        data: xr.DataArray,
        threshold: float,
        method: str = "absolute"
    ) -> List[Dict]:
        """Find temporal patterns in data.
        
        Args:
            data: Input time series
            threshold: Detection threshold
            method: Threshold method ('absolute' or 'percentile')
            
        Returns:
            List of detected patterns
        """
        # Apply threshold
        if method == "percentile":
            thresh_value = np.percentile(data, threshold)
        else:
            thresh_value = threshold
            
        binary = data > thresh_value
        
        # Find runs of True values
        runs = np.split(np.arange(len(binary)), np.where(np.diff(binary))[0] + 1)
        
        # Get properties of runs that exceed minimum duration
        patterns = []
        for run in runs:
            if len(run) >= self.min_duration and binary[run[0]]:
                patterns.append({
                    "start_idx": run[0],
                    "end_idx": run[-1],
                    "duration": len(run),
                    "max_value": data[run].max().item(),
                    "mean_value": data[run].mean().item()
                })
        
        return patterns
    
    def track_patterns(
        self,
        data: xr.DataArray,
        threshold: float,
        method: str = "absolute",
        max_speed: Optional[float] = None
    ) -> List[Dict]:
        """Track patterns through space and time.
        
        Args:
            data: Input data array (must have time dimension)
            threshold: Detection threshold
            method: Threshold method ('absolute' or 'percentile')
            max_speed: Maximum allowed displacement speed between timesteps
            
        Returns:
            List of tracked patterns
        """
        tracks = []
        current_tracks = []
        
        for t in range(len(data.time)):
            # Find patterns at current time
            labels, props = self.find_spatial_patterns(
                data.isel(time=t),
                threshold,
                method
            )
            
            # Match with existing tracks
            unmatched = list(range(len(props)))
            for track in current_tracks:
                last_pos = track["positions"][-1]
                best_match = None
                best_dist = float("inf")
                
                for i, prop in enumerate(props):
                    if i not in unmatched:
                        continue
                        
                    dist = np.sqrt(
                        (prop["centroid"][0] - last_pos[0])**2 +
                        (prop["centroid"][1] - last_pos[1])**2
                    )
                    
                    if (max_speed is None or dist <= max_speed) and dist < best_dist:
                        best_match = i
                        best_dist = dist
                
                if best_match is not None:
                    track["positions"].append(props[best_match]["centroid"])
                    track["sizes"].append(props[best_match]["size"])
                    track["max_values"].append(props[best_match]["max_value"])
                    track["mean_values"].append(props[best_match]["mean_value"])
                    unmatched.remove(best_match)
                else:
                    track["end_time"] = t - 1
                    if t - track["start_time"] >= self.min_duration:
                        tracks.append(track)
                    current_tracks.remove(track)
            
            # Start new tracks for unmatched patterns
            for i in unmatched:
                current_tracks.append({
                    "start_time": t,
                    "positions": [props[i]["centroid"]],
                    "sizes": [props[i]["size"]],
                    "max_values": [props[i]["max_value"]],
                    "mean_values": [props[i]["mean_value"]]
                })
        
        # Finish any remaining tracks
        for track in current_tracks:
            track["end_time"] = len(data.time) - 1
            if track["end_time"] - track["start_time"] >= self.min_duration:
                tracks.append(track)
        
        return tracks
