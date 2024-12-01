# WX-Extreme: Weather Extreme Event Detection and Validation

A Python package for detecting and validating extreme weather events in meteorological datasets, with special support for comparing weather forecasts (like Pangu-Weather) against ground truth (like ERA5).

> **Note:** "WX" is the traditional meteorological abbreviation for "weather", commonly used in aviation and weather forecasting.

## Inspiration and Differences from WeatherBench2

WX-Extreme was inspired by [WeatherBench2](https://sites.research.google/weatherbench/), a benchmark dataset for machine learning in weather forecasting. While WeatherBench2 provides an excellent foundation for general forecast evaluation, WX-Extreme addresses specific limitations when it comes to extreme event prediction:

### Key Differences

1. **Event-Specific Focus**
   - WeatherBench2: General metrics (RMSE, ACC) for overall forecast skill
   - WX-Extreme: Specialized detection and validation of extreme events with:
     - Configurable thresholds (percentile/absolute)
     - Temporal persistence requirements
     - Spatial coherence validation
     - Event-specific metrics (POD, FAR)

2. **Operational Considerations**
   - WeatherBench2: Research-oriented evaluation framework
   - WX-Extreme: Operational forecasting focus:
     - Multiple forecast initialization times
     - Lead time dependency analysis
     - Forecast skill decay assessment
     - Rolling evaluation windows

3. **Regional Analysis**
   - WeatherBench2: Global metrics and evaluation
   - WX-Extreme: Support for regional studies:
     - Flexible coordinate system handling
     - Region-specific thresholds
     - Local pattern analysis
     - Area-weighted metrics

4. **Memory Efficiency**
   - WeatherBench2: Full data loading approach
   - WX-Extreme: Production-ready data handling:
     - Intelligent chunking strategies
     - Memory-efficient processing
     - Parallel computation support
     - Cloud storage integration

5. **User Focus**
   - WeatherBench2: Research benchmark dataset
   - WX-Extreme: Operational tool with:
     - Simple, intuitive API
     - Clear documentation
     - Practical examples
     - Real-world use cases

## Features

- Extreme event detection using percentile or absolute thresholds
- Support for spatial and temporal persistence criteria
- Built-in support for ERA5 and Pangu-Weather data formats
- Validation metrics for extreme event prediction
- Visualization tools for event comparison
- Operational forecast evaluation:
  - Multiple initialization times
  - Lead time analysis
  - Forecast skill decay
  - Rolling evaluation windows

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wx-extreme.git
cd wx-extreme

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Here's a simple example of validating Pangu-Weather forecasts against ERA5 data:

```python
from wx_extreme.core.detector import ExtremeEventDetector
import xarray as xr

# Load your data (example using provided datasets)
forecast = xr.open_dataset("path/to/forecast.zarr")
analysis = xr.open_dataset("path/to/analysis.zarr")

# Create detector
detector = ExtremeEventDetector(
    threshold_method="percentile",
    threshold_value=95,  # 95th percentile
    min_duration=3      # 3 time steps persistence
)

# Detect events
forecast_events = detector.detect_events(forecast)
analysis_events = detector.detect_events(analysis)

# Calculate validation metrics
hits = (forecast_events & analysis_events).sum().values
misses = (~forecast_events & analysis_events).sum().values
false_alarms = (forecast_events & ~analysis_events).sum().values

pod = hits / (hits + misses)  # Probability of Detection
far = false_alarms / (hits + false_alarms)  # False Alarm Ratio
```

## Operational Forecast Evaluation

The package supports evaluation of operational forecasting systems:

```python
# Load forecast dataset with multiple initialization times
forecast_ds = xr.open_dataset("forecasts.zarr")
era5_ds = xr.open_dataset("era5.zarr")

# Evaluate forecast skill by lead time
skill_df = evaluate_forecast_skill(
    forecast_ds, 
    era5_ds,
    lead_times=range(24, 241, 24)  # 1-10 days
)

# Plot skill metrics
fig = plot_forecast_skill(skill_df)
fig.savefig('forecast_skill.png')
```

This will analyze:
- Forecast performance at different lead times
- Skill decay over time
- Initialization time dependence
- Systematic biases

## Example Scripts

Check out `examples/validate_forecast.py` for a complete example that:
1. Loads Pangu-Weather forecasts and ERA5 analysis data
2. Handles coordinate system differences
3. Detects extreme events
4. Calculates validation metrics
5. Creates comparison plots
6. Evaluates operational forecast skill

To run the example:
```bash
python examples/validate_forecast.py
```

## Data Requirements

The package works with any NetCDF/Zarr dataset that has:
- Temperature data in Kelvin or Celsius
- Dimensions: time, latitude, longitude
- For forecasts: init_time and fcst_hour dimensions
- Coordinates in either 0-360° or -180-180° longitude convention

Supported data sources:
- ERA5 reanalysis (available through Copernicus/GCS)
- Pangu-Weather forecasts
- Any similar gridded weather data

## API Reference

### ExtremeEventDetector

Main class for detecting extreme events.

```python
detector = ExtremeEventDetector(
    threshold_method="percentile",  # or "absolute"
    threshold_value=95,            # percentile or absolute value
    min_duration=3,               # minimum event duration
    spatial_scale=None           # minimum spatial scale (optional)
)
```

Methods:
- `detect_events(data)`: Detect extreme events in the input data
- `_compute_threshold(data)`: Compute detection threshold
- `_apply_duration_filter(events)`: Apply temporal persistence criteria
- `_apply_spatial_filter(events)`: Apply spatial coherence criteria

### Forecast Evaluation

Functions for operational forecast evaluation:

- `evaluate_forecast_skill()`: Analyze forecast performance by lead time
- `plot_forecast_skill()`: Visualize forecast skill metrics
- `normalize_coords()`: Handle coordinate system differences

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:
```
@software{wx_extreme,
  title = {WX-Extreme: Weather Extreme Event Detection and Validation},
  author = {Al-Ekram Elahee Hridoy},
  year = {2024},
  url = {https://github.com/alhridoy/wx-extreme}
}
```
