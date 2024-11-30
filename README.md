# WX-Extreme

A Python package for detecting and evaluating extreme weather events in climate data.

## WeatherBench2 Limitations and WX-Extreme Solutions

WX-Extreme extends WeatherBench2's capabilities by addressing several key limitations:

### 1. Extreme Event Focus
WeatherBench2:
- Focuses on general forecast skill (RMSE, ACC)
- Lacks specific extreme event detection
- No duration or spatial coherence analysis

WX-Extreme Solutions:
- Flexible threshold definitions (percentile/absolute)
- Minimum duration constraints
- Spatial coherence requirements
- Event-specific evaluation metrics

### 2. Physical Consistency
WeatherBench2:
- Limited physical consistency checks
- No grid-aware validations
- Basic thermodynamic relationships

WX-Extreme Solutions:
- Comprehensive physical validation suite
- Grid-aware spatial metrics
- Advanced thermodynamic checks:
  - Potential temperature relationships
  - Hydrostatic balance
  - Pattern prediction scoring

### 3. Spatial Analysis
WeatherBench2:
- Global metrics only
- No regional analysis capabilities
- Fixed grid assumptions

WX-Extreme Solutions:
- Grid spacing calculations
- Area-weighted metrics
- Regional event detection
- Flexible grid handling
- Spatial coherence validation

### 4. Statistical Robustness
WeatherBench2:
- Basic statistical measures
- Limited extreme value analysis
- No event duration statistics

WX-Extreme Solutions:
- Advanced statistical tools:
  - Exceedance probabilities
  - Percentile-based thresholds
  - Duration statistics
  - Event frequency analysis
  - Pattern correlation metrics

### 5. Visualization
WeatherBench2:
- Basic plotting capabilities
- Limited comparison tools
- No event-specific visualization

WX-Extreme Solutions:
- Specialized visualization suite:
  - Event heatmaps
  - Model comparison plots
  - Bias assessment visualizations
  - Spatial pattern analysis plots

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Requirements

The package works with weather/climate data that meets these specifications:

### Required Data Format
- **File Format**: NetCDF (.nc) or any xarray-compatible format
- **Dimensions**: 
  - time
  - latitude (degrees North, -90 to 90)
  - longitude (degrees East, -180 to 180)
- **Variables**:
  - Temperature (°C or K)
  - Pressure levels (Pa) - for physical consistency evaluation

### Recommended Data Sources
1. **ERA5 Reanalysis**
   - Resolution: 0.25° x 0.25°
   - Variables: 2m temperature, surface pressure
   - [Download from CDS](https://cds.climate.copernicus.eu/)

2. **CMIP6 Model Outputs**
   - Variables: tas (surface temperature), ps (surface pressure)
   - [Download from ESGF](https://esgf-node.llnl.gov/projects/cmip6/)

3. **Weather Model Forecasts**
   - GFS (Global Forecast System)
   - ECMWF forecasts

## Quick Start

```python
from wx_extreme.core.detector import ExtremeEventDetector
import xarray as xr

# Load your data
data = xr.open_dataset('temperature.nc')
temperature = data['t2m']

# Initialize detector
detector = ExtremeEventDetector(
    threshold_method="percentile",
    threshold_value=95,
    min_duration=3
)

# Detect events
events = detector.detect_events(temperature)
```

See `examples/model_evaluation.py` for a complete tutorial.

## Features

1. **Event Detection**
   - Percentile-based thresholds
   - Absolute value thresholds
   - Spatial coherence requirements
   - Minimum duration constraints

2. **Event Evaluation**
   - Frequency analysis
   - Intensity metrics
   - Duration statistics
   - Spatial extent

3. **Model Evaluation**
   - Pattern prediction scoring
   - Physical consistency checks
   - Bias assessment
   - Spatial correlation analysis

## Dependencies

- numpy
- xarray
- pandas
- matplotlib
- scipy
- netCDF4
