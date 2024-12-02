# WX-Extreme

Advanced Evaluation Framework for Extreme Weather Events in AI Weather Models

## Overview

WX-Extreme is a Python package for evaluating extreme weather events in machine learning weather forecasting models. The package provides:

- Flexible threshold-based event detection
- Comprehensive forecast verification metrics
- Spatial and temporal coherence analysis
- Model validation tools

## Installation

```bash
# Install from source
git clone https://github.com/alhridoy/wea-extream.git
cd wea-extream
pip install -e .
```

## Core Features

### 1. Event Detection
```python
from wx_extreme.core.detector import ExtremeEventDetector

# Initialize detector with options
detector = ExtremeEventDetector(
    threshold_method="percentile",  # or "absolute"
    threshold_value=95,            # percentile or absolute value
    min_duration=3,               # minimum event duration
    spatial_scale=2.0             # spatial coherence in grid points
)

# Detect events
events = detector.detect_events(data)
```

### 2. Forecast Metrics
```python
from wx_extreme.core.metrics import calculate_forecast_metrics

# Calculate comprehensive metrics
metrics = calculate_forecast_metrics(
    forecast=model_output,
    observation=reference_data
)

# Available metrics include:
# - Basic: bias, MAE, RMSE, MSE
# - Pattern: correlation, ACC
# - Skill Scores: MSE skill score, Murphy score
# - Categorical: POD, FAR, CSI, HSS, ETS (when threshold provided)
```

## Basic Usage

### 1. Detecting Heat Waves
```python
import xarray as xr
from wx_extreme.core.detector import ExtremeEventDetector

# Load your temperature data (must have time, latitude, longitude dimensions)
data = xr.open_dataset('temperature.nc')
temp = data['t2m']

# For temperature in Kelvin, convert to Celsius
if temp.max() > 100:
    temp = temp - 273.15

# Initialize detector for heat waves
detector = ExtremeEventDetector(
    threshold_method="percentile",  # Use percentile threshold
    threshold_value=95,            # 95th percentile
    min_duration=3,               # At least 3 time steps
    spatial_scale=2.0             # Minimum 2 grid points spatial extent
)

# Detect heat waves
heatwaves = detector.detect_events(temp)
```

### 2. Evaluating Model Forecasts
```python
from wx_extreme.core.metrics import calculate_forecast_metrics

# Load forecast and observation data
forecast = xr.open_dataset('forecast.nc')['t2m']
obs = xr.open_dataset('observation.nc')['t2m']

# Calculate verification metrics
metrics = calculate_forecast_metrics(
    forecast=forecast,
    observation=obs,
    dim=['latitude', 'longitude']  # Dimensions for pattern metrics
)

# Print key metrics
print(f"Bias: {metrics['bias']:.2f}°C")
print(f"RMSE: {metrics['rmse']:.2f}°C")
print(f"Pattern Correlation: {metrics['pattern_correlation']:.3f}")

# For categorical verification (e.g., extreme events)
forecast.attrs['threshold'] = 30  # Set temperature threshold (°C)
categorical_metrics = calculate_forecast_metrics(forecast, obs)
print(f"Hit Rate: {categorical_metrics['pod']:.2f}")
print(f"False Alarm Ratio: {categorical_metrics['far']:.2f}")
print(f"Critical Success Index: {categorical_metrics['csi']:.2f}")
```

### 3. Analyzing Results
```python
import matplotlib.pyplot as plt

# Plot detected events for a specific time
plt.figure(figsize=(10, 5))
heatwaves.isel(time=0).plot()
plt.title('Detected Heat Waves')
plt.savefig('heatwave_map.png')

# Calculate event statistics
total_events = heatwaves.sum().item()
event_days = heatwaves.sum('time').values
max_temp = temp.where(heatwaves).max().item()

print(f"Total event grid points: {total_events}")
print(f"Maximum temperature during events: {max_temp:.1f}°C")
```

### Example Data Sources

The package works with any NetCDF or Zarr data that has the required dimensions (time, latitude, longitude). Some compatible data sources:

1. ERA5 Reanalysis:
```python
import cdsapi

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'variable': '2m_temperature',
        'year': '2023',
        'month': '07',
        'day': list(range(1, 32)),
        'time': [f'{h:02d}:00' for h in range(24)],
        'format': 'netcdf'
    },
    'era5_temp.nc'
)
```

2. GFS Forecast Data:
```python
import xarray as xr
from datetime import datetime

date = datetime.now().strftime('%Y%m%d')
url = f'https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date}'
gfs = xr.open_dataset(url)
temp = gfs['tmp2m']  # 2-meter temperature
```

## Validation Results

### Forecast Skill
![Forecast Skill](plots/com_forecast_skill.png)

The plot shows:
- Top: Temperature bias (°C)
- Middle: Root Mean Square Error (°C)
- Bottom: Pattern correlation

### Forecast Validation
![Forecast Validation](plots/forecast_validation.png)

The validation plot demonstrates:
- Top left: Model forecast temperature field
- Top right: Ground truth temperature field
- Bottom left: Detected extreme events in forecast
- Bottom right: Detected extreme events in observations

## Credits and Acknowledgments

This project builds upon several groundbreaking works in ML weather forecasting:

- **WeatherBench2** ([Link](https://sites.research.google/weatherbench/)) - The foundational benchmark for ML weather forecasting that inspired this project
- **Pangu-Weather** ([Paper](https://arxiv.org/abs/2211.02556)) - Huawei's transformer-based weather forecasting model
- **Aurora** ([Paper](https://arxiv.org/pdf/2405.13063)) - A foundation model for weather forecasting
- **GraphCast** ([Paper](https://arxiv.org/abs/2212.12794)) - Google DeepMind's graph neural network approach
- **FourCastNet** ([Paper](https://arxiv.org/abs/2202.11214)) - NVIDIA's Fourier neural operator model
- **ERA5** ([Documentation](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)) - ECMWF's reanalysis dataset

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{wx_extreme_2024,
  title={WX-Extreme: Advanced Evaluation Framework for Extreme Weather Events in ML Models},
  author={Hridoy, Al-Ekram Elahee},
  year={2024},
  url={https://github.com/alhridoy/weatherbench2},
}
```

Please also cite the relevant papers for WeatherBench2, Pangu-Weather, and other models used in your evaluation:

```bibtex
@article{weatherbench2_2023,
  title={WeatherBench 2: A benchmark for the next generation of data-driven weather forecasts},
  author={Rasp, Stephan and others},
  journal={Bulletin of the American Meteorological Society},
  year={2023}
}

@article{pangu_weather_2022,
  title={Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast},
  author={Du, Kaifeng and others},
  journal={arXiv preprint arXiv:2211.02556},
  year={2022}
}

@article{aurora_2023,
  title={Aurora: A foundation model for weather forecasting},
  author={Lam, Ryan and others},
  journal={arXiv preprint arXiv:2405.13063},
  year={2024}
}
```
