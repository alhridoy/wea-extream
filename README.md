# WX-Extreme

Advanced Evaluation Framework for Extreme Weather Events in AI Weather Models

## Overview

WX-Extreme is a Python package for evaluating extreme weather events in machine learning weather forecasting models. This project is inspired by [WeatherBench2](https://sites.research.google/weatherbench/) and addresses specific challenges in evaluating extreme weather events in ML weather models.

### Key Differences from WeatherBench2
While WeatherBench2 provides an excellent foundation for general forecast evaluation, WX-Extreme focuses specifically on:
- Event-specific detection and validation
- Operational forecast evaluation with multiple initialization times
- Regional analysis capabilities
- Memory-efficient processing for large datasets
- Extreme event-specific metrics

## Models and Data Sources

This project currently supports evaluation of:
- [Pangu-Weather](https://arxiv.org/abs/2211.02556) - A transformer-based weather forecasting model
- [Aurora](https://arxiv.org/pdf/2405.13063) - A foundation model for weather forecasting
- [GraphCast](https://arxiv.org/abs/2212.12794) - Google DeepMind's graph neural network-based weather model
- [FourCastNet](https://arxiv.org/abs/2202.11214) - NVIDIA's Fourier neural operator-based model
- [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) - ECMWF's fifth generation reanalysis dataset

## Installation

```bash
# Clone the repository
git clone https://github.com/alhridoy/wea-extream.git
cd wea-extream

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from wx_extreme.core.detector import ExtremeEventDetector
from wx_extreme.core.evaluator import evaluate_extremes

# Initialize detector
detector = ExtremeEventDetector(
    threshold_method="percentile",
    threshold_value=95,
    min_duration=3
)

# Detect events
events = detector.detect_events(data)

# Evaluate results
metrics = evaluate_extremes(data, events)
```

## Basic Usage

### 1. Detecting Extreme Events

```python
import xarray as xr
from wx_extreme.core.detector import ExtremeEventDetector

# Load your temperature data (example with ERA5)
data = xr.open_dataset('temperature.nc')
temp = data['t2m']  # 2-meter temperature

# Initialize detector
detector = ExtremeEventDetector(
    threshold_method="percentile",  # or "absolute"
    threshold_value=95,            # 95th percentile
    min_duration=3                 # minimum 3 days
)

# Detect extreme events
events = detector.detect_events(temp)
```

### 2. Evaluating Model Forecasts

```python
from wx_extreme.core.evaluator import evaluate_extremes
from wx_extreme.core.metrics import MLModelMetrics

# Load forecast and observation data
forecast = xr.open_dataset('forecast.nc')['t2m']
obs = xr.open_dataset('observation.nc')['t2m']

# Basic evaluation
metrics = evaluate_extremes(forecast, events, reference=obs)
print(f"Mean intensity: {metrics['mean_intensity']:.2f}°C")
print(f"Event duration: {metrics['mean_duration']:.1f} days")

# Advanced ML metrics
ml_metrics = MLModelMetrics()
skill_score = ml_metrics.extreme_value_skill_score(
    forecast, 
    obs, 
    threshold=30  # °C
)
pattern_score = ml_metrics.pattern_prediction_score(
    forecast, 
    obs, 
    spatial_scale=1.0
)
```

### 3. Visualizing Results

```python
import matplotlib.pyplot as plt
from wx_extreme.utils.plot_utils import plot_model_comparison_heatmap

# Plot temperature bias
fig, ax = plt.subplots(figsize=(10, 6))
plot_model_comparison_heatmap(
    forecast,
    obs,
    metric='bias',
    title='Temperature Bias',
    ax=ax
)
plt.savefig('bias_map.png')

# Plot extreme events
events.plot(
    figsize=(12, 6),
    cmap='Reds',
    add_colorbar=True
)
plt.title('Detected Extreme Events')
plt.savefig('events_map.png')
```

### 4. Working with Different Data Sources

#### ERA5 Data

```python
import cdsapi

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'variable': '2m_temperature',
        'year': '2023',
        'month': '07',
        'day': list(map(str, range(1, 32))),
        'time': [f'{h:02d}:00' for h in range(24)],
        'format': 'netcdf'
    },
    'era5_temp.nc'
)
```

#### Pangu-Weather Data

```python
import requests
import xarray as xr

# Download Pangu-Weather forecast
base_url = "https://data.pangeo.io/pangeo-weather/"
forecast = xr.open_zarr(base_url + "forecast.zarr")

# Process specific variables
t2m = forecast['t2m'].sel(level=2)
```

#### GFS Data

```python
from datetime import datetime
import xarray as xr

# Open GFS forecast
date = datetime.now().strftime('%Y%m%d')
url = f'https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date}'
ds = xr.open_dataset(url)
t2m = ds['tmp2m']
```

### 5. Batch Processing

```python
import pandas as pd
from wx_extreme.core.detector import ExtremeEventDetector

# Process multiple time periods
dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
detector = ExtremeEventDetector(threshold_method="percentile", threshold_value=95)

results = []
for date in dates:
    # Load data for each month
    data = xr.open_dataset(f'data_{date.strftime("%Y%m")}.nc')
    
    # Detect events
    events = detector.detect_events(data['t2m'])
    
    # Store results
    results.append({
        'date': date,
        'event_count': events.sum().item(),
        'max_intensity': data['t2m'].where(events).max().item()
    })

# Create summary DataFrame
summary = pd.DataFrame(results)
```

For more detailed examples and use cases, check out the example notebooks in the `examples/` directory:
- `model_evaluation.ipynb`: Comprehensive model evaluation workflow
- `heatwave_detection.py`: Detailed heatwave analysis example
- `validate_forecast.py`: Forecast validation pipeline
- `analyze_heatwave.py`: In-depth heatwave case study

## Validation Results

### Pangu-Weather Forecast Skill

We evaluated Pangu-Weather's 2-meter temperature forecasts against ERA5 reanalysis data for June-July 2021, focusing on the Western North American heat wave period. The evaluation includes both general forecast skill and extreme event detection capabilities.

#### Basic Forecast Skill
![Basic Forecast Skill](plots/forecast_skill.png)

The plot shows:
- Top: Temperature bias (°C)
- Middle: Root Mean Square Error (°C)
- Bottom: Pattern correlation

#### Forecast Validation
![Forecast Validation](plots/forecast_validation.png)

The validation plot shows:
- Top left: Forecast temperature field
- Top right: ERA5 analysis temperature field
- Bottom left: Detected extreme events in forecast
- Bottom right: Detected extreme events in analysis

Each line represents a different forecast initialization time, showing how forecast skill evolves with lead time.

#### Comprehensive Metrics
![Comprehensive Metrics](plots/comprehensive_metrics.png)

The comprehensive evaluation includes:
1. Basic Error Metrics:
   - Bias: Systematic error in forecasts
   - RMSE: Overall magnitude of errors
   - MAE: Average absolute error

2. Pattern Metrics:
   - Pattern Correlation: Spatial correlation
   - ACC: Anomaly Correlation Coefficient

3. Skill Scores:
   - MSE Skill Score: Improvement over climatology
   - Murphy Score: Forecast skill accounting for variance

4. Extreme Event Metrics:
   - POD: Probability of Detection
   - FAR: False Alarm Ratio
   - CSI: Critical Success Index
   - HSS: Heidke Skill Score
   - ETS: Equitable Threat Score

Key findings:
- Strong pattern correlation (>0.98) throughout the forecast period
- RMSE increases from ~2.2°C at day 1 to ~3.8°C at day 10
- Small cold bias that increases with forecast lead time
- High extreme event detection skill (HSS > 0.7) for the first 5 days

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
