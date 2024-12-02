# WX-Extreme

Advanced Evaluation Framework for Extreme Weather Events in ML Models

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
- [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) - ECMWF's fifth generation reanalysis dataset
- [GraphCast](https://arxiv.org/abs/2212.12794) - Google DeepMind's graph neural network-based weather model
- [FourCastNet](https://arxiv.org/abs/2202.11214) - NVIDIA's Fourier neural operator-based model

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

## User Guide

### Data Requirements

Your input data should be in one of these formats:
1. **NetCDF (.nc) files** with:
   - Temperature data in Kelvin or Celsius
   - Dimensions: time, latitude, longitude
   - For forecasts: init_time and fcst_hour dimensions
   - Coordinates in either 0-360° or -180-180° longitude convention

2. **Zarr (.zarr) files** with similar structure

### Example: Evaluating a Heat Wave Forecast

```python
import xarray as xr
from wx_extreme.core.detector import ExtremeEventDetector
from wx_extreme.core.evaluator import evaluate_extremes

# 1. Load your forecast and ground truth data
forecast = xr.open_dataset("path/to/your/forecast.nc")  # Your model's forecast
ground_truth = xr.open_dataset("path/to/ground_truth.nc")  # ERA5 or observations

# 2. Initialize the detector
detector = ExtremeEventDetector(
    threshold_method="percentile",  # or "absolute"
    threshold_value=95,            # 95th percentile
    min_duration=3,               # minimum 3 timesteps
    spatial_scale=2.0            # minimum 2° spatial extent
)

# 3. Detect extreme events
forecast_events = detector.detect_events(forecast)
truth_events = detector.detect_events(ground_truth)

# 4. Evaluate the results
metrics = evaluate_extremes(
    forecast=forecast,
    ground_truth=ground_truth,
    forecast_events=forecast_events,
    truth_events=truth_events
)

print(metrics)
```

### Example: Operational Forecast Evaluation

```python
import pandas as pd
from wx_extreme.core.metrics import evaluate_forecast_skill

# Load multiple forecast initializations
forecasts = xr.open_dataset("path/to/forecasts.nc")
era5 = xr.open_dataset("path/to/era5.nc")

# Evaluate skill by lead time
skill_df = evaluate_forecast_skill(
    forecasts,
    era5,
    lead_times=range(24, 241, 24)  # 1-10 days
)

# Plot results
from wx_extreme.utils.plot_utils import plot_forecast_skill
fig = plot_forecast_skill(skill_df)
fig.savefig('forecast_skill.png')
```

### Accessing Pre-prepared Data

We provide easy access to common datasets:

```python
from wx_extreme.core.metrics import get_panguweather_t2_forecasts, load_era5_data

# Load Pangu-Weather forecasts
ml_forecasts = get_panguweather_t2_forecasts()

# Load ERA5 data
era5_data = load_era5_data()
```

### Memory Efficiency for Large Datasets

For large datasets, use chunking:

```python
import dask

# Set up chunking strategy
chunks = {
    'time': 50,
    'latitude': 20,
    'longitude': 20
}

# Load data with chunks
data = xr.open_dataset("large_dataset.nc", chunks=chunks)

# Use dask for parallel processing
with dask.config.set(scheduler='threads'):
    result = detector.detect_events(data)
```

## Validation Results

### Pangu-Weather Forecast Skill

We evaluated Pangu-Weather's 2-meter temperature forecasts against ERA5 reanalysis data for June-July 2021. The results show:

- Strong pattern correlation (>0.98) throughout the forecast period
- RMSE increases from ~2.2°C at day 1 to ~3.8°C at day 10
- Small cold bias that increases with forecast lead time

![Forecast Skill Metrics](plots/forecast_skill.png)

The plot shows:
- Top: Temperature bias (°C)
- Middle: Root Mean Square Error (°C)
- Bottom: Pattern correlation

Each line represents a different forecast initialization time, showing how forecast skill evolves with lead time.

## Credits and Acknowledgments

This project builds upon several groundbreaking works in ML weather forecasting:

- **WeatherBench2** ([Link](https://sites.research.google/weatherbench/)) - The foundational benchmark for ML weather forecasting that inspired this project
- **Pangu-Weather** ([Paper](https://arxiv.org/abs/2211.02556)) - Huawei's transformer-based weather forecasting model
- **GraphCast** ([Paper](https://arxiv.org/abs/2212.12794)) - Google DeepMind's graph neural network approach
- **FourCastNet** ([Paper](https://arxiv.org/abs/2202.11214)) - NVIDIA's Fourier neural operator model
- **ERA5** ([Documentation](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)) - ECMWF's reanalysis dataset

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use WX-Extreme in your research, please cite:

```bibtex
@software{wx_extreme2024,
  author = {Al-Ekram Elahee Hridoy},
  title = {WX-Extreme: Advanced Evaluation Framework for Extreme Weather Events in ML Models},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/alhridoy/wea-extream}
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
```
