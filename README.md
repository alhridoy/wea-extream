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
