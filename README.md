# WX-Extreme

Advanced Evaluation Framework for Extreme Weather Events in ML Models

## Overview

WX-Extreme is a Python package for evaluating extreme weather events in machine learning weather forecasting models. It provides tools for:
- Detecting extreme weather events using various methods
- Evaluating forecast skill for extreme events
- Comparing ML model predictions with traditional numerical weather prediction
- Analyzing spatial and temporal patterns of extreme events

## Installation

```bash
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
  url = {https://github.com/alhridoy/wx-extreme}
}
```
