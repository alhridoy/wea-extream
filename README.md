# WX-Extreme: Advanced Weather Prediction Model Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

WX-Extreme is a comprehensive Python package for evaluating machine learning weather prediction models, with a special focus on extreme events and physical consistency. It provides tools for detecting extreme weather events, validating physical relationships, and assessing forecast quality from an end-user perspective.

## 🌟 Key Features

- **Extreme Event Detection**
  - Percentile-based thresholds
  - Return period analysis
  - Peaks-over-threshold method
  - Compound event detection

- **Physical Consistency Validation**
  - Hydrostatic balance
  - Geostrophic wind relationships
  - Thermal wind balance
  - Conservation laws

- **Statistical Analysis**
  - Extreme value distributions
  - Return period calculations
  - Exceedance probabilities
  - Block maxima analysis

- **Spatial Analysis**
  - Grid metrics calculations
  - Distance computations
  - Area calculations
  - Coordinate transformations

- **Visualization**
  - Publication-quality plots
  - Map visualizations
  - Return period curves
  - Wind field plotting

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/wx-extreme.git
cd wx-extreme

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 📖 Quick Start

```python
import xarray as xr
from wx_extreme.utils import spatial_utils, met_utils, stat_utils, plot_utils

# Load your weather data
ds = xr.open_dataset('weather_data.nc')

# Calculate grid metrics
dx, dy = spatial_utils.get_grid_spacing(ds.latitude, ds.longitude)

# Analyze extremes
temp_95 = stat_utils.calculate_percentile(ds.temperature, q=95, dim='time')
exceed_prob = stat_utils.exceedance_probability(ds.temperature, threshold=25)

# Check physical consistency
theta = met_utils.potential_temperature(ds.temperature, ds.pressure)

# Create visualization
plot_utils.plot_field(temp_95, colorbar_label='Temperature (°C)')
```

## 📚 Documentation

For detailed documentation and examples, check out our [examples](examples/) directory:

- [Basic Usage](examples/basic_usage.ipynb): Introduction to core functionality
- [Extreme Event Analysis](examples/extreme_events.ipynb): Detailed extreme event detection
- [Physical Validation](examples/physics.ipynb): Physical consistency checks
- [Visualization Guide](examples/visualization.ipynb): Advanced plotting examples

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [WeatherBench2](https://github.com/google-research/weatherbench2) for inspiration and benchmarking
- [xarray](https://xarray.dev/) for excellent data structures
- [cartopy](https://scitools.org.uk/cartopy/docs/latest/) for map visualizations

## 📧 Contact

For questions and feedback:
- Create an issue in the repository
- Email: your.email@example.com
- Twitter: [@wx_extreme](https://twitter.com/wx_extreme)

## 📝 Citation

If you use WX-Extreme in your research, please cite:

```bibtex
@software{wx_extreme2023,
  author = {Your Name},
  title = {WX-Extreme: Advanced Weather Prediction Model Evaluation},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/your-username/wx-extreme}
}
