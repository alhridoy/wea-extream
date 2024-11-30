import xarray as xr
import matplotlib.pyplot as plt
from wx_extreme.core.detector import ExtremeHeatDetector
from wx_extreme.core.evaluator import evaluate_heat_events

# Load temperature data (example)
ds = xr.tutorial.open_dataset('air_temperature')
temperature = ds.air

# Create detector
detector = ExtremeHeatDetector(
    threshold_method="percentile",
    threshold_value=95,
    min_duration=3,
    spatial_scale=2.0
)

# Detect events
events = detector.detect_events(temperature)

# Evaluate events
metrics = evaluate_heat_events(temperature, events)
print("Heat wave metrics:", metrics)

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

temperature.mean('time').plot(ax=ax1)
ax1.set_title('Mean Temperature')

events.mean('time').plot(ax=ax2)
ax2.set_title('Heat Wave Frequency')

plt.tight_layout()
plt.show() 