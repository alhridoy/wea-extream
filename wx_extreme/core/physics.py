"""Physical consistency validation module."""

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy import constants

from wx_extreme.utils import thermo


@dataclasses.dataclass
class PhysicsValidator:
    """Validator for physical consistency in weather forecasts.
    
    This class implements various checks for physical relationships and
    conservation laws that should be maintained in weather forecasts:
    1. Hydrostatic balance
    2. Geostrophic wind balance
    3. Thermal wind relationship
    4. Mass conservation
    5. Energy conservation
    6. Moisture conservation
    """
    
    def __init__(
        self,
        tolerance: float = 0.1,
        scale_dependent: bool = True,
    ):
        """Initialize the validator.
        
        Args:
            tolerance: Maximum allowed relative error in physical relationships
            scale_dependent: Whether to adjust tolerance based on spatial scale
        """
        self.tolerance = tolerance
        self.scale_dependent = scale_dependent
        
        # Physical constants
        self.g = constants.g  # gravitational acceleration
        self.R = constants.R  # gas constant
        self.cp = 1004.0  # specific heat at constant pressure
        self.omega = 7.2921e-5  # Earth's angular velocity
        self.L = 2.5e6  # latent heat of vaporization

    def validate(
        self,
        dataset: xr.Dataset,
        region: Optional[str] = None,
    ) -> xr.Dataset:
        """Validate physical consistency of the forecast.
        
        Args:
            dataset: Forecast dataset to validate
            region: Optional region name to focus validation
            
        Returns:
            Dataset containing validation metrics
        """
        metrics = {}
        
        # Basic thermodynamic checks
        if all(var in dataset for var in ["temperature", "pressure"]):
            metrics["hydrostatic_balance"] = self._check_hydrostatic_balance(
                dataset.temperature,
                dataset.pressure,
                dataset.geopotential if "geopotential" in dataset else None,
            )
        
        # Wind balance checks
        if all(var in dataset for var in [
            "u_component_of_wind",
            "v_component_of_wind",
            "pressure",
        ]):
            metrics["geostrophic_balance"] = self._check_geostrophic_balance(
                dataset.u_component_of_wind,
                dataset.v_component_of_wind,
                dataset.pressure,
                dataset.latitude,
            )
            
            if "temperature" in dataset:
                metrics["thermal_wind"] = self._check_thermal_wind(
                    dataset.u_component_of_wind,
                    dataset.v_component_of_wind,
                    dataset.temperature,
                    dataset.pressure,
                    dataset.latitude,
                )
        
        # Conservation checks
        if "pressure" in dataset:
            metrics["mass_conservation"] = self._check_mass_conservation(
                dataset.pressure
            )
            
        if all(var in dataset for var in [
            "temperature",
            "specific_humidity",
            "surface_pressure",
        ]):
            metrics["energy_conservation"] = self._check_energy_conservation(
                dataset.temperature,
                dataset.specific_humidity,
                dataset.surface_pressure,
            )
            
            metrics["moisture_conservation"] = self._check_moisture_conservation(
                dataset.specific_humidity,
                dataset.surface_pressure,
            )
        
        return xr.Dataset(metrics)

    def _check_hydrostatic_balance(
        self,
        temperature: xr.DataArray,
        pressure: xr.DataArray,
        geopotential: Optional[xr.DataArray] = None,
    ) -> xr.DataArray:
        """Check hydrostatic balance: dp/dz = -ρg."""
        # Calculate density from ideal gas law
        density = pressure / (self.R * temperature)
        
        # Calculate vertical pressure gradient
        if "level" in pressure.dims:
            dp_dz = pressure.differentiate("level")
            if geopotential is not None:
                dz = geopotential.differentiate("level") / self.g
                dp_dz = dp_dz / dz
        else:
            return xr.DataArray(np.nan)  # Can't check without vertical levels
        
        # Compare with theoretical value
        theoretical = -density * self.g
        error = abs((dp_dz - theoretical) / theoretical)
        
        if self.scale_dependent:
            # Allow larger errors at smaller scales
            error = error.coarsen(
                latitude=5, longitude=5, boundary="trim"
            ).mean()
        
        return error

    def _check_geostrophic_balance(
        self,
        u_wind: xr.DataArray,
        v_wind: xr.DataArray,
        pressure: xr.DataArray,
        latitude: xr.DataArray,
    ) -> xr.DataArray:
        """Check geostrophic wind balance."""
        # Coriolis parameter
        f = 2 * self.omega * np.sin(np.deg2rad(latitude))
        
        # Calculate pressure gradients
        dx = spatial_utils.get_dx(latitude, longitude)
        dy = spatial_utils.get_dy(latitude)
        
        dp_dx = pressure.differentiate("longitude") / dx
        dp_dy = pressure.differentiate("latitude") / dy
        
        # Calculate geostrophic wind components
        ug = -(1 / (f * density)) * dp_dy
        vg = (1 / (f * density)) * dp_dx
        
        # Compare with actual winds
        u_error = abs((u_wind - ug) / ug)
        v_error = abs((v_wind - vg) / vg)
        
        if self.scale_dependent:
            # Allow larger errors at smaller scales
            u_error = u_error.coarsen(
                latitude=5, longitude=5, boundary="trim"
            ).mean()
            v_error = v_error.coarsen(
                latitude=5, longitude=5, boundary="trim"
            ).mean()
        
        return (u_error + v_error) / 2

    def _check_thermal_wind(
        self,
        u_wind: xr.DataArray,
        v_wind: xr.DataArray,
        temperature: xr.DataArray,
        pressure: xr.DataArray,
        latitude: xr.DataArray,
    ) -> xr.DataArray:
        """Check thermal wind relationship."""
        if "level" not in pressure.dims:
            return xr.DataArray(np.nan)
        
        # Calculate vertical wind shear
        du_dz = u_wind.differentiate("level")
        dv_dz = v_wind.differentiate("level")
        
        # Calculate horizontal temperature gradients
        dx = spatial_utils.get_dx(latitude, longitude)
        dy = spatial_utils.get_dy(latitude)
        
        dT_dx = temperature.differentiate("longitude") / dx
        dT_dy = temperature.differentiate("latitude") / dy
        
        # Thermal wind equations
        f = 2 * self.omega * np.sin(np.deg2rad(latitude))
        theoretical_du_dz = -(self.R / (f * pressure)) * dT_dy
        theoretical_dv_dz = (self.R / (f * pressure)) * dT_dx
        
        # Compare with actual wind shear
        u_error = abs((du_dz - theoretical_du_dz) / theoretical_du_dz)
        v_error = abs((dv_dz - theoretical_dv_dz) / theoretical_dv_dz)
        
        if self.scale_dependent:
            u_error = u_error.coarsen(
                latitude=5, longitude=5, boundary="trim"
            ).mean()
            v_error = v_error.coarsen(
                latitude=5, longitude=5, boundary="trim"
            ).mean()
        
        return (u_error + v_error) / 2

    def _check_mass_conservation(
        self,
        pressure: xr.DataArray,
    ) -> xr.DataArray:
        """Check conservation of mass (surface pressure tendency)."""
        if "time" not in pressure.dims:
            return xr.DataArray(np.nan)
        
        # Calculate pressure tendency
        dp_dt = pressure.differentiate("time")
        
        # Pressure tendency should be small compared to pressure
        relative_tendency = abs(dp_dt / pressure)
        
        if self.scale_dependent:
            relative_tendency = relative_tendency.coarsen(
                latitude=5, longitude=5, boundary="trim"
            ).mean()
        
        return relative_tendency

    def _check_energy_conservation(
        self,
        temperature: xr.DataArray,
        specific_humidity: xr.DataArray,
        surface_pressure: xr.DataArray,
    ) -> xr.DataArray:
        """Check conservation of total energy."""
        if "time" not in temperature.dims:
            return xr.DataArray(np.nan)
        
        # Calculate total energy (internal + latent)
        energy = (
            self.cp * temperature +
            self.L * specific_humidity
        ) * surface_pressure
        
        # Energy tendency should be small
        energy_tendency = abs(
            energy.differentiate("time") / energy
        )
        
        if self.scale_dependent:
            energy_tendency = energy_tendency.coarsen(
                latitude=5, longitude=5, boundary="trim"
            ).mean()
        
        return energy_tendency

    def _check_moisture_conservation(
        self,
        specific_humidity: xr.DataArray,
        surface_pressure: xr.DataArray,
    ) -> xr.DataArray:
        """Check conservation of total moisture."""
        if "time" not in specific_humidity.dims:
            return xr.DataArray(np.nan)
        
        # Total column water
        water = specific_humidity * surface_pressure
        
        # Water tendency should be small
        water_tendency = abs(
            water.differentiate("time") / water
        )
        
        if self.scale_dependent:
            water_tendency = water_tendency.coarsen(
                latitude=5, longitude=5, boundary="trim"
            ).mean()
        
        return water_tendency
