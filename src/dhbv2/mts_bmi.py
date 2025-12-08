"""
BMI wrapper for interfacing δHBV2.0 MTS (hourly) with the NOAA-OWP NextGen
framework.

@Leo Lonzarich
"""

import json
import logging
import os
import time
from typing import Optional, Union, Any

import numpy as np
import torch
import yaml
from bmipy import Bmi
from dmg.core.utils.factory import import_data_sampler
from dmg.core.utils.dates import Dates

from dmg import ModelHandler
from numpy.typing import NDArray
from sklearn.exceptions import DataDimensionalityWarning
from dhbv2.utils import bmi_array

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

root_path = os.path.dirname(os.path.abspath(__file__))


# -------------------------------------------- #
# Dynamic input variables (CSDMS standard names)
# -------------------------------------------- #
_dynamic_input_vars = [
    ('atmosphere_water__liquid_equivalent_precipitation_rate', 'mm h-1'),
    ('land_surface_air__temperature', 'degK'),
    ('land_surface_water__potential_evaporation_volume_flux', 'mm h-1'),
]

# ------------------------------------------- #
# Static input variables (CSDMS standard names)
# ------------------------------------------- #
_static_input_vars = [
    ('ratio__mean_potential_evapotranspiration__mean_precipitation', '-'),
    ('atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate', 'mm d-1'),
    ('land_surface_water__Hargreaves_potential_evaporation_volume_flux', 'mm d-1'),
    ('land_vegetation__normalized_diff_vegetation_index', '-'),
    ('free_land_surface_water', 'mm d-1'),
    ('basin__mean_of_slope', 'm km-1'),
    ('soil_sand__grid', 'km2'),
    ('soil_clay__grid', 'km2'),
    ('soil_silt__grid', 'km2'),
    ('land_surface_water__glacier_fraction', 'percent'),
    ('soil_clay__attr', 'percent'),
    ('soil_gravel__attr', 'percent'),
    ('soil_sand__attr', 'percent'),
    ('soil_silt__attr', 'percent'),
    ('basin__mean_of_elevation', 'm'),
    ('atmosphere_water__daily_mean_of_temperature', 'degC'),
    ('land_surface_water__permafrost_fraction', '-'),
    ('bedrock__permeability', 'm2'),
    ('p_seasonality', '-'),
    ('land_surface_water__potential_evaporation_volume_flux_seasonality', '-'),
    ('land_surface_water__snow_fraction', 'percent'),
    ('atmosphere_water__precipitation_falling_as_snow_fraction', 'percent'),
    ('soil_clay__volume_fraction', 'percent'),
    ('soil_gravel__volume_fraction', 'percent'),
    ('soil_sand__volume_fraction', 'percent'),
    ('soil_silt__volume_fraction', 'percent'),
    ('soil_active-layer__porosity', '-'),
    ('basin__area', 'km2'),
]

# ------------------------------------- #
# Output variables (CSDMS standard names)
# ------------------------------------- #
_output_vars = [
    ('land_surface_water__runoff_volume_flux', 'm3 s-1'),
]

# ---------------------------------------------- #
# Internal variable names <-> CSDMS standard names
# ---------------------------------------------- #
_var_name_internal_map = {
    # ----------- Dynamic inputs -----------
    'precip': 'atmosphere_water__liquid_equivalent_precipitation_rate',
    'temp': 'land_surface_air__temperature',
    'pet': 'land_surface_water__potential_evaporation_volume_flux',
    # ----------- Static inputs -----------
    'aridity': 'ratio__mean_potential_evapotranspiration__mean_precipitation',
    'meanP': 'atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate',
    'ETPOT_Hargr': 'land_surface_water__Hargreaves_potential_evaporation_volume_flux',
    'NDVI': 'land_vegetation__normalized_diff_vegetation_index',
    'FW': 'free_land_surface_water',
    'meanslope': 'basin__mean_of_slope',
    'SoilGrids1km_sand': 'soil_sand__grid',
    'SoilGrids1km_clay': 'soil_clay__grid',
    'SoilGrids1km_silt': 'soil_silt__grid',
    'glaciers': 'land_surface_water__glacier_fraction',
    'HWSD_clay': 'soil_clay__attr',
    'HWSD_gravel': 'soil_gravel__attr',
    'HWSD_sand': 'soil_sand__attr',
    'HWSD_silt': 'soil_silt__attr',
    'meanelevation': 'basin__mean_of_elevation',
    'meanTa': 'atmosphere_water__daily_mean_of_temperature',
    'permafrost': 'land_surface_water__permafrost_fraction',
    'permeability': 'bedrock__permeability',
    'seasonality_P': 'p_seasonality',
    'seasonality_PET': 'land_surface_water__potential_evaporation_volume_flux_seasonality',
    'snow_fraction': 'land_surface_water__snow_fraction',
    'snowfall_fraction': 'atmosphere_water__precipitation_falling_as_snow_fraction',
    'T_clay': 'soil_clay__volume_fraction',
    'T_gravel': 'soil_gravel__volume_fraction',
    'T_sand': 'soil_sand__volume_fraction',
    'T_silt': 'soil_silt__volume_fraction',
    'Porosity': 'soil_active-layer__porosity',
    'uparea': 'basin__area',
    # ----------- Outputs -----------
    'streamflow': 'land_surface_water__runoff_volume_flux',
}

_var_name_external_map = {v: k for k, v in _var_name_internal_map.items()}


def map_to_external(name: str):
    """Return the external name (exposed via BMI) for a given internal name."""
    return _var_name_internal_map[name]


def map_to_internal(name: str):
    """Return the internal name for a given external name (exposed via BMI)."""
    return _var_name_external_map[name]


# =============================================================================#
# MAIN BMI >>>>
# =============================================================================#


class MtsDeltaModelBmi(Bmi):
    """
    δHBV2.0 MTS BMI: NextGen-compatible, differentiable, physics-informed ML
    model for hydrologic forecasting (Yang et al., 2025; Song et al., 2024).

    Note: BMI can only run forward inference. Training code will be released in
        the δMG package (https://github.com/mhpi/generic_deltamodel) at a later
        date.
    """

    _att_map = {
        'model_name': 'δHBV2.0 MTS',
        'version': '0.1',
        'author_name': 'Leo Lonzarich',
        'time_step_size': 3600,
        'time_units': 's',
    }

    def __init__(
        self,
        verbose=False,
    ) -> None:
        """Create a δHBV2.0 MTS BMI ready for initialization.

        Parameters
        ----------
        config_path
            Path to the BMI configuration file.
        verbose
            Enables debug print statements if True.
        """
        super().__init__()
        self._name = self._att_map['model_name']
        self._time_units = self._att_map['time_units']
        self._time_step_size = self._att_map['time_step_size']
        self._model = None
        self._initialized = False
        self.verbose = verbose

        self._var_loc = 'node'
        self._var_grid_id = 0

        self._timestep = 0
        self._start_time = 0.0
        self._end_time = np.finfo('d').max

        self.bmi_config = None
        self.model_config = None

        # Track BMI processing time
        t_start = time.time()
        self.proc_time = 0.0

        # Initialize input/output vars
        self._dynamic_var = self._set_vars(_dynamic_input_vars, bmi_array([]))
        self._static_var = self._set_vars(_static_input_vars, bmi_array([]))
        self._output_vars = self._set_vars(_output_vars, bmi_array([]))

        self.proc_time += time.time() - t_start
        if self.verbose:
            log.info(f"BMI init took {time.time() - t_start} s")

    @staticmethod
    def _set_vars(
        vars: list[tuple[str, str]],
        var_value: NDArray,
    ) -> dict[str, dict[str, Union[NDArray, str]]]:
        """Set the values of given variables."""
        var_dict = {}
        for item in vars:
            var_dict[item[0]] = {'value': var_value.copy(), 'units': item[1]}
        return var_dict

    def initialize(self, config_path: Optional[str] = None) -> None:
        """(Control function) Initialize the BMI model.

        This BMI operates in two modes:
            (Necessesitated by the fact that dhBV 2.0's internal NN must forward
            on all data at once. <-- Forwarding on each timestep one-by-one with
            saving/loading hidden states would slash LSTM performance. However,
            feeding in hidden states day-by-day leeds to great efficiency losses
            vs simply feeding all data at once due to carrying gradients at each
            step.)

            1) Feed all input data to BMI before
                'bmi.initialize()'. Then internal model is forwarded on all data
                and generates predictions during '.initialize()'.

            2) Run '.initialize()', then pass data day by day as normal during
                'bmi.update()'. If forwarding period is sufficiently small (say,
                <100 days), then forwarding LSTM on individual days with saved
                states is reasonable.

        To this end, a configuration file can be specified either during
        `bmi.__init__()`, or during `.initialize()`. If running BMI as type (1),
        config must be passed in the former, otherwise passed in the latter for (2).

        Parameters
        ----------
        config_path
            Path to the BMI configuration file.
        """
        t_start = time.time()

        # Read BMI configuration file
        try:
            with open(config_path) as f:
                self.bmi_config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load BMI configuration: {e}") from e

        # Read model configuration
        try:
            model_config_path = os.path.join(
                root_path,
                '..',
                '..',
                'ngen_resources/data/dhbv2_mts/',
                self.bmi_config.get('model_config'),
            )
            with open(model_config_path) as f:
                self.model_config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model configuration: {e}") from e

        self.model_config = self.initialize_config(self.model_config)
        self.model_config['model_dir'] = os.path.join(
            root_path,
            '..',
            '..',
            'ngen_resources/data/dhbv2_mts/',
            self.model_config.get('model_dir'),
        )
        self.device = self.model_config['device']
        self.internal_dtype = self.model_config['dtype']
        self.external_dtype = eval(self.bmi_config['dtype'])
        self.sampler = import_data_sampler(self.model_config['data_sampler'])(
            self.model_config,
        )

        # Load static variables from BMI config
        for name in self._static_var.keys():
            ext_name = map_to_internal(name)
            if ext_name in self.bmi_config.keys():
                self._static_var[name]['value'] = bmi_array(self.bmi_config[ext_name])
            else:
                log.warning(f"Static variable '{name}' not in BMI config. Skipping.")

        # Set simulation parameters
        self._time_step_size = self.bmi_config.get(
            'time_step_size',
            self._time_step_size,
        )
        self.current_time = self.bmi_config.get('start_time', self._start_time)
        self._end_time = self.bmi_config.get('end_time', self._end_time)

        # Load a trained model
        try:
            self._model = self._load_model(self.model_config).to(self.device)
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}") from e

        # Track BMI runtime
        self.proc_time += time.time() - t_start
        if self.verbose:
            log.info(
                f"BMI Initialize took {time.time() - t_start:.4f} s | Total runtime: {self.proc_time:.4f} s",
            )

    def update(self) -> None:
        """(Control function) Advance model state by one time step."""
        t_start = time.time()

        data_dict = self._format_inputs()
        predictions = self._do_forward(data_dict)
        self._format_outputs(predictions)

        self._timestep += 1

        # Track BMI runtime
        self.proc_time += time.time() - t_start
        if self.verbose:
            log.info(
                f"BMI Update took {time.time() - t_start:.4f} s | Total runtime: {self.proc_time:.4f} s",
            )

    def update_until(self, end_time: float) -> None:
        """(Control function) Update model until a particular time.

        Note: Models should be trained standalone with dPLHydro_PMI first before
        forward predictions with this BMI.

        Parameters
        ----------
        end_time
            Time to run model until.
        """
        t_start = time.time()

        if end_time < self.get_current_time():
            log.warning(
                f"No update performed: end_time ({end_time}) <= current time ({self.get_current_time()}).",
            )
            return None

        n_steps, remainder = divmod(
            end_time - self.get_current_time(),
            self.get_time_step(),
        )

        if remainder != 0:
            log.warning(
                f"End time is not multiple of time step size. Updating until: {end_time - remainder}",
            )

        for _ in range(int(n_steps)):
            self.update()
        # self.update_frac(n_steps - int(n_steps))  # Fractional step updates.

        # Track BMI runtime
        self.proc_time += time.time() - t_start
        if self.verbose:
            log.info(
                f"BMI Update Until took {time.time() - t_start:.4f} s | Total runtime: {self.proc_time:.4f} s",
            )

    def finalize(self) -> None:
        """(Control function) Finalize model."""
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()
        self._initialized = False

        if self.verbose:
            log.info("BMI model finalized.")

    # =========================================================================#
    # Helper functions for BMI
    # =========================================================================#

    def _do_forward(self, data_dict: dict[str, Any]):
        """Forward model on the pre-formatted dictionary."""
        with torch.no_grad():
            self.prediction = self._model.dpl_model(data_dict)

            # The model output is usually a Dict.
            # We want 'Qs' (Streamflow)
            # Output Shape typically: (Window, Batch, 1)
            # We only want the specific prediction for the center timestep

            # Note: Depending on your model architecture, the prediction
            # might correspond to the last step or the center step.
            # In Sequence-to-One, it's the last step.
            # In Sequence-to-Sequence, we take the center or last.

            # Assuming we want the last value of the High-Freq window:
            target_var = self.model_config['train']['target'][0]  # e.g. 'Qs'

            # Get the result tensor
            pred_tensor = self.prediction[target_var]  # Shape (Time, Batch, 1)

            # We take the middle or last index?
            # In your script: hourly_predict.append(output['Qs'][-current_window_size:,:,0]...)
            # For a single step BMI, we usually just want the one value at T.
            # Let's assume the model outputs a sequence matching the input window.
            # We take the center value (which corresponds to current_time).

            center_idx = pred_tensor.shape[0] // 2
            final_val = pred_tensor[center_idx, :, :].cpu().numpy()

            # Format into dictionary for _format_outputs
            # Internal map expects 'streamflow' usually
            return {'streamflow': final_val}

    @staticmethod
    def _load_model(config: dict):
        """Load a pre-trained model based on the configuration."""
        try:
            return ModelHandler(config, verbose=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}") from e

    def _get_daily_mean_window(
        self,
        var_name,
        current_idx,
        lookback_days,
        window_day_width,
    ):
        """
        Calculates daily means from hourly data for the specific lookback
        window.
        """
        # 1. Calculate hourly indices corresponding to the daily lookback
        # We assume 1 day = 24 hours.
        hours_per_day = 24

        # The daily input window starts at (T - 365 days) and ends at (T - 14 days) roughly
        # This matches the logic: start_daily_index : start_daily_index + lookback - 2*window

        # Calculate the start hour of the daily window
        daily_start_hour = current_idx - (lookback_days * hours_per_day)

        # We need enough hours to cover the 'lookback_days' duration
        # Adjust this length based on exactly how many "daily pixels" the model expects
        # Based on your previous script: n_days = lookback_days - 2 * window_size_day
        n_days_needed = lookback_days - 2 * window_day_width

        if daily_start_hour < 0:
            # Handle Warmup edge case: Pad with first value or zeros
            return np.zeros(
                (n_days_needed, 1),
            )  # Shape (Time, Batch=1) assuming 1 basin

        # Extract the chunk of hourly data covering these days
        total_hours_needed = n_days_needed * hours_per_day
        hourly_chunk = self._dynamic_var[var_name]['value'][
            daily_start_hour : daily_start_hour + total_hours_needed
        ]

        # Reshape to (Days, 24, Batch) and mean over axis 1 (hours)
        # Assuming shape is (Time, Batch, 1) -> (Days, 24, Batch, 1)
        # Note: Your _dynamic_var seems to be (Time, 1) based on init,
        # but _format_inputs suggests it might grow. Assuming (Time, 1).

        if hourly_chunk.shape[0] < total_hours_needed:
            # End of data edge case
            return np.zeros((n_days_needed, 1))

        # Reshape to (Days, 24) (assuming single basin/batch for BMI)
        daily_means = hourly_chunk.reshape(n_days_needed, 24).mean(axis=1)

        # Return shape (Days, 1)
        return daily_means[..., np.newaxis]

    def _format_outputs(self, outputs):
        """Format model outputs as BMI outputs."""
        for name in self._output_vars.keys():
            internal_name = map_to_internal(name)
            if outputs is None:
                log.error("No outputs to format. Check model predictions.")
                output_val = np.zeros(1)
            elif not isinstance(outputs[internal_name], np.ndarray):
                output_val = outputs[internal_name].detach().cpu().numpy()
            else:
                output_val = outputs[internal_name]

            self._output_vars[name]['value'] = np.append(
                self._output_vars[name]['value'],
                output_val,
            )

    def _format_inputs(self):
        """
        Prepares inputs for a SINGLE timestep (self._timestep).
        Performs windowing and normalization immediately.
        """
        # --- Constants from your Model Config ---
        # You might want to move these to __init__
        window_size_hour = 7 * 24  # 168
        lookback_days = 365
        window_size_day = 7
        eps = 1e-6

        # Helper aliases
        hf_config = self.model_config['delta_model']['nn_model']['high_freq_model']
        var_x_list = hf_config['forcings']  # e.g., ['Pr', 'T']
        var_c_list = hf_config['attributes']  # e.g., ['area']

        current_idx = self._timestep

        # --- 1. Hourly Window (High Freq) ---
        # Window: [t - 168, t + 168]
        h_start = current_idx - window_size_hour
        h_end = current_idx + window_size_hour

        # Safety check for warmup
        if h_start < 0:
            log.warning(
                f"Timestep {current_idx} is inside warmup period. Padding with zeros.",
            )
            # Create dummy data if we are at the very start
            # In production, you should start BMI after warmup period

        x_hf_list = []
        x_hf_norm_list = []

        for name in var_x_list:
            internal_name = map_to_internal(name)
            # Retrieve Stats: [min, max, mean, std]
            stats = self.norm_stats[internal_name]
            mu, sigma = stats[2], stats[3]

            # Extract raw
            raw_val = self._dynamic_var[name]['value'][h_start:h_end]
            x_hf_list.append(raw_val)

            # Normalize
            norm_val = (raw_val - mu) / (sigma + eps)
            x_hf_norm_list.append(norm_val)

        # Concatenate: (Time, Feat) -> (Time, 1, Feat) for Batch=1
        x_phy_high = np.concatenate(x_hf_list, axis=-1)[..., np.newaxis].transpose(
            0,
            2,
            1,
        )
        xc_nn_high = np.concatenate(x_hf_norm_list, axis=-1)[..., np.newaxis].transpose(
            0,
            2,
            1,
        )

        # --- 2. Daily Window (Low Freq) ---
        x_lf_list = []
        x_lf_norm_list = []

        # For daily stats, we need to map internal name to daily stats name if they differ
        # Assuming your json has 'mean_daily' or similar, OR we use the same stats.
        # Your previous script used specific daily stats keys.
        # I will assume standard stats for now, adjust keys if needed.

        for name in var_x_list:
            internal_name = map_to_internal(name)
            # You might need specific daily stats here if your JSON has them separately
            stats = self.norm_stats[internal_name]
            mu, sigma = stats[2], stats[3]

            # Get Aggregated Daily Data
            daily_val = self._get_daily_mean_window(
                name,
                current_idx,
                lookback_days,
                window_size_day,
            )

            x_lf_list.append(daily_val)
            x_lf_norm_list.append((daily_val - mu) / (sigma + eps))

        # Concatenate (Time, 1, Feat)
        x_phy_low = np.concatenate(x_lf_list, axis=-1)[..., np.newaxis].transpose(
            0,
            2,
            1,
        )
        xc_nn_low = np.concatenate(x_lf_norm_list, axis=-1)[..., np.newaxis].transpose(
            0,
            2,
            1,
        )

        # --- 3. Static Attributes ---
        c_norm_list = []
        for name in var_c_list:
            internal_name = map_to_internal(name)
            stats = self.norm_stats[internal_name]
            mu, sigma = stats[2], stats[3]

            raw_c = self._static_var[name]['value']
            norm_c = (raw_c - mu) / (sigma + eps)
            c_norm_list.append(norm_c)

        # Shape: (1, Feat) -> (1, Feat)
        c_nn_norm = np.concatenate(c_norm_list, axis=-1)
        if c_nn_norm.ndim == 1:
            c_nn_norm = c_nn_norm[np.newaxis, :]

        # --- 4. Package for Model ---
        # We perform the repeat/broadcast here manually

        # Convert to Tensor
        d_out = {
            'x_phy_high_freq': torch.from_numpy(x_phy_high).float().to(self.device),
            'x_phy_low_freq': torch.from_numpy(x_phy_low).float().to(self.device),
            'xc_nn_norm_high_freq': torch.from_numpy(xc_nn_high)
            .float()
            .to(self.device),
            'xc_nn_norm_low_freq': torch.from_numpy(xc_nn_low).float().to(self.device),
            'c_nn_norm': torch.from_numpy(c_nn_norm).float().to(self.device),
        }

        # Broadcast attributes (Append static to dynamic)
        # Target shape: (Time, Batch, Feat + Attr)

        # High Freq Append
        c_exp_h = (
            d_out['c_nn_norm']
            .unsqueeze(0)
            .repeat(d_out['xc_nn_norm_high_freq'].shape[0], 1, 1)
        )
        d_out['xc_nn_norm_high_freq'] = torch.cat(
            (d_out['xc_nn_norm_high_freq'], c_exp_h),
            dim=-1,
        )

        # Low Freq Append
        c_exp_l = (
            d_out['c_nn_norm']
            .unsqueeze(0)
            .repeat(d_out['xc_nn_norm_low_freq'].shape[0], 1, 1)
        )
        d_out['xc_nn_norm_low_freq'] = torch.cat(
            (d_out['xc_nn_norm_low_freq'], c_exp_l),
            dim=-1,
        )

        # Add Aux Data
        ac_name = map_to_external(
            self.model_config['observations']['upstream_area_name'],
        )
        el_name = map_to_external(self.model_config['observations']['elevation_name'])

        d_out['ac_all'] = (
            torch.from_numpy(self._static_var[ac_name]['value']).float().to(self.device)
        )
        d_out['elev_all'] = (
            torch.from_numpy(self._static_var[el_name]['value']).float().to(self.device)
        )

        return d_out

    def normalize(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Normalize data for neural network."""
        self.load_norm_stats()
        x_nn_norm = self._to_norm(x_nn, _dynamic_input_vars)
        c_nn_norm = self._to_norm(c_nn, _static_input_vars)

        # Remove nans
        x_nn_norm[x_nn_norm != x_nn_norm] = 0
        c_nn_norm[c_nn_norm != c_nn_norm] = 0

        c_nn_norm_repeat = np.repeat(
            np.expand_dims(c_nn_norm, 0),
            x_nn_norm.shape[0],
            axis=0,
        )

        xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm_repeat), axis=2)
        del x_nn_norm, x_nn

        return xc_nn_norm, c_nn_norm

    def _to_norm(
        self,
        data: NDArray[np.float32],
        vars: list[str],
    ) -> NDArray[np.float32]:
        """Standard Gaussian data normalization."""
        log_norm_vars = self.model_config["model"]["phy"]["use_log_norm"]

        data_norm = np.zeros(data.shape)

        for k, var in enumerate(vars):
            stat = self.norm_stats[map_to_internal(var[0])]

            if len(data.shape) == 3:
                if map_to_internal(var[0]) in log_norm_vars:
                    data[:, :, k] = np.log10(np.sqrt(data[:, :, k]) + 0.1)
                data_norm[:, :, k] = (data[:, :, k] - stat[2]) / stat[3]
            elif len(data.shape) == 2:
                if var[0] in log_norm_vars:
                    data[:, k] = np.log10(np.sqrt(data[:, k]) + 0.1)
                data_norm[:, k] = (data[:, k] - stat[2]) / stat[3]
            else:
                raise DataDimensionalityWarning("Data dimension must be 2 or 3.")
        return data_norm

    def load_norm_stats(self) -> None:
        """Load normalization statistics."""
        path = os.path.join(
            self.model_config["model_path"],
            "..",
            "normalization_statistics.json",
        )
        try:
            with open(os.path.abspath(path)) as f:
                self.norm_stats = json.load(f)
        except ValueError as e:
            raise ValueError("Normalization statistics not found.") from e

    def _process_predictions(self, predictions):
        """Process model predictions and store them in output variables."""
        for var_name, prediction in predictions.items():
            if var_name in self._output_vars:
                self._output_vars[var_name]["value"] = prediction.cpu().numpy()
            else:
                log.warning(f"Output variable '{var_name}' not recognized. Skipping.")

    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ) -> list[dict[str, np.ndarray]]:
        """Merge list of batch data dictionaries into a single dictionary."""
        data = {}
        try:
            if target_key:
                return torch.cat([x[target_key] for x in batch_list], dim=1).numpy()

            for key in batch_list[0].keys():
                if len(batch_list[0][key].shape) == 3:
                    dim = 1
                else:
                    dim = 0
                data[key] = (
                    torch.cat([d[key] for d in batch_list], dim=dim).cpu().numpy()
                )
            return data
        except ValueError as e:
            raise ValueError(f"Error concatenating batch data: {e}") from e

    @staticmethod
    def _fill_nan(array_3d):
        # Define the x-axis for interpolation
        x = np.arange(array_3d.shape[1])

        # Iterate over the first and third dimensions to interpolate the second dimension
        for i in range(array_3d.shape[0]):
            for j in range(array_3d.shape[2]):
                # Select the 1D slice for interpolation
                slice_1d = array_3d[i, :, j]

                # Find indices of NaNs and non-NaNs
                nans = np.isnan(slice_1d)
                non_nans = ~nans

                # Only interpolate if there are NaNs and at least two non-NaN values for reference
                if np.any(nans) and np.sum(non_nans) > 1:
                    # Perform linear interpolation using numpy.interp
                    array_3d[i, :, j] = np.interp(
                        x,
                        x[non_nans],
                        slice_1d[non_nans],
                        left=None,
                        right=None,
                    )
        return array_3d

    def array_to_tensor(self) -> None:
        """Converts input values into Torch tensor object to be read by model."""
        raise NotImplementedError("array_to_tensor")

    def tensor_to_array(self) -> None:
        """
        Converts model output Torch tensor into date + gradient arrays to be
        passed out of BMI for backpropagation, loss, optimizer tuning.
        """
        raise NotImplementedError("tensor_to_array")

    def get_tensor_slice(self):
        """Get tensor of input data for a single timestep."""
        # sample_dict = take_sample_test(self.bmi_config, self.dataset_dict)
        # self.input_tensor = torch.Tensor()

        raise NotImplementedError("get_tensor_slice")

    def get_var_type(self, var_name):
        """Data type of variable."""
        return str(self.get_value_ptr(var_name).dtype)

    def get_var_units(self, var_standard_name):
        """Get units of variable.

        Parameters
        ----------
        var_standard_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Variable units.
        """
        # Combine input/output variable dicts: NOTE: should add to init.
        return {**self._dynamic_var, **self._output_vars}[var_standard_name]["units"]

    def get_var_nbytes(self, var_name):
        """Get units of variable."""
        return self.get_value_ptr(var_name).nbytes

    def get_var_itemsize(self, name):
        """Get item size of variable."""
        return self.get_value_ptr(name).itemsize

    def get_var_location(self, name):
        """Location of variable."""
        if name in {**self._dynamic_var, **self._output_vars}.keys():
            return self._var_loc
        else:
            raise KeyError(f"Variable '{name}' not supported.")

    def get_var_grid(self, var_name):
        """Grid id for a variable."""
        if var_name in {**self._dynamic_var, **self._output_vars}.keys():
            return self._var_grid_id
        else:
            raise KeyError(f"Variable '{var_name}' not supported.")

    def get_grid_rank(self, grid_id: int):
        """Rank of grid."""
        if grid_id == 0:
            return 1
        raise RuntimeError(f"Unsupported grid rank: {grid_id!s}. only support 0")

    def get_grid_size(self, grid_id):
        """Size of grid."""
        if grid_id == 0:
            return 1
        raise RuntimeError(f"unsupported grid size: {grid_id!s}. only support 0")

    def get_value_ptr(self, var_standard_name: str) -> np.ndarray:
        """Reference to values."""
        return {**self._dynamic_var, **self._static_var, **self._output_vars}[
            var_standard_name
        ]["value"]

    def get_value(self, var_name: str, dest: NDArray):
        """Return copy of variable values."""
        # TODO: will need to properly account for multiple basins.
        try:
            dest[:] = self.get_value_ptr(var_name)[self._timestep - 1,].flatten()
        except RuntimeError as e:
            raise e
        return dest

    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at indices."""
        dest[:] = self.get_value_ptr(var_name).take(indices)
        return dest

    def set_value(self, var_name, values: np.ndarray):
        """Set variable value."""
        for dict in [self._dynamic_var, self._static_var, self._output_vars]:
            if var_name in dict.keys():
                dict[var_name]["value"] = values
                break

    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices."""
        if not isinstance(src, list):
            src = [src]

        for dict in [self._dynamic_var, self._static_var, self._output_vars]:
            if name in dict.keys():
                for i in inds:
                    dict[name]["value"][i] = src[i]
                break

    def get_component_name(self):
        """Name of the component."""
        return self._name

    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._dynamic_var)

    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_vars)

    def get_input_var_names(self):
        """Get names of input variables."""
        return list(self._dynamic_var.keys())

    def get_output_var_names(self):
        """Get names of output variables."""
        return list(self._output_vars.keys())

    def get_grid_shape(self, grid_id, shape):
        """Number of rows and columns of uniform rectilinear grid."""
        # var_name = self._grids[grid_id][0]
        # shape[:] = self.get_value_ptr(var_name).shape
        # return shape
        raise NotImplementedError("get_grid_shape")

    def get_grid_spacing(self, grid_id, spacing):
        """Spacing of rows and columns of uniform rectilinear grid."""
        # spacing[:] = self._model.spacing
        # return spacing
        raise NotImplementedError("get_grid_spacing")

    def get_grid_origin(self, grid_id, origin):
        """Origin of uniform rectilinear grid."""
        # origin[:] = self._model.origin
        # return origin
        raise NotImplementedError("get_grid_origin")

    def get_grid_type(self, grid_id):
        """Type of grid."""
        if grid_id == 0:
            return "scalar"
        raise RuntimeError(f"unsupported grid type: {grid_id!s}. only support 0")

    def get_start_time(self):
        """Start time of model."""
        return self._start_time

    def get_end_time(self):
        """End time of model."""
        return self._end_time

    def get_current_time(self):
        """Current time of model."""
        return self._timestep * self._att_map["time_step_size"] + self._start_time

    def get_time_step(self):
        """Time step size of model."""
        return self._att_map["time_step_size"]

    def get_time_units(self):
        """Time units of model."""
        return self._att_map["time_units"]

    def get_grid_edge_count(self, grid):
        """Get grid edge count."""
        raise NotImplementedError("get_grid_edge_count")

    def get_grid_edge_nodes(self, grid, edge_nodes):
        """Get grid edge nodes."""
        raise NotImplementedError("get_grid_edge_nodes")

    def get_grid_face_count(self, grid):
        """Get grid face count."""
        raise NotImplementedError("get_grid_face_count")

    def get_grid_face_nodes(self, grid, face_nodes):
        """Get grid face nodes."""
        raise NotImplementedError("get_grid_face_nodes")

    def get_grid_node_count(self, grid):
        """Get grid node count."""
        raise NotImplementedError("get_grid_node_count")

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        """Get grid nodes per face."""
        raise NotImplementedError("get_grid_nodes_per_face")

    def get_grid_face_edges(self, grid, face_edges):
        """Get grid face edges."""
        raise NotImplementedError("get_grid_face_edges")

    def get_grid_x(self, grid, x):
        """Get grid x-coordinates."""
        raise NotImplementedError("get_grid_x")

    def get_grid_y(self, grid, y):
        """Get grid y-coordinates."""
        raise NotImplementedError("get_grid_y")

    def get_grid_z(self, grid, z):
        """Get grid z-coordinates."""
        raise NotImplementedError("get_grid_z")

    def initialize_config(
        self,
        config: Union[dict, dict],
    ) -> dict[str, Any]:
        """Parse and initialize configuration settings.

        Parameters
        ----------
        config
            Configuration settings from Hydra.

        Returns
        -------
        dict
            Formatted configuration settings.
        """
        config["device"], config["dtype"] = self.set_system_spec(config)

        # Convert date ranges to integer values.
        train_time = Dates(config["train"], config["model"]["rho"])
        test_time = Dates(config["test"], config["model"]["rho"])
        sim_time = Dates(config["sim"], config["model"]["rho"])
        all_time = Dates(config["observations"], config["model"]["rho"])

        exp_time_start = min(
            train_time.start_time,
            train_time.end_time,
            test_time.start_time,
            test_time.end_time,
        )
        exp_time_end = max(
            train_time.start_time,
            train_time.end_time,
            test_time.start_time,
            test_time.end_time,
        )

        config["train_time"] = [train_time.start_time, train_time.end_time]
        config["test_time"] = [test_time.start_time, test_time.end_time]
        config["sim_time"] = [sim_time.start_time, sim_time.end_time]
        config["experiment_time"] = [exp_time_start, exp_time_end]
        config["all_time"] = [all_time.start_time, all_time.end_time]

        if config.get("model_dir") is None:
            config["model_dir"] = ""
        config["plot_dir"] = ""
        config["sim_dir"] = ""
        config["log_dir"] = ""

        # Convert string back to data type.
        config["dtype"] = eval(config["dtype"])
        config["model"]["phy"]["nearzero"] = float(config["model"]["phy"]["nearzero"])

        # Raytune
        config["do_tune"] = config.get("do_tune", False)

        return config

    def set_system_spec(self, config: dict) -> tuple[str, str]:
        """Set the device and data type for the model on user's system.

        Parameters
        ----------
        cuda_devices
            List of CUDA devices to use. If None, the first available device is used.

        Returns
        -------
        tuple[str, str]
            The device type and data type for the model.
        """
        if config["device"] == "cpu":
            device = torch.device("cpu")
        elif config["device"] == "mps":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                raise ValueError("MPS is not available on this system.")
        elif config["device"] == "cuda":
            # Set the first device as the active device.
            if (
                torch.cuda.is_available()
                and config["gpu_id"] < torch.cuda.device_count()
            ):
                device = torch.device(f"cuda:{config['gpu_id']}")
                torch.cuda.set_device(device)
            else:
                raise ValueError(
                    f"Selected CUDA device {config['gpu_id']} is not available.",
                )
        else:
            raise ValueError(f"Invalid device: {config['device']}")

        dtype = torch.float32
        return str(device), str(dtype)

    # def scale_output(self) -> None:
    #     """
    #     Scale and return more meaningful output from wrapped model.
    #     """
    #     models = self.config['hydro_models'][0]

    #     # TODO: still have to finish finding and undoing scaling applied before
    #     # model run. (See some checks used in bmi_lstm.py.)

    #     # Strip unnecessary time and variable dims. This gives 1D array of flow
    #     # at each basin.
    #     # TODO: setup properly for multiple models later.
    #     self.streamflow_cms = self.preds[models]['flow_sim'].squeeze()
