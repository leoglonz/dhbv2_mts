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
from dmg.core.utils.dates import Dates

from dmg import MtsModelHandler
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
    ('catchment__area', 'km2'),
    ('basin__length', 'km'),
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
    'P': 'atmosphere_water__liquid_equivalent_precipitation_rate',
    'Temp': 'land_surface_air__temperature',
    'PET': 'land_surface_water__potential_evaporation_volume_flux',
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
    'catchsize': 'catchment__area',
    'lengthkm': 'basin__length',
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

        This is a multitimescale (hourly) version of the δHBV2.0 BMI at
        (dhbv2/bmi.py).

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
            log.debug(f"BMI init took {time.time() - t_start} s")

    @staticmethod
    def _set_vars(
        vars: list[tuple[str, str]],
        var_value: NDArray,
    ) -> dict[str, dict[str, Union[NDArray, str]]]:
        """Set the values of given variables.

        Returns
        -------
        dict
            Dictionary of variable names mapping to their values and units.
            e.g.,
            {
                'var_name_1': {'value': array([...]), 'units': 'unit_1'},
                'var_name_2': {'value': array([...]), 'units': 'unit_2'},
                ...
            }
        """
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
            self._model = self._load_model(self.model_config, verbose=self.verbose).to(
                self.device,
            )
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
            prediction = self._model.dpl_model(data_dict)
            prediction = {
                'streamflow': prediction['Qs'].detach().cpu().numpy(),
            }
        return prediction

    @staticmethod
    def _load_model(config: dict, verbose: bool = False) -> MtsModelHandler:
        """Load a pre-trained model based on the configuration."""
        try:
            model = MtsModelHandler(config, verbose=verbose)
            model.dpl_model.eval()
            model.dpl_model.phy_model.high_freq_model.use_distr_routing = False
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}") from e

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

            if output_val.ndim != 1:
                output_val = output_val.squeeze()
            self._output_vars[name]['value'] = np.append(
                self._output_vars[name]['value'],
                output_val,
            )

    def _format_inputs(self):
        """
        Prepare model inputs for a single timestep (self._timestep).
        Performs windowing and normalization immediately.

        TODO: cleanup
        """
        self._load_norm_stats()

        eps = 1e-6
        mean_dyn_hourly = np.asarray(
            self.norm_stats['mean']['dyn_input'],
            dtype=np.float32,
        )
        std_dyn_hourly = np.asarray(
            self.norm_stats['std']['dyn_input'],
            dtype=np.float32,
        )

        mean_attr = np.asarray(
            self.norm_stats['mean']['static_input'],
            dtype=np.float32,
        )
        std_attr = np.asarray(self.norm_stats['std']['static_input'], dtype=np.float32)
        mean_attr_rout = np.asarray(
            self.norm_stats['mean']['rout_static_input'],
            dtype=np.float32,
        )
        std_attr_rout = np.asarray(
            self.norm_stats['std']['rout_static_input'],
            dtype=np.float32,
        )

        while mean_dyn_hourly.ndim < 3:
            mean_dyn_hourly = mean_dyn_hourly[np.newaxis, ...]
            std_dyn_hourly = std_dyn_hourly[np.newaxis, ...]

        while mean_attr.ndim < 2:
            mean_attr = mean_attr[np.newaxis, ...]
            std_attr = std_attr[np.newaxis, ...]

        while mean_attr_rout.ndim < 2:
            mean_attr_rout = mean_attr_rout[np.newaxis, ...]
            std_attr_rout = std_attr_rout[np.newaxis, ...]

        var_x_list = self.model_config['model']['nn']['hif_model']['forcings']
        var_c_list = self.model_config['model']['nn']['hif_model']['attributes']
        var_c_list2 = self.model_config['model']['nn']['hif_model']['attributes2']

        n_units = self._dynamic_var['land_surface_air__temperature']['value'].shape[0]
        outlet_topo = torch.eye(n_units)

        hourly_forcing = []
        for var in var_x_list:
            hourly_forcing.append(
                np.expand_dims(
                    self._dynamic_var[map_to_external(var)]['value'],
                    axis=-1,
                ),
            )
        hourly_forcing = np.concatenate(hourly_forcing, axis=-1)

        attr = []
        for var in var_c_list:
            attr.append(
                np.expand_dims(
                    self._static_var[map_to_external(var)]['value'],
                    axis=-1,
                ),
            )
        attr = np.concatenate(attr, axis=-1)

        attr_rout = []
        for var in var_c_list2:
            attr_rout.append(
                np.expand_dims(
                    self._static_var[map_to_external(var)]['value'],
                    axis=-1,
                ),
            )
        attr_rout = np.concatenate(attr_rout, axis=-1)

        # Normalization
        hourly_forcing_norm = (hourly_forcing - mean_dyn_hourly) / (
            std_dyn_hourly + eps
        )
        attr_norm = (attr - mean_attr) / (std_attr + eps)
        attr_norm_rout = (attr_rout - mean_attr_rout) / (std_attr_rout + eps)

        # 7 days warmup + 7 days prediction, we only give 1 timestep, use cached states
        x_phy_high_freq = torch.from_numpy(hourly_forcing).permute([1, 0, 2])
        xc_nn_norm_high_freq = torch.from_numpy(hourly_forcing_norm).permute([1, 0, 2])

        c_nn_norm = torch.from_numpy(attr_norm)
        xc_nn_norm_high_freq = torch.cat(
            (
                xc_nn_norm_high_freq,
                c_nn_norm.unsqueeze(0).repeat(xc_nn_norm_high_freq.shape[0], 1, 1),
            ),
            dim=-1,
        )
        rc_nn_norm = torch.from_numpy(attr_norm_rout)

        elev_all = torch.from_numpy(
            self._static_var[map_to_external('meanelevation')]['value'],
        )
        ac_all = torch.from_numpy(self._static_var[map_to_external('uparea')]['value'])
        areas = torch.from_numpy(
            self._static_var[map_to_external('catchsize')]['value'],
        )

        if elev_all.ndim < 2:
            elev_all = elev_all.unsqueeze(0)
        if ac_all.ndim < 2:
            ac_all = ac_all.unsqueeze(0)
        if areas.ndim < 2:
            areas = areas.unsqueeze(0)

        return {
            'xc_nn_norm_high_freq': xc_nn_norm_high_freq,
            'c_nn_norm': c_nn_norm,
            'rc_nn_norm': rc_nn_norm,
            'x_phy_high_freq': x_phy_high_freq,
            'ac_all': ac_all,
            'elev_all': elev_all,
            'areas': areas,
            'outlet_topo': outlet_topo,
        }

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

    def _load_norm_stats(self) -> None:
        """Load normalization statistics."""
        path = os.path.join(
            self.model_config["model_dir"],
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

    def set_value(self, var_name, values: list):
        """Set variable value."""
        if not isinstance(values, list):
            values = [values]
        for dict in [self._dynamic_var, self._static_var, self._output_vars]:
            if var_name in dict.keys():
                dict[var_name]["value"] = np.expand_dims(
                    np.array(values),
                    axis=1,
                )  # [time, space]
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
        config['device'], config['dtype'] = self.set_system_spec(config)

        # Convert date ranges to integer values.
        rho = config['model']['rho']
        # train_time = Dates(config['train'], rho)
        # test_time = Dates(config['test'], rho)
        sim_time = Dates(config['sim'], rho)
        # all_time = Dates(config['observations'], rho)

        # exp_time_start = min(
        #     train_time.start_time,
        #     train_time.end_time,
        #     test_time.start_time,
        #     test_time.end_time,
        # )
        # exp_time_end = max(
        #     train_time.start_time,
        #     train_time.end_time,
        #     test_time.start_time,
        #     test_time.end_time,
        # )

        # config['train_time'] = [train_time.start_time, train_time.end_time]
        # config['test_time'] = [test_time.start_time, test_time.end_time]
        config['sim_time'] = [sim_time.start_time, sim_time.end_time]
        # config['experiment_time'] = [exp_time_start, exp_time_end]
        # config['all_time'] = [all_time.start_time, all_time.end_time]

        if config.get('model_dir') is None:
            config['model_dir'] = ''
        config['plot_dir'] = ''
        config['sim_dir'] = ''
        config['log_dir'] = ''

        # Convert string back to data type.
        config['dtype'] = eval(config['dtype'])

        for name in ['hif_model', 'lof_model']:
            config['model']['phy'][name]['nearzero'] = float(
                config['model']['phy'][name]['nearzero'],
            )

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
