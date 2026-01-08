"""
Fetch catchment attributes from CAMELS netCDF and update existing BMI config
YAML files accordingly.

TODO: This script should support adding catchments attributes for any catchment
in the CONUS HF.
"""

import xarray as xr
from ruamel.yaml import YAML
from pathlib import Path

# Setup pathing
pkg_root = Path(__file__).parent.parent.parent


### -------- Settings -------- ###
CAT_ID = 2455
ATTRS_PATH = '/projects/mhpi/yxs275/hourly_model/mtsHBV/data/CAMELS_HFs_attr_new.nc'
TARGET_PATH = f'{pkg_root}/ngen_resources/data/dhbv_2_mts/config/bmi_cat-{CAT_ID}.yaml'
### -------------------------- ###


if __name__ == '__main__':
    attrs_ds = xr.open_dataset(ATTRS_PATH).sel(gauge=CAT_ID)

    yml = YAML()
    yml.preserve_quotes = True

    # Load data
    with open(TARGET_PATH) as f:
        config_data = yml.load(f)

    # Update existing keys
    for var_name in attrs_ds.data_vars:
        key = str(var_name)
        if key in config_data:
            val = attrs_ds[var_name].item()
            config_data[key] = val

    # Dump back (comments and structure are preserved automatically)
    with open(TARGET_PATH, 'w') as f:
        yml.dump(config_data, f)
