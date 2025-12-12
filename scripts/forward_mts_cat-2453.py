"""
Forward BMI on single catchment with pseudo-NextGen operating behavior.

We use catchment 67 which is included with NOAA-OWP/ngen.

@leoglonz
"""

import os
from pathlib import Path
import random

import xarray as xr
import numpy as np
from dhbv2.mts_bmi import MtsDeltaModelBmi as Bmi

### Configuration Settings (single-catchment) ###
BMI_CONFIG_PATH = "ngen_resources/data/dhbv2_mts/config/bmi_cat-2453.yaml"
FORCING_PATH = (
    "ngen_resources/data/forcing/camels_2010-01-01_00_00_00_2011-12-30_23_00_00.nc"
)
### ----------------------------------------- ###

# Setup pathing
pkg_root = Path(__file__).parent.parent
bmi_config_path = os.path.join(pkg_root, Path(BMI_CONFIG_PATH))
forcing_path = os.path.join(pkg_root, Path(FORCING_PATH))


# Create dHBV 2.0 BMI instance
print(">>> Creating DeltaModelBmi instance")
model = Bmi(verbose=True)


### BMI initialization ###
print(">>> Initializing the BMI")
model.initialize(config_path=bmi_config_path)


print("[Preparing data]")
forcing_xr = xr.open_dataset(forcing_path)
t_steps = len(forcing_xr["time"])

print(
    "[Looping through timesteps | Setting forcing/attribute values & forwarding model]",
)

for t in range(t_steps):
    # print(f"\n--- Timestep {t + 1}/{t_steps} ---")

    # Set forcing values
    model.set_value(
        "atmosphere_water__liquid_equivalent_precipitation_rate",
        forcing_xr["precip_rate"][t] + random.randrange(0, 4),
    )
    model.set_value(
        "land_surface_air__temperature",
        forcing_xr["TMP_2maboveground"][t],
    )
    model.set_value(
        "land_surface_water__potential_evaporation_volume_flux",
        forcing_xr["PET_hargreaves"][t],
    )
    print(
        ">>> Forcings set"
        f" | Precip: {forcing_xr['precip_rate'][t]:.4f} m/s,"
        f" Temp: {forcing_xr['TMP_2maboveground'][t]:.2f} K,"
        f" PET: {forcing_xr['PET_hargreaves'][t]:.4f} m3/s",
    )

    # if f_dict['precip_rate'][t] > 0.01:
    #     print(">>> High precip event!")

    ### BMI update ###
    # print(">>> Doing BMI model update")
    model.update()

    dest_array = np.zeros(1)
    model.get_value("land_surface_water__runoff_volume_flux", dest_array)
    runoff = dest_array[0]

    print(
        f"Result: Streamflow at time {model.get_current_time()} ({model.get_time_units()}) is {runoff:.4f} m3/s",
    )

    if t > 100:
        print(">>> Stopping the loop")
        break

### BMI finalization ###
print(">>> Finalizing the BMI")
model.finalize()
