"""
Forward BMI on single catchment with pseudo-NextGen operating behavior.

We use catchment 67 which is included with NOAA-OWP/ngen.

@leoglonz
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
from dhbv2.bmi_mts import DeltaModelBmi as Bmi

### Configuration Settings (single-catchment) ###
BMI_CONFIG_PATH = "ngen_resources/data/dhbv2_mts/config/bmi_cat-67.yaml"
FORCING_PATH = (
    "ngen_resources/data/forcing/cat-67_2015-12-01 00_00_00_2015-12-30 23_00_00.csv"
)
### ----------------------------------------- ###

# Setup pathing
pkg_root = Path(__file__).parent.parent
bmi_config_path = os.path.join(pkg_root, Path(BMI_CONFIG_PATH))
forcing_path = os.path.join(pkg_root, Path(FORCING_PATH))


# Create dHBV 2.0 BMI instance
print(">>> Creating DeltaModelBmi instance")
model = Bmi()


### BMI initialization ###
print(">>> Initializing the BMI")
model.initialize(config_path=bmi_config_path)


print("[Preparing data]")
f_dict = pd.read_csv(forcing_path)
t_steps = len(f_dict["time"])

print(
    "[Looping through timesteps | Setting forcing/attribute values & forwarding model]",
)

for t in range(t_steps):
    print(f"\n--- Timestep {t + 1}/{t_steps} ---")

    # Set forcing values
    model.set_value(
        "atmosphere_water__liquid_equivalent_precipitation_rate",
        f_dict["precip_rate"][t],
    )
    model.set_value(
        "land_surface_air__temperature",
        f_dict["TMP_2maboveground"][t],
    )
    model.set_value(
        "land_surface_water__potential_evaporation_volume_flux",
        f_dict["PET_hargreaves"][t],
    )

    ### BMI update ###
    print(">>> Doing BMI model update")
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
