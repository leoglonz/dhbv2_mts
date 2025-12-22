# Running Standalone

You can run the models in "standalone" mode using Python scripts. This is useful for debugging, quick testing, or running inference without compiling the full NextGen engine.

</br>

## Scripts

The `scripts/` directory contains helpers for both Daily and MTS models.

### (1) Running the Daily Model

The `forward.py` script mimics the BMI lifecycle for the daily model.

```bash
python scripts/forward.py
```

- Config: scripts/forward.py defaults to cat-88306. You may need to edit the BASIN_ID variable or path strings to match your data.

- Input: Expects NetCDF forcing by default, but can be adapted for CSV.

### (2) Running the MTS (Hourly) Model

The `forward_mts_cat-67.py` script runs the hourly multi-timescale model for a specific test catchment (cat-67).

- Config: Uses `ngen_resources/data/dhbv2_mts/config/bmi_cat-67.yaml`.

- Input: Uses CSV forcing files.

- Output: Prints streamflow (m3/s) for each hour to the console.

</br>

## Configuration Files

Standalone runs rely on **BMI config files** (YAML). These define the physics options and point to the static attributes.

Example `bmi_cat-67.yaml`:

```yaml
catchment_id: 'cat-67'
time_step: 1 hour
model_config: models/hfv2.2_15yr/config.yaml
states_name: initial_states_2009.pt
use_daily_states: false
# ... list of static attributes (aridity, meanP, etc.) ...
```
