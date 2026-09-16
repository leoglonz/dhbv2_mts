# Running Standalone

dhbv2 models can be run "standalone" with provided Python scripts. This may be useful for debugging, intuiting the structure, or running inference without compiling the full NextGen engine.

</br>

## Scripts

The `./scripts/` directory contains BMI forward examples for both daily and MTS (hourly) models.

### (1) Running the Daily Model

> *Coming soon.*

### (2) Running the MTS (Hourly) Model

The `mts_forward_example.py` script runs MTS for a single catchment (e.g., cat-2453; 2454, 2455 also available).

```bash
python scripts/mts_forward_example.py
```

- **Config**: Uses `./ngen_resources/data/dhbv_2_mts/config/bmi_cat-2453.yaml`.

- **Input**: Uses NetCDF forcing file `./ngen_resources/data/forcing/camels_subset_2008-01-09 00_00_00_2010-12-30 23_00_00.nc`.

- **Output**: Hourly runoff (m h-1) timeseries, saved to `./output/dhbv_2_mts_cat-2453_runoff.npy` (`DHBV2_MTS_OUTPUT` to write elsewhere).

The forcing file's units are not the units the BMI declares, and standalone there is no NextGen unit-conversion layer to bridge them — the script converts by hand at the top of its forcing block. If you adapt it to another dataset, keep those conversions in step with `get_var_units()`; see [8-validation](./8-validation.md#forcing-units-the-failure-this-catches).

To check this run against the committed benchmark, see [8-validation](./8-validation.md).

</br>

## Configuration Files

Standalone runs rely on yaml **BMI config files**. These define the physics options and provide static catchment attributes.

Example `bmi_cat-2453.yaml`:

```yaml
# ... list of static attributes (aridity, meanP, etc.) ...

catchment_id: cat-2453
model_dir: ./data/dhbv_2_mts/model/dhbv_2_mts/
dtype: float32
verbose: false

pet_method: penman_monteith  # or 'hargreaves'
latitude: 45.38943493686819

warmup:
  cycle_days: 14  # how long the hourly model runs before it
                  # takes fresh states from the daily model
  daily_mode: periodic  # or 'cold' to skip daily spin-up
  daily_warmup_days: 351
  hourly_mode: periodic  # or 'cold' to skip hourly spin-up
  hourly_warmup_hours: 336
```

The model emits zero runoff until it has `daily_warmup_days` of daily history and `hourly_warmup_hours` of hourly history. For example, for the above values, the first non-zero runoff arrives at step 8760.
