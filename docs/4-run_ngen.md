# Running with NextGen

To run δHBV 2.0 within the NextGen framework, you must configure a **Realization** file (JSON) that points to the correct Python class and BMI configuration.

</br>

## Python Types

* **Daily Model:** `dhbv2.bmi.DeltaModelBmi`
* **Hourly (MTS) Model:** `dhbv2.mts_bmi.MtsDeltaModelBmi`

</br>

## Configuration Examples

### (1) Daily Simulation (`realization_cat-67.json`)

```json
{
  "global": { "time": { "output_interval": 86400 } },
  "catchments": {
    "cat-67": {
      "formulations": [
        {
          "name": "bmi_python",
          "params": {
            "python_type": "dhbv2.bmi.DeltaModelBmi",
            "model_type_name": "Differentiable Model",
            "init_config": "/path/to/ngen_resources/data/dhbv2/config/bmi_cat-67.yaml",
            "uses_forcing_file": true,
            "main_output_variable": "land_surface_water__runoff_volume_flux"
          }
        }
      ]
    }
  }
}
```

### (2) Hourly MTS Simulation (`realization_cat-67.json`)

> Note: The time step and `output_interval` must be set to `3600` (seconds).

```json
{
  "global": { "time": { "output_interval": 3600 } },
  "catchments": {
    "cat-67": {
      "formulations": [
        {
          "name": "bmi_python",
          "params": {
            "python_type": "dhbv2.mts_bmi.MtsDeltaModelBmi",
            "model_type_name": "未HBV2.0 MTS",
            "init_config": "/path/to/ngen_resources/data/dhbv2_mts/config/bmi_cat-67.yaml",
            "uses_forcing_file": true,
            "main_output_variable": "land_surface_water__runoff_volume_flux"
          }
        }
      ]
    }
  }
}
```

</br>

## Execution

Run the NextGen engine pointing to the realization file:

```bash
./cmake_build/ngen \
    /path/to/catchment_data.geojson "cat-67" \
    /path/to/nexus_data.geojson "nex-67" \
    /path/to/realization_cat-67.json
```
