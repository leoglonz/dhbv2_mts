# Validation

Before trusting a dhbv2 result, confirm your install reproduces included benchmarks. This step validates against catchment `cat-2453`.

Run the provided data/realization/config example through whatever NextGen build you have, then point the validation suite at the output;

* [BMI config](../ngen_resources/data/dhbv_2_mts/config/bmi_cat-2453.yaml)
* [NextGen realization](../ngen_resources/data/dhbv_2_mts/realizations/realization_troute_cat-2453.json)

| Leg | Question it answers | Needs |
| --- | --- | --- |
| **1. Standalone** | Does the model itself reproduce the benchmark hydrograph? | Python only |
| **2. NextGen** | Is my realization feeding the model what it expects? | your own ngen |
| **3. T-Route** | Is the routing stack attached and conserving water? | your own ngen + t-route |

> NOTE: The daily (non-MTS) δHBV2.0 model has no benchmark case yet.

</br>

## Quick Start

```bash
# 1. Standalone leg
python scripts/mts_forward_example.py

# 2. The provided example The t-route realization writes
#    cat-2453.csv *and* troute_output_*.nc, so one run covers legs 2 and 3;
#    use realization_nc_cat-2453.json instead to skip routing.
ngen <gpkg> cat-2453 <gpkg> nex-2454 \
    ./data/dhbv_2_mts/realizations/realization_troute_cat-2453.json

# 3. Validate whatever you produced
pytest tests/test_validation.py --run-dir=/path/to/that/output
```

`--run-dir` is searched recursively, so any output layout works as long as it contains `cat-2453.csv` and, for routing, `troute_output_*.nc`. A clean run ends with:

```
standalone vs benchmark:  max|diff|=0.000e+00 ... NSE=1.000000000
ngen vs benchmark:        max|diff|=1.460e-10 ... NSE=1.000000000
ngen vs standalone:       max|diff|=1.460e-10 ... NSE=1.000000000
t-route window: 17328 steps, 2009-01-08 01:00:00 -> 2010-12-31 00:00:00
t-route vs unrouted:      n=17327 vol_ratio=1.000037 r=0.995869
standalone runoff ratio: 0.5209
13 passed
```

</br>

## The Benchmark Case

| | |
| --- | --- |
| Catchment | `cat-2453` (CAMELS subset, HydroFabric v2.2) |
| Forcing | `ngen_resources/data/forcing/camels_subset_2008-01-09 ... .nc` |
| Window | `2008-01-09 00:00` -> `2010-12-30 23:00`, hourly (26,088 steps) |
| Output | `land_surface_water__runoff_volume_flux` in `m h-1` |
| Spin-up | Steps 0-8759 are zero; first flow at step 8760 (`2009-01-08 00:00`) |
| Tolerance | `max\|diff\| <= 1e-6 m h-1` |

`tests/benchmarks/mts_cat-2453_runoff.npz` holds the runoff series plus config and forcing hashes, and the package versions that produced it. **If a check fails, compare those against your own files first**; a changed BMI config or forcing explains a mismatch without implicating your install:

```python
import json, numpy as np

with np.load('tests/benchmarks/mts_cat-2453_runoff.npz') as d:
    print(json.loads(str(d['metadata'])))
```

The tolerance leaves four orders of headroom for a different BLAS, CPU or PyTorch build -- NextGen lands at `~1e-10`, its CSV resolution limit -- while still catching misconfiguration.

</br>

## Leg 1 -- Standalone

Runs the BMI standalone in Python. Writes `output/dhbv_2_mts_cat-2453_runoff.npy`, where the suite looks by default; set `DHBV2_MTS_OUTPUT` to write elsewhere — see [4-run_standalone](./4-run_standalone.md).

</br>

## Leg 2 -- NextGen

Runs the same BMI inside NextGen over the same window, using `realization_nc_cat-2453.json`. Point `--run-dir` at wherever it wrote `cat-2453.csv`. Any NextGen build will do; see [5-run_ngen](./5-run_ngen.md).

### Forcing units: the failure this catches

NextGen reads each forcing variable's declared units and converts to whatever the BMI asks for via `get_var_units()`. **Standalone there is no such layer**, so a driver script must do the same conversions by hand:

| Forcing variable | File units | BMI declares | Conversion |
| --- | --- | --- | --- |
| `precip_rate` | `mm s^-1` | `mm h-1` | x 3600 |
| `TMP_2maboveground` | `K` | `degC` | - 273.15 |
| `PRES_surface` | `Pa` | `Pa` | none |
| `SPFH_2maboveground` | `kg/kg` | `g g-1` | none |
| `DLWRF_surface`, `DSWRF_surface` | `W/m^2` | `W m-2` | none |
| `UGRD_10maboveground`, `VGRD_10maboveground` | `m/s` | `m s-1` | none |

</br>

## Leg 3 -- T-Route

Adds Muskingum-Cunge routing via `realization_troute_cat-2453.json`. It writes `cat-2453.csv` alongside `stream_output/troute_output_*.nc`.

</br>

## Interpreting a Failure

| Symptom | Likely cause |
| --- | --- |
| All legs fail together | Weights, BMI config, or forcing differ from the benchmark. Compare the recorded hashes. |
| Standalone passes, NextGen fails | The realization. Check `variables_names_map` and forcing units. |
| Both pass, t-route fails | Routing config: `start_datetime` inside warmup, or `nts`/`max_loop_size`/`qts_subdivisions` inconsistent with the window. |
| t-route routed too few hours | `nts` is routed hours x `qts_subdivisions`, not hours. |
| Runoff ratio ≥ 1.0 | A forcing-unit error, most likely temperature. Not a model-skill problem. |
| `max\|diff\|` in `1e-6`–`1e-4` | Possibly a genuine BLAS/CPU/PyTorch difference. If NSE and volume ratio are ~1.0 the hydrograph is intact -- report it as an [issue](https://github.com/mhpi/dhbv2/issues) with your `package_versions`. |
| A leg is skipped | Its artifact was not found. Check `--run-dir` contains `cat-2453.csv` / `troute_output_*.nc`; the skip message says what was missing and where it looked. |

</br>

## Regenerating the Benchmark

> Maintainers only, for building new benchmarks to accompany latest model releases.

```bash
python scripts/mts_forward_example.py
python scripts/utils/make_mts_benchmark.py
```

</br>

To validate the NextGen and T-Route build itself, independently of this model,
see the Validation section of [5-run_ngen](./5-run_ngen.md).
