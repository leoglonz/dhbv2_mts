# Validation

Before you trust a δHBV2.0 result, confirm your install reproduces a known one.
This page describes a benchmark case for catchment `cat-2453` and four checks
against it, each answering a different question:

| Leg | Question it answers | Needs Docker |
| --- | --- | --- |
| **1. Standalone** | Does the model itself reproduce the benchmark hydrograph? | no |
| **2. NextGen** | Is my realization feeding the model what it expects? | yes |
| **3. T-Route** | Is the routing stack attached and conserving water? | yes |
| **4. CSV forcing** | Do both forcing providers agree? | yes |

Leg 2 matters more than it looks. The standalone driver and NextGen reach the
same BMI by different routes, and a realization can be wrong in ways the
standalone path cannot see — see [Forcing units](#forcing-units-the-failure-this-catches).

</br>

## Quick Start

```bash
# All four legs, then the comparison (~3.5 min)
./scripts/run_validation.sh

# Or one leg at a time
./scripts/run_validation.sh standalone
./scripts/run_validation.sh ngen
./scripts/run_validation.sh troute
./scripts/run_validation.sh csv
./scripts/run_validation.sh check      # compare existing artifacts only
```

A clean run ends with:

```
standalone vs benchmark:        max|diff|=0.000e+00 ... NSE=1.000000000
ngen vs benchmark:              max|diff|=1.460e-10 ... NSE=1.000000000
ngen vs standalone:             max|diff|=1.460e-10 ... NSE=1.000000000
ngen (CSV forcing) vs benchmark:max|diff|=8.335e-12 ... NSE=1.000000000
ngen (routing run) vs benchmark:max|diff|=1.460e-10 ... NSE=1.000000000
t-route window: 17328 steps, 2009-01-08 01:00:00 -> 2010-12-31 00:00:00
t-route vs unrouted:            n=17327 vol_ratio=1.000037 r=0.995869
standalone runoff ratio: 0.5209
15 passed
```

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NGEN_IMAGE` | `localbuild/ngen:latest` | NextGen image tag to run |
| `MOUNT_LOCAL_SRC` | `0` | Set to `1` to run the image against this checkout's `dhbv2`/`hydrodl2`/`dmg` instead of the copies baked into the image |

</br>

## The Benchmark Case

| | |
| --- | --- |
| Catchment | `cat-2453` (CAMELS subset, HydroFabric v2.2) |
| Forcing | `ngen_resources/data/forcing/camels_subset_2008-01-09 00_00_00_2010-12-30 23_00_00.nc` |
| Window | `2008-01-09 00:00:00` -> `2010-12-30 23:00:00`, hourly (26,088 steps) |
| Output | `land_surface_water__runoff_volume_flux` in `m h-1` |
| Spin-up | Steps 0–8759 are zero; first flow at step 8760 (`2009-01-08 00:00`) |

The committed benchmark is `tests/benchmarks/mts_cat-2453_runoff.npz`. It holds
the runoff series **and** the provenance needed to tell "this run disagrees"
from "this run isn't comparable":

```python
import json, numpy as np
with np.load('tests/benchmarks/mts_cat-2453_runoff.npz') as d:
    print(json.loads(str(d['metadata'])))
```

```json
{
  "catchment": "cat-2453",
  "units": "m h-1",
  "start_time": "2008-01-09 00:00:00",
  "end_time": "2010-12-30 23:00:00",
  "n_steps": 26088,
  "bmi_config_sha256": "131e3582...",
  "model_config_sha256": "13855381...",
  "forcing_sha256": "b98e1da8...",
  "package_versions": {"dhbv2": "...", "hydrodl2": "...", "dmg": "...", "torch": "...", "numpy": "..."}
}
```

If a check fails, compare those hashes against your own files first. A changed
BMI config or a different forcing file explains a mismatch without implicating
your install.

### Tolerance

`max|diff| <= 1e-6 m h-1`, applied to every comparison against the benchmark.

That number is not arbitrary. Repeat runs on one machine are bit-identical
(`0.000e+00`), and NextGen lands at `~1e-10` — the resolution limit of its
9-significant-figure CSV output. So the tolerance leaves about four orders of
magnitude of headroom for a different BLAS, CPU, or PyTorch build, while still
catching real misconfiguration: a single wrong forcing unit moves the
hydrograph by `~1e-3`, a thousand times the tolerance.

</br>

## Leg 1 — Standalone

Runs the BMI directly from Python, with no NextGen involved.

```bash
./scripts/run_validation.sh standalone
```

This writes `output/validation/standalone_cat-2453.npy` (26,088 float64 values
in `m h-1`). Takes about a minute on CPU.

Under the hood it is `scripts/mts_forward_example.py` with
`DHBV2_MTS_OUTPUT` pointed at the validation artifact, so you can also run that
script directly for interactive work — see [4-run_standalone](./4-run_standalone.md).

</br>

## Leg 2 — NextGen

Runs the same BMI inside NextGen, over the same window, using
`realization_validation_cat-2453.json`.

```bash
./scripts/run_validation.sh ngen
```

This writes `output/validation/cat-2453.csv`. It needs a built NextGen image;
see [5-run_ngen](./5-run_ngen.md).

### Forcing units: the failure this catches

Every BMI input variable declares its units through `get_var_units()`. NextGen
reads the `units` attribute on each forcing variable, compares it to what the
BMI declares, and converts with udunits. **Standalone there is no such layer**,
so a driver script has to perform exactly the same conversions by hand:

| Forcing variable | File units | BMI declares | Conversion |
| --- | --- | --- | --- |
| `precip_rate` | `mm s^-1` | `mm h-1` | × 3600 |
| `TMP_2maboveground` | `K` | `degC` | − 273.15 |
| `PRES_surface` | `Pa` | `Pa` | none |
| `SPFH_2maboveground` | `kg/kg` | `g g-1` | none |
| `DLWRF_surface`, `DSWRF_surface` | `W/m^2` | `W m-2` | none |
| `UGRD_10maboveground`, `VGRD_10maboveground` | `m/s` | `m s-1` | none |

Miss one and the two paths silently disagree. Feeding temperature in Kelvin,
for example, puts the catchment permanently above freezing: the snow module
never accumulates, PET collapses, and runoff rises to **1.03×** precipitation —
more water leaving the catchment than entering it.

Running only the standalone leg cannot detect this, because the standalone
benchmark would have been generated with the same mistake. Leg 2 compares two
independently-converted paths, so it can.

### Realization notes

Two things about `variables_names_map` are easy to get wrong:

- **Map to the bare name, never `precip_rate[mm h-1]`.** The bracket declares
  units; it is not part of the variable name. ngen strips it while reading a
  CSV header, so a map entry carrying it matches nothing — and the NetCDF
  provider, which has no such column at all, aborts outright with
  `Got request for variable precip_rate[mm h-1] but it was not found in the
  cache`. Use `precip_rate` with both providers.
- **The archived CSVs under `ngen_resources/data/forcing/depr/` cannot drive
  the MTS model.** They carry only precipitation, temperature, and Hargreaves
  PET — 3 of the 8 inputs it requires. The CSV example this suite exercises is
  `ngen_resources/data/forcing/cat-2453_2008-01-09 00_00_00_2010-12-30 23_00_00.csv`,
  generated from the NetCDF by `scripts/make_csv_forcing.py` with all 8.

</br>

## Leg 3 — T-Route

Runs NextGen with Muskingum-Cunge routing via
`realization_troute_cat-2453.json`, over the same full window as leg 2.

```bash
./scripts/run_validation.sh troute
```

This writes `output/validation_troute/cat-2453.csv` and a single
`output/validation_troute/stream_output/troute_output_200901080000.nc`.

### What gets routed

T-Route routes the **full post-spin-up period**: 17,328 hours from
`2009-01-08 01:00` to `2010-12-31 00:00`. It does not route the 8,760 spin-up
hours, which are zero — routing them would cost time and dilute the check.

Three values in `routing_config.yaml` are pinned to the realization's window
and must move with it:

```yaml
compute_parameters:
  restart_parameters:
    start_datetime: 2009-01-08 00:00:00   # realization start + spin-up length
  forcing_parameters:
    qts_subdivisions: 12                  # ngen timestep / t-route dt = 3600/300
    dt: 300                               # t-route internal timestep [s]
    nts: 207936.0                         # routed hours x qts_subdivisions
                                          #   = (26088 - 8760) x 12
    max_loop_size: 26088.0                # ngen timesteps in the run
output_parameters:
  stream_output:
    stream_output_time: -1                # one file for the whole run
```

`stream_output_time` deserves a note: it is **hours of routed flow per output
file**, and the t-route default of `1` would emit 17,328 separate NetCDFs here.
`-1` writes them all to one 964 KB file instead.

Nothing warns you when these drift. A too-small `nts` silently routes a
shorter period than the model simulated, which is why the suite asserts the
routed series is exactly 17,328 steps rather than merely non-empty.

### What is checked

Volume conservation, not a stored benchmark. Over a single 2.1 km reach,
Muskingum-Cunge should neither create nor destroy water and should barely
attenuate the hydrograph, so the suite asserts routed/unrouted volume within
1% and correlation above 0.95. Measured over the full period: `1.000037` and
`0.9959`.

The suite also checks that the routing run's own unrouted runoff matches the
benchmark, so a routing discrepancy can never be blamed on that run having
simulated something different.

</br>

## Leg 4 — CSV Forcing

Runs NextGen through the `CsvPerFeature` provider instead of `NetCDF`, using
`realization_cat-2453.json`.

```bash
./scripts/run_validation.sh csv
```

This generates `ngen_resources/data/forcing/cat-2453_2008-01-09 00_00_00_2010-12-30 23_00_00.csv`
if absent (via `scripts/make_csv_forcing.py`) and writes
`output/validation_csv/cat-2453.csv`.
It runs the shorter 9,312-step window that realization ships with, and is
compared against the benchmark's first 9,312 steps.

Both providers must hand the model identical numbers. They currently agree to
`8.3e-12` — the CSV is written at `%.9g`, the shortest decimal form that
round-trips the NetCDF's float32 values exactly. The leg exists because the
CSV example is generated data that can silently fall out of step with the
NetCDF it came from, exactly the failure this page is here to prevent. See
[Forcing providers](./5-run_ngen.md#forcing-providers).

For routing background and the DDR alternative, see [7-routing](./7-routing.md).

</br>

## Water Balance

One check needs no benchmark at all:

```
runoff / precipitation over the post-spin-up period must be between 0 and 1
```

Benchmark value: **0.521**. With temperature fed in Kelvin it was **1.026** —
physically impossible, and caught without any reference run. This is the check
worth carrying over to a new catchment, a retrained model, or a forcing dataset
for which no benchmark exists yet.

</br>

## Interpreting a Failure

| Symptom | Likely cause |
| --- | --- |
| All legs fail together | Model weights, BMI config, or forcing differ from the benchmark. Compare the recorded hashes. |
| Standalone passes, NextGen fails | The realization. Check `variables_names_map` and forcing units. |
| Both pass, t-route fails | Routing config: `start_datetime` inside spin-up, or `nts`/`max_loop_size`/`qts_subdivisions` inconsistent with the realization window. |
| t-route routed fewer hours than expected | `nts` is too small for the window. It is routed hours × `qts_subdivisions`, not hours. |
| Only the CSV leg fails | The CSV has drifted from the NetCDF. Regenerate with `scripts/make_csv_forcing.py`, and check its headers still carry `[units]` and that `variables_names_map` uses the bare name. |
| Runoff ratio ≥ 1.0 | A forcing-unit error, most likely temperature. Not a model-skill problem. |
| `max|diff|` between `1e-6` and `1e-4` | Possibly a genuine numerical-environment difference (BLAS, CPU, PyTorch build). Check NSE and volume ratio: if both are ~1.0 the hydrograph is intact. Report it as an [issue](https://github.com/mhpi/dhbv2/issues) with your `package_versions`. |
| A leg is skipped | Its artifact was not produced. The skip message names the command that produces it. |

A leg is skipped **only** when its artifact is absent. An artifact that exists
but disagrees always fails — it is never quietly reported as a pass.

</br>

## Regenerating the Benchmark

Maintainers only, and only when a change is *intended* to move the numbers:

```bash
python scripts/mts_forward_example.py
python scripts/make_mts_benchmark.py
```

A benchmark refreshed to silence a failing validation run defeats the purpose
of having one. Regenerate deliberately, and say why in the commit message.

</br>

## Validating the NextGen Build Itself

The checks above validate δHBV2.0 within NextGen. To validate the NextGen and
T-Route build independently of this model — `ngen --info`, the stock unit
tests, `test_routing_pybind`, and the LowerColorado_TX T-Route examples — see
the Validation section of [5-run_ngen](./5-run_ngen.md).

</br>

## Daily Model

> The daily (non-MTS) δHBV2.0 model has no benchmark case yet.
