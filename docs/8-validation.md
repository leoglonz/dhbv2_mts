# Validation

Before you trust a δHBV2.0 result, confirm your install reproduces a known one.
This page describes a benchmark case for catchment `cat-2453` and four checks
against it, each answering a different question:

| Leg | Question it answers | Needs |
| --- | --- | --- |
| **1. Standalone** | Does the model itself reproduce the benchmark hydrograph? | Python only |
| **2. NextGen** | Is my realization feeding the model what it expects? | your own ngen |
| **3. T-Route** | Is the routing stack attached and conserving water? | your own ngen + t-route |

Leg 2 matters more than it looks. The standalone driver and NextGen reach the
same BMI by different routes, and a realization can be wrong in ways the
standalone path cannot see — see [Forcing units](#forcing-units-the-failure-this-catches).

</br>

## Quick Start

Nothing here runs NextGen for you, and nothing needs Docker. Run the shipped
`cat-2453` example through whatever NextGen build you already have, then point
the suite at its output.

```bash
# 1. The standalone leg needs no ngen at all (~1 min on CPU)
python scripts/mts_forward_example.py

# 2. Run the shipped example through your own ngen. The t-route realization
#    writes cat-2453.csv *and* troute_output_*.nc, so one run covers legs 2
#    and 3; use realization_nc_cat-2453.json instead to skip routing.
ngen <gpkg> cat-2453 <gpkg> nex-2454 \
    ./data/dhbv_2_mts/realizations/realization_troute_cat-2453.json

# 3. Validate whatever you produced
pytest tests/test_validation.py --run-dir=/path/to/that/output
```

`--run-dir` is searched recursively, so any output layout works as long as it
contains `cat-2453.csv` and, for routing, `troute_output_*.nc`.

A clean run ends with:

```
standalone vs benchmark:        max|diff|=0.000e+00 ... NSE=1.000000000
ngen vs benchmark:              max|diff|=1.460e-10 ... NSE=1.000000000
ngen vs standalone:             max|diff|=1.460e-10 ... NSE=1.000000000
t-route window: 17328 steps, 2009-01-08 01:00:00 -> 2010-12-31 00:00:00
t-route vs unrouted:            n=17327 vol_ratio=1.000037 r=0.995869
standalone runoff ratio: 0.5209
13 passed
```

Run it with no `--run-dir` at all and you still get 8 checks — benchmark
integrity, the standalone leg, and the water balance — with the ngen and
t-route legs skipped rather than failed.

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
python scripts/mts_forward_example.py
```

This writes `output/dhbv_2_mts_cat-2453_runoff.npy` (26,088 float64 values in
`m h-1`), which is where the suite looks by default. Takes about a minute on
CPU. Set `DHBV2_MTS_OUTPUT` to write elsewhere — see
[4-run_standalone](./4-run_standalone.md).

</br>

## Leg 2 — NextGen

Runs the same BMI inside NextGen, over the same window, using
`realization_nc_cat-2453.json`.

```bash
ngen <gpkg> cat-2453 <gpkg> nex-2454 \
    ./data/dhbv_2_mts/realizations/realization_nc_cat-2453.json
```

Point `--run-dir` at wherever that wrote `cat-2453.csv`. Any NextGen build will
do; see [5-run_ngen](./5-run_ngen.md) if you still need one.

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
  PET — 3 of the 8 inputs it requires. The usable CSV example is
  `ngen_resources/data/forcing/cat-2453_2008-01-09 00_00_00_2010-12-30 23_00_00.csv`,
  which ships with all 8 and covers the full window.

</br>

## Leg 3 — T-Route

Runs NextGen with Muskingum-Cunge routing via
`realization_troute_cat-2453.json`, over the same full window as leg 2.

```bash
ngen <gpkg> cat-2453 <gpkg> nex-2454 \
    ./data/dhbv_2_mts/realizations/realization_troute_cat-2453.json
```

This writes `cat-2453.csv` alongside a single
`stream_output/troute_output_200901080000.nc`. Because it produces both, one
`--run-dir` covers legs 2 and 3 together.

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

Because a single run supplies both legs, the runoff compared against the
benchmark in leg 2 *is* the series that fed t-route — so a routing discrepancy
can never be blamed on that run having simulated something different.

</br>

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
| Runoff ratio ≥ 1.0 | A forcing-unit error, most likely temperature. Not a model-skill problem. |
| `max|diff|` between `1e-6` and `1e-4` | Possibly a genuine numerical-environment difference (BLAS, CPU, PyTorch build). Check NSE and volume ratio: if both are ~1.0 the hydrograph is intact. Report it as an [issue](https://github.com/mhpi/dhbv2/issues) with your `package_versions`. |
| A leg is skipped | Its artifact was not found. Check `--run-dir` actually contains `cat-2453.csv` / `troute_output_*.nc`; the skip message says what was missing and where it looked. |

A leg is skipped **only** when its artifact is absent. An artifact that exists
but disagrees always fails — it is never quietly reported as a pass.

</br>

## Regenerating the Benchmark

Maintainers only, and only when a change is *intended* to move the numbers:

```bash
python scripts/mts_forward_example.py
python scripts/utils/make_mts_benchmark.py
```

It promotes `output/dhbv_2_mts_cat-2453_runoff.npy` by default, refuses any run
whose length is not the benchmark case's 26,088 steps, and records the hashes
and package versions above alongside the series.

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
