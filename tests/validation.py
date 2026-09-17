"""Supporting fixtures for δHBV2.0 MTS validation suite.

The suite compares three runoff series that should all describe the same
hydrograph for catchment ``cat-2453``:

1. **standalone** -- the BMI driven directly by ``scripts/mts_forward_example.py``
2. **ngen** -- the same BMI driven by NextGen through its Python BMI adapter
3. **t-route** -- NextGen's runoff routed through Muskingum-Cunge

Leg 1 answers "is the model reproducing the benchmark", leg 2 answers "is my
NextGen wiring feeding the model what it expects", and leg 3 answers "is the
routing stack attached and conserving water".

@leoglonz
"""

import glob
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------- #
#  Fixed properties of the shipped validation case
# ---------------------------------------------------------------------------- #

CATCHMENT = 'cat-2453'
CATCHMENT_FEATURE_ID = 2453
NEXUS = 'nex-2454'

#: Catchment drainage area [km2], from the shipped BMI config (`catchsize`).
CATCHMENT_AREA_KM2 = 7.7805003825041545

#: Full forcing window covered by the shipped NetCDF file.
SIM_START = '2008-01-09 00:00:00'
SIM_END = '2010-12-30 23:00:00'
N_STEPS_FULL = 26088

#: First step at which the model has cleared spin-up and emits flow, with the
#: shipped `bmi_cat-2453.yaml` (351 daily + 336 hourly warmup, 14-day cycle).
FIRST_FLOW_STEP = 8760

#: The routed leg runs the same window as the others, but t-route only routes
#: the post-spin-up remainder -- `start_datetime` in `routing_config.yaml` is
#: set to where flow begins, and `nts` to the hours that follow.
ROUTED_START = '2009-01-08 00:00:00'
N_STEPS_ROUTED = N_STEPS_FULL - FIRST_FLOW_STEP

#: `realization_cat-2453.json` (the CsvPerFeature example) runs a shorter
#: window than the validation realizations -- it is a usage example first.
N_STEPS_CSV_EXAMPLE = 9312

RUNOFF_VAR = 'land_surface_water__runoff_volume_flux'
RUNOFF_UNITS = 'm h-1'

DATA_ROOT = PKG_ROOT / 'ngen_resources' / 'data'
FORCING_PATH = (
    DATA_ROOT / 'forcing' / 'camels_subset_2008-01-09 00_00_00_2010-12-30 23_00_00.nc'
)
MODEL_CONFIG_PATH = DATA_ROOT / 'dhbv_2_mts' / 'model' / 'dhbv_2_mts' / 'config.yaml'
BENCHMARK_PATH = PKG_ROOT / 'tests' / 'benchmarks' / 'mts_cat-2453_runoff.npz'

#: Where `scripts/run_validation.sh` deposits the artifacts under test.
VALIDATION_DIR = PKG_ROOT / 'output' / 'validation'
VALIDATION_TROUTE_DIR = PKG_ROOT / 'output' / 'validation_troute'
VALIDATION_CSV_DIR = PKG_ROOT / 'output' / 'validation_csv'

STANDALONE_RUN = VALIDATION_DIR / f'standalone_{CATCHMENT}.npy'
NGEN_RUN = VALIDATION_DIR / f'{CATCHMENT}.csv'
NGEN_ROUTED_RUN = VALIDATION_TROUTE_DIR / f'{CATCHMENT}.csv'
NGEN_CSV_FORCING_RUN = VALIDATION_CSV_DIR / f'{CATCHMENT}.csv'
TROUTE_STREAM_DIR = VALIDATION_TROUTE_DIR / 'stream_output'

# ---------------------------------------------------------------------------- #
#  Tolerances
# ---------------------------------------------------------------------------- #

#: Agreement required to call a setup correct, in m h-1. Repeat runs on one
#: machine are bit-identical and ngen-vs-standalone lands at ~1e-10 (the limit
#: of ngen's 9-significant-figure CSV), so this leaves ~4 orders of headroom for
#: a different BLAS or CPU while still catching any real misconfiguration --
#: a single wrong forcing unit moves the series by ~1e-3.
ATOL = 1e-6

#: Muskingum-Cunge over one 2.1 km reach should neither create nor destroy
#: water. Anything outside this means the routing stack is misconfigured.
ROUTED_VOLUME_TOL = 0.01
ROUTED_MIN_CORR = 0.95


def sha256(path) -> str:
    """Return the SHA-256 hex digest of a file.

    Parameters
    ----------
    path : str or Path
        File to digest.

    Returns
    -------
    str
        Hex digest, or ``'<missing>'`` if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return '<missing>'
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def bmi_config_path(catchment: str = CATCHMENT) -> Path:
    """Return the path to a catchment's BMI config.

    Parameters
    ----------
    catchment : str, optional
        Catchment id, e.g. ``'cat-2453'``.

    Returns
    -------
    Path
        Path to the BMI yaml.
    """
    return DATA_ROOT / 'dhbv_2_mts' / 'config' / f'bmi_{catchment}.yaml'


def package_versions() -> dict[str, str]:
    """Return installed versions of the packages that determine the numbers.

    Returns
    -------
    dict
        Package name to version string, ``'<not installed>'`` when absent.
    """
    versions = {}
    for name in ('dhbv2', 'hydrodl2', 'dmg', 'torch', 'numpy'):
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', '<unknown>')
        except Exception:  # noqa: BLE001 - a missing package is a valid answer
            versions[name] = '<not installed>'
    return versions


# ---------------------------------------------------------------------------- #
#  Loaders
# ---------------------------------------------------------------------------- #


def load_benchmark() -> tuple[np.ndarray, dict]:
    """Load the committed benchmark runoff series and its provenance.

    Returns
    -------
    tuple
        ``(runoff, metadata)`` -- runoff in m h-1 and the provenance dict.
    """
    with np.load(BENCHMARK_PATH, allow_pickle=False) as data:
        runoff = data['runoff'].astype(np.float64)
        metadata = json.loads(str(data['metadata']))
    return runoff, metadata


def load_ngen_csv(path) -> pd.Series:
    """Load a NextGen per-catchment output CSV as a time-indexed runoff series.

    Parameters
    ----------
    path : str or Path
        Path to ngen's ``cat-*.csv`` output.

    Returns
    -------
    pd.Series
        Runoff [m h-1] indexed by timestamp.
    """
    frame = pd.read_csv(path)
    frame.columns = [c.strip() for c in frame.columns]
    frame['Time'] = pd.to_datetime(frame['Time'].astype(str).str.strip())
    return frame.set_index('Time')[RUNOFF_VAR].astype(np.float64)


def load_troute_flow(
    stream_dir=TROUTE_STREAM_DIR,
    feature_id: int = CATCHMENT_FEATURE_ID,
) -> pd.Series:
    """Load routed flow for one feature from t-route's per-hour NetCDF output.

    Parameters
    ----------
    stream_dir : str or Path, optional
        Directory holding ``troute_output_*.nc``.
    feature_id : int, optional
        Numeric hydrofabric feature id, e.g. ``2453``.

    Returns
    -------
    pd.Series
        Routed flow [m3 s-1] indexed by timestamp.
    """
    import xarray as xr

    files = sorted(glob.glob(str(Path(stream_dir) / 'troute_output_*.nc')))
    if not files:
        raise FileNotFoundError(f"No troute_output_*.nc under {stream_dir}")

    times, flows = [], []
    for file in files:
        with xr.open_dataset(file) as dataset:
            flow = dataset['flow'].sel(feature_id=feature_id).values.ravel()
            times.extend(pd.to_datetime(dataset['time'].values))
            flows.extend(np.asarray(flow, dtype=np.float64))

    return pd.Series(flows, index=pd.DatetimeIndex(times)).sort_index()


def runoff_to_cms(runoff: pd.Series, area_km2: float = CATCHMENT_AREA_KM2) -> pd.Series:
    """Convert a runoff depth rate to a volumetric flow rate.

    Parameters
    ----------
    runoff : pd.Series
        Runoff [m h-1].
    area_km2 : float, optional
        Contributing area [km2].

    Returns
    -------
    pd.Series
        Flow [m3 s-1].
    """
    return runoff * area_km2 * 1e6 / 3600.0


def forcing_precip_mm_h(catchment: str = CATCHMENT) -> np.ndarray:
    """Return the catchment's precipitation forcing in the BMI's declared units.

    Parameters
    ----------
    catchment : str, optional
        Catchment id.

    Returns
    -------
    np.ndarray
        Precipitation [mm h-1], one value per hourly step.
    """
    import xarray as xr

    with xr.open_dataset(FORCING_PATH) as dataset:
        series = (
            dataset.set_coords('ids')
            .swap_dims({'catchment-id': 'ids'})
            .sel(ids=catchment)['precip_rate']
            .values
        )
    return np.asarray(series, dtype=np.float64) * 3600.0


# ---------------------------------------------------------------------------- #
#  Metrics
# ---------------------------------------------------------------------------- #


def compare(sim: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Compute agreement metrics between two aligned runoff series.

    Parameters
    ----------
    sim : np.ndarray
        Series under test.
    ref : np.ndarray
        Reference series.

    Returns
    -------
    dict
        ``max_abs_diff``, ``mean_abs_diff``, ``nse``, ``volume_ratio``,
        and ``pearson_r``.
    """
    sim = np.asarray(sim, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    diff = np.abs(sim - ref)

    denom = np.sum((ref - ref.mean()) ** 2)
    nse = 1.0 - np.sum((sim - ref) ** 2) / denom if denom > 0 else np.nan

    ref_total = ref.sum()
    volume_ratio = sim.sum() / ref_total if ref_total != 0 else np.nan

    varies = sim.std() > 0 and ref.std() > 0
    pearson_r = np.corrcoef(sim, ref)[0, 1] if varies else np.nan

    return {
        'max_abs_diff': float(diff.max()),
        'mean_abs_diff': float(diff.mean()),
        'nse': float(nse),
        'volume_ratio': float(volume_ratio),
        'pearson_r': float(pearson_r),
    }


def format_metrics(label: str, metrics: dict[str, float]) -> str:
    """Render a metrics dict as a single readable line.

    Parameters
    ----------
    label : str
        Name of the comparison.
    metrics : dict
        Output of :func:`compare`.

    Returns
    -------
    str
        Formatted summary line.
    """
    return (
        f"{label}: max|diff|={metrics['max_abs_diff']:.3e} "
        f"mean|diff|={metrics['mean_abs_diff']:.3e} "
        f"NSE={metrics['nse']:.9f} "
        f"vol_ratio={metrics['volume_ratio']:.9f} "
        f"r={metrics['pearson_r']:.9f}"
    )


def align(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    """Restrict two time-indexed series to their common timestamps.

    Parameters
    ----------
    left, right : pd.Series
        Time-indexed series to align.

    Returns
    -------
    tuple
        ``(left_values, right_values, n_overlap)``.
    """
    index = left.index.intersection(right.index)
    return left.reindex(index).to_numpy(), right.reindex(index).to_numpy(), len(index)


def missing_artifact_reason(path, produced_by: str) -> Optional[str]:
    """Return a skip reason if a validation artifact has not been produced.

    A *missing* artifact is a legitimate skip -- the user has not run that leg.
    An artifact that exists but disagrees is never skipped; that is the failure
    the suite is here to report.

    Parameters
    ----------
    path : str or Path
        Expected artifact location.
    produced_by : str
        Command that produces it, quoted back to the user.

    Returns
    -------
    str or None
        Skip reason, or None if the artifact is present.
    """
    if Path(path).exists():
        return None
    return f"{Path(path).name} not found at {path}. Produce it with: {produced_by}"
