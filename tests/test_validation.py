"""End-to-end validation that a δHBV2.0 MTS install reproduces the benchmark.


Run the shipped `cat-2453` example through whatever ngen you already have, then
point the suite at the output:

    ngen <gpkg> cat-2453 <gpkg> nex-2454 \
        ./data/dhbv_2_mts/realizations/realization_troute_cat-2453.json

    pytest tests/test_validation.py --run-dir=/path/to/that/output

The t-route realization writes both `cat-2453.csv` and `troute_output_*.nc`,
so one run covers both ngen legs. Use `realization_nc_cat-2453.json` instead
if you are not validating routing; the t-route tests then skip.

Each leg skips only when its artifact is absent, and says how to produce it. An
artifact that exists but disagrees with the benchmark always fails.

Legs
----
standalone   BMI driven directly by Python, against the committed benchmark.
             Needs no ngen at all: `python scripts/mts_forward_example.py`.
ngen         The same BMI driven by your ngen, against the committed benchmark.
             Catches forcing-unit and variable-mapping errors in a realization,
             which are invisible to the standalone leg.
t-route      Your ngen runoff routed by Muskingum-Cunge, checked for volume
             conservation against the unrouted series that fed it.

A water-balance guard runs against whatever series are available and needs no
benchmark at all: runoff may not exceed precipitation.

@leoglonz
"""

from pathlib import Path

import numpy as np
import pytest

from validation import (
    ATOL,
    BENCHMARK_PATH,
    CATCHMENT,
    FIRST_FLOW_STEP,
    N_STEPS_FULL,
    N_STEPS_ROUTED,
    ROUTED_MIN_CORR,
    ROUTED_VOLUME_TOL,
    STANDALONE_RUN,
    align,
    compare,
    find_ngen_csv,
    find_troute_dir,
    forcing_precip_mm_h,
    format_metrics,
    load_benchmark,
    load_ngen_csv,
    load_troute_flow,
    require_artifact,
    runoff_to_cms,
)

RUN_STANDALONE = "python scripts/mts_forward_example.py"

# Where to look first when a leg disagrees with the benchmark. The standalone
# leg passing while ngen fails points at the realization, not the model.
HINT_REALIZATION = (
    "Check `variables_names_map`, and that every forcing's units are ones ngen "
    "can convert to what the BMI declares via get_var_units()."
)


def assert_matches(sim, ref, label: str, capsys, hint: str = '') -> None:
    """Assert two runoff series agree within ATOL, reporting metrics either way.

    Parameters
    ----------
    sim
        Series under test, as an array or pandas Series.
    ref
        Reference series.
    label
        Name of the comparison, used in the reported line.
    capsys
        pytest capture fixture, so metrics print on success as well as failure.
    hint
        Where to look first if the comparison fails.
    """
    metrics = compare(np.asarray(sim), np.asarray(ref))
    with capsys.disabled():
        print('\n' + format_metrics(label, metrics))

    assert metrics['max_abs_diff'] <= ATOL, (
        f"{label} departs by {metrics['max_abs_diff']:.3e} m/h "
        f"(tolerance {ATOL:.0e}).\n{format_metrics('  metrics', metrics)}"
        + (f"\n  {hint}" if hint else '')
    )


# ---------------------------------------------------------------------------- #
#  Fixtures
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='module')
def run_dir(request):
    """Directory holding the validator's own ngen output, from ``--run-dir``."""
    path = request.config.getoption('--run-dir')
    if not path:
        pytest.skip(
            "No --run-dir given. Run the shipped cat-2453 example through your "
            "own ngen, then re-run with --run-dir=<that run's output>.",
        )
    return Path(path)


@pytest.fixture(scope='module')
def benchmark():
    """Committed benchmark runoff series and its metadata."""
    try:
        return load_benchmark()
    except FileNotFoundError:
        pytest.skip(
            f"No committed benchmark at {BENCHMARK_PATH}. Regenerate with "
            f"scripts/utils/make_mts_benchmark.py (maintainers only).",
        )


@pytest.fixture(scope='module')
def standalone():
    """Runoff series from the standalone BMI driver."""
    require_artifact(STANDALONE_RUN, RUN_STANDALONE)
    return np.load(STANDALONE_RUN).astype(np.float64)


@pytest.fixture(scope='module')
def ngen(run_dir):
    """Runoff series from the validator's own ngen run."""
    return load_ngen_csv(find_ngen_csv(run_dir))


@pytest.fixture(scope='module')
def troute(run_dir):
    """Routed flow from the validator's own t-route output."""
    return load_troute_flow(find_troute_dir(run_dir))


# ---------------------------------------------------------------------------- #
#  Benchmark integrity
# ---------------------------------------------------------------------------- #


class TestBenchmarkIntegrity:
    """The benchmark itself must describe the case these tests assume."""

    def test_length(self, benchmark):
        """Benchmark should span the full shipped forcing window."""
        runoff, _ = benchmark
        assert runoff.shape == (N_STEPS_FULL,)

    def test_units_are_metres_per_hour(self, benchmark):
        """Benchmark should be stored in the BMI's external units, m h-1.

        A benchmark accidentally stored in mm h-1 is 1000x too large and would
        make every comparison fail for the wrong reason.
        """
        runoff, meta = benchmark
        assert meta['units'] == 'm h-1'
        assert runoff.max() < 0.1, (
            f"Peak runoff {runoff.max():.4g} is too large to be m h-1."
        )

    def test_spinup_then_flow(self, benchmark):
        """Benchmark should be zero through spin-up, then produce flow."""
        runoff, _ = benchmark
        assert np.all(runoff[:FIRST_FLOW_STEP] == 0.0)
        assert np.count_nonzero(runoff[FIRST_FLOW_STEP:]) > 0

    def test_metadata_is_complete(self, benchmark):
        """Provenance should be recorded so mismatches can be diagnosed."""
        _, meta = benchmark
        for key in (
            'catchment',
            'start_time',
            'end_time',
            'bmi_config_sha256',
            'model_config_sha256',
            'forcing_sha256',
            'package_versions',
        ):
            assert key in meta, f"Benchmark metadata missing '{key}'"
        assert meta['catchment'] == CATCHMENT


# ---------------------------------------------------------------------------- #
#  Leg 1: standalone vs benchmark
# ---------------------------------------------------------------------------- #


class TestStandaloneReproducesBenchmark:
    """Standalone BMI run must reproduce the committed hydrograph."""

    def test_length_matches(self, standalone, benchmark):
        """A short or long run is a failure, not a reason to skip."""
        runoff, _ = benchmark
        assert standalone.shape == runoff.shape, (
            f"Standalone run has {standalone.size} steps, benchmark has "
            f"{runoff.size}; it did not cover the full forcing window."
        )

    def test_matches_benchmark(self, standalone, benchmark, capsys):
        """Runoff must match the benchmark to within ATOL."""
        runoff, meta = benchmark
        assert_matches(
            standalone,
            runoff,
            'standalone vs benchmark',
            capsys,
            hint=(
                f"Benchmark generated {meta.get('generated_utc')} with "
                f"{meta.get('package_versions')}. If configs changed, compare "
                f"the recorded hashes before assuming a code regression."
            ),
        )


# ---------------------------------------------------------------------------- #
#  Leg 2: ngen vs benchmark
# ---------------------------------------------------------------------------- #


class TestNgenReproducesBenchmark:
    """NextGen must feed the BMI the same forcings as standalone."""

    def test_length_matches(self, ngen, benchmark):
        """ngen should emit one row per hour of the full window."""
        runoff, _ = benchmark
        assert len(ngen) == runoff.size, (
            f"ngen produced {len(ngen)} steps, benchmark has {runoff.size}; "
            f"check `time.start_time`/`end_time` in the realization."
        )

    def test_matches_benchmark(self, ngen, benchmark, capsys):
        """ngen runoff must match the benchmark to within ATOL.

        Catches realization errors. A forcing mapped to the wrong variable, or a
        unit ngen cannot convert, moves the hydrograph by ~1e-3 m/h -- three
        orders above tolerance.
        """
        runoff, _ = benchmark
        assert_matches(ngen, runoff, 'ngen vs benchmark', capsys, HINT_REALIZATION)

    def test_matches_standalone(self, ngen, standalone, capsys):
        """The two drive paths must agree with each other.

        Independent of the benchmark, so it still works after an intentional
        model change that has not yet been promoted.
        """
        assert_matches(ngen, standalone, 'ngen vs standalone', capsys)


# ---------------------------------------------------------------------------- #
#  Leg 3: t-route
# ---------------------------------------------------------------------------- #


class TestTrouteRouting:
    """Routed flow must be a plausible transformation of runoff."""

    def test_covers_full_post_spinup_period(self, troute, capsys):
        """t-route must route every hour after spin-up, not a subset.

        `nts` and `max_loop_size` in routing_config.yaml are pinned to the
        realization's window; if the window moves and they do not, t-route
        silently routes a shorter period than the model simulated.
        """
        with capsys.disabled():
            print(
                f"\nt-route window: {len(troute)} steps, "
                f"{troute.index[0]} -> {troute.index[-1]}",
            )

        assert len(troute) == N_STEPS_ROUTED, (
            f"t-route routed {len(troute)} hours; the post-spin-up period is "
            f"{N_STEPS_ROUTED} hours. Check `nts` (routed hours x "
            f"qts_subdivisions = {N_STEPS_ROUTED * 12}) and `max_loop_size` in "
            f"routing_config.yaml."
        )
        assert troute.max() > 0.0, (
            "All routed flow is zero; the routing window likely sits inside "
            "model spin-up. Check `start_datetime` in routing_config.yaml "
            "against the realization's `start_time` plus spin-up length."
        )

    def test_conserve_volume(self, troute, ngen, capsys):
        """Muskingum-Cunge over a single reach must not create or destroy water."""
        routed, unrouted, n_overlap = align(troute, runoff_to_cms(ngen))
        assert n_overlap > 0, (
            "t-route and ngen output share no timestamps. Check "
            "`start_datetime` in routing_config.yaml against the realization."
        )

        metrics = compare(routed, unrouted)
        with capsys.disabled():
            print(
                f"\nt-route vs unrouted: n={n_overlap} "
                f"vol_ratio={metrics['volume_ratio']:.6f} "
                f"r={metrics['pearson_r']:.6f} mean={routed.mean():.5f} m3/s",
            )

        assert abs(metrics['volume_ratio'] - 1.0) <= ROUTED_VOLUME_TOL, (
            f"Routed volume is {metrics['volume_ratio']:.4f}x the runoff that "
            f"fed it (tolerance {ROUTED_VOLUME_TOL:.0%}). Check "
            f"`qts_subdivisions`, `dt`, and the geopackage channel parameters."
        )
        assert metrics['pearson_r'] >= ROUTED_MIN_CORR, (
            f"Routed and unrouted hydrographs correlate at only "
            f"{metrics['pearson_r']:.4f}; expect near-unity over one short reach."
        )


# ---------------------------------------------------------------------------- #
#  Benchmark-free physical guard
# ---------------------------------------------------------------------------- #


class TestWaterBalance:
    """Runoff cannot exceed precipitation."""

    @staticmethod
    def _runoff_ratio(runoff_m_h: np.ndarray) -> float:
        precip_mm_h = forcing_precip_mm_h()[: runoff_m_h.size]
        post = slice(FIRST_FLOW_STEP, None)
        return float(runoff_m_h[post].sum() * 1000.0 / precip_mm_h[post].sum())

    def test_standalone_runoff_ratio(self, standalone, capsys):
        """Post-spin-up runoff must be a sane fraction of precipitation."""
        ratio = self._runoff_ratio(standalone)
        with capsys.disabled():
            print(f"\nstandalone runoff ratio: {ratio:.4f}")

        assert 0.0 < ratio < 1.0, (
            f"Runoff/precipitation = {ratio:.4f} over the post-spin-up period. "
            f"A ratio at or above 1.0 points at a forcing-unit error "
            f"(temperature in K rather than degC suppresses snow and PET) "
            f"rather than at model skill."
        )

    def test_benchmark_runoff_ratio(self, benchmark, capsys):
        """The committed benchmark must itself satisfy the water balance."""
        runoff, _ = benchmark
        ratio = self._runoff_ratio(runoff)
        with capsys.disabled():
            print(f"\nbenchmark runoff ratio: {ratio:.4f}")
        assert 0.0 < ratio < 1.0
