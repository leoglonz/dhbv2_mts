"""End-to-end validation of δHBV2.0 MTS install for benchmark reproduction.

Unlike the rest of the suite, these tests do not exercise BMI plumbing in
isolation -- they compare *whole simulations* produced by three routes to the
same hydrograph, and they need artifacts that a prior run has to produce:

    ./scripts/run_validation.sh

Each leg skips only when its artifact is absent, and says which command
produces it. An artifact that exists but disagrees with the benchmark always
fails; it must never be reported as a pass.

Legs
----
standalone   BMI driven directly by Python, against the committed benchmark.
ngen         The same BMI driven by NextGen, against the committed benchmark.
             Catches forcing-unit and variable-mapping errors in a realization,
             which are invisible to the standalone leg.
t-route      NextGen runoff routed by Muskingum-Cunge, checked for volume
             conservation against the unrouted series that fed it.

A water-balance guard runs against whatever series are available and needs no
benchmark at all: runoff may not exceed precipitation.

@leoglonz
"""

import numpy as np
import pytest

from validation import (
    ATOL,
    CATCHMENT,
    N_STEPS_CSV_EXAMPLE,
    NGEN_CSV_FORCING_RUN,
    FIRST_FLOW_STEP,
    N_STEPS_FULL,
    N_STEPS_ROUTED,
    NGEN_ROUTED_RUN,
    NGEN_RUN,
    ROUTED_MIN_CORR,
    ROUTED_VOLUME_TOL,
    STANDALONE_RUN,
    TROUTE_STREAM_DIR,
    align,
    compare,
    format_metrics,
    forcing_precip_mm_h,
    load_benchmark,
    load_ngen_csv,
    load_troute_flow,
    missing_artifact_reason,
    runoff_to_cms,
)

RUN_STANDALONE = "python scripts/mts_forward_example.py"
RUN_NGEN = "./scripts/run_validation.sh ngen"
RUN_TROUTE = "./scripts/run_validation.sh troute"
RUN_CSV = "./scripts/run_validation.sh csv"


# ---------------------------------------------------------------------------- #
#  Fixtures
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='module')
def benchmark():
    """Committed benchmark runoff series and its provenance metadata."""
    try:
        return load_benchmark()
    except FileNotFoundError:
        pytest.skip(
            "No committed benchmark. Regenerate with "
            "scripts/make_mts_benchmark.py (maintainers only).",
        )


@pytest.fixture(scope='module')
def standalone():
    """Runoff series from the standalone BMI driver."""
    reason = missing_artifact_reason(STANDALONE_RUN, RUN_STANDALONE)
    if reason:
        pytest.skip(reason)
    return np.load(STANDALONE_RUN).astype(np.float64)


@pytest.fixture(scope='module')
def ngen():
    """Runoff series from the full-window NextGen run."""
    reason = missing_artifact_reason(NGEN_RUN, RUN_NGEN)
    if reason:
        pytest.skip(reason)
    return load_ngen_csv(NGEN_RUN)


@pytest.fixture(scope='module')
def ngen_routed():
    """Unrouted runoff from the NextGen run that fed t-route."""
    reason = missing_artifact_reason(NGEN_ROUTED_RUN, RUN_TROUTE)
    if reason:
        pytest.skip(reason)
    return load_ngen_csv(NGEN_ROUTED_RUN)


@pytest.fixture(scope='module')
def ngen_csv_forcing():
    """Runoff from the NextGen run driven by CSV forcing rather than NetCDF."""
    reason = missing_artifact_reason(NGEN_CSV_FORCING_RUN, RUN_CSV)
    if reason:
        pytest.skip(reason)
    return load_ngen_csv(NGEN_CSV_FORCING_RUN)


@pytest.fixture(scope='module')
def troute():
    """Routed flow from t-route's stream output."""
    reason = missing_artifact_reason(TROUTE_STREAM_DIR, RUN_TROUTE)
    if reason:
        pytest.skip(reason)
    return load_troute_flow()


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
            f"Peak runoff {runoff.max():.4g} m/h ({runoff.max() * 1000:.4g} mm/h) "
            f"is too large to be m h-1; the benchmark may be in mm h-1."
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
    """The standalone driver must reproduce the committed hydrograph."""

    def test_length_matches(self, standalone, benchmark):
        """A short or long run is a failure, not a reason to skip."""
        runoff, _ = benchmark
        assert standalone.shape == runoff.shape, (
            f"Standalone run has {standalone.size} steps, benchmark has "
            f"{runoff.size}. The run did not cover the full forcing window."
        )

    def test_matches_benchmark(self, standalone, benchmark, capsys):
        """Runoff must match the benchmark to within ATOL."""
        runoff, meta = benchmark
        assert standalone.shape == runoff.shape, (
            "length mismatch; see test_length_matches"
        )

        metrics = compare(standalone, runoff)
        with capsys.disabled():
            print('\n' + format_metrics('standalone vs benchmark', metrics))

        assert metrics['max_abs_diff'] <= ATOL, (
            f"Standalone run departs from the benchmark by "
            f"{metrics['max_abs_diff']:.3e} m/h (tolerance {ATOL:.0e}).\n"
            f"{format_metrics('  metrics', metrics)}\n"
            f"  benchmark generated {meta.get('generated_utc')} with "
            f"{meta.get('package_versions')}\n"
            f"  If configs changed, compare the recorded hashes before assuming "
            f"a code regression."
        )


# ---------------------------------------------------------------------------- #
#  Leg 2: ngen vs benchmark
# ---------------------------------------------------------------------------- #


class TestNgenReproducesBenchmark:
    """NextGen must feed the BMI the same forcings the standalone driver does."""

    def test_length_matches(self, ngen, benchmark):
        """ngen should emit one row per hour of the full window."""
        runoff, _ = benchmark
        assert len(ngen) == runoff.size, (
            f"ngen produced {len(ngen)} steps, benchmark has {runoff.size}. "
            f"Check `time.start_time`/`end_time` in the realization."
        )

    def test_matches_benchmark(self, ngen, benchmark, capsys):
        """ngen runoff must match the benchmark to within ATOL.

        This is the leg that catches realization errors. A forcing mapped to the
        wrong variable, or a unit ngen cannot convert, moves the hydrograph by
        ~1e-3 m/h -- three orders above tolerance.
        """
        runoff, _ = benchmark
        assert len(ngen) == runoff.size, "length mismatch; see test_length_matches"

        metrics = compare(ngen.to_numpy(), runoff)
        with capsys.disabled():
            print('\n' + format_metrics('ngen vs benchmark', metrics))

        assert metrics['max_abs_diff'] <= ATOL, (
            f"NextGen output departs from the benchmark by "
            f"{metrics['max_abs_diff']:.3e} m/h (tolerance {ATOL:.0e}).\n"
            f"{format_metrics('  metrics', metrics)}\n"
            f"  The standalone leg passing while this one fails points at the "
            f"realization, not the model: check `variables_names_map` and that "
            f"every forcing's units are ones ngen can convert to the units the "
            f"BMI declares via get_var_units()."
        )

    def test_matches_standalone(self, ngen, standalone, capsys):
        """The two drive paths must agree with each other.

        Independent of the benchmark, so it still works after an intentional
        model change that has not yet been promoted.
        """
        assert len(ngen) == standalone.size, (
            f"ngen has {len(ngen)} steps, standalone has {standalone.size}"
        )

        metrics = compare(ngen.to_numpy(), standalone)
        with capsys.disabled():
            print('\n' + format_metrics('ngen vs standalone', metrics))

        assert metrics['max_abs_diff'] <= ATOL, (
            f"NextGen and standalone disagree by {metrics['max_abs_diff']:.3e} m/h "
            f"(tolerance {ATOL:.0e}).\n{format_metrics('  metrics', metrics)}"
        )


class TestCsvForcingProvider:
    """The two forcing providers must be interchangeable.

    `realization_cat-2453.json` reads per-feature CSVs and
    `realization_nc_cat-2453.json` reads the NetCDF they were generated from.
    Both must hand the model identical numbers, which depends on the CSV
    carrying `NAME[units]` headers that ngen can convert from.
    """

    def test_matches_benchmark(self, ngen_csv_forcing, benchmark, capsys):
        """CSV-driven runoff must match the benchmark over its shorter window."""
        runoff, _ = benchmark
        assert len(ngen_csv_forcing) == N_STEPS_CSV_EXAMPLE, (
            f"CSV-forced run has {len(ngen_csv_forcing)} steps, expected "
            f"{N_STEPS_CSV_EXAMPLE}"
        )

        metrics = compare(
            ngen_csv_forcing.to_numpy(),
            runoff[:N_STEPS_CSV_EXAMPLE],
        )
        with capsys.disabled():
            print('\n' + format_metrics('ngen (CSV forcing) vs benchmark', metrics))

        assert metrics['max_abs_diff'] <= ATOL, (
            f"CSV-forced run departs from the benchmark by "
            f"{metrics['max_abs_diff']:.3e} m/h (tolerance {ATOL:.0e}).\n"
            f"Regenerate the CSV with scripts/make_csv_forcing.py, and check "
            f"that its column headers still carry units ngen can convert "
            f"(`NAME[units]`) and that `variables_names_map` references the "
            f"bare name without the bracket."
        )


# ---------------------------------------------------------------------------- #
#  Leg 3: t-route
# ---------------------------------------------------------------------------- #


class TestTrouteRouting:
    """Routed flow must be a plausible transformation of the runoff that fed it."""

    def test_unrouted_run_matches_benchmark(self, ngen_routed, benchmark, capsys):
        """The runoff that fed t-route must itself match the benchmark.

        Confirms the routed leg simulated the same thing as the other two, so a
        routing discrepancy cannot be blamed on a different simulation.
        """
        runoff, _ = benchmark
        assert len(ngen_routed) == runoff.size, (
            f"Routed run has {len(ngen_routed)} steps, expected {runoff.size}"
        )

        metrics = compare(ngen_routed.to_numpy(), runoff)
        with capsys.disabled():
            print('\n' + format_metrics('ngen (routing run) vs benchmark', metrics))

        assert metrics['max_abs_diff'] <= ATOL

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
            f"{N_STEPS_ROUTED} hours. Check `nts` "
            f"(routed hours x qts_subdivisions = {N_STEPS_ROUTED} x 12 = "
            f"{N_STEPS_ROUTED * 12}) and `max_loop_size` in routing_config.yaml."
        )
        assert troute.max() > 0.0, (
            "All routed flow is zero. The routing window likely sits inside "
            "model spin-up: check `start_datetime` in routing_config.yaml "
            "against the realization's `start_time` plus spin-up length."
        )

    def test_conserves_volume(self, troute, ngen_routed, capsys):
        """Muskingum-Cunge over a single reach must not create or destroy water."""
        unrouted_cms = runoff_to_cms(ngen_routed)
        routed, unrouted, n_overlap = align(troute, unrouted_cms)

        assert n_overlap > 0, (
            "t-route output and ngen output share no timestamps. Check "
            "`start_datetime` in routing_config.yaml against the realization."
        )

        volume_ratio = routed.sum() / unrouted.sum()
        correlation = np.corrcoef(routed, unrouted)[0, 1]
        with capsys.disabled():
            print(
                f"\nt-route vs unrouted: n={n_overlap} "
                f"vol_ratio={volume_ratio:.6f} r={correlation:.6f} "
                f"mean={routed.mean():.5f} m3/s",
            )

        assert abs(volume_ratio - 1.0) <= ROUTED_VOLUME_TOL, (
            f"Routed volume is {volume_ratio:.4f}x the runoff that fed it "
            f"(tolerance {ROUTED_VOLUME_TOL:.0%}). Routing is not conserving "
            f"water -- check `qts_subdivisions`, `dt`, and the geopackage "
            f"channel parameters."
        )
        assert correlation >= ROUTED_MIN_CORR, (
            f"Routed and unrouted hydrographs correlate at only "
            f"{correlation:.4f}. Expect near-unity over a single short reach."
        )


# ---------------------------------------------------------------------------- #
#  Benchmark-free physical guard
# ---------------------------------------------------------------------------- #


class TestWaterBalance:
    """Runoff cannot exceed precipitation, whatever the benchmark says.

    This is the check that would have caught feeding Kelvin to a model that
    declares degC: with no snow and a broken PET, the runoff ratio went above
    1.0. It needs no stored benchmark, so it is still meaningful on a new
    catchment or a retrained model.
    """

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
            f"A ratio at or above 1.0 means the catchment is yielding more "
            f"water than it receives, which points at a forcing-unit error "
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
