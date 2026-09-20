"""Reaquire δHBV2.0 MTS validation benchmark.

Usage:
    python scripts/mts_forward_example.py
    python scripts/utils/make_mts_benchmark.py [run.npy]

The benchmark is an npz carrying the runoff series. The BMI
and model config hashes, the forcing file hash, the simulation window, and the
versions of the three packages that produced it are also included to ensure
reproducibility and comparability.

@leoglonz
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tests'))
from validation import (
    BENCHMARK_PATH,
    CATCHMENT,
    FORCING_PATH,
    MODEL_CONFIG_PATH,
    N_STEPS_FULL,
    RUNOFF_UNITS,
    SIM_END,
    SIM_START,
    STANDALONE_RUN,
    bmi_config_path,
    package_versions,
    sha256,
)


def main() -> int:
    """Write ``tests/benchmarks/`` from a completed standalone run."""
    run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else STANDALONE_RUN
    if not run_path.exists():
        print(f"No run found at {run_path}. Run scripts/mts_forward_example.py first.")
        return 1

    runoff = np.load(run_path).astype(np.float64)
    if runoff.size != N_STEPS_FULL:
        print(
            f"{run_path} has {runoff.size} steps, but the benchmark case is "
            f"{N_STEPS_FULL}. Refusing to promote a run of the wrong window.",
        )
        return 1

    meta = {
        'catchment': CATCHMENT,
        'units': RUNOFF_UNITS,
        'start_time': SIM_START,
        'end_time': SIM_END,
        'n_steps': int(runoff.size),
        'time_step_seconds': 3600,
        'source_run': str(run_path.name),
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'bmi_config_sha256': sha256(bmi_config_path(CATCHMENT)),
        'model_config_sha256': sha256(MODEL_CONFIG_PATH),
        'forcing_sha256': sha256(FORCING_PATH),
        'package_versions': package_versions(),
    }

    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BENCHMARK_PATH,
        runoff=runoff,
        metadata=np.array(json.dumps(meta, indent=2)),
    )
    print(f"Wrote {BENCHMARK_PATH} ({runoff.size} steps)")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
