"""Derive NextGen per-feature CSV forcing from the shipped NetCDF forcing file.

NextGen can read forcing through either provider, and this repo ships an
example of each: `realization_cat-2453.json` uses `CsvPerFeature`,
`realization_nc_cat-2453.json` uses `NetCDF`. Both must produce the same
hydrograph, so the CSV is generated from the NetCDF rather than maintained
separately.

Two details make the two providers interchangeable:

* **Units travel in the header.** ngen parses a `NAME[units]` (or `NAME(units)`)
  column header, strips the bracket, and converts the values to whatever the
  BMI declares through `get_var_units()`. The units written here are copied
  verbatim from the NetCDF variable attributes, so both providers hand the
  model identical numbers. The realization's `variables_names_map` must
  reference the *bare* name -- ngen has already removed the bracket by the time
  the map is applied.
* **Values round-trip exactly.** The NetCDF stores float32; `%.9g` is the
  shortest decimal form that recovers a float32 without loss, so a CSV-driven
  run reproduces a NetCDF-driven run bit for bit.

Usage
-----
    python scripts/make_csv_forcing.py
    python scripts/make_csv_forcing.py cat-2454 cat-2455

@leoglonz
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PKG_ROOT = Path(__file__).parent.parent
FORCING_DIR = PKG_ROOT / 'ngen_resources' / 'data' / 'forcing'
NETCDF_PATH = FORCING_DIR / 'camels_subset_2008-01-09 00_00_00_2010-12-30 23_00_00.nc'

#: The eight forcings the MTS BMI consumes, in the order they are written.
VARIABLES = [
    'precip_rate',
    'TMP_2maboveground',
    'SPFH_2maboveground',
    'DLWRF_surface',
    'DSWRF_surface',
    'PRES_surface',
    'UGRD_10maboveground',
    'VGRD_10maboveground',
]

#: float32 round-trips exactly at 9 significant decimal digits.
VALUE_FORMAT = '%.9g'

#: Timestamp form used in forcing filenames, matching the NetCDF file.
NAME_STAMP_FORMAT = '%Y-%m-%d %H_%M_%S'


def write_csv(catchment: str) -> Path:
    """Write one catchment's forcing CSV beside the NetCDF file.

    Parameters
    ----------
    catchment : str
        Catchment id, e.g. ``'cat-2453'``.

    Returns
    -------
    Path
        The file that was written.
    """
    with xr.open_dataset(NETCDF_PATH) as dataset:
        selected = (
            dataset.set_coords('ids')
            .swap_dims({'catchment-id': 'ids'})
            .sel(ids=catchment)
        )
        # `Time` is an epoch-seconds variable, not a datetime coordinate.
        timestamps = pd.to_datetime(
            selected['Time'].values,
            unit=selected['Time'].attrs.get('units', 's'),
            origin=pd.Timestamp(
                selected['Time'].attrs.get('epoch_start', '01/01/1970 00:00:00'),
            ),
        )
        columns = {name: selected[name].values for name in VARIABLES}
        units = {name: selected[name].attrs['units'] for name in VARIABLES}

    header = 'time,' + ','.join(f'{name}[{units[name]}]' for name in VARIABLES)
    stacked = np.column_stack([columns[name] for name in VARIABLES])

    out_path = FORCING_DIR / (
        f'{catchment}'
        f'_{timestamps[0]:{NAME_STAMP_FORMAT}}'
        f'_{timestamps[-1]:{NAME_STAMP_FORMAT}}.csv'
    )

    # A realization matches forcing with `.*{{id}}.*\.csv`, so a second CSV for
    # the same catchment -- an earlier date range, say -- leaves ngen with two
    # candidates and no way to choose. Say so rather than quietly adding one.
    stale = [
        other for other in FORCING_DIR.glob(f'{catchment}*.csv') if other != out_path
    ]
    if stale:
        print(
            f"  WARNING: {len(stale)} other CSV(s) for {catchment} are already "
            f"here and will also match a realization's file_pattern. "
            f"Remove them: " + ', '.join(f.name for f in stale),
        )

    with open(out_path, 'w') as out:
        out.write(header + '\n')
        for stamp, row in zip(timestamps, stacked):
            values = ','.join(VALUE_FORMAT % v for v in row)
            out.write(f"{stamp:%Y-%m-%d %H:%M:%S},{values}\n")

    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path.name}: {len(timestamps)} rows, {size_mb:.1f} MB")
    return out_path


def main() -> int:
    """Write a CSV for each catchment named on the command line."""
    catchments = sys.argv[1:] or ['cat-2453']
    for catchment in catchments:
        write_csv(catchment)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
