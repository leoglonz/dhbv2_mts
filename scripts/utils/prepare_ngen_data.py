"""Building hourly forcing dataset for CAMELS catchments.
Creates NETCDF and CSV files suitable for ngen forcing input.

Mirrors structure of ngen forcing files:
- ./ngen/data/forcing/cats-27_52_67-2015_12_01-2015_12_30.nc
- ./ngen/data/forcing/cat-67_2015-12-01 00_00_00_2015-12-30 23_00_00.csv
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr

# Setup pathing
pkg_root = Path(__file__).parent.parent.parent


### -------- Settings -------- ###
CAT_IDS = [2453, 2454, 2455]
T_START = '2008-01-09 00:00:00'  # 2004-10-01 00:00:00
T_END = '2015-12-30 23:00:00'  # 2018-09-30 23:00:00

DATA_PATH = '/gpfs/yxs275/data/hourly/CAMELS_HF/forcing/forcing_1990_2018_gauges_hourly_00000_00499.nc'
OUT_DIR = os.path.join(pkg_root, "ngen_resources/data/forcing/")

SAVE_CSV = True
### -------------------------- ###


def transform_dataset(ds_in, cat_ids):
    """Transform CAMELS dataset to match target structure."""
    ds = ds_in.sel(gauge=cat_ids)
    ds = ds.rename({'gauge': 'catchment-id'})
    ds = ds.sel(time=slice(T_START, T_END))

    # Rename existing vars
    ds = ds.rename(
        {
            'P': 'precip_rate[mm h-1]',
            'T': 'TMP_2maboveground',
            'PET': 'PET_hargreaves',
        },
    )

    for var in ds.variables:
        ds[var].encoding = {}

    # Create 'ids' var
    raw_ids = ds['catchment-id'].values.astype(str)
    cat_ids = np.char.add('cat-', raw_ids)
    ds['ids'] = (('catchment-id',), cat_ids)

    # Create 'Time' var (nanoseconds)
    time_values = ds['time'].values.view('int64').astype('float64')
    time_broadcasted = np.tile(time_values, (ds.sizes['catchment-id'], 1))
    ds['Time'] = (('catchment-id', 'time'), time_broadcasted)
    ds['Time'].attrs = {'units': 'ns'}

    # Create zero-filled vars
    zero_vars = [
        'APCP_surface',
        'DLWRF_surface',
        'DSWRF_surface',
        'PRES_surface',
        'SPFH_2maboveground',
        'UGRD_10maboveground',
        'VGRD_10maboveground',
    ]

    shape = (ds.sizes['catchment-id'], ds.sizes['time'])
    zeros = np.zeros(shape, dtype='float64')

    for var in zero_vars:
        ds[var] = (('catchment-id', 'time'), zeros)
        ds[var].encoding = {'_FillValue': None}

    # Convert to temp to kelvin
    ds['TMP_2maboveground'] = ds['TMP_2maboveground'].astype('float64') + np.float64(
        273.15,
    )

    # Standardize units
    ds['TMP_2maboveground'].attrs['units'] = 'degK'
    ds['precip_rate[mm h-1]'].attrs['units'] = 'mm h-1'
    ds['PET_hargreaves'].attrs['units'] = 'mm h-1'
    ds['APCP_surface'].attrs['units'] = 'kg m-2'
    ds['DLWRF_surface'].attrs['units'] = 'W m-2'
    ds['DSWRF_surface'].attrs['units'] = 'W m-2'
    ds['PRES_surface'].attrs['units'] = 'Pa'
    ds['SPFH_2maboveground'].attrs['units'] = 'g g-1'
    ds['UGRD_10maboveground'].attrs['units'] = 'm s-1'
    ds['VGRD_10maboveground'].attrs['units'] = 'm s-1'

    ds['TMP_2maboveground'].encoding = {'dtype': 'float64'}

    return ds


def export_to_csv(ds, out_dir):
    """Exports each catchment in xarray dataset to a separate csv."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created directory: {out_dir}")

    # time_idx = pd.to_datetime(ds['time'].values)

    for i in range(ds.sizes['catchment-id']):
        cat_name = ds['ids'].values[i]

        # Select data for this catchment and convert to dataframe
        df = ds.isel({'catchment-id': i}).to_dataframe()

        if 'ids' in df.columns:
            df = df.drop(columns=['ids'])

        df = df[['precip_rate[mm h-1]', 'TMP_2maboveground', 'PET_hargreaves']]

        fmt_t_start = T_START.replace(':', '_')
        fmt_t_end = T_END.replace(':', '_')
        csv_path = os.path.join(
            OUT_DIR,
            f"{cat_name}_{fmt_t_start}_{fmt_t_end}.csv",
        )
        df.to_csv(csv_path, index_label='time')
        print(f"Exported {csv_path}")


def verify_integrity(original_path, new_path, t_start, t_end, n_cat):
    """Verify that the data transformation preserved integrity."""
    print("Verifying data integrity...")
    ds_old = xr.open_dataset(original_path)
    ds_new = xr.open_dataset(new_path)

    # Slice old dataset exactly like the new one for comparison
    ds_old_sub = ds_old.isel(gauge=slice(0, n_cat)).sel(time=slice(t_start, t_end))

    # 1. Verify precipitation
    np.testing.assert_array_equal(
        ds_old_sub['P'].values,
        ds_new['precip_rate'].values,
        err_msg="Precipitation values drifted!",
    )
    print("✓ Precipitation matches exactly.")

    # 2. Verify temperature
    expected_temp = ds_old_sub['T'].values
    print(ds_old_sub['T'].sel(time='2008-01-09T00:00:00')[0].item())
    print((ds_new['TMP_2maboveground'].values - 273.15)[0][0].item())
    np.testing.assert_allclose(
        expected_temp,
        ds_new['TMP_2maboveground'].values - 273.15,
        rtol=1e-16,
        err_msg="Temperature math introduced errors!",
    )
    print("✓ Temperature conversion matches within float64 tolerance.")

    print("SUCCESS: No numerical discrepancies detected.")


if __name__ == '__main__':
    with xr.open_dataset(DATA_PATH) as camels_xr:
        formatted_ds = transform_dataset(camels_xr, CAT_IDS)

    if SAVE_CSV:
        export_to_csv(formatted_ds, OUT_DIR)

    final_nc = formatted_ds.drop_vars(['time', 'catchment-id'])

    fmt_t_start = T_START.replace(':', '_')
    fmt_t_end = T_END.replace(':', '_')
    nc_path = os.path.join(
        OUT_DIR,
        f"camels_{fmt_t_start}_{fmt_t_end}.nc",
    )

    # Save to nc
    final_nc.to_netcdf(
        nc_path,
        format='NETCDF4',
        engine='netcdf4',
    )
    print(f"Saved NETCDF to: {nc_path}")

    # Run verification
    # verify_integrity(camels_path, out_path, t_start, t_end, n_cat)
