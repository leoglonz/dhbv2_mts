"""Building hourly CAMELS forcing dataset.
3 catchments, 2010-2012.

Mirrors structure of ./ngen/data/forcing/cat-67_2015-12-01 00_00_00_2015-12-30 23_00_00.csv
"""

import pandas as pd
import numpy as np
import xarray as xr

n_cat = 3
t_start = '2010-01-01 00:00:00'
t_end = '2011-12-30 23:00:00'

example_path = '/projects/mhpi/leoglonz/ciroh-ua/ciroh-ua-ngen/data/forcing/cat-67_2015-12-01 00_00_00_2015-12-30 23_00_00.csv'
camels_path = '/gpfs/yxs275/data/hourly/CAMELS_HF/forcing/forcing_1990_2018_gauges_hourly_00000_00499.nc'
out_path = f'/projects/mhpi/leoglonz/ciroh-ua/dhbv2_mts/ngen_resources/data/forcing/camels_{t_start.replace(":", "_").replace(" ", "_")}_{t_end.replace(":", "_").replace(" ", "_")}.nc'

df = pd.read_csv(example_path)
camels_xr = xr.open_dataset(camels_path)

out_df = pd.DataFrame()


def transform_dataset(ds_in):
    """Transform CAMELS dataset to match target structure."""
    # 1. Rename Dimensions first
    ds = ds_in.rename({'gauge': 'catchment-id'})

    # --- INSERT SUBSETTING HERE (Before creating new vars or dropping coords) ---
    # Use .isel (index select) for the first 100 catchments
    # Use .sel (label select) for the specific date range
    ds = ds.isel({'catchment-id': slice(0, n_cat)})
    ds = ds.sel(time=slice(t_start, t_end))
    # --------------------------------------------------------------------------

    # 2. Rename Existing Variables
    ds = ds.rename(
        {
            'P': 'APCP_surface',
            'T': 'TMP_2maboveground',
            'PET': 'PET_hargreaves',
        },
    )

    # 3. Create 'ids' Variable
    raw_ids = ds['catchment-id'].values.astype(str)

    # Prepend "cat-" using numpy's string operations
    # cat_ids will be like ["cat-2453", "cat-2454", ...]
    cat_ids = np.char.add('cat-', raw_ids)

    ds['ids'] = (('catchment-id',), cat_ids)

    # 4. Create 'Time' Variable with Nanosecond Units
    # .view('int64') accesses the raw nanoseconds from the datetime64[ns] object
    # .astype('float64') converts that integer count to a float
    time_values = ds['time'].values.view('int64').astype('float64')

    time_broadcasted = np.tile(
        time_values,
        (ds.sizes['catchment-id'], 1),
    )

    ds['Time'] = (('catchment-id', 'time'), time_broadcasted)

    # Explicitly add the unit metadata to match your target
    ds['Time'].attrs = {'units': 'ns'}

    # 5. Create Zero-Filled Variables
    zero_vars = [
        'DLWRF_surface',
        'DSWRF_surface',
        'PRES_surface',
        'SPFH_2maboveground',
        'UGRD_10maboveground',
        'VGRD_10maboveground',
        'precip_rate',
    ]

    # Create zeros based on the new subset shape
    shape = (ds.sizes['catchment-id'], ds.sizes['time'])
    zeros = np.zeros(shape, dtype='float32')

    for var in zero_vars:
        ds[var] = (('catchment-id', 'time'), zeros)

    # 6. NOW it is safe to drop the coordinates
    ds = ds.drop_vars(['time', 'catchment-id'])

    return ds


formatted_ds = transform_dataset(camels_xr)

# save to nc
formatted_ds.to_netcdf(out_path)
