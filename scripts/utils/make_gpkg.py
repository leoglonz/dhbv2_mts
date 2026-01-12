"""
Create a geopackage from the CONUS NextGen HydroFabric v2.2 for a selection of
catchments. Used as a ngen input.

Get the latest hydrofabric here: https://www.lynker-spatial.com

Available layers:
    flowpaths (default), divides, lakes, nexus, pois, hydrolocations,
    flowpath-attributes, flowpath-attributes-ml, network, divide-attributes

Currently configured to extract 'divides', 'nexus', 'flowpaths', and 'network'
layers.

@leoglonz
"""

import geopandas as gpd
from pathlib import Path
import sqlite3

# Setup pathing
pkg_root = Path(__file__).parent.parent.parent


### -------- Settings -------- ###
HF_PATH = '/projects/mhpi/data/hydrofabric/v2.2/conus_nextgen.gpkg'
SAVE_PATH = f'{pkg_root}/ngen_resources/data/geo/camels_subset_hf2.gpkg'
CAT_IDS = [2453, 2454, 2455]
### -------------------------- ###


if __name__ == '__main__':
    divides = gpd.read_file(HF_PATH, layer='divides')
    nexus = gpd.read_file(HF_PATH, layer='nexus')
    flowpaths = gpd.read_file(HF_PATH, layer='flowpaths')
    network = gpd.read_file(HF_PATH, layer='network')
    flowpath_attributes = gpd.read_file(HF_PATH, layer='flowpath-attributes')
    flowpath_attributes_ml = gpd.read_file(HF_PATH, layer='flowpath-attributes-ml')
    divide_attributes = gpd.read_file(HF_PATH, layer='divide-attributes')

    cats = [f'cat-{cat_id}' for cat_id in CAT_IDS]
    cats_alt = [f'wb-{cat_id}' for cat_id in CAT_IDS]
    divides_new = divides[divides['divide_id'].isin(cats)]

    toid = divides_new['toid']
    nexs = [f'nex-{toid_val[4:]}' for toid_val in toid]
    nexus_new = nexus[nexus['id'].isin(nexs)]
    flowpaths_new = flowpaths[flowpaths['divide_id'].isin(cats)]
    network_new = network[network['divide_id'].isin(cats)]
    flowpath_attributes_new = flowpath_attributes[
        flowpath_attributes['id'].isin(cats_alt)
    ]
    flowpath_attributes_ml_new = flowpath_attributes_ml[
        flowpath_attributes_ml['id'].isin(cats_alt)
    ]
    divide_attributes_new = divide_attributes[divide_attributes['divide_id'].isin(cats)]

    divides_new.to_file(SAVE_PATH, layer='divides', driver='GPKG', mode='w')
    nexus_new.to_file(SAVE_PATH, layer='nexus', driver='GPKG', mode='a')
    flowpaths_new.to_file(SAVE_PATH, layer='flowpaths', driver='GPKG', mode='a')
    with sqlite3.connect(SAVE_PATH) as conn:
        network_new.to_sql('network', conn, if_exists='replace', index=False)
        flowpath_attributes_new.to_sql(
            'flowpath-attributes',
            conn,
            if_exists='replace',
            index=False,
        )
        flowpath_attributes_ml_new.to_sql(
            'flowpath-attributes-ml',
            conn,
            if_exists='replace',
            index=False,
        )
        divide_attributes_new.to_sql(
            'divide-attributes',
            conn,
            if_exists='replace',
            index=False,
        )

    print(f"Saved GPKG to: {SAVE_PATH}")
