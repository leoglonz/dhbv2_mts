"""
Create a geopackage from the CONUS NextGen HydroFabric v2.2 for a selection of
catchments. Used as a ngen input.

Get the latest hydrofabric here: https://www.lynker-spatial.com

Available layers:
    flowpaths (default), divides, lakes, nexus, pois, hydrolocations,
    flowpath-attributes, flowpath-attributes-ml, network, divide-attributes

Currently configured to only extract 'divides', 'nexus', and 'flowpaths' layers.

@leoglonz
"""

import geopandas as gpd
from pathlib import Path

# Setup pathing
pkg_root = Path(__file__).parent


### -------- Settings -------- ###
HF_PATH = '/projects/mhpi/data/hydrofabric/v2.2/conus_nextgen.gpkg'
SAVE_PATH = f'{pkg_root}/camels_hf2.gpkg'
CAT_IDS = [2453, 2454, 2455]
### -------------------------- ###


if __name__ == '__main__':
    divides = gpd.read_file(HF_PATH, layer='divides')
    nexus = gpd.read_file(HF_PATH, layer='nexus')
    flowpaths = gpd.read_file(HF_PATH, layer='flowpaths')

    cats = [f'cat-{cat_id}' for cat_id in CAT_IDS]
    divides_new = divides[divides['divide_id'].isin(cats)]

    toid = divides_new['toid']
    nexs = [f'nex-{toid_val[4:]}' for toid_val in toid]
    nexus_new = nexus[nexus['id'].isin(nexs)]
    flowpaths_new = flowpaths[flowpaths['divide_id'].isin(cats)]

    divides_new.to_file(SAVE_PATH, layer='divides', driver='GPKG', mode='w')
    nexus_new.to_file(SAVE_PATH, layer='nexus', driver='GPKG', mode='a')
    flowpaths_new.to_file(SAVE_PATH, layer='flowpaths', driver='GPKG', mode='a')

    print(f"Saved GPKG to: {SAVE_PATH}")
