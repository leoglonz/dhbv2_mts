import os
import re
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
from icechunk.xarray import to_icechunk

# Setup pathing
root = Path(__file__).parent


# --- Configuration ---
CSV_INPUT_DIR = Path("/projects/mhpi/leoglonz/ciroh-ua/noaa-ngen/data/output")
LOCAL_SAVE_DESTINATION = f'{root}/data/runoff'
BATCH_SIZE = 50  # Number of catchments to commit at once


def extract_id_from_filename(filename):
    """
    Extracts numerical ID from filename.
    Adjust the regex based on your actual naming convention.
    """
    match = re.search(r'\d+', filename)
    return match.group() if match else filename


def load_csv_to_ds(file_path):
    """Reads a single CSV and returns a 1D Xarray Dataset."""
    df = pd.read_csv(file_path)
    # Ensure time is datetime objects
    df['Time'] = pd.to_datetime(df['Time'])

    divide_id = extract_id_from_filename(file_path.name)

    # Create dataset
    ds = xr.Dataset(
        data_vars={
            'runoff': (
                ["time"],
                df['land_surface_water__runoff_volume_flux'].values.astype(np.float32),
            ),
        },
        coords={
            'time': (["time"], df['Time'].values),
            'divide_id': divide_id,
        },
    )
    # Expand dimension so we can concatenate along divide_id
    return ds.expand_dims("divide_id")


def main():
    """Main processing function."""
    # 1. Initialize Icechunk Repository
    if not os.path.exists(LOCAL_SAVE_DESTINATION):
        storage_config = icechunk.local_filesystem_storage(
            path=str(LOCAL_SAVE_DESTINATION),
        )
        repo = icechunk.Repository.create(storage_config)
    else:
        storage_config = icechunk.local_filesystem_storage(
            path=str(LOCAL_SAVE_DESTINATION),
        )
        repo = icechunk.Repository.open(storage_config)

    session = repo.writable_session("main")

    # 2. Gather CSV files
    csv_files = list(CSV_INPUT_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} files. Starting conversion...")

    # 3. Process in batches
    is_first_batch = True

    for i in range(0, len(csv_files), BATCH_SIZE):
        batch_files = list(CSV_INPUT_DIR.glob("cat-*.csv"))
        batch_ds_list = [load_csv_to_ds(f) for f in batch_files]

        # Combine batch into one dataset
        combined_batch = xr.concat(batch_ds_list, dim="divide_id")

        # Icechunk requires object type for strings usually to match your reference script
        combined_batch["divide_id"] = combined_batch["divide_id"].astype(object)

        if is_first_batch:
            # Initial write
            to_icechunk(combined_batch, session)
            msg = f"Initial commit with {len(batch_files)} catchments"
            is_first_batch = False
        else:
            # Append along the divide_id dimension
            to_icechunk(combined_batch, session, append_dim='divide_id')
            msg = f"Appended batch {i // BATCH_SIZE} with {len(batch_files)} catchments"

        # Commit batch
        latest_snapshot = session.commit(msg)
        print(f"{msg}: {latest_snapshot}")

        # Re-open session for next batch
        session = repo.writable_session("main")

    print("Processing complete.")


if __name__ == "__main__":
    main()
