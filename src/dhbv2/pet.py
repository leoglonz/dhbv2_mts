import numpy as np


def hargreaves(
    tmin: np.ndarray,
    tmax: np.ndarray,
    tmean: np.ndarray,
    lat: np.ndarray,
    day_of_year: np.ndarray,
) -> np.ndarray:
    """Calculate potential evapotranspiration (PET) using Hargreaves method.

    Parameters
    ----------
    tmin
        Minimum daily temperature (°C)
    tmax
        Maximum daily temperature (°C)
    tmean
        Mean daily temperature (°C)
    lat
        Latitude (degrees)
    day_of_year
        Day of the year (1-365/366)

    Returns
    -------
    ndarray
        Potential evapotranspiration (mm/day)
    """
    pet = np.zeros(tmin.shape, dtype=np.float32) * np.NaN
    for ii in np.arange(len(pet[:, 0])):
        trange = tmax[ii, :] - tmin[ii, :]
        trange[trange < 0] = 0

        latitude = np.deg2rad(lat[ii, :])

        SOLAR_CONSTANT = 0.0820

        sol_dec = 0.409 * np.sin((2.0 * np.pi / 365.0) * day_of_year[ii, :] - 1.39)

        sha = np.arccos(np.clip(-np.tan(latitude) * np.tan(sol_dec), -1, 1))

        ird = 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year[ii, :]))

        tmp1 = (24.0 * 60.0) / np.pi
        tmp2 = sha * np.sin(latitude) * np.sin(sol_dec)
        tmp3 = np.cos(latitude) * np.cos(sol_dec) * np.sin(sha)
        et_rad = tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)

        pet[ii, :] = 0.0023 * (tmean[ii, :] + 17.8) * trange**0.5 * 0.408 * et_rad

    pet[pet < 0] = 0

    return pet
