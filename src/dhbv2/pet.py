import numpy as np
import math
import pandas as pd


def hargreaves_pet(
    tmin: np.ndarray,
    tmax: np.ndarray,
    tmean: np.ndarray,
    lat: np.ndarray,
    day_of_year: np.ndarray,
) -> np.ndarray:
    """
    Calculate potential evapotranspiration (PET) using Hargreaves method and
    daily forcings.

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
    SOLAR_CONSTANT = 0.0820

    trange = np.maximum(tmax - tmin, 0)

    sol_dec = 0.409 * np.sin((2.0 * np.pi / 365.0) * day_of_year - 1.39)
    sha = np.arccos(np.clip(-np.tan(lat) * np.tan(sol_dec), -1, 1))
    ird = 1 + 0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year)

    et_rad = (
        (24.0 * 60.0 / np.pi)
        * SOLAR_CONSTANT
        * ird
        * (
            sha * np.sin(lat) * np.sin(sol_dec)
            + np.cos(lat) * np.cos(sol_dec) * np.sin(sha)
        )
    )

    pet = 0.0023 * (tmean + 17.8) * np.sqrt(trange) * 0.408 * et_rad
    pet = np.maximum(pet, 0)
    return pet


def calc_hourly_hargreaves_pet(
    temp,
    srad,
    lat,
    time_idx,
):
    """
    Compute hourly PET using daily Hargreaves but with hourly inputs.
    PET is allocated to hours according to fractional shortwave radiation.

    Returns
    -------
    ndarray
        PET in mm/hour
    """
    # 1. Convert inputs
    lat_rad = np.deg2rad(lat)
    Nh = len(temp)

    # 2. Extract daily groupings
    time_index = pd.to_datetime(time_idx)
    days = time_index.normalize().unique()
    Ndays = len(days)

    # Prepare arrays
    tmin_d = np.zeros(Ndays)
    tmax_d = np.zeros(Ndays)
    tmean_d = np.zeros(Ndays)
    doy_d = np.zeros(Ndays, dtype=int)
    sw_sum_d = np.zeros(Ndays)

    # Map each hour to its day index
    day_index = time_index.normalize()
    day_to_idx = {d: i for i, d in enumerate(days)}
    idx_daily = np.array([day_to_idx[d] for d in day_index])

    # 3. Aggregate hourly → daily
    for d_idx, day in enumerate(days):
        mask = idx_daily == d_idx
        th = temp[mask]
        sw = srad[mask]

        tmin_d[d_idx] = th.min()
        tmax_d[d_idx] = th.max()
        tmean_d[d_idx] = th.mean()
        sw_sum_d[d_idx] = sw.sum()
        doy_d[d_idx] = day.timetuple().tm_yday

    # 4. Compute DAILY PET
    pet_daily = hargreaves_pet(tmin_d, tmax_d, tmean_d, lat_rad, doy_d)

    # 5. Distribute daily PET to hourly PET
    pet_hourly = np.zeros(Nh)

    for h in range(Nh):
        d = idx_daily[h]
        daily_sw = sw_sum_d[d]
        # If no radiation (polar night or bad data), spread uniformly
        if daily_sw > 0:
            frac = srad[h] / daily_sw
        else:
            frac = 1.0 / np.sum(idx_daily == d)
        pet_hourly[h] = pet_daily[d] * frac

    return pet_hourly


def hourly_pet_penman_monteith(
    temp,
    spfh,
    dswrf,
    dlwrf,
    pressure,
    ugrd_10m,
    vgrd_10m,
    albedo=0.23,
):
    """
    Computes hourly PET (mm/hour) using the FAO Penman-Monteith formulation.

    Parameters
    ----------
    temp
        Air temperature in Kelvin.
    spfh
        Specific humidity (kg/kg or g/g depending on AORC version).
    dswrf
        Downward shortwave radiation in W/m².
    dlwrf
        Downward longwave radiation in W/m².
    pressure
        Atmospheric pressure in Pascals.
    ugrd_10m
        U-component of wind at 10 meters in m/s.
    vgrd_10m
        V-component of wind at 10 meters in m/s.
    albedo
        Surface albedo (default is 0.23 for grass).
    """
    # --- Temperature (C) ---
    T = temp - 273.15

    # --- Convert specific humidity (AORC sometimes in g/g) ---
    if spfh > 0.02:  # crude check: g/g range is like 0–20 g/kg
        q = spfh / 1000.0
    else:
        q = spfh

    # --- Wind speed at 10 m and convert to 2 m (FAO) ---
    u10 = math.sqrt(ugrd_10m**2 + vgrd_10m**2)
    u2 = u10 * 4.87 / math.log(67.8 * 10 - 5.42)

    # --- Atmospheric pressure (kPa) ---
    P_kPa = pressure / 1000.0

    # --- Psychrometric constant (kPa/°C) ---
    gamma = 0.000665 * P_kPa

    # --- Saturation vapor pressure (kPa) ---
    es = 0.6108 * math.exp((17.27 * T) / (T + 237.3))

    # --- Actual vapor pressure (kPa) ---
    # e = q * P / (0.622 + 0.378 q)
    ea = (q * P_kPa) / (0.622 + 0.378 * q)

    # --- Slope of saturation vapor pressure curve (kPa/°C) ---
    delta = (4098 * es) / (T + 237.3) ** 2

    # --- Net radiation ---
    # Convert downward radiation to MJ/m2/hr
    Rs = dswrf * 0.0036  # shortwave (incoming)
    Rl_down = dlwrf * 0.0036  # longwave (downwelling)

    # Estimate upwelling longwave using Stefan–Boltzmann
    sigma = 4.903e-9  # MJ/K4/m2/day   (FAO56 units)
    # Convert sigma for hourly: divide by 24
    sigma_hr = sigma / 24.0

    Rl_up = sigma_hr * (temp**4)  # VERY approximate without emissivity

    Rn = (1 - albedo) * Rs + (Rl_down - Rl_up)

    # --- Soil heat flux G (FAO: zero for hourly) ---
    G = 0

    # --- Penman–Monteith FAO-56 equation for hourly ET (mm/hr) ---
    ET = (
        0.408 * delta * (Rn - G) + gamma * (900.0 / (T + 273.15)) * u2 * (es - ea)
    ) / (delta + gamma * (1 + 0.34 * u2))

    # ensure non-negative
    return max(0.0, ET)
