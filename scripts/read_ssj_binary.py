#!/usr/bin/env python
"""
Reader for JHU/APL DMSP SSJ/4/5 binary data files.

Format specification from:
  - Tom Sotirelis, JHU/APL (read_ssj_file.pro, conv_struc.pro)
  - http://sd-www.jhuapl.edu/Aurora/software/index.html

File format: 128 bytes per record, LITTLE-ENDIAN byte order, 1 record per second.

Record layout (128 bytes):
  Bytes   0:    Satellite number (uint8)
  Bytes   1:    Year - last two digits (uint8)
  Bytes 2-3:    Day of year (int16)
  Bytes 4-5:    Geodetic latitude, tenths of degree (int16)
  Bytes 6-7:    Geodetic longitude, tenths of degree (int16)
  Bytes 8-47:   20 electron flux counts (20 x int16)
  Bytes 48-87:  20 ion flux counts (20 x int16)
  Bytes 88-91:  Seconds into day (int32)
  Bytes 92-93:  Version number * 1000 (int16)
  Byte  94:      Original position source flag (uint8)
  Byte  95:      Filler (zero)
  Byte  96:      NORAD error flag (uint8, 0=good, 1=error)
  Byte  97:      Geomagnetic error flag (uint8, 0=good, 1=error)
  Bytes 98-101:  NORAD latitude, ten-thousandths of degree (int32)
  Bytes 102-105: NORAD longitude, ten-thousandths of degree (int32)
  Bytes 106-109: NORAD altitude, decimeters (int32)
  Bytes 110-113: Magnetic latitude, ten-thousandths of degree (int32)
  Bytes 114-117: Magnetic longitude, ten-thousandths of degree (int32)
  Bytes 118-121: MLT, hundred-thousandths of hour (int32)
  Bytes 122-123: PACE Model Year (int16)
  Bytes 124-127: Filler (zeros)

Channel layout:
  SSJ/4 (F06-F15): 20 channels each for electrons and ions
    Electrons: channels 0-9 (high-E, 30keV to 1keV), channels 10-19 (low-E, 1keV to 30eV)
    Ions:       channels 0-9 (high-E), channels 10-19 (low-E)
    Channel 9 (e) / channel 10 (i) is the redundant 1keV overlap channel

  SSJ/5 (F16-F20): 20 channels each for electrons and ions
    Channels 0-8 and 10-19 contain data (19 unique channels)
    Channel 9 is the redundant 1keV overlap (set to zero or status)
    To get 19 physical channels: [0:9] + [10:20] -> indices 0-18

Count encoding:
  Values 0-32000: direct counts
  Values > 32000: rescaled as count = 32000 + (value - 32000) * 100

Channel energies (eV), 19 channels in decreasing order:
  [30000, 20400, 13900, 9450, 6460, 4400, 3000, 2040, 1392, 949,
   646, 440, 300, 204, 139, 95, 65, 44, 30]

Reference:
  Wang et al. (2023), "Horse-Collar Auroral Morphology Driven by
  Flow Shear at the Poleward Edge of the Ring Current"

@author: Auto-generated from JHU/APL IDL source code
"""

import struct
import numpy as np
from datetime import datetime, timedelta

# SSJ/5 channel energies in eV (19 channels, high to low)
CHANNEL_ENERGIES = np.array([
    30000, 20400, 13900, 9450, 6460, 4400, 3000, 2040, 1392, 949,
    646, 440, 300, 204, 139, 95, 65, 44, 30
], dtype=np.float64)

RECORD_SIZE = 128  # bytes


def rescale_counts(counts):
    """
    Rescale SSJ count values that exceed 32000.

    In the JHU/APL format, count values > 32000 are encoded as:
        actual_count = 32000 + (stored_value - 32000) * 100

    Parameters
    ----------
    counts : int or np.ndarray
        Raw count values from the binary file.

    Returns
    -------
    rescaled : same type as input
        Rescaled count values.
    """
    counts = np.asarray(counts, dtype=np.float64)
    mask = counts > 32000
    counts[mask] = 32000.0 + (counts[mask] - 32000.0) * 100.0
    return counts


def get_19_channels(counts_20, sat_number):
    """
    Extract 19 physical channels from 20-channel SSJ data,
    removing the redundant 1 keV overlap channel.

    For SSJ/5 (F16-F20):
        Redundant channel is index 9.
        Physical channels: [0:9] + [10:20] = 19 channels

    For SSJ/4 (F06-F15):
        Electron redundant channel: index 9
        Ion redundant channel: index 10
        (This function removes index 9 for both species;
         for SSJ/4 ions the user should handle index 10 separately.)

    Parameters
    ----------
    counts_20 : np.ndarray, shape (20,) or (20, N)
        20-channel count values.
    sat_number : int
        DMSP satellite number (e.g., 16, 17, 18).

    Returns
    -------
    counts_19 : np.ndarray, shape (19,) or (19, N)
        19-channel count values with redundant channel removed.
    """
    if sat_number >= 16:
        # SSJ/5: remove channel 9 (redundant 1 keV)
        return np.concatenate([counts_20[:9], counts_20[10:20]], axis=0)
    else:
        # SSJ/4: remove channel 9 (electron redundant) or 10 (ion redundant)
        return np.concatenate([counts_20[:9], counts_20[10:20]], axis=0)


def read_ssj_file(filepath):
    """
    Read a JHU/APL DMSP SSJ/4/5 binary data file.

    Parameters
    ----------
    filepath : str
        Path to the binary data file (e.g., '2015apr10.f18').

    Returns
    -------
    data : dict
        Dictionary containing all parsed data fields:
        - 'satellite'     : int, DMSP satellite number (Fxx)
        - 'year'         : int, full year (e.g., 2015)
        - 'doy'          : int, day of year (1-366)
        - 'glat'         : np.ndarray, geodetic latitude (degrees)
        - 'glon'         : np.ndarray, geodetic longitude (degrees)
        - 'eflux'        : np.ndarray, shape (20, N), electron raw counts
        - 'iflux'        : np.ndarray, shape (20, N), ion raw counts
        - 'eflux_19'     : np.ndarray, shape (19, N), electron counts (19 ch)
        - 'iflux_19'     : np.ndarray, shape (19, N), ion counts (19 ch)
        - 'eflux_rescaled': np.ndarray, rescaled electron counts
        - 'iflux_rescaled': np.ndarray, rescaled ion counts
        - 'sod'          : np.ndarray, seconds of day
        - 'datetime'     : np.ndarray of datetime objects
        - 'version'      : int, data file version
        - 'orig_pos_flag': np.ndarray, original position source flag
        - 'norad_err'    : np.ndarray, NORAD error flag
        - 'geomag_err'   : np.ndarray, geomagnetic error flag
        - 'nlat'         : np.ndarray, NORAD latitude (degrees)
        - 'nlon'         : np.ndarray, NORAD longitude (degrees)
        - 'nalt'         : np.ndarray, NORAD altitude (meters)
        - 'mlat'         : np.ndarray, magnetic latitude (degrees)
        - 'mlon'         : np.ndarray, magnetic longitude (degrees)
        - 'mlt'          : np.ndarray, magnetic local time (hours)
        - 'pace_year'    : int, PACE model year
        - 'n_records'    : int, number of records read
        - 'channel_energies': np.ndarray, 19 channel energies (eV)

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file size is not a multiple of 128 bytes.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    file_size = len(raw)
    if file_size % RECORD_SIZE != 0:
        raise ValueError(
            f"File size {file_size} is not a multiple of record size "
            f"({RECORD_SIZE} bytes). File may be corrupted or not in "
            f"JHU/APL SSJ format."
        )

    n_records = file_size // RECORD_SIZE
    data = np.frombuffer(raw, dtype=np.uint8).reshape(n_records, RECORD_SIZE)

    # --- Parse header fields (same for all records) ---
    sat = int(data[0, 0])
    yr_two_digit = int(data[0, 1])
    year = yr_two_digit + 2000 if yr_two_digit < 80 else yr_two_digit + 1900

    # --- Parse per-record fields ---

    # Use numpy dtype views for efficient parsing of multi-byte fields.
    # This avoids the overhead of struct.unpack on large arrays.

    # Day of year (int16 LE, bytes 2-3)
    doy = data[:, 2:4].view('<i2').flatten()

    # Geodetic latitude (int16 LE, bytes 4-5, tenths of degree)
    glat = data[:, 4:6].view('<i2').flatten().astype(np.float64) / 10.0

    # Geodetic longitude (int16 LE, bytes 6-7, tenths of degree)
    glon = data[:, 6:8].view('<i2').flatten().astype(np.float64) / 10.0

    # Electron flux counts (20 x int16 LE, bytes 8-47)
    eflux = data[:, 8:48].view('<i2').reshape(20, n_records).copy()

    # Ion flux counts (20 x int16 LE, bytes 48-87)
    iflux = data[:, 48:88].view('<i2').reshape(20, n_records).copy()

    # Seconds of day (int32 LE, bytes 88-91)
    sod = data[:, 88:92].view('<i4').flatten()

    # Version (int16 LE, bytes 92-93)
    version = data[0, 92:94].view('<h')[0]

    # Byte flags
    orig_pos_flag = data[:, 94].astype(np.uint8)
    norad_err = data[:, 96].astype(np.uint8)
    geomag_err = data[:, 97].astype(np.uint8)

    # NORAD position (int32 LE each)
    nlat = data[:, 98:102].view('<i4').flatten().astype(np.float64) / 10000.0
    nlon = data[:, 102:106].view('<i4').flatten().astype(np.float64) / 10000.0
    nalt = data[:, 106:110].view('<i4').flatten().astype(np.float64) / 10.0  # decimeters to meters

    # Magnetic coordinates (int32 LE each)
    mlat = data[:, 110:114].view('<i4').flatten().astype(np.float64) / 10000.0
    mlon = data[:, 114:118].view('<i4').flatten().astype(np.float64) / 10000.0

    # MLT (int32 LE, bytes 118-121, hundred-thousandths of hour)
    mlt = data[:, 118:122].view('<i4').flatten().astype(np.float64) / 100000.0

    # PACE Model Year (int16 LE, bytes 122-123)
    pace_year = data[0, 122:124].view('<h')[0]

    # --- Derived fields ---

    # Rescale counts (>32000 encoding)
    eflux_rescaled = rescale_counts(eflux)
    iflux_rescaled = rescale_counts(iflux)

    # Get 19 physical channels (remove redundant 1 keV channel)
    eflux_19 = get_19_channels(eflux, sat)
    iflux_19 = get_19_channels(iflux, sat)
    eflux_19_rescaled = rescale_counts(eflux_19)
    iflux_19_rescaled = rescale_counts(iflux_19)

    # Create datetime array
    base_date = datetime(year, 1, 1)
    datetimes = np.array(
        [base_date + timedelta(days=int(d) - 1, seconds=int(s))
         for d, s in zip(doy, sod)],
        dtype=object
    )

    # Use NORAD position where available, fall back to original position
    use_norad = norad_err == 0
    glat_final = np.where(use_norad, nlat, glat)
    glon_final = np.where(use_norad, nlon, glon)

    return {
        'satellite': sat,
        'year': year,
        'doy': doy,
        'glat': glat_final,
        'glon': glon_final,
        'glat_original': glat,
        'glon_original': glon,
        'eflux': eflux,
        'iflux': iflux,
        'eflux_19': eflux_19,
        'iflux_19': iflux_19,
        'eflux_rescaled': eflux_rescaled,
        'iflux_rescaled': iflux_rescaled,
        'eflux_19_rescaled': eflux_19_rescaled,
        'iflux_19_rescaled': iflux_19_rescaled,
        'sod': sod,
        'datetime': datetimes,
        'version': version,
        'orig_pos_flag': orig_pos_flag,
        'norad_err': norad_err,
        'geomag_err': geomag_err,
        'nlat': nlat,
        'nlon': nlon,
        'nalt': nalt,
        'mlat': mlat,
        'mlon': mlon,
        'mlt': mlt,
        'pace_year': pace_year,
        'n_records': n_records,
        'channel_energies': CHANNEL_ENERGIES,
    }


def find_polar_passes(mlat, min_abs_mlat=50.0):
    """
    Find polar pass segments where |mlat| > min_abs_mlat.

    Parameters
    ----------
    mlat : np.ndarray
        Magnetic latitude array.
    min_abs_mlat : float
        Minimum |mlat| to consider as polar (degrees).

    Returns
    -------
    passes : list of tuples
        List of (start_index, end_index) tuples for each polar pass.
    """
    in_polar = np.abs(mlat) > min_abs_mlat
    # Find transitions
    transitions = np.diff(in_polar.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    if len(starts) == 0 or len(ends) == 0:
        return []

    # Align starts and ends
    if starts[0] > ends[0]:
        starts = np.concatenate([[0], starts])
    if ends[-1] < starts[-1]:
        ends = np.concatenate([ends, [len(mlat)]])

    passes = list(zip(starts, ends))
    return passes


def main():
    """Demonstrate reading a JHU/APL SSJ binary file."""
    import sys
    import os

    # Default file paths
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'dmsp'
    )

    test_files = [
        os.path.join(data_dir, '2015apr10.f18'),
        os.path.join(data_dir, '2015apr10.f17'),
    ]

    # Filter to existing files
    test_files = [f for f in test_files if os.path.exists(f)]

    if not test_files:
        print(f"No test files found in {data_dir}")
        print("Place .fxx files there and re-run.")
        return

    filepath = test_files[0]
    print(f"Reading: {filepath}")
    print(f"File size: {os.path.getsize(filepath)} bytes")

    data = read_ssj_file(filepath)

    print(f"\n{'='*60}")
    print(f"JHU/APL SSJ Binary File Summary")
    print(f"{'='*60}")
    print(f"  Satellite:       F{data['satellite']:02d}")
    print(f"  Year:            {data['year']}")
    print(f"  Day of Year:      {data['doy'][0]}")
    print(f"  Number of records: {data['n_records']}")
    print(f"  Version:         {data['version']/1000:.3f}")
    print(f"  Time range:       {data['datetime'][0]} to {data['datetime'][-1]}")
    print(f"  GeoLat range:     [{data['glat'].min():.2f}, {data['glat'].max():.2f}]")
    print(f"  GeoLon range:     [{data['glon'].min():.2f}, {data['glon'].max():.2f}]")
    print(f"  NoradAlt range:    [{data['nalt'].min():.1f}, {data['nalt'].max():.1f}] m")
    print(f"  MagLat range:     [{data['mlat'].min():.2f}, {data['mlat'].max():.2f}]")
    print(f"  MLT range:        [{data['mlt'].min():.2f}, {data['mlt'].max():.2f}] h")
    print(f"  NORAD error recs:  {np.sum(data['norad_err'] > 0)}")
    print(f"  Geomag error recs: {np.sum(data['geomag_err'] > 0)}")

    # Show first few records
    print(f"\n{'='*60}")
    print(f"First 5 records:")
    print(f"{'='*60}")
    print(f"{'Rec':>5s} {'UT':>10s} {'GeoLat':>8s} {'GeoLon':>8s} "
          f"{'MagLat':>8s} {'MLT':>6s} {'Alt(km)':>7s} "
          f"{'e_ch0':>6s} {'e_ch9':>6s} {'e_ch18':>6s} "
          f"{'i_ch0':>6s} {'i_ch9':>6s} {'i_ch18':>6s}")
    print(f"{'---':>5s} {'---':>10s} {'---':>8s} {'---':>8s} "
          f"{'---':>8s} {'---':>6s} {'---':>7s} "
          f"{'---':>6s} {'---':>6s} {'---':>6s} "
          f"{'---':>6s} {'---':>6s} {'---':>6s}")

    for i in range(min(5, data['n_records'])):
        dt = data['datetime'][i]
        tstr = dt.strftime('%H:%M:%S')
        print(f"{i:5d} {tstr:>10s} {data['glat'][i]:8.2f} {data['glon'][i]:8.2f} "
              f"{data['mlat'][i]:8.2f} {data['mlt'][i]:6.2f} "
              f"{data['nalt'][i]/1000:7.1f} "
              f"{data['eflux_19'][0,i]:6d} {data['eflux_19'][9,i]:6d} "
              f"{data['eflux_19'][18,i]:6d} "
              f"{data['iflux_19'][0,i]:6d} {data['iflux_19'][9,i]:6d} "
              f"{data['iflux_19'][18,i]:6d}")

    # Find polar passes
    passes = find_polar_passes(data['mlat'])
    print(f"\n{'='*60}")
    print(f"Polar passes (|mlat| > 50 deg): {len(passes)}")
    print(f"{'='*60}")
    for idx, (s, e) in enumerate(passes[:10]):
        print(f"  Pass {idx+1}: records {s}-{e} "
              f"({e-s}s, mlat=[{data['mlat'][s]:.1f},{data['mlat'][e]:.1f}]), "
              f"MLT=[{data['mlt'][s]:.1f},{data['mlt'][e]:.1f}])")

    # Channel energies
    print(f"\n{'='*60}")
    print(f"Channel energies (eV): {data['channel_energies']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
