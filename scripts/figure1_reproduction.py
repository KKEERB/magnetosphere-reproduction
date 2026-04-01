#!/usr/bin/env python
"""
Figure 1 Reproduction: IMF, solar wind conditions and auroral electrojet
Wang et al. (2023) - Communications Earth & Environment

Layout (6 panels):
  (a) Three IMF components (BX, BY, BZ) in GSE coordinates
  (b) Solar wind number density and speed
  (c) Solar wind dynamic pressure (PDyn) and SYM-H index
  (d) Provisional auroral electrojet indices (AL and AU)
  (e) Zoomed IMF for 8:30-12:30 UT on April 10
  (f) Zoomed PDyn for 8:30-12:30 UT on April 10

Data source: OMNI HRO2 1-minute data (lagged 7 min for bow shock propagation)
Period: 9-11 April 2015 (with zoom on April 10, 8:30-12:30 UT)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')


def load_omni_data():
    """Load OMNI HRO2 1-minute CDF data."""
    from spacepy import pycdf

    cdf_path = os.path.join(PROJECT_DIR, 'omni_data', 'hro2_1min', '2015',
                            'omni_hro2_1min_20150401_v01.cdf')
    cdf = pycdf.CDF(cdf_path)

    epoch = np.array(cdf['Epoch'][:])

    data = {
        'time': epoch,
        'bx': cdf['BX_GSE'][:].astype(np.float64),
        'by': cdf['BY_GSE'][:].astype(np.float64),
        'bz': cdf['BZ_GSE'][:].astype(np.float64),
        'speed': cdf['flow_speed'][:].astype(np.float64),
        'vx': cdf['Vx'][:].astype(np.float64),
        'density': cdf['proton_density'][:].astype(np.float64),
        'pressure': cdf['Pressure'][:].astype(np.float64),
        'sym_h': cdf['SYM_H'][:].astype(np.float64),
        'al': cdf['AL_INDEX'][:].astype(np.float64),
        'au': cdf['AU_INDEX'][:].astype(np.float64),
    }

    # Replace fill values with NaN
    for key in ['bx', 'by', 'bz', 'speed', 'vx', 'density', 'pressure',
                'sym_h', 'al', 'au']:
        data[key][data[key] > 9000] = np.nan

    # Time lag: shift data 7 minutes backward for bow shock propagation
    # (paper states IMF data lagged by 7 min)
    lag_minutes = 7
    epoch_dt = np.array([datetime.fromisoformat(str(t)) if hasattr(t, '__str__') else t
                         for t in epoch], dtype='datetime64[m]')
    data['time_lagged'] = epoch_dt - np.timedelta64(lag_minutes, 'm')

    return data


def select_date_range(data, date_start, date_end):
    """Select data for a specific date range."""
    mask = (data['time_lagged'] >= np.datetime64(date_start)) & \
           (data['time_lagged'] < np.datetime64(date_end))
    result = {}
    for key in data:
        if isinstance(data[key], np.ndarray):
            result[key] = data[key][mask]
        else:
            result[key] = data[key]
    return result


def create_figure1(data, outdir):
    """
    Create Figure 1 reproduction: 6-panel overview of IMF and solar wind.
    """
    # Select April 9-11 data
    d = select_date_range(data, '2015-04-09', '2015-04-12')

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), dpi=200,
                             gridspec_kw={'hspace': 0.15})
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)

    t = d['time_lagged']

    # Colors matching paper style
    c_bx = '#888888'
    c_by = '#4477AA'
    c_bz = '#CC3311'
    c_density = '#228833'
    c_speed = '#EE7733'
    c_pdyn = '#CC3311'
    c_symh = '#4477AA'
    c_al = '#228833'
    c_au = '#CC3311'

    # (a) IMF components
    ax = axes[0]
    ax.plot(t, d['bx'], color=c_bx, linewidth=0.6, label='$B_X$')
    ax.plot(t, d['by'], color=c_by, linewidth=0.6, label='$B_Y$')
    ax.plot(t, d['bz'], color=c_bz, linewidth=0.8, label='$B_Z$')
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axhline(15, color=c_bz, linewidth=0.3, linestyle='--', alpha=0.5)
    ax.set_ylabel('IMF (nT)')
    ax.set_xlim(t[0], t[-1])
    ax.set_xticklabels([])
    ax.legend(loc='upper left', ncol=3, fontsize=8, framealpha=0.8)
    ax.text(-0.02, 0.5, '(a)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='center', ha='right')
    # Highlight northward IMF period
    ax.axvspan(np.datetime64('2015-04-10T09:17'), np.datetime64('2015-04-10T12:17'),
               color='lightyellow', alpha=0.5)

    # (b) Solar wind density and speed
    ax = axes[1]
    ax.plot(t, d['density'], color=c_density, linewidth=0.6, label='$N_p$ (cm$^{-3}$)')
    ax.set_ylabel('$N_p$ (cm$^{-3}$)', color=c_density)
    ax.set_xlim(t[0], t[-1])
    ax.set_xticklabels([])
    ax.tick_params(axis='y', labelcolor=c_density)
    ax.text(-0.02, 0.5, '(b)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='center', ha='right')

    ax2 = ax.twinx()
    ax2.plot(t, d['speed'], color=c_speed, linewidth=0.6, label='$V_{sw}$ (km/s)')
    ax2.set_ylabel('$V_{sw}$ (km/s)', color=c_speed)
    ax2.tick_params(axis='y', labelcolor=c_speed)

    ax.axvspan(np.datetime64('2015-04-10T09:17'), np.datetime64('2015-04-10T12:17'),
               color='lightyellow', alpha=0.5)

    # (c) PDyn and SYM-H
    ax = axes[2]
    ax.plot(t, d['pressure'], color=c_pdyn, linewidth=0.6, label='$P_{Dyn}$ (nPa)')
    ax.set_ylabel('$P_{Dyn}$ (nPa)', color=c_pdyn)
    ax.set_xlim(t[0], t[-1])
    ax.set_xticklabels([])
    ax.tick_params(axis='y', labelcolor=c_pdyn)
    ax.text(-0.02, 0.5, '(c)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='center', ha='right')

    ax2 = ax.twinx()
    ax2.plot(t, d['sym_h'], color=c_symh, linewidth=0.6, label='SYM-H (nT)')
    ax2.set_ylabel('SYM-H (nT)', color=c_symh)
    ax2.tick_params(axis='y', labelcolor=c_symh)

    ax.axvspan(np.datetime64('2015-04-10T09:17'), np.datetime64('2015-04-10T12:17'),
               color='lightyellow', alpha=0.5)

    # (d) AL and AU indices
    ax = axes[3]
    ax.plot(t, d['al'], color=c_al, linewidth=0.6, label='AL (nT)')
    ax.plot(t, d['au'], color=c_au, linewidth=0.6, label='AU (nT)')
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.set_ylabel('Index (nT)')
    ax.set_xlim(t[0], t[-1])
    ax.set_xticklabels([])
    ax.legend(loc='lower left', ncol=2, fontsize=8, framealpha=0.8)
    ax.text(-0.02, 0.5, '(d)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='center', ha='right')
    ax.axvspan(np.datetime64('2015-04-10T09:17'), np.datetime64('2015-04-10T12:17'),
               color='lightyellow', alpha=0.5)

    # (e) Zoomed IMF for 8:30-12:30 UT on April 10
    ax = axes[4]
    dz = select_date_range(data, '2015-04-10T08:30', '2015-04-10T12:30')
    tz = dz['time_lagged']
    ax.plot(tz, dz['bx'], color=c_bx, linewidth=0.8, label='$B_X$')
    ax.plot(tz, dz['by'], color=c_by, linewidth=0.8, label='$B_Y$')
    ax.plot(tz, dz['bz'], color=c_bz, linewidth=1.0, label='$B_Z$')
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axhline(15, color=c_bz, linewidth=0.3, linestyle='--', alpha=0.5)
    ax.axhline(-15, color=c_bz, linewidth=0.3, linestyle='--', alpha=0.5)
    # Red dotted lines marking focus period
    ax.axvline(np.datetime64('2015-04-10T09:17'), color='red', linewidth=0.8,
               linestyle=':', alpha=0.7)
    ax.axvline(np.datetime64('2015-04-10T12:17'), color='red', linewidth=0.8,
               linestyle=':', alpha=0.7)
    ax.set_ylabel('IMF (nT)')
    ax.set_xlim(tz[0], tz[-1])
    ax.set_xticklabels([])
    ax.legend(loc='upper right', ncol=3, fontsize=8, framealpha=0.8)
    ax.text(-0.02, 0.5, '(e)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='center', ha='right')
    # Gray shading for focus period
    ax.axvspan(np.datetime64('2015-04-10T09:17'), np.datetime64('2015-04-10T12:17'),
               color='gray', alpha=0.15)

    # (f) Zoomed PDyn for 8:30-12:30 UT on April 10
    ax = axes[5]
    ax.plot(tz, dz['pressure'], color=c_pdyn, linewidth=0.8, label='$P_{Dyn}$')
    ax.axvline(np.datetime64('2015-04-10T09:17'), color='red', linewidth=0.8,
               linestyle=':', alpha=0.7)
    ax.axvline(np.datetime64('2015-04-10T12:17'), color='red', linewidth=0.8,
               linestyle=':', alpha=0.7)
    ax.set_ylabel('$P_{Dyn}$ (nPa)')
    ax.set_xlim(tz[0], tz[-1])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.text(-0.02, 0.5, '(f)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='center', ha='right')
    ax.axvspan(np.datetime64('2015-04-10T09:17'), np.datetime64('2015-04-10T12:17'),
               color='gray', alpha=0.15)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    fig.suptitle('Figure 1: Overview of IMF, Solar Wind and Auroral Electrojet\n'
                 '9-11 April 2015 (Wang et al., 2023)',
                 fontsize=12, fontweight='bold', y=0.99)

    plt.savefig(os.path.join(outdir, 'figure1_imf_overview.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: figure1_imf_overview.png")


def verify_key_values(data):
    """Verify key values mentioned in the paper."""
    dz = select_date_range(data, '2015-04-10T09:17', '2015-04-10T12:17')
    print("\n=== Key values verification (9:17-12:17 UT, April 10) ===")
    print(f"  IMF Bz: min={np.nanmin(dz['bz']):.1f}, max={np.nanmax(dz['bz']):.1f} nT")
    print(f"  IMF Bz > 15 nT: {np.sum(dz['bz'] > 15)}/{len(dz['bz'])} minutes ({100*np.sum(dz['bz']>15)/len(dz['bz']):.0f}%)")
    print(f"  |By/Bz| < 0.38: {np.sum(np.abs(dz['by']) / np.abs(dz['bz']) < 0.38)}/{np.sum((np.abs(dz['bz'])>0)&(np.abs(dz['by'])>0))} minutes")
    print(f"  Solar wind speed: mean={np.nanmean(dz['speed']):.0f} km/s")
    print(f"  Density: mean={np.nanmean(dz['density']):.1f} cm^-3")
    print(f"  PDyn: mean={np.nanmean(dz['pressure']):.1f} nPa")


def main():
    print("\n" + "=" * 60)
    print("Figure 1 Reproduction - Wang et al. (2023)")
    print("=" * 60)

    print("\nLoading OMNI data...")
    data = load_omni_data()

    print("Verifying key values...")
    verify_key_values(data)

    print("\nGenerating Figure 1...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    create_figure1(data, OUTPUT_DIR)

    print("Done.")


if __name__ == '__main__':
    main()
