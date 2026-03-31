#!/usr/bin/env python
"""
DMSP SSJ5 particle data analysis for the Wang et al. (2023) HCA event.

Reads JHU/APL binary SSJ5 data files and produces spectrograms
of precipitating electron and ion fluxes for polar passes.

SSJ5 instrument:
- Measures precipitating electrons and ions from 30 eV to 30 keV
- 19 logarithmically-spaced energy channels
- Time resolution: 1 second per record
- Used to identify auroral boundaries and particle precipitation features

Data source: JHU/APL SSJ binary files (128 bytes/record, little-endian)
  http://sd-www.jhuapl.edu/Aurora/data/data_step1.cgi

Reference:
  Wang et al. (2023), "Horse-Collar Auroral Morphology Driven by
  Flow Shear at the Poleward Edge of the Ring Current"
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from datetime import datetime

# Add scripts directory to path for importing read_ssj_binary
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from read_ssj_binary import read_ssj_file, find_polar_passes, CHANNEL_ENERGIES


# --- Bin edges for spectrogram plotting (20 edges for 19 channels) ---
# Geometric mean of adjacent channel energies gives bin boundaries.
BIN_EDGES = np.zeros(20)
BIN_EDGES[0] = CHANNEL_ENERGIES[0] * (CHANNEL_ENERGIES[0] / CHANNEL_ENERGIES[1]) ** 0.5
BIN_EDGES[-1] = CHANNEL_ENERGIES[-1] * (CHANNEL_ENERGIES[-1] / CHANNEL_ENERGIES[-2]) ** 0.5
for i in range(1, 19):
    BIN_EDGES[i] = np.sqrt(CHANNEL_ENERGIES[i-1] * CHANNEL_ENERGIES[i])


def _spectrogram(ax, times, flux_19, energies, bin_edges, cmap_name='Spectral_r'):
    """
    Plot a particle flux spectrogram on the given axes.

    Parameters
    ----------
    ax : matplotlib Axes
    times : 1D array of datetime objects, length N
    flux_19 : 2D array, shape (19, N), flux counts
    energies : 1D array, shape (19,), channel center energies (eV)
    bin_edges : 1D array, shape (20,), bin boundary energies (eV)
    cmap_name : str

    Returns
    -------
    im : mappable
    """
    mpl_times = mdates.date2num(times)

    # Build 1D arrays: N time points, 20 energy edges
    # pcolormesh with shading='flat': C must be (N-1, 19)
    # So we trim one time point from C.
    # Alternatively, use shading='auto' with matching dimensions.
    # Best approach: use pcolormesh with 1D x (N points), 1D y (20 edges),
    # and C shape (N, 19) -- this works with shading='auto' (treated as 'nearest').
    # For shading='flat', we need C (N-1, 19), which loses one time point.

    # Use shading='flat' properly: trim last time point from C, keep all 20 edges
    C = flux_19[:, :-1] if flux_19.shape[1] > 1 else flux_19

    cmap = plt.colormaps[cmap_name].copy()
    cmap.set_bad('white', 0.1)
    cmap.set_over('black')
    cmap.set_under('grey')

    valid = np.isfinite(C)
    if valid.any():
        z_min = max(1.0, np.nanpercentile(C[valid], 1))
        z_max = np.nanpercentile(C[valid], 99)
    else:
        z_min, z_max = 1.0, 1000.0

    masked = np.ma.masked_where(~np.isfinite(C), C)

    im = ax.pcolormesh(mpl_times, bin_edges, masked,
                       cmap=cmap, shading='flat',
                       norm=mcolors.LogNorm(vmin=z_min, vmax=z_max))

    ax.set_yscale('log')
    ax.set_ylim([energies.min(), energies.max()])
    ax.set_ylabel('Channel Energy [eV]')

    return im


def load_data():
    """
    Load F17 and F18 SSJ5 binary data files for 2015-04-10.

    Returns
    -------
    datasets : dict
        Keys 'F17', 'F18' mapping to the dict returned by read_ssj_file.
    """
    data_dir = os.path.join(PROJECT_DIR, 'data', 'dmsp')
    datasets = {}

    for fname in ['2015apr10.f17', '2015apr10.f18']:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            data = read_ssj_file(fpath)
            sat_key = f'F{data["satellite"]:02d}'
            datasets[sat_key] = data
            print(f"  {sat_key}: {data['n_records']} records, "
                  f"DOY={data['doy'][0]}, "
                  f"MLat=[{data['mlat'].min():.1f}, {data['mlat'].max():.1f}], "
                  f"Alt={data['nalt'].mean()/1000:.1f} km")
        else:
            print(f"  WARNING: {fpath} not found, skipping")

    return datasets


def plot_polar_pass_spectrogram(data, pass_idx, pass_slice, sat_key,
                                outdir, species='electron'):
    """
    Plot a spectrogram of particle flux for a single polar pass.

    Parameters
    ----------
    data : dict
        Output from read_ssj_file.
    pass_idx : int
        Pass number (0-based) for labeling.
    pass_slice : tuple of (start, end)
        Record index range for the polar pass.
    sat_key : str
        Satellite identifier, e.g. 'F18'.
    outdir : str
        Directory to save output plots.
    species : str
        'electron' or 'ion'.
    """
    s, e = pass_slice
    n = e - s

    # Extract pass data
    datetimes = data['datetime'][s:e]
    mlat = data['mlat'][s:e]
    mlt = data['mlt'][s:e]
    energies = data['channel_energies']

    if species == 'electron':
        flux_19 = data['eflux_19_rescaled'][:, s:e]
        label = 'Electron'
    else:
        flux_19 = data['iflux_19_rescaled'][:, s:e]
        label = 'Ion'

    # Replace zeros and negative values with NaN for log-scale plotting
    flux_plot = flux_19.copy().astype(np.float64)
    flux_plot[flux_plot <= 0] = np.nan

    # Build mpl_times for tick formatting
    mpl_times = mdates.date2num(datetimes)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)

    im = _spectrogram(ax, datetimes, flux_plot, energies, BIN_EDGES)

    plt.colorbar(im, ax=ax, label='Counts', pad=0.02)

    ax.set_yscale('log')
    ax.set_ylim([energies.min(), energies.max()])
    ax.set_ylabel('Channel Energy [eV]')

    # X-axis: time with latitude and MLT labels
    ax.xaxis_date()
    plotwidth_h = (datetimes[-1] - datetimes[0]).total_seconds() / 3600.
    if plotwidth_h <= 1:
        majloc = mdates.MinuteLocator(interval=5)
    elif plotwidth_h <= 3:
        majloc = mdates.MinuteLocator(interval=10)
    else:
        majloc = mdates.MinuteLocator(interval=15)

    ax.xaxis.set_major_locator(majloc)

    # Custom tick labels: HH:MM / MLat / MLT
    xticks = ax.get_xticks().tolist()
    xlabels = []
    for tick in xticks:
        ind = np.nonzero(mpl_times == tick)[0]
        if len(ind) > 0:
            idx = ind[0]
            tickstr = datetimes[idx].strftime('%H:%M')
            tickstr += f'\n{mlat[idx]:.1f}'
            tickstr += f'\n{mlt[idx]:.1f}h'
            xlabels.append(tickstr)
        else:
            dtime = mdates.num2date(tick)
            xlabels.append(dtime.strftime('%H:%M'))

    ax.xaxis.set_major_locator(mdates.ticker.FixedLocator(xticks))
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_xlabel('UT / MLat(deg) / MLT(h)')

    # Hemisphere and direction
    hemi = 'N' if mlat[len(mlat)//2] > 0 else 'S'
    direction = 'poleward' if mlat[-1] > mlat[0] else 'equatorward'

    t0 = datetimes[0].strftime('%H:%M:%S')
    t1 = datetimes[-1].strftime('%H:%M:%S')
    ax.set_title(
        f'DMSP {sat_key} {label} Spectrogram - {hemi} hemisphere '
        f'({direction})\n'
        f'2015-04-10 {t0} - {t1} UTC  |  Pass #{pass_idx+1}',
        fontsize=10)

    plt.tight_layout()

    fname_out = os.path.join(
        outdir,
        f'ssj_{sat_key}_{species}_pass{pass_idx+1:02d}_{hemi}_{t0.replace(":","")}-{t1.replace(":","")}.png')
    fig.savefig(fname_out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {fname_out}")


def plot_combined_pass(data, pass_idx, pass_slice, sat_key, outdir):
    """
    Plot combined electron + ion spectrograms and energy flux profile
    for a single polar pass.

    Parameters
    ----------
    data : dict
        Output from read_ssj_file.
    pass_idx : int
        Pass number for labeling.
    pass_slice : tuple of (start, end)
        Record index range.
    sat_key : str
        Satellite identifier.
    outdir : str
        Output directory.
    """
    s, e = pass_slice
    datetimes = data['datetime'][s:e]
    mlat = data['mlat'][s:e]
    mlt = data['mlt'][s:e]
    energies = data['channel_energies']

    eflux = data['eflux_19_rescaled'][:, s:e].astype(np.float64)
    iflux = data['iflux_19_rescaled'][:, s:e].astype(np.float64)

    # Total energy proxy: sum across channels (weighted by energy)
    # This is a crude proxy; proper calibration requires geometric factors.
    e_total = np.nansum(eflux * energies[:, np.newaxis], axis=0)
    i_total = np.nansum(iflux * energies[:, np.newaxis], axis=0)

    # Replace zeros with NaN
    eflux_plot = eflux.copy()
    iflux_plot = iflux.copy()
    eflux_plot[eflux_plot <= 0] = np.nan
    iflux_plot[iflux_plot <= 0] = np.nan

    mpl_times = mdates.date2num(datetimes)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=150,
                             gridspec_kw={'height_ratios': [1, 1, 0.6]})

    # --- Electron and Ion spectrograms ---
    for ax_idx, (flux, label) in enumerate([(eflux_plot, 'Electron'),
                                             (iflux_plot, 'Ion')]):
        ax = axes[ax_idx]
        im = _spectrogram(ax, datetimes, flux, energies, BIN_EDGES)
        plt.colorbar(im, ax=ax, label='Counts', pad=0.01)
        ax.set_ylabel(f'{label}\nEnergy [eV]')

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # --- Energy flux proxy (bottom panel) ---
    ax3 = axes[2]
    ax3.semilogy(mpl_times, e_total, 'r-', linewidth=0.5, alpha=0.8, label='Electron')
    ax3.semilogy(mpl_times, i_total, 'b-', linewidth=0.5, alpha=0.8, label='Ion')
    ax3.set_ylabel('Total Flux\nProxy')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.xaxis_date()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.set_xlabel('UTC Time (2015-04-10)')
    ax3.grid(True, alpha=0.3)

    # Latitude on twin axis
    ax3t = ax3.twinx()
    ax3t.plot(mpl_times, mlat, 'k--', linewidth=0.8, alpha=0.5)
    ax3t.set_ylabel('MLat', fontsize=8)
    ax3t.tick_params(axis='y', labelsize=7)

    hemi = 'N' if mlat[len(mlat)//2] > 0 else 'S'
    t0 = datetimes[0].strftime('%H:%M:%S')
    t1 = datetimes[-1].strftime('%H:%M:%S')

    fig.suptitle(
        f'DMSP {sat_key} Particle Spectrograms - {hemi} hemisphere\n'
        f'2015-04-10 {t0} - {t1} UTC  |  Pass #{pass_idx+1}',
        fontsize=11, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname_out = os.path.join(
        outdir,
        f'ssj_{sat_key}_combined_pass{pass_idx+1:02d}_{hemi}_{t0.replace(":","")}-{t1.replace(":","")}.png')
    fig.savefig(fname_out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {fname_out}")


def plot_all_passes_overview(datasets, outdir):
    """
    Create a summary figure showing all polar passes for each satellite
    as a stack of spectrograms.

    Parameters
    ----------
    datasets : dict
        Keys 'F17', 'F18' mapping to read_ssj_file output.
    outdir : str
        Output directory.
    """
    for sat_key, data in datasets.items():
        passes = find_polar_passes(data['mlat'])
        n_passes = len(passes)

        if n_passes == 0:
            continue

        print(f"\n  Creating overview for {sat_key} ({n_passes} passes)...")

        # Stack electron spectrograms vertically (one per pass)
        n_cols = min(n_passes, 6)
        n_rows = (n_passes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows),
                                 dpi=150, squeeze=False)
        axes_flat = axes.flatten()

        for pi, (s, e) in enumerate(passes):
            ax = axes_flat[pi]

            eflux = data['eflux_19_rescaled'][:, s:e].astype(np.float64)
            eflux[eflux <= 0] = np.nan

            mlat_pass = data['mlat'][s:e]
            mlt_pass = data['mlt'][s:e]
            dt_pass = data['datetime'][s:e]

            mpl_times = mdates.date2num(dt_pass)
            lat_grid = np.linspace(mlat_pass[0], mlat_pass[-1], e - s)

            # Simple latitude-based x-axis instead of time for compact view
            masked = np.ma.masked_where(~np.isfinite(eflux), eflux)
            valid = np.isfinite(eflux)
            if valid.any():
                z_min = max(1.0, np.nanpercentile(eflux[valid], 1))
                z_max = np.nanpercentile(eflux[valid], 98)
            else:
                z_min, z_max = 1.0, 1000.0

            cmap = plt.colormaps['Spectral_r'].copy()
            cmap.set_bad('white', 0.1)

            # Use 1D lat_grid (N points) and BIN_EDGES (20 edges).
            # pcolormesh shading='flat': C must be (N-1, 19).
            # Trim last time point from data to get N-1 columns.
            C = masked[:, :-1] if masked.shape[1] > 1 else masked
            ax.pcolormesh(lat_grid, BIN_EDGES, C,
                          cmap=cmap, shading='flat',
                          norm=mcolors.LogNorm(vmin=z_min, vmax=z_max))
            ax.set_yscale('log')
            ax.set_ylim([30, 30000])
            ax.set_xlim([mlat_pass.min(), mlat_pass.max()])

            hemi = 'N' if mlat_pass[len(mlat_pass)//2] > 0 else 'S'
            t0 = dt_pass[0].strftime('%H:%M')
            mlt_mid = mlt_pass[len(mlt_pass)//2]
            ax.set_title(f'#{pi+1} {hemi} {t0} MLT={mlt_mid:.0f}h',
                         fontsize=7)
            ax.tick_params(labelsize=5)

            if pi == 0:
                ax.set_ylabel('eV', fontsize=7)

        # Hide unused axes
        for pi in range(n_passes, len(axes_flat)):
            axes_flat[pi].set_visible(False)

        fig.suptitle(
            f'DMSP {sat_key} Electron Spectrograms - All Polar Passes\n'
            f'2015-04-10  (x-axis: magnetic latitude)',
            fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname_out = os.path.join(outdir, f'ssj_{sat_key}_all_passes_overview.png')
        fig.savefig(fname_out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {fname_out}")


def print_pass_summary(datasets):
    """
    Print a summary table of all polar passes.

    Parameters
    ----------
    datasets : dict
        Keys 'F17', 'F18' mapping to read_ssj_file output.
    """
    print(f"\n{'='*90}")
    print(f"Polar Pass Summary (|MLat| > 50 deg)")
    print(f"{'='*90}")
    print(f"{'Sat':>5s} {'Pass':>5s} {'Hemi':>5s} {'Start UT':>12s} {'End UT':>12s} "
          f"{'MLat range':>15s} {'MLT range':>12s} {'Duration':>9s}")
    print(f"{'---':>5s} {'---':>5s} {'---':>5s} {'---':>12s} {'---':>12s} "
          f"{'---':>15s} {'---':>12s} {'---':>9s}")

    for sat_key, data in sorted(datasets.items()):
        passes = find_polar_passes(data['mlat'])
        for idx, (s, e) in enumerate(passes):
            dt0 = data['datetime'][s]
            dt1 = data['datetime'][e - 1]
            mlat_s, mlat_e = data['mlat'][s], data['mlat'][e - 1]
            mlt_s, mlt_e = data['mlt'][s], data['mlt'][e - 1]
            hemi = 'NH' if (mlat_s + mlat_e) / 2 > 0 else 'SH'
            dur = (e - s) / 60.0

            print(f"{sat_key:>5s} {idx+1:5d} {hemi:>5s} "
                  f"{dt0.strftime('%H:%M:%S'):>12s} {dt1.strftime('%H:%M:%S'):>12s} "
                  f"[{mlat_s:6.1f},{mlat_e:6.1f}] "
                  f"[{mlt_s:5.1f},{mlt_e:5.1f}]h "
                  f"{dur:7.1f} min")

    print(f"{'='*90}\n")


def main():
    """Main analysis routine."""
    print("\n" + "=" * 60)
    print("DMSP SSJ5 Particle Data Analysis")
    print("Event: 2015-04-10 (Wang et al., 2023)")
    print("=" * 60)

    outdir = os.path.join(PROJECT_DIR, 'output')
    os.makedirs(outdir, exist_ok=True)

    # Load data
    print("\nLoading SSJ5 binary data:")
    datasets = load_data()

    if not datasets:
        print("ERROR: No data files found.")
        return

    # Print pass summary
    print_pass_summary(datasets)

    # Generate plots for each satellite
    for sat_key, data in datasets.items():
        passes = find_polar_passes(data['mlat'])
        print(f"\nGenerating spectrograms for {sat_key} "
              f"({len(passes)} polar passes)...")

        for idx, (s, e) in enumerate(passes):
            # Combined electron + ion spectrogram (most useful)
            plot_combined_pass(data, idx, (s, e), sat_key, outdir)

    # Overview figure (all passes stacked)
    print("\nGenerating overview figures:")
    plot_all_passes_overview(datasets, outdir)

    print(f"\nAll plots saved to: {outdir}/")
    print("Done.")


if __name__ == '__main__':
    main()
