#!/usr/bin/env python
"""
Auroral boundary detection from DMSP SSJ5 binary data.

Dual-mode algorithm:
1. Standard mode: EQ1/PO1/PO2/EQ2 for normal auroral oval (two distinct
   precipitation regions separated by polar cap)
2. HCA mode: poleward edge detection for northward IMF conditions where
   the polar cap is very small or absent (precipitation extends to pole)

Also detects the equatorward boundary of the auroral zone for both modes.

Reference:
  Wang et al. (2023), Communications Earth & Environment
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from read_ssj_binary import read_ssj_file, find_polar_passes, CHANNEL_ENERGIES


def hardy_integrate(counts_19, energies):
    """Integrate top 9 electron channels (>1keV) using Hardy algorithm."""
    c = energies.copy()
    flux = counts_19.copy().astype(np.float64)
    flux[flux < 0] = 0
    flux[~np.isfinite(flux)] = 0
    integrated = (
        (c[0] - c[1]) * flux[0] +
        0.5 * (c[0] - c[2]) * flux[1] +
        0.5 * (c[1] - c[3]) * flux[2] +
        0.5 * (c[2] - c[4]) * flux[3] +
        0.5 * (c[3] - c[5]) * flux[4] +
        0.5 * (c[4] - c[6]) * flux[5] +
        0.5 * (c[5] - c[7]) * flux[6] +
        0.5 * (c[6] - c[8]) * flux[7] +
        (c[7] - c[8]) * flux[8]
    )
    return integrated


def moving_average(x, window_size):
    """Weighted moving average with edge-padding."""
    half_w = int(np.floor(0.5 * window_size))
    padded = np.concatenate([x[:half_w], x, x[-half_w:]])
    shape = (len(padded) - window_size + 1, window_size)
    strides = (padded.strides[0], padded.strides[0])
    windowed = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.nanmean(windowed, axis=-1)


def detect_boundaries(intflux, mlat, sod):
    """
    Detect auroral boundaries with dual-mode approach.

    Parameters
    ----------
    intflux : np.ndarray, shape (N,)
        Smoothed integrated electron flux proxy.
    mlat : np.ndarray, shape (N,)
        Magnetic latitude.
    sod : np.ndarray, shape (N,)
        Seconds of day.

    Returns
    -------
    bnd : dict
        Boundary detection results.
    """
    MIN_LAT = 50.0
    FLUX_FRAC = 0.15

    n = len(intflux)
    hemi = 'N' if np.nanmean(mlat) > 0 else 'S'
    sign = 1 if hemi == 'N' else -1

    result = {
        'mode': None, 'eq1': None, 'po1': None, 'po2': None, 'eq2': None,
        'eq1_mlat': None, 'po1_mlat': None, 'po2_mlat': None, 'eq2_mlat': None,
        'poleward_edge_mlat': None, 'equatorward_edge_mlat': None,
        'fom': None, 'failure_reason': None,
    }

    valid = np.isfinite(intflux) & (intflux > 0)
    if np.sum(valid) < 30:
        result['failure_reason'] = 'Not enough valid data'
        return result

    idx_pass = np.flatnonzero(np.abs(mlat) > MIN_LAT)
    if len(idx_pass) < 30:
        result['failure_reason'] = 'Pass too short'
        return result

    flux_min = max(np.nanpercentile(intflux[valid], 10),
                   FLUX_FRAC * np.nanmax(intflux))
    above = intflux > flux_min

    # --- Find contiguous above-threshold segments ---
    delta = np.diff(above.astype(int))
    starts = np.where(delta == 1)[0] + 1
    ends = np.where(delta == -1)[0] + 1

    if len(starts) == 0 or len(ends) == 0:
        result['failure_reason'] = 'No above-threshold segments'
        return result

    if starts[0] > ends[0]:
        starts = np.concatenate([[0], starts])
    if ends[-1] < starts[-1]:
        ends = np.concatenate([ends, [n]])

    # Filter to polar region
    segments = []
    for s, e in zip(starts, ends):
        mid_mlat = np.nanmean(mlat[s:e])
        if np.abs(mid_mlat) > MIN_LAT and (e - s) >= 10:
            segments.append({'si': s, 'ei': e, 'twidth': sod[e-1] - sod[s],
                             'area': np.nansum(intflux[s:e]),
                             'mlat_mean': mid_mlat})

    if len(segments) < 1:
        result['failure_reason'] = 'No valid segments in polar region'
        return result

    # --- Detect poleward edge (highest-latitude precipitation) ---
    max_lat_seg = max(segments, key=lambda s: sign * s['mlat_mean'])
    result['poleward_edge_mlat'] = sign * np.max(np.abs(mlat[max_lat_seg['si']:max_lat_seg['ei']]))

    # --- Detect equatorward edge (lowest-latitude precipitation) ---
    min_lat_seg = min(segments, key=lambda s: sign * s['mlat_mean'])
    result['equatorward_edge_mlat'] = sign * np.min(np.abs(mlat[min_lat_seg['si']:min_lat_seg['ei']]))

    # --- Try standard two-boundary mode ---
    # Merge segments separated by <30 seconds
    MIN_GAP = 30
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        if seg['si'] - merged[-1]['ei'] < MIN_GAP:
            merged[-1]['ei'] = seg['ei']
            merged[-1]['area'] += seg['area']
            merged[-1]['twidth'] = sod[merged[-1]['ei']-1] - sod[merged[-1]['si']]
            merged[-1]['mlat_mean'] = np.nanmean(mlat[merged[-1]['si']:merged[-1]['ei']])
        else:
            merged.append(seg.copy())

    segments = merged

    # For standard mode, need at least 2 segments with pole between them
    idx_pole = np.argmax(np.abs(mlat))

    if len(segments) >= 2:
        # Find segments on each side of the pole
        before_pole = [(i, s) for i, s in enumerate(segments) if s['ei'] < idx_pole]
        after_pole = [(i, s) for i, s in enumerate(segments) if s['si'] > idx_pole]

        if before_pole and after_pole:
            # Standard mode: two-boundary detection
            s1 = max(before_pole, key=lambda x: x[1]['area'])[1]
            s2 = max(after_pole, key=lambda x: x[1]['area'])[1]

            max_area = max(s['area'] for s in segments)
            gap_time = abs(sod[s2['si']] - sod[s1['ei']])

            # Check if there's a significant gap (polar cap)
            if gap_time > 60:  # at least 60s gap
                result['mode'] = 'standard'
                result['eq1'] = min(s1['si'], n-1)
                result['po1'] = min(s1['ei'], n-1)
                result['po2'] = min(s2['si'], n-1)
                result['eq2'] = min(s2['ei'], n-1)
                result['eq1_mlat'] = mlat[min(s1['si'], n-1)]
                result['po1_mlat'] = mlat[min(s1['ei'], n-1)]
                result['po2_mlat'] = mlat[min(s2['si'], n-1)]
                result['eq2_mlat'] = mlat[min(s2['ei'], n-1)]
                result['fom'] = (s1['area'] + s2['area']) / max_area + gap_time / 1200.
                return result

    # --- HCA mode: precipitation extends to pole ---
    # Check if there's precipitation near the pole
    pole_region = max(0, idx_pole - 60)
    pole_end = min(n, idx_pole + 60)
    pole_flux = intflux[pole_region:pole_end]
    pole_above = np.sum(pole_flux > flux_min) / len(pole_flux)

    if pole_above > 0.3:
        # Significant precipitation near pole → HCA mode
        result['mode'] = 'HCA'

        # Find the highest-latitude edge where flux drops below threshold
        # on the equatorward side (from pole toward equator on both sides)
        # Ascending side (lower latitudes to pole)
        asc_segs = [s for s in segments if s['mlat_mean'] * sign < mlat[idx_pole] * sign]
        if asc_segs:
            s_asc = max(asc_segs, key=lambda s: sign * s['mlat_mean'])
            ei = min(s_asc['ei'], n - 1)
            result['po1'] = ei
            result['po1_mlat'] = mlat[ei]

        # Descending side (pole to lower latitudes)
        desc_segs = [s for s in segments if s['mlat_mean'] * sign <= mlat[idx_pole] * sign]
        if desc_segs:
            s_desc = max(desc_segs, key=lambda s: sign * s['mlat_mean'])
            ei = min(s_desc['ei'], n - 1)
            result['po2'] = ei
            result['po2_mlat'] = mlat[ei]

        # Equatorward edges
        if asc_segs:
            s_eq1 = min(asc_segs, key=lambda s: sign * s['mlat_mean'])
            si = min(s_eq1['si'], n - 1)
            result['eq1'] = si
            result['eq1_mlat'] = mlat[si]

        if desc_segs:
            s_eq2 = min(desc_segs, key=lambda s: sign * s['mlat_mean'])
            si = min(s_eq2['si'], n - 1)
            result['eq2'] = si
            result['eq2_mlat'] = mlat[si]

        return result

    result['mode'] = 'standard'
    result['failure_reason'] = 'Ambiguous precipitation structure'
    return result


def plot_pass_with_boundaries(data, pass_start, pass_end, pass_num,
                               sat_key, boundaries, outdir):
    """Plot SSJ5 spectrogram with auroral boundary overlays."""
    s, e = pass_start, pass_end
    datetimes = data['datetime'][s:e]
    mlat = data['mlat'][s:e]
    mlt = data['mlt'][s:e]
    energies = data['channel_energies']

    eflux = data['eflux_19_rescaled'][:, s:e].astype(np.float64)
    iflux = data['iflux_19_rescaled'][:, s:e].astype(np.float64)
    eflux[eflux <= 0] = np.nan
    iflux[iflux <= 0] = np.nan

    # Bin edges
    bin_edges = np.zeros(20)
    bin_edges[0] = energies[0] * (energies[0] / energies[1]) ** 0.5
    bin_edges[-1] = energies[-1] * (energies[-1] / energies[-2]) ** 0.5
    for i in range(1, 19):
        bin_edges[i] = np.sqrt(energies[i-1] * energies[i])

    mpl_times = mdates.date2num(datetimes)
    hemi = 'N' if mlat[len(mlat)//2] > 0 else 'S'
    t0 = datetimes[0].strftime('%H:%M:%S')
    t1 = datetimes[-1].strftime('%H:%M:%S')

    fig = plt.figure(figsize=(14, 11), dpi=150)
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 0.4, 0.5], hspace=0.35)

    # Electron spectrogram
    ax1 = fig.add_subplot(gs[0])
    C_e = eflux[:, :-1] if eflux.shape[1] > 1 else eflux
    masked_e = np.ma.masked_where(~np.isfinite(C_e), C_e)
    if np.isfinite(C_e).any():
        vmin = max(1.0, np.nanpercentile(C_e[np.isfinite(C_e)], 1))
        vmax = np.nanpercentile(C_e[np.isfinite(C_e)], 99)
    else:
        vmin, vmax = 1.0, 1000.0
    cmap = plt.colormaps['Spectral_r'].copy()
    cmap.set_bad('white', 0.1)
    im1 = ax1.pcolormesh(mpl_times, bin_edges, masked_e, cmap=cmap,
                          shading='flat', norm=LogNorm(vmin=vmin, vmax=vmax))
    ax1.set_yscale('log')
    ax1.set_ylim([30, 30000])
    ax1.set_ylabel('Electron\nEnergy [eV]')
    ax1.set_title(f'DMSP {sat_key} {hemi} Pass #{pass_num+1}  '
                  f'2015-04-10 {t0}-{t1} UTC',
                  fontsize=11, fontweight='bold')
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    cb1 = plt.colorbar(im1, ax=ax1, pad=0.01)
    cb1.set_label('Counts', fontsize=8)

    # Ion spectrogram
    ax2 = fig.add_subplot(gs[1])
    C_i = iflux[:, :-1] if iflux.shape[1] > 1 else iflux
    masked_i = np.ma.masked_where(~np.isfinite(C_i), C_i)
    if np.isfinite(C_i).any():
        vmin_i = max(1.0, np.nanpercentile(C_i[np.isfinite(C_i)], 1))
        vmax_i = np.nanpercentile(C_i[np.isfinite(C_i)], 99)
    else:
        vmin_i, vmax_i = 1.0, 1000.0
    im2 = ax2.pcolormesh(mpl_times, bin_edges, masked_i, cmap=cmap,
                          shading='flat', norm=LogNorm(vmin=vmin_i, vmax=vmax_i))
    ax2.set_yscale('log')
    ax2.set_ylim([30, 30000])
    ax2.set_ylabel('Ion\nEnergy [eV]')
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    cb2 = plt.colorbar(im2, ax=ax2, pad=0.01)
    cb2.set_label('Counts', fontsize=8)

    # Integrated flux profile
    ax3 = fig.add_subplot(gs[2])
    intflux = hardy_integrate(data['eflux_19_rescaled'][:, s:e], energies)
    intflux_smooth = moving_average(intflux, 15)
    ax3.semilogy(mpl_times, intflux_smooth, 'k-', lw=0.8)
    ax3.set_ylabel('Int. Flux\n(>1keV)')
    ax3.xaxis_date()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.grid(True, alpha=0.3)

    # MLat profile
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(mpl_times, mlat, 'k-', lw=0.8)
    ax4.set_ylabel('MLat')
    ax4.set_xlabel('UTC Time')
    ax4.xaxis_date()
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.grid(True, alpha=0.3)

    # Draw boundaries
    if boundaries:
        mode = boundaries.get('mode', 'unknown')
        bnd_info = []

        for key, color, label, ls in [('eq1', 'blue', 'EQ1', '--'),
                                       ('po1', 'red', 'PO1', '-'),
                                       ('po2', 'magenta', 'PO2', '-'),
                                       ('eq2', 'deepskyblue', 'EQ2', '--')]:
            idx = boundaries.get(key)
            if idx is not None and s <= idx < e:
                local_t = mpl_times[idx - s]
                local_lat = mlat[idx - s]
                for ax in [ax1, ax2, ax3]:
                    ax.axvline(local_t, color=color, lw=1.5, ls=ls, alpha=0.8)
                ax4.axvline(local_t, color=color, lw=1.5, ls=ls, alpha=0.8,
                            label=f'{label} ({local_lat:.1f})')
                bnd_info.append(f'{label}={local_lat:.1f}')

        if bnd_info:
            ax4.legend(loc='upper right', fontsize=8, ncol=2)

        mode_str = f'Mode: {mode}'
        if boundaries.get('failure_reason'):
            mode_str += f' ({boundaries["failure_reason"]})'
        if bnd_info:
            mode_str += ' | ' + ' '.join(bnd_info)
        if boundaries.get('poleward_edge_mlat') is not None:
            mode_str += f' | Poleward edge: {boundaries["poleward_edge_mlat"]:.1f}'

        fig.text(0.5, 0.01, mode_str, ha='center', fontsize=9,
                 color='darkblue' if mode == 'HCA' else 'black',
                 style='italic')

    plt.savefig(os.path.join(outdir,
                f'boundary_{sat_key}_pass{pass_num+1:02d}_{hemi}_{t0.replace(":","")}-{t1.replace(":","")}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_boundary_evolution(datasets, outdir):
    """
    Plot auroral boundary latitude evolution vs time.
    Key figure for showing polar cap shrinkage under northward IMF.
    """
    fig, (ax_n, ax_s) = plt.subplots(2, 1, figsize=(14, 8), dpi=150,
                                       sharex=True)

    for sat_key, data in sorted(datasets.items()):
        passes = find_polar_passes(data['mlat'])
        for pi, (s, e) in enumerate(passes):
            mlat = data['mlat'][s:e]
            hemi = 'N' if mlat[len(mlat)//2] > 0 else 'S'
            sign = 1 if hemi == 'N' else -1
            ax = ax_n if hemi == 'N' else ax_s

            dt_mid = data['datetime'][(s+e)//2]
            mlt_mid = data['mlt'][(s+e)//2]

            eflux = data['eflux_19_rescaled'][:, s:e]
            energies = data['channel_energies']
            intflux = hardy_integrate(eflux, energies)
            intflux_smooth = moving_average(intflux, 15)
            sod = data['sod'][s:e]

            bnd = detect_boundaries(intflux_smooth, mlat, sod)

            color = 'C0' if sat_key == 'F17' else 'C1'
            marker_eq = 'o'
            marker_po = '^'

            if bnd.get('mode') == 'standard' and bnd.get('failure_reason') is None:
                for key, marker in [('eq1', 'o'), ('po1', '^'), ('po2', 'v'), ('eq2', 's')]:
                    idx = bnd[key]
                    if idx is not None:
                        lat = sign * np.abs(mlat[idx])
                        ax.plot(dt_mid, lat, marker, color=color, markersize=4,
                                markeredgecolor='black', markeredgewidth=0.3)
                        # Label first few
                        if pi < 3:
                            ax.annotate(f'{sat_key}', (dt_mid, lat), fontsize=6,
                                        xytext=(3, 3), textcoords='offset points')

            elif bnd.get('mode') == 'HCA':
                # HCA: mark poleward edge
                pw = bnd.get('poleward_edge_mlat')
                ew = bnd.get('equatorward_edge_mlat')
                if pw is not None:
                    ax.plot(dt_mid, pw, '*', color='red', markersize=8,
                            markeredgecolor='black', markeredgewidth=0.3)
                if ew is not None:
                    ax.plot(dt_mid, ew, 'o', color=color, markersize=4,
                            markeredgecolor='black', markeredgewidth=0.3)

    for ax, title in [(ax_n, 'Northern Hemisphere'), (ax_s, 'Southern Hemisphere')]:
        ax.set_ylabel('|MLat| (degrees)')
        ax.set_title(f'Auroral Boundary Evolution - {title}\n'
                     f'2015-04-10 (Wang et al. 2023 event)',
                     fontsize=11)
        ax.set_ylim([45, 92])
        ax.grid(True, alpha=0.3)
        ax.axhline(80, color='grey', ls=':', alpha=0.5, label='80 deg')
        ax.axhline(85, color='grey', ls=':', alpha=0.5, label='85 deg')

    ax_s.set_xlabel('UTC Time (2015-04-10)')
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C0',
               markeredgecolor='black', markersize=6, label='F17 EQ'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='C0',
               markeredgecolor='black', markersize=6, label='F17 PO'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C1',
               markeredgecolor='black', markersize=6, label='F18 EQ'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='C1',
               markeredgecolor='black', markersize=6, label='F18 PO'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markeredgecolor='black', markersize=10, label='HCA poleward edge'),
    ]
    ax_n.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=3)

    ax_n.xaxis_date()
    ax_s.xaxis_date()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'boundary_evolution_20150410.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: boundary_evolution_20150410.png")


def main():
    print("\n" + "=" * 60)
    print("Auroral Boundary Detection - Wang et al. (2023)")
    print("Event: 2015-04-10 (Northward IMF / HCA)")
    print("=" * 60)

    outdir = os.path.join(PROJECT_DIR, 'output')
    os.makedirs(outdir, exist_ok=True)

    data_dir = os.path.join(PROJECT_DIR, 'data', 'dmsp')
    datasets = {}

    for fname in ['2015apr10.f17', '2015apr10.f18']:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            data = read_ssj_file(fpath)
            sat_key = f'F{data["satellite"]:02d}'
            datasets[sat_key] = data
            print(f"\n  {sat_key}: {data['n_records']} records")

    if not datasets:
        print("ERROR: No data files found.")
        return

    print(f"\n{'='*80}")
    print(f"{'Sat':>5s} {'Pass':>5s} {'Hemi':>5s} {'UT':>14s} "
          f"{'MLT':>8s} {'Mode':>8s} {'EQ1':>6s} {'PO1':>6s} {'PO2':>6s} {'EQ2':>6s} "
          f"{'PwEdge':>6s} {'FOM':>5s}")
    print(f"{'---':>5s} {'---':>5s} {'---':>5s} {'---':>14s} "
          f"{'---':>8s} {'---':>8s} {'---':>6s} {'---':>6s} {'---':>6s} {'---':>6s} "
          f"{'---':>6s} {'---':>5s}")

    for sat_key, data in sorted(datasets.items()):
        passes = find_polar_passes(data['mlat'])
        for pi, (s, e) in enumerate(passes):
            mlat = data['mlat'][s:e]
            sod = data['sod'][s:e]
            energies = data['channel_energies']
            eflux = data['eflux_19_rescaled'][:, s:e]

            intflux = hardy_integrate(eflux, energies)
            intflux_smooth = moving_average(intflux, 15)
            bnd = detect_boundaries(intflux_smooth, mlat, sod)

            hemi = 'N' if mlat[len(mlat)//2] > 0 else 'S'
            dt = data['datetime'][s].strftime('%H:%M') + '-' + data['datetime'][e-1].strftime('%H:%M')
            mlt = f"{data['mlt'][s]:.1f}-{data['mlt'][e-1]:.1f}"

            mode = bnd.get('mode', '?')
            eq1 = f"{bnd['eq1_mlat']:.1f}" if bnd.get('eq1_mlat') else '-'
            po1 = f"{bnd['po1_mlat']:.1f}" if bnd.get('po1_mlat') else '-'
            po2 = f"{bnd['po2_mlat']:.1f}" if bnd.get('po2_mlat') else '-'
            eq2 = f"{bnd['eq2_mlat']:.1f}" if bnd.get('eq2_mlat') else '-'
            pw = f"{bnd.get('poleward_edge_mlat', 0):.1f}" if bnd.get('poleward_edge_mlat') else '-'
            fom = f"{bnd['fom']:.1f}" if bnd.get('fom') else '-'

            print(f"{sat_key:>5s} {pi+1:5d} {hemi:>5s} {dt:>14s} "
                  f"{mlt:>8s} {mode:>8s} {eq1:>6s} {po1:>6s} {po2:>6s} {eq2:>6s} "
                  f"{pw:>6s} {fom:>5s}")

            plot_pass_with_boundaries(data, s, e, pi, sat_key, bnd, outdir)

    # Boundary evolution plot
    print(f"\n  Generating boundary evolution plot...")
    plot_boundary_evolution(datasets, outdir)

    print(f"\nAll plots saved to: {outdir}/")
    print("Done.")


if __name__ == '__main__':
    main()
