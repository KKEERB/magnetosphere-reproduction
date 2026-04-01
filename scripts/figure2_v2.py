#!/usr/bin/env python
"""
Figure 2 Reproduction: Auroral and plasma observations (SSJ5 spectrograms)
Wang et al. (2023) - Communications Earth & Environment

Layout matching paper:
  Row 1: [SSUSI placeholder] x 2 (NH and SH for ~11:00 and ~12:00 UT)
  Row 2: Electron spectrograms (NH, SH) x 2 time groups
  Row 3: Ion spectrograms (NH, SH) x 2 time groups

Panel mapping:
  (a,b) SSUSI LBHS polar projection (placeholder - no SSUSI data available)
  (c,d) Electron energy flux ~11:00 UT (NH, SH)
  (e,f) Ion energy flux ~11:00 UT (NH, SH)
  (g,h) SSUSI ~12:00 UT (placeholder)
  (i,j) Electron energy flux ~12:00 UT (NH, SH)
  (k,l) Ion energy flux ~12:00 UT (NH, SH)

Pass selection (closest to paper):
  ~11:00 NH: F18 #14 (11:09-11:33 UT)
  ~11:00 SH: F17 #13 (10:25-10:49 UT)
  ~12:00 NH: F17 #16 (12:55-13:21 UT)
  ~12:00 SH: F17 #15 (12:06-12:30 UT)
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
sys.path.insert(0, SCRIPT_DIR)

from read_ssj_binary import read_ssj_file, find_polar_passes, CHANNEL_ENERGIES


def plot_spectrogram(ax, times, flux_19, energies, bin_edges,
                     species_label='', cmap_name='hot'):
    """
    Plot a particle flux spectrogram matching paper style.
    X-axis: time, Y-axis: energy (log), Color: flux (log).
    """
    mpl_times = mdates.date2num(times)

    # Trim for pcolormesh shading='flat'
    C = flux_19[:, :-1] if flux_19.shape[1] > 1 else flux_19

    # Apply light smoothing along time axis for cleaner display
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    for i in range(C.shape[0]):
        valid_mask = np.isfinite(C[i])
        if valid_mask.any():
            C[i, valid_mask] = np.convolve(C[i, valid_mask], kernel, mode='same')

    valid = np.isfinite(C)
    if valid.any():
        z_min = max(0.5, np.nanpercentile(C[valid], 0.5))
        z_max = np.nanpercentile(C[valid], 99.5)
    else:
        z_min, z_max = 0.5, 500.0

    cmap = plt.colormaps[cmap_name].copy()
    cmap.set_bad('white', 1.0)

    masked = np.ma.masked_where(~valid, C)

    im = ax.pcolormesh(mpl_times, bin_edges, masked,
                        cmap=cmap, shading='flat',
                        norm=LogNorm(vmin=z_min, vmax=z_max))

    ax.set_yscale('log')
    ax.set_ylim([energies.min(), energies.max()])
    ax.set_ylabel(f'{species_label}\nEnergy [eV]', fontsize=8)

    return im


def create_figure2_ssj5(outdir):
    """Create Figure 2 with SSJ5 spectrograms matching paper layout."""
    # Load data
    f17 = read_ssj_file(os.path.join(PROJECT_DIR, 'data', 'dmsp', '2015apr10.f17'))
    f18 = read_ssj_file(os.path.join(PROJECT_DIR, 'data', 'dmsp', '2015apr10.f18'))

    # Bin edges for 19 energy channels
    bin_edges = np.zeros(20)
    bin_edges[0] = CHANNEL_ENERGIES[0] * (CHANNEL_ENERGIES[0] / CHANNEL_ENERGIES[1]) ** 0.5
    bin_edges[-1] = CHANNEL_ENERGIES[-1] * (CHANNEL_ENERGIES[-1] / CHANNEL_ENERGIES[-2]) ** 0.5
    for i in range(1, 19):
        bin_edges[i] = np.sqrt(CHANNEL_ENERGIES[i-1] * CHANNEL_ENERGIES[i])

    # Define passes matching paper's Figure 2
    # (satellite, pass_number, species_groups)
    passes_config = {
        # ~11:00 UT group
        '11_NH': {'data': f18, 'pass_idx': 13, 'label': '~11:00 UT'},  # F18 #14, 0-indexed=13
        '11_SH': {'data': f17, 'pass_idx': 12, 'label': '~11:00 UT'},  # F17 #13, 0-indexed=12
        # ~12:00 UT group
        '12_NH': {'data': f17, 'pass_idx': 15, 'label': '~12:00 UT'},  # F17 #16, 0-indexed=15
        '12_SH': {'data': f17, 'pass_idx': 14, 'label': '~12:00 UT'},  # F17 #15, 0-indexed=14
    }

    # Create figure: 6 rows x 2 columns
    # Row 0: SSUSI (placeholder)
    # Row 1-2: Electron NH, SH
    # Row 3: SSUSI (placeholder)
    # Row 4-5: Ion NH, SH
    fig = plt.figure(figsize=(14, 20), dpi=200)
    gs = fig.add_gridspec(6, 2, height_ratios=[1.2, 1, 1, 1.2, 1, 1],
                          hspace=0.3, wspace=0.15)

    col_idx = 0
    for group_key in ['11', '12']:
        nh_cfg = passes_config[f'{group_key}_NH']
        sh_cfg = passes_config[f'{group_key}_SH']

        # Get pass data
        nh_data = nh_cfg['data']
        nh_passes = find_polar_passes(nh_data['mlat'])
        nh_s, nh_e = nh_passes[nh_cfg['pass_idx']]

        sh_data = sh_cfg['data']
        sh_passes = find_polar_passes(sh_data['mlat'])
        sh_s, sh_e = sh_passes[sh_cfg['pass_idx']]

        # --- SSUSI placeholder row ---
        ax_sususi = fig.add_subplot(gs[0 if col_idx == 0 else 3, col_idx])
        sat_key_nh = f'F{nh_data["satellite"]:02d}'
        sat_key_sh = f'F{sh_data["satellite"]:02d}'
        ax_sususi.text(0.5, 0.5,
                       f'SSUSI LBHS Auroral Image\n'
                       f'({nh_cfg["label"]})\n'
                       f'[Data not available]',
                       transform=ax_sususi.transAxes,
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax_sususi.set_xticks([])
        ax_sususi.set_yticks([])
        panel = '(a)' if col_idx == 0 else '(g)'
        ax_sususi.set_title(f'{panel} {nh_cfg["label"]}', fontsize=10, fontweight='bold')

        # --- Electron spectrograms ---
        for row_offset, (cfg, hemi, sp_label) in enumerate([
            (nh_cfg, 'NH', 'Electron'),
            (sh_cfg, 'SH', 'Electron'),
        ]):
            data = cfg['data']
            ps = find_polar_passes(data['mlat'])
            s, e = ps[cfg['pass_idx']]

            ax = fig.add_subplot(gs[1 + row_offset, col_idx])

            times = data['datetime'][s:e]
            flux = data['eflux_19_rescaled'][:, s:e].astype(np.float64)
            flux[flux <= 0] = np.nan

            # Handle negative values (error flags)
            flux[flux < 0] = np.nan

            im = plot_spectrogram(ax, times, flux, CHANNEL_ENERGIES, bin_edges,
                                  species_label=sp_label)

            dt0 = times[0].strftime('%H:%M')
            dt1 = times[-1].strftime('%H:%M')
            mlat_range = data['mlat'][s:e]
            mlt_range = data['mlt'][s:e]
            sat_key = f'F{data["satellite"]:02d}'

            panel_idx = 2 + col_idx * 4 + row_offset
            panels = ['(c)', '(d)', '(e)', '(f)', '(i)', '(j)', '(k)', '(l)']
            panel_label = panels[panel_idx] if panel_idx < len(panels) else ''

            ax.set_title(f'{panel_label} {sat_key} {sp_label} {hemi}\n'
                        f'{dt0}-{dt1} UT  '
                        f'MLT=[{mlt_range[0]:.0f},{mlt_range[-1]:.0f}]',
                        fontsize=8)

            ax.xaxis_date()
            if row_offset == 1:  # bottom electron row
                ax.set_xlabel('UT', fontsize=8)
            else:
                ax.set_xticklabels([])

            plt.colorbar(im, ax=ax, pad=0.02, aspect=15, label='Flux')

        # --- Ion spectrograms ---
        for row_offset, (cfg, hemi, sp_label) in enumerate([
            (nh_cfg, 'NH', 'Ion'),
            (sh_cfg, 'SH', 'Ion'),
        ]):
            data = cfg['data']
            ps = find_polar_passes(data['mlat'])
            s, e = ps[cfg['pass_idx']]

            ax = fig.add_subplot(gs[4 + row_offset, col_idx])

            times = data['datetime'][s:e]
            flux = data['iflux_19_rescaled'][:, s:e].astype(np.float64)
            flux[flux <= 0] = np.nan
            flux[flux < 0] = np.nan

            im = plot_spectrogram(ax, times, flux, CHANNEL_ENERGIES, bin_edges,
                                  species_label=sp_label)

            dt0 = times[0].strftime('%H:%M')
            dt1 = times[-1].strftime('%H:%M')
            mlat_range = data['mlat'][s:e]
            mlt_range = data['mlt'][s:e]
            sat_key = f'F{data["satellite"]:02d}'

            panel_idx = 4 + col_idx * 4 + row_offset
            panels = ['(c)', '(d)', '(e)', '(f)', '(i)', '(j)', '(k)', '(l)']
            panel_label = panels[panel_idx] if panel_idx < len(panels) else ''

            ax.set_title(f'{panel_label} {sat_key} {sp_label} {hemi}\n'
                        f'{dt0}-{dt1} UT  '
                        f'MLT=[{mlt_range[0]:.0f},{mlt_range[-1]:.0f}]',
                        fontsize=8)

            ax.xaxis_date()
            if row_offset == 1:
                ax.set_xlabel('UT', fontsize=8)
            else:
                ax.set_xticklabels([])

            plt.colorbar(im, ax=ax, pad=0.02, aspect=15, label='Flux')

        col_idx += 1

    fig.suptitle('Figure 2: Auroral and Plasma Observations\n'
                 '2015-04-10 (Wang et al., 2023)',
                 fontsize=13, fontweight='bold', y=0.99)

    plt.savefig(os.path.join(outdir, 'figure2_ssj5_spectrograms.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: figure2_ssj5_spectrograms.png")


def main():
    print("\n" + "=" * 60)
    print("Figure 2 Reproduction (SSJ5 Spectrograms)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    create_figure2_ssj5(OUTPUT_DIR)

    print("Done.")


if __name__ == '__main__':
    main()
