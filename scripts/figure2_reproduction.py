#!/usr/bin/env python
"""
Figure 2 Reproduction: Auroral and plasma observations
Wang et al. (2023) - Communications Earth & Environment

Layout:
  Row 1: SSUSI auroral images (LBHS band) in polar projection
  Row 2-3: SSJ5 electron and ion energy flux spectrograms
  Two time groups: ~11:00 UT and ~12:00 UT on 2015-04-10

Key features to show:
  - Horse-Collar Aurora (HCA) morphology
  - Transpolar arcs at dawn/dusk MLT sectors
  - Precipitation extending to very high latitudes (polar cap closure)
  - Conjugate observations (N/S hemispheres)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from read_ssj_binary import read_ssj_file, find_polar_passes, CHANNEL_ENERGIES


def load_ssj5_data():
    """Load F17 and F18 data."""
    data_dir = os.path.join(PROJECT_DIR, 'data', 'dmsp')
    datasets = {}
    for fname in ['2015apr10.f17', '2015apr10.f18']:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            data = read_ssj_file(fpath)
            sat_key = f'F{data["satellite"]:02d}'
            datasets[sat_key] = data
    return datasets


def find_pass_near_time(data, target_hour, hemisphere='N'):
    """Find the polar pass closest to a target hour."""
    passes = find_polar_passes(data['mlat'])
    best = None
    best_diff = 999
    for pi, (s, e) in enumerate(passes):
        dt_mid = data['datetime'][(s + e) // 2]
        hour = dt_mid.hour + dt_mid.minute / 60.
        mlat_mid = data['mlat'][(s + e) // 2]
        hemi_ok = (hemisphere == 'N' and mlat_mid > 0) or (hemisphere == 'S' and mlat_mid < 0)
        if hemi_ok and abs(hour - target_hour) < best_diff:
            best_diff = abs(hour - target_hour)
            best = (pi, s, e)
    return best


def plot_spectrogram_panel(ax, data, s, e, species='electron', show_lat=True):
    """
    Plot a single spectrogram panel for Figure 2.
    X-axis: MLat, Y-axis: Energy, Color: counts.
    """
    energies = data['channel_energies']
    mlat = data['mlat'][s:e]
    mlt = data['mlt'][s:e]

    if species == 'electron':
        flux = data['eflux_19_rescaled'][:, s:e].astype(np.float64)
        label = 'Electron'
    else:
        flux = data['iflux_19_rescaled'][:, s:e].astype(np.float64)
        label = 'Ion'

    flux[flux <= 0] = np.nan

    # Bin edges
    bin_edges = np.zeros(20)
    bin_edges[0] = energies[0] * (energies[0] / energies[1]) ** 0.5
    bin_edges[-1] = energies[-1] * (energies[-1] / energies[-2]) ** 0.5
    for i in range(1, 19):
        bin_edges[i] = np.sqrt(energies[i-1] * energies[i])

    lat_grid = np.linspace(mlat[0], mlat[-1], e - s)
    C = flux[:, :-1] if flux.shape[1] > 1 else flux
    masked = np.ma.masked_where(~np.isfinite(C), C)

    if np.isfinite(C).any():
        vmin = max(1.0, np.nanpercentile(C[np.isfinite(C)], 0.5))
        vmax = np.nanpercentile(C[np.isfinite(C)], 99)
    else:
        vmin, vmax = 1.0, 1000.0

    cmap = plt.colormaps['Spectral_r'].copy()
    cmap.set_bad('white', 0.1)

    im = ax.pcolormesh(lat_grid, bin_edges, masked, cmap=cmap,
                        shading='flat', norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_yscale('log')
    ax.set_ylim([30, 30000])
    ax.set_ylabel(f'{label}\nEnergy [eV]', fontsize=8)

    # Latitude range annotation
    hemi = 'N' if mlat[len(mlat)//2] > 0 else 'S'
    ax.text(0.02, 0.95, f'{hemi} hemisphere',
            transform=ax.transAxes, fontsize=7,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.5))

    return im


def create_figure2(datasets, outdir):
    """
    Create Figure 2 reproduction:
    Top: SSUSI auroral images (or placeholder if not available)
    Middle: Electron spectrograms for key passes
    Bottom: Ion spectrograms for key passes
    Two columns: ~11:00 UT and ~12:00 UT
    """
    # Select key passes
    # For ~11:00 UT: F17 pass closest to 11:00 (NH) and conjugate SH pass
    # For ~12:00 UT: F18 pass closest to 12:00 (NH) and conjugate SH pass

    time_groups = [
        {'label': '~11:00 UT', 'hour': 11.0},
        {'label': '~12:00 UT', 'hour': 12.0},
    ]

    fig = plt.figure(figsize=(16, 20), dpi=200)
    # Layout: 5 rows x 2 columns
    # Row 0: SSUSI auroral images
    # Row 1: Electron spectrograms (NH)
    # Row 2: Electron spectrograms (SH)
    # Row 3: Ion spectrograms (NH)
    # Row 4: Ion spectrograms (SH)
    gs = fig.add_gridspec(5, 2, height_ratios=[1.2, 1, 1, 1, 1],
                          hspace=0.25, wspace=0.2)

    panel_labels = [
        ['(a)', '(b)', '(g)', '(h)'],  # SSUSI
        ['(c)', '(d)', '(i)', '(j)'],  # Electron NH
        ['(e)', '(f)', '(k)', '(l)'],  # Electron SH / Ion NH
    ]

    # Check for SSUSI images
    ssusi_dir = os.path.join(PROJECT_DIR, '..', 'data', 'ssusi')
    if not os.path.exists(ssusi_dir):
        ssusi_dir = 'C:/Users/Lenovo/data/ssusi'

    ssusi_images = {}
    for sat in ['F17', 'F18']:
        sat_dir = os.path.join(ssusi_dir, sat)
        if os.path.exists(sat_dir):
            files = [f for f in os.listdir(sat_dir) if f.endswith('.nc')]
            ssusi_images[sat] = sorted(files)

    for col, tg in enumerate(time_groups):
        hour = tg['hour']
        label = tg['label']

        # Find best NH and SH passes for this time
        nh_pass = None
        sh_pass = None
        for sat_key in ['F17', 'F18']:
            data = datasets[sat_key]
            nh = find_pass_near_time(data, hour, 'N')
            sh = find_pass_near_time(data, hour, 'S')
            if nh:
                if nh_pass is None or abs(hour - (data['datetime'][(nh[1]+nh[2])//2].hour + data['datetime'][(nh[1]+nh[2])//2].minute/60.)) < \
                   abs(hour - (datasets[nh_pass[0]]['datetime'][(nh_pass[1]+nh_pass[2])//2].hour + datasets[nh_pass[0]]['datetime'][(nh_pass[1]+nh_pass[2])//2].minute/60.)):
                    nh_pass = (sat_key,) + nh
            if sh:
                if sh_pass is None or abs(hour - (data['datetime'][(sh[1]+sh[2])//2].hour + data['datetime'][(sh[1]+sh[2])//2].minute/60.)) < \
                   abs(hour - (datasets[sh_pass[0]]['datetime'][(sh_pass[1]+sh_pass[2])//2].hour + datasets[sh_pass[0]]['datetime'][(sh_pass[1]+sh_pass[2])//2].minute/60.)):
                    sh_pass = (sat_key,) + sh

        # --- Row 0: SSUSI auroral image ---
        ax_sususi = fig.add_subplot(gs[0, col])
        try:
            # Try to find an SSUSI image near this time
            sat_key = nh_pass[0] if nh_pass else 'F17'
            # Look for SSUSI image
            found_image = False
            for search_dir in [os.path.join(PROJECT_DIR, 'output'), 'C:/Users/Lenovo/data/ssusi']:
                if os.path.exists(search_dir):
                    for f in os.listdir(search_dir):
                        if 'ssusi' in f.lower() and ('auroral' in f.lower() or 'lbhs' in f.lower() or 'F17' in f or 'F18' in f) and f.endswith('.png'):
                            img_path = os.path.join(search_dir, f)
                            img = Image.open(img_path)
                            ax_sususi.imshow(img)
                            ax_sususi.set_xticks([])
                            ax_sususi.set_yticks([])
                            found_image = True
                            break
                if found_image:
                    break

            if not found_image:
                ax_sususi.text(0.5, 0.5,
                              f'SSUSI LBHS Auroral Image\n{sat_key} Northern Hemisphere\n{label}',
                              transform=ax_sususi.transAxes,
                              ha='center', va='center', fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                ax_sususi.set_xticks([])
                ax_sususi.set_yticks([])
        except Exception:
            ax_sususi.text(0.5, 0.5, f'SSUSI Image\n{label}',
                          transform=ax_sususi.transAxes,
                          ha='center', va='center', fontsize=10)
            ax_sususi.set_xticks([])
            ax_sususi.set_yticks([])

        plabel = '(a)' if col == 0 else '(b)'
        ax_sususi.set_title(f'{plabel} SSUSI LBHS - {label}', fontsize=11, fontweight='bold')

        # --- Rows 1-4: Spectrograms ---
        for row, (pass_info, species, hemi_label) in enumerate([
            (nh_pass, 'electron', 'NH'),
            (sh_pass, 'electron', 'SH'),
            (nh_pass, 'ion', 'NH'),
            (sh_pass, 'ion', 'SH'),
        ]):
            ax = fig.add_subplot(gs[row + 1, col])

            if pass_info is None:
                ax.text(0.5, 0.5, f'No {hemi_label} pass\nnear {label}',
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                continue

            sat_key, pi, s, e = pass_info
            data = datasets[sat_key]

            im = plot_spectrogram_panel(ax, data, s, e, species=species)

            dt0 = data['datetime'][s].strftime('%H:%M')
            dt1 = data['datetime'][e-1].strftime('%H:%M')
            mlt0 = data['mlt'][s]
            mlt1 = data['mlt'][e-1]
            mlat_s = data['mlat'][s]
            mlat_e = data['mlat'][e-1]

            # Panel labels
            panel_idx = row * 2 + col
            all_labels = ['(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
            if panel_idx < len(all_labels):
                plabel = all_labels[panel_idx]

            species_short = 'e$^-$' if species == 'electron' else 'ion'
            ax.set_title(f'{plabel} {sat_key} {species_short} {hemi_label}\n'
                        f'{dt0}-{dt1} UT  MLT=[{mlt0:.0f},{mlt1:.0f}]',
                        fontsize=8)
            ax.set_xlabel('MLat (deg)', fontsize=8)

            plt.colorbar(im, ax=ax, pad=0.02, aspect=15)

    fig.suptitle('Figure 2: Auroral and Plasma Observations\n'
                 '2015-04-10 Northward IMF Event (Wang et al., 2023)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(outdir, 'figure2_reproduction.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: figure2_reproduction.png")


def create_figure2_compact(datasets, outdir):
    """
    Create a compact version of Figure 2 focusing on the key NH passes
    around 11:00 and 12:00 UT.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), dpi=200, squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    time_targets = [11.0, 12.0]
    species_list = ['electron', 'ion']

    for col, hour in enumerate(time_targets):
        # Find best NH pass
        best_pass = None
        best_sat = None
        for sat_key in ['F17', 'F18']:
            data = datasets[sat_key]
            p = find_pass_near_time(data, hour, 'N')
            if p:
                dt_mid = data['datetime'][(p[1]+p[2])//2]
                h_mid = dt_mid.hour + dt_mid.minute/60.
                if best_pass is None or abs(h_mid - hour) < abs(best_pass[2] - hour):
                    best_pass = (sat_key, p[0], p[1], p[2])
                    best_sat = sat_key

        for row, species in enumerate(species_list):
            ax = axes[row, col]

            if best_pass is None:
                ax.text(0.5, 0.5, f'No data near {hour:.0f}:00 UT',
                       transform=ax.transAxes, ha='center', va='center')
                continue

            sat_key, pi, s, e = best_pass
            data = datasets[sat_key]

            im = plot_spectrogram_panel(ax, data, s, e, species=species)

            dt0 = data['datetime'][s].strftime('%H:%M')
            dt1 = data['datetime'][e-1].strftime('%H:%M')
            mlt0 = data['mlt'][s]
            mlt1 = data['mlt'][e-1]

            species_short = 'Electron' if species == 'electron' else 'Ion'
            ax.set_title(f'DMSP {sat_key} {species_short}\n'
                        f'~{hour:.0f}:00 UT ({dt0}-{dt1})  '
                        f'MLT=[{mlt0:.0f},{mlt1:.0f}]',
                        fontsize=9, fontweight='bold')
            ax.set_xlabel('MLat (deg)', fontsize=8)

            plt.colorbar(im, ax=ax, pad=0.02, aspect=15)

        # Add SuperDARN convection map for this time in the bottom row
        ax_sd = axes[3, col]
        sd_file = os.path.join(PROJECT_DIR, 'data', 'superdarn',
                                f'map-nth-20150410-{int(hour):02d}00.jpg')
        if os.path.exists(sd_file):
            img = Image.open(sd_file)
            ax_sd.imshow(img)
            ax_sd.set_xticks([])
            ax_sd.set_yticks([])
            ax_sd.set_title(f'SuperDARN Convection\n~{hour:.0f}:00 UT',
                          fontsize=9, fontweight='bold')
        else:
            ax_sd.text(0.5, 0.5, f'SuperDARN map\n{hour:.0f}:00 UT',
                      transform=ax_sd.transAxes, ha='center', va='center')
            ax_sd.set_xticks([])
            ax_sd.set_yticks([])

    fig.suptitle('Figure 2: Auroral and Plasma Observations - 2015-04-10\n'
                 'Wang et al. (2023) Reproduction',
                 fontsize=13, fontweight='bold')

    plt.savefig(os.path.join(outdir, 'figure2_compact.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: figure2_compact.png")


def create_figure3_superdarn_imf(outdir):
    """
    Create Figure 3: SuperDARN convection + IMF conditions.
    Shows the relationship between IMF Bz and convection pattern.
    """
    fig = plt.figure(figsize=(16, 10), dpi=200)

    # Top row: IMF conditions (reuse existing plot if available)
    imf_file = os.path.join(outdir, 'imf_timeseries_20150410.png')
    if os.path.exists(imf_file):
        ax_imf = fig.add_subplot(2, 2, (1, 2))
        img = Image.open(imf_file)
        ax_imf.imshow(img)
        ax_imf.set_xticks([])
        ax_imf.set_yticks([])
        ax_imf.set_title('(a) IMF and Solar Wind Conditions', fontsize=11, fontweight='bold')
    else:
        ax_imf = fig.add_subplot(2, 2, (1, 2))
        ax_imf.text(0.5, 0.5, 'IMF Timeseries (imf_timeseries_20150410.png)',
                   transform=ax_imf.transAxes, ha='center', va='center')

    # Bottom row: SuperDARN at two key times
    for col, (ut_time, label, desc) in enumerate([
        ('0004', '00:04 UT', 'Pre-event: Southward IMF'),
        ('1100', '11:00 UT', 'Event: Northward IMF + DLR'),
    ]):
        ax = fig.add_subplot(2, 2, 3 + col)
        sd_file = os.path.join(PROJECT_DIR, 'data', 'superdarn',
                                f'map-nth-20150410-{ut_time}.jpg')
        if os.path.exists(sd_file):
            img = Image.open(sd_file)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            panel = '(b)' if col == 0 else '(c)'
            ax.set_title(f'{panel} SuperDARN {label}\n{desc}',
                        fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'SuperDARN {ut_time} UT',
                   transform=ax.transAxes, ha='center', va='center')

    fig.suptitle('Figure 3: Solar Wind Driving and Convection Response\n'
                 '2015-04-10 (Wang et al., 2023)',
                 fontsize=13, fontweight='bold')

    plt.savefig(os.path.join(outdir, 'figure3_imf_convection.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: figure3_imf_convection.png")


def main():
    print("\n" + "=" * 60)
    print("Figure Reproduction - Wang et al. (2023)")
    print("Event: 2015-04-10")
    print("=" * 60)

    outdir = os.path.join(PROJECT_DIR, 'output')
    os.makedirs(outdir, exist_ok=True)

    print("\nLoading SSJ5 data...")
    datasets = load_ssj5_data()
    for sat_key in sorted(datasets.keys()):
        print(f"  {sat_key}: {datasets[sat_key]['n_records']} records")

    print("\nGenerating Figure 2 (Auroral + Plasma)...")
    create_figure2(datasets, outdir)
    create_figure2_compact(datasets, outdir)

    print("\nGenerating Figure 3 (SuperDARN + IMF)...")
    create_figure3_superdarn_imf(outdir)

    print(f"\nAll figures saved to: {outdir}/")
    print("Done.")


if __name__ == '__main__':
    main()
