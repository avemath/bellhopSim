"""
Publication-quality plotting functions for BELLHOP simulation results.

All functions use consistent styling: oceanographic depth convention
(depth increases downward on Y-axis), dark blue water, visible grid,
labeled axes with units.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection


# Shared style constants
WATER_COLOR = '#1a6fa3'
WATER_ALPHA = 0.08
BOTTOM_COLOR = '#8B6914'
SURFACE_COLOR = '#2196F3'
GRID_ALPHA = 0.25
LABEL_SIZE = 11
TITLE_SIZE = 13


def _get_depth(env):
    """Extract water depth from env dict."""
    d = env.get('depth', None)
    if d is None:
        return 1000.0
    if hasattr(d, '__len__'):
        return float(np.max(d))
    return float(d)


def _get_max_range(env):
    """Extract max receiver range in km from env dict."""
    rr = env.get('rx_range', None)
    if rr is None:
        return 10.0
    return float(np.max(rr))


def plot_ssp(ssp, ax=None, title='Sound Speed Profile', color=WATER_COLOR):
    """Plot a sound speed profile with depth increasing downward.

    Parameters
    ----------
    ssp : numpy.ndarray
        Array of shape (N, 2): column 0 = depth (m), column 1 = speed (m/s)
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; creates new figure if None
    title : str
        Plot title
    color : str
        Line color

    Returns
    -------
    matplotlib.axes.Axes
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4, 7))

    ax.plot(ssp[:, 1], ssp[:, 0], color=color, linewidth=2.5)

    # Shade min-speed (SOFAR) region
    min_idx = np.argmin(ssp[:, 1])
    min_depth = ssp[min_idx, 0]
    ax.axhline(min_depth, color='crimson', linestyle='--', linewidth=1,
               label=f'SOFAR axis: {min_depth:.0f} m  ({ssp[min_idx,1]:.0f} m/s)')

    ax.invert_yaxis()
    ax.set_xlabel('Sound Speed (m/s)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Depth (m)', fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=9)

    c_min, c_max = ssp[:, 1].min(), ssp[:, 1].max()
    margin = (c_max - c_min) * 0.05 if c_max > c_min else 5.0
    ax.set_xlim(c_min - margin, c_max + margin)

    if standalone:
        plt.tight_layout()
    return ax


def plot_rays(ray_data, env, ax=None, title=None, max_rays=None, alpha=0.6):
    """Ray diagram with ocean floor, surface, and rays colored by launch angle.

    Parameters
    ----------
    ray_data : pandas.DataFrame or list
        Output from pm.compute_rays(). Each row is one ray path; columns
        'x' (range in m) and 'y' (depth in m) provide the path.
    env : dict
        arlpy environment dictionary
    ax : matplotlib.axes.Axes, optional
        Axes to draw on
    title : str, optional
        Plot title
    max_rays : int, optional
        Limit number of displayed rays for clarity
    alpha : float
        Ray line transparency

    Returns
    -------
    matplotlib.axes.Axes
    """
    import pandas as _pd

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 5))

    water_depth = _get_depth(env)
    max_range_km = _get_max_range(env)

    # Draw water background
    ax.set_facecolor('#e8f4f8')

    # Draw bottom as filled region
    max_range_m = max_range_km * 1000.0
    bottom_x = [0, max_range_m, max_range_m, 0]
    bottom_y = [water_depth, water_depth, water_depth * 1.05, water_depth * 1.05]
    ax.fill(bottom_x, bottom_y, color=BOTTOM_COLOR, alpha=0.9, zorder=2)
    ax.axhline(water_depth, color=BOTTOM_COLOR, linewidth=2, zorder=3)

    # Draw surface
    ax.axhline(0, color=SURFACE_COLOR, linewidth=2, alpha=0.8, zorder=3, label='Surface')

    # Plot rays
    if ray_data is None:
        ax.text(0.5, 0.5, 'No ray data', transform=ax.transAxes,
                ha='center', va='center', fontsize=13, color='red')
    elif isinstance(ray_data, _pd.DataFrame):
        rays_list = [ray_data]
    else:
        rays_list = ray_data if hasattr(ray_data, '__iter__') else [ray_data]

    if ray_data is not None:
        rays_list = ray_data if hasattr(ray_data, '__len__') else [ray_data]
        if hasattr(ray_data, 'iterrows'):
            # Single DataFrame where each row is a ray
            n_rays = len(ray_data)
            if max_rays is not None:
                step = max(1, n_rays // max_rays)
                indices = range(0, n_rays, step)
            else:
                indices = range(n_rays)

            cmap = cm.get_cmap('RdYlBu')
            for i in indices:
                row = ray_data.iloc[i]
                if 'x' in row and 'y' in row:
                    x = np.asarray(row['x'])
                    y = np.asarray(row['y'])
                    # Color by first launch angle if available
                    color_val = (i / max(n_rays - 1, 1))
                    ax.plot(x * 1e3 if x.max() < 500 else x,
                            y, color=cmap(color_val),
                            linewidth=0.8, alpha=alpha, zorder=4)

    ax.set_xlim(0, max_range_m)
    ax.set_ylim(water_depth * 1.05, -water_depth * 0.02)
    ax.set_xlabel('Range (m)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Depth (m)', fontsize=LABEL_SIZE)
    ax.grid(True, alpha=GRID_ALPHA)

    src_depth = float(env.get('tx_depth', 0))
    ax.plot(0, src_depth, 'r*', markersize=14, zorder=10, label=f'Source ({src_depth:.0f} m)')

    freq = env.get('frequency', '')
    default_title = f'Ray Diagram  |  f = {freq} Hz  |  depth = {water_depth:.0f} m'
    ax.set_title(title or default_title, fontsize=TITLE_SIZE)
    ax.legend(fontsize=9, loc='upper right')

    if standalone:
        plt.tight_layout()
    return ax


def plot_transmission_loss(tl_data, env, ax=None, dynamic_range=60, title=None, cmap='jet'):
    """2D transmission loss color map (depth vs range).

    Parameters
    ----------
    tl_data : pandas.DataFrame
        Output from pm.compute_transmission_loss(). Index = depths (m),
        columns = ranges (km).
    env : dict
        arlpy environment dictionary
    ax : matplotlib.axes.Axes, optional
        Axes to draw on
    dynamic_range : float
        Color scale range in dB (default 60). The color axis spans
        [TL_min, TL_min + dynamic_range].
    title : str, optional
        Plot title
    cmap : str
        Colormap name

    Returns
    -------
    matplotlib.axes.Axes, matplotlib.colorbar.Colorbar
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 5))

    water_depth = _get_depth(env)
    freq = env.get('frequency', '?')
    src_depth = float(env.get('tx_depth', 0))

    if tl_data is None:
        ax.text(0.5, 0.5, 'No TL data', transform=ax.transAxes,
                ha='center', va='center', fontsize=13, color='red')
        return ax, None

    # tl_data: DataFrame index=depths, columns=ranges
    # arlpy returns complex pressure amplitudes (|p| << 1), not TL in dB.
    # Convert: TL_dB = -20 * log10(|p|)
    depths = np.asarray(tl_data.index, dtype=float)
    ranges_km = np.asarray(tl_data.columns, dtype=float)
    pressure_abs = np.abs(np.asarray(tl_data.values, dtype=complex))
    pressure_abs = np.where(pressure_abs < 1e-10, np.nan, pressure_abs)
    tl_matrix = -20.0 * np.log10(pressure_abs)

    vmin = np.nanmin(tl_matrix)
    vmax = vmin + dynamic_range

    im = ax.pcolormesh(ranges_km, depths, tl_matrix,
                       vmin=vmin, vmax=vmax, cmap=cmap,
                       shading='auto')

    # Draw ocean floor
    ax.axhline(water_depth, color=BOTTOM_COLOR, linewidth=3, label='Bottom')
    ax.fill_between([ranges_km.min(), ranges_km.max()],
                    [water_depth] * 2, [water_depth * 1.15] * 2,
                    color=BOTTOM_COLOR, alpha=0.8)

    # Mark source
    ax.plot(ranges_km.min(), src_depth, 'r*', markersize=14, zorder=10,
            label=f'Source ({src_depth:.0f} m)')

    ax.set_ylim(min(water_depth * 1.08, depths.max() * 1.05), -water_depth * 0.02)
    ax.set_xlim(ranges_km.min(), ranges_km.max())
    ax.set_xlabel('Range (km)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Depth (m)', fontsize=LABEL_SIZE)
    ax.grid(True, alpha=GRID_ALPHA, color='white')
    ax.legend(fontsize=9, loc='upper right')

    default_title = f'Transmission Loss  |  f = {freq} Hz  |  {dynamic_range} dB dynamic range'
    ax.set_title(title or default_title, fontsize=TITLE_SIZE)

    if standalone:
        cb = plt.colorbar(im, ax=ax, label='TL (dB re 1 m)', fraction=0.035)
        plt.tight_layout()
    else:
        cb = plt.colorbar(im, ax=ax, label='TL (dB re 1 m)', fraction=0.035)

    return ax, cb


def plot_tl_slice(tl_data, depth_m, env, ax=None, title=None, color=WATER_COLOR):
    """Horizontal TL slice at a given depth vs range.

    Parameters
    ----------
    tl_data : pandas.DataFrame
        Transmission loss DataFrame (index = depths, columns = ranges km)
    depth_m : float
        Depth at which to extract the horizontal slice
    env : dict
        arlpy environment dictionary
    ax : matplotlib.axes.Axes, optional
    title : str, optional
    color : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    if tl_data is None:
        ax.text(0.5, 0.5, 'No TL data', transform=ax.transAxes,
                ha='center', va='center')
        return ax

    depths = np.asarray(tl_data.index, dtype=float)
    ranges_km = np.asarray(tl_data.columns, dtype=float)

    # Find nearest depth index
    depth_idx = int(np.argmin(np.abs(depths - depth_m)))
    actual_depth = depths[depth_idx]

    pressure_abs = np.abs(np.asarray(tl_data.iloc[depth_idx, :], dtype=complex))
    pressure_abs = np.where(pressure_abs < 1e-10, np.nan, pressure_abs)
    tl_slice = -20.0 * np.log10(pressure_abs)

    ax.plot(ranges_km, tl_slice, color=color, linewidth=2,
            label=f'TL at {actual_depth:.0f} m depth')
    ax.invert_yaxis()
    ax.set_xlabel('Range (km)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Transmission Loss (dB)', fontsize=LABEL_SIZE)

    freq = env.get('frequency', '?')
    default_title = f'TL vs Range at {actual_depth:.0f} m  |  f = {freq} Hz'
    ax.set_title(title or default_title, fontsize=TITLE_SIZE)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=10)

    if standalone:
        plt.tight_layout()
    return ax


def plot_arrivals(arr_data, env=None, rx_range_idx=0, rx_depth_idx=0, ax=None, title=None):
    """Stem plot of multipath arrival amplitudes vs arrival time.

    Parameters
    ----------
    arr_data : pandas.DataFrame
        Output from pm.compute_arrivals(). Contains columns 'time_of_arrival'
        and 'amplitude' (complex).
    env : dict, optional
        arlpy environment dictionary (for title annotation)
    rx_range_idx : int
        Index into receiver range array to plot
    rx_depth_idx : int
        Index into receiver depth array to plot
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import pandas as _pd

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    if arr_data is None:
        ax.text(0.5, 0.5, 'No arrivals data', transform=ax.transAxes,
                ha='center', va='center', fontsize=13, color='red')
        if standalone:
            plt.tight_layout()
        return ax

    # Filter to the requested rx range/depth if multi-indexed
    try:
        if isinstance(arr_data.index, _pd.MultiIndex):
            idx_levels = arr_data.index.levels
            r_val = idx_levels[0][rx_range_idx] if len(idx_levels) > 0 else None
            d_val = idx_levels[1][rx_depth_idx] if len(idx_levels) > 1 else None
            if r_val is not None and d_val is not None:
                df = arr_data.loc[(r_val, d_val)]
            else:
                df = arr_data
        else:
            df = arr_data
    except (KeyError, IndexError):
        df = arr_data

    if 'time_of_arrival' in df.columns and 'amplitude' in df.columns:
        times = np.asarray(df['time_of_arrival'], dtype=float) * 1000.0  # convert to ms
        amps = np.abs(np.asarray(df['amplitude'], dtype=complex))

        # Normalize
        if amps.max() > 0:
            amps_norm = amps / amps.max()
        else:
            amps_norm = amps

        markerline, stemlines, baseline = ax.stem(
            times, amps_norm, linefmt=f'{WATER_COLOR}', markerfmt='o',
            basefmt='k-'
        )
        markerline.set_markerfacecolor(WATER_COLOR)
        markerline.set_markersize(8)
        stemlines.set_linewidth(1.5)
        stemlines.set_alpha(0.8)

        ax.set_xlabel('Arrival Time (ms)', fontsize=LABEL_SIZE)
        ax.set_ylabel('Normalized Amplitude', fontsize=LABEL_SIZE)

        freq = env.get('frequency', '?') if env else '?'
        default_title = f'Multipath Arrivals  |  f = {freq} Hz  |  {len(times)} paths'
        ax.set_title(title or default_title, fontsize=TITLE_SIZE)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.set_ylim(-0.05, 1.15)
    else:
        ax.text(0.5, 0.5, f'Unexpected data format:\n{df.columns.tolist()}',
                transform=ax.transAxes, ha='center', va='center', fontsize=10)

    if standalone:
        plt.tight_layout()
    return ax


def plot_impulse_response(ir, fs, ax=None, title='Channel Impulse Response', color=WATER_COLOR):
    """Plot a discrete-time channel impulse response.

    Parameters
    ----------
    ir : numpy.ndarray
        Impulse response samples (real or complex magnitudes)
    fs : float
        Sample rate in Hz
    ax : matplotlib.axes.Axes, optional
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    t_ms = np.arange(len(ir)) / fs * 1000.0
    ax.stem(t_ms, np.abs(ir), linefmt=color, markerfmt='o', basefmt='k-')
    ax.set_xlabel('Time (ms)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Amplitude', fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.grid(True, alpha=GRID_ALPHA)

    if standalone:
        plt.tight_layout()
    return ax


def plot_scenario_comparison(tl_results, labels, env_list, depth_m=None):
    """Overlay TL-vs-range curves from multiple scenarios for comparison.

    Parameters
    ----------
    tl_results : list of pandas.DataFrame
        TL DataFrames from pm.compute_transmission_loss() for each scenario
    labels : list of str
        Scenario names
    env_list : list of dict
        arlpy environment dicts (one per scenario)
    depth_m : float, optional
        Depth at which to extract TL slice; defaults to source depth of first env

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(tl_results)))

    for i, (tl, label, env) in enumerate(zip(tl_results, labels, env_list)):
        if tl is None:
            continue
        depths = np.asarray(tl.index, dtype=float)
        ranges_km = np.asarray(tl.columns, dtype=float)

        d_ref = depth_m if depth_m is not None else float(env.get('tx_depth', depths[len(depths)//2]))
        depth_idx = int(np.argmin(np.abs(depths - d_ref)))

        p_abs = np.abs(np.asarray(tl.iloc[depth_idx, :], dtype=complex))
        p_abs = np.where(p_abs < 1e-10, np.nan, p_abs)
        tl_slice = -20.0 * np.log10(p_abs)
        ax.plot(ranges_km, tl_slice, color=colors[i], linewidth=2, label=label)

    ax.invert_yaxis()
    ax.set_xlabel('Range (km)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Transmission Loss (dB)', fontsize=LABEL_SIZE)
    ax.set_title(f'TL Comparison at {depth_m:.0f} m depth' if depth_m else 'TL Comparison',
                 fontsize=TITLE_SIZE)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def annotate_convergence_zones(ax, cz_ranges, y_pos=None):
    """Add vertical shaded bands at convergence zone ranges.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    cz_ranges : list of float
        Range values (km) at which convergence zones were identified
    y_pos : float, optional
        Y position for text labels (defaults to top of axes)
    """
    ylim = ax.get_ylim()
    y_label = y_pos if y_pos is not None else min(ylim)

    for r in cz_ranges:
        ax.axvline(r, color='yellow', linewidth=1.5, alpha=0.8, linestyle='--')
        ax.axvspan(r - 5, r + 5, alpha=0.12, color='yellow')
        ax.text(r, y_label, f'CZ\n{r:.0f} km', fontsize=8, color='goldenrod',
                ha='center', va='top', fontweight='bold')
