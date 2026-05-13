"""
Sound speed profile (SSP) library for underwater acoustic propagation.

All profiles return a numpy array of shape (N, 2) where column 0 is depth
in meters and column 1 is sound speed in m/s.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def munk_profile(c0=1500.0, z_axis=1300.0, B=1300.0, epsilon=0.00737, z_max=5000.0, n_points=200):
    """Classic Munk deep-ocean profile with SOFAR channel at ~1300 m.

    The Munk profile is the canonical model for deep ocean sound propagation
    and is used throughout the BELLHOP literature.

    Parameters
    ----------
    c0 : float
        Reference sound speed at channel axis (m/s), default 1500
    z_axis : float
        Depth of the SOFAR channel axis (m), default 1300
    B : float
        Thermocline scale depth (m), default 1300
    epsilon : float
        Perturbation strength (dimensionless), default 0.00737
    z_max : float
        Maximum depth to compute profile (m)
    n_points : int
        Number of depth samples

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 2): column 0 = depth (m), column 1 = speed (m/s)
    """
    depths = np.linspace(0.0, z_max, n_points)
    zeta = 2.0 * (depths - z_axis) / B
    speeds = c0 * (1.0 + epsilon * (zeta + np.exp(-zeta) - 1.0))
    return np.column_stack([depths, speeds])


def isothermal_profile(speed=1500.0, z_max=5000.0, n_points=10):
    """Constant sound speed with depth — the simplest possible SSP.

    Parameters
    ----------
    speed : float
        Sound speed (m/s), constant at all depths
    z_max : float
        Maximum depth (m)
    n_points : int
        Number of depth samples (only 2 needed, more for consistency)

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 2)
    """
    depths = np.linspace(0.0, z_max, n_points)
    speeds = np.full(n_points, speed)
    return np.column_stack([depths, speeds])


def surface_duct_profile(duct_depth=100.0, duct_speed=1520.0, z_max=5000.0, n_points=200):
    """Surface duct profile: elevated near-surface speed, thermocline drop, deep increase.

    A surface duct traps sound near the surface between the surface and a
    shadow zone below the duct. Common in polar regions and after storms.

    Parameters
    ----------
    duct_depth : float
        Depth of the base of the surface duct (m), default 100
    duct_speed : float
        Sound speed at the surface (m/s), default 1520
    z_max : float
        Maximum depth (m)
    n_points : int
        Number of depth samples

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 2)
    """
    depths = np.linspace(0.0, z_max, n_points)
    speeds = np.zeros(n_points)

    thermocline_base = duct_depth * 3.0
    deep_min_speed = 1480.0
    deep_gradient = 0.017  # m/s per meter

    for i, z in enumerate(depths):
        if z <= duct_depth:
            # Surface duct: nearly constant with slight decrease
            speeds[i] = duct_speed - 0.02 * z
        elif z <= thermocline_base:
            # Thermocline: steep drop from duct speed to deep minimum
            frac = (z - duct_depth) / (thermocline_base - duct_depth)
            speeds[i] = (duct_speed - 0.02 * duct_depth) * (1.0 - frac) + deep_min_speed * frac
        else:
            # Deep isothermal / slight pressure increase
            speeds[i] = deep_min_speed + deep_gradient * (z - thermocline_base)

    return np.column_stack([depths, speeds])


def arctic_profile(surface_speed=1435.0, z_max=4000.0, n_points=200):
    """Arctic profile: speed increases monotonically from cold surface downward.

    In the Arctic, the surface is at or near freezing so sound speed is
    minimum at the surface and increases monotonically with depth due to
    pressure. This produces completely upward-refracting rays.

    Parameters
    ----------
    surface_speed : float
        Sound speed at surface (m/s), default 1435 (near-freezing water)
    z_max : float
        Maximum depth (m)
    n_points : int
        Number of depth samples

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 2)
    """
    depths = np.linspace(0.0, z_max, n_points)
    # Halocline effect in top 200 m, then pressure-dominated increase
    speeds = np.zeros(n_points)
    for i, z in enumerate(depths):
        if z < 200.0:
            speeds[i] = surface_speed + 0.1 * z
        else:
            speeds[i] = surface_speed + 20.0 + 0.016 * (z - 200.0)
    return np.column_stack([depths, speeds])


def shallow_water_profile(total_depth=200.0, surface_speed=1520.0, bottom_speed=1490.0, n_points=100):
    """Realistic continental shelf profile with warm surface, sharp thermocline, isothermal bottom.

    Parameters
    ----------
    total_depth : float
        Water column depth (m), default 200
    surface_speed : float
        Sound speed at surface (m/s), warm mixed layer
    bottom_speed : float
        Sound speed at bottom (m/s), cold bottom water
    n_points : int
        Number of depth samples

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 2)
    """
    depths = np.linspace(0.0, total_depth, n_points)
    speeds = np.zeros(n_points)

    mixed_depth = total_depth * 0.1       # 10% of total depth
    thermocline_base = total_depth * 0.4  # thermocline from 10% to 40%

    for i, z in enumerate(depths):
        if z <= mixed_depth:
            # Warm mixed surface layer: very slight negative gradient
            speeds[i] = surface_speed - 0.05 * z
        elif z <= thermocline_base:
            # Sharp thermocline: rapid decrease
            frac = (z - mixed_depth) / (thermocline_base - mixed_depth)
            top_speed = surface_speed - 0.05 * mixed_depth
            speeds[i] = top_speed + (bottom_speed - top_speed) * frac
        else:
            # Nearly isothermal bottom water with slight pressure increase
            dz = z - thermocline_base
            speeds[i] = bottom_speed + 0.005 * dz

    return np.column_stack([depths, speeds])


def custom_profile(depth_speed_pairs):
    """Linearly interpolated profile from user-defined (depth, speed) pairs.

    Parameters
    ----------
    depth_speed_pairs : list of tuples or array-like of shape (N, 2)
        Each row is (depth_m, speed_ms). Depths must be monotonically increasing.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 2) (same as input, sorted by depth)
    """
    arr = np.array(depth_speed_pairs, dtype=float)
    # Sort by depth
    idx = np.argsort(arr[:, 0])
    return arr[idx]


def plot_profile(ssp, ax=None, label=None, color=None, title=None):
    """Plot a sound speed profile with oceanographic convention.

    Depth on Y-axis increasing downward, speed on X-axis.

    Parameters
    ----------
    ssp : numpy.ndarray
        Array of shape (N, 2): depth (m) and speed (m/s)
    ax : matplotlib Axes, optional
        Axes to plot on; creates new figure if None
    label : str, optional
        Legend label
    color : str, optional
        Line color
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 7))

    kwargs = {}
    if label is not None:
        kwargs['label'] = label
    if color is not None:
        kwargs['color'] = color

    ax.plot(ssp[:, 1], ssp[:, 0], linewidth=2, **kwargs)
    ax.invert_yaxis()
    ax.set_xlabel('Sound Speed (m/s)', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=12)
    elif label:
        ax.set_title(label, fontsize=12)

    # Annotate min/max and channel axis
    min_idx = np.argmin(ssp[:, 1])
    ax.axhline(y=ssp[min_idx, 0], color='red', linestyle='--', alpha=0.5,
               label=f'SOFAR axis: {ssp[min_idx, 0]:.0f} m')

    if label:
        ax.legend(fontsize=9)

    return ax


def plot_comparison(ssp_list, labels=None, title='Sound Speed Profile Comparison'):
    """Overlay multiple SSPs on the same axes for direct comparison.

    Parameters
    ----------
    ssp_list : list of numpy.ndarray
        List of SSP arrays from any profile function
    labels : list of str, optional
        Legend labels for each profile
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(ssp_list)))
    fig, ax = plt.subplots(figsize=(5, 8))

    for i, ssp in enumerate(ssp_list):
        label = labels[i] if labels and i < len(labels) else f'Profile {i+1}'
        ax.plot(ssp[:, 1], ssp[:, 0], linewidth=2, color=colors[i], label=label)

    ax.invert_yaxis()
    ax.set_xlabel('Sound Speed (m/s)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def get_sofar_depth(ssp):
    """Return the depth of the sound channel axis (minimum sound speed).

    Parameters
    ----------
    ssp : numpy.ndarray
        SSP array of shape (N, 2)

    Returns
    -------
    float
        Depth in meters where sound speed is minimum (SOFAR channel axis)
    """
    min_idx = np.argmin(ssp[:, 1])
    return float(ssp[min_idx, 0])


def get_speed_range(ssp):
    """Return (min_speed, max_speed) tuple for the profile.

    Parameters
    ----------
    ssp : numpy.ndarray
        SSP array of shape (N, 2)

    Returns
    -------
    tuple of float
        (min_speed_ms, max_speed_ms)
    """
    return float(np.min(ssp[:, 1])), float(np.max(ssp[:, 1]))
