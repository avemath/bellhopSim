"""
BELLHOP environment construction helpers.

Provides functions to build arlpy environment dictionaries for common
ocean acoustic scenarios and write them to disk for inspection.
"""

import os
import numpy as np
from datetime import datetime

# Path to the envfiles directory relative to this file's parent
_ENVFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'envfiles')

# Bottom type lookup: name -> (sound_speed m/s, density kg/m3, attenuation dB/lambda).
# arlpy expects density in kg/m3 (it divides by 1000 when writing the .env file).
BOTTOM_TYPES = {
    'mud':    {'soundspeed': 1450.0, 'density': 1200.0, 'attenuation': 0.5},
    'sand':   {'soundspeed': 1650.0, 'density': 1900.0, 'attenuation': 0.8},
    'gravel': {'soundspeed': 1800.0, 'density': 2000.0, 'attenuation': 0.6},
    'rock':   {'soundspeed': 3000.0, 'density': 2500.0, 'attenuation': 0.1},
}


def _apply_bottom(env, bottom_type):
    """Apply bottom acoustic parameters from the lookup table to env dict."""
    bt = bottom_type.lower()
    if bt not in BOTTOM_TYPES:
        raise ValueError(f"Unknown bottom type '{bottom_type}'. Choose from: {list(BOTTOM_TYPES.keys())}")
    params = BOTTOM_TYPES[bt]
    env['bottom_soundspeed'] = params['soundspeed']
    env['bottom_density'] = params['density']
    env['bottom_absorption'] = params['attenuation']
    return env


# ── Units boundary ────────────────────────────────────────────────────────────
# arlpy works in meters for every range (receiver ranges, bathymetry ranges,
# ray x coordinates, TL columns). This project works in km. All BELLHOP runs go
# through create_env() and the run_* helpers below so the conversion happens
# in exactly one place.

M_PER_KM = 1000.0


def _as_arlpy_axis(values):
    """Return a float array, or a plain float for a single value.

    arlpy's .env writer only accepts a scalar or an array of length >= 2;
    a 1-element array raises TypeError under NumPy 2.
    """
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    return float(arr[0]) if arr.size == 1 else arr


def create_env(depth, ssp, frequency, tx_depth, rx_depths_m, rx_ranges_km,
               bottom_type=None, **kwargs):
    """Build an arlpy environment with receiver ranges given in km.

    Parameters
    ----------
    depth : float or array-like of shape (N, 2)
        Water depth in meters, or range-dependent bathymetry as
        (range_km, depth_m) pairs.
    ssp : numpy.ndarray
        Sound speed profile of shape (N, 2): depth (m), speed (m/s).
    frequency : float
        Frequency in Hz
    tx_depth : float
        Source depth in meters
    rx_depths_m : float or array-like
        Receiver depths in meters
    rx_ranges_km : float or array-like
        Receiver ranges in km (converted to meters for arlpy)
    bottom_type : str, optional
        Key into BOTTOM_TYPES; applied if given
    **kwargs
        Passed through to arlpy.uwapm.create_env2d (nbeams, min_angle, ...)

    Returns
    -------
    dict
        arlpy environment dictionary (meters, arlpy convention)
    """
    import arlpy.uwapm as pm

    if np.ndim(depth) == 2:
        bathy = np.asarray(depth, dtype=float).copy()
        bathy[:, 0] *= M_PER_KM
        depth = bathy.tolist()
    else:
        depth = float(depth)

    env = pm.create_env2d(
        depth=depth,
        soundspeed=np.asarray(ssp, dtype=float).tolist(),
        frequency=float(frequency),
        tx_depth=float(tx_depth),
        rx_depth=_as_arlpy_axis(rx_depths_m),
        rx_range=_as_arlpy_axis(np.asarray(rx_ranges_km, dtype=float) * M_PER_KM),
        **kwargs,
    )
    if bottom_type is not None:
        _apply_bottom(env, bottom_type)
    return env


def run_rays(env):
    """Run BELLHOP ray tracing; ray x coordinates are returned in km."""
    import arlpy.uwapm as pm
    rays = pm.compute_rays(env)
    if rays is None:
        return None
    rays = rays.copy()
    rays['ray'] = [np.column_stack([np.asarray(r)[:, 0] / M_PER_KM, np.asarray(r)[:, 1]])
                   for r in rays['ray']]
    return rays


def run_tl(env, mode=None):
    """Run BELLHOP transmission loss; columns (ranges) are returned in km.

    Values are complex pressure as returned by arlpy; TL_dB = -20*log10(|p|).
    """
    import arlpy.uwapm as pm
    tl = pm.compute_transmission_loss(env, mode=mode or pm.incoherent)
    if tl is None:
        return None
    tl = tl.copy()
    tl.columns = np.asarray(tl.columns, dtype=float) / M_PER_KM
    return tl


def run_arrivals(env):
    """Run BELLHOP arrivals. Returns None if BELLHOP finds no eigenrays."""
    import arlpy.uwapm as pm
    try:
        return pm.compute_arrivals(env)
    except ValueError:
        # pandas >= 2: pd.concat([]) raises when there are no arrivals
        return None


def build_deep_ocean_env(
    ssp,
    water_depth=5000.0,
    src_depth=100.0,
    rx_depths=None,
    rx_ranges=None,
    frequency=250.0,
    bottom_type='sand',
    surface_loss_db=0.0,
    n_rays=100,
):
    """Build an arlpy environment dictionary for deep ocean propagation.

    Parameters
    ----------
    ssp : numpy.ndarray
        Sound speed profile array of shape (N, 2): depth (m) and speed (m/s).
        Must span from 0 to at least water_depth.
    water_depth : float
        Water column depth in meters, default 5000
    src_depth : float
        Source depth in meters
    rx_depths : array-like, optional
        Receiver depths in meters; defaults to 10 depths from 0 to water_depth
    rx_ranges : array-like, optional
        Receiver ranges in km; defaults to 50 ranges from 1 to 100 km
    frequency : float
        Acoustic frequency in Hz
    bottom_type : str
        Sediment type: 'mud', 'sand', 'gravel', or 'rock'
    surface_loss_db : float
        Surface reflection loss in dB per bounce (default 0, perfect surface)
    n_rays : int
        Number of ray fan rays for ray tracing runs

    Returns
    -------
    dict
        arlpy environment dictionary ready for pm.compute_rays() etc.
    """
    if rx_depths is None:
        rx_depths = np.linspace(0.0, water_depth, 10)
    if rx_ranges is None:
        rx_ranges = np.linspace(1.0, 100.0, 50)

    env = create_env(water_depth, _clip_ssp(ssp, water_depth), frequency, src_depth,
                     rx_depths, rx_ranges, bottom_type=bottom_type, nbeams=n_rays)
    env['surface_absorption'] = float(surface_loss_db)

    return env


def build_shallow_water_env(
    ssp,
    water_depth=200.0,
    src_depth=30.0,
    rx_depths=None,
    rx_ranges=None,
    frequency=1000.0,
    bottom_type='sand',
    surface_loss_db=0.0,
    n_rays=200,
):
    """Build an arlpy environment dictionary for shallow water propagation.

    Shallow water (<300 m) is characterized by dominant bottom interaction,
    modal interference, and high transmission loss relative to deep ocean.

    Parameters
    ----------
    ssp : numpy.ndarray
        Sound speed profile array of shape (N, 2)
    water_depth : float
        Water column depth in meters, default 200
    src_depth : float
        Source depth in meters
    rx_depths : array-like, optional
        Receiver depths; defaults to 10 depths from 1 to water_depth-1 m
    rx_ranges : array-like, optional
        Receiver ranges in km; defaults to 50 ranges from 0.1 to 20 km
    frequency : float
        Acoustic frequency in Hz, default 1000
    bottom_type : str
        Sediment type
    surface_loss_db : float
        Surface reflection loss in dB per bounce
    n_rays : int
        Number of ray fan rays

    Returns
    -------
    dict
        arlpy environment dictionary
    """
    if rx_depths is None:
        rx_depths = np.linspace(1.0, water_depth - 1.0, 10)
    if rx_ranges is None:
        rx_ranges = np.linspace(0.1, 20.0, 50)

    env = create_env(water_depth, _clip_ssp(ssp, water_depth), frequency, src_depth,
                     rx_depths, rx_ranges, bottom_type=bottom_type, nbeams=n_rays)
    env['surface_absorption'] = float(surface_loss_db)

    return env


def _clip_ssp(ssp, max_depth):
    """Clip or extend an SSP to exactly span [0, max_depth].

    Clips entries deeper than max_depth and adds an endpoint at max_depth
    if needed by linear interpolation.

    Parameters
    ----------
    ssp : numpy.ndarray
        Array of shape (N, 2)
    max_depth : float
        Target maximum depth

    Returns
    -------
    numpy.ndarray
        Clipped/extended SSP
    """
    arr = np.asarray(ssp, dtype=float)
    # Ensure starts at 0
    if arr[0, 0] > 0.0:
        arr = np.vstack([[0.0, arr[0, 1]], arr])

    # Check if we need to extend or clip
    if arr[-1, 0] < max_depth:
        # Extrapolate the last gradient to max_depth
        if len(arr) >= 2:
            dz = arr[-1, 0] - arr[-2, 0]
            dc = arr[-1, 1] - arr[-2, 1]
            grad = dc / dz if dz > 0 else 0.0
        else:
            grad = 0.0
        extra_z = max_depth
        extra_c = arr[-1, 1] + grad * (extra_z - arr[-1, 0])
        arr = np.vstack([arr, [extra_z, extra_c]])
    else:
        # Clip to max_depth
        mask = arr[:, 0] <= max_depth
        arr = arr[mask]
        if arr[-1, 0] < max_depth:
            # Interpolate to exactly max_depth using the original (unclipped) ssp
            # so that the interpolation brackets max_depth correctly.
            from scipy.interpolate import interp1d
            orig = np.asarray(ssp, dtype=float)
            f = interp1d(orig[:, 0], orig[:, 1], fill_value='extrapolate')
            arr = np.vstack([arr, [max_depth, float(f(max_depth))]])

    return arr


def save_env_file(env, scenario_name='scenario', envfiles_dir=None):
    """Write the current arlpy environment to a .env file for inspection.

    The file is saved in the envfiles/ directory with a timestamp in the
    filename so runs do not overwrite each other.

    Parameters
    ----------
    env : dict
        arlpy environment dictionary
    scenario_name : str
        Descriptive name prefix for the filename
    envfiles_dir : str, optional
        Directory to save to; defaults to project envfiles/

    Returns
    -------
    str
        Absolute path to the saved .env file
    """
    import arlpy.uwapm as pm

    if envfiles_dir is None:
        envfiles_dir = _ENVFILES_DIR
    os.makedirs(envfiles_dir, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = scenario_name.replace(' ', '_').replace('/', '-')
    fname_base = os.path.join(envfiles_dir, f'{safe_name}_{ts}')

    # Use arlpy's internal ray computation to generate the file
    # We directly call the env file writer
    pm.compute_rays(env, fname_base=fname_base)

    env_file = fname_base + '.env'
    if os.path.exists(env_file):
        return env_file
    return fname_base + '.*'


def describe_env(env):
    """Print a human-readable summary of the environment parameters.

    Parameters
    ----------
    env : dict
        arlpy environment dictionary
    """
    print("=" * 50)
    print("BELLHOP Environment Summary")
    print("=" * 50)
    print(f"  Frequency:       {env.get('frequency', 'N/A')} Hz")
    print(f"  Water depth:     {env.get('depth', 'N/A')} m")
    print(f"  Source depth:    {env.get('tx_depth', 'N/A')} m")

    rd = env.get('rx_depth', None)
    rr = env.get('rx_range', None)
    if rd is not None:
        rd = np.atleast_1d(rd)
        print(f"  Rx depths:       {rd.min():.1f} to {rd.max():.1f} m ({len(rd)} depths)")
    if rr is not None:
        rr = np.atleast_1d(rr) / M_PER_KM
        print(f"  Rx ranges:       {rr.min():.2f} to {rr.max():.2f} km ({len(rr)} ranges)")

    print(f"  Bottom speed:    {env.get('bottom_soundspeed', 'N/A')} m/s")
    print(f"  Bottom density:  {env.get('bottom_density', 'N/A')} kg/m³")
    print(f"  Bottom absorb:   {env.get('bottom_absorption', 'N/A')} dB/λ")

    ssp = env.get('soundspeed', None)
    if ssp is not None and not isinstance(ssp, (int, float)):
        ssp = np.asarray(ssp)
        if ssp.ndim == 2:
            print(f"  SSP range:       {ssp[:,1].min():.1f} to {ssp[:,1].max():.1f} m/s")
    print("=" * 50)
