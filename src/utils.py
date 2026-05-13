"""
Utility functions for post-processing BELLHOP simulation results.

Includes delay spread estimation, SNR computation, convergence zone
detection, impulse response generation, and result export.
"""

import os
import json
import numpy as np
from datetime import datetime

_OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')


def estimate_delay_spread(arr_data):
    """Compute the maximum multipath delay spread across all receivers.

    Delay spread is the difference between the earliest and latest
    significant arrival times. It directly determines the coherence
    bandwidth and ISI length in communication systems.

    Parameters
    ----------
    arr_data : pandas.DataFrame
        Output from pm.compute_arrivals().

    Returns
    -------
    float
        Delay spread in seconds. Returns 0 if no arrivals.
    """
    if arr_data is None or len(arr_data) == 0:
        return 0.0

    if 'time_of_arrival' not in arr_data.columns:
        return 0.0

    times = np.asarray(arr_data['time_of_arrival'], dtype=float)
    times = times[np.isfinite(times)]

    if len(times) < 2:
        return 0.0

    return float(times.max() - times.min())


def compute_snr_estimate(tl_db, src_level_db=200.0, noise_level_db=60.0, directivity_index_db=0.0):
    """Passive/active sonar equation SNR estimate.

    SNR = SL - TL - NL + DI

    where:
      SL = source level (dB re 1 µPa @ 1 m)
      TL = transmission loss (dB, positive)
      NL = ambient noise level (dB re 1 µPa)
      DI = array directivity index (dB)

    Parameters
    ----------
    tl_db : float or numpy.ndarray
        Transmission loss value(s) in dB (positive convention)
    src_level_db : float
        Source level in dB re 1 µPa @ 1 m, default 200
    noise_level_db : float
        Ambient noise level in dB re 1 µPa, default 60
    directivity_index_db : float
        Receiver directivity index in dB, default 0

    Returns
    -------
    float or numpy.ndarray
        SNR in dB at the receiver location(s)
    """
    return src_level_db - np.abs(tl_db) - noise_level_db + directivity_index_db


def find_convergence_zones(tl_data, threshold_db=5.0, reference_depth_idx=None):
    """Identify ranges where transmission loss is anomalously low (convergence zones).

    A convergence zone is detected where TL dips more than threshold_db
    below the local smoothed TL trend, indicating anomalously low loss
    characteristic of deep-water convergence zone propagation.

    Parameters
    ----------
    tl_data : pandas.DataFrame
        TL DataFrame from pm.compute_transmission_loss().
        Index = depths (m), columns = ranges (km).
    threshold_db : float
        Minimum dB dip below local trend to flag as a CZ, default 5
    reference_depth_idx : int, optional
        Depth index to use for CZ detection; defaults to middle depth

    Returns
    -------
    list of float
        Ranges (km) of identified convergence zone centers
    """
    if tl_data is None or tl_data.empty:
        return []

    import pandas as pd
    from scipy.signal import find_peaks

    depths = np.asarray(tl_data.index, dtype=float)
    ranges_km = np.asarray(tl_data.columns, dtype=float)

    if reference_depth_idx is None:
        reference_depth_idx = len(depths) // 2

    # arlpy returns complex pressure; convert to TL dB = -20*log10(|p|)
    pressure_abs = np.abs(np.asarray(tl_data.iloc[reference_depth_idx, :], dtype=complex))
    pressure_abs = np.where(pressure_abs < 1e-10, np.nan, pressure_abs)
    tl_slice = -20.0 * np.log10(pressure_abs)

    # Remove NaN/zero
    valid = np.isfinite(tl_slice) & (tl_slice > 0)
    if not valid.any():
        return []

    # Smooth the TL slice to get background trend (running median)
    window = max(5, len(ranges_km) // 20)
    smoothed = pd.Series(tl_slice).rolling(window, center=True, min_periods=1).median().values

    # Convergence zones are where TL is significantly below the smoothed trend
    dip = smoothed - tl_slice  # positive = TL lower than trend = better propagation
    peaks, props = find_peaks(dip, height=threshold_db, distance=max(3, len(ranges_km)//30))

    cz_ranges = [float(ranges_km[p]) for p in peaks]
    return cz_ranges


def _amp_col(df):
    """Return the amplitude column name used by this arlpy version."""
    if 'arrival_amplitude' in df.columns:
        return 'arrival_amplitude'
    return 'amplitude'


def channel_to_impulse_response(arr_data, fs=10000.0, duration=None, rx_range_km=None, rx_depth_m=None):
    """Convert BELLHOP arrivals to a discrete-time channel impulse response.

    Each arrival is placed at the nearest sample to its arrival time,
    with complex amplitude. The result is a causal filter suitable for
    convolving with a communication signal to simulate acoustic multipath.

    Parameters
    ----------
    arr_data : pandas.DataFrame
        Arrivals from pm.compute_arrivals().
    fs : float
        Output sample rate in Hz, default 10000
    duration : float, optional
        IR duration in seconds. Defaults to delay_spread * 2 + 50 ms.
    rx_range_km : float, optional
        Select a specific receiver range for multi-receiver data
    rx_depth_m : float, optional
        Select a specific receiver depth for multi-receiver data

    Returns
    -------
    numpy.ndarray
        Complex-valued impulse response of length int(duration * fs)
    """
    if arr_data is None or len(arr_data) == 0:
        n_samples = int((duration or 0.1) * fs)
        return np.zeros(n_samples, dtype=complex)

    df = arr_data.copy()

    # Filter by range/depth if specified and data is multi-indexed
    if rx_range_km is not None or rx_depth_m is not None:
        try:
            import pandas as pd
            if isinstance(df.index, pd.MultiIndex):
                ranges = df.index.get_level_values(0)
                ddepths = df.index.get_level_values(1)
                mask = np.ones(len(df), dtype=bool)
                if rx_range_km is not None:
                    closest_r = float(ranges[np.argmin(np.abs(ranges - rx_range_km))])
                    mask &= (ranges == closest_r)
                if rx_depth_m is not None:
                    closest_d = float(ddepths[np.argmin(np.abs(ddepths - rx_depth_m))])
                    mask &= (ddepths == closest_d)
                df = df[mask]
        except Exception:
            pass

    amp_col = _amp_col(df)
    if 'time_of_arrival' not in df.columns or amp_col not in df.columns:
        n_samples = int((duration or 0.1) * fs)
        return np.zeros(n_samples, dtype=complex)

    times = np.asarray(df['time_of_arrival'], dtype=float)
    amps = np.asarray(df[amp_col], dtype=complex)

    # Remove invalid entries
    valid = np.isfinite(times) & np.isfinite(np.abs(amps))
    times = times[valid]
    amps = amps[valid]

    if len(times) == 0:
        n_samples = int((duration or 0.1) * fs)
        return np.zeros(n_samples, dtype=complex)

    # Make times relative to first arrival
    t0 = times.min()
    times_rel = times - t0

    if duration is None:
        duration = times_rel.max() * 2.0 + 0.05  # 50 ms padding

    n_samples = max(1, int(np.ceil(duration * fs)))
    ir = np.zeros(n_samples, dtype=complex)

    # Place arrivals at nearest sample
    sample_indices = np.round(times_rel * fs).astype(int)
    for idx, amp in zip(sample_indices, amps):
        if 0 <= idx < n_samples:
            ir[idx] += amp

    return ir


def compute_arrival_stats(arr_data):
    """Compute summary statistics for a set of BELLHOP arrivals.

    Parameters
    ----------
    arr_data : pandas.DataFrame
        Output from pm.compute_arrivals()

    Returns
    -------
    dict with keys:
        n_arrivals : int
        delay_spread_ms : float
        strongest_amplitude : float
        second_strongest : float
        dominance_ratio : float
        coherence_bandwidth_hz : float
    """
    stats = {
        'n_arrivals': 0,
        'delay_spread_ms': 0.0,
        'strongest_amplitude': 0.0,
        'second_strongest': 0.0,
        'dominance_ratio': float('inf'),
        'coherence_bandwidth_hz': float('inf'),
    }

    if arr_data is None or len(arr_data) == 0:
        return stats

    amp_col = _amp_col(arr_data)
    if amp_col not in arr_data.columns or 'time_of_arrival' not in arr_data.columns:
        return stats

    amps = np.abs(np.asarray(arr_data[amp_col], dtype=complex))
    times = np.asarray(arr_data['time_of_arrival'], dtype=float)

    valid = np.isfinite(amps) & np.isfinite(times) & (amps > 0)
    amps = amps[valid]
    times = times[valid]

    if len(amps) == 0:
        return stats

    stats['n_arrivals'] = int(len(amps))
    delay_spread = float(times.max() - times.min()) if len(times) > 1 else 0.0
    stats['delay_spread_ms'] = delay_spread * 1000.0

    sorted_amps = np.sort(amps)[::-1]
    stats['strongest_amplitude'] = float(sorted_amps[0])
    if len(sorted_amps) > 1:
        stats['second_strongest'] = float(sorted_amps[1])
        stats['dominance_ratio'] = float(sorted_amps[0] / sorted_amps[1]) if sorted_amps[1] > 0 else float('inf')

    if delay_spread > 0:
        stats['coherence_bandwidth_hz'] = float(1.0 / delay_spread)
    else:
        stats['coherence_bandwidth_hz'] = float('inf')

    return stats


def export_results(env, results, filename=None, outputs_dir=None):
    """Save environment and simulation results to a JSON file.

    The JSON file captures all environment parameters and key results
    for later analysis or reproducibility.

    Parameters
    ----------
    env : dict
        arlpy environment dictionary
    results : dict
        Simulation results (scalars and short arrays; DataFrames are summarized)
    filename : str, optional
        Output filename (without extension); auto-generated with timestamp if None
    outputs_dir : str, optional
        Directory to save to; defaults to project outputs/

    Returns
    -------
    str
        Absolute path to the saved JSON file
    """
    if outputs_dir is None:
        outputs_dir = _OUTPUTS_DIR
    os.makedirs(outputs_dir, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'bellhop_results_{ts}'

    filepath = os.path.join(outputs_dir, filename + '.json')

    def _serialize(obj):
        """Convert numpy arrays and DataFrames to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.complexfloating,)):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                return {'type': 'DataFrame', 'shape': list(obj.shape),
                        'columns_range': [float(obj.columns.min()), float(obj.columns.max())],
                        'index_range': [float(obj.index.min()), float(obj.index.max())]}
        except ImportError:
            pass
        return str(obj)

    # Build serializable env
    env_serial = {}
    for k, v in env.items():
        try:
            env_serial[k] = _serialize(v) if isinstance(v, (np.ndarray, np.generic)) else v
        except Exception:
            env_serial[k] = str(v)

    export_data = {
        'timestamp': datetime.now().isoformat(),
        'environment': env_serial,
        'results': {k: _serialize(v) if isinstance(v, (np.ndarray, np.generic)) else v
                    for k, v in results.items()},
    }

    with open(filepath, 'w') as fh:
        json.dump(export_data, fh, indent=2, default=_serialize)

    print(f'Results saved to: {filepath}')
    return filepath


def wsl_safe_tmpdir():
    """Return a temporary directory path that Windows bellhop.exe can access.

    Returns
    -------
    str
        Path to a Windows-accessible temp directory under /mnt/c/
    """
    candidates = [
        '/mnt/c/Users/avery/AppData/Local/Temp/bellhopSim',
        '/mnt/c/Windows/Temp/bellhopSim',
        '/mnt/c/Temp/bellhopSim',
    ]
    for c in candidates:
        try:
            os.makedirs(c, exist_ok=True)
            # Test write access
            test_file = os.path.join(c, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return c
        except (OSError, PermissionError):
            continue

    # Fallback: use Linux /tmp and hope WSL interop handles it
    import tempfile
    return tempfile.gettempdir()
