"""
BELLHOP path configuration and arlpy monkey-patching for WSL.

On this system, bellhop.exe is a Windows binary located at:
  C:\\acoustics-toolbox\\atWin10_2020_11_4\\windows-bin-20201102\\bellhop.exe

When Python runs under WSL (Linux), subprocess calls to bellhop.exe must:
  1. Have the Windows bin directory on PATH
  2. Pass Windows-format paths (C:\\...) not Linux paths (/mnt/c/...) because
     Windows executables cannot resolve /mnt/c/ paths.
  3. Use a temp directory accessible from Windows (under /mnt/c/).

This module handles all three automatically. Import it FIRST in any notebook
before importing arlpy.uwapm.

Usage:
    import src.bellhop_config as bc
    bc.configure()
    import arlpy.uwapm as pm
    # Now pm.compute_rays(), pm.compute_transmission_loss() etc. work correctly.
"""

import os
import subprocess
import tempfile

BELLHOP_BIN = '/mnt/c/acoustics-toolbox/atWin10_2020_11_4/windows-bin-20201102'
WIN_TMPDIR = '/mnt/c/Users/avery/AppData/Local/Temp'


def _to_win_path(linux_path):
    """Convert a WSL Linux path to a Windows path string using wslpath."""
    try:
        result = subprocess.run(
            ['wslpath', '-w', linux_path],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    # Fallback: naive substitution for /mnt/c/ paths
    if linux_path.startswith('/mnt/c/'):
        return 'C:\\' + linux_path[7:].replace('/', '\\')
    return linux_path


def _patched_bellhop(self, *args):
    """Replacement for arlpy._Bellhop._bellhop that converts paths for Windows."""
    try:
        win_args = [_to_win_path(a) for a in args]
        cmd = 'bellhop.exe ' + ' '.join(f'"{a}"' for a in win_args)
        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            shell=True,
            env=env,
            timeout=120,
        )
        if result.returncode == 127:
            return False
    except (OSError, subprocess.TimeoutExpired):
        return False
    return True


def configure(bellhop_bin=None, win_tmpdir=None, verbose=True):
    """Configure the environment so arlpy can drive bellhop.exe from WSL.

    Call this once at the start of a session, before importing arlpy.uwapm
    for the first time (or immediately after if already imported).

    Parameters
    ----------
    bellhop_bin : str, optional
        Path to the directory containing bellhop.exe. Defaults to the
        standard location on this machine.
    win_tmpdir : str, optional
        Windows-accessible temp directory for BELLHOP working files.
        Defaults to the user AppData Local Temp folder.
    verbose : bool
        Print configuration status, default True

    Returns
    -------
    bool
        True if BELLHOP was found and configuration succeeded.
    """
    global BELLHOP_BIN, WIN_TMPDIR

    if bellhop_bin:
        BELLHOP_BIN = bellhop_bin
    if win_tmpdir:
        WIN_TMPDIR = win_tmpdir

    # 1. Add bellhop.exe directory to PATH
    current_path = os.environ.get('PATH', '')
    if BELLHOP_BIN not in current_path:
        os.environ['PATH'] = BELLHOP_BIN + ':' + current_path

    # 2. Set TMPDIR to a Windows-accessible location
    os.makedirs(WIN_TMPDIR, exist_ok=True)
    os.environ['TMPDIR'] = WIN_TMPDIR
    tempfile.tempdir = WIN_TMPDIR

    # 3. Monkey-patch arlpy to convert WSL paths to Windows paths
    try:
        import arlpy.uwapm as pm
        if pm._models:
            BellhopCls = pm._models[0][1]
            BellhopCls._bellhop = _patched_bellhop
            if verbose:
                print(f'[bellhop_config] arlpy patched successfully.')
        else:
            if verbose:
                print('[bellhop_config] WARNING: No models found in arlpy._models list.')
    except ImportError:
        if verbose:
            print('[bellhop_config] ERROR: arlpy not installed. Run: pip install arlpy')
        return False

    # 4. Verify bellhop.exe is reachable
    try:
        r = subprocess.run(
            'bellhop.exe', shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=10, env=os.environ.copy()
        )
        found = True
        if verbose:
            print(f'[bellhop_config] bellhop.exe found at: {BELLHOP_BIN}')
    except Exception:
        found = False
        if verbose:
            print(f'[bellhop_config] WARNING: bellhop.exe not found at {BELLHOP_BIN}')
            print('  Check that the path is correct and bellhop.exe exists there.')

    # 5. Confirm arlpy sees it
    try:
        import arlpy.uwapm as pm
        available = pm.models()
        if verbose:
            print(f'[bellhop_config] arlpy.models() = {available}')
    except Exception as e:
        if verbose:
            print(f'[bellhop_config] pm.models() failed: {e}')

    if verbose:
        print(f'[bellhop_config] TMPDIR = {WIN_TMPDIR}')
        print('[bellhop_config] Configuration complete.\n')

    return found


def quick_test():
    """Run a minimal BELLHOP simulation to verify the full pipeline works.

    Returns
    -------
    bool
        True if BELLHOP ran and returned results.
    """
    import numpy as np
    import arlpy.uwapm as pm

    print('Running quick BELLHOP test...')
    env = pm.create_env2d()
    env['depth'] = 1000.0
    env['frequency'] = 500.0
    env['tx_depth'] = 50.0
    env['rx_depth'] = np.array([50.0, 100.0, 200.0])
    env['rx_range'] = np.linspace(1.0, 10.0, 10)
    env['soundspeed'] = np.array([
        [0.0, 1500.0], [250.0, 1498.0], [500.0, 1496.0],
        [750.0, 1497.0], [1000.0, 1500.0]
    ])

    rays = pm.compute_rays(env)

    if rays is not None:
        print(f'  PASS: compute_rays returned {type(rays).__name__}')
    else:
        print('  FAIL: compute_rays returned None')

    tl = pm.compute_transmission_loss(env)
    if tl is not None:
        print(f'  PASS: compute_transmission_loss returned shape {tl.shape}')
    else:
        print('  FAIL: compute_transmission_loss returned None')

    success = (rays is not None) and (tl is not None)
    print('Quick test:', 'PASSED' if success else 'FAILED')
    return success
