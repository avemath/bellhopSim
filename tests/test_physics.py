"""
Regression checks for the physics bugs fixed in this repo.

Run from the project root:
    python -m pytest tests/

Tests marked "bellhop" run BELLHOP and are skipped if bellhop.exe is not
reachable. Everything else is pure Python.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import src.environment_builder as eb
import src.plotting as pl
import src.profiles as prof
import src.utils as ut


@pytest.fixture(scope='module')
def bellhop():
    import src.bellhop_config as bc
    if not os.path.exists(os.path.join(bc.BELLHOP_BIN, 'bellhop.exe')):
        pytest.skip('bellhop.exe not available')
    if not bc.configure(verbose=False):
        pytest.skip('BELLHOP could not be configured')
    import arlpy.uwapm as pm
    return pm


def _munk_env(rx_ranges_km, rx_depths_m=1300.0, **kwargs):
    ssp = eb._clip_ssp(prof.munk_profile(), 5000.0)
    return eb.create_env(5000.0, ssp, 500.0, 1300.0, rx_depths_m, rx_ranges_km,
                         bottom_type='sand', **kwargs)


# ── Bug 1: receiver ranges in km were passed to arlpy, which uses meters ─────

def test_create_env_converts_km_to_meters():
    env = _munk_env(np.linspace(0.5, 100.0, 5))
    assert np.isclose(np.min(env['rx_range']), 500.0)
    assert np.isclose(np.max(env['rx_range']), 100000.0)


def test_create_env_converts_bathymetry_ranges():
    ssp = eb._clip_ssp(prof.munk_profile(z_max=3500.0), 3500.0)
    bathy_km = np.array([[0.0, 200.0], [60.0, 800.0], [120.0, 3500.0]])
    env = eb.create_env(bathy_km, ssp, 500.0, 50.0, 100.0, [1.0, 120.0])
    depth = np.asarray(env['depth'])
    assert np.allclose(depth[:, 0], [0.0, 60000.0, 120000.0])
    assert np.allclose(depth[:, 1], bathy_km[:, 1])


def test_plot_range_reads_meters_as_km():
    env = _munk_env(np.linspace(0.5, 100.0, 5))
    assert np.isclose(pl._get_max_range(env), 100.0)


@pytest.mark.bellhop
def test_bellhop_ranges_come_back_in_km(bellhop):
    env = _munk_env(np.linspace(1.0, 20.0, 10), rx_depths_m=np.linspace(0, 5000, 11),
                    nbeams=30, min_angle=-15, max_angle=15)
    tl = eb.run_tl(env)
    assert np.isclose(float(np.max(tl.columns)), 20.0)
    rays = eb.run_rays(env)
    max_x = max(float(np.asarray(r)[:, 0].max()) for r in rays['ray'])
    assert 19.0 < max_x < 25.0, max_x


# ── Bug 2: bottom density in g/cm3 where arlpy expects kg/m3 ─────────────────

@pytest.mark.parametrize('name', sorted(eb.BOTTOM_TYPES))
def test_bottom_density_is_kg_per_m3(name):
    rho = eb.BOTTOM_TYPES[name]['density']
    assert 1000.0 < rho < 3500.0, f'{name}: {rho} is not a sediment density in kg/m3'
    # arlpy writes density / 1000 (g/cm3) into the .env file; sediment must be
    # denser than seawater (~1.03 g/cm3) or the bottom reflects almost losslessly.
    assert rho / 1000.0 > 1.03


@pytest.mark.bellhop
def test_sand_bottom_removes_steep_multibounce_paths(bellhop):
    # With the old near-zero density, 117 eigenrays arrived at 65 km, the last
    # one 242 s after launch. With a real sand bottom the steep paths die out.
    arr = eb.run_arrivals(_munk_env(65.0, nbeams=200))
    stats = ut.compute_arrival_stats(arr)
    assert stats['n_arrivals'] < 60
    assert float(arr['time_of_arrival'].max()) < 100.0


# ── Bug 3: BER sim ignored channel phase and timing ──────────────────────────

def test_ideal_channel_matches_theory():
    rng = np.random.default_rng(0)
    for snr_db in (0.0, 4.0):
        ber, _, _ = ut.simulate_bpsk_link(np.array([1.0 + 0j]), snr_db, 200000, 4, rng)
        theory = float(ut.bpsk_awgn_ber(snr_db))
        assert abs(ber - theory) / theory < 0.1, (snr_db, ber, theory)


def test_receiver_syncs_to_delayed_rotated_strongest_path():
    # Weak first arrival, strong arrival 30 samples (7.5 symbols) later with a
    # 2-rad phase. The old model (real part of taps, timing from the first
    # arrival) gives BER near 0.5 here; a synced receiver sees only mild ISI.
    h = np.zeros(40, dtype=complex)
    h[0] = 0.3
    h[30] = np.exp(1j * 2.0)
    ber, _, _ = ut.simulate_bpsk_link(h, 30.0, 20000, 4, np.random.default_rng(1))
    assert ber < 1e-3, ber


# ── Single-receiver envs: arlpy's writer rejects 1-element arrays ────────────

def test_single_receiver_is_passed_as_scalar():
    env = _munk_env([65.0], rx_depths_m=[1300.0])
    assert isinstance(env['rx_range'], float)
    assert isinstance(env['rx_depth'], float)


@pytest.mark.bellhop
def test_single_receiver_arrivals_run(bellhop):
    arr = eb.run_arrivals(_munk_env(np.array([65.0]), rx_depths_m=np.array([1300.0]), nbeams=50))
    assert arr is not None and len(arr) > 0
