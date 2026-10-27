"""
Export case-study figures and the numbers behind them, headless.

Runs the same scenarios as the notebooks (Munk deep-water defaults from the
interactive notebook, the frequency sweep and comms simulator from the advanced
notebook) through BELLHOP, then writes PNGs and figures.json to docs/figures/.

Usage (from the project root):
    python scripts/export_figures.py

All BELLHOP runs go through src/environment_builder.py (create_env, run_rays,
run_tl, run_arrivals), which take and return ranges in km, and the comms link
uses src/utils.py (simulate_bpsk_link, bpsk_awgn_ber). The notebooks use the
same functions.
"""

import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import src.bellhop_config as bc
import src.profiles as prof
import src.environment_builder as eb
import src.plotting as pl
import src.utils as ut

OUT_DIR = os.path.join(ROOT, 'docs', 'figures')
DPI = 150

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})

# ── Scenario parameters ──────────────────────────────────────────────────────
# Deep-water scenario: interactive notebook defaults (Sections 3 to 5)
DEEP = {
    'ssp': 'Munk (c0=1500 m/s, axis 1300 m, B=1300 m, epsilon=0.00737)',
    'water_depth_m': 5000.0,
    'source_depth_m': 1300.0,
    'frequency_hz': 500.0,
    'bottom_type': 'sand',
    # Narrower fan than the notebook default (+/-80 deg) so the trapped SOFAR
    # paths are legible; Munk rays steeper than ~14 deg reach a boundary.
    'rays': {'max_range_km': 100.0, 'n_rays': 50,
             'min_angle_deg': -20.0, 'max_angle_deg': 20.0},
    'tl': {'mode': 'incoherent', 'min_range_km': 0.5, 'max_range_km': 100.0,
           'n_ranges': 100, 'rx_min_depth_m': 0.0, 'rx_max_depth_m': 4950.0,
           'n_rx_depths': 51, 'n_beams': 200, 'dynamic_range_db': 60.0,
           'cz_threshold_db': 4.0},
    'arrivals': {'rx_range_km': 65.0, 'rx_depth_m': 1300.0, 'n_beams': 200},
}

# Frequency sweep: advanced notebook Section 2
SWEEP = {
    'ssp': 'Munk (defaults)',
    'water_depth_m': 5000.0,
    'source_depth_m': 100.0,
    'frequencies_hz': [100, 500, 1000, 2500, 5000],
    'bottom_type': 'sand',
    'mode': 'incoherent',
    'min_range_km': 0.5,
    'max_range_km': 100.0,
    'n_ranges': 80,
    'n_rx_depths': 51,
    'n_beams': 200,
    'tl_slice_depth_m': 100.0,
}

# Comms simulator: advanced notebook Section 4
COMMS = {
    'ssp': 'shallow_water_profile(total_depth=1000 m)',
    'water_depth_m': 1000.0,
    'source_depth_m': 50.0,
    'rx_range_km': 15.0,
    'rx_depth_m': 100.0,
    'frequency_hz': 3500.0,
    'bottom_type': 'sand',
    'n_beams': 1000,
    'sample_rate_hz': 10000.0,
    'samples_per_symbol': 4,
    'n_symbols_per_trial': 20000,
    'n_trials_per_snr': 5,
    'random_seed': 42,
    'snr_db': list(range(-2, 22, 2)),
}


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {os.path.relpath(path, ROOT)}')
    return name


def deep_water(pm, results):
    d = DEEP
    wd = d['water_depth_m']
    ssp_use = eb._clip_ssp(prof.munk_profile(), wd)
    sofar_depth = prof.get_sofar_depth(ssp_use)
    c_min, c_max = prof.get_speed_range(ssp_use)

    # ── Rays ──
    r = d['rays']
    env_rays = eb.create_env(
        wd, ssp_use, d['frequency_hz'], d['source_depth_m'],
        rx_depths_m=np.linspace(0, wd, 21),
        rx_ranges_km=np.linspace(0.1, r['max_range_km'], 25),
        bottom_type=d['bottom_type'],
        min_angle=r['min_angle_deg'], max_angle=r['max_angle_deg'],
        nbeams=r['n_rays'],
    )
    rays = eb.run_rays(env_rays)
    if rays is None or len(rays) == 0:
        raise RuntimeError('run_rays returned no rays')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'width_ratios': [1, 2.6]})
    pl.plot_ssp(ssp_use, ax=axes[0], title='Munk sound speed profile')
    pl.plot_rays(rays, env_rays, ax=axes[1],
                 title=(f'Ray trace: {len(rays)} rays, source at {d["source_depth_m"]:.0f} m, '
                        f'{d["frequency_hz"]:.0f} Hz, {d["bottom_type"]} bottom'))
    fig.tight_layout()
    _save(fig, 'ssp_and_rays.png')

    # ── TL map ──
    t = d['tl']
    env_tl = eb.create_env(
        wd, ssp_use, d['frequency_hz'], d['source_depth_m'],
        rx_depths_m=np.linspace(t['rx_min_depth_m'], t['rx_max_depth_m'], t['n_rx_depths']),
        rx_ranges_km=np.linspace(t['min_range_km'], t['max_range_km'], t['n_ranges']),
        bottom_type=d['bottom_type'],
        nbeams=t['n_beams'],
    )
    tl = eb.run_tl(env_tl, mode=pm.incoherent)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    _, cb = pl.plot_transmission_loss(
        tl, env_tl, ax=ax, dynamic_range=t['dynamic_range_db'],
        title=(f'Incoherent transmission loss: Munk profile, source at '
               f'{d["source_depth_m"]:.0f} m, {d["frequency_hz"]:.0f} Hz'))
    cb.set_label('Transmission loss (dB re 1 m)')
    fig.tight_layout()
    _save(fig, 'tl_map.png')

    # TL at the source depth, at a few ranges, for reference
    p_abs = np.abs(np.asarray(tl.values, dtype=complex))
    p_abs = np.where(p_abs < 1e-10, np.nan, p_abs)
    tl_db = -20.0 * np.log10(p_abs)
    depths = np.asarray(tl.index, dtype=float)
    ranges_km = np.asarray(tl.columns, dtype=float)
    di = int(np.argmin(np.abs(depths - d['source_depth_m'])))
    cz = ut.find_convergence_zones(tl, threshold_db=t['cz_threshold_db'])

    # ── Arrivals ──
    a = d['arrivals']
    env_arr = eb.create_env(
        wd, ssp_use, d['frequency_hz'], d['source_depth_m'],
        rx_depths_m=a['rx_depth_m'],
        rx_ranges_km=a['rx_range_km'],
        bottom_type=d['bottom_type'],
        nbeams=a['n_beams'],
    )
    arr = eb.run_arrivals(env_arr)
    stats = ut.compute_arrival_stats(arr)
    if stats['n_arrivals'] == 0:
        raise RuntimeError('compute_arrivals returned no valid arrivals')
    spread_ms = stats['delay_spread_ms']
    spread20_ms, n20 = ut.delay_spread_within(arr, 20.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    pl.plot_arrivals(arr, env_arr, ax=ax, title=(
        f'Multipath arrivals at {a["rx_range_km"]:.0f} km range, {a["rx_depth_m"]:.0f} m depth: '
        f'{stats["n_arrivals"]} paths'))
    ax.set_ylim(-0.05, 1.3)
    # Only the spread within 20 dB of the strongest arrival is marked: the full eigenray
    # spread is set by echoes far too weak to matter for a receiver.
    y = 1.16
    ax.annotate('', xy=(spread20_ms, y), xytext=(0, y),
                arrowprops=dict(arrowstyle='<->', color='#0b0b0b', lw=1.5))
    ax.text(spread20_ms + spread_ms * 0.01, y, f'Within 20 dB of strongest ({n20} paths) = {spread20_ms:,.0f} ms',
            ha='left', va='center', fontsize=11, color='#0b0b0b')
    ax.axvline(spread20_ms, color='#52514e', linestyle=':', linewidth=1)
    bounce_pal = [pl.WATER_COLOR, '#27ae60', '#e67e22', '#c0392b']
    handles = [Line2D([0], [0], color=c, marker='o', lw=2, label=lbl) for c, lbl in
               zip(bounce_pal, ['0 boundary bounces', '1 bounce', '2 bounces', '3+ bounces'])]
    ax.legend(handles=handles, loc='center right', title='Surface + bottom bounces')
    fig.tight_layout()
    _save(fig, 'arrivals.png')

    results['deep_water'] = {
        'parameters': d,
        'sofar_axis_depth_m': sofar_depth,
        'sound_speed_range_m_s': [c_min, c_max],
        'rays_traced': int(len(rays)),
        'tl_at_source_depth_db': {
            'depth_m': float(depths[di]),
            **{f'{rk:g}_km': float(tl_db[di, int(np.argmin(np.abs(ranges_km - rk)))])
               for rk in (10, 30, 65, 100)},
        },
        'convergence_zones_detected_km': cz,
        'convergence_zone_detection_depth_m': float(depths[len(depths) // 2]),
        'arrivals': {
            'n_arrivals': stats['n_arrivals'],
            'delay_spread_ms': spread_ms,
            'delay_spread_definition': 'latest minus earliest eigenray, all eigenrays (src/utils.py)',
            'coherence_bandwidth_hz': stats['coherence_bandwidth_hz'],
            'coherence_bandwidth_definition': '1 / delay spread',
            'n_arrivals_within_20db': n20,
            'delay_spread_within_20db_ms': spread20_ms,
            'coherence_bandwidth_within_20db_hz': 1000.0 / spread20_ms if spread20_ms > 0 else None,
            'strongest_to_second_amplitude_ratio': stats['dominance_ratio'],
        },
    }


def frequency_sweep(pm, results):
    s = SWEEP
    wd = s['water_depth_m']
    ssp_use = eb._clip_ssp(prof.munk_profile(z_max=wd), wd)
    rx_depths = np.linspace(0, wd, s['n_rx_depths'])
    rx_ranges_km = np.linspace(s['min_range_km'], s['max_range_km'], s['n_ranges'])

    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
    markers = ['o', 's', '^', 'D', 'v']
    fig, ax = plt.subplots(figsize=(12, 5.5))
    summary = {}
    for i, freq in enumerate(s['frequencies_hz']):
        env = eb.create_env(
            wd, ssp_use, float(freq), s['source_depth_m'],
            rx_depths_m=rx_depths, rx_ranges_km=rx_ranges_km,
            bottom_type=s['bottom_type'], nbeams=s['n_beams'],
        )
        tl = eb.run_tl(env, mode=pm.incoherent)
        depths = np.asarray(tl.index, dtype=float)
        rk = np.asarray(tl.columns, dtype=float)
        di = int(np.argmin(np.abs(depths - s['tl_slice_depth_m'])))
        p_abs = np.abs(np.asarray(tl.iloc[di, :], dtype=complex))
        p_abs = np.where(p_abs < 1e-10, np.nan, p_abs)
        tl_slice = -20.0 * np.log10(p_abs)
        if not np.isfinite(tl_slice).any():
            raise RuntimeError(f'No finite TL at {freq} Hz')
        ax.plot(rk, tl_slice, color=colors[i], linewidth=2, marker=markers[i],
                markersize=5, markevery=8, label=f'{freq} Hz')
        last = np.where(np.isfinite(tl_slice))[0][-1]
        ax.annotate(f'{freq} Hz', xy=(rk[last], tl_slice[last]), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=10, color='#0b0b0b')
        summary[f'{freq}_hz'] = {
            f'tl_db_at_{r:g}_km': float(tl_slice[int(np.argmin(np.abs(rk - r)))])
            for r in (10, 50, 100)
        }
        print(f'  sweep: {freq} Hz done')

    ax.invert_yaxis()
    ax.set_xlim(0, s['max_range_km'] * 1.08)
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Transmission loss (dB re 1 m)')
    ax.set_title(f'Incoherent TL vs range at {s["tl_slice_depth_m"]:.0f} m depth: '
                 f'Munk profile, source at {s["source_depth_m"]:.0f} m')
    ax.grid(True, alpha=0.25)
    ax.legend(title='Frequency', loc='lower left')
    fig.tight_layout()
    _save(fig, 'frequency_sweep.png')

    results['frequency_sweep'] = {
        'parameters': s,
        'rx_depth_used_m': float(depths[di]),
        'tl_by_frequency': summary,
    }


def comms_link(pm, results):
    c = COMMS
    wd = c['water_depth_m']
    ssp_use = eb._clip_ssp(prof.shallow_water_profile(total_depth=wd), wd)
    env = eb.create_env(
        wd, ssp_use, c['frequency_hz'], c['source_depth_m'],
        rx_depths_m=c['rx_depth_m'],
        rx_ranges_km=c['rx_range_km'],
        bottom_type=c['bottom_type'],
        nbeams=c['n_beams'],
    )
    arr = eb.run_arrivals(env)
    stats = ut.compute_arrival_stats(arr)
    if stats['n_arrivals'] == 0:
        raise RuntimeError('Comms scenario returned no arrivals')
    fs = c['sample_rate_hz']
    ir = ut.channel_to_impulse_response(arr, fs=fs)
    ir_norm = ir / (np.abs(ir).max() + 1e-12)
    spread20_ms, n20 = ut.delay_spread_within(arr, 20.0)

    # Shared link model (src/utils.py, same as the advanced notebook):
    # receiver synced to the strongest arrival, no equalizer.
    rng = np.random.default_rng(c['random_seed'])
    sps, n_sym = c['samples_per_symbol'], c['n_symbols_per_trial']

    def ber_at(snr_db, h):
        errors = sum(round(ut.simulate_bpsk_link(h, snr_db, n_sym, sps, rng)[0] * n_sym)
                     for _ in range(c['n_trials_per_snr']))
        return errors / (n_sym * c['n_trials_per_snr'])

    snr = np.array(c['snr_db'], dtype=float)
    ber_ch = np.array([ber_at(x, ir_norm) for x in snr])
    ber_awgn_sim = np.array([ber_at(x, np.array([1.0 + 0j])) for x in snr])
    ber_theory = ut.bpsk_awgn_ber(snr)
    n_bits = n_sym * c['n_trials_per_snr']

    fig, ax = plt.subplots(figsize=(9, 6))
    snr_fine = np.linspace(snr[0], snr[-1], 200)
    ax.semilogy(snr_fine, ut.bpsk_awgn_ber(snr_fine), '--',
                color='#52514e', linewidth=2, label='Theoretical BPSK, AWGN only')
    nz = ber_ch > 0
    ax.semilogy(snr[nz], ber_ch[nz], 'o-', color='#2a78d6', linewidth=2, markersize=8,
                label=f'Simulated BPSK, BELLHOP multipath channel ({stats["n_arrivals"]} paths)')
    ax.set_ylim(max(0.5 / n_bits, 1e-6), 1)
    ax.set_xlabel('SNR per sample, on total received power (dB)')
    ax.set_ylabel('Bit error rate')
    ax.set_title(f'BPSK BER vs SNR: {wd:.0f} m water, {c["rx_range_km"]:.0f} km range, '
                 f'{c["frequency_hz"]:.0f} Hz\nReceiver synced to strongest path, no equalizer')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc='lower left')
    fig.tight_layout()
    _save(fig, 'ber_vs_snr.png')

    symbol_rate = fs / sps
    results['comms'] = {
        'parameters': c,
        'link_model': ('complex baseband NRZ BPSK through the complex BELLHOP impulse '
                       'response; receiver synchronized to the strongest arrival (delay and '
                       'phase); one sample per symbol; sign detector; no equalizer'),
        'snr_definition': 'received signal power (after channel) / complex noise power, per sample',
        'theory_curve': '0.5*erfc(sqrt(SNR)), BPSK in AWGN with this SNR definition',
        'bits_per_snr_point': n_bits,
        'channel': {
            'n_arrivals': stats['n_arrivals'],
            'delay_spread_ms': stats['delay_spread_ms'],
            'coherence_bandwidth_hz': stats['coherence_bandwidth_hz'],
            'n_arrivals_within_20db': n20,
            'delay_spread_within_20db_ms': spread20_ms,
            'impulse_response_length_samples': int(len(ir)),
            'symbol_rate_baud': symbol_rate,
            'isi_span_symbols': int(stats['delay_spread_ms'] / 1000.0 * symbol_rate),
        },
        'ber': [
            {'snr_db': float(x), 'ber_bellhop_channel': float(b),
             'ber_awgn_simulated': float(a), 'ber_awgn_theory': float(t)}
            for x, b, a, t in zip(snr, ber_ch, ber_awgn_sim, ber_theory)
        ],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Configuring BELLHOP...')
    if not bc.configure(verbose=False):
        sys.exit('bellhop.exe not reachable; check BELLHOP_BIN in src/bellhop_config.py')
    import arlpy.uwapm as pm

    results = {
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'generator': 'scripts/export_figures.py',
        'figures': ['ssp_and_rays.png', 'tl_map.png', 'arrivals.png',
                    'ber_vs_snr.png', 'frequency_sweep.png'],
        'dpi': DPI,
    }
    print('Deep-water scenario (rays, TL, arrivals)...')
    deep_water(pm, results)
    print('Comms simulator...')
    comms_link(pm, results)
    print('Frequency sweep...')
    frequency_sweep(pm, results)

    path = os.path.join(OUT_DIR, 'figures.json')
    with open(path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'  wrote {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    main()
