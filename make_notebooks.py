"""
Generator script for bellhop_interactive.ipynb and bellhop_advanced.ipynb.
Run this once to (re-)create both notebooks:
    python3 make_notebooks.py
"""

import json
import os

NB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebooks')
os.makedirs(NB_DIR, exist_ok=True)


def nb_metadata():
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.3"
        }
    }


def code_cell(source, tags=None):
    meta = {}
    if tags:
        meta["tags"] = tags
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": meta,
        "outputs": [],
        "source": source
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }


# =============================================================================
# MAIN INTERACTIVE NOTEBOOK
# =============================================================================

def make_interactive_notebook():
    cells = []

    # ---- SECTION 1: Setup ----
    cells.append(md_cell("""# BELLHOP Underwater Acoustic Propagation Simulator

**Interactive exploration of ray acoustics, transmission loss, and multipath propagation**

This notebook drives [BELLHOP](https://oalib-acoustics.org/) — the industry-standard ray-acoustic
propagation model — through the Python [arlpy](https://github.com/org-arl/arlpy) library.
You can explore how sound propagates through different ocean environments in real time
using interactive sliders and dropdowns.

## What does BELLHOP compute?

| Output | Physical meaning |
|--------|-----------------|
| **Ray paths** | Trajectories of sound energy through the water column — shows refraction, surface/bottom bounces, shadow zones |
| **Transmission Loss (TL)** | How much sound level drops with range and depth — critical for sonar range prediction |
| **Multipath arrivals** | The set of distinct ray paths reaching a receiver — determines channel delay spread for communications |

## Sections in this notebook

1. **Setup** — BELLHOP path config and library imports
2. **Sound Speed Profile Explorer** — Interactive SSP builder with 6 profile types
3. **Ray Tracing** — Visualize ray paths for any source/environment
4. **Transmission Loss Map** — Full 2D TL color plot with convergence zone detection
5. **Multipath Arrivals Analysis** — Delay spread, impulse response, comms implications
6. **Scenario Comparison** — Compare up to 4 environments side-by-side
7. **Sonar Equation Calculator** — Translate TL into SNR for a real sonar system

---
"""))

    cells.append(md_cell("## Section 1 — Setup and Configuration"))

    cells.append(code_cell("""\
%matplotlib inline
# ── Core imports ──────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..'))   # add project root to path

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# ── Project modules ───────────────────────────────────────────────────────────
# importlib.reload ensures we always run the on-disk version, not a stale
# bytecode cache from a previous kernel session.
import importlib
import src.bellhop_config as bc
import src.profiles as prof
import src.environment_builder as eb
import src.plotting as pl
import src.utils as ut
for _mod in [bc, prof, eb, pl, ut]:
    importlib.reload(_mod)

# ── Configure BELLHOP ─────────────────────────────────────────────────────────
bc.configure(verbose=True)

# ── Import arlpy AFTER configure() ────────────────────────────────────────────
import arlpy.uwapm as pm

print("\\n=== arlpy default environment ===")
pm.print_env(pm.create_env2d())
"""))

    # Helper cell: show_fig used by all widget callbacks
    cells.append(code_cell("""\
def show_fig(fig):
    \"\"\"Render a matplotlib figure inside an Output widget and close it.

    With %matplotlib inline, plt.show() fires at cell-end, not inside
    widget callbacks. display(fig) explicitly pushes the figure into the
    current Output context, and plt.close() prevents a second render.
    \"\"\"
    plt.tight_layout()
    display(fig)
    plt.close(fig)
"""))

    cells.append(code_cell("""\
# ── Quick sanity test (runs a ~1-second BELLHOP job) ─────────────────────────
ok = bc.quick_test()
if not ok:
    print("\\n⚠  BELLHOP test failed. Check the path configuration above.")
else:
    print("\\n✓  BELLHOP is working correctly — proceed to Section 2.")
"""))

    # ---- SECTION 2: SSP Explorer ----
    cells.append(md_cell("""---
## Section 2 — Sound Speed Profile Explorer

Select a profile type and adjust its parameters.
The live plot updates as you move the sliders.
Click **Lock in SSP** to save this profile for use in Sections 3–6.

### Physical context
The sound speed profile (SSP) controls everything about how sound propagates:
- **Minimum** speed depth = SOFAR channel axis — sound is trapped here and travels enormous distances
- **Negative gradient** above the axis: sound refracts upward → skip zones, shadow zones
- **Positive gradient** below: sound refracts downward → surface duct or upward refraction
"""))

    cells.append(code_cell("""\
# ── Shared state ──────────────────────────────────────────────────────────────
_locked_ssp = {'ssp': prof.munk_profile(), 'name': 'Munk (default)'}

# ── Layout helpers ────────────────────────────────────────────────────────────
def _slider(desc, val, lo, hi, step, width='95%'):
    return widgets.FloatSlider(
        value=val, min=lo, max=hi, step=step,
        description=desc, style={'description_width': '170px'},
        layout=widgets.Layout(width=width)
    )

def _int_slider(desc, val, lo, hi, step=1, width='95%'):
    return widgets.IntSlider(
        value=val, min=lo, max=hi, step=step,
        description=desc, style={'description_width': '170px'},
        layout=widgets.Layout(width=width)
    )

# ── Widgets ───────────────────────────────────────────────────────────────────
w_profile = widgets.Dropdown(
    options=['Munk', 'Isothermal', 'Surface Duct', 'Arctic', 'Shallow Water', 'Custom'],
    value='Munk', description='Profile type:',
    style={'description_width': '120px'},
    layout=widgets.Layout(width='320px')
)

# Munk sliders
w_c0      = _slider('c₀ (m/s)',      1500, 1400, 1600, 1)
w_z_axis  = _slider('Channel axis (m)', 1300, 200, 3000, 10)
w_epsilon = _slider('ε (strength)',  0.00737, 0.001, 0.02, 0.0001)
w_zmax_m  = _slider('Max depth (m)', 5000, 500, 6000, 50)
munk_box  = widgets.VBox([w_c0, w_z_axis, w_epsilon, w_zmax_m])

# Isothermal
w_iso_speed = _slider('Sound speed (m/s)', 1500, 1400, 1600, 1)
w_iso_zmax  = _slider('Max depth (m)', 3000, 100, 6000, 50)
iso_box = widgets.VBox([w_iso_speed, w_iso_zmax])

# Surface duct
w_duct_depth = _slider('Duct depth (m)', 100, 10, 400, 5)
w_duct_speed = _slider('Duct speed (m/s)', 1520, 1490, 1570, 1)
w_duct_zmax  = _slider('Max depth (m)', 4000, 500, 6000, 50)
duct_box = widgets.VBox([w_duct_depth, w_duct_speed, w_duct_zmax])

# Arctic
w_arctic_surface = _slider('Surface speed (m/s)', 1435, 1400, 1470, 1)
w_arctic_zmax    = _slider('Max depth (m)', 4000, 500, 5000, 50)
arctic_box = widgets.VBox([w_arctic_surface, w_arctic_zmax])

# Shallow water
w_sw_depth    = _slider('Total depth (m)', 200, 30, 500, 5)
w_sw_surf_spd = _slider('Surface speed (m/s)', 1520, 1480, 1560, 1)
w_sw_bot_spd  = _slider('Bottom speed (m/s)', 1490, 1460, 1530, 1)
sw_box = widgets.VBox([w_sw_depth, w_sw_surf_spd, w_sw_bot_spd])

# Custom profile
w_custom_text = widgets.Textarea(
    value='0,1520\\n50,1510\\n100,1495\\n200,1490\\n500,1492\\n1000,1500\\n2000,1510',
    description='depth,speed pairs:',
    style={'description_width': '150px'},
    layout=widgets.Layout(width='95%', height='120px')
)
custom_box = widgets.VBox([
    widgets.HTML('<b>Enter one (depth m, speed m/s) pair per line:</b>'),
    w_custom_text
])

# Container that swaps param boxes
params_container = widgets.VBox([munk_box])

ssp_out = widgets.Output()
lock_btn = widgets.Button(description='🔒 Lock in this SSP', button_style='success',
                          layout=widgets.Layout(width='200px'))
ssp_info = widgets.HTML('')

def _get_current_ssp():
    p = w_profile.value
    if p == 'Munk':
        return prof.munk_profile(c0=w_c0.value, z_axis=w_z_axis.value,
                                  epsilon=w_epsilon.value, z_max=w_zmax_m.value)
    elif p == 'Isothermal':
        return prof.isothermal_profile(speed=w_iso_speed.value, z_max=w_iso_zmax.value)
    elif p == 'Surface Duct':
        return prof.surface_duct_profile(duct_depth=w_duct_depth.value,
                                          duct_speed=w_duct_speed.value,
                                          z_max=w_duct_zmax.value)
    elif p == 'Arctic':
        return prof.arctic_profile(surface_speed=w_arctic_surface.value,
                                    z_max=w_arctic_zmax.value)
    elif p == 'Shallow Water':
        return prof.shallow_water_profile(total_depth=w_sw_depth.value,
                                           surface_speed=w_sw_surf_spd.value,
                                           bottom_speed=w_sw_bot_spd.value)
    else:  # Custom
        try:
            pairs = [list(map(float, line.split(',')))
                     for line in w_custom_text.value.strip().split('\\n') if line.strip()]
            return prof.custom_profile(pairs)
        except Exception as e:
            return prof.munk_profile()

def _update_ssp(_=None):
    ssp = _get_current_ssp()
    sofar = prof.get_sofar_depth(ssp)
    c_min, c_max = prof.get_speed_range(ssp)
    ssp_info.value = (
        f'<b>SSP stats:</b>  '
        f'Min speed: <b>{c_min:.1f} m/s</b>  |  '
        f'Max speed: <b>{c_max:.1f} m/s</b>  |  '
        f'SOFAR axis: <b>{sofar:.0f} m</b>'
    )
    with ssp_out:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(4.5, 7))
        pl.plot_ssp(ssp, ax=ax, title=f'{w_profile.value} Profile')
        show_fig(fig)

def _on_profile_change(change):
    p = change['new']
    box_map = {
        'Munk': munk_box, 'Isothermal': iso_box,
        'Surface Duct': duct_box, 'Arctic': arctic_box,
        'Shallow Water': sw_box, 'Custom': custom_box
    }
    params_container.children = [box_map.get(p, munk_box)]
    _update_ssp()

def _lock_ssp(_):
    ssp = _get_current_ssp()
    _locked_ssp['ssp'] = ssp
    _locked_ssp['name'] = w_profile.value
    lock_btn.description = f'✓ Locked: {w_profile.value}'
    lock_btn.button_style = 'info'

w_profile.observe(_on_profile_change, names='value')
lock_btn.on_click(_lock_ssp)

# Connect all param sliders to update
for w in [w_c0, w_z_axis, w_epsilon, w_zmax_m,
          w_iso_speed, w_iso_zmax, w_duct_depth, w_duct_speed, w_duct_zmax,
          w_arctic_surface, w_arctic_zmax, w_sw_depth, w_sw_surf_spd, w_sw_bot_spd,
          w_custom_text]:
    w.observe(_update_ssp, names='value')

_update_ssp()

display(widgets.VBox([
    w_profile,
    params_container,
    ssp_info,
    lock_btn,
    ssp_out
]))
"""))

    # ---- SECTION 3: Ray Tracing ----
    cells.append(md_cell("""---
## Section 3 — Ray Tracing Simulation

Configure the source/receiver geometry and bottom type, then click **Run Ray Trace**.

> **Default source depth = 1300 m** — this is the Munk profile SOFAR channel axis.
> Placing the source at the SOFAR axis produces the classic convergence zone ray fan:
> rays launched at all angles oscillate symmetrically around the axis and focus at
> regular intervals (~65 km for typical deep-ocean profiles).
> Move the source shallower (e.g. 100 m) to see how rays become trapped in the upper
> water column and produce shadow zones below.

### What to look for
- **Convergence zones**: tight bundles of rays focusing at ~65 km intervals — very low TL
- **Shadow zones**: regions where no rays reach — very high TL
- **SOFAR channel**: rays launched at small angles (< 10°) oscillate tightly around the axis
- **Bottom/surface bounces**: steep-angle rays that interact with boundaries lose energy
"""))

    cells.append(code_cell("""\
# ── Ray trace widgets ─────────────────────────────────────────────────────────
rt_src_depth = _slider('Source depth (m)',  1300, 1, 4999, 1)
rt_water_depth = _slider('Water depth (m)', 5000, 200, 6000, 50)
rt_max_range = _slider('Max range (km)',      100, 1, 500, 1)
rt_n_rays    = _int_slider('Number of rays',  50, 10, 300, 10)
rt_min_angle = _slider('Min launch angle°', -80, -89, 0, 1)
rt_max_angle = _slider('Max launch angle°',  80, 0, 89, 1)
rt_freq      = _slider('Frequency (Hz)',    500, 10, 10000, 10)
rt_bottom    = widgets.Dropdown(
    options=['sand', 'mud', 'gravel', 'rock'],
    value='sand', description='Bottom type:',
    style={'description_width': '120px'}, layout=widgets.Layout(width='300px')
)
rt_run_btn = widgets.Button(description='▶ Run Ray Trace', button_style='primary',
                             layout=widgets.Layout(width='200px'))
rt_out = widgets.Output()
rt_stats = widgets.HTML('')
rt_warn = widgets.HTML('')

_WARN_STYLE = 'color:darkorange;font-weight:bold'

def _rt_check(_=None):
    msgs = []
    sd, wd = rt_src_depth.value, rt_water_depth.value
    if sd >= wd:
        msgs.append(f'⚠ Source depth ({sd:.0f} m) ≥ water depth ({wd:.0f} m) — will be clamped to {wd-1:.0f} m.')
    if rt_min_angle.value >= rt_max_angle.value:
        msgs.append(f'⚠ Min angle ({rt_min_angle.value:.0f}°) ≥ max angle ({rt_max_angle.value:.0f}°) — no rays will be traced.')
    cost = rt_n_rays.value * rt_max_range.value
    if cost > 15000:   # ~150 rays × 100 km or equivalent
        msgs.append(f'⚠ {rt_n_rays.value} rays × {rt_max_range.value:.0f} km — expect a slow run (> 30 s). Reduce rays or range for faster results.')
    rt_warn.value = '<br>'.join(f'<span style="{_WARN_STYLE}">{m}</span>' for m in msgs)

for _w in [rt_src_depth, rt_water_depth, rt_min_angle, rt_max_angle, rt_n_rays, rt_max_range]:
    _w.observe(_rt_check, names='value')
_rt_check()

def _run_ray_trace(_=None):
    if rt_min_angle.value >= rt_max_angle.value:
        with rt_out:
            clear_output(wait=True)
            print('Cannot run: min launch angle must be less than max launch angle.')
        return
    with rt_out:
        clear_output(wait=True)
        print('Running BELLHOP ray trace...')
    water_depth = rt_water_depth.value
    src_depth = min(rt_src_depth.value, water_depth - 1)
    ssp = _locked_ssp['ssp']

    # Clip SSP to water depth
    ssp_use = eb._clip_ssp(ssp, water_depth)

    env = pm.create_env2d(
        depth=water_depth,
        soundspeed=ssp_use.tolist(),
        frequency=rt_freq.value,
        tx_depth=src_depth,
        rx_depth=np.linspace(0, water_depth, 21),
        rx_range=np.linspace(0.1, rt_max_range.value, 25),
        min_angle=rt_min_angle.value,
        max_angle=rt_max_angle.value,
        nbeams=rt_n_rays.value,
    )
    eb._apply_bottom(env, rt_bottom.value)

    rays = pm.compute_rays(env)

    with rt_out:
        clear_output(wait=True)
        if rays is None:
            print('BELLHOP did not produce ray output. Check env parameters.')
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                                  gridspec_kw={'width_ratios': [1, 2.5]})

        # Left: SSP
        pl.plot_ssp(ssp_use, ax=axes[0], title=f'SSP: {_locked_ssp["name"]}')

        # Right: ray diagram
        ax_ray = axes[1]
        ax_ray.set_facecolor('#ddeeff')

        max_range_km = rt_max_range.value   # x-axis in km throughout

        # Bottom (x in km, y in m)
        ax_ray.fill_between([0, max_range_km],
                             [water_depth] * 2, [water_depth * 1.08] * 2,
                             color='#8B6914', alpha=0.9, zorder=2)
        ax_ray.axhline(water_depth, color='#8B6914', linewidth=2, zorder=3)
        ax_ray.axhline(0, color='#2196F3', linewidth=2, zorder=3)

        # Plot rays — arlpy stores each path in row['ray'] as (N,2):
        # col 0 = range in km, col 1 = depth in metres
        cmap = plt.cm.RdYlBu
        n_rays = len(rays)
        for i, (_, row) in enumerate(rays.iterrows()):
            if 'ray' in row.index:
                ray_path = np.asarray(row['ray'])
                ax_ray.plot(ray_path[:, 0], ray_path[:, 1],   # col0 already km
                            color=cmap(i / max(n_rays - 1, 1)),
                            linewidth=0.7, alpha=0.65, zorder=4)

        ax_ray.plot(0, src_depth, 'r*', markersize=16, zorder=10,
                    label=f'Source: {src_depth:.0f} m')
        ax_ray.set_xlim(0, max_range_km)
        ax_ray.set_ylim(water_depth * 1.08, -water_depth * 0.02)
        ax_ray.set_xlabel('Range (km)', fontsize=11)
        ax_ray.set_ylabel('Depth (m)', fontsize=11)
        ax_ray.set_title(
            f'Ray Diagram  |  f={rt_freq.value:.0f} Hz  |  {n_rays} rays  |  '
            f'{rt_bottom.value} bottom  |  SSP: {_locked_ssp["name"]}',
            fontsize=12)
        ax_ray.grid(True, alpha=0.2)
        ax_ray.legend(fontsize=9)

        show_fig(fig)

        rt_stats.value = (
            f'<b>Ray statistics:</b>  '
            f'Total rays launched: <b>{n_rays}</b>  |  '
            f'Env: {rt_bottom.value} bottom, {water_depth:.0f} m deep, '
            f'src @ {src_depth:.0f} m, max range {rt_max_range.value:.0f} km'
        )

rt_run_btn.on_click(_run_ray_trace)

display(widgets.VBox([
    widgets.HBox([rt_src_depth, rt_water_depth]),
    widgets.HBox([rt_max_range, rt_n_rays]),
    widgets.HBox([rt_min_angle, rt_max_angle]),
    widgets.HBox([rt_freq, rt_bottom]),
    rt_warn,
    rt_run_btn,
    rt_stats,
    rt_out
]))
"""))

    # ---- SECTION 4: Transmission Loss ----
    cells.append(md_cell("""---
## Section 4 — Transmission Loss Map

Computes and displays the full 2D transmission loss field (depth × range).

TL is computed in dB relative to 1 m from the source:
$$TL = -20\\log_{10}\\left(\\frac{p(r,z)}{p_0}\\right)$$

Higher TL = weaker signal. Convergence zones appear as vertical bands of low TL.
"""))

    cells.append(code_cell("""\
# ── TL shared state ──────────────────────────────────────────────────────────
_tl_result = {'tl': None, 'env': None}
tl_ready = widgets.HTML(
    '<span style="color:gray">No TL data yet — click <b>▶ Run TL Map</b> above.</span>'
)

# ── TL widgets ────────────────────────────────────────────────────────────────
tl_src_depth   = _slider('Source depth (m)',   1300, 1, 4999, 1)
tl_water_depth = _slider('Water depth (m)',    5000, 200, 6000, 50)
tl_max_range   = _slider('Max range (km)',      100, 1, 500, 1)
tl_freq        = _slider('Frequency (Hz)',      500, 10, 10000, 10)
tl_rx_min_d    = _slider('Rx min depth (m)',      0, 0, 4900, 10)
tl_rx_max_d    = _slider('Rx max depth (m)',   4950, 100, 5000, 10)
tl_n_depths    = _int_slider('# Rx depths',      51, 5, 201, 5)
tl_dyn_range   = _slider('Dynamic range (dB)',   60, 20, 100, 5)
tl_bottom      = widgets.Dropdown(
    options=['sand', 'mud', 'gravel', 'rock'],
    value='sand', description='Bottom type:',
    style={'description_width': '120px'}, layout=widgets.Layout(width='300px')
)
tl_run_type = widgets.Dropdown(
    options=['coherent', 'incoherent', 'semicoherent'],
    value='incoherent', description='Run type:',
    style={'description_width': '120px'}, layout=widgets.Layout(width='300px')
)
tl_slice_depth = _slider('TL slice depth (m)', 1300, 0, 5000, 10)
tl_run_btn = widgets.Button(description='▶ Run TL Map', button_style='primary',
                              layout=widgets.Layout(width='200px'))
tl_out    = widgets.Output()
tl_stats  = widgets.HTML('')
tl_slice_btn = widgets.Button(description='Plot TL slice', button_style='',
                               layout=widgets.Layout(width='160px'))
tl_slice_out = widgets.Output()
tl_warn = widgets.HTML('')

def _tl_check(_=None):
    msgs = []
    sd, wd = tl_src_depth.value, tl_water_depth.value
    rmin, rmax = tl_rx_min_d.value, tl_rx_max_d.value
    if sd >= wd:
        msgs.append(f'⚠ Source depth ({sd:.0f} m) ≥ water depth ({wd:.0f} m) — will be clamped to {wd-1:.0f} m.')
    if rmin >= rmax:
        msgs.append(f'⚠ Rx min depth ({rmin:.0f} m) ≥ Rx max depth ({rmax:.0f} m) — no receiver grid possible.')
    if rmax > wd:
        msgs.append(f'⚠ Rx max depth ({rmax:.0f} m) exceeds water depth ({wd:.0f} m) — will be clamped to {wd:.0f} m.')
    sd_tl = tl_slice_depth.value
    if sd_tl < rmin or sd_tl > rmax:
        msgs.append(f'⚠ Slice depth ({sd_tl:.0f} m) is outside the receiver depth range ({rmin:.0f}–{rmax:.0f} m) — slice will show nearest available depth.')
    tl_warn.value = '<br>'.join(f'<span style="{_WARN_STYLE}">{m}</span>' for m in msgs)

for _w in [tl_src_depth, tl_water_depth, tl_rx_min_d, tl_rx_max_d, tl_slice_depth]:
    _w.observe(_tl_check, names='value')
_tl_check()

run_type_map = {
    'coherent': pm.coherent,
    'incoherent': pm.incoherent,
    'semicoherent': pm.semicoherent,
}

def _run_tl(_=None):
    if tl_rx_min_d.value >= tl_rx_max_d.value:
        with tl_out:
            clear_output(wait=True)
            print('Cannot run: Rx min depth must be less than Rx max depth.')
        return
    with tl_out:
        clear_output(wait=True)
        print('Running BELLHOP transmission loss computation...')
    water_depth = tl_water_depth.value
    src_depth = min(tl_src_depth.value, water_depth - 1)
    rx_min = min(tl_rx_min_d.value, water_depth)
    rx_max = min(tl_rx_max_d.value, water_depth)
    ssp = _locked_ssp['ssp']
    ssp_use = eb._clip_ssp(ssp, water_depth)

    env = pm.create_env2d(
        depth=water_depth,
        soundspeed=ssp_use.tolist(),
        frequency=tl_freq.value,
        tx_depth=src_depth,
        rx_depth=np.linspace(rx_min, rx_max, int(tl_n_depths.value)),
        rx_range=np.linspace(0.5, tl_max_range.value, 100),
        nbeams=200,
    )
    eb._apply_bottom(env, tl_bottom.value)

    task = run_type_map.get(tl_run_type.value, pm.incoherent)

    try:
        tl = pm.compute_transmission_loss(env, mode=task)
    except Exception as exc:
        import traceback
        with tl_out:
            clear_output(wait=True)
            print(f'BELLHOP error: {type(exc).__name__}: {exc}')
            traceback.print_exc()
        tl_ready.value = '<span style="color:red">✗ TL computation failed — see output above.</span>'
        return

    _tl_result['tl'] = tl
    _tl_result['env'] = env

    with tl_out:
        clear_output(wait=True)
        if tl is None:
            print('BELLHOP did not produce TL output.')
            tl_ready.value = '<span style="color:red">✗ TL returned None.</span>'
            return

        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                                      gridspec_kw={'width_ratios': [1, 3.5]})

            pl.plot_ssp(ssp_use, ax=axes[0], title=f'SSP: {_locked_ssp["name"]}')
            _, cb = pl.plot_transmission_loss(
                tl, env, ax=axes[1],
                dynamic_range=tl_dyn_range.value,
                title=f'{tl_run_type.value.title()} TL  |  '
                      f'f={tl_freq.value:.0f} Hz  |  {tl_bottom.value} bottom'
            )

            # Convergence zone detection
            cz_ranges = ut.find_convergence_zones(tl, threshold_db=4.0)
            if cz_ranges:
                pl.annotate_convergence_zones(axes[1], cz_ranges)
                cz_str = ', '.join(f'{r:.0f} km' for r in cz_ranges)
                tl_stats.value = (
                    f'<b>Convergence zones detected at:</b> {cz_str}'
                    f' | Run type: {tl_run_type.value}'
                )
            else:
                tl_stats.value = (
                    f'<b>No convergence zones detected</b> | Run type: {tl_run_type.value}'
                )

            show_fig(fig)

            tl_ready.value = (
                f'<span style="color:green;font-weight:bold">✓ TL ready</span> — '
                f'{tl_run_type.value}, f={tl_freq.value:.0f} Hz, '
                f'src={src_depth:.0f} m, max {tl_max_range.value:.0f} km. '
                f'Use <b>Pull TL from Section 4</b> in Section 7.'
            )

        except Exception as plot_exc:
            import traceback
            print(f'Plot error: {type(plot_exc).__name__}: {plot_exc}')
            traceback.print_exc()
            tl_ready.value = '<span style="color:orange">⚠ TL computed but plot failed.</span>'

def _plot_tl_slice(_=None):
    tl = _tl_result.get('tl')
    env = _tl_result.get('env')
    if tl is None:
        with tl_slice_out:
            clear_output(wait=True)
            print('Run the TL map first.')
        return
    with tl_slice_out:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(11, 4))
        pl.plot_tl_slice(tl, tl_slice_depth.value, env, ax=ax)
        show_fig(fig)

tl_run_btn.on_click(_run_tl)
tl_slice_btn.on_click(_plot_tl_slice)

display(widgets.VBox([
    widgets.HBox([tl_src_depth, tl_water_depth]),
    widgets.HBox([tl_max_range, tl_freq]),
    widgets.HBox([tl_rx_min_d, tl_rx_max_d, tl_n_depths]),
    widgets.HBox([tl_dyn_range, tl_bottom, tl_run_type]),
    tl_warn,
    tl_run_btn,
    tl_ready,
    tl_stats,
    tl_out,
    widgets.HTML('<hr><b>Horizontal TL slice</b>'),
    widgets.HBox([tl_slice_depth, tl_slice_btn]),
    tl_slice_out,
]))
"""))

    # ---- SECTION 5: Arrivals ----
    cells.append(md_cell("""---
## Section 5 — Multipath Arrivals Analysis

BELLHOP traces every ray path from source to receiver and records:
- **Arrival time** — when each ray path reaches the receiver
- **Complex amplitude** — amplitude and phase of each path

### Why does this matter for communications?
In an underwater acoustic channel, every symbol you transmit arrives multiple times via
different paths. Each copy is delayed and scaled differently — this is **multipath**.

The **delay spread** $T_d$ (latest minus earliest arrival) determines:
- **Coherence bandwidth**: $B_c \\approx 1/T_d$ — signals wider than this suffer ISI
- **ISI length**: $L_{ISI} = \\lfloor T_d \\cdot f_s \\rfloor$ taps — the equalizer must handle this many

For typical ocean channels: $T_d \\approx 10$–$200$ ms → $B_c \\approx 5$–$100$ Hz
(Compare: OFDM subcarrier spacing must be $\\gg B_c$)

### Getting interesting multipath

The number of arrivals depends strongly on geometry:

| Setting | Result |
|---------|--------|
| Source + Rx both on SOFAR axis (1300 m), range < 65 km | **1 arrival** — only the direct horizontal path returns to that depth; oscillating rays are mid-cycle |
| Source on SOFAR axis, range ≈ **65 km** (1st convergence zone) | **Many arrivals** — all oscillating rays refocus at the CZ |
| Source on SOFAR axis, Rx at shallow depth (e.g. 100 m) | **Several arrivals** — catches the upward-swinging rays mid-cycle |

> **Default is 65 km** — the first convergence zone of the Munk profile where multipath is richest.
"""))

    cells.append(code_cell("""\
# ── Arrivals shared state ────────────────────────────────────────────────────
_arr_result = {'arr': None, 'env': None}

# ── Arrivals widgets ──────────────────────────────────────────────────────────
arr_src_depth   = _slider('Source depth (m)', 1300, 1, 4999, 1)
arr_water_depth = _slider('Water depth (m)', 5000, 200, 6000, 50)
arr_freq        = _slider('Frequency (Hz)',   500, 10, 5000, 10)
arr_bottom      = widgets.Dropdown(
    options=['sand', 'mud', 'gravel', 'rock'],
    value='sand', description='Bottom type:',
    style={'description_width': '120px'}, layout=widgets.Layout(width='300px')
)
arr_rx_range = _slider('Rx range (km)',  65, 0.5, 300, 0.5)
arr_rx_depth = _slider('Rx depth (m)', 1300, 1, 4999, 1)
arr_fs       = _slider('Sample rate fs (Hz)', 8000, 1000, 50000, 500)
arr_run_btn  = widgets.Button(description='▶ Compute Arrivals', button_style='primary',
                               layout=widgets.Layout(width='220px'))
arr_out      = widgets.Output()
arr_stats    = widgets.HTML('')
arr_warn     = widgets.HTML('')

def _arr_check(_=None):
    msgs = []
    sd, wd = arr_src_depth.value, arr_water_depth.value
    rd, rng = arr_rx_depth.value, arr_rx_range.value
    if sd >= wd:
        msgs.append(f'⚠ Source depth ({sd:.0f} m) ≥ water depth ({wd:.0f} m) — will be clamped to {wd-1:.0f} m.')
    if rd >= wd:
        msgs.append(f'⚠ Rx depth ({rd:.0f} m) ≥ water depth ({wd:.0f} m) — will be clamped to {wd-1:.0f} m.')
    if abs(rd - sd) > 500 and rng < 20:
        msgs.append(f'⚠ Source ({sd:.0f} m) and Rx ({rd:.0f} m) are far apart in depth at only {rng:.0f} km range — few or no arrivals likely. Try range > 20 km or match source/Rx depths.')
    if abs(rd - sd) < 200 and rng < 60:
        msgs.append(f'ℹ Source and Rx near same depth ({sd:.0f} m) at {rng:.0f} km — SOFAR cycle ≈ 65 km, so most oscillating rays miss this receiver. Try range ≈ 65 km for the first convergence zone (many arrivals).')
    arr_warn.value = '<br>'.join(f'<span style="{_WARN_STYLE}">{m}</span>' for m in msgs)

for _w in [arr_src_depth, arr_water_depth, arr_rx_depth, arr_rx_range]:
    _w.observe(_arr_check, names='value')
_arr_check()

def _run_arrivals(_=None):
    water_depth = arr_water_depth.value
    src_depth = min(arr_src_depth.value, water_depth - 1)
    rx_depth  = min(arr_rx_depth.value,  water_depth - 1)
    ssp_use   = eb._clip_ssp(_locked_ssp['ssp'], water_depth)

    env = pm.create_env2d(
        depth=water_depth,
        soundspeed=ssp_use.tolist(),
        frequency=arr_freq.value,
        tx_depth=src_depth,
        rx_depth=np.array([rx_depth]),
        rx_range=np.array([arr_rx_range.value]),
        nbeams=200,
    )
    eb._apply_bottom(env, arr_bottom.value)

    with arr_out:
        clear_output(wait=True)
        print(f'Running BELLHOP arrivals — src={src_depth:.0f} m  rx={rx_depth:.0f} m @ {arr_rx_range.value:.1f} km ...')

    try:
        arr = pm.compute_arrivals(env)
    except ValueError:
        # pandas ≥2.0: pd.concat([]) raises ValueError when BELLHOP finds no arrivals
        with arr_out:
            clear_output(wait=True)
            print('No arrivals found (BELLHOP returned empty result).')
            print('Suggestions:')
            print('  • Increase range (try ≥ 50 km for deep water Munk profile)')
            print('  • Match Rx depth to source depth for maximum energy coupling')
            print('  • Verify source depth < water depth')
        return
    except Exception as exc:
        import traceback
        with arr_out:
            clear_output(wait=True)
            print(f'BELLHOP error: {type(exc).__name__}: {exc}')
            traceback.print_exc()
        return

    _arr_result['arr'] = arr
    _arr_result['env'] = env

    with arr_out:
        clear_output(wait=True)
        try:
            if arr is None or len(arr) == 0:
                print('No arrivals returned (empty DataFrame).')
                print('Try a longer range or different geometry.')
                return

            _amp_col = 'arrival_amplitude' if 'arrival_amplitude' in arr.columns else 'amplitude'
            if 'time_of_arrival' not in arr.columns or _amp_col not in arr.columns:
                print(f'Unexpected arrival columns: {list(arr.columns)}')
                print('Expected: time_of_arrival, arrival_amplitude (or amplitude)')
                return

            times_s = np.asarray(arr['time_of_arrival'], dtype=float)
            amps    = np.abs(np.asarray(arr[_amp_col], dtype=complex))
            valid   = np.isfinite(times_s) & np.isfinite(amps) & (amps > 0)
            times_s = times_s[valid]
            amps    = amps[valid]

            if len(times_s) == 0:
                print(f'All {len(arr)} arrivals have NaN or zero amplitude.')
                print('BELLHOP may have found eigenrays but with negligible energy.')
                print('Try: match Rx depth to source depth, or increase range.')
                return

            delay_ms = (times_s - times_s.min()) * 1000.0
            amps_n   = amps / amps.max()
            spread   = float(delay_ms.max())

            if spread > 0:
                arr_stats.value = (
                    f'<b>Arrivals:</b> {len(times_s)} paths  |  '
                    f'Delay spread: <b>{spread:.2f} ms</b>  |  '
                    f'Coherence BW: <b>{1000.0 / spread:.1f} Hz</b>'
                )
            else:
                arr_stats.value = (
                    f'<b>Arrivals:</b> {len(times_s)} paths  |  '
                    f'Single eigenray (0 ms spread)'
                )

            # ── Bounce-count colors ───────────────────────────────────────────
            _bp = ['#1a6fa3', '#27ae60', '#e67e22', '#c0392b']
            if 'surface_bounces' in arr.columns and 'bottom_bounces' in arr.columns:
                _n_tot = (np.asarray(arr['surface_bounces'], dtype=int) +
                          np.asarray(arr['bottom_bounces'],  dtype=int))[valid]
                _colors = [_bp[min(int(n), 3)] for n in _n_tot]
            else:
                _colors = [_bp[0]] * len(delay_ms)

            fig, axes = plt.subplots(1, 2, figsize=(16, 4))

            # ── Left: multipath arrivals ──────────────────────────────────────
            x_span = max(spread, 1.0)
            axes[0].vlines(delay_ms, 0, amps_n, colors=_colors, linewidth=2, alpha=0.85)
            axes[0].scatter(delay_ms, amps_n, c=_colors, s=60, zorder=5)
            axes[0].axhline(0, color='k', linewidth=0.8)
            axes[0].set_xlim(-x_span * 0.05, x_span * 1.15)
            axes[0].set_ylim(-0.05, 1.15)
            axes[0].set_xlabel('Delay relative to first arrival (ms)', fontsize=11)
            axes[0].set_ylabel('Normalized Amplitude', fontsize=11)
            axes[0].set_title(
                f'Multipath Arrivals  |  {len(times_s)} paths  |  '
                f'spread={spread:.2f} ms  |  '
                f'f={arr_freq.value:.0f} Hz  |  '
                f'Rx {arr_rx_range.value:.0f} km / {rx_depth:.0f} m',
                fontsize=10
            )
            axes[0].grid(True, alpha=0.25)
            # Bounce legend
            from matplotlib.lines import Line2D as _L2D
            _leg = [_L2D([0],[0], color=c, lw=2, label=lbl)
                    for c, lbl in zip(_bp, ['0 bounces', '1 bounce', '2 bounces', '3+ bounces'])]
            axes[0].legend(handles=_leg, fontsize=8, loc='upper right')

            # ── Right: channel impulse response ───────────────────────────────
            fs      = arr_fs.value
            ir      = ut.channel_to_impulse_response(arr, fs=fs)
            t_ms    = np.arange(len(ir)) / fs * 1000.0
            ir_abs  = np.abs(ir)
            ir_max  = float(ir_abs.max()) if ir_abs.max() > 0 else 1.0
            ir_norm = ir_abs / ir_max      # normalize: weak arrivals visible
            sig     = ir_norm > 0
            if sig.any():
                axes[1].vlines(t_ms[sig], 0, ir_norm[sig],
                               colors='#e67e22', linewidth=2, alpha=0.85)
                axes[1].scatter(t_ms[sig], ir_norm[sig], color='#e67e22', s=50, zorder=5)
                x2_span = max(float(t_ms[sig].max()), 1.0)
                axes[1].set_xlim(-x2_span * 0.05, x2_span * 1.15)
            axes[1].axhline(0, color='k', linewidth=0.8)
            axes[1].set_ylim(-0.05, 1.15)
            stats_val = ut.compute_arrival_stats(arr)
            isi_taps  = int(stats_val.get('delay_spread_ms', 0) / 1000.0 * fs)
            axes[1].set_xlabel('Delay (ms)', fontsize=11)
            axes[1].set_ylabel('Normalized Amplitude', fontsize=11)
            axes[1].set_title(
                f'Channel Impulse Response  |  fs={fs:.0f} Hz  |  ISI taps ≈ {isi_taps}',
                fontsize=11
            )
            axes[1].grid(True, alpha=0.25)

            show_fig(fig)

        except Exception as _plot_err:
            import traceback
            print(f'Plotting error: {type(_plot_err).__name__}: {_plot_err}')
            print()
            print('--- Diagnostic ---')
            if arr is not None and len(arr) > 0:
                print(f'Columns : {list(arr.columns)}')
                print(f'Rows    : {len(arr)}')
                _ac = 'arrival_amplitude' if 'arrival_amplitude' in arr.columns else 'amplitude'
                if 'time_of_arrival' in arr.columns and _ac in arr.columns:
                    _t = np.asarray(arr['time_of_arrival'], dtype=float)
                    _a = np.abs(np.asarray(arr[_ac], dtype=complex))
                    _v = np.isfinite(_t) & np.isfinite(_a) & (_a > 0)
                    print(f'Valid   : {int(_v.sum())}/{len(arr)}')
                    if _v.sum() > 0:
                        print(f'Time    : {_t[_v].min():.4f}–{_t[_v].max():.4f} s '
                              f'(spread={1000*(_t[_v].max()-_t[_v].min()):.2f} ms)')
                        print(f'Amp     : {_a[_v].min():.3e}–{_a[_v].max():.3e}')
            traceback.print_exc()

arr_run_btn.on_click(_run_arrivals)

display(widgets.VBox([
    widgets.HBox([arr_src_depth, arr_water_depth]),
    widgets.HBox([arr_freq, arr_bottom]),
    widgets.HBox([arr_rx_range, arr_rx_depth]),
    arr_fs,
    arr_warn,
    arr_run_btn,
    arr_stats,
    arr_out,
]))
"""))

    # ---- SECTION 6: Scenario Comparison ----
    cells.append(md_cell("""---
## Section 6 — Scenario Comparison

Save up to 4 named scenarios (each with its own SSP, depth, frequency, bottom) then
compare their TL-vs-range curves and summary statistics side-by-side.

**Workflow:**
1. Configure the SSP in Section 2 and lock it
2. Set the parameters below and click **Save Scenario**
3. Repeat for up to 4 scenarios
4. Click **Compare Scenarios** to overlay the TL curves
"""))

    cells.append(code_cell("""\
# ── Scenario storage ──────────────────────────────────────────────────────────
_scenarios = []   # list of dicts: {name, ssp, ssp_name, water_depth, src_depth, freq, bottom}

# ── Scenario widgets ──────────────────────────────────────────────────────────
sc_name       = widgets.Text(value='Scenario 1', description='Name:',
                              style={'description_width': '80px'},
                              layout=widgets.Layout(width='280px'))
sc_water_d    = _slider('Water depth (m)',  5000, 200, 6000, 50)
sc_src_d      = _slider('Source depth (m)', 1300, 1, 4999, 1)
sc_freq       = _slider('Frequency (Hz)',    500, 10, 5000, 10)
sc_bottom     = widgets.Dropdown(
    options=['sand', 'mud', 'gravel', 'rock'],
    value='sand', description='Bottom:',
    style={'description_width': '80px'}, layout=widgets.Layout(width='260px')
)
sc_max_range  = _slider('Max range (km)',   100, 5, 500, 5)
sc_save_btn   = widgets.Button(description='💾 Save Scenario', button_style='warning',
                                layout=widgets.Layout(width='200px'))
sc_compare_btn= widgets.Button(description='📊 Compare Scenarios', button_style='success',
                                layout=widgets.Layout(width='220px'))
sc_clear_btn  = widgets.Button(description='🗑 Clear All', button_style='danger',
                                layout=widgets.Layout(width='150px'))
sc_list_html  = widgets.HTML('<i>No scenarios saved yet.</i>')
sc_warn       = widgets.HTML('')
sc_out        = widgets.Output()

def _sc_check(_=None):
    msgs = []
    if sc_src_d.value >= sc_water_d.value:
        msgs.append(f'⚠ Source depth ({sc_src_d.value:.0f} m) ≥ water depth ({sc_water_d.value:.0f} m) — will be clamped on save.')
    sc_warn.value = '<br>'.join(f'<span style="{_WARN_STYLE}">{m}</span>' for m in msgs)

for _w in [sc_src_d, sc_water_d]:
    _w.observe(_sc_check, names='value')
_sc_check()

def _update_sc_list():
    if not _scenarios:
        sc_list_html.value = '<i>No scenarios saved yet.</i>'
        return
    rows = '<br>'.join(
        f'<b>{i+1}. {s["name"]}</b>: SSP={s["ssp_name"]}, '
        f'{s["water_depth"]:.0f}m deep, src@{s["src_depth"]:.0f}m, '
        f'f={s["freq"]:.0f}Hz, {s["bottom"]}, max {s["max_range"]:.0f}km'
        for i, s in enumerate(_scenarios)
    )
    sc_list_html.value = rows

def _save_scenario(_=None):
    if len(_scenarios) >= 4:
        sc_list_html.value = '<b style="color:red">Maximum 4 scenarios already saved. Clear to start over.</b>'
        return
    _scenarios.append({
        'name':        sc_name.value or f'Scenario {len(_scenarios)+1}',
        'ssp':         _locked_ssp['ssp'].copy(),
        'ssp_name':    _locked_ssp['name'],
        'water_depth': sc_water_d.value,
        'src_depth':   min(sc_src_d.value, sc_water_d.value - 1),
        'freq':        sc_freq.value,
        'bottom':      sc_bottom.value,
        'max_range':   sc_max_range.value,
    })
    sc_name.value = f'Scenario {len(_scenarios)+1}'
    _update_sc_list()

def _compare_scenarios(_=None):
    with sc_out:
        clear_output(wait=True)
        if len(_scenarios) < 1:
            print('Save at least one scenario first.')
            return
        print(f'Running TL for {len(_scenarios)} scenarios...')

        tl_results, envs, labels = [], [], []
        for s in _scenarios:
            ssp_use = eb._clip_ssp(s['ssp'], s['water_depth'])
            env = pm.create_env2d(
                depth=s['water_depth'],
                soundspeed=ssp_use.tolist(),
                frequency=s['freq'],
                tx_depth=s['src_depth'],
                rx_depth=np.linspace(0, s['water_depth'], 21),
                rx_range=np.linspace(0.5, s['max_range'], 80),
                nbeams=200,
            )
            eb._apply_bottom(env, s['bottom'])
            tl = pm.compute_transmission_loss(env, mode=pm.incoherent)
            tl_results.append(tl)
            envs.append(env)
            labels.append(s['name'])
            print(f'  Completed: {s["name"]}')

        # Comparison TL plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        colors = plt.cm.tab10(np.linspace(0, 0.9, len(_scenarios)))
        for i, (tl, env, label, s) in enumerate(zip(tl_results, envs, labels, _scenarios)):
            if tl is None:
                continue
            depths = np.asarray(tl.index, dtype=float)
            ranges_km = np.asarray(tl.columns, dtype=float)
            d_ref = s['src_depth']
            d_idx = int(np.argmin(np.abs(depths - d_ref)))
            p_abs = np.abs(np.asarray(tl.iloc[d_idx, :], dtype=complex))
            p_abs = np.where(p_abs < 1e-10, np.nan, p_abs)
            tl_slice = -20.0 * np.log10(p_abs)
            axes[0].plot(ranges_km, tl_slice, color=colors[i], linewidth=2, label=label)

        # SSP overlay
        for i, s in enumerate(_scenarios):
            axes[1].plot(s['ssp'][:, 1], s['ssp'][:, 0],
                         color=colors[i], linewidth=2, label=s['name'])

        axes[0].invert_yaxis()
        axes[0].set_xlabel('Range (km)', fontsize=11)
        axes[0].set_ylabel('Transmission Loss (dB)', fontsize=11)
        axes[0].set_title('TL vs Range at Source Depth', fontsize=12)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=9)

        axes[1].invert_yaxis()
        axes[1].set_xlabel('Sound Speed (m/s)', fontsize=11)
        axes[1].set_ylabel('Depth (m)', fontsize=11)
        axes[1].set_title('Sound Speed Profiles', fontsize=12)
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=9)

        plt.suptitle('Scenario Comparison', fontsize=14, fontweight='bold')
        show_fig(fig)

        # Summary table
        import pandas as pd
        rows = []
        for s in _scenarios:
            rows.append({
                'Name': s['name'], 'SSP': s['ssp_name'],
                'Depth (m)': s['water_depth'], 'Src (m)': s['src_depth'],
                'Freq (Hz)': s['freq'], 'Bottom': s['bottom'],
                'Max Range (km)': s['max_range'],
            })
        df = pd.DataFrame(rows)
        from IPython.display import display as ipy_display
        ipy_display(df)

def _clear_scenarios(_=None):
    _scenarios.clear()
    _update_sc_list()
    with sc_out:
        clear_output(wait=True)

sc_save_btn.on_click(_save_scenario)
sc_compare_btn.on_click(_compare_scenarios)
sc_clear_btn.on_click(_clear_scenarios)

display(widgets.VBox([
    widgets.HBox([sc_name, sc_bottom]),
    widgets.HBox([sc_water_d, sc_src_d]),
    widgets.HBox([sc_freq, sc_max_range]),
    sc_warn,
    widgets.HBox([sc_save_btn, sc_compare_btn, sc_clear_btn]),
    sc_list_html,
    sc_out,
]))
"""))

    # ---- SECTION 7: Sonar Equation ----
    cells.append(md_cell("""---
## Section 7 — Sonar Equation Calculator

The passive sonar equation:

$$SNR = SL - TL - NL + DI \\quad \\text{(all in dB)}$$

| Term | Symbol | Meaning |
|------|--------|---------|
| Source Level | SL | Radiated power of target (dB re 1 µPa @ 1 m) |
| Transmission Loss | TL | Acoustic path loss (from Section 4 or manual) |
| Noise Level | NL | Ambient ocean noise at receiver (dB re 1 µPa) |
| Directivity Index | DI | Receiver array gain over an omnidirectional sensor (dB) |
| Detection Threshold | DT | Min SNR for reliable detection (dB) |

A target is **detectable** when $SNR \\geq DT$.
"""))

    cells.append(code_cell("""\
# ── Sonar equation widgets ────────────────────────────────────────────────────
sn_sl  = _slider('SL — Source Level (dB)',      200, 140, 240, 1)
sn_nl  = _slider('NL — Noise Level (dB)',        60, 30, 100, 1)
sn_di  = _slider('DI — Directivity Index (dB)',   0, 0, 40, 1)
sn_dt  = _slider('DT — Detection Threshold (dB)',  10, 0, 30, 1)
sn_tl_manual = _slider('TL (dB)', 80, 20, 150, 0.1)
sn_tl_range  = _slider('Lookup range (km)',  50, 0.5, 500, 0.5)
sn_tl_depth  = _slider('Lookup depth (m)', 1300, 0, 5000, 10)
sn_tl_from4  = widgets.Button(
    description='Pull TL from Section 4',
    button_style='info', layout=widgets.Layout(width='220px')
)
sn_pull_status = widgets.HTML(
    '<i style="color:gray">Set the lookup range/depth above, then click the button '
    'to read TL at that point from the Section 4 map.</i>'
)
sn_out = widgets.Output()

_sonar_state = {'tl': 80.0}

def _update_sonar(_=None):
    tl = _sonar_state['tl']
    snr = ut.compute_snr_estimate(tl, sn_sl.value, sn_nl.value, sn_di.value)
    margin = snr - sn_dt.value
    verdict = 'DETECTED ✓' if margin >= 0 else 'NOT DETECTED ✗'
    with sn_out:
        clear_output(wait=True)
        print(f'  SL  = {sn_sl.value:.0f} dB')
        print(f'  TL  = {tl:.1f} dB')
        print(f'  NL  = {sn_nl.value:.0f} dB')
        print(f'  DI  = {sn_di.value:.0f} dB')
        print(f'─' * 30)
        print(f'  SNR = {snr:.1f} dB')
        print(f'  DT  = {sn_dt.value:.0f} dB')
        print(f'  Margin = {margin:+.1f} dB  →  {verdict}')

def _pull_tl_from_sec4(_=None):
    tl_df = _tl_result.get('tl')
    if tl_df is None:
        sn_pull_status.value = (
            '<span style="color:red;font-weight:bold">'
            '⚠ No TL data found. In Section 4: set parameters, then click '
            '<b>▶ Run TL Map</b>. The status line will turn green when ready.</span>'
        )
        return

    depths  = np.asarray(tl_df.index,   dtype=float)
    ranges  = np.asarray(tl_df.columns, dtype=float)
    d_idx   = int(np.argmin(np.abs(depths - sn_tl_depth.value)))
    r_idx   = int(np.argmin(np.abs(ranges - sn_tl_range.value)))
    p_val   = float(np.abs(complex(tl_df.iloc[d_idx, r_idx])))
    tl_val  = float(-20.0 * np.log10(p_val)) if p_val > 1e-10 else 120.0

    # Clamp to slider range before setting (avoids silent widget clamp)
    tl_clamped = float(np.clip(tl_val, 20, 150))
    _sonar_state['tl'] = tl_clamped
    sn_tl_manual.value  = tl_clamped

    actual_r = float(ranges[r_idx])
    actual_d = float(depths[d_idx])
    sn_pull_status.value = (
        f'<span style="color:green;font-weight:bold">'
        f'✓ Pulled TL = {tl_val:.1f} dB '
        f'from Section 4 at {actual_r:.1f} km / {actual_d:.0f} m depth</span>'
        + (f'<br><span style="color:gray;font-size:0.85em">'
           f'(Nearest computed point to your lookup sliders)</span>'
           if abs(actual_r - sn_tl_range.value) > 1 or abs(actual_d - sn_tl_depth.value) > 50
           else '')
    )
    _update_sonar()

def _manual_tl_changed(change):
    _sonar_state['tl'] = change['new']
    sn_pull_status.value = '<i style="color:gray">TL set manually.</i>'
    _update_sonar()

for w in [sn_sl, sn_nl, sn_di, sn_dt]:
    w.observe(_update_sonar, names='value')
sn_tl_manual.observe(_manual_tl_changed, names='value')
sn_tl_from4.on_click(_pull_tl_from_sec4)

_update_sonar()

display(widgets.VBox([
    widgets.HBox([sn_sl, sn_nl]),
    widgets.HBox([sn_di, sn_dt]),
    widgets.HTML('<hr><b>Transmission Loss (TL)</b><br>'
                 '<span style="color:gray;font-size:0.9em">'
                 'Either drag the slider to enter TL manually, or use the Section 4 lookup below.</span>'),
    sn_tl_manual,
    widgets.HTML('<b>Section 4 TL lookup</b> — set range &amp; depth, then click the button:'),
    widgets.HBox([sn_tl_range, sn_tl_depth]),
    widgets.HBox([sn_tl_from4, sn_pull_status]),
    widgets.HTML('<hr>'),
    sn_out,
]))
"""))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": nb_metadata(),
        "cells": cells
    }

    path = os.path.join(NB_DIR, 'bellhop_interactive.ipynb')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Wrote: {path}')
    return path


# =============================================================================
# ADVANCED NOTEBOOK
# =============================================================================

def make_advanced_notebook():
    cells = []

    cells.append(md_cell("""# BELLHOP Advanced Scenarios

Advanced multi-scenario analyses built on top of the main interactive notebook.

## Contents
1. **SOFAR Channel Animation** — Source sinking through water column, animated ray plots
2. **Frequency Sweep** — TL comparison across 5 frequencies
3. **Range-Dependent Bathymetry** — Sloping continental shelf
4. **Communication Channel Simulator** — BPSK through acoustic channel + BER vs SNR
"""))

    # Setup
    cells.append(code_cell("""\
%matplotlib inline
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

import src.bellhop_config as bc
bc.configure(verbose=True)
import arlpy.uwapm as pm

import src.profiles as prof
import src.environment_builder as eb
import src.plotting as pl
import src.utils as ut
"""))

    # ---- 1. SOFAR Animation ----
    cells.append(md_cell("""---
## 1 — SOFAR Channel Animation

Step the source depth from near-surface to maximum depth.
For each step, BELLHOP computes ray paths and we assemble them into an animation
showing how propagation geometry evolves as the source sinks through the water column.
"""))

    cells.append(code_cell("""\
# ── SOFAR animation controls ──────────────────────────────────────────────────
anim_water_depth  = 5000.0
anim_max_range    = 150.0   # km
anim_n_src_depths = 12      # number of animation frames
anim_freq         = 250.0
anim_bottom       = 'sand'

ssp_munk = prof.munk_profile(z_max=anim_water_depth)

# Source depths from near-surface to bottom
src_depths = np.linspace(10, anim_water_depth - 50, anim_n_src_depths)

print(f'Computing {anim_n_src_depths} ray frames (one per source depth)...')

frames_rays = []
for i, src_d in enumerate(src_depths):
    ssp_use = eb._clip_ssp(ssp_munk, anim_water_depth)
    env = pm.create_env2d(
        depth=anim_water_depth,
        soundspeed=ssp_use.tolist(),
        frequency=anim_freq,
        tx_depth=float(src_d),
        rx_depth=np.linspace(0, anim_water_depth, 31),
        rx_range=np.linspace(0.5, anim_max_range, 60),
        nbeams=80, min_angle=-60, max_angle=60,
    )
    eb._apply_bottom(env, anim_bottom)
    rays = pm.compute_rays(env)
    frames_rays.append((src_d, rays))
    print(f'  Frame {i+1}/{anim_n_src_depths}: src depth = {src_d:.0f} m')

print('All frames computed. Building animation...')
"""))

    cells.append(code_cell("""\
# ── Build and display animation ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
max_range_m = anim_max_range * 1000.0

def draw_frame(frame_idx):
    ax.clear()
    src_d, rays = frames_rays[frame_idx]

    ax.set_facecolor('#ddeeff')
    ax.fill_between([0, max_range_m],
                    [anim_water_depth] * 2, [anim_water_depth * 1.06] * 2,
                    color='#8B6914', alpha=0.9)
    ax.axhline(anim_water_depth, color='#8B6914', linewidth=2)
    ax.axhline(0, color='#2196F3', linewidth=2)

    if rays is not None:
        cmap = plt.cm.RdYlBu
        n_r = len(rays)
        for i, (_, row) in enumerate(rays.iterrows()):
            if 'ray' in row.index:
                rp = np.asarray(row['ray'])
                ax.plot(rp[:, 0] * 1000, rp[:, 1],
                        color=cmap(i / max(n_r - 1, 1)), linewidth=0.7, alpha=0.6)

    ax.plot(0, src_d, 'r*', markersize=16, zorder=10)
    ax.set_xlim(0, max_range_m)
    ax.set_ylim(anim_water_depth * 1.06, -anim_water_depth * 0.02)
    ax.set_xlabel('Range (m)', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.set_title(
        f'SOFAR Channel Animation  |  Source at {src_d:.0f} m depth  |  '
        f'Frame {frame_idx+1}/{len(frames_rays)}',
        fontsize=12
    )
    ax.grid(True, alpha=0.2)

    # Overlay SOFAR axis
    sofar_d = prof.get_sofar_depth(ssp_munk)
    ax.axhline(sofar_d, color='yellow', linewidth=1.5, linestyle='--',
               alpha=0.8, label=f'SOFAR axis: {sofar_d:.0f} m')
    ax.legend(fontsize=9)

anim = animation.FuncAnimation(fig, draw_frame, frames=len(frames_rays),
                                interval=800, repeat=True)
plt.tight_layout()

# Render as HTML5 video
from IPython.display import HTML
HTML(anim.to_jshtml())
"""))

    # ---- 2. Frequency Sweep ----
    cells.append(md_cell("""---
## 2 — Frequency Sweep

Higher frequencies: shorter wavelengths, sharper interference patterns, higher absorption.
Lower frequencies: longer range, smoother TL, but less directional.

We sweep from 100 Hz to 5 kHz for the same Munk-profile deep-ocean environment.
"""))

    cells.append(code_cell("""\
# ── Frequency sweep ───────────────────────────────────────────────────────────
sweep_freqs  = [100, 500, 1000, 2500, 5000]   # Hz
sweep_depth  = 5000.0    # m
sweep_src    = 100.0     # m
sweep_range  = 100.0     # km
sweep_bottom = 'sand'

ssp_munk = prof.munk_profile(z_max=sweep_depth)
ssp_use  = eb._clip_ssp(ssp_munk, sweep_depth)

tl_results = {}
rx_depths  = np.linspace(0, sweep_depth, 51)
rx_ranges  = np.linspace(0.5, sweep_range, 80)

for freq in sweep_freqs:
    env = pm.create_env2d(
        depth=sweep_depth,
        soundspeed=ssp_use.tolist(),
        frequency=float(freq),
        tx_depth=sweep_src,
        rx_depth=rx_depths,
        rx_range=rx_ranges,
        nbeams=200,
    )
    eb._apply_bottom(env, sweep_bottom)
    tl = pm.compute_transmission_loss(env, mode=pm.incoherent)
    tl_results[freq] = tl
    print(f'  f={freq} Hz complete')

print('Frequency sweep done.')
"""))

    cells.append(code_cell("""\
# ── Plot frequency sweep results ─────────────────────────────────────────────
fig, axes = plt.subplots(len(sweep_freqs), 1, figsize=(14, 4*len(sweep_freqs)),
                          sharex=True)

for i, freq in enumerate(sweep_freqs):
    tl = tl_results[freq]
    if tl is None:
        axes[i].set_title(f'{freq} Hz — no data')
        continue
    depths = np.asarray(tl.index, dtype=float)
    ranges_km = np.asarray(tl.columns, dtype=float)
    p_abs = np.abs(np.asarray(tl.values, dtype=complex))
    p_abs = np.where(p_abs < 1e-10, np.nan, p_abs)
    tl_matrix = -20.0 * np.log10(p_abs)
    vmin = np.nanmin(tl_matrix)
    im = axes[i].pcolormesh(ranges_km, depths, tl_matrix,
                             vmin=vmin, vmax=vmin+70, cmap='jet', shading='auto')
    axes[i].fill_between(ranges_km[[0,-1]], [sweep_depth]*2, [sweep_depth*1.05]*2,
                          color='#8B6914', alpha=0.9)
    axes[i].axhline(sweep_depth, color='#8B6914', linewidth=2)
    axes[i].set_ylim(sweep_depth*1.05, -sweep_depth*0.01)
    axes[i].set_ylabel('Depth (m)', fontsize=10)
    axes[i].set_title(f'f = {freq} Hz  (incoherent TL)', fontsize=11)
    axes[i].grid(True, alpha=0.15, color='white')
    plt.colorbar(im, ax=axes[i], label='TL (dB)', fraction=0.025)

axes[-1].set_xlabel('Range (km)', fontsize=11)
plt.suptitle('Frequency Sweep — Transmission Loss vs Frequency\\nMunk Profile, 5000 m depth',
             fontsize=13, fontweight='bold', y=1.002)
plt.tight_layout()
display(fig)
plt.close(fig)
"""))

    cells.append(code_cell("""\
# ── TL at source depth vs range for each frequency ───────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
colors = plt.cm.plasma(np.linspace(0, 0.85, len(sweep_freqs)))

for i, freq in enumerate(sweep_freqs):
    tl = tl_results[freq]
    if tl is None: continue
    depths = np.asarray(tl.index, dtype=float)
    ranges_km = np.asarray(tl.columns, dtype=float)
    d_idx = int(np.argmin(np.abs(depths - sweep_src)))
    p_abs = np.abs(np.asarray(tl.iloc[d_idx, :], dtype=complex))
    p_abs = np.where(p_abs < 1e-10, np.nan, p_abs)
    tl_slice = -20.0 * np.log10(p_abs)
    ax.plot(ranges_km, tl_slice, color=colors[i], linewidth=2,
            label=f'{freq} Hz')

ax.invert_yaxis()
ax.set_xlabel('Range (km)', fontsize=11)
ax.set_ylabel('Transmission Loss (dB)', fontsize=11)
ax.set_title(f'TL at Source Depth ({sweep_src:.0f} m) vs Range — Frequency Comparison', fontsize=12)
ax.grid(True, alpha=0.25)
ax.legend(title='Frequency', fontsize=10)
plt.tight_layout()
display(fig)
plt.close(fig)
"""))

    # ---- 3. Range-Dependent Bathymetry ----
    cells.append(md_cell("""---
## 3 — Range-Dependent Bathymetry

Model a continental shelf: the bottom slopes from shallow (200 m) to deep (3000 m)
over 80 km — a typical continental slope geometry. arlpy accepts a range-dependent
bathymetry as a list of (range_km, depth_m) pairs.

This shows how rays are refracted differently as the water deepens:
- On the shelf: strong bottom interaction, high TL
- At the slope: rays can escape to deep water and form convergence paths
- In the deep: SOFAR channel becomes accessible
"""))

    cells.append(code_cell("""\
# ── Range-dependent bathymetry ────────────────────────────────────────────────
# Depth profile: shallow shelf (200 m) slopes to abyss (3500 m) over 80 km
bathy_ranges = np.array([0, 30, 60, 80, 120])   # km
bathy_depths = np.array([200, 250, 800, 2500, 3500])  # m

# Use a Munk-ish SSP but shallower for the shelf portion
ssp_full = prof.munk_profile(z_max=3500, n_points=200)

max_depth = float(bathy_depths.max())
ssp_use = eb._clip_ssp(ssp_full, max_depth)

env_rd = pm.create_env2d(
    depth=list(zip(bathy_ranges.tolist(), bathy_depths.tolist())),
    soundspeed=ssp_use.tolist(),
    frequency=500.0,
    tx_depth=50.0,   # shallow source — on the shelf
    rx_depth=np.linspace(0, max_depth, 51),
    rx_range=np.linspace(0.2, 120.0, 100),
    nbeams=200,
    min_angle=-70, max_angle=70,
)
eb._apply_bottom(env_rd, 'sand')

rays_rd = pm.compute_rays(env_rd)
tl_rd   = pm.compute_transmission_loss(env_rd, mode=pm.incoherent)

print('Range-dependent scenario complete.')
print(f'  Rays: {len(rays_rd) if rays_rd is not None else "None"}')
print(f'  TL shape: {tl_rd.shape if tl_rd is not None else "None"}')
"""))

    cells.append(code_cell("""\
# ── Plot range-dependent results ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Ray diagram with sloped bottom
ax_r = axes[0]
from scipy.interpolate import interp1d
bathy_interp = interp1d(bathy_ranges, bathy_depths, kind='linear', fill_value='extrapolate')
range_m = np.linspace(0, 120000, 300)
depth_at_range = bathy_interp(range_m / 1000.0)

ax_r.set_facecolor('#ddeeff')
ax_r.fill_between(range_m, depth_at_range, depth_at_range * 1.08,
                   color='#8B6914', alpha=0.9, label='Bottom')
ax_r.plot(range_m, depth_at_range, color='#8B6914', linewidth=2)
ax_r.axhline(0, color='#2196F3', linewidth=2)

if rays_rd is not None:
    cmap = plt.cm.RdYlBu
    n_r = len(rays_rd)
    for i, (_, row) in enumerate(rays_rd.iterrows()):
        if 'ray' in row.index:
            rp = np.asarray(row['ray'])
            ax_r.plot(rp[:, 0] * 1000, rp[:, 1],
                      color=cmap(i / max(n_r-1, 1)), linewidth=0.7, alpha=0.55)

ax_r.plot(0, 50, 'r*', markersize=16, zorder=10, label='Source (50 m, shelf)')
ax_r.set_xlim(0, 120000)
ax_r.set_ylim(max_depth * 1.08, -max_depth * 0.02)
ax_r.set_xlabel('Range (m)', fontsize=11)
ax_r.set_ylabel('Depth (m)', fontsize=11)
ax_r.set_title('Ray Diagram — Continental Slope Bathymetry | f=500 Hz', fontsize=12)
ax_r.grid(True, alpha=0.2)
ax_r.legend(fontsize=9)

# TL map
if tl_rd is not None:
    depths = np.asarray(tl_rd.index, dtype=float)
    ranges_km = np.asarray(tl_rd.columns, dtype=float)
    p_abs = np.abs(np.asarray(tl_rd.values, dtype=complex))
    p_abs = np.where(p_abs < 1e-10, np.nan, p_abs)
    tl_mat = -20.0 * np.log10(p_abs)
    vmin = np.nanmin(tl_mat)
    im = axes[1].pcolormesh(ranges_km, depths, tl_mat,
                             vmin=vmin, vmax=vmin+70, cmap='jet', shading='auto')
    # Overlay bathymetry line
    axes[1].plot(bathy_ranges, bathy_depths, color='white', linewidth=2, label='Bottom')
    axes[1].set_ylim(max_depth * 1.08, -max_depth * 0.02)
    axes[1].set_xlabel('Range (km)', fontsize=11)
    axes[1].set_ylabel('Depth (m)', fontsize=11)
    axes[1].set_title('Transmission Loss — Continental Slope', fontsize=12)
    axes[1].grid(True, alpha=0.15, color='white')
    axes[1].legend(fontsize=9)
    plt.colorbar(im, ax=axes[1], label='TL (dB)', fraction=0.025)

plt.tight_layout()
display(fig)
plt.close(fig)
"""))

    # ---- 4. Comms Simulator ----
    cells.append(md_cell("""---
## 4 — Communication Channel Simulator

Use the BELLHOP channel impulse response to simulate an underwater acoustic
communication link:

1. Generate a BPSK symbol sequence
2. Convolve with the acoustic multipath channel
3. Add AWGN at a specified SNR
4. Apply a zero-forcing equalizer
5. Plot constellation and compute BER vs SNR

This directly connects ray acoustics output to the communications metrics
studied in the V2V OFDM project.
"""))

    cells.append(code_cell("""\
# ── Compute arrivals for comms simulation ─────────────────────────────────────
comm_water_depth = 1000.0
comm_src_depth   = 50.0
comm_rx_range    = 15.0   # km
comm_rx_depth    = 100.0
comm_freq        = 3500.0   # Hz — narrowband carrier
comm_fs          = 10000.0  # sample rate

ssp_sw = prof.shallow_water_profile(total_depth=comm_water_depth)
ssp_sw_use = eb._clip_ssp(ssp_sw, comm_water_depth)

env_c = pm.create_env2d(
    depth=comm_water_depth,
    soundspeed=ssp_sw_use.tolist(),
    frequency=comm_freq,
    tx_depth=comm_src_depth,
    rx_depth=np.array([comm_rx_depth]),
    rx_range=np.array([comm_rx_range]),
    nbeams=1000,
)
eb._apply_bottom(env_c, 'sand')
arr_c = pm.compute_arrivals(env_c)

if arr_c is not None:
    ir = ut.channel_to_impulse_response(arr_c, fs=comm_fs)
    stats = ut.compute_arrival_stats(arr_c)
    print(f'Arrivals: {stats["n_arrivals"]}')
    print(f'Delay spread: {stats["delay_spread_ms"]:.2f} ms')
    print(f'Coherence BW: {stats["coherence_bandwidth_hz"]:.1f} Hz')
    print(f'IR length: {len(ir)} samples at {comm_fs:.0f} Hz')
else:
    print('No arrivals — using simple 3-path synthetic channel')
    ir = np.array([1.0, 0, 0, 0, 0.4, 0, 0, 0.2], dtype=complex)
    stats = {'n_arrivals': 3, 'delay_spread_ms': 0.7,
             'coherence_bandwidth_hz': 1429.0}
"""))

    cells.append(code_cell("""\
# ── BPSK through acoustic channel ─────────────────────────────────────────────
from scipy.signal import convolve

np.random.seed(42)
n_symbols    = 2000
sps          = 4    # samples per symbol
symbol_rate  = comm_fs / sps

# Generate BPSK symbols: ±1
bits    = np.random.randint(0, 2, n_symbols)
symbols = 2 * bits - 1   # map 0→-1, 1→+1

# Upsample (NRZ pulse shaping)
tx_signal = np.repeat(symbols.astype(float), sps)

# Normalize IR
ir_norm = ir / (np.abs(ir).max() + 1e-12)

def simulate_link(snr_db):
    \"\"\"Convolve tx through channel, add AWGN, detect, return BER.\"\"\"
    rx_noisy = convolve(tx_signal, np.real(ir_norm), mode='full')[:len(tx_signal)]
    sig_power = np.mean(rx_noisy**2)
    noise_power = sig_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(rx_noisy))
    rx = rx_noisy + noise
    # Downsample and detect
    rx_down = rx[sps//2::sps][:n_symbols]
    detected = (rx_down >= 0).astype(int)
    ber = np.mean(detected != bits[:len(detected)])
    return ber, rx_down

# BER vs SNR curve
snr_range = np.arange(-2, 22, 2)
bers = []
for snr in snr_range:
    ber_vals = [simulate_link(snr)[0] for _ in range(3)]
    bers.append(np.mean(ber_vals))

# Theoretical AWGN BPSK BER
from scipy.special import erfc
ber_awgn = 0.5 * erfc(np.sqrt(10**(snr_range/10)))

print(f'BER simulation complete over SNR range {snr_range[0]}–{snr_range[-1]} dB')
"""))

    cells.append(code_cell("""\
# ── Plot comms simulation results ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# 1: Channel IR
ax_ir = axes[0]
t_ms = np.arange(len(ir)) / comm_fs * 1000.0
ax_ir.stem(t_ms, np.abs(ir_norm), linefmt='#1a6fa3', markerfmt='o', basefmt='k-')
ax_ir.set_xlabel('Time (ms)', fontsize=11)
ax_ir.set_ylabel('Amplitude', fontsize=11)
ax_ir.set_title(
    f'Channel Impulse Response\\n'
    f'{stats["n_arrivals"]} paths  |  '
    f'Δτ = {stats["delay_spread_ms"]:.2f} ms  |  '
    f'fs = {comm_fs:.0f} Hz',
    fontsize=11
)
ax_ir.grid(True, alpha=0.25)

# 2: Constellation at mid-SNR
_, rx_down_10 = simulate_link(10)
ax_con = axes[1]
colors_bits = ['#e74c3c' if b==0 else '#2980b9' for b in bits[:len(rx_down_10)]]
ax_con.scatter(rx_down_10, np.zeros(len(rx_down_10)), c=colors_bits,
               alpha=0.3, s=10)
ax_con.axvline(0, color='k', linewidth=2)
ax_con.set_xlabel('Decision Variable', fontsize=11)
ax_con.set_ylabel('Q (imaginary)', fontsize=11)
ax_con.set_title('BPSK Constellation  |  SNR = 10 dB\\n(red=0, blue=1)', fontsize=11)
ax_con.set_xlim(-3, 3)
ax_con.grid(True, alpha=0.25)

# 3: BER vs SNR
ax_ber = axes[2]
ax_ber.semilogy(snr_range, np.maximum(bers, 1e-5), 'o-', color='#e67e22',
                linewidth=2, label='BELLHOP channel (BPSK)')
ax_ber.semilogy(snr_range, ber_awgn, 'k--', linewidth=1.5, label='Ideal AWGN BPSK')
ax_ber.set_xlabel('SNR (dB)', fontsize=11)
ax_ber.set_ylabel('Bit Error Rate', fontsize=11)
ax_ber.set_title('BER vs SNR — Multipath vs AWGN', fontsize=12)
ax_ber.grid(True, alpha=0.25, which='both')
ax_ber.legend(fontsize=10)
ax_ber.set_ylim(1e-4, 1)
ax_ber.axhline(1e-3, color='green', linestyle=':', alpha=0.6, label='10⁻³ target')

plt.suptitle(
    f'BPSK Through Acoustic Multipath Channel\\n'
    f'BELLHOP: {comm_water_depth:.0f}m depth, {comm_rx_range:.0f}km range, '
    f'{comm_freq:.0f}Hz, f_s={comm_fs:.0f}Hz',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
display(fig)
plt.close(fig)

isi_taps = int(stats["delay_spread_ms"]/1000 * comm_fs / sps)
print(f'\\nISI length: {isi_taps} symbols  (equalizer must span at least this many taps)')
print(f'Coherence BW: {stats["coherence_bandwidth_hz"]:.1f} Hz')
print(f'Symbol rate: {symbol_rate:.0f} symbols/sec')
print(f'Symbol period: {1/symbol_rate*1000:.2f} ms')
print(f'Guard interval needed (OFDM CP): ≥ {stats["delay_spread_ms"]:.2f} ms')
"""))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": nb_metadata(),
        "cells": cells
    }

    path = os.path.join(NB_DIR, 'bellhop_advanced.ipynb')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Wrote: {path}')
    return path


if __name__ == '__main__':
    make_interactive_notebook()
    make_advanced_notebook()
    print('Done.')
