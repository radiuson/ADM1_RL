"""
Scenario Schematic — Illustrative disturbance profiles
=======================================================

Synthetic line plots showing the characteristic disturbance pattern
of each scenario.  No simulation needed.

Panels:
  (a) Feed multiplier  c_f  (loading disturbance)
  (b) Ambient temp     T_env [°C]  (thermal disturbance)

Usage:
    python analysis/plot_scenario_schematic.py [--output-dir /path/to/figures]
"""

import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── time axis ─────────────────────────────────────────────────────────────────
T_DAYS = 30
t = np.linspace(0, T_DAYS, 2880)

# ── scenario colours & labels ─────────────────────────────────────────────────
SCENARIOS = {
    'nominal':          ('Nominal',       '#2166AC'),
    'high_load':        ('High Load',     '#D6604D'),
    'low_load':         ('Low Load',      '#4DAC26'),
    'shock_load':       ('Shock Load',    '#F4A582'),
    'temperature_drop': ('Temp. Drop',    '#762A83'),
    'cold_winter':      ('Cold Winter',   '#E66101'),
}

# ── feed multiplier c_f per scenario ─────────────────────────────────────────
def feed_mult(sc):
    if sc == 'nominal':
        return np.ones_like(t)
    if sc == 'high_load':
        return np.full_like(t, 1.5)
    if sc == 'low_load':
        return np.full_like(t, 0.6)
    if sc == 'shock_load':
        # step up at day 5 → step down at day 15
        y = np.ones_like(t)
        y[(t >= 5) & (t < 15)] = 1.8
        return y
    # thermal scenarios: nominal feed
    return np.ones_like(t)

# ── ambient temperature T_env per scenario ────────────────────────────────────
def t_env(sc):
    if sc == 'cold_winter':
        return np.full_like(t, 5.0)
    if sc == 'temperature_drop':
        # linear ramp from 25 → 20 over day 10–11, then stable
        y = np.full_like(t, 25.0)
        mask = (t >= 10) & (t <= 11)
        y[mask] = 25.0 - 5.0 * (t[mask] - 10.0)
        y[t > 11] = 20.0
        return y
    # other scenarios: stable 25 °C
    return np.full_like(t, 25.0)

# ── build figure ──────────────────────────────────────────────────────────────

COLOR_FEED = '#1a7abf'   # blue  — feed multiplier
COLOR_TEMP = '#cf3b2f'   # red   — ambient temperature
COLOR_BG   = '#f6f8fa'   # GitHub-style background
COLOR_GRID = '#e1e4e8'   # subtle gridlines

# Normalise to [0, 1] across all scenarios so both fit one axis
FEED_MIN, FEED_MAX = 0.6, 1.8
TEMP_MIN, TEMP_MAX = 5.0, 25.0

def _norm_feed(y):
    return (y - FEED_MIN) / (FEED_MAX - FEED_MIN)

def _norm_temp(y):
    return (y - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)


def _draw_thermometer(ax, x_line_end, y_line, color, size=0.075):
    """
    Cartoon thermometer connected to the end of the temperature line.

    x_line_end, y_line : axes-fraction coordinates of the line endpoint.
    The stem's left-centre aligns with the line end so it looks attached.
    """
    from matplotlib.patches import Circle, FancyBboxPatch, Arc
    from matplotlib.transforms import blended_transform_factory

    trans = ax.transAxes
    OUTLINE = '#222222'
    LW      = 2.2          # cartoon outline width (points)

    # geometry (all in axes fraction)
    bulb_r  = size * 0.60
    stem_w  = size * 0.38
    stem_h  = size * 1.80
    cx      = x_line_end + size * 0.10   # centre-x of stem/bulb
    # bulb centre sits at the line y; stem rises above it
    by      = y_line                     # bulb centre y

    # ── bulb ─────────────────────────────────────────────────────────────
    # filled circle (back)
    bulb_bg = Circle((cx, by), radius=bulb_r,
                     transform=trans, facecolor=color, edgecolor=OUTLINE,
                     linewidth=LW, clip_on=False, zorder=11)
    ax.add_patch(bulb_bg)

    # shine dot (cartoon highlight)
    shine = Circle((cx - bulb_r * 0.28, by + bulb_r * 0.32),
                   radius=bulb_r * 0.22,
                   transform=trans, facecolor='white', edgecolor='none',
                   alpha=0.75, clip_on=False, zorder=13)
    ax.add_patch(shine)

    # ── stem ─────────────────────────────────────────────────────────────
    stem_x  = cx - stem_w / 2
    stem_y  = by + bulb_r * 0.35          # overlap with bulb slightly

    stem = FancyBboxPatch(
        (stem_x, stem_y), stem_w, stem_h,
        boxstyle='round,pad=0.008',
        transform=trans, facecolor=color, edgecolor=OUTLINE,
        linewidth=LW, clip_on=False, zorder=10,
    )
    ax.add_patch(stem)

    # white inner tube
    tube_w = stem_w * 0.42
    tube_h = stem_h * 0.78
    tube = FancyBboxPatch(
        (cx - tube_w / 2, stem_y + stem_h * 0.10),
        tube_w, tube_h,
        boxstyle='round,pad=0.004',
        transform=trans, facecolor='white', edgecolor='none',
        clip_on=False, zorder=12,
    )
    ax.add_patch(tube)


def build_one(sc: str, label: str, color: str, output_dir: pathlib.Path) -> None:
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       9,
        'axes.linewidth':  0.6,
    })

    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    yf = _norm_feed(feed_mult(sc))
    yt = _norm_temp(t_env(sc))

    rng = np.random.default_rng(0)

    def fuzzy_line(ax_, x, y, color, lw_main=2.2, n_fuzz=6, zorder=3):
        for i in range(n_fuzz):
            jitter = rng.normal(0, 0.008, size=len(y))
            ax_.plot(x, y + jitter, color=color,
                     lw=lw_main * 0.55, alpha=0.18,
                     solid_capstyle='round', zorder=zorder - 1)
        with plt.rc_context({'path.sketch': (1.2, 6, 1.0)}):
            ax_.plot(x, y, color=color, lw=lw_main,
                     solid_capstyle='round', zorder=zorder)

    # ── blue feed: area fill + fuzzy line ─────────────────────────────────
    ax.fill_between(t, 0, yf,
                    color=COLOR_FEED, alpha=0.18, linewidth=0, zorder=2)
    fuzzy_line(ax, t, yf, COLOR_FEED)

    # ── red temperature fuzzy line ─────────────────────────────────────────
    fuzzy_line(ax, t, yt, COLOR_TEMP)

    ax.set_xlim(0, T_DAYS)
    ax.set_ylim(-0.08, 1.18)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.grid(False)

    # no tick labels, keep tick marks
    ax.tick_params(labelbottom=False, labelleft=False,
                   direction='out', length=3, width=0.6,
                   color='#888888')

    # spines: only bottom + left, light color
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#444d56')
        ax.spines[sp].set_linewidth(1.2)

    # ── thermometer connected to right end of temperature line ────────────
    y_end_data = float(yt[-1])
    ylim = ax.get_ylim()
    y_ax = (y_end_data - ylim[0]) / (ylim[1] - ylim[0])   # axes fraction
    x_ax = 1.0   # right plot edge in axes coords
    _draw_thermometer(ax, x_ax, y_ax, COLOR_TEMP, size=0.09)

    fig.subplots_adjust(left=0.04, right=0.88, top=0.96, bottom=0.08)

    stem = f'scenario_{sc}'
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / f'{stem}.png', bbox_inches='tight', dpi=200)
    print(f'  Saved: {stem}.png')
    plt.close(fig)


def build_figures(output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sc, (label, color) in SCENARIOS.items():
        build_one(sc, label, color, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output-dir', type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1] / 'results' / 'figures' / 'scenarios')
    args = parser.parse_args()
    build_figures(args.output_dir)


if __name__ == '__main__':
    main()
    # also regenerate the combined overview
    import sys
    if '--no-combo' not in sys.argv:
        pass  # combined figure removed; run with --output-dir to choose destination
