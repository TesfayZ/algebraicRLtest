"""Render the paper's figures from the CSVs the experiments already wrote.

Reads nothing but `Results/` and `audit/Results/`, so it never reruns an
experiment and can never disagree with one: if a number moved, rerun the owning
script and then rerun this. Output is vector PDF into `doc/figures/`, sized to
the ICLR text block so nothing is rescaled at \\includegraphics time.

Three figures, each carrying something a table does not.

  shaping_curves     the E3 learning curves. The table reports AUC and final
                     return; only the curve shows that the faulty potentials
                     track the correct one for the whole of training and that
                     the random one never leaves the unshaped baseline.
  amplification      separation against step on the chaotic environment. The
                     claim is a growth *rate* over fifteen decades, which is a
                     slope on a log axis and a run of numbers in prose.
  calibration        false-alarm rate against the nominal level, with Wilson
                     intervals. The claim is "these intervals lie above the
                     line and those lie below", which is what a forest plot
                     shows and a table of interval endpoints does not.

Colours are Okabe-Ito, validated for CVD separation against a white surface.
The mass-error sweep is an ordered magnitude, so it takes a single-hue ramp
and not categorical hues.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "Results")
AUDIT_RESULTS = os.path.join(HERE, "audit", "Results")
FIGDIR = os.path.join(HERE, "doc", "figures")

#: ICLR's text block is 5.5in wide; figures are drawn at that width so that
#: \includegraphics[width=\textwidth] is the identity.
TEXTWIDTH = 5.5

# Scale factors to enlarge the figure while keeping the relative text size.
# This gives more room for the panel (b) title without having it clipped.
WIDTH_SCALE = 6.0 / 5.5      # enlarge width by ~9%
HEIGHT_SCALE = 2.7 / 2.35    # enlarge height by ~15%

# Okabe-Ito. Categorical slots are assigned in fixed order and never cycled.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
GREY = "#6E6E6E"

#: Single-hue ramp, light to dark, for the ordered mass-error sweep.
FAULT_RAMP = ["#F3B78A", "#E08040", "#B44F0A", "#6E2C00"]

INK = "#1a1a1a"
MUTED = "#767676"

# Scale all font sizes so that after the figure is scaled back to \textwidth
# they appear at the original intended point sizes.
SIZE = 8 * WIDTH_SCALE
plt.rcParams.update({
    "font.family": "serif",
    "font.size": SIZE,
    "axes.labelsize": SIZE,
    "axes.titlesize": 8.5 * WIDTH_SCALE,
    "legend.fontsize": 7 * WIDTH_SCALE,
    "xtick.labelsize": 7 * WIDTH_SCALE,
    "ytick.labelsize": 7 * WIDTH_SCALE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "grid.color": "#e2e2e2",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class ReducedInput(Exception):
    """The CSV is a reduced run and cannot support the paper's figure.

    Raised rather than drawn around. A --quick pass writes fewer conditions,
    seeds and null pairs into the same paths; a figure rendered from those
    would carry the paper's own axis labels over data that is not the paper's.
    """


def recess(ax):
    """Grid and spines recede; the data does not."""
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name + ".pdf")
    # Suppressing the creation date makes the output byte-reproducible, so
    # rerunning this on unchanged CSVs leaves the tree clean and a diff on a
    # figure means the data moved.
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  wrote doc/figures/{name}.pdf")
    return path


# --------------------------------------------------------------------------
# Figure 1: E3 learning curves
# --------------------------------------------------------------------------

#: condition key -> label, in the order the panels draw them.
CONTROL_SERIES = [
    ("none", "PPO, unshaped", GREY, 1.2, "-"),
    ("random", "random, matched range", PURPLE, 1.4, "-"),
    ("randstep", "random, matched increment", PURPLE, 1.2, "--"),
    ("rnd", "RND", ORANGE, 1.4, "-"),
    ("discovered", "discovered energy", BLUE, 2.0, "-"),
]

FAULT_SERIES = [
    ("faulty_m2_1.1", r"$m_2$ off by 10%", FAULT_RAMP[0]),
    ("faulty_m2_1.3", r"$m_2$ off by 30%", FAULT_RAMP[1]),
    ("faulty_m2_2", r"$m_2$ off by 100%", FAULT_RAMP[2]),
    ("faulty_m2_4", r"$m_2$ off by 300%", FAULT_RAMP[3]),
]


def aggregate_curves(rows):
    """Aggregate curve rows at the independent unit of each condition.

    The random control is several polynomial draws crossed with PPO seeds.  Its
    independent unit is the polynomial draw, so each draw is averaged across
    seeds before the panel's mean and standard error are formed.  Pooling all
    seed runs would treat ten evaluations of one polynomial as ten independent
    random-potential draws and understate the uncertainty the control is meant
    to show.
    """
    steps = sorted(
        int(k.split("_")[1]) for k in rows[0] if k.startswith("curve_")
    )
    out = {}
    random = {}
    for row in rows:
        cond = row["condition"]
        if cond.startswith("random_") or cond.startswith("randstep_"):
            base, draw = cond.rsplit("_", 1)
            random.setdefault(base, {}).setdefault(draw, []).append(
                [float(row[f"curve_{s}"]) for s in steps])
        else:
            out.setdefault(cond, []).append(
                [float(row[f"curve_{s}"]) for s in steps])
    curves = {}
    for cond, seeds in out.items():
        a = np.asarray(seeds, dtype=float)
        curves[cond] = (
            np.asarray(steps, dtype=float),
            a.mean(axis=0),
            a.std(axis=0, ddof=1) / np.sqrt(a.shape[0]),
        )
    for cond, by_draw in random.items():
        draw_means = np.asarray([np.mean(seed_curves, axis=0)
                                 for seed_curves in by_draw.values()], dtype=float)
        curves[cond] = (
            np.asarray(steps, dtype=float),
            draw_means.mean(axis=0),
            draw_means.std(axis=0, ddof=1) / np.sqrt(draw_means.shape[0]),
        )
    return curves


def load_curves():
    """Load and aggregate E3 curves from the committed results CSV."""
    return aggregate_curves(read_csv(os.path.join(RESULTS, "shaping.csv")))


def figure_shaping():
    curves = load_curves()
    wanted = [c for c, *_ in CONTROL_SERIES] + [c for c, *_ in FAULT_SERIES]
    missing = [c for c in wanted if c not in curves]
    if missing:
        raise ReducedInput(
            "Results/shaping.csv is missing " + ", ".join(missing)
            + ". The four faulty conditions come from E3 under --param-sweep; "
              "without it they collapse to one."
        )

    last = max(curves["none"][0])

    # Enlarge the figure so the title of panel (b) fits comfortably.
    # The figure will be scaled down at inclusion time, but font sizes
    # have been increased proportionally so they appear as intended.
    width = TEXTWIDTH * WIDTH_SCALE
    height = 2.35 * HEIGHT_SCALE

    # Give more horizontal room to panel (b) – its title is longer.
    # Also keep enough space between panels so y-label of (b) doesn't overlap.
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(width, height),
        gridspec_kw={"width_ratios": [0.40, 0.60]})

    for ax in (axa, axb):
        recess(ax)
        ax.set_xlabel("environment steps")
        ax.set_xlim(0, last * 1.07)
        ticks = [t for t in (0, last / 3, 2 * last / 3, last)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            ["0"] + [f"{t / 1000:.0f}k" for t in ticks[1:]]
        )

    # Panel A: the controls. Identity, so categorical hues.
    lo_a, hi_a = np.inf, -np.inf
    for cond, label, colour, lw, ls in CONTROL_SERIES:
        steps, mean, se = curves[cond]
        axa.fill_between(steps, mean - se, mean + se, color=colour,
                         alpha=0.13, linewidth=0)
        axa.plot(steps, mean, color=colour, linewidth=lw, linestyle=ls,
                 label=label, zorder=3)
        lo_a = min(lo_a, float(np.min(mean - se)))
        hi_a = max(hi_a, float(np.max(mean + se)))
    axa.set_ylabel("unshaped return")
    # Limits from the data, not from a literal. The random control sits at the
    # environment's truncation limit, which is the panel's most important line,
    # and any hardcoded floor tuned to one run puts it off the axis.
    pad = 0.06 * (hi_a - lo_a)
    axa.set_ylim(lo_a - pad, hi_a + pad)
    axa.set_title("(a) shaping vs controls", loc="left")
    # Legend at lower right, but with tighter spacing and smaller font to avoid touching y-axis.
    axa.legend(loc="lower right", frameon=False, handlelength=1.2,
               fontsize=6.0 * WIDTH_SCALE, labelspacing=0.2, borderaxespad=0.1)

    # Panel B: the mass-error sweep, drawn against the correct potential
    # rather than against the axis. Plotting the raw curves would put five
    # overlapping lines on one panel and show only that they overlap; the
    # question is whether each sits inside the seed-to-seed spread of the
    # correct one, which is what the band is.
    steps, ref, ref_se = curves["discovered"]
    axb.fill_between(steps, -ref_se, ref_se, color=BLUE, alpha=0.16,
                     linewidth=0,
                     label="correct energy, $\\pm 1$ s.e. over seeds")
    axb.plot(steps, np.zeros_like(steps), color=BLUE, linewidth=1.6, zorder=4)
    for cond, label, colour in FAULT_SERIES:
        steps, mean, _ = curves[cond]
        axb.plot(steps, mean - ref, color=colour, linewidth=1.3, label=label,
                 zorder=3)
    axb.set_ylabel("return, minus correct potential")
    axb.set_title("(b) faulty potentials vs correct energy", loc="left")
    lo_b = min(float(np.min(curves[c][1] - ref)) for c, *_ in FAULT_SERIES)
    hi_b = max(float(np.max(curves[c][1] - ref)) for c, *_ in FAULT_SERIES)
    span = hi_b - lo_b
    axb.set_ylim(lo_b - 0.06 * span, hi_b + 0.45 * span)   # headroom for legend
    axb.legend(loc="upper right", frameon=False, handlelength=1.6,
               fontsize=6.3 * WIDTH_SCALE, labelspacing=0.3)

    # Adjust gap to avoid overlap of y-label of (b) with (a)
    fig.subplots_adjust(wspace=0.28)
    return save(fig, "shaping_curves")


# --------------------------------------------------------------------------
# Figure 2: separation growth on the chaotic environment
# --------------------------------------------------------------------------

#: Everything at or below this is the float64 floor of the comparison; a
#: bitwise-identical pair reports exactly zero and is drawn on it.
SEP_FLOOR = 1e-17


def figure_amplification():
    rows = read_csv(os.path.join(AUDIT_RESULTS, "amplification.csv"))
    traces = {}
    for row in rows:
        key = (row["env_id"], row["older"], row["newer"])
        traces.setdefault(key, []).append(
            (int(row["step"]), float(row["separation"]))
        )

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.4))
    recess(ax)
    ax.grid(True, axis="both")

    styles = {"InvertedDoublePendulum-v4": (VERMILLION, "InvertedDoublePendulum"),
              "Reacher-v4": (BLUE, "Reacher")}
    seen = set()
    dashes = {}
    rates = []
    for (env, older, newer), pts in sorted(traces.items()):
        pts.sort()
        step = np.array([p[0] for p in pts], dtype=float)
        raw = np.array([p[1] for p in pts], dtype=float)
        sep = np.clip(raw, SEP_FLOOR, None)
        colour, label = styles[env]
        # Colour carries the environment; the release pair is the secondary
        # dimension and takes a line style, never a second hue.
        ls = "-" if dashes.setdefault(env, 0) == 0 else (0, (4, 2))
        dashes[env] += 1
        ax.plot(step, sep, color=colour, linestyle=ls, linewidth=1.5,
                label=label if env not in seen else None, zorder=3)
        seen.add(env)
        # A bitwise-identical pair reports exactly zero, which a log axis
        # cannot place. Drawn on the floor it would read as "separates by
        # 1e-17", which is weaker than the truth, so say so in the figure and
        # not only in the caption.
        if not np.any(raw > 0):
            ax.text(step[-1] * 0.55, SEP_FLOOR * 0.62, "bitwise identical",
                    ha="center", va="top", fontsize=6.5, color=colour)
        # Steps per decade, measured off the same trace the reader is looking
        # at instead of typed in beside it. The window and the two-point form
        # are the audit script's, so the annotation and the prose cannot
        # drift apart; a least-squares slope over the same span is a different
        # number and would.
        lo, hi = 50.0, 400.0
        if env.startswith("InvertedDoublePendulum") and {lo, hi} <= set(step):
            a = raw[step == lo][0]
            b = raw[step == hi][0]
            if a > 0 and b > 0:
                decades = np.log10(b / a) / (hi - lo)
                if decades > 0:
                    rates.append(1.0 / decades)

    ax.set_yscale("log")
    ax.set_ylim(3e-18, 1e2)
    ax.set_xlim(0, 500)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("max separation between releases")

    ax.axhline(1e1, color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)),
               zorder=2)
    ax.text(497, 1.4e1, "observation clip", ha="right", va="bottom",
            fontsize=6.5, color=MUTED)
    ax.axhline(2.220446049250313e-16, color=MUTED, linewidth=0.7,
               linestyle=(0, (1, 2)), zorder=2)
    ax.text(497, 3.2e-16, "one ulp at step 0", ha="right", va="bottom",
            fontsize=6.5, color=MUTED)

    lo, hi = min(rates), max(rates)
    rate_text = (f"a decade per {lo:.0f} steps" if hi - lo < 1.5
                 else f"a decade per {lo:.0f} to {hi:.0f} steps")
    ax.annotate(rate_text, xy=(230, 1e-4), xytext=(300, 2e-8),
                fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))

    ax.legend(loc="upper left", frameon=False, handlelength=1.8)
    return save(fig, "amplification")


# --------------------------------------------------------------------------
# Figure 3: false-alarm calibration
# --------------------------------------------------------------------------

ALPHA = 0.01


def figure_calibration():
    rows = read_csv(os.path.join(RESULTS, "false_alarm_summary.csv"))
    n_pairs = {int(r["n_pairs"]) for r in rows}
    if len(n_pairs) != 1:
        raise ReducedInput(
            f"Results/false_alarm_summary.csv mixes pair counts {sorted(n_pairs)}; "
            "one axis label cannot describe all of them."
        )
    n_pairs = n_pairs.pop()

    order, seen = [], set()
    for row in rows:
        key = (row["system"], row["noise"], row["floor"])
        if key not in seen:
            seen.add(key)
            order.append(key)

    by_key = {}
    for row in rows:
        by_key[(row["system"], row["noise"], row["floor"], row["unit"])] = row

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.7))
    recess(ax)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    units = [("block", VERMILLION, "per block", -0.16),
             ("trajectory", BLUE, "per trajectory", 0.16)]

    labels = []
    for i, (system, noise, floor) in enumerate(order):
        y = len(order) - 1 - i
        if float(noise) == 0:
            sigma = r"\sigma = 0"
        else:
            sigma = rf"\sigma = 10^{{{int(round(np.log10(float(noise))))}}}"
        fl = r"$\rho_{\min}$" if float(floor) > 0 else "no floor"
        labels.append((y, f"{system.capitalize()}, ${sigma}$, {fl}"))
        for unit, colour, label, off in units:
            row = by_key[(system, noise, floor, unit)]
            rate = float(row["family_rate"])
            lo, hi = float(row["wilson_lo"]), float(row["wilson_hi"])
            above = lo > ALPHA
            ax.plot([lo, hi], [y + off] * 2, color=colour, linewidth=1.3,
                    solid_capstyle="butt", zorder=3)
            ax.plot([rate], [y + off], marker="o", markersize=4.5,
                    color=colour, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=4,
                    label=label if i == 0 else None)
            ax.text(hi + 0.0035, y + off, f"{rate:.3f}", va="center",
                    fontsize=6.5,
                    color=INK if above else MUTED,
                    fontweight="bold" if above else "normal")

    ax.axvline(ALPHA, color=INK, linewidth=0.9, linestyle=(0, (3, 2)),
               zorder=2)
    ax.text(ALPHA + 0.0015, len(order) - 0.35, r"nominal $\alpha = 0.01$",
            fontsize=6.5, color=INK)

    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([t for _, t in labels])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(order) - 0.15)
    ax.set_xlim(-0.002, 0.148)
    ax.set_xlabel(f"family-wise false-alarm rate over {n_pairs} healthy pairs, "
                  "Wilson 95%")
    ax.legend(loc="lower right", frameon=False, handlelength=0.8,
              numpoints=1)
    return save(fig, "calibration")


def main():
    print("Rendering paper figures from committed CSVs...")
    try:
        figure_shaping()
        figure_amplification()
        figure_calibration()
    except ReducedInput as exc:
        print(f"\nrefusing to draw: {exc}")
        print("Rerun the owning experiment at the paper's settings "
              "(run_all_experiments.py without --quick), then rerun this.")
        raise SystemExit(1)
    print("Done.")


if __name__ == "__main__":
    main()