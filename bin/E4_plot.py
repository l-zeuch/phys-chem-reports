#!/usr/bin/env python3

# built in
import argparse
import contextlib
import csv
import io
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# 3rd party
import numpy as np
from matplotlib import pyplot

# globals
# Adjust these paths as needed. The program as-is assumes your data is in:
# data/CV_XXXmVs[_stirred]/Voltammogram/Current vs Potential.csv
PATH_PREFIX = "E4/data/"
NAME_PREFIX = "CV_"
PATH_POSTFIX = "/Voltammogram/Current vs Potential.csv"
FIT_WINDOWS = {
    # These windows fitted my data well. Adjust as needed. (Look at the plot!)
    # For the forward scan, I don't recommend going too close to your lowest scan potential.
    # The same applies to the reverse scan, of course.
    "forward": (0.0, 200.0, "pos"),
    "reverse": (600.0, 700.0, "neg"),
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot voltammogram data from CSV files, and analyse them.\n"
            f'This program assumes you have your recorded data in a directory "{PATH_PREFIX}",\n'
            f'with the actual CSV files located in "{PATH_PREFIX}{NAME_PREFIX}XXXmVs[_stirred]{PATH_POSTFIX}".\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("scanrate", type=str, help='Scanrate to plot (numeric part, e.g. "100" for 100 mVs).')
    parser.add_argument(
        "--stirred",
        action="store_true",
        help="Indicate if the solution was stirred during the measurement.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the generated report to a text file in addition to printing it.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show the plot of the data.",
    )
    args = parser.parse_args()

    potentials, currents = load_file(args.scanrate.zfill(3), args.stirred)
    runs_potentials, runs_currents = find_runs(potentials, currents)

    extremas = compute_extrema_stats(runs_potentials, runs_currents)
    fits = compute_linear_fits(runs_potentials, runs_currents)
    peaks = analyze_peaks(extremas, fits)

    report = render_report(args.scanrate, args.stirred, extremas, peaks, fits)
    print(report)
    if args.save:
        fname = f"CV_report_{args.scanrate}mVs"
        if args.stirred:
            fname += "_stirred"
        fname += ".txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {fname}")

    if args.plot:
        ctx = PlotContext(
            potentials=runs_potentials,
            currents=runs_currents,
            scanrate=args.scanrate,
            stirred=args.stirred,
            extremas=extremas,
            fits=fits,
            peaks=peaks,
        )
        plot_data(ctx)


def render_report(scanrate: str, stirred: bool, extremas, peaks, fits) -> str:
    """Capture the printed report as a string."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"Report for scanrate {scanrate} mV/s {'(stirred)' if stirred else ''}")
        print(f"  Detected {len(extremas.runs)} runs.")
        print_extrema_report(extremas)
        print_analysis_report(scanrate, extremas, peaks)
        print_fit_report(fits)
    return buf.getvalue()

# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class RunExtrema:
    max_current: float | None
    max_pot: float | None
    min_current: float | None
    min_pot: float | None


@dataclass
class ExtremaStats:
    runs: List[RunExtrema]
    avg_max_current: float
    err_avg_max_current: float
    avg_min_current: float
    err_avg_min_current: float
    avg_max_pot: float
    err_avg_max_pot: float
    avg_min_pot: float
    err_avg_min_pot: float
    halfway_pot: float
    err_halfway_pot: float


@dataclass
class FitAvg:
    slope: float | None
    intercept: float | None
    slope_sem: float | None
    intercept_sem: float | None
    x_min: float
    x_max: float


@dataclass
class FitWindowResult:
    lo: float
    hi: float
    direction: str
    per_run: List[Dict[str, float | None]]
    avg: FitAvg


@dataclass
class PeakAnalysis:
    i_pa_raw: float | None
    i_pc_raw: float | None
    fit_at_pa: float | None
    fit_at_pc: float | None
    i_pa_bg: float | None
    i_pc_bg: float | None
    i_pa_bg_err: float | None
    i_pc_bg_err: float | None
    ratio_bg: float | None
    ratio_bg_err: float | None


@dataclass
class PlotContext:
    potentials: List[List[float]]
    currents: List[List[float]]
    scanrate: str
    stirred: bool
    extremas: ExtremaStats
    fits: Dict[str, FitWindowResult]
    peaks: PeakAnalysis


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def load_file(name: str, stirred: bool = False) -> Tuple[List[float], List[float]]:
    fname = _make_fname(name, stirred)
    return _read_csv(fname)


def _make_fname(name: str, stirred: bool) -> str:
    fname = f"{PATH_PREFIX}{NAME_PREFIX}{name}mVs"
    if stirred:
        fname += "_stirred"
    fname += PATH_POSTFIX
    return fname


def _read_csv(fname: str) -> Tuple[List[float], List[float]]:
    potentials: List[float] = []
    currents: List[float] = []
    with open(fname, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header if present
        for row in reader:
            if len(row) < 2:
                continue
            potentials.append(float(row[0]))
            currents.append(float(row[1]))
    return potentials, currents


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #

def _sem(data: Sequence[float]) -> float:
    """Return standard error of the mean (s / sqrt(n)). 0.0 for n <= 1."""
    n = len(data)
    if n <= 1:
        return 0.0
    return statistics.stdev(data) / math.sqrt(n)


def find_runs(potentials: Sequence, currents: Sequence) -> Tuple[List[List[float]], List[List[float]]]:
    """Split potentials/currents into runs based on potential crossing zero (with smoothing + min separation)."""
    pot_arr = np.asarray(potentials, dtype=float)
    n = len(pot_arr)
    if n == 0:
        return [], []

    window = 11 if n >= 11 else (5 if n >= 5 else 3)

    kernel = np.ones(window) / window
    smoothed = np.convolve(pot_arr, kernel, mode="same")

    candidate_crossings: List[int] = [
        i for i in range(1, n) if (smoothed[i - 1] < 0 and smoothed[i] >= 0)
    ]

    min_sep = max(5, window)
    crossings: List[int] = []
    last = -min_sep * 2
    for idx in candidate_crossings:
        if idx - last >= min_sep:
            crossings.append(idx)
            last = idx

    splits: List[Tuple[int, int]] = []
    start = 0
    for idx in crossings:
        splits.append((start, idx + 1))
        start = idx
    splits.append((start, n))

    potentials_runs = _apply_splits(potentials, splits)
    currents_runs = _apply_splits(currents, splits)
    return potentials_runs, currents_runs


def _apply_splits(lst: Sequence, splits: List[Tuple[int, int]]) -> List[List]:
    """Apply index splits to a sequence and return list-of-lists slices."""
    return [list(lst[start:end]) for (start, end) in splits]


def compute_extrema_stats(runs_potentials: List[List[float]], runs_currents: List[List[float]]) -> ExtremaStats:
    """Compute per-run maxima/minima and their potentials, plus averages and SEM."""
    runs_data: List[RunExtrema] = []
    for pot_run, cur_run in zip(runs_potentials, runs_currents):
        if not cur_run:
            runs_data.append(RunExtrema(None, None, None, None))
            continue
        max_idx = int(np.argmax(cur_run))
        min_idx = int(np.argmin(cur_run))
        runs_data.append(
            RunExtrema(
                max_current=float(cur_run[max_idx]),
                max_pot=float(pot_run[max_idx]),
                min_current=float(cur_run[min_idx]),
                min_pot=float(pot_run[min_idx]),
            )
        )

    max_currents = [r.max_current for r in runs_data if r.max_current is not None]
    min_currents = [r.min_current for r in runs_data if r.min_current is not None]
    max_pots = [r.max_pot for r in runs_data if r.max_pot is not None]
    min_pots = [r.min_pot for r in runs_data if r.min_pot is not None]

    return ExtremaStats(
        runs=runs_data,
        avg_max_current=statistics.fmean(max_currents) if max_currents else 0.0,
        err_avg_max_current=_sem(max_currents),
        avg_min_current=statistics.fmean(min_currents) if min_currents else 0.0,
        err_avg_min_current=_sem(min_currents),
        avg_max_pot=statistics.fmean(max_pots) if max_pots else 0.0,
        err_avg_max_pot=_sem(max_pots),
        avg_min_pot=statistics.fmean(min_pots) if min_pots else 0.0,
        err_avg_min_pot=_sem(min_pots),
        halfway_pot=statistics.fmean(max_pots + min_pots) if (max_pots and min_pots) else 0.0,
        err_halfway_pot=_sem(max_pots + min_pots),
    )


# --------------------------------------------------------------------------- #
# Linear fits
# --------------------------------------------------------------------------- #

def compute_linear_fits(
    runs_potentials: List[List[float]], runs_currents: List[List[float]]
) -> Dict[str, FitWindowResult]:
    """Fit straight lines per run in configured windows with direction filtering."""
    fit_results: Dict[str, FitWindowResult] = {}

    for wname, (lo, hi, direction) in FIT_WINDOWS.items():
        per_run = []
        slopes = []
        intercepts = []
        x_min = None
        x_max = None

        for pot_run, cur_run in zip(runs_potentials, runs_currents):
            m, b, r2, x_used = _fit_linear_window(pot_run, cur_run, lo, hi, direction)
            per_run.append({"slope": m, "intercept": b, "r2": r2})
            if m is not None and b is not None and len(x_used) > 0:
                slopes.append(m)
                intercepts.append(b)
                local_min = float(np.min(x_used))
                local_max = float(np.max(x_used))
                x_min = local_min if x_min is None else min(x_min, local_min)
                x_max = local_max if x_max is None else max(x_max, local_max)

        avg_slope = statistics.fmean(slopes) if slopes else None
        avg_intercept = statistics.fmean(intercepts) if intercepts else None
        slope_sem = _sem(slopes) if slopes else None
        intercept_sem = _sem(intercepts) if intercepts else None

        fit_results[wname] = FitWindowResult(
            lo=lo,
            hi=hi,
            direction=direction,
            per_run=per_run,
            avg=FitAvg(
                slope=avg_slope,
                intercept=avg_intercept,
                slope_sem=slope_sem,
                intercept_sem=intercept_sem,
                x_min=x_min if x_min is not None else lo,
                x_max=x_max if x_max is not None else hi,
            ),
        )

    return fit_results


def _fit_linear_window(
    pot_run: Sequence[float],
    cur_run: Sequence[float],
    lo: float,
    hi: float,
    direction: str,
) -> Tuple[float | None, float | None, float | None, np.ndarray]:
    """Fit y = m x + b on points with lo <= x <= hi, filtered by sweep direction."""
    pot_arr = np.asarray(pot_run, dtype=float)
    cur_arr = np.asarray(cur_run, dtype=float)

    if len(pot_arr) < 2:
        return None, None, None, np.array([])

    deriv = np.gradient(pot_arr)
    dir_mask = deriv >= 0 if direction == "pos" else deriv <= 0
    bounds_mask = (pot_arr >= lo) & (pot_arr <= hi)
    mask = dir_mask & bounds_mask

    if np.count_nonzero(mask) < 2:
        mask = bounds_mask

    x_used = pot_arr[mask]
    y_used = cur_arr[mask]
    if x_used.size < 2:
        return None, None, None, np.array([])

    m, b = np.polyfit(x_used, y_used, 1)
    y_pred = m * x_used + b
    ss_res = float(np.sum((y_used - y_pred) ** 2))
    ss_tot = float(np.sum((y_used - np.mean(y_used)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(m), float(b), r2, x_used


# --------------------------------------------------------------------------- #
# Peak analysis (background-corrected peaks with error propagation)
# --------------------------------------------------------------------------- #

def analyze_peaks(extremas: ExtremaStats, fits: Dict[str, FitWindowResult]) -> PeakAnalysis:
    out = PeakAnalysis(
        i_pa_raw=extremas.avg_max_current,
        i_pc_raw=extremas.avg_min_current,
        fit_at_pa=None,
        fit_at_pc=None,
        i_pa_bg=None,
        i_pc_bg=None,
        i_pa_bg_err=None,
        i_pc_bg_err=None,
        ratio_bg=None,
        ratio_bg_err=None,
    )

    sigma_I_pa = extremas.err_avg_max_current or 0.0
    sigma_I_pc = extremas.err_avg_min_current or 0.0
    sigma_x_pa = extremas.err_avg_max_pot or 0.0
    sigma_x_pc = extremas.err_avg_min_pot or 0.0

    # Forward (anodic) peak vs forward fit
    fwd = fits.get("forward")
    if fwd and fwd.avg.slope is not None and fwd.avg.intercept is not None and out.i_pa_raw is not None:
        m = fwd.avg.slope
        b = fwd.avg.intercept
        sigma_m = fwd.avg.slope_sem or 0.0
        sigma_b = fwd.avg.intercept_sem or 0.0
        x_pa = extremas.avg_max_pot
        fit_pa = m * x_pa + b
        out.fit_at_pa = fit_pa
        diff_pa = out.i_pa_raw - fit_pa
        out.i_pa_bg = float(np.abs(diff_pa))
        var_pa = (sigma_I_pa ** 2) + (x_pa * sigma_m) ** 2 + (m * sigma_x_pa) ** 2 + (sigma_b ** 2)
        out.i_pa_bg_err = float(np.sqrt(var_pa)) if var_pa > 0 else 0.0

    # Reverse (cathodic) peak vs reverse fit
    rev = fits.get("reverse")
    if rev and rev.avg.slope is not None and rev.avg.intercept is not None and out.i_pc_raw is not None:
        m = rev.avg.slope
        b = rev.avg.intercept
        sigma_m = rev.avg.slope_sem or 0.0
        sigma_b = rev.avg.intercept_sem or 0.0
        x_pc = extremas.avg_min_pot
        fit_pc = m * x_pc + b
        out.fit_at_pc = fit_pc
        diff_pc = out.i_pc_raw - fit_pc
        out.i_pc_bg = float(np.abs(diff_pc))
        var_pc = (sigma_I_pc ** 2) + (x_pc * sigma_m) ** 2 + (m * sigma_x_pc) ** 2 + (sigma_b ** 2)
        out.i_pc_bg_err = float(np.sqrt(var_pc)) if var_pc > 0 else 0.0

    # Ratio of background-corrected peaks
    ipa, ipc = out.i_pa_bg, out.i_pc_bg
    ipa_err, ipc_err = out.i_pa_bg_err, out.i_pc_bg_err
    if ipa not in (None, 0) and ipc not in (None, 0) and ipa_err is not None and ipc_err is not None:
        out.ratio_bg = ipa / ipc
        rel_var = (ipa_err / ipa) ** 2 + (ipc_err / ipc) ** 2
        out.ratio_bg_err = float(np.sqrt(rel_var)) * out.ratio_bg if rel_var > 0 else 0.0

    return out


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

def print_extrema_report(extremas: ExtremaStats) -> None:
    """Print per-run extrema and the averaged value with SEM."""
    print("\n=== Extrema Summary ===")
    for i, r in enumerate(extremas.runs, start=1):
        if r.max_current is None:
            print(f"  Run {i}: no data")
            continue
        print(
            f"  Run {i}: "
            f"max = {r.max_current:.2f} µA at {r.max_pot:.2f} mV | "
            f"min = {r.min_current:.2f} µA at {r.min_pot:.2f} mV"
        )
    print(
        "  Avg Max Current: "
        f"{extremas.avg_max_current:.2f} ± {extremas.err_avg_max_current:.2f} µA "
        f"at {extremas.avg_max_pot:.2f} ± {extremas.err_avg_max_pot:.2f} mV"
    )
    print(
        "  Avg Min Current: "
        f"{extremas.avg_min_current:.2f} ± {extremas.err_avg_min_current:.2f} µA "
        f"at {extremas.avg_min_pot:.2f} ± {extremas.err_avg_min_pot:.2f} mV"
    )


def print_fit_report(fits: Dict[str, FitWindowResult]) -> None:
    """Print per-run fit parameters and averages."""
    for wname, info in fits.items():
        lo, hi = info.lo, info.hi
        print(
            f"\n=== Linear fit: {wname} ===\n"
            f"  Window: [{lo:.0f}, {hi:.0f}] mV (dir: {info.direction})"
        )
        for i, run_fit in enumerate(info.per_run, start=1):
            m, b, r2 = run_fit["slope"], run_fit["intercept"], run_fit["r2"]
            if m is None:
                print(f"  Run {i}: insufficient points in window")
            else:
                print(f"  Run {i}: slope = {m:.4f} µA/mV, intercept = {b:.4f} µA, R² = {r2:.4f}")
        avg = info.avg
        if avg.slope is not None:
            sem_m = avg.slope_sem or 0.0
            sem_b = avg.intercept_sem or 0.0
            print(
                f"  Avg: slope = {avg.slope:.4f} ± {sem_m:.4f} µA/mV, "
                f"intercept = {avg.intercept:.4f} ± {sem_b:.4f} µA"
            )
        else:
            print("  Avg: no data")


def print_analysis_report(scanrate: str, extremas: ExtremaStats, peaks: PeakAnalysis) -> None:
    max_pot = extremas.avg_max_pot
    min_pot = extremas.avg_min_pot
    pot_diff = max_pot - min_pot
    halfway = extremas.halfway_pot

    ipa = peaks.i_pa_bg
    ipc = peaks.i_pc_bg
    ipa_err = peaks.i_pa_bg_err
    ipc_err = peaks.i_pc_bg_err
    ratio = peaks.ratio_bg
    ratio_err = peaks.ratio_bg_err

    print("\n=== Analysis Report ===")
    print(f"  Scanrate: {scanrate} mV/s")
    print(f"  Max current potential (anodic): {max_pot:.2f} ± {extremas.err_avg_max_pot:.2f} mV")
    print(f"  Min current potential (cathodic): {min_pot:.2f} ± {extremas.err_avg_min_pot:.2f} mV")
    print(
        f"  ΔE (max - min): {pot_diff:.2f} ± "
        f"{extremas.err_avg_max_pot + extremas.err_avg_min_pot:.2f} mV"
    )
    if halfway is not None:
        print(
            f"  Halfway potential: {halfway:.2f} ± {extremas.err_halfway_pot:.2f} mV"
        )

    if ipa is not None:
        if ipa_err is not None:
            print(f"  i_pa (background-corrected): {ipa:.3f} ± {ipa_err:.3f} µA")
        else:
            print(f"  i_pa (background-corrected): {ipa:.3f} µA")

    if ipc is not None:
        if ipc_err is not None:
            print(f"  i_pc (background-corrected): {ipc:.3f} ± {ipc_err:.3f} µA")
        else:
            print(f"  i_pc (background-corrected): {ipc:.3f} µA")

    if ratio is not None:
        if ratio_err is not None:
            print(f"  i_pa / i_pc: {ratio:.3f} ± {ratio_err:.3f}")
        else:
            print(f"  i_pa / i_pc: {ratio:.3f}")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_data(ctx: PlotContext) -> None:
    """Plot runs, annotate averages, and overlay average linear fits with SEM bands."""
    potentials = ctx.potentials
    currents = ctx.currents
    extremas = ctx.extremas
    fits = ctx.fits
    peaks = ctx.peaks
    pyplot.xlabel("Potential (mV)")
    pyplot.ylabel("Current (µA)")
    pyplot.grid(True)

    for i in range(len(potentials)):
        pyplot.plot(potentials[i], currents[i])

    # annotate average extrema (maximum) with error bars
    pyplot.annotate(
        f'Max Current: ({extremas.avg_max_current:.2f} ± {extremas.err_avg_max_current:.2f}) µA\n'
        f'at ({extremas.avg_max_pot:.2f} ± {extremas.err_avg_max_pot:.2f}) mV',
        xy=(extremas.avg_max_pot, extremas.avg_max_current),
        xytext=(extremas.avg_max_pot + 20, extremas.avg_max_current),
        fontsize=8,
    )
    pyplot.errorbar(
        extremas.avg_max_pot,
        extremas.avg_max_current,
        yerr=float(extremas.err_avg_max_current),
        capsize=6,
        color="black",
    )

    # annotate average extrema (minimum) with error bars
    pyplot.annotate(
        f'Min Current: ({extremas.avg_min_current:.2f} ± {extremas.err_avg_min_current:.2f}) µA\n'
        f'at ({extremas.avg_min_pot:.2f} ± {extremas.err_avg_min_pot:.2f}) mV',
        xy=(extremas.avg_min_pot, extremas.avg_min_current),
        xytext=(extremas.avg_min_pot - 160, extremas.avg_min_current),
        fontsize=8,
    )
    pyplot.errorbar(
        extremas.avg_min_pot,
        extremas.avg_min_current,
        yerr=float(extremas.err_avg_min_current),
        capsize=6,
        color="black",
    )

    # overlay average fit lines and ±SEM bands
    colors = {"forward": "gray", "reverse": "brown"}
    for wname, info in fits.items():
        avg = info.avg
        if avg.slope is None or avg.intercept is None:
            continue
        if wname == "forward":
            x_start = avg.x_min
            x_end = extremas.avg_max_pot
        else:
            x_start = avg.x_max
            x_end = extremas.avg_min_pot
        xfit = np.linspace(min(x_start, x_end), max(x_start, x_end), 50)
        yfit = avg.slope * xfit + avg.intercept
        base_color = colors.get(wname, "k")
        pyplot.plot(xfit, yfit, "--", color=base_color, linewidth=1.5, label=f"Avg Fit: {wname} ± SEM\n y = ({avg.slope:.4f} ± {avg.slope_sem:.4f})x + ({avg.intercept:.4f} ± {avg.intercept_sem:.4f})")
        # ±SEM bands
        if avg.slope_sem is not None and avg.intercept_sem is not None:
            y_hi = (avg.slope + avg.slope_sem) * xfit + (avg.intercept + avg.intercept_sem)
            y_lo = (avg.slope - avg.slope_sem) * xfit + (avg.intercept - avg.intercept_sem)
            pyplot.plot(xfit, y_hi, ":", color=base_color, alpha=0.45, linewidth=1.0)
            pyplot.plot(xfit, y_lo, ":", color=base_color, alpha=0.45, linewidth=1.0)

    # mark background-corrected peak offsets (vertical lines from fit to peak)
    # forward (anodic)
    fwd = fits.get("forward")
    if fwd and fwd.avg.slope is not None and fwd.avg.intercept is not None:
        x_pa = extremas.avg_max_pot
        y_fit_pa = fwd.avg.slope * x_pa + fwd.avg.intercept
        y_peak_pa = extremas.avg_max_current
        diff_pa = peaks.i_pa_bg
        diff_pa_err = peaks.i_pa_bg_err
        pyplot.plot([x_pa, x_pa], [y_fit_pa, y_peak_pa], color="gray", linewidth=2, alpha=0.7)
        if diff_pa is not None and diff_pa_err is not None:
            pyplot.annotate(
                f"i_pa = ({diff_pa:.3f} ± {diff_pa_err:.3f}) µA",
                xy=(x_pa, (y_fit_pa + y_peak_pa) / 2),
                xytext=(x_pa + 20, (y_fit_pa + y_peak_pa) / 2),
                fontsize=8,
                color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

    # reverse (cathodic)
    rev = fits.get("reverse")
    if rev and rev.avg.slope is not None and rev.avg.intercept is not None:
        x_pc = extremas.avg_min_pot
        y_fit_pc = rev.avg.slope * x_pc + rev.avg.intercept
        y_peak_pc = extremas.avg_min_current
        diff_pc = peaks.i_pc_bg
        diff_pc_err = peaks.i_pc_bg_err
        pyplot.plot([x_pc, x_pc], [y_fit_pc, y_peak_pc], color="brown", linewidth=2, alpha=0.7)
        if diff_pc is not None and diff_pc_err is not None:
            pyplot.annotate(
                f"i_pc = ({diff_pc:.3f} ± {diff_pc_err:.3f}) µA",
                xy=(x_pc, (y_fit_pc + y_peak_pc) / 2),
                xytext=(x_pc - 160, (y_fit_pc + y_peak_pc) / 2),
                fontsize=8,
                color="brown",
                arrowprops=dict(arrowstyle="->", color="brown", lw=0.8),
            )

    # halfway potential line
    halfway_pot = extremas.halfway_pot
    pyplot.axvline(x=halfway_pot, color="green", linestyle="--")
    pyplot.annotate(
        f"Halfway Potential: ({halfway_pot:.2f} ± {extremas.err_halfway_pot:.2f}) mV",
        xy=(halfway_pot, extremas.avg_max_current * 0.5),
        xytext=(halfway_pot * 0.5, extremas.avg_max_current * (2/3)),
        fontsize=8,
        color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
    )

    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    main()
