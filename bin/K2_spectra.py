#!/usr/bin/env python3

import argparse
import csv
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot

PATH_PREFIX = "K2/data/"
CUTOFF_NM = 220.0

def main():
    parser = argparse.ArgumentParser(description="Plot spectra (single-pass or multi-pass).")
    parser.add_argument("-f", "--file", default="water", help="Base filename (without .csv) in data/")
    parser.add_argument("--multi", action="store_true", help="Treat file as multi-pass data.")
    parser.add_argument("--time", action="store_true", help="Plot the time-dependant extinction at a single wavelength.")
    args = parser.parse_args()

    if args.multi:
        wavelengths, absorbance_passes = load_multi(args.file)
        plot_multi(wavelengths, absorbance_passes)
    elif args.time:
        t, absorbances = load_single(args.file, cutoff=False)
        plot_time(t, absorbances)
    else:
        wavelengths, absorbances = load_single(args.file)
        plot_single(wavelengths, absorbances)


def _load_csv(name: str, cutoff=True) -> Tuple[List[float], List[List[float]]]:
    """
    Internal loader that reads alternating wavelength/absorbance columns.
    Returns (wavelengths, list_of_absorbance_columns) filtered by CUTOFF_NM.
    """
    fname = f"{PATH_PREFIX}{name}.csv"
    print(f"Loading file: {fname}")

    raw_wl: List[float] = []
    raw_abs_passes: List[List[float]] = []

    with open(fname, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        if not header:
            raise ValueError("Empty CSV file.")
        if len(header) % 2 != 0:
            raise ValueError("Expected alternating wavelength/absorbance columns.")

        num_passes = len(header) // 2
        raw_abs_passes = [[] for _ in range(num_passes)]

        for row in reader:
            if len(row) < num_passes * 2:
                continue
            raw_wl.append(float(row[0]))
            for i in range(num_passes):
                raw_abs_passes[i].append(float(row[2 * i + 1]))

    if cutoff:
        # Apply cutoff mask
        wl_array = np.asarray(raw_wl)
        mask = wl_array >= CUTOFF_NM
        wl_filtered = wl_array[mask].tolist()
        abs_filtered: List[List[float]] = []
        for abs_pass in raw_abs_passes:
            abs_array = np.asarray(abs_pass)
            abs_filtered.append(abs_array[mask].tolist())

        return wl_filtered, abs_filtered

    return raw_wl, raw_abs_passes


def load_single(name: str, cutoff=True) -> Tuple[List[float], List[float]]:
    wavelengths, absorbance_passes = _load_csv(name, cutoff=cutoff)
    if not absorbance_passes:
        raise ValueError("No absorbance data found.")
    return wavelengths, absorbance_passes[0]


def load_multi(name: str) -> Tuple[List[float], List[List[float]]]:
    return _load_csv(name)


def _denoise(absorbances: List[float], window_size: int = 11) -> List[float]:
    """Simple moving average denoising."""
    abs_array = np.asarray(absorbances)
    kernel = np.ones(window_size) / window_size
    denoised = np.convolve(abs_array, kernel, mode='same')
    # remove first and last few points which are less reliable
    denoised[:window_size//2] = abs_array[:window_size//2]
    denoised[-(window_size//2):] = abs_array[-(window_size//2):]
    return denoised.tolist()


def plot_time(t, absorbances: List[float]) -> None:
    denoised = _denoise(absorbances)
    pyplot.plot(t, absorbances, label="Rohdaten")
    pyplot.plot(t, denoised, label="Geglättet", color="orange", alpha=0.7)
    pyplot.xlabel("Zeit (Minuten)")
    pyplot.ylabel("Extinktion bei 617 nm")
    pyplot.tight_layout()
    pyplot.grid(True)
    pyplot.legend()
    pyplot.show()


def plot_single(wavelengths: List[float], absorbances: List[float]) -> None:
    pyplot.plot(wavelengths, absorbances)
    pyplot.xlabel("Wellenlänge (nm)")
    pyplot.ylabel("Extinktion")
    pyplot.tight_layout()
    pyplot.axvline(617.0, color="red", linestyle="--", label="Maximum Malachitgrün (617 nm)")
    pyplot.grid(True)
    pyplot.legend()
    pyplot.show()


def plot_multi(wavelengths: List[float], absorbance_passes: List[List[float]]) -> None:
    for idx, abs_values in enumerate(absorbance_passes):
        pyplot.plot(wavelengths, abs_values, label=f"{idx * 10} Minuten")
    pyplot.xlabel("Wellenlänge (nm)")
    pyplot.ylabel("Extinktion")
    pyplot.axvline(279.0, color="gray", linestyle="--", label="Isosbestischer Punkt (279 nm)")
    pyplot.axvline(617.0, color="k", linestyle="--", label="Maximum Malachitgrün (617 nm)")
    pyplot.axvline(253.0, color="b", linestyle="--", label="Maximum Malachitgrün-Carbinol (253 nm)")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.grid(True)
    pyplot.show()


if __name__ == "__main__":
    main()
