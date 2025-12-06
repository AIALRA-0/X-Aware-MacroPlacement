#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/thermal_cli.py

Unified CLI for macro thermal analysis.

Features:
- Input:
    * --plc-file:  CircuitTraining / MacroPlacement .plc placement file
      (no power, we synthesize macro power via log-normal model);
    * --macro-csv: CSV describing macros. It can be either:
        - macro power CSV:  name,x_um,y_um,power_W
        - layout CSV:      macro_name,x_um,y_um,width_um,height_um
          (power is synthesized from area).

- Output (under --out-dir, default: ./outputs):
    * macro_power/<prefix>_macro_power.csv
    * figures/<prefix>_temperature.png
    * figures/<prefix>_grad_norm.png
    * metrics/<prefix>_thermal_stats.json
    * (optional) data/thermal/<prefix>_temp.npy
                 data/thermal/<prefix>_grad_norm.npy
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# allow "from src.thermal.thermal_model import ..." when this file is in scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.thermal.thermal_model import compute_thermal_from_macros, Macro  # noqa: E402


# -------------------------
# Helpers: parsing & power
# -------------------------

def parse_macros_from_plc_text(plc_text: str) -> pd.DataFrame:
    """
    Very lightweight .plc parser.

    Each valid line is assumed to be:
        <inst_name> <x> <y> ...

    We keep only (name, x_um, y_um).
    """
    records: List[Tuple[str, float, float]] = []
    for line in plc_text.splitlines():
        line = line.strip()
        if (not line) or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
        except ValueError:
            continue
        records.append((name, x, y))

    if not records:
        raise ValueError("No macros parsed from .plc file. Check format or parser.")

    df = pd.DataFrame(records, columns=["name", "x_um", "y_um"])
    return df


def simulate_power_lognormal(
    macros_df: pd.DataFrame,
    base_power: float = 1.0,
    log_sigma: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthesize macro power using a log-normal distribution.

    - Treat all macros as area=1 (no area info in plc).
    - Normalize so mean(power) ≈ base_power.
    """
    rng = np.random.default_rng(seed)
    n = len(macros_df)
    if n == 0:
        raise ValueError("Empty macro DataFrame when synthesizing power.")

    noise = rng.lognormal(mean=0.0, sigma=log_sigma, size=n)
    noise = noise / noise.mean()
    power = base_power * noise

    out = macros_df.copy()
    out["power_W"] = power.astype(np.float32)
    return out


def detect_csv_type(df: pd.DataFrame) -> str:
    """
    Detect CSV schema:
    - "macro_power": name/x/y/power_W (or macro_name/x/y/power_W)
    - "layout":      macro_name/x/y/width/height (no power_W)
    """
    cols = set(df.columns)

    if {"x_um", "y_um", "power_W"}.issubset(cols) and (
        "name" in cols or "macro_name" in cols
    ):
        return "macro_power"

    if {"macro_name", "x_um", "y_um", "width_um", "height_um"}.issubset(cols):
        return "layout"

    raise ValueError(
        "Unsupported CSV schema. Expected either:\n"
        "  (name,x_um,y_um,power_W) or\n"
        "  (macro_name,x_um,y_um,width_um,height_um)."
    )


def normalize_macro_power_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a macro power CSV to standard columns:
        name, x_um, y_um, power_W
    """
    cols = set(df.columns)
    if "name" in cols:
        name_col = "name"
    elif "macro_name" in cols:
        name_col = "macro_name"
    else:
        raise ValueError("macro_power CSV must have 'name' or 'macro_name' column.")

    out = pd.DataFrame(
        {
            "name": df[name_col].astype(str),
            "x_um": df["x_um"].astype(float),
            "y_um": df["y_um"].astype(float),
            "power_W": df["power_W"].astype(float),
        }
    )
    return out


def layout_csv_to_macro_power(
    df: pd.DataFrame,
    alpha_w_per_um2: float = 1e-5,
    noise_std: float = 0.2,
    seed: int = 42,
):
    """
    From layout CSV with columns:
        macro_name,x_um,y_um,width_um,height_um
    generate:
        macros_df: name,x_um,y_um,power_W
        chip_w_um, chip_h_um: estimated chip size from macro bbox * 1.1
    """
    required = {"macro_name", "x_um", "y_um", "width_um", "height_um"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Layout CSV missing columns: {missing}")

    rng = np.random.default_rng(seed)
    area = df["width_um"].values * df["height_um"].values
    n = len(df)
    if n == 0:
        raise ValueError("Layout CSV is empty.")

    noise = rng.normal(loc=0.0, scale=noise_std, size=n)
    noise = np.clip(noise, -0.5, 0.5)
    power_w = alpha_w_per_um2 * area * (1.0 + noise)

    macros_df = pd.DataFrame(
        {
            "name": df["macro_name"].astype(str),
            "x_um": df["x_um"].astype(float),
            "y_um": df["y_um"].astype(float),
            "power_W": power_w.astype(np.float32),
        }
    )

    x_max = float(df["x_um"].max())
    y_max = float(df["y_um"].max())
    chip_w_um = x_max * 1.1
    chip_h_um = y_max * 1.1

    return macros_df, chip_w_um, chip_h_um


def plot_field(field: np.ndarray, title: str, out_path: str) -> None:
    """
    Save a 2D field as a heatmap PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots()
    im = ax.imshow(field, origin="lower")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Macro thermal analysis CLI (PLC / CSV → CSV + PNG + JSON)."
    )

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--plc-file",
        type=str,
        help="Placement .plc file (no power; log-normal power will be synthesized).",
    )
    group.add_argument(
        "--macro-csv",
        type=str,
        help=(
            "Macro CSV. Either:\n"
            "  name,x_um,y_um,power_W\n"
            "or\n"
            "  macro_name,x_um,y_um,width_um,height_um"
        ),
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Output root directory (default: ./outputs).",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for output files; default: input basename without extension.",
    )

    # chip geometry
    p.add_argument(
        "--chip-width-um",
        type=float,
        default=None,
        help="Chip width in µm (optional override; default: inferred from data).",
    )
    p.add_argument(
        "--chip-height-um",
        type=float,
        default=None,
        help="Chip height in µm (optional override; default: inferred from data).",
    )

    # grid & kernel
    p.add_argument("--nx", type=int, default=128, help="Grid columns (nx).")
    p.add_argument("--ny", type=int, default=128, help="Grid rows (ny).")
    p.add_argument(
        "--sigma-um",
        type=float,
        default=80.0,
        help="Gaussian kernel sigma in µm (controls thermal diffusion extent).",
    )

    # saving
    p.add_argument(
        "--save-npy",
        action="store_true",
        help="If set, save temp/grad_norm arrays as .npy under data/thermal/.",
    )

    # power model for PLC (log-normal)
    p.add_argument(
        "--base-power",
        type=float,
        default=1.0,
        help="Mean macro power (relative units) for PLC log-normal model.",
    )
    p.add_argument(
        "--log-sigma",
        type=float,
        default=0.4,
        help="σ of log-normal power variation for PLC input.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for power synthesis.",
    )

    # power model for layout CSV (area → power)
    p.add_argument(
        "--alpha-w-per-um2",
        type=float,
        default=1e-5,
        help="α in W/µm² for area-based power when CSV has width/height.",
    )
    p.add_argument(
        "--noise-std",
        type=float,
        default=0.2,
        help="Std of Gaussian noise for area-based power.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_dir = os.path.join(args.out_dir, "metrics")
    figures_dir = os.path.join(args.out_dir, "figures")
    macro_power_dir = os.path.join(args.out_dir, "macro_power")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(macro_power_dir, exist_ok=True)

    if args.prefix is None:
        if args.plc_file:
            base = os.path.basename(args.plc_file)
        else:
            base = os.path.basename(args.macro_csv)
        args.prefix = os.path.splitext(base)[0]

    # ----------------------
    # Build macro power table
    # ----------------------
    if args.plc_file:
        # read plc
        with open(args.plc_file, "r", encoding="utf-8", errors="ignore") as f:
            plc_text = f.read()

        macros_df = parse_macros_from_plc_text(plc_text)
        print(f"[INFO] Parsed {len(macros_df)} macros from {args.plc_file}")

        macros_df = simulate_power_lognormal(
            macros_df,
            base_power=args.base_power,
            log_sigma=args.log_sigma,
            seed=args.seed,
        )

        # infer chip size if not overridden
        if args.chip_width_um is None:
            args.chip_width_um = float(macros_df["x_um"].max()) * 1.1
        if args.chip_height_um is None:
            args.chip_height_um = float(macros_df["y_um"].max()) * 1.1

    else:
        df_in = pd.read_csv(args.macro_csv)
        csv_type = detect_csv_type(df_in)
        print(f"[INFO] Detected CSV type: {csv_type}")

        if csv_type == "macro_power":
            macros_df = normalize_macro_power_df(df_in)
            if args.chip_width_um is None:
                args.chip_width_um = float(macros_df["x_um"].max()) * 1.1
            if args.chip_height_um is None:
                args.chip_height_um = float(macros_df["y_um"].max()) * 1.1
        else:  # layout
            macros_df, chip_w, chip_h = layout_csv_to_macro_power(
                df_in,
                alpha_w_per_um2=args.alpha_w_per_um2,
                noise_std=args.noise_std,
                seed=args.seed,
            )
            if args.chip_width_um is None:
                args.chip_width_um = chip_w
            if args.chip_height_um is None:
                args.chip_height_um = chip_h

        print(f"[INFO] Using {len(macros_df)} macros from {args.macro_csv}")

    print(
        f"[INFO] Chip size: {args.chip_width_um:.1f}µm x {args.chip_height_um:.1f}µm, "
        f"Grid: {args.nx} x {args.ny}, sigma = {args.sigma_um:.1f}µm"
    )

    # Save macro power CSV
    macro_csv_path = os.path.join(
        macro_power_dir, f"{args.prefix}_macro_power.csv"
    )
    macros_df.to_csv(macro_csv_path, index=False)
    print(f"[INFO] Saved macro power CSV to {macro_csv_path}")

    # ----------------------
    # Thermal computation
    # ----------------------
    macros_list: List[Macro] = [
        (row.name, float(row.x_um), float(row.y_um), float(row.power_W))
        for row in macros_df.itertuples(index=False)
    ]

    result = compute_thermal_from_macros(
        macros=macros_list,
        chip_w_um=args.chip_width_um,
        chip_h_um=args.chip_height_um,
        nx=args.nx,
        ny=args.ny,
        sigma_um=args.sigma_um,
    )

    temp_grid = result["temp_grid"]
    grad_norm = result["grad_norm"]
    metrics = result["metrics"]

    print("[INFO] Thermal metrics:")
    print(json.dumps(metrics, indent=2))

    # save metrics
    metrics_path = os.path.join(
        metrics_dir, f"{args.prefix}_thermal_stats.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved metrics JSON to {metrics_path}")

    # save npy if requested
    if args.save_npy:
        npy_dir = os.path.join("data", "thermal")
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, f"{args.prefix}_temp.npy"), temp_grid)
        np.save(os.path.join(npy_dir, f"{args.prefix}_grad_norm.npy"), grad_norm)
        print(
            f"[INFO] Saved temp/grad_norm NPY to data/thermal/{args.prefix}_*.npy"
        )

    # save figures
    temp_png = os.path.join(figures_dir, f"{args.prefix}_temperature.png")
    grad_png = os.path.join(figures_dir, f"{args.prefix}_grad_norm.png")

    plot_field(temp_grid, "Temperature Field", temp_png)
    plot_field(grad_norm, "Gradient Norm ||∇T|| Field", grad_png)

    print(f"[INFO] Saved temperature heatmap to {temp_png}")
    print(f"[INFO] Saved grad-norm heatmap to {grad_png}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
