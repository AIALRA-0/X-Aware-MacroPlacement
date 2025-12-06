#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/thermal_gui.py

Simple Streamlit GUI for macro thermal analysis:

- Input types:
    * PLC (.plc): parse (name, x, y), synthesize power via log-normal model;
      button "Convert PLC → CSV" to download macro power CSV.
    * CSV (.csv):
        - If columns include power_W: treated as macro power CSV
          (name,x_um,y_um,power_W or macro_name,x_um,y_um,power_W).
        - If columns are layout-style: macro_name,x_um,y_um,width_um,height_um,
          synthesize power from area.

- Adjustable parameters:
    * Grid: nx, ny;
    * Gaussian kernel sigma (µm);
    * Power model parameters.

- Output:
    * Preview macro table;
    * Numeric statistics (temperature / grad-norm: mean, p90, max);
    * Heatmap previews for T and ||∇T||.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# import thermal core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.thermal.thermal_model import compute_thermal_from_macros, Macro  # noqa: E402


# -------------------------
# Helpers (same logic as CLI)
# -------------------------

def parse_macros_from_plc_text(plc_text: str) -> pd.DataFrame:
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
        raise ValueError("No macros parsed from PLC file.")
    return pd.DataFrame(records, columns=["name", "x_um", "y_um"])


def simulate_power_lognormal(
    macros_df: pd.DataFrame,
    base_power: float = 1.0,
    log_sigma: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(macros_df)
    if n == 0:
        raise ValueError("Empty macro DataFrame.")
    noise = rng.lognormal(mean=0.0, sigma=log_sigma, size=n)
    noise = noise / noise.mean()
    power = base_power * noise
    out = macros_df.copy()
    out["power_W"] = power.astype(np.float32)
    return out


def detect_csv_type(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"x_um", "y_um", "power_W"}.issubset(cols) and (
        "name" in cols or "macro_name" in cols
    ):
        return "macro_power"
    if {"macro_name", "x_um", "y_um", "width_um", "height_um"}.issubset(cols):
        return "layout"
    raise ValueError(
        "Unsupported CSV format. Expect either:\n"
        "  (name,x_um,y_um,power_W) or\n"
        "  (macro_name,x_um,y_um,width_um,height_um)."
    )


def normalize_macro_power_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    if "name" in cols:
        name_col = "name"
    elif "macro_name" in cols:
        name_col = "macro_name"
    else:
        raise ValueError("CSV must have 'name' or 'macro_name'.")
    return pd.DataFrame(
        {
            "name": df[name_col].astype(str),
            "x_um": df["x_um"].astype(float),
            "y_um": df["y_um"].astype(float),
            "power_W": df["power_W"].astype(float),
        }
    )


def layout_csv_to_macro_power(
    df: pd.DataFrame,
    alpha_w_per_um2: float = 1e-5,
    noise_std: float = 0.2,
    seed: int = 42,
):
    required = {"macro_name", "x_um", "y_um", "width_um", "height_um"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Layout CSV missing columns: {missing}")

    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        raise ValueError("Layout CSV is empty.")

    area = df["width_um"].values * df["height_um"].values
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


def show_thermal_result(result: dict):
    temp_grid = result["temp_grid"]
    grad_norm = result["grad_norm"]
    metrics = result["metrics"]

    st.subheader("Metrics")
    st.json(metrics)

    st.subheader("Heatmaps")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        im1 = ax1.imshow(temp_grid, origin="lower")
        ax1.set_title("Temperature Field")
        fig1.colorbar(im1, ax=ax1)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(grad_norm, origin="lower")
        ax2.set_title("Gradient Norm ||∇T||")
        fig2.colorbar(im2, ax=ax2)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


# -------------------------
# Streamlit app
# -------------------------

def main():
    st.set_page_config(page_title="Macro Thermal Analysis", layout="wide")

    st.title("Macro Thermal Analysis")

    # Sidebar: global parameters
    st.sidebar.header("Grid & Kernel")
    nx = st.sidebar.number_input("Grid Nx", min_value=16, max_value=512, value=128, step=16)
    ny = st.sidebar.number_input("Grid Ny", min_value=16, max_value=512, value=128, step=16)
    sigma_um = st.sidebar.number_input(
        "Gaussian sigma (µm)", min_value=1.0, max_value=500.0, value=80.0, step=5.0
    )

    st.sidebar.header("Power Model (PLC)")
    base_power = st.sidebar.number_input(
        "Base power (log-normal mean)", min_value=0.01, value=1.0, step=0.1
    )
    log_sigma = st.sidebar.slider(
        "Log-normal sigma", min_value=0.0, max_value=1.0, value=0.4, step=0.05
    )
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1)

    st.sidebar.header("Power Model (Layout CSV)")
    alpha_w_per_um2 = st.sidebar.number_input(
        "Alpha (W/µm²)", min_value=1e-7, max_value=1e-3, value=1e-5, format="%.1e"
    )
    noise_std = st.sidebar.slider(
        "Area power noise std", min_value=0.0, max_value=0.5, value=0.2, step=0.05
    )

    # Main content
    st.subheader("Input")

    input_type = st.radio(
        "Input type",
        ["PLC (.plc)", "CSV (.csv)"],
        horizontal=True,
    )

    if input_type.startswith("PLC"):
        plc_file = st.file_uploader("Upload PLC file", type=["plc", "txt"])
        if plc_file is None:
            st.info("Please upload a .plc file.")
            return

        try:
            plc_text = plc_file.read().decode("utf-8", errors="ignore")
            macros_df = parse_macros_from_plc_text(plc_text)
        except Exception as e:
            st.error(f"Failed to parse PLC file: {e}")
            return

        st.markdown("**Parsed macros (head)**")
        st.dataframe(macros_df.head(), use_container_width=True)

        # default chip size from coordinates
        chip_w_default = float(macros_df["x_um"].max()) * 1.1
        chip_h_default = float(macros_df["y_um"].max()) * 1.1

        col_geom1, col_geom2 = st.columns(2)
        with col_geom1:
            chip_w_um = st.number_input(
                "Chip width (µm)", min_value=1.0, value=chip_w_default, step=10.0
            )
        with col_geom2:
            chip_h_um = st.number_input(
                "Chip height (µm)", min_value=1.0, value=chip_h_default, step=10.0
            )

        # Convert PLC → CSV button
        if st.button("Convert PLC → CSV"):
            macros_with_power = simulate_power_lognormal(
                macros_df,
                base_power=base_power,
                log_sigma=log_sigma,
                seed=seed,
            )
            st.markdown("**Macro power table (head)**")
            st.dataframe(macros_with_power.head(), use_container_width=True)

            csv_bytes = macros_with_power.to_csv(index=False).encode("utf-8")
            base_prefix = Path(plc_file.name).stem
            csv_name = f"{base_prefix}_macro_power.csv"
            st.download_button(
                "Download macro power CSV",
                data=csv_bytes,
                file_name=csv_name,
                mime="text/csv",
            )

        # Run thermal button
        if st.button("Run Thermal"):
            macros_with_power = simulate_power_lognormal(
                macros_df,
                base_power=base_power,
                log_sigma=log_sigma,
                seed=seed,
            )
            macros_list: List[Macro] = [
                (row.name, float(row.x_um), float(row.y_um), float(row.power_W))
                for row in macros_with_power.itertuples(index=False)
            ]
            result = compute_thermal_from_macros(
                macros=macros_list,
                chip_w_um=chip_w_um,
                chip_h_um=chip_h_um,
                nx=int(nx),
                ny=int(ny),
                sigma_um=float(sigma_um),
            )
            show_thermal_result(result)

    else:
        csv_file = st.file_uploader("Upload CSV file", type=["csv"])
        if csv_file is None:
            st.info("Please upload a CSV file.")
            return

        try:
            df_in = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

        st.markdown("**CSV preview (head)**")
        st.dataframe(df_in.head(), use_container_width=True)

        try:
            csv_type = detect_csv_type(df_in)
        except Exception as e:
            st.error(f"Failed to detect CSV type: {e}")
            return

        if csv_type == "macro_power":
            macros_df = normalize_macro_power_df(df_in)
            chip_w_default = float(macros_df["x_um"].max()) * 1.1
            chip_h_default = float(macros_df["y_um"].max()) * 1.1
        else:
            macros_df, chip_w_default, chip_h_default = layout_csv_to_macro_power(
                df_in,
                alpha_w_per_um2=alpha_w_per_um2,
                noise_std=noise_std,
                seed=seed,
            )

        st.markdown("**Macro power table (head)**")
        st.dataframe(macros_df.head(), use_container_width=True)

        col_geom1, col_geom2 = st.columns(2)
        with col_geom1:
            chip_w_um = st.number_input(
                "Chip width (µm)", min_value=1.0, value=chip_w_default, step=10.0
            )
        with col_geom2:
            chip_h_um = st.number_input(
                "Chip height (µm)", min_value=1.0, value=chip_h_default, step=10.0
            )

        if st.button("Run Thermal"):
            macros_list: List[Macro] = [
                (row.name, float(row.x_um), float(row.y_um), float(row.power_W))
                for row in macros_df.itertuples(index=False)
            ]
            result = compute_thermal_from_macros(
                macros=macros_list,
                chip_w_um=chip_w_um,
                chip_h_um=chip_h_um,
                nx=int(nx),
                ny=int(ny),
                sigma_um=float(sigma_um),
            )
            show_thermal_result(result)


if __name__ == "__main__":
    main()
