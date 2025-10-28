#!/usr/bin/env python3
"""
Build a Sun-centered x–y map of **Σ_SFR(x,y)** inferred from a 3D dust map
(Edenhofer+2023 via dustmaps) using a Kennicutt–Schmidt law, and overlay the
Herbig star locations from a TSV.

- Σ_gas is inferred from A_V integrated over |z| <= Zmax (pc), with
  Σ_gas ≈ 21.3 Msun/pc^2 per mag(A_V).
- Σ_SFR = A_KS * (Σ_gas)^N with defaults A_KS=2.5e-4, N=1.4.
- A_V is derived from the dustmap "E" units with A_V = 2.8 * E (per pc).
- The map uses the density mode (extinction per pc) at (l,b,d) points corresponding
  to a rectilinear (x,y,z) grid centered on the Sun, then integrates along z.

This script is vectorized and chunked for memory safety.
"""

import argparse
import math
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

from dustmaps.config import config as dustmaps_config
import dustmaps.edenhofer2023 as eden
from dustmaps.edenhofer2023 import Edenhofer2023Query

def icrs_to_gal_lbd(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ICRS (deg) -> Galactic (l,b) deg via astropy."""
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    gal = sky.galactic
    return gal.l.deg, gal.b.deg

def xy_to_lbd(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sun-centered Cartesian (pc) -> Galactic (l,b,d)."""
    d = np.sqrt(x**2 + y**2 + z**2).astype(np.float64)
    d = np.where(d < 1e-6, 1e-6, d)
    l = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    b = np.degrees(np.arcsin(z / d))
    return l, b, d

def compute_AV_slab_from_dust(x_grid: np.ndarray, y_grid: np.ndarray, z_vals: np.ndarray,
                              dmin: float = 69.0, dmax: float = 1250.0,
                              AV_per_E: float = 2.8,
                              chunk_size: int = 500_000,
                              verbose: bool = True) -> np.ndarray:
    """
    For each (x,y), integrate A_V per pc along z over the slab defined by z_vals.
    Returns AV_slab with shape (Ny, Nx) (matching np.meshgrid(indexing='xy')).
    """
    NX = x_grid.size
    NY = y_grid.size
    NZ = z_vals.size

    XX, YY, ZZ = np.meshgrid(x_grid, y_grid, z_vals, indexing="xy")
    # Flatten for vectorized querying in chunks
    Xf = XX.ravel().astype(np.float64)
    Yf = YY.ravel().astype(np.float64)
    Zf = ZZ.ravel().astype(np.float64)

    # Convert to Galactic (l,b,d)
    l_deg, b_deg, d_pc = xy_to_lbd(Xf, Yf, Zf)

    # Build SkyCoord once per chunk
    q_den = Edenhofer2023Query(integrated=False, load_samples=False)

    AV_per_pc_accum = np.zeros_like(Xf, dtype=np.float64)

    N = Xf.size
    i = 0
    while i < N:
        j = min(i + chunk_size, N)
        if verbose:
            print(f"Querying dustmap: {i:,} – {j:,} / {N:,} points...", file=sys.stderr)

        coords = SkyCoord(l_deg[i:j] * u.deg, b_deg[i:j] * u.deg, d_pc[i:j] * u.pc, frame="galactic")
        # E per pc:
        E_per_pc_chunk = q_den.query(coords, mode="mean")
        # Mask out-of-range distances
        mask = (d_pc[i:j] < dmin) | (d_pc[i:j] > dmax) | ~np.isfinite(E_per_pc_chunk)
        E_per_pc_chunk = np.where(mask, 0.0, E_per_pc_chunk)
        # Convert to A_V per pc
        AV_per_pc_accum[i:j] = AV_per_E * E_per_pc_chunk
        i = j

    AV_per_pc = AV_per_pc_accum.reshape((NY, NX, NZ))
    # Integrate along z: simple Riemann sum (Δz is uniform)
    # z_vals were used with indexing='xy' so third axis is z
    if NZ < 2:
        raise ValueError("z_vals must have at least 2 points.")
    dZ = float(z_vals[1] - z_vals[0])
    AV_slab = (AV_per_pc * dZ).sum(axis=2)
    return AV_slab

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tsv", required=True, help="Path to Herbig catalog TSV with columns: ra, dec, distance (pc)")
    ap.add_argument("--out", default="sfr_map_xy_from_dust.png", help="Output PNG path")
    ap.add_argument("--npz", default="sfr_grid_xy_from_dust.npz", help="Output NPZ (grid + Σ_SFR)")
    ap.add_argument("--zmax", type=float, default=200.0, help="Vertical half-thickness for integration (pc)")
    ap.add_argument("--dz", type=float, default=10.0, help="Vertical step for integration (pc)")
    ap.add_argument("--ks_A", type=float, default=2.5e-4, help="Kennicutt–Schmidt A (Msun yr^-1 kpc^-2)")
    ap.add_argument("--ks_N", type=float, default=1.4, help="Kennicutt–Schmidt N")
    ap.add_argument("--av_to_gas", type=float, default=21.3, help="Σ_gas [Msun/pc^2] per mag(A_V)")
    ap.add_argument("--av_per_E", type=float, default=2.8, help="A_V per E unit from Edenhofer map")
    ap.add_argument("--grid", type=int, default=201, help="Grid size per axis (Nx=Ny)")
    ap.add_argument("--margin", type=float, default=200.0, help="Margin (pc) beyond star extent")
    ap.add_argument("--dmin", type=float, default=69.0, help="Dust map valid inner radius (pc)")
    ap.add_argument("--dmax", type=float, default=1250.0, help="Dust map valid outer radius (pc)")
    ap.add_argument("--chunk", type=int, default=500_000, help="Dust query chunk size")
    ap.add_argument("--fetch", action="store_true", help="Fetch Edenhofer dust map if not already present")
    ap.add_argument("--dust_dir", default=None, help="dustmaps data directory; defaults to ./dustmaps_data")
    args = ap.parse_args()

    # Configure dustmaps location and fetch data if requested
    if args.dust_dir is not None:
        dustmaps_config["data_dir"] = args.dust_dir
    else:
        # Ensure default exists
        dustmaps_config["data_dir"] = os.path.expanduser("./dustmaps_data")

    if args.fetch:
        print("Fetching Edenhofer 2023 map (only needed once)...", file=sys.stderr)
        eden.fetch()  # downloads data if missing

    # Load catalog
    cat = pd.read_csv(args.tsv, sep="\t")
    if not {"ra", "dec", "distance"}.issubset(cat.columns):
        raise SystemExit("TSV must include columns: ra, dec, distance (pc)")

    ra_deg = cat["ra"].to_numpy(float)
    dec_deg = cat["dec"].to_numpy(float)
    d_pc = cat["distance"].to_numpy(float)

    # Convert to Galactic and Sun-centered x,y for plotting
    l_deg, b_deg = icrs_to_gal_lbd(ra_deg, dec_deg)
    l = np.deg2rad(l_deg)
    b = np.deg2rad(b_deg)
    x_star = d_pc * np.cos(b) * np.cos(l)
    y_star = d_pc * np.cos(b) * np.sin(l)

    # Grid bounds
    margin = float(args.margin)
    xmin = float(np.floor(min(x_star.min(), -x_star.max()) - margin))
    xmax = float(np.ceil(max(x_star.max(), -x_star.min()) + margin))
    ymin = float(np.floor(min(y_star.min(), -y_star.max()) - margin))
    ymax = float(np.ceil(max(y_star.max(), -y_star.min()) + margin))
    rmax = float(np.ceil(max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))))
    xmin, xmax, ymin, ymax = -rmax, rmax, -rmax, rmax

    NX = NY = int(args.grid)
    x_grid = np.linspace(xmin, xmax, NX)
    y_grid = np.linspace(ymin, ymax, NY)
    z_vals = np.arange(-args.zmax, args.zmax + args.dz, args.dz)

    # Build A_V slab via dust map
    AV_slab = compute_AV_slab_from_dust(
        x_grid, y_grid, z_vals,
        dmin=args.dmin, dmax=args.dmax,
        AV_per_E=args.av_per_E,
        chunk_size=args.chunk,
        verbose=True
    )

    # Convert to Σ_gas and then Σ_SFR
    Sigma_gas = args.av_to_gas * AV_slab  # Msun/pc^2
    Sigma_SFR = args.ks_A * np.power(np.maximum(Sigma_gas, 0.0), args.ks_N)  # Msun yr^-1 kpc^-2

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    bg = np.log10(np.maximum(Sigma_SFR, np.percentile(Sigma_SFR, 1) * 1e-6))
    im = ax.imshow(
        bg,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal",
    )
    ax.scatter(x_star, y_star, s=10)
    ax.set_xlabel("x from Sun [pc]")
    ax.set_ylabel("y from Sun [pc]")
    ax.set_title("Σ_SFR(x, y) from dust (Kennicutt–Schmidt) + Herbig locations")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log10 Σ_SFR  [Msun yr$^{-1}$ kpc$^{-2}$]")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}", file=sys.stderr)

    # Save grid and map for reuse
    np.savez_compressed(
        args.npz,
        x_grid=x_grid,
        y_grid=y_grid,
        Sigma_SFR=Sigma_SFR,
        AV_slab=AV_slab,
        Sigma_gas=Sigma_gas,
        meta=dict(
            zmax=args.zmax, dz=args.dz, ks_A=args.ks_A, ks_N=args.ks_N,
            av_to_gas=args.av_to_gas, av_per_E=args.av_per_E,
            dmin=args.dmin, dmax=args.dmax
        ),
    )
    print(f"Saved grid to {args.npz}", file=sys.stderr)


if __name__ == "__main__":
    main()
