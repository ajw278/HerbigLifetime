#!/usr/bin/env python3
"""
Fit a simple radial completeness + lifetime model for Herbig stars.

This script is designed to plug into your *existing* file conventions:

Inputs
------
--sfr_npz
    NPZ produced by `sfr_map_xy_from_dust.py`, containing:
        x_grid, y_grid, and either Sigma_SFR_smoothed or Sigma_SFR
    (units: Msun / yr / kpc^2)

--star_pack
    NPZ 'star_pack' produced by `build_ppp_all_packs.py`, containing:
        x_pc, y_pc, Mstar, etc.

Model
-----
We compress the SFR field to a ring-averaged Σ_SFR(R) profile and
fit a simple model for the expected number of Herbig stars in each
(R, M) bin:

    lambda_ij = k_H * Sigma_SFR(R_i) * tau(M_j) * f_det(R_i, M_j)

where

  * tau(M) = tau0 * (M / M0)^alpha_tau
      - tau0 = 3 Myr at M0 = 1 Msun (fixed)
      - alpha_tau is a free parameter (expected < 0)

  * f_det(R, M) is a logistic function in log R and log M:
        logit f_det = a0 + a_R * ln(R / R0) + a_M * ln(M / M0)

  * k_H is a global normalisation factor absorbing survey geometry,
    IMF fraction into the Herbig range, etc.

The SFR normalisation is anchored to Quintana+ via the total SFR
inside 1 kpc:
    SFR_Quintana(<=1 kpc) = 2896 Msun / Myr

By default we compute a scaling factor so that the *mean* Σ_SFR
inside 1 kpc produces this total when integrated over π (1 kpc)^2,
mirroring the message printed at the end of `sfr_map_xy_from_dust.py`.
You can override this with --sfr_scale if you prefer a fixed value.
"""

import argparse
import sys

import numpy as np
import pymc as pm
import arviz as az
from utils import chabrier2003_unnorm_pdf, xi_norm_on_interval, imf_mean_mass


# Quintana+ (2025) SFR constraint inside 1 kpc
SFR_QUINTANA = 2896.0          # Msun / Myr (central)
SFR_QUINTANA_LO = 2895.0       # central - 1
SFR_QUINTANA_HI = 2896.0 + 417.0  # central + 417

K_GEOM_LO = SFR_QUINTANA_LO / SFR_QUINTANA
K_GEOM_HI = SFR_QUINTANA_HI / SFR_QUINTANA
K_GEOM_MODE = 1.0
K_GEOM_C = (K_GEOM_MODE - K_GEOM_LO) / (K_GEOM_HI - K_GEOM_LO)

LOG_K_GEOM_SIGMA = (np.log(K_GEOM_HI) - np.log(K_GEOM_LO)) / (2 * 1.96)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def compute_quintana_scale(
    x_grid,
    y_grid,
    Sigma_SFR,
    sfr_target_msun_per_myr=2896.0,
    radius_target_pc=1000.0,
    verbose=True,
):
    """
    Compute the SFR normalisation factor needed to match the
    Quintana+ SFR inside radius_target_pc.

    We follow the same logic as the message printed in
    `sfr_map_xy_from_dust.py`:

        SFR_target = sfr_target_msun_per_myr [Msun/Myr]
        Area(<=1 kpc) = π * (1 kpc)^2  [kpc^2]
        Mean Σ_SFR_required = SFR_target / (Area * 1e6)  [Msun/yr/kpc^2]

    We estimate the map-averaged Σ_SFR (over finite, positive pixels)
    and take:

        sfr_scale = Mean_required / Mean_map

    Parameters
    ----------
    x_grid, y_grid : 1D arrays (pc)
    Sigma_SFR      : 2D array (Ny, Nx), Msun/yr/kpc^2 (unscaled)
    sfr_target_msun_per_myr : float
    radius_target_pc        : float

    Returns
    -------
    sfr_scale : float
        Multiplicative factor to apply to Sigma_SFR.
    """

    # Cylindrical radius grid
    XX, YY = np.meshgrid(x_grid, y_grid, indexing="xy")
    R_pc = np.sqrt(XX**2 + YY**2)

    # Prefer to average within R <= radius_target_pc, but if for any
    # reason that yields no valid pixels, fall back to all valid pixels.
    mask = np.isfinite(Sigma_SFR) & (Sigma_SFR > 0) & (R_pc <= radius_target_pc)
    if not np.any(mask):
        mask = np.isfinite(Sigma_SFR) & (Sigma_SFR > 0)

    if not np.any(mask):
        raise SystemExit("No valid Σ_SFR pixels to compute Quintana normalisation.")

    mean_Sigma_SFR = float(np.nanmean(Sigma_SFR[mask]))  # Msun/yr/kpc^2

    # Desired mean Σ_SFR inside 1 kpc for the given total SFR:
    #    SFR_target [Msun/Myr] = mean_required [Msun/yr/kpc^2] * π * (1 kpc)^2 * 1e6
    area_kpc2 = np.pi * (radius_target_pc / 1000.0) ** 2  # π * 1^2 = π, but keep general
    mean_required = sfr_target_msun_per_myr / (area_kpc2 * 1e6)

    sfr_scale = mean_required / mean_Sigma_SFR

    if verbose:
        print(
            f"[info] Map-averaged Σ_SFR (R <= {radius_target_pc:.0f} pc): "
            f"{mean_Sigma_SFR:.3e} Msun/yr/kpc^2",
            file=sys.stderr,
        )
        print(
            f"[info] To match SFR={sfr_target_msun_per_myr:.1f} Msun/Myr inside "
            f"{radius_target_pc/1000.0:.1f} kpc, "
            f"normalisation factor would be sfr_scale={sfr_scale:.3f}",
            file=sys.stderr,
        )

    return sfr_scale


def load_sfr_profile(
    sfr_npz_path,
    dR_pc,
    R_max_pc=None,
    sfr_scale=None,
    sfr_target_msun_per_myr=2896.0,
    radius_target_pc=1000.0,
):
    """
    Load Σ_SFR(x,y) from NPZ and compress to ring-averaged Σ_SFR(R).

    Parameters
    ----------
    sfr_npz_path : str
        Path to NPZ from `sfr_map_xy_from_dust.py`.
    dR_pc : float
        Ring width in cylindrical radius [pc].
    R_max_pc : float or None
        Maximum radius to include; if None we use the max radius
        supported by the SFR grid (where Σ_SFR > 0).
    sfr_scale : float or None
        If provided, multiply Σ_SFR by this factor.
        If None, compute a Quintana-based factor.
    sfr_target_msun_per_myr : float
        Total SFR inside radius_target_pc (default: 2896 Msun/Myr).
    radius_target_pc : float
        Radius used for the Quintana normalisation (default: 1 kpc).

    Returns
    -------
    R_centers_pc : (n_R,) array
    Sigma_SFR_R  : (n_R,) array, ring-averaged Σ_SFR after scaling
    R_edges_pc   : (n_R+1,) array
    sfr_scale    : float, the factor actually used
    """
    D = np.load(sfr_npz_path, allow_pickle=True)
    if "x_grid" not in D.files or "y_grid" not in D.files:
        raise SystemExit("sfr_npz must contain x_grid and y_grid.")

    x_grid = D["x_grid"]  # pc
    y_grid = D["y_grid"]  # pc

    if "Sigma_SFR_smoothed" in D.files:
        Sigma_SFR = D["Sigma_SFR_smoothed"]
    elif "Sigma_SFR" in D.files:
        Sigma_SFR = D["Sigma_SFR"]
    else:
        raise SystemExit("sfr_npz must contain Sigma_SFR_smoothed or Sigma_SFR.")

    if Sigma_SFR.shape != (y_grid.size, x_grid.size):
        raise SystemExit("Σ_SFR grid shape mismatch with x_grid/y_grid.")

    # Compute scaling factor if not supplied
    if sfr_scale is None:
        sfr_scale = compute_quintana_scale(
            x_grid,
            y_grid,
            Sigma_SFR,
            sfr_target_msun_per_myr=sfr_target_msun_per_myr,
            radius_target_pc=radius_target_pc,
            verbose=True,
        )

    Sigma_SFR_scaled = Sigma_SFR * float(sfr_scale)

    # Cylindrical radius per cell
    XX, YY = np.meshgrid(x_grid, y_grid, indexing="xy")
    R_grid = np.sqrt(XX**2 + YY**2)

    valid = np.isfinite(Sigma_SFR_scaled) & (Sigma_SFR_scaled > 0)
    if not np.any(valid):
        raise SystemExit("No valid Σ_SFR cells after scaling.")

    R_grid_valid = R_grid[valid]
    if R_max_pc is None:
        R_max_pc = float(np.nanmax(R_grid_valid))

    if not np.isfinite(R_max_pc) or R_max_pc <= 0:
        raise SystemExit("Invalid R_max_pc derived from Σ_SFR grid.")

    # Define radial rings
    R_edges_pc = np.arange(0.0, R_max_pc + dR_pc, dR_pc, dtype=float)
    if R_edges_pc.size < 2:
        R_edges_pc = np.array([0.0, R_max_pc], float)
    n_ring = R_edges_pc.size - 1
    R_centers_pc = 0.5 * (R_edges_pc[:-1] + R_edges_pc[1:])

    # Define radial rings
    R_edges_pc = np.arange(0.0, R_max_pc + dR_pc, dR_pc, dtype=float)
    if R_edges_pc.size < 2:
        R_edges_pc = np.array([0.0, R_max_pc], float)
    n_ring = R_edges_pc.size - 1
    R_centers_pc = 0.5 * (R_edges_pc[:-1] + R_edges_pc[1:])

    # Flatten arrays for ring averaging
    R_flat = R_grid.ravel()
    Sigma_flat = Sigma_SFR_scaled.ravel()
    valid_flat = valid.ravel()

    # Ring-averaged Σ_SFR(R) [Msun / yr / kpc^2]
    Sigma_SFR_R = np.zeros(n_ring, float)
    for i in range(n_ring):
        in_ring = (
            valid_flat
            & (R_flat >= R_edges_pc[i])
            & (R_flat < R_edges_pc[i + 1])
        )
        if np.any(in_ring):
            Sigma_SFR_R[i] = float(np.mean(Sigma_flat[in_ring]))
        else:
            Sigma_SFR_R[i] = 0.0

    # NEW: ring area A_i in kpc^2  (R_edges in pc)
    area_ring_kpc2 = np.pi * (R_edges_pc[1:]**2 - R_edges_pc[:-1]**2) / 1e6

    return R_centers_pc, Sigma_SFR_R, R_edges_pc, area_ring_kpc2, sfr_scale



def load_herbig_counts(
    star_pack_path,
    R_edges_pc,
    M_min,
    M_max,
    n_M_bins,
):
    """
    Load Herbig star positions and masses from star_pack, and
    bin into (R, M) counts.

    Parameters
    ----------
    star_pack_path : str
        NPZ 'star_pack' from build_ppp_all_packs.py.
    R_edges_pc : array
        Radial bin edges [pc].
    M_min, M_max : float
        Minimum/maximum mass for the Herbig range [Msun].
    n_M_bins : int
        Number of mass bins.

    Returns
    -------
    N_obs_RM : (n_R, n_M) array of counts (int)
    M_edges  : (n_M+1,) array
    M_centers: (n_M,) array
    R_star_pc: (N,) array, cylindrical radius of stars used
    M_star   : (N,) array, stellar mass of stars used
    """
    D = np.load(star_pack_path, allow_pickle=True)

    required = ("x_pc", "y_pc", "Mstar")
    if not all(k in D.files for k in required):
        missing = [k for k in required if k not in D.files]
        raise SystemExit(f"star_pack missing required keys: {missing}")

    x_star = np.asarray(D["x_pc"], float)
    y_star = np.asarray(D["y_pc"], float)
    M_star = np.asarray(D["Mstar"], float)

    # Cylindrical radius
    R_star_pc = np.sqrt(x_star**2 + y_star**2)

    # Use only stars with finite R and positive, finite mass
    mask = np.isfinite(R_star_pc) & np.isfinite(M_star) & (M_star > 0)
    R_star_pc = R_star_pc[mask]
    M_star = M_star[mask]

    if R_star_pc.size == 0:
        raise SystemExit("No valid stars with finite (R, M) in star_pack.")

    # Mass bins
    M_edges = np.linspace(M_min, M_max, n_M_bins + 1)
    M_centers = 0.5 * (M_edges[:-1] + M_edges[1:])

    # Counts
    N_obs_RM, R_edges_out, M_edges_out = np.histogram2d(
        R_star_pc,
        M_star,
        bins=[R_edges_pc, M_edges],
    )

    # Ensure edges match
    if not np.allclose(R_edges_pc, R_edges_out):
        raise SystemExit("R_edges from histogram2d differ from input R_edges.")
    if not np.allclose(M_edges, M_edges_out):
        raise SystemExit("M_edges from histogram2d differ from constructed edges.")

    return N_obs_RM, M_edges, M_centers, R_star_pc, M_star


def build_and_sample_model(
    R_centers_pc,
    Sigma_SFR_R,
    area_ring_kpc2,
    N_obs_RM,
    M_centers,
    Mbar,
    bin_frac,
    tau0_Myr=3.0,
    R0_pc=500.0,
    M0_Msun=1.0,
    draws=2000,
    tune=2000,
    chains=4,
    cores=4,
    target_accept=0.9,
):
    """
    Build and sample the PyMC model:

        lambda_ij = k_H * Sigma_SFR_R[i] * tau(M_j) * f_det(R_i, M_j)

    with tau(M) = tau0 * (M/M0)^alpha_tau and
    logit f_det = a0 + a_R ln(R/R0) + a_M ln(M/M0).
    """
        # Filter out rings with zero Σ_SFR_R
    mask_valid = Sigma_SFR_R > 0
    if not np.any(mask_valid):
        raise SystemExit("All Σ_SFR_R rings are zero; nothing to fit.")

    R_centers_pc = R_centers_pc[mask_valid]
    Sigma_SFR_R  = Sigma_SFR_R[mask_valid]
    area_ring_kpc2 = area_ring_kpc2[mask_valid]
    N_obs_RM = N_obs_RM[mask_valid, :]

    # SFR in each ring: Msun / yr
    SFR_ring_yr = Sigma_SFR_R * area_ring_kpc2
    # Convert to Msun / Myr
    SFR_ring_myr = SFR_ring_yr * 1e6

    # Log R, log M
    logR = np.log(R_centers_pc / R0_pc)
    logM = np.log(M_centers / M0_Msun)

    with pm.Model() as model:

        # Global geometric normalisation ONLY

        # Global geometric normalisation:
        # Σ_SFR has already been scaled to match Quintana inside 1 kpc,
        # so k_geom should be O(1) but allowed to go up OR down.
        log_k_geom = pm.TruncatedNormal("log_k_geom", mu=0.0, sigma=LOG_K_GEOM_SIGMA, lower=0.0)

        k_geom = pm.Deterministic("k_geom", pm.math.exp(log_k_geom))    
        # Completeness coefficients
        a0 = pm.Normal("a0", mu=0.0, sigma=2.0)
        a_R = pm.Normal("a_R", mu=-1.0, sigma=1.0)
        a_M = pm.Normal("a_M", mu=1.0, sigma=1.0)

        # Lifetime normalisation at 1 Msun (Myr), log-normal prior.
        # Interpret tau0_Myr argument as the prior median.
        log_tau0 = pm.Normal(
            "log_tau0",
            mu=np.log(tau0_Myr),
            sigma=0.4,          # adjust if you want wider/narrower than ~3–10 Myr
        )
        tau0 = pm.Deterministic("tau0_Myr", pm.math.exp(log_tau0))

        # Lifetime exponent
        alpha_tau = pm.Normal("alpha_tau", mu=-1.5, sigma=0.5)

        # Lifetime at each mass bin [Myr]
        tau_M = tau0 * (M_centers / M0_Msun)**alpha_tau      # (n_M,)
        tau_2d = tau_M[None, :]                              # (1, n_M)


        # SFR per ring [Msun/Myr], broadcast over mass bins
        SFR_2d = SFR_ring_myr[:, None]                            # (n_R, 1)

        # IMF number fractions per bin (constant)
        bin_frac = np.asarray(bin_frac, float)
        frac_2d = bin_frac[None, :]                               # (1, n_M)

        # Completeness
        logit_f = a0 + a_R * logR[:, None] + a_M * logM[None, :]
        f_det = pm.Deterministic("f_det", pm.math.sigmoid(logit_f))

        # Expected counts λ_ij
        lambda_RM = pm.Deterministic(
            "lambda_RM",
            pm.math.exp(log_k_geom) *
            SFR_2d * tau_2d * (frac_2d / Mbar) * f_det
        )

        N_obs = pm.Poisson("N_obs", mu=lambda_RM, observed=N_obs_RM)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            chains=chains,
            cores=cores,
            compute_convergence_checks=True,
            progressbar=True,
        )


    return model, idata


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--sfr_npz", required=True,
                    help="NPZ from sfr_map_xy_from_dust.py (x_grid,y_grid,Sigma_SFR[_smoothed])")
    ap.add_argument("--star_pack", required=True,
                    help="star_pack NPZ from build_ppp_all_packs.py")
    ap.add_argument("--dR_pc", type=float, default=50.0,
                    help="Ring width in cylindrical radius [pc]")
    ap.add_argument("--R_max_pc", type=float, default=None,
                    help="Optional max radius [pc] for rings; "
                         "default uses max radius supported by Σ_SFR grid.")
    ap.add_argument("--M_min", type=float, default=1.5,
                    help="Minimum mass for Herbig range [Msun]")
    ap.add_argument("--M_max", type=float, default=8.0,
                    help="Maximum mass for Herbig range [Msun]")
    ap.add_argument("--n_M_bins", type=int, default=6,
                    help="Number of mass bins")
    ap.add_argument("--tau0_Myr", type=float, default=3.0,
                    help="Lifetime at 1 Msun [Myr], fixed")
    ap.add_argument("--R0_pc", type=float, default=500.0,
                    help="Reference radius [pc] for completeness regression")
    ap.add_argument("--M0_Msun", type=float, default=1.0,
                    help="Reference mass [Msun] for lifetime/completeness")
    ap.add_argument("--sfr_scale", type=float, default=None,
                    help="If set, use this Σ_SFR scaling instead of computing from Quintana.")
    ap.add_argument("--sfr_target_msun_per_myr", type=float, default=2896.0,
                    help="Target SFR inside radius_target_pc [Msun/Myr] (Quintana+ value).")
    ap.add_argument("--radius_target_pc", type=float, default=1000.0,
                    help="Radius used for Quintana normalisation [pc].")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.9)
    ap.add_argument("--out", default="",
                    help="If set, save inference data to this netcdf file.")
    args = ap.parse_args()

    # 1) SFR profile as a function of radius
    R_centers_pc, Sigma_SFR_R, R_edges_pc, area_ring_kpc2, sfr_scale_used = load_sfr_profile(
        sfr_npz_path=args.sfr_npz,
        dR_pc=args.dR_pc,
        R_max_pc=args.R_max_pc,
        sfr_scale=args.sfr_scale,
        sfr_target_msun_per_myr=args.sfr_target_msun_per_myr,
        radius_target_pc=args.radius_target_pc,
    )


     # --- DEBUG: check local SFDR inside 1 kpc ---
    XX, YY = np.meshgrid(R_centers_pc, R_centers_pc, indexing="xy")  # fake, but we only need a 1D mask
    # Better: build a proper radius array for the ring centres
    mask_R1kpc = R_centers_pc <= 1000.0
    mean_scaled = np.nanmean(Sigma_SFR_R[mask_R1kpc])             # Msun/yr/kpc^2
    mean_scaled_myr = mean_scaled * 1e6                           # Msun/Myr/kpc^2

    print(f"[debug] Mean Σ_SFR (R<=1 kpc) after scaling: {mean_scaled:.3e} Msun/yr/kpc^2 "
          f"= {mean_scaled_myr:.1f} Msun/Myr/kpc^2", file=sys.stderr)

    ''''
    import matplotlib.pyplot as plt
    plt.scatter(R_centers_pc, Sigma_SFR_R)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()'''

    print(f"[info] Using Σ_SFR scaling factor sfr_scale={sfr_scale_used:.3f}", file=sys.stderr)

    # 2) Herbig counts in (R, M)
    N_obs_RM, M_edges, M_centers, R_star_pc, M_star = load_herbig_counts(
        star_pack_path=args.star_pack,
        R_edges_pc=R_edges_pc,
        M_min=args.M_min,
        M_max=args.M_max,
        n_M_bins=args.n_M_bins,
    )
    print(f"[info] Binned {R_star_pc.size} stars into N_R={R_centers_pc.size}, N_M={M_centers.size}", file=sys.stderr)

    # --- IMF-derived quantities: <M> and bin number fractions ---
    Mmin_all, Mmax_all = 0.08, 120.0
    Mgrid = np.logspace(np.log10(Mmin_all), np.log10(Mmax_all), 20000)
    xi_vals = xi_norm_on_interval(Mgrid, Mmin=Mmin_all, Mmax=Mmax_all)

    # mean stellar mass <M>
    Mbar = imf_mean_mass(Mmin=Mmin_all, Mmax=Mmax_all)

    # number fraction in each mass bin
    bin_frac = np.zeros(M_centers.size, float)
    for j in range(M_centers.size):
        Mlo, Mhi = M_edges[j], M_edges[j+1]
        mask = (Mgrid >= Mlo) & (Mgrid < Mhi)
        if not np.any(mask):
            continue
        bin_frac[j] = np.trapz(xi_vals[mask], Mgrid[mask])

    # sanity check: total Herbig fraction (optional)
    print(f"[info] Total IMF number fraction in Herbig mass range: {bin_frac.sum():.3e}", file=sys.stderr)


    # 3) Build + sample model
    model, idata = build_and_sample_model(
        R_centers_pc=R_centers_pc,
        Sigma_SFR_R=Sigma_SFR_R,
        area_ring_kpc2=area_ring_kpc2,
        N_obs_RM=N_obs_RM,
        M_centers=M_centers,
        Mbar=Mbar,
        bin_frac=bin_frac,
        tau0_Myr=args.tau0_Myr,
        R0_pc=args.R0_pc,
        M0_Msun=args.M0_Msun,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
    )

    # Quick summary
    summary = az.summary(
        idata,
        var_names=["k_geom", "log_k_geom", "tau0_Myr", "a0", "a_R", "a_M", "alpha_tau"],
        round_to=3,
    )
    print(summary, file=sys.stderr)

    # Save if requested
    if args.out:
        az.to_netcdf(idata, args.out)
        print(f"[OK] Saved trace to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
