#!/usr/bin/env python3
"""
Estimate the number of Herbig Ae/Be stars and total PMS intermediate-mass stars within 1 kpc,
using the renormalised Σ_SFR map and fitted lifetime model.

Inputs:
    --sfr_grid    sfr_grid_xy_from_dust.npz  (Σ_SFR_smoothed in Msun/yr/kpc^2, already Quintana-scaled)
    --tau0_Myr    Best-fit τ0 [Myr]
    --alpha_tau   Best-fit slope α_τ
    --Mmin        Minimum stellar mass [Msun] (default 1.5)
    --Mmax        Maximum stellar mass [Msun] (default 12)
    --Rmax_pc     Outer radius for integration [pc] (default 1000)
"""

import numpy as np
import argparse
from scipy.integrate import simpson

# --- IMF helper: Chabrier (2003) system IMF ---
def chabrier_imf(M):
    """dN/dlogM (unnormalised)"""
    M = np.asarray(M)
    imf = np.zeros_like(M)
    # Chabrier 2003 lognormal for M<1 Msun, power law for M>1 Msun
    mask_low = M < 1.0
    mask_high = ~mask_low
    imf[mask_low] = np.exp(-((np.log10(M[mask_low]) - np.log10(0.079))**2) / (2 * 0.69**2))
    imf[mask_high] = M[mask_high] ** -1.3
    return imf

def normalize_imf(Mgrid, imf):
    dlogM = np.log(Mgrid[1]/Mgrid[0])
    norm = np.trapz(imf, dx=dlogM)
    return imf / norm

# --- PMS lifetime scaling ---
def t_pms_Myr(M):
    """Approximate PMS lifetime [Myr]."""
    return 10.0 * M**-2.5

# --- Main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfr_grid", required=True)
    ap.add_argument("--tau0_Myr", type=float, required=True)
    ap.add_argument("--alpha_tau", type=float, required=True)
    ap.add_argument("--sfr_norm", type=float, default=1.0)
    ap.add_argument("--Mmin", type=float, default=1.5)
    ap.add_argument("--Mmax", type=float, default=12.0)
    ap.add_argument("--Rmax_pc", type=float, default=1000.0)
    args = ap.parse_args()

    data = np.load(args.sfr_grid, allow_pickle=True)
    x = data["x_grid"]; y = data["y_grid"]
    Sigma_SFR = data["Sigma_SFR_smoothed"]  # Msun/yr/kpc^2, renormalised

    XX, YY = np.meshgrid(x, y, indexing="xy")
    R_pc = np.sqrt(XX**2 + YY**2)
    mask = np.isfinite(Sigma_SFR) & (R_pc <= args.Rmax_pc)
    dx_pc = x[1] - x[0]; dy_pc = y[1] - y[0]
    A_pix_kpc2 = (dx_pc/1000.0)*(dy_pc/1000.0)

    total_SFR = np.nansum(Sigma_SFR[mask] * A_pix_kpc2)*args.sfr_norm  # Msun/yr
    print(f"Total SFR(<{args.Rmax_pc:.0f} pc): {1e6*total_SFR:.2f} Msun/Myr")

    # Mass grid for IMF integration
    Mgrid = np.logspace(np.log10(0.08), np.log10(120), 2000)
    xi = normalize_imf(Mgrid, chabrier_imf(Mgrid))
    dlogM = np.log(Mgrid[1]/Mgrid[0])

    # --- Compute fractions ---
    total_number = np.trapz(xi, dx=dlogM)
    mask_H = (Mgrid >= args.Mmin) & (Mgrid <= args.Mmax)
    f_H = np.trapz(xi[mask_H], dx=dlogM) / total_number
    Mbar = np.trapz(Mgrid * xi, dx=dlogM) / total_number

    # --- Herbig lifetime model ---
    tau_Myr = args.tau0_Myr * (Mgrid/Mgrid[1000])**args.alpha_tau  # 1 Msun reference ~ index ~1000

    # IMF-averaged effective Herbig lifetime in the 1.5–12 Msun range
    tau_eff_H = np.trapz(tau_Myr[mask_H]*xi[mask_H], dx=dlogM) / np.trapz(xi[mask_H], dx=dlogM)

    # --- PMS lifetime for same range ---
    tau_pms = t_pms_Myr(Mgrid)
    tau_eff_PMS = np.trapz(tau_pms[mask_H]*xi[mask_H], dx=dlogM) / np.trapz(xi[mask_H], dx=dlogM)

    # --- Expected numbers ---
    N_Herbig = (total_SFR / Mbar) * f_H * tau_eff_H * 1e6  # Msun/yr × yr→Myr
    N_PMS = (total_SFR / Mbar) * f_H * tau_eff_PMS * 1e6

    print(f"IMF fraction f_H(1.5–12 Msun) = {f_H:.3f}")
    print(f"Mean stellar mass Mbar = {Mbar:.2f} Msun")
    print(f"Effective Herbig lifetime ⟨τ_H⟩ = {tau_eff_H:.2f} Myr")
    print(f"Effective PMS lifetime ⟨t_PMS⟩ = {tau_eff_PMS:.2f} Myr")
    print("")
    print(f"⇒ Expected number of Herbigs (<{args.Rmax_pc:.0f} pc): {N_Herbig:.0f}")
    print(f"⇒ Expected number of PMS stars (<{args.Rmax_pc:.0f} pc): {N_PMS:.0f}")

if __name__ == "__main__":
    main()
