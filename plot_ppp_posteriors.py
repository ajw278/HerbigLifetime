#!/usr/bin/env python3
# plot_ppp_posteriors.py
"""
Visualize PPP NUTS results (ArviZ netcdf) for the Herbig PPP model.

Features
--------
- Trace + posterior plots for main parameters
- Lifetime curve tau(M) with 68% band across Herbig mass range
- Vertical profile f_z(z) with 68% band
- Selection curves p_det vs mu and A_V (if selection coefficients present)
- Optional recomputation of Lambda (normalization) if packs are provided

Usage
-----
python plot_ppp_posteriors.py --trace ppp_trace.nc --outdir figs \
    --mass_pack ppp_mass_pack.npz \
    --grid_pack ppp_grid_pack.npz --mu_ref 10 --k_lambda 1.0 --compute_lambda

"""

import argparse, os, json
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# ---------------- helpers ----------------

def flatten_draws(idata, name):
    """Return 1D numpy array of posterior draws for var `name` or None if missing."""
    if "posterior" not in idata.groups():
        return None
    post = idata.posterior
    if name not in post:
        return None
    x = post[name].values  # shape (chain, draw[, ...])
    return np.reshape(x, (-1,) + x.shape[2:])  # squash chains/draws

def pick_first(*candidates):
    """Return the name of the first variable that exists in the posterior."""
    for nm in candidates:
        if nm is not None:
            return nm
    return None

def compute_tau_M(draws_dict, M_grid, M_pivot=2.5):
    """
    Build tau(M) curves for all posterior draws.
    Supports either:
      - pivoted: log10_tau_p + beta*(log10(M)-log10(M_pivot))
      - classic: tau0_Myr -> tau0_yr = 10**log10_tau0_Myr * 1e6; tau = tau0_yr * M**beta
    """
    beta = draws_dict.get("beta")  # (n_draw,)
    if beta is None:
        raise RuntimeError("No 'beta' found in posterior.")
    n = beta.shape[0]
    M = np.asarray(M_grid)[None, :]  # (1, nM)

    if "log10_tau_p" in draws_dict:
        log10_tau_p = draws_dict["log10_tau_p"][:, None]  # (n_draw,1)
        log10_Mp = np.log10(M_pivot)
        log10_tau = log10_tau_p + beta[:, None] * (np.log10(M) - log10_Mp)
        tau = 10.0**log10_tau
        return tau  # years
    elif "log10_tau0_Myr" in draws_dict:
        tau0_yr = (10.0**draws_dict["log10_tau0_Myr"]) * 1e6  # (n_draw,)
        tau = (tau0_yr[:, None]) * (M ** beta[:, None])
        return tau  # years
    else:
        raise RuntimeError("Expected 'log10_tau_p' or 'log10_tau0_Myr' in posterior.")

def fz_of_z(draws_dict, z_grid_pc):
    """Compute f_z(z) for each draw using Gaussian vertical profile."""
    h = draws_dict.get("h_z_pc")
    if h is None:
        raise RuntimeError("No 'h_z_pc' found in posterior.")
    z = np.asarray(z_grid_pc)[None, :]
    SQ2PI = np.sqrt(2.0*np.pi)
    fz = (1.0/(SQ2PI*h[:, None])) * np.exp(-0.5*(z/h[:, None])**2)
    return fz  # pc^-1

def logistic(x):
    return 1.0/(1.0 + np.exp(-x))

def selection_curves(draws_dict, mu_grid, Av_grid, logM_ref, mu_ref=10.0, k_lambda=1.0):
    """Compute p_det over grids for posterior draws; returns median & 16/84 bands."""
    need = ["a0", "a_mu", "a_Av", "a_logM"]
    if any(nm not in draws_dict for nm in need):
        return None
    a0     = draws_dict["a0"][:, None]
    a_mu   = draws_dict["a_mu"][:, None]
    a_Av   = draws_dict["a_Av"][:, None]
    a_logM = draws_dict["a_logM"][:, None]

    mu = mu_grid[None, :]
    Av = Av_grid[None, :]
    logM = float(logM_ref)

    # 1) p_det vs mu at Av=Av0 (use median Av_grid mid-point)
    Av0 = float(np.median(Av_grid))
    eta_mu = a0 + a_mu*(mu - mu_ref) + a_Av*(k_lambda*Av0) + a_logM*logM
    p_mu = logistic(eta_mu)

    # 2) p_det vs Av at mu=mu0 (use median mu_grid mid-point)
    mu0 = float(np.median(mu_grid))
    eta_Av = a0 + a_mu*(mu0 - mu_ref) + a_Av*(k_lambda*Av) + a_logM*logM
    p_Av = logistic(eta_Av)

    def bands(A):
        return np.percentile(A, [16, 50, 84], axis=0)

    return dict(
        mu=dict(grid=mu_grid, p16_50_84=bands(p_mu)),
        Av=dict(grid=Av_grid, p16_50_84=bands(p_Av)),
        settings=dict(mu0=mu0, Av0=Av0, mu_ref=mu_ref, k_lambda=k_lambda, logM_ref=logM)
    )

def ensure_outdir(d):
    os.makedirs(d, exist_ok=True)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--trace", required=True, help="ArviZ netcdf from PyMC (e.g., ppp_trace.nc)")
    ap.add_argument("--outdir", default="ppp_figs", help="Directory to save figures")
    # Lifetime curve controls
    ap.add_argument("--Mmin_herbig", type=float, default=1.5)
    ap.add_argument("--Mmax_herbig", type=float, default=8.0)
    ap.add_argument("--M_pivot", type=float, default=2.5)
    ap.add_argument("--nM", type=int, default=200)
    # Selection plots
    ap.add_argument("--mu_min", type=float, default=5.0)
    ap.add_argument("--mu_max", type=float, default=12.0)
    ap.add_argument("--Av_min", type=float, default=0.0)
    ap.add_argument("--Av_max", type=float, default=5.0)
    ap.add_argument("--k_lambda", type=float, default=1.0, help="Band-to-V factor (A_band = k_lambda * A_V)")
    ap.add_argument("--mu_ref", type=float, default=10.0)
    # Optional packs for labels/defaults
    ap.add_argument("--mass_pack", default="", help="If given, read Herbig mass range from here")
    ap.add_argument("--grid_pack", default="", help="If given with --compute_lambda, recompute Λ")
    ap.add_argument("--compute_lambda", action="store_true", help="Recompute Λ distribution (requires grid & mass packs)")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    idata = az.from_netcdf(args.trace)

    # Collect draws we need
    names = ["log10_tau_p", "log10_tau0_Myr", "beta", "h_z_pc", "a0", "a_mu", "a_Av", "a_logM", "s_birth"]
    draws = {}
    for nm in names:
        arr = flatten_draws(idata, nm)
        if arr is not None:
            draws[nm] = arr.squeeze()

    # If mass_pack exists, take Herbig range from it
    MminH, MmaxH = args.Mmin_herbig, args.Mmax_herbig
    if args.mass_pack and os.path.exists(args.mass_pack):
        MP = np.load(args.mass_pack)
        MminH = float(MP.get("Mmin_herbig", MminH))
        MmaxH = float(MP.get("Mmax_herbig", MmaxH))

    # ----------- 1) ArviZ trace & posterior ----------
    try:
        az.plot_trace(idata, var_names=[nm for nm in ["log10_tau_p","log10_tau0_Myr","beta","h_z_pc","a0","a_mu","a_Av","a_logM","s_birth"] if nm in idata.posterior])
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "trace_main.png"), dpi=160)
        plt.close()
    except Exception as e:
        print("[warn] trace plot skipped:", e)

    try:
        az.plot_posterior(idata, var_names=[nm for nm in ["log10_tau_p","log10_tau0_Myr","beta","h_z_pc","a0","a_mu","a_Av","a_logM","s_birth"] if nm in idata.posterior], hdi_prob=0.68)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "posterior_main.png"), dpi=160)
        plt.close()
    except Exception as e:
        print("[warn] posterior plot skipped:", e)

    # ----------- 2) Lifetime curve tau(M) ------------
    try:
        M_grid = np.logspace(np.log10(MminH), np.log10(MmaxH), args.nM)
        tau = compute_tau_M(draws, M_grid, M_pivot=args.M_pivot)  # (ndraws, nM)
        p16, p50, p84 = np.percentile(tau, [16,50,84], axis=0)
        plt.figure(figsize=(6,4))
        plt.fill_between(M_grid, p16, p84, alpha=0.25, label="68% band")
        plt.plot(M_grid, p50, lw=2, label="median")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Mass $M$ [$M_\\odot$]"); plt.ylabel("$\\tau(M)$ [yr]")
        plt.title("Posterior lifetime curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "posterior_tau_of_M.png"), dpi=180)
        plt.close()
    except Exception as e:
        print("[warn] tau(M) plot skipped:", e)

    # ----------- 3) Vertical profile f_z(z) ----------
    try:
        z_grid = np.linspace(-300, 300, 601)  # pc
        fz = fz_of_z(draws, z_grid)  # (ndraws, nz)
        p16, p50, p84 = np.percentile(fz, [16,50,84], axis=0)
        plt.figure(figsize=(6,4))
        plt.fill_between(z_grid, p16, p84, alpha=0.25, label="68% band")
        plt.plot(z_grid, p50, lw=2, label="median")
        plt.xlabel("$z$ [pc]"); plt.ylabel("$f_z(z)$ [pc$^{-1}$]")
        plt.title("Posterior vertical profile")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "posterior_fz.png"), dpi=180)
        plt.close()
    except Exception as e:
        print("[warn] f_z plot skipped:", e)

    # ----------- 4) Selection curves -----------------
    try:
        if all(nm in draws for nm in ["a0","a_mu","a_Av","a_logM"]):
            mu_grid = np.linspace(args.mu_min, args.mu_max, 200)
            Av_grid = np.linspace(args.Av_min, args.Av_max, 200)
            # Use a representative mass ~2.5 Msun
            logM_ref = np.log10(2.5)
            sel = selection_curves(draws, mu_grid, Av_grid, logM_ref, mu_ref=args.mu_ref, k_lambda=args.k_lambda)
            if sel is not None:
                # p_det vs mu at Av0
                g = sel["mu"]["grid"]; p16,p50,p84 = sel["mu"]["p16_50_84"]
                plt.figure(figsize=(6,4))
                plt.fill_between(g, p16, p84, alpha=0.25)
                plt.plot(g, p50, lw=2)
                plt.ylim(0,1); plt.xlabel("$\\mu$ (distance modulus)"); plt.ylabel("$p_\\mathrm{det}$")
                plt.title(f"Selection vs $\\mu$ (A_V={sel['settings']['Av0']:.2f}, logM={sel['settings']['logM_ref']:.2f})")
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, "selection_vs_mu.png"), dpi=180)
                plt.close()

                # p_det vs Av at mu0
                g = sel["Av"]["grid"]; p16,p50,p84 = sel["Av"]["p16_50_84"]
                plt.figure(figsize=(6,4))
                plt.fill_between(g, p16, p84, alpha=0.25)
                plt.plot(g, p50, lw=2)
                plt.ylim(0,1); plt.xlabel("$A_V$ [mag]"); plt.ylabel("$p_\\mathrm{det}$")
                plt.title(f"Selection vs $A_V$ ($\\mu$={sel['settings']['mu0']:.2f}, logM={sel['settings']['logM_ref']:.2f})")
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, "selection_vs_Av.png"), dpi=180)
                plt.close()
        else:
            print("[info] selection coefficients not all present; skipping selection plots.")
    except Exception as e:
        print("[warn] selection plots skipped:", e)

    # ----------- 5) Summary table --------------------
    try:
        wanted = [nm for nm in ["log10_tau_p","log10_tau0_Myr","beta","h_z_pc","a0","a_mu","a_Av","a_logM","s_birth"] if nm in idata.posterior]
        summ = az.summary(idata, var_names=wanted, hdi_prob=0.68, kind="stats")
        # Save a simple text + CSV
        txt = summ.to_string()
        with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
            f.write(txt + "\n")
        summ.to_csv(os.path.join(args.outdir, "summary.csv"))
        print("[OK] wrote summary to", os.path.join(args.outdir, "summary.{txt,csv}"))
    except Exception as e:
        print("[warn] summary failed:", e)

    # ----------- 6) Optional: recompute Lambda -------
    if args.compute_lambda and args.grid_pack and args.mass_pack and \
       os.path.exists(args.grid_pack) and os.path.exists(args.mass_pack):
        try:
            G = np.load(args.grid_pack)
            MP = np.load(args.mass_pack)
            Sigma_birth = G["Sigma_birth_yr_pc2"].astype(float)
            z_grid = G["z_pc"].astype(float)
            mu_grid_cells = G["mu"].astype(float)
            dV = G["dV_pc3"].astype(float)
            footprint = G["footprint"].astype(bool)
            Av_grid_cells = G["A_V"].astype(float) if "A_V" in G.files else np.zeros_like(mu_grid_cells)

            # Mass quadrature
            Mq = MP["Mq"].astype(float); Wq = MP["Wq"].astype(float); xi_q = MP["xi_q"].astype(float)
            log10_Mq = np.log10(Mq)

            # Grab equal number of posterior draws (thin if large)
            n_draws = min(400, draws["beta"].shape[0])
            sel_idx = np.linspace(0, draws["beta"].shape[0]-1, n_draws).astype(int)

            # Build Lambda per draw
            SQ2PI = np.sqrt(2*np.pi)
            Lambda = np.zeros(n_draws, float)
            for j, k in enumerate(sel_idx):
                # lifetime
                if "log10_tau_p" in draws:
                    log10_tau_p = draws["log10_tau_p"][k]
                    beta = draws["beta"][k]
                    tau_q = 10**(log10_tau_p + beta*(log10_Mq - np.log10(args.M_pivot)))
                else:
                    tau0_yr = (10**draws["log10_tau0_Myr"][k]) * 1e6
                    beta = draws["beta"][k]
                    tau_q = tau0_yr * (Mq**beta)

                # vertical profile
                h = draws["h_z_pc"][k]
                fz = (1.0/(SQ2PI*h)) * np.exp(-0.5*(z_grid/h)**2)

                # selection
                if all(nm in draws for nm in ["a0","a_mu","a_Av","a_logM"]):
                    a0 = draws["a0"][k]; a_mu = draws["a_mu"][k]
                    a_Av = draws["a_Av"][k]; a_logM = draws["a_logM"][k]
                    eta = a0 + a_mu*(mu_grid_cells - args.mu_ref) + a_Av*(args.k_lambda*Av_grid_cells)
                    pdet = 1/(1+np.exp(-(eta[:, None] + a_logM*log10_Mq[None, :])))  # (G,Q)
                else:
                    pdet = np.ones((mu_grid_cells.size, Mq.size), float)

                mass_sum = np.sum(pdet * (Wq*xi_q*tau_q)[None, :], axis=1)  # (G,)
                cell_base = (Sigma_birth * fz * dV) * footprint
                Lambda[j] = np.sum(cell_base * mass_sum)

            # Plot Lambda histogram
            plt.figure(figsize=(6,4))
            plt.hist(Lambda, bins=30, density=True)
            plt.xlabel("$\\Lambda$ (expected count)")
            plt.ylabel("density")
            plt.title("Posterior of $\\Lambda$ (recomputed)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "Lambda_hist.png"), dpi=180)
            plt.close()

            # Save numbers
            np.savez_compressed(os.path.join(args.outdir, "Lambda_values.npz"), Lambda=Lambda)
            print("[OK] recomputed Lambda; mean=%.2f median=%.2f" % (np.mean(Lambda), np.median(Lambda)))
        except Exception as e:
            print("[warn] Lambda recomputation failed:", e)

    print("[DONE] Plots saved in:", args.outdir)

if __name__ == "__main__":
    main()
