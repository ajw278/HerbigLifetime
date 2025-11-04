#!/usr/bin/env python3
"""
PPP likelihood for Herbig stars with distance+mass marginalization.

Inputs:
  --grid_pack        : from build_ppp_all_packs.py (HEALPix×distance cells)
  --mass_pack        : Chabrier IMF info (Herbig mass quadrature) + <M>
  --star_marg_pack   : per-star distance & mass nodes/weights, A_V(d), and Σ_birth at nodes
Options:
  --mu_ref           : center for distance-modulus term in selection (default 10)
  --k_lambda         : A_band / A_V coefficient for selection (default 1.0 uses A_V)
  --sample           : set to run NUTS sampling
  --draws, --tune    : sampling hyperparameters
  --out              : netcdf path to save trace (optional)
"""

import argparse
import numpy as np

# PyMC 5 / PyTensor
import pymc as pm
import pytensor.tensor as pt
from utils import  xi_norm_on_interval

# ---------- helpers ----------
SQ2PI = np.sqrt(2.0 * np.pi)

import numpy as np

def load_grid_standardized(path):
    """
    Load a PPP grid pack and return a standardized view that works for BOTH:
      - rectangular packs (npix × Kd), possibly flattened, and
      - ragged packs (per-pixel variable Kd) with row_ptr.

    Returns a dict with 1-D cellwise arrays:
      npix:int
      ragged:bool
      Gcells:int
      row_ptr: (npix+1,) int64 or None
      mu: (Gcells,) float64                 distance modulus per cell (center)
      z:  (Gcells,) float64                 z (pc) per cell (center)
      dV: (Gcells,) float64                 cell volumes (pc^3)
      Sigma_birth: (Gcells,) float64        stars / (yr pc^2) at cell center
      Av: (Gcells,) float64                 A_V to the cell center (0 if missing)
      footprint: (Gcells,) bool             valid mask
    """
    P = np.load(path, allow_pickle=True)
    files = set(P.files)

    def as1d(name, default=None, dtype=np.float64):
        if name not in P.files:
            if default is None:
                raise KeyError(f"[grid] missing '{name}'")
            arr = np.asarray(default, dtype=dtype)
        else:
            arr = np.asarray(P[name], dtype=dtype)
        if arr.ndim > 1:
            arr = arr.ravel()
        return arr

    # Detect ragged/adaptive pack via 'row_ptr'
    if "row_ptr" in files:
        row_ptr = np.asarray(P["row_ptr"], dtype=np.int64)
        npix = int(row_ptr.size - 1)
        Gcells = int(row_ptr[-1])
        ragged = True

        mu  = as1d("mu")
        z   = as1d("z_pc")
        dV  = as1d("dV_pc3")
        Sig = as1d("Sigma_birth_yr_pc2")
        Av  = as1d("A_V", default=np.zeros(Gcells, float)) if ("A_V" in files) else np.zeros(Gcells, float)

        fp = np.asarray(P["footprint"]).astype(bool)
        if fp.ndim > 1:
            fp = fp.ravel()
        if fp.size != Gcells:
            # be forgiving: if someone saved per-pixel, broadcast
            if fp.size == npix:
                fp = np.repeat(fp, np.diff(row_ptr))
            else:
                raise ValueError(f"[grid] 'footprint' size {fp.size} != Gcells {Gcells}")

        return dict(npix=npix, ragged=ragged, Gcells=Gcells, row_ptr=row_ptr,
                    mu=mu.astype(float), z=z.astype(float), dV=dV.astype(float),
                    Sigma_birth=Sig.astype(float), Av=Av.astype(float),
                    footprint=fp)

    # Rectangular (legacy) pack
    # We accept either flattened (Gcells,) or (npix, Kd).
    ragged = False
    if "npix" in files:
        npix = int(P["npix"])
    else:
        # try to infer from something 2D
        npix = None

    def flatten2d(name, fallback=None, dtype=np.float64):
        if name not in P.files:
            if fallback is None:
                raise KeyError(f"[grid] missing '{name}'")
            arr = np.asarray(fallback, dtype=dtype)
        else:
            arr = np.asarray(P[name], dtype=dtype)
        if arr.ndim == 2:
            return arr.reshape(-1)
        return arr

    # Prefer directly stored quantities
    mu  = flatten2d("mu")
    z   = flatten2d("z_pc")
    dV  = flatten2d("dV_pc3")
    Sig = flatten2d("Sigma_birth_yr_pc2")
    Av  = flatten2d("A_V", fallback=np.zeros_like(mu))

    fp = np.asarray(P["footprint"]).astype(bool) if "footprint" in files else np.ones_like(mu, bool)
    if fp.ndim > 1:
        fp = fp.ravel()

    # Consistency
    Gcells = mu.size
    if not (z.size == dV.size == Sig.size == fp.size == Gcells):
        raise ValueError("[grid] field sizes are inconsistent after flattening.")

    return dict(npix=int(npix) if npix is not None else -1,
                ragged=ragged, Gcells=Gcells, row_ptr=None,
                mu=mu.astype(float), z=z.astype(float), dV=dV.astype(float),
                Sigma_birth=Sig.astype(float), Av=Av.astype(float),
                footprint=fp)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--grid_pack", required=True)
    ap.add_argument("--mass_pack", required=True)
    ap.add_argument("--star_marg_pack", required=True)
    ap.add_argument("--mu_ref", type=float, default=10.0)
    ap.add_argument("--k_lambda", type=float, default=1.0)
    ap.add_argument("--sample", action="store_true", help="Run NUTS sampling")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--target_accept", type=float, default=0.9)
    ap.add_argument("--out", default="", help="If set, save trace to this netcdf file")
    ap.add_argument("--zsun_mu_pc", type=float, default=20.8,
                help="Prior mean for Sun height above the Galactic mid-plane [pc]")
    ap.add_argument("--zsun_sigma_pc", type=float, default=2.0,
                    help="Prior sigma for z_sun [pc] (small, informative)")
    args = ap.parse_args()

    # ---- load packs ----
    G = np.load(args.grid_pack, allow_pickle=True)
    M = np.load(args.mass_pack, allow_pickle=True)
    S = np.load(args.star_marg_pack, allow_pickle=True)

    # ---------- GRID (works for rectangular or ragged) ----------
    grid = load_grid_standardized(args.grid_pack)

    footprint   = grid["footprint"]
    Sigma_birth = np.where(np.isfinite(grid["Sigma_birth"]) & footprint, grid["Sigma_birth"], 0.0)

    z_grid  = np.nan_to_num(grid["z"],  nan=0.0, posinf=0.0, neginf=0.0)
    mu_grid = np.nan_to_num(grid["mu"], nan=0.0, posinf=0.0, neginf=0.0)
    dV_grid = np.where(np.isfinite(grid["dV"]), grid["dV"], 0.0)

    Av_grid = grid["Av"] if grid["Av"].size else np.zeros_like(mu_grid)
    Av_grid = np.nan_to_num(Av_grid, nan=0.0, posinf=0.0, neginf=0.0)

    footprint_f = footprint.astype(np.float64)  # 1.0 in support, 0.0 outside)

    # ---------- MASS QUADRATURE (Herbig range) ----------
    Mq   = M["Mq"].astype(np.float64)    # (Q,)
    Wq   = M["Wq"].astype(np.float64)    # (Q,)
    xi_q = M["xi_q"].astype(np.float64)  # (Q,)
    log10_Mq = np.log10(Mq)

    # ---------- STAR NODES (N, Kd/Km) ----------
    # SANITIZE star-node arrays
    birth_nodes = S["birth_nodes"].astype(np.float64)  # Σ_birth at (x,y) nodes
    birth_nodes = np.where(np.isfinite(birth_nodes) & (birth_nodes > 0.0), birth_nodes, 1e-300)

    z_nodes  = np.nan_to_num(S["z_nodes"].astype(np.float64),  nan=0.0, posinf=0.0, neginf=0.0)
    mu_nodes = np.nan_to_num(S["mu_nodes"].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Av_nodes = np.nan_to_num(S["Av_nodes"].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    d_w = S["d_weights"].astype(np.float64)
    d_w = np.where(np.isfinite(d_w) & (d_w > 0.0), d_w, 1e-300)

    M_nodes = S["M_nodes"].astype(np.float64)
    M_nodes = np.where(np.isfinite(M_nodes) & (M_nodes > 0.0), M_nodes, 1.0)
    logM_nodes = np.log10(M_nodes)

    M_w = S["M_weights"].astype(np.float64)
    M_w = np.where(np.isfinite(M_w) & (M_w > 0.0), M_w, 1e-300)

    # IMF factor at star mass nodes (normalized on [0.08,120] Msun)
    xi_nodes = xi_norm_on_interval(M_nodes)

    N, Kd = birth_nodes.shape
    Km = M_nodes.shape[1]
    Q  = Mq.size

    # Cast masks & constants
    footprint_f = footprint.astype(np.float64)  # 1.0 where valid, 0.0 else
    Av_grid_scaled = args.k_lambda * Av_grid
    Av_nodes_scaled = args.k_lambda * Av_nodes

    # ---- PyMC model ----
    with pm.Model() as model:
        # ----- Priors -----
        log10_tau0_Myr = pm.Normal("log10_tau0_Myr", mu=np.log10(3.0), sigma=0.2)
        beta           = pm.Normal("beta", mu=-0.7, sigma=1.0)
        h_z_pc         = pm.HalfNormal("h_z_pc", sigma=60.0)

        # selection coefficients
        a0     = pm.Normal("a0", mu=0.0, sigma=3.0)
        a_mu   = pm.Normal("a_mu", mu=-0.5, sigma=0.5)
        a_Av   = pm.Normal("a_Av", mu=-1.0, sigma=1.0)
        a_logM = pm.Normal("a_logM", mu=1.0, sigma=1.0)

        # global scale for birth map (absorbs Σ_SFR→Σ_birth calibration etc.)
        s_birth = pm.LogNormal("s_birth", mu=0.0, sigma=1.0)

        # ----- Deterministic components -----
        tau0_yr = (10.0 ** log10_tau0_Myr) * 1.0e6  # years

        z_sun_pc = pm.Normal("z_sun_pc", mu=args.zsun_mu_pc, sigma=args.zsun_sigma_pc)

        # ----- Deterministic components -----
        SQ2PI = np.sqrt(2.0 * np.pi)

        def fz(z):
            return (1.0 / (SQ2PI * h_z_pc)) * pm.math.exp(-0.5 * (z / h_z_pc) ** 2)

        # Use mid-plane heights in BOTH normalization grid and per-star nodes
        z_grid_mid  = pt.as_tensor_variable(z_grid)  + z_sun_pc   # (G,)
        z_nodes_mid = pt.as_tensor_variable(z_nodes) + z_sun_pc   # (N, Kd)

        fz_grid  = fz(z_grid_mid)          # replaces previous fz(pt.as_tensor_variable(z_grid))
        fz_nodes = fz(z_nodes_mid)         # replaces previous fz(pt.as_tensor_variable(z_nodes))

        # ---------- Normalization integral Λ ----------
        # cell base factor (stars/yr per cell, up to the mass sum and s_birth)
        cell_base = s_birth * pt.as_tensor_variable(Sigma_birth) * fz_grid * pt.as_tensor_variable(dV_grid)
        cell_base = cell_base * pt.as_tensor_variable(footprint_f)  # zero outside support

        # selection term over mass grid (broadcast add)
        s_g = a0 + a_mu * (pt.as_tensor_variable(mu_grid) - args.mu_ref) + a_Av * pt.as_tensor_variable(Av_grid_scaled)
        # shapes: s_g (G,), log10_Mq (Q,)
        eta_gq = s_g[:, None] + a_logM * pt.as_tensor_variable(log10_Mq)[None, :]
        pdet_gq = pm.math.sigmoid(eta_gq)  # (G,Q)

        # lifetime factor on mass grid (Q,)
        tau_q = tau0_yr * (pt.as_tensor_variable(Mq) ** beta)  # years
        mass_kernel_q = pt.as_tensor_variable(Wq * xi_q) * tau_q  # (Q,)

        mass_sum_grid = pt.sum(pdet_gq * mass_kernel_q[None, :], axis=1)  # (G,)

        Lambda = pt.sum(cell_base * mass_sum_grid)  # scalar

        # ---------- Data term (star-by-star, marginalized) ----------
        # selection term on star nodes
        s_ik = a0 + a_mu * (pt.as_tensor_variable(mu_nodes) - args.mu_ref) + a_Av * pt.as_tensor_variable(Av_nodes_scaled)  # (N,Kd)
        # broadcast with mass nodes
        eta_ikl = s_ik[:, :, None] + a_logM * pt.as_tensor_variable(logM_nodes)[:, None, :]  # (N,Kd,Km)

        # log p_det for numerical stability
        log_pdet_ikl = pm.math.log(pm.math.sigmoid(eta_ikl))  # (N,Kd,Km)

        # lifetime and IMF at star mass nodes
        tau_nodes_mass = tau0_yr * (pt.as_tensor_variable(M_nodes) ** beta)       # (N,Km)
        xi_nodes_t = pt.as_tensor_variable(xi_nodes)                               # (N,Km)

        # safe positives
        birth_nodes_t = pt.clip(pt.as_tensor_variable(birth_nodes), 1e-300, np.inf)  # (N,Kd)
        fz_nodes_t    = pt.clip(fz_nodes, 1e-300, np.inf)                            # (N,Kd)
        d_w_t         = pt.clip(pt.as_tensor_variable(d_w), 1e-300, 1.0)            # (N,Kd)
        M_w_t         = pt.clip(pt.as_tensor_variable(M_w), 1e-300, 1.0)            # (N,Km)

        # log λ at each (k,l)
        log_lambda_ikl = (
            pt.log(s_birth) +
            pt.log(birth_nodes_t)[:, :, None] +
            pt.log(fz_nodes_t)[:, :, None] +
            pt.log(xi_nodes_t)[:, None, :] +
            pt.log(tau_nodes_mass)[:, None, :] +
            log_pdet_ikl
        )  # (N,Kd,Km)

        # add log-weights and log-sum-exp over (k,l)
        log_w_ikl = pt.log(d_w_t)[:, :, None] + pt.log(M_w_t)[:, None, :]
        log_intensity_i = pm.math.logsumexp(log_lambda_ikl + log_w_ikl, axis=(1, 2))  # (N,)

        # PPP log-likelihood
        pm.Potential("PPP_loglike", pt.sum(log_intensity_i) - Lambda)

        # -------- sampling (optional) --------
        idata = None
        if args.sample:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                target_accept=args.target_accept,
                chains=4,
                cores=4,
                compute_convergence_checks=True,
                progressbar=True,
            )
            if args.out:
                import arviz as az
                az.to_netcdf(idata, args.out)
                print(f"[OK] saved trace to {args.out}")

    # If not sampling, just build the model to catch shape errors
    if not args.sample:
        print("[OK] Built PPP model graph successfully.")

if __name__ == "__main__":
    main()
