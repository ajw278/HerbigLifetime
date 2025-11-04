#!/usr/bin/env python3
"""
Map expected number of Herbig stars in HEALPix pixels for a given mass range,
and overlay real stars from a catalog (filtered by the same mass range).

Requirements:
  numpy, matplotlib, arviz, healpy, (optional) pandas

Inputs:
  --grid_pack     : NPZ with HEALPix grid (distance nodes, A_V along LoS, etc.)
  --sfr_npz       : NPZ with x_grid, y_grid, Sigma_SFR (from your dust-based map)
  --trace         : ArviZ netcdf with posterior samples
  --catalog       : TSV/CSV with at least (ra, dec) and stellar mass column (default Mstar)

Core model pieces:
  logit p_det = a0 + a_mu*(mu - mu_ref) + a_Av*(k_lambda * A_V) + a_logM * log10(M)
  tau(M)      = 10^{log10_tau_p + beta*(log10 M - log10 M_pivot)}   (pivot form)   OR
                10^{log10_tau0_Myr + 6 + beta * log10 M}            (tau0 in Myr)
  f_z(z)      = N(0, h_z_pc) along z (with z_sun shift)
  IMF         : Chabrier (2003) system IMF; uses power-law tail above 1 Msun (sufficient for Herbig range)

Expected observed count per voxel:
  dN = s_birth * Sigma_SFR(x,y) * f_z(z) * [IMF⊗tau⊗p_det over M-bin] * dV

Notes:
  - We use posterior MEDIANS for speed; can extend to sample-averaging if needed.
  - Sigma_SFR is in Msun/yr/kpc^2; absolute scaling handled by s_birth (from your fit).
  - We integrate over M on a log grid inside the requested [Mmin, Mmax].
"""
import argparse, os, sys
import numpy as np
import arviz as az
import healpy as hp
import matplotlib.pyplot as plt
from utils import  xi_norm_on_interval
from mpl_setup import *

DEG2_PER_SR = (180.0/np.pi)**2

try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    HAVE_ASTROPY = True
except Exception:
    HAVE_ASTROPY = False

# ---------- small helpers ----------

def logistic(x):  # stable enough for our ranges
    return 1.0/(1.0 + np.exp(-x))

def load_catalog_icrs(catalog_path, col_ra, col_dec, col_mass):
    try:
        import pandas as pd
        sep = "\t" if catalog_path.lower().endswith(".tsv") else ","
        df = pd.read_csv(catalog_path, sep=sep)
        ra  = df[col_ra].to_numpy(float)
        dec = df[col_dec].to_numpy(float)
        M   = df[col_mass].to_numpy(float)
    except Exception:
        data = np.genfromtxt(catalog_path, names=True, dtype=None, encoding=None)
        ra  = np.asarray(data[col_ra], float)
        dec = np.asarray(data[col_dec], float)
        M   = np.asarray(data[col_mass], float)
    return ra, dec, M

def catalog_galactic(ra_deg, dec_deg):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sky = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs").galactic
    return sky.l.deg, sky.b.deg

def overlay_points(l_deg, b_deg, sel, dot_size=10.0, color="w", edge="k", alpha=0.9, zorder=15):
    if np.any(sel):
        hp.projscatter(
            l_deg[sel], b_deg[sel], lonlat=True,
            s=dot_size, c=color, marker="o",
            edgecolors=edge, linewidths=0.6,
            alpha=alpha, zorder=zorder
        )


def load_post_medians(trace_path):
    idata = az.from_netcdf(trace_path)
    post = idata.posterior
    med = {}
    for k in ["s_birth","beta","h_z_pc","a0","a_mu","a_Av","a_logM","z_sun_pc","log10_tau_p","log10_tau0_Myr"]:
        if k in post:
            med[k] = np.median(post[k].values)  # chain,draw -> scalar
    if "z_sun_pc" not in med:
        med["z_sun_pc"] = 20.8  # pc, safe default
    return med

def infer_dr(G, dtype):
    """
    Return dr as (npix, Kd):
      priority: 'dr' -> 'delta_d_pc' -> np.diff(d_nodes).
    """
    npix = G["npix"]; Kd = G["Kd"]
    # Try direct fields
    for key in ("dr", "delta_d_pc"):
        if key in G:
            dr = np.asarray(G[key], dtype=dtype)
            if dr.ndim == 2 and dr.shape == (npix, Kd):
                return dr
            if dr.ndim == 1:
                if dr.size == Kd:
                    return np.broadcast_to(dr[None, :], (npix, Kd)).astype(dtype)
                if dr.size == npix*Kd:
                    return dr.reshape(npix, Kd)
    # Fall back to radial nodes
    d_nodes = np.asarray(G["d_nodes"], dtype=dtype)  # (Kd,)
    dr1 = np.diff(d_nodes, prepend=d_nodes[:1])
    dr1 = np.where(dr1 <= 0, np.median(dr1[dr1>0]), dr1)
    return np.broadcast_to(dr1[None, :], (npix, Kd)).astype(dtype)

def rho_dust_from_AV(Av, dr, mode="auto"):
    """
    Volumetric dust proxy ρ_dust:
      - If Av looks cumulative along k (monotonic increasing), use finite difference: dA_V/dr.
      - If not, assume Av is already per-pc extinction density (i.e. proportional to ρ).
    Returns array shaped like Av.
    """
    Av = np.asarray(Av)
    # crude monotonicity check along k
    is_cumulative = np.nanmedian((Av[:, -1] - Av[:, 0]) >= 0) > 0.5 and np.nanmedian(np.all(np.diff(Av, axis=1) >= -1e-6, axis=1)) > 0.5
    if mode == "per_pc":
        is_cumulative = False
    if mode == "cumulative":
        is_cumulative = True

    if is_cumulative:
        dAv = np.diff(Av, axis=1, prepend=Av[:, :1])
        rho = dAv / np.maximum(dr, 1e-9)  # mag pc^-2  (∝ dust volume density up to a constant)
    else:
        rho = Av  # already per-pc extinction density
    # clip tiny negatives from numeric noise
    return np.where(rho > 0, rho, 0.0)


def build_obs_count_map(nside, ra_deg, dec_deg, mass, mmin, mmax):
    """Return counts per HEALPix pixel for sources in [mmin,mmax]."""
    # to Galactic
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        sky = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs").galactic
        l_deg = sky.l.deg
        b_deg = sky.b.deg
    except Exception:
        raise SystemExit("[obs] astropy is required to convert RA/Dec to Galactic.")

    sel = (np.isfinite(l_deg) & np.isfinite(b_deg) &
           np.isfinite(mass) & (mass >= mmin) & (mass <= mmax))
    if not np.any(sel):
        return np.zeros(hp.nside2npix(nside), dtype=np.float64)

    pix = hp.ang2pix(nside, l_deg[sel], b_deg[sel], lonlat=True)
    npix = hp.nside2npix(nside)
    counts = np.bincount(pix, minlength=npix).astype(np.float64)
    return counts

def masked_gaussian_smooth(hpmap, fwhm_rad, mask=None):
    """
    Smooth on the sphere with a Gaussian kernel using masked renormalization:
    smooth(data*mask) / smooth(mask).
    """
    if mask is None:
        mask = np.isfinite(hpmap).astype(float)
    else:
        mask = mask.astype(float)
    data = np.nan_to_num(hpmap, nan=0.0, posinf=0.0, neginf=0.0)

    num = hp.smoothing(data * mask, fwhm=fwhm_rad, verbose=False)
    den = hp.smoothing(mask,      fwhm=fwhm_rad, verbose=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        sm = num / (den + 1e-12)
    sm[~np.isfinite(sm)] = np.nan
    return sm


def compute_rho_means_stream(mu, Av, dV, fz, dr, method, dtype, B=256):
    """
    Streaming pass to compute per-pixel mean ⟨ρ⟩_i with chosen weighting.
      method: 'plain'  → mean over k
              'volume' → Σ ρ dV / Σ dV
              'fz'     → Σ ρ (fz dV) / Σ (fz dV)
    Returns rho_mean: (npix,)
    """
    npix, Kd = Av.shape
    num = np.zeros(npix, dtype=dtype)
    den = np.zeros(npix, dtype=dtype)

    for i0 in range(0, npix, B):
        i1   = min(npix, i0 + B)
        Av_b = Av[i0:i1]          # (B,Kd)
        dV_b = dV[i0:i1]
        fz_b = fz[i0:i1]
        dr_b = dr[i0:i1]

        rho_b = rho_dust_from_AV(Av_b, dr_b)  # (B,Kd)

        if method == "plain":
            num[i0:i1] = np.sum(rho_b, axis=1)
            den[i0:i1] = rho_b.shape[1]
        elif method == "volume":
            w = dV_b
            num[i0:i1] = np.sum(rho_b * w, axis=1)
            den[i0:i1] = np.sum(w, axis=1)
        else:  # 'fz'
            w = fz_b * dV_b
            num[i0:i1] = np.sum(rho_b * w, axis=1)
            den[i0:i1] = np.sum(w, axis=1)

    rho_mean = num / np.maximum(den, 1e-30)
    rho_mean = np.where(np.isfinite(rho_mean), rho_mean, 0.0)
    return rho_mean


def tau_years(M, med, M_pivot=2.5):
    M = np.asarray(M, float)
    if "log10_tau_p" in med:
        return 10.0**(med["log10_tau_p"] + med["beta"]*(np.log10(M) - np.log10(M_pivot)))
    elif "log10_tau0_Myr" in med:
        return 10.0**(med["log10_tau0_Myr"] + 6.0 + med["beta"]*np.log10(M))
    else:
        raise KeyError("Posterior must contain either 'log10_tau_p' or 'log10_tau0_Myr' (plus 'beta').")

def load_pack(path, dtype=np.float32):
    P = np.load(path, allow_pickle=True)
    keys = set(P.files)

    # Must have these (your pack’s names)
    for k in ["nside","npix","mu","A_V","z_pc","dV_pc3","Sigma_SFR_Msun_yr_kpc2","d_pc"]:
        if k not in keys:
            raise SystemExit(f"[pack] missing '{k}' in {path}. Keys: {list(P.keys())}")

    nside = int(P["nside"])
    npix  = int(P["npix"])

    # --- Infer Kd from mu (robust even when flattened) ---
    mu_raw = np.asarray(P["mu"], dtype=dtype)
    if mu_raw.ndim == 2 and mu_raw.shape[0] == npix:
        Kd = int(mu_raw.shape[1])
        mu = mu_raw
    elif mu_raw.ndim == 1:
        if mu_raw.size % npix != 0:
            raise ValueError(f"[pack] mu.size={mu_raw.size} not divisible by npix={npix}")
        Kd = int(mu_raw.size // npix)
        mu = mu_raw.reshape(npix, Kd)
    else:
        raise ValueError(f"[pack] mu has unexpected shape {mu_raw.shape}")

    G = npix * Kd  # total voxels

    def as_2d(name):
        """Return array shaped (npix, Kd) from flattened or 2D."""
        arr = np.asarray(P[name], dtype=dtype)
        if arr.ndim == 2:
            if arr.shape == (npix, Kd):
                return arr
            if arr.size == G:
                return arr.reshape(npix, Kd)
            raise ValueError(f"[pack] {name} has 2D shape {arr.shape}, expected ({npix},{Kd})")
        if arr.ndim == 1:
            if arr.size == G:
                return arr.reshape(npix, Kd)
            raise ValueError(f"[pack] {name} has 1D size {arr.size}, expected {G}")
        raise ValueError(f"[pack] {name} has ndim={arr.ndim}, expected 1 or 2")

    # Per-voxel arrays
    Av   = as_2d("A_V")
    z_pc = as_2d("z_pc")
    dV   = as_2d("dV_pc3")

    # Radial nodes: d_pc may be (Kd,), (G,), or (npix,Kd)
    dpc_raw = np.asarray(P["d_pc"], dtype=dtype)
    if dpc_raw.ndim == 1 and dpc_raw.size == Kd:
        d_nodes = dpc_raw
    elif dpc_raw.ndim == 1 and dpc_raw.size == G:
        d_nodes = dpc_raw.reshape(npix, Kd)[0, :]
    elif dpc_raw.ndim == 2 and dpc_raw.shape == (npix, Kd):
        d_nodes = dpc_raw[0, :]
    else:
        raise ValueError(f"[pack] d_pc has shape/size {dpc_raw.shape}, cannot recover nodes")

    # Σ_SFR: allow per-pixel, per-voxel, or flattened
    SFR_raw = np.asarray(P["Sigma_SFR_Msun_yr_kpc2"], dtype=dtype)
    if SFR_raw.ndim == 2 and SFR_raw.shape == (npix, Kd):
        Sigma_SFR = SFR_raw
        sfr_mode = "per_voxel"
    elif SFR_raw.ndim == 1 and SFR_raw.size == G:
        Sigma_SFR = SFR_raw.reshape(npix, Kd)
        sfr_mode = "per_voxel"
    elif SFR_raw.ndim == 1 and SFR_raw.size == npix:
        Sigma_SFR = SFR_raw  # keep 1D; broadcast later
        sfr_mode = "per_pixel"
    else:
        raise ValueError(f"[pack] Sigma_SFR shape/size {SFR_raw.shape} unexpected")

    footprint = (np.asarray(P["footprint"], bool)
                 if "footprint" in keys else np.ones(npix, bool))
    ell = np.asarray(P["ell"], dtype=dtype) if "ell" in keys else None
    bee = np.asarray(P["bee"], dtype=dtype) if "bee" in keys else None

    print(f"[pack] nside={nside} npix={npix} Kd={Kd} (G={G})  mu.shape={mu.shape}")
    if isinstance(Sigma_SFR, np.ndarray) and Sigma_SFR.ndim == 1:
        print(f"[pack] Sigma_SFR per-pixel: shape=({Sigma_SFR.shape[0]},)")
    else:
        print(f"[pack] Sigma_SFR per-voxel: shape={Sigma_SFR.shape}")

    return dict(
        nside=nside, npix=npix, Kd=Kd, d_nodes=d_nodes,
        mu=mu, Av=Av, z=z_pc, dV=dV,
        Sigma_SFR=Sigma_SFR, sfr_mode=sfr_mode,
        footprint=footprint, ell=ell, bee=bee
    )


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--grid_pack", required=True, help="NPZ with HEALPix voxel grid (fields listed in file header)")
    ap.add_argument("--trace",     required=True, help="ArviZ netcdf with posterior")
    ap.add_argument("--catalog",   required=True, help="TSV/CSV with at least ra, dec, mass column")
    ap.add_argument("--col_ra",    default="ra")
    ap.add_argument("--col_dec",   default="dec")
    ap.add_argument("--col_mass",  default="Mstar")

    # selection / physics knobs
    ap.add_argument("--mu_ref",    type=float, default=10.0)
    ap.add_argument("--k_lambda",  type=float, default=1.0)
    ap.add_argument("--Mmin",      type=float, default=1.5)
    ap.add_argument("--Mmax",      type=float, default=8.0)
    ap.add_argument("--Mq",        type=int,   default=48)
    ap.add_argument("--M_pivot",   type=float, default=2.5)

    # plotting
    ap.add_argument("--out",       default="expected_herbig_healpix.png")
    ap.add_argument("--title",     default="Expected observed Herbigs per pixel")
    ap.add_argument("--cmap",      default="viridis")
    ap.add_argument("--vmax_pct",  type=float, default=99.5)
    ap.add_argument("--dot_size",  type=float, default=10.0)

    ap.add_argument("--dtype", choices=["float32","float64"], default="float32",
                help="Computation dtype for big arrays")
    ap.add_argument("--chunk_pix", type=int, default=256,
                    help="Number of pixels processed per block")
    ap.add_argument("--per_sr", action="store_true",
                help="Output expected number per steradian instead of per pixel")
    ap.add_argument("--per_deg2", action="store_true",
                help="Output expected number per square degree instead of per pixel (implies per_sr)")

    ap.add_argument("--use_dust_weight", action="store_true",
                help="Down/up-weight voxel contributions by local dust density normalized by the pixel-mean.")
    ap.add_argument("--dust_mean", choices=["plain", "volume", "fz"], default="fz",
                    help="How to compute the per-pixel mean dust density: "
                        "'plain' (simple average over k), 'volume' (∝ dV), or 'fz' (∝ fz·dV).")
    ap.add_argument("--dust_gamma", type=float, default=1.0,
                    help="Exponent γ for (rho / mean)^γ.")
    ap.add_argument("--dust_eps", type=float, default=1e-6,
                    help="Small floor added to rho and mean to avoid 0/0.")
    ap.add_argument("--dust_clip", type=float, default=5.0,
                    help="Symmetric clip for the normalized weight; "
                        "w = clip( (rho/mean)^γ , 1/clip, clip ).")

    ap.add_argument("--make_obs_map", action="store_true",
                help="Also make an observed density map from the catalog.")
    ap.add_argument("--smooth_fwhm_deg", type=float, default=6.0,
                    help="Gaussian FWHM (deg) for spherical smoothing of the observed map.")
    ap.add_argument("--side_by_side", action="store_true",
                    help="Plot expected and observed maps in one figure (two panels).")
    args = ap.parse_args()

    # Load grid & posterior

    print("[info] loading grid pack and posterior medians...")
    med = load_post_medians(args.trace)
    DT = np.float32 if args.dtype == "float32" else np.float64
    G   = load_pack(args.grid_pack, dtype=DT)


    print(f"[info] grid has nside={G['nside']}, npix={G['npix']}")
    print(f"[info] using dtype={DT.__name__} for computations")

    mu   = np.asarray(G["mu"],   dtype=DT)
    Av   = np.asarray(G["Av"],   dtype=DT)
    z_pc = np.asarray(G["z"],    dtype=DT)
    dV   = np.asarray(G["dV"],   dtype=DT)
    SFR  = np.asarray(G["Sigma_SFR"], dtype=DT)
    print(f"[info] median SFR: {np.nanmedian(SFR):.3e} Msun/yr/kpc^2   ")
    if SFR.ndim == 1:
        SFR = SFR[:, None] * np.ones_like(mu, dtype=DT)
    #mask_pix = G["footprint"]
    npix, Kd = mu.shape

    z_mid = z_pc + DT(med["z_sun_pc"])
    hz    = DT(med["h_z_pc"])
    fz    = np.exp(-0.5*(z_mid/hz)**2, dtype=DT) / (DT(np.sqrt(2.0*np.pi)) * hz)  # (npix,Kd)

    # mass quadrature
    print("[info] preparing mass quadrature...")
    Mgrid    = np.logspace(np.log10(args.Mmin), np.log10(args.Mmax), args.Mq).astype(DT)
    logM_vec = np.log10(Mgrid).astype(DT)
    Mw       = xi_norm_on_interval(Mgrid, Mmin=0.08, Mmax=120.0)

    #Mgrid_all = np.logspace(np.log10(0.08), np.log10(120.0), 512)
    #Mw_all       = xi_norm_on_interval(Mgrid_all, Mmin=0.08, Mmax=120.0)

    tau_M_vec= tau_years(Mgrid, med, M_pivot=args.M_pivot).astype(DT)
    w_tau_vec= (Mw * tau_M_vec).astype(DT)

    '''plt.plot(Mgrid, w_tau_vec, marker='o')
    plt.plot(Mgrid, Mw, marker='x', label='IMF only')
    plt.plot(Mgrid, tau_M_vec/1e6, marker='s', label='tau(M) [Myr]')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Mass [Msun]")
    plt.ylabel("Weights")
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.show()'''
    
    a0    = DT(med["a0"]); a_mu = DT(med["a_mu"])
    a_Av  = DT(med["a_Av"]); a_logM = DT(med["a_logM"])
    mu_ref = DT(args.mu_ref); kL = DT(args.k_lambda)
    s_birth = DT(med["s_birth"])                   


    # Precompute fixed factors
    z_mid = z_pc + med["z_sun_pc"]                      # (npix,Kd)
    fz = np.exp(-0.5*(z_mid/med["h_z_pc"])**2) / (np.sqrt(2.0*np.pi)*med["h_z_pc"])  # (npix,Kd)


    # --- voxel geometry & dust proxy prep ---
    dr = infer_dr(G, DT)  # (npix, Kd)

    rho_mean = None
    if args.use_dust_weight:
        rho_mean = compute_rho_means_stream(mu, Av, dV, fz, dr,
                                            method=args.dust_mean, dtype=DT,
                                            B=int(args.chunk_pix))


    print("[info] integrating over mass grid and pixels...")

    npix, Kd = mu.shape
    expected = np.zeros(npix, dtype=DT)
    B = int(args.chunk_pix)



    # Pixel solid angle (sr) — from pack if available, else from HEALPix geometry
    omega_sr = None
    if "delta_omega_sr" in G:
        om = np.asarray(G["delta_omega_sr"], dtype=DT)
        if om.ndim == 0:
            omega_sr = np.full(npix, float(om), dtype=DT)
        elif om.ndim == 1 and om.size == npix:
            omega_sr = om.astype(DT)
        elif om.ndim == 1 and om.size == npix * Kd:
            # stored per-voxel; take the pixel value (they're constant over k)
            omega_sr = om.reshape(npix, Kd)[:, 0].astype(DT)
        elif om.ndim == 2 and om.shape == (npix, Kd):
            omega_sr = om[:, 0].astype(DT)
    # Fallback to HEALPix equal-area size if missing / odd
    if omega_sr is None:
        import healpy as hp
        omega_val = hp.nside2pixarea(G["nside"])  # sr per pixel
        omega_sr  = np.full(npix, omega_val, dtype=DT)

    
    '''def logistic32(x):
        # numerically safe, avoids huge temporaries
        return DT(1.0) / (DT(1.0) + np.exp(-x, dtype=DT))'''

    for i0 in range(0, npix, B):
        i1   = min(npix, i0 + B)
        mu_b = mu[i0:i1]          # (B,Kd)
        Av_b = Av[i0:i1]          # (B,Kd)

        # base term independent of M
        base = a0 + a_mu * (mu_b - mu_ref) + a_Av * (kL * Av_b)   # (B, Kd), DT
        
        eff  = np.zeros_like(base, dtype=DT)                       # (B, Kd)

        for m in range(logM_vec.size):
            lm = logM_vec[m].item()      # Python float
            wt = w_tau_vec[m].item()     # Python float

            eta_m = base + a_logM * lm   # (B, Kd)

            # stable logistic to avoid overflow
            p = np.where(
                eta_m >= 0,
                1.0 / (1.0 + np.exp(-eta_m)),
                np.exp(eta_m) / (1.0 + np.exp(eta_m)),
            ).astype(DT)

            eff += wt * p                 # (B, Kd)
        
        if G["sfr_mode"] == "per_pixel":
            sfr_blk = G["Sigma_SFR"][i0:i1][:, None]        # (B,1) → broadcasts to (B,Kd)
        else:
            sfr_blk = G["Sigma_SFR"][i0:i1]                 # (B,Kd)

        if args.use_dust_weight:
            rho_blk = rho_dust_from_AV(Av_b, dr[i0:i1])  # (B,Kd)
            # normalized weight with exponent and symmetric clip
            mean_blk = rho_mean[i0:i1][:, None]          # (B,1)
            w_d = np.power((rho_blk + args.dust_eps) / (mean_blk + args.dust_eps),
                        args.dust_gamma).astype(DT)
            if args.dust_clip and args.dust_clip > 0:
                c = DT(args.dust_clip)
                w_d = np.clip(w_d, 1.0/c, c)
        else:
            w_d = 1.0

        dN_blk = 1e-6 * s_birth * sfr_blk * fz[i0:i1] * eff * dV[i0:i1] * w_d
        '''plt.plot(mu_b[0,:], dN_blk[0,:], label=f"Pixel {i0} dN_blk" )
        plt.plot(mu_b[0,:], s_birth*sfr_blk[0,:], label=f"Pixel {i0} sfr_blk" )
        plt.plot(mu_b[0,:], fz[i0:i1][0,:], label=f"Pixel {i0} fz" )
        plt.plot(mu_b[0,:], eff[0,:], label=f"Pixel {i0} eff" )
        plt.plot(mu_b[0,:], dV[i0:i1][0,:], label=f"Pixel {i0} dV" )
        plt.xlabel("mu")
        plt.ylabel("dN_blk components")
        plt.legend()
        plt.yscale('log')
        plt.show()'''

        if "footprint" in G:
            fp = G["footprint"]
            # ensure (npix, Kd) for block
            if fp.ndim == 1 and fp.size == G["npix"] * G["Kd"]:
                fp_blk = fp.reshape(G["npix"], G["Kd"])[i0:i1]       # (B, Kd)
            elif fp.ndim == 2 and fp.shape == (G["npix"], G["Kd"]):
                fp_blk = fp[i0:i1]
            elif fp.ndim == 1 and fp.size == G["npix"]:
                # per-pixel mask—broadcast across Kd
                fp_blk = fp[i0:i1, None]
            else:
                fp_blk = None

            if fp_blk is not None:
                dN_blk = np.where(fp_blk, dN_blk, 0.0)
        expected[i0:i1] = np.nansum(dN_blk, axis=1)

        # free large temporaries early
        del mu_b, Av_b, base, eff, dN_blk

    '''for i0 in range(0, npix, chunk):
        i1 = min(npix, i0+chunk)
        mu_blk = mu[i0:i1][:, :, None]      # (B,Kd,1)
        Av_blk = Av[i0:i1][:, :, None]      # (B,Kd,1)

        eta = (med["a0"]
               + med["a_mu"] * (mu_blk - args.mu_ref)
               + med["a_Av"] * (args.k_lambda * Av_blk)
               + med["a_logM"] * logM)                      # (B,Kd,Mq)
        pdet = logistic(eta)                                 # (B,Kd,Mq)
        eff  = (Mw * tau_M * pdet).sum(axis=2)               # (B,Kd)

        dN_blk = med["s_birth"] * SFR[i0:i1] * fz[i0:i1] * eff * dV[i0:i1]  # (B,Kd)
        expected[i0:i1] = np.nansum(dN_blk, axis=1)'''

    unit_label = "$E[N]$ pix$^{-1}$"
    if args.per_sr or args.per_deg2:
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = expected / omega_sr
        unit_label = "$E[N]$ sr$^{-1}$"
        if args.per_deg2:
            DEG2_PER_SR = (180.0/np.pi)**2  # ≈ 3282.80635
            expected = expected / DEG2_PER_SR
            unit_label = r"$E[N]$ deg$^{-2}$"

    # Optional: blank out pixels with zero/invalid solid angle
    bad = ~np.isfinite(omega_sr) | (omega_sr <= 0)
    expected[bad] = np.nan
    # Zero out outside footprint, if provided

    obs_counts = None
    if args.make_obs_map:
        # Read catalog
        try:
            import pandas as pd
            sep = "\t" if args.catalog.lower().endswith(".tsv") else ","
            df = pd.read_csv(args.catalog, sep=sep)
            ra  = df[args.col_ra].to_numpy(float)
            dec = df[args.col_dec].to_numpy(float)
            M   = df[args.col_mass].to_numpy(float)
        except Exception:
            data = np.genfromtxt(args.catalog, names=True, dtype=None, encoding=None)
            ra  = np.asarray(data[args.col_ra], float)
            dec = np.asarray(data[args.col_dec], float)
            M   = np.asarray(data[args.col_mass], float)

        obs_counts = build_obs_count_map(G["nside"], ra, dec, M, args.Mmin, args.Mmax)  # counts/pixel

        # convert to desired surface density BEFORE smoothing
        if getattr(args, "per_sr", False) or getattr(args, "per_deg2", False):
            obs_field = obs_counts / omega_sr  # counts / sr
            obs_unit  = r"counts sr$^{-1}$"
            if getattr(args, "per_deg2", False):
                obs_field = obs_field / DEG2_PER_SR
                obs_unit  = r"counts deg$^{-2}$"
        else:
            obs_field = obs_counts.astype(float)
            obs_unit  = "counts/pixel"

        # make an optional mask (e.g., pack footprint) to keep smoothing honest near edges
        fp = G.get("footprint", None)
        if fp is not None:
            if fp.ndim == 2 and fp.shape == (npix, Kd):
                mask_pix = fp.any(axis=1)
            elif fp.ndim == 1 and fp.size == npix*Kd:
                mask_pix = fp.reshape(npix, Kd).any(axis=1)
            elif fp.ndim == 1 and fp.size == npix:
                mask_pix = fp.astype(bool)
            else:
                mask_pix = np.ones(npix, bool)
        else:
            mask_pix = np.ones(npix, bool)

        fwhm_rad = np.deg2rad(args.smooth_fwhm_deg)
        obs_smoothed = masked_gaussian_smooth(obs_field, fwhm_rad, mask=mask_pix)

    # --- catalog to Galactic + selection mask for mass bin ---
    ra, dec, M = load_catalog_icrs(args.catalog, args.col_ra, args.col_dec, args.col_mass)
    l_pts, b_pts = catalog_galactic(ra, dec)
    sel_pts = (M >= args.Mmin) & (M <= args.Mmax) & np.isfinite(l_pts) & np.isfinite(b_pts)


    if args.side_by_side and args.make_obs_map:
        fig = plt.figure(figsize=(14, 6.2))
        vmax1 = float(int(np.nanpercentile(expected, args.vmax_pct)+0.5))
        vmax2 =  float(int(np.nanpercentile(obs_smoothed, args.vmax_pct)+0.5))

        hp1 = hp.mollview(expected, title=f"Expected  [{args.Mmin:.2g}–{args.Mmax:.2g} $M_\\odot$]",
                    unit=unit_label, cmap=args.cmap, min=0.0, max=vmax1, cbar=True, coord="G",
                    fig=fig.number, sub=(1,2,1))
        overlay_points(l_pts, b_pts, sel_pts, dot_size=args.dot_size)
        hp2 = hp.mollview(obs_smoothed, title=f"Observed (smoothed {args.smooth_fwhm_deg:.1f}°)",
                    unit=obs_unit, cmap=args.cmap, min=0.0, max=vmax1, cbar=True, coord="G",
                    fig=fig.number, sub=(1,2,2))
        overlay_points(l_pts, b_pts, sel_pts, dot_size=args.dot_size)
        
        plt.savefig(args.out, dpi=220, bbox_inches="tight")
        print(f"[OK] saved {os.path.abspath(args.out)}")
    else:
        # expected (as before)
        vmax =  float(int(np.nanpercentile(expected, args.vmax_pct)+0.5))
        hp.mollview(expected,
            title=f"{args.title}  [{args.Mmin:.2g}–{args.Mmax:.2g} $M_\\odot$]",
            unit=unit_label, cmap=args.cmap, min=0.0, max=vmax, cbar=True, coord="G"
        )
        overlay_points(l_pts, b_pts, sel_pts, dot_size=args.dot_size)
        
        plt.savefig(args.out, dpi=220, bbox_inches="tight")
        print(f"[OK] saved {os.path.abspath(args.out)}")

        # optional separate observed map
        if args.make_obs_map:
            out2 = os.path.splitext(args.out)[0] + "_observed.png"
            vmax2 = np.nanpercentile(obs_smoothed, args.vmax_pct)
            plt.figure()
            hp.mollview(obs_smoothed, title=f"Observed (smoothed {args.smooth_fwhm_deg:.1f}°)",
                        unit=obs_unit, cmap=args.cmap, min=0.0, max=vmax2, cbar=True, coord="G")
            overlay_points(l_pts, b_pts, sel_pts, dot_size=args.dot_size)
            plt.savefig(out2, dpi=220, bbox_inches="tight")
            print(f"[OK] saved {os.path.abspath(out2)}")
    
    '''# Plot Mollweide map (Galactic coords)
    vmax = float(int(np.nanpercentile(expected, args.vmax_pct)+0.5))
    print(vmax)
    hp.mollview(
        expected, title=f"{args.title}  [{args.Mmin:.2g}–{args.Mmax:.2g} $M_\\odot$]",
        unit=unit_label, cmap=args.cmap, min=0.0, max=vmax, cbar=True, coord="G"
    )

    # Overlay real stars in same mass bin
    # Load catalog robustly (pandas or NumPy)
    try:
        import pandas as pd  # optional
        sep = "\t" if args.catalog.lower().endswith(".tsv") else ","
        df = pd.read_csv(args.catalog, sep=sep)
        ra  = df[args.col_ra].to_numpy(float)
        dec = df[args.col_dec].to_numpy(float)
        M   = df[args.col_mass].to_numpy(float)
    except Exception:
        data = np.genfromtxt(args.catalog, names=True, dtype=None, encoding=None)
        ra  = np.asarray(data[args.col_ra], float)
        dec = np.asarray(data[args.col_dec], float)
        M   = np.asarray(data[args.col_mass], float)

    if HAVE_ASTROPY:
        sky = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs").galactic
        l_pts = sky.l.deg; b_pts = sky.b.deg
        sel = (M >= args.Mmin) & (M <= args.Mmax) & np.isfinite(l_pts) & np.isfinite(b_pts)
        if np.any(sel):
            hp.projscatter(
                l_pts[sel], b_pts[sel], lonlat=True,
                s=args.dot_size, c="w", marker="o", edgecolors="k", linewidths=0.6, alpha=0.9
            )
    else:
        print("[warn] astropy not available; skipping star overlay.", file=sys.stderr)

    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"[OK] saved {os.path.abspath(args.out)}")'''

    # ---- build observed count map from the catalog ----

if __name__ == "__main__":
    main()
