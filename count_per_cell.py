#!/usr/bin/env python3
# count_per_cell_3d.py
"""
Count Herbig stars per (HEALPix pixel i, radial shell k) cell using the same grid
as in the likelihood (from ppp_grid_pack.npz). Print cells with N>=minN (default 2).

Assumes the pack contains:
  - nside, npix
  - mu (npix, Kd) or d_pc (npix, Kd)  [we'll use d_pc if present for distances]
  - dr or delta_d_pc (optional; improves shell edges)
  - ell, bee (optional; Galactic lon/lat per pixel, deg)
"""
import numpy as np
from utils import *
import numpy as np
import argparse, pandas as pd, healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd

def read_catalog(path, col_ra, col_dec, col_dist, maybe_name="Name"):
    try:
        import pandas as pd
        sep = "\t" if path.lower().endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep)
        ra  = df[col_ra].to_numpy(float)
        dec = df[col_dec].to_numpy(float)
        d   = df[col_dist].to_numpy(float)
        names = df[maybe_name].astype(str).to_numpy() if maybe_name in df.columns else None
    except Exception:
        data = np.genfromtxt(path, names=True, dtype=None, encoding=None)
        ra  = np.asarray(data[col_ra], float)
        dec = np.asarray(data[col_dec], float)
        d   = np.asarray(data[col_dist], float)
        names = np.asarray(data[maybe_name], str) if maybe_name in data.dtype.names else None
    return ra, dec, d, names

def icrs_to_gal(ra_deg, dec_deg):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sky = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs").galactic
    return sky.l.deg, sky.b.deg

def get_dr_row(dr, i, npix, K):
    """
    Normalize any dr to a row for pixel i:
    - None → None (midpoint edges)
    - scalar → scalar
    - (K,) → that row for all pixels
    - (npix,K) or flattened (npix*K,) → row i
    - (1,) → scalar
    """
    if dr is None:
        return None
    arr = np.asarray(dr)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        if arr.size == K:
            return arr
        if arr.size == npix*K:
            return arr.reshape(npix, K)[i]
        if arr.size == 1:
            return float(arr[0])
        return None
    if arr.ndim == 2 and arr.shape == (npix, K):
        return arr[i]
    return None



def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_pack", required=True)
    ap.add_argument("--catalog",   required=True)
    ap.add_argument("--col_ra",    default="ra")
    ap.add_argument("--col_dec",   default="dec")
    ap.add_argument("--col_dist",  default="distance")
    ap.add_argument("--minN",      type=int, default=2)
    ap.add_argument("--max_print", type=int, default=200)
    ap.add_argument("--use_names", action="store_true")
    ap.add_argument("--save_tsv",  default="")
    args = ap.parse_args()

    G = load_grid_layout(args.grid_pack)

    # Load catalog
    df = pd.read_csv(args.catalog, sep="\t" if args.catalog.endswith(".tsv") else ",")
    ra  = df[args.col_ra].to_numpy(float)
    dec = df[args.col_dec].to_numpy(float)
    dist= df[args.col_dist].to_numpy(float)
    names = df["Name"].to_numpy(str) if args.use_names and "Name" in df.columns else None

    # Map stars to HEALPix pixel (Galactic)
    sky = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs").galactic
    l_deg = sky.l.deg; b_deg = sky.b.deg

    # Need NSIDE; try to infer from npix in pack
    if G["ragged"]:
        npix = G["npix"]
    else:
        npix = G["npix"]
    # infer NSIDE from npix
    import healpy as hp
    nside = hp.npix2nside(npix)
    pix = hp.ang2pix(nside, l_deg, b_deg, lonlat=True)

    # Allocate counts per cell
    if G["ragged"]:
        counts = np.zeros(G["G"], dtype=np.int32)
    else:
        counts = np.zeros(G["npix"] * G["Kd"], dtype=np.int32)

    # Optional name storage
    cell_names = {} if args.use_names else None

    # Count
    if G["ragged"]:
        row_ptr = G["row_ptr"]; d_lo = G["d_lo"]; d_hi = G["d_hi"]
        for j in range(dist.size):
            p = int(pix[j])
            if p < 0 or p >= G["npix"] or not np.isfinite(dist[j]): 
                continue
            s0, s1 = row_ptr[p], row_ptr[p+1]
            if s1 <= s0: 
                continue
            lo = d_lo[s0:s1]; hi = d_hi[s0:s1]
            k = np.where((dist[j] >= lo) & (dist[j] < hi))[0]
            if k.size:
                idx = s0 + int(k[0])
                counts[idx] += 1
                if cell_names is not None:
                    cell_names.setdefault(idx, []).append(names[j] if names is not None else f"star{j}")
    else:
        d_pc = G["d_pc"]; Kd = G["Kd"]; npix = G["npix"]
        for j in range(dist.size):
            p = int(pix[j])
            if p < 0 or p >= npix or not np.isfinite(dist[j]): 
                continue
            dc = d_pc[p]  # (Kd,)
            dr_row = get_dr_row(G["dr"], p, npix, Kd)
            edges = edges_from_centers(dc, dr_row)
            k = np.searchsorted(edges, dist[j], side="right") - 1
            if 0 <= k < Kd:
                idx = p*Kd + int(k)
                counts[idx] += 1
                if cell_names is not None:
                    cell_names.setdefault(idx, []).append(names[j] if names is not None else f"star{j}")

    # Summaries
    binc = np.bincount(counts)
    for n, c in enumerate(binc):
        if c:
            print(f"N={n}: {c} cells")

    # Print cells with >= minN
    printed = 0
    if G["ragged"]:
        row_ptr = G["row_ptr"]; d_lo = G["d_lo"]; d_hi = G["d_hi"]
        for idx in np.where(counts >= args.minN)[0]:
            # find pixel by row_ptr
            p = int(np.searchsorted(row_ptr, idx, side="right") - 1)
            k = int(idx - row_ptr[p])
            print(f"pix={p:5d}  cell={k:3d}  d=[{d_lo[idx]:.1f},{d_hi[idx]:.1f}] pc  N={counts[idx]}")
            if cell_names and printed < args.max_print:
                print("   names:", ", ".join(cell_names.get(idx, [])))
            printed += 1
            if printed >= args.max_print:
                break
    else:
        d_pc = G["d_pc"]; Kd = G["Kd"]
        for idx in np.where(counts >= args.minN)[0]:
            p = idx // Kd; k = idx % Kd
            dc = d_pc[p]
            dr_row = get_dr_row(G["dr"], p, G["npix"], Kd)
            edges = edges_from_centers(dc, dr_row)
            print(f"pix={p:5d}  cell={k:3d}  d=[{edges[k]:.1f},{edges[k+1]:.1f}] pc  N={counts[idx]}")
            if cell_names and printed < args.max_print:
                print("   names:", ", ".join(cell_names.get(idx, [])))
            printed += 1
            if printed >= args.max_print:
                break

    # Optional TSV
    if args.save_tsv:
        import csv
        rows = []
        if G["ragged"]:
            row_ptr = G["row_ptr"]
            for idx, N in enumerate(counts):
                p = int(np.searchsorted(row_ptr, idx, side="right") - 1)
                k = int(idx - row_ptr[p])
                rows.append(dict(pixel=p, cell=k, d_lo=G["d_lo"][idx], d_hi=G["d_hi"][idx], N=N))
        else:
            d_pc = G["d_pc"]; Kd = G["Kd"]
            for idx, N in enumerate(counts):
                p = idx // Kd; k = idx % Kd
                edges = edges_from_centers(d_pc[p], get_dr_row(G["dr"], p, G["npix"], Kd))
                rows.append(dict(pixel=p, cell=k, d_lo=edges[k], d_hi=edges[k+1], N=N))
        pd.DataFrame(rows).to_csv(args.save_tsv, index=False)
        print(f"[OK] wrote {args.save_tsv}")


if __name__ == "__main__":
    main()
