#!/usr/bin/env python3
# plot_ppp_corner.py

import argparse, re, os, importlib
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl

# ------------- existing helpers (unchanged, trimmed) ----------------

def load_draws(trace_path):
    idata = az.from_netcdf(trace_path)
    post = idata.posterior
    return idata, post

def flatten_scalar_params(post, allow_names=None, exclude_regex=r".*__.*"):
    scalars = {}
    patt = re.compile(exclude_regex) if exclude_regex else None
    for name in post.data_vars:
        if allow_names is not None and name not in allow_names:
            continue
        if patt is not None and patt.match(name):
            continue
        arr = post[name].values  # (chain, draw[, dims...])
        if arr.ndim == 2:
            scalars[name] = arr.reshape(-1)
    return scalars

def have_vars(post, names):
    return [nm for nm in names if nm in post.data_vars]

def compute_tau_derivatives(draws, M_vals=(1.5, 2.5, 8.0), M_pivot=2.5):
    out = {}
    have_pivot = ("log10_tau_p" in draws) and ("beta" in draws)
    have_classic = ("log10_tau0_Myr" in draws) and ("beta" in draws)
    if not (have_pivot or have_classic):
        return out
    beta = draws["beta"]
    for M in M_vals:
        if have_pivot:
            val = draws["log10_tau_p"] + beta * (np.log10(M) - np.log10(M_pivot))
        else:
            val = draws["log10_tau0_Myr"] + 6.0 + beta * np.log10(M)
        out[f"log10_tau@{M:.1f}Msun"] = val
    return out

def stack_for_corner(draws_dict, order=None):
    if order is None:
        order = list(draws_dict.keys())
    X = np.column_stack([draws_dict[k] for k in order])
    labels = order
    return X, labels

def thin_rows(X, nth=5, max_draws=2000, seed=123):
    X2 = X[::max(1, nth)]
    if X2.shape[0] > max_draws:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X2.shape[0], size=max_draws, replace=False)
        X2 = X2[idx]
    return X2

# --------------------- NEW helpers for labels/colorbar ---------------------

def load_labels_module(module_name):
    """Import module and return LATEX_LABELS dict or empty dict."""
    if not module_name:
        return {}
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "LATEX_LABELS", {})
    except Exception:
        return {}

def to_tex(label_str):
    """Wrap a LaTeX label in $...$ if not already."""
    if label_str.startswith("$") and label_str.endswith("$"):
        return label_str
    return f"${label_str}$"

def build_level_colorbar(fig, levels, cmap_name="Blues", label="Credible level", fontsize=60.0, ticksize=40.0):
    """
    Add a discrete colorbar that explains the filled contour levels.
    This is an indicative legend: colors match the palette used, not exact contour mappables.
    """
    levels = np.asarray(levels, float)
    # Discrete colors: darkest for most confident (e.g., 68%), lighter for larger areas
    base_cmap = plt.get_cmap(cmap_name)
    n = len(levels) + 1  # n bands: [0, lvl1], [lvl1,lvl2], ..., [lvlK,1]
    # pick shades from dark to light
    shades = np.linspace(0.8, 0.35, n)  # tweak for aesthetics
    colors = [base_cmap(s) for s in shades]
    cmap = ListedColormap(colors)
    bounds = np.concatenate(([0.0], levels, [1.0]))
    norm = BoundaryNorm(bounds, ncolors=cmap.N)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Place a single shared colorbar to the right
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.03, pad=0.02)
    # Put ticks exactly at the levels
    cbar.set_ticks(levels)
    cbar.set_ticklabels([f"{int(l*100)}%" for l in levels])
    cbar.set_label(label)
    cbar.ax.yaxis.label.set_size(fontsize)
    cbar.ax.tick_params(labelsize=fontsize, size=ticksize)

    return cbar

# ------------------------------- main --------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--vars", default="", help="Comma-separated var names; if empty, auto core set")
    ap.add_argument("--exclude_regex", default=r".*__.*")
    ap.add_argument("--add_tau_derived", action="store_true", default=False)
    ap.add_argument("--M_pivot", type=float, default=2.5)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--max_draws", type=int, default=2000)
    ap.add_argument("--out", default="corner_ppp.png")
    ap.add_argument("--out_pdf", default="")
    ap.add_argument("--use_corner", action="store_true")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.68, 0.95], help="credible levels")
    ap.add_argument("--cmap", default="Blues", help="colormap name for filled contours")
    ap.add_argument("--fontsize", type=float, default=12.0, help="label/title fontsize")
    ap.add_argument("--ticksize", type=float, default=12.0, help="tick label size")
    ap.add_argument("--labels_module", default="ppp_labels", help="Python module exposing LATEX_LABELS dict")
    ap.add_argument("--usetex", action="store_true", help="Use LaTeX engine (requires system LaTeX); default off")
    args = ap.parse_args()

    if args.usetex:
        plt.rcParams["text.usetex"] = True

    # Load labels map
    label_map = load_labels_module(args.labels_module)

    idata, post = load_draws(args.trace)

    # Choose variables
    user_vars = [v.strip() for v in args.vars.split(",") if v.strip()] if args.vars else []
    if user_vars:
        present = have_vars(post, user_vars)
        var_list = present
        missing = [v for v in user_vars if v not in present]
        if missing:
            print("[warn] Missing variables:", ", ".join(missing))
    else:
        core = ["log10_tau_p", "log10_tau0_Myr", "beta", "h_z_pc",
                "a0", "a_mu", "a_Av", "a_logM", "s_birth"]
        var_list = have_vars(post, core)
        if not var_list:
            var_list = [nm for nm, da in post.data_vars.items() if da.values.ndim == 2]

    draws = flatten_scalar_params(post, allow_names=set(var_list), exclude_regex=args.exclude_regex)

    # Add derived taus
    if args.add_tau_derived:
        derived = compute_tau_derivatives(draws, (1.5, 2.5, 8.0), M_pivot=args.M_pivot)
        draws.update(derived)
        ordered = list(var_list) + list(derived.keys())
    else:
        ordered = list(var_list)

    # Stack & thin
    X, names = stack_for_corner(draws, ordered)
    X = thin_rows(X, nth=args.thin, max_draws=args.max_draws)

    # Build LaTeX labels
    labels_tex = [to_tex(label_map.get(nm, nm)) for nm in names]

    # Try corner first (if requested)
    use_corner = False
    if args.use_corner:
        try:
            import corner
            use_corner = True
        except Exception as e:
            print("[info] 'corner' not available, falling back to ArviZ. Reason:", e)

    if use_corner:
        fig = corner.corner(
            X,
            labels=labels_tex,
            show_titles=True,
            quantiles=[0.16, 0.50, 0.84],
            levels=args.levels,
            plot_datapoints=False,
            fill_contours=True,
            label_kwargs=dict(fontsize=args.fontsize),
            title_kwargs=dict(fontsize=args.fontsize),
            contourf_kwargs=dict(cmap=args.cmap, colors=None),
            hist2d_kwargs=dict(cmap=args.cmap, colors=None),
            colors=None,
            contour_kwargs=dict(colors="k")
            #color="C0",  # use default cycle color; fills use cmap internally
        )
        # Bigger tick labels
        for ax in fig.axes:
            ax.tick_params(labelsize=args.ticksize)

        # Add a discrete colorbar describing the credible levels
        #build_level_colorbar(fig, levels=args.levels, cmap_name=args.cmap, label="Credible level", ticksize=args.ticksize, fontsize=args.fontsize)

        fig.tight_layout()
        fig.savefig(args.out, dpi=220)
        if args.out_pdf:
            fig.savefig(args.out_pdf)
        plt.close(fig)
        print(f"[OK] saved corner plot to {args.out}" + (f" and {args.out_pdf}" if args.out_pdf else ""))
    else:
        # ArviZ fallback: build an InferenceData with scalar arrays keyed by ORIGINAL names
        data_for_az = {nm: X[:, i] for i, nm in enumerate(names)}
        id_for_plot = az.from_dict(posterior={k: v[:, None] for k, v in data_for_az.items()})
        # Map labels
        labeller = az.labels.MapLabeller(var_name_map={nm: labels_tex[i] for i, nm in enumerate(names)})

        ax = az.plot_pair(
            id_for_plot,
            var_names=names,
            kind="kde",
            marginals=True,
            kde_kwargs=dict(fill_kwargs={"alpha": 0.25}, hdi_probs=args.levels,
            contourf_kwargs={"cmap": "Blues"},
            contour_kwargs={"colors": "k"}),
             marginal_kwargs={
                "fill_kwargs": {"alpha": 0.0, "facecolor": "none"},  # ⟵ disable 1D fill
                "plot_kwargs": {"color": "k", "alpha": 1.0, "lw": 1.4},
                # IMPORTANT: do NOT pass 'quantiles' here; we’ll draw lines ourselves
            },
            #marginal_kwargs=dict(quantiles=[0.14, 0.5, 0.84], 
            #                    fill_kwargs=None, #{"alpha": 0.12, "facecolor": "gray"},  # fill ONLY
            #                    plot_kwargs={"alpha": 1.0, "color": "orange", "lw": 20},  # outline line
            #                    ),
            labeller=labeller,
            textsize=args.fontsize,
            figsize=(3*len(names), 3*len(names)),
        )
        # ax can be np.ndarray of axes; make all ticks larger
        for a in np.ravel(ax):
            a.tick_params(labelsize=args.ticksize)

        def draw_diag_quantiles(ax_grid, X, names, color="k"):
            A = np.asarray(ax_grid)
            for i, _ in enumerate(names):
                p16, p50, p84 = np.percentile(X[:, i], [16, 50, 84])
                A[i, i].axvline(p16, color=color, lw=1.2, ls="--", zorder=10)
                A[i, i].axvline(p50, color=color, lw=1.5, ls="-",  zorder=11)
                A[i, i].axvline(p84, color=color, lw=1.2, ls="--", zorder=10)

        draw_diag_quantiles(ax, X, names)

        fig = plt.gcf()
        #build_level_colorbar(fig, levels=args.levels, cmap_name=args.cmap, label="Credible level", ticksize=args.ticksize, fontsize=args.fontsize)
        plt.tight_layout()
        plt.savefig(args.out, dpi=220)
        if args.out_pdf:
            plt.savefig(args.out_pdf)
        plt.close()
        print(f"[OK] saved corner-like pair plot to {args.out}" + (f" and {args.out_pdf}" if args.out_pdf else ""))

if __name__ == "__main__":
    main()
