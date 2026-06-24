#!/usr/bin/env python3
"""Aggregate per-run .z result files into RMSE/bias summaries and PDF plots."""
import os
import re
import glob
import argparse
import warnings

import numpy
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt


use_tex = True                          # needs a LaTeX install + Latin Modern
mpl.rcParams.update({
    "text.usetex": use_tex,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "font.size": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
if not use_tex:
    mpl.rcParams.update({"text.usetex": False, "font.family": "serif"})

FIGSIZE = (3 * 1.618, 3)                # 4.854 x 3 in golden-ratio panel

combine_ips = True                      # pool IPS_SN into a single "IPS" curve

plot_estimators = ["OnPolicy", "IPS", "PI", "PI_SN", "DM_tree"]

estimator_labels = {
    "OnPolicy": r"On-policy",
    "IPS": r"IPS \& wIPS" if use_tex else "IPS & wIPS",
    "PI": r"PI",
    "PI_SN": r"wPI",
    "DM_tree": r"DM-tree",
}

line_styles = {
    "OnPolicy": dict(color="darkblue",  linestyle=(0, (5, 1)), band_color="darkblue",  band_alpha=0.12),
    "IPS":      dict(color="darkred",   linestyle="dashed",    band_color="darkred",   band_alpha=0.12),
    "PI":       dict(color="black",     linestyle="solid",     band_color="grey",      band_alpha=0.12),
    "PI_SN":    dict(color="darkgreen", linestyle="dashdot",   band_color="darkgreen", band_alpha=0.12),
    "DM_tree":  dict(color="purple",    linestyle="dotted",    band_color="purple",    band_alpha=0.12),
}

metric_labels = {
    "CarouselExpStreams": r"linear reward",
    "CarouselAnyStream": r"nonlinear reward",
    "SlateGroupExposure": r"Slate Group Exposure",
    "SlateAWRF": r"Slate AWRF",
    "SlateNDKL": r"Slate NDKL"
}

LEGEND_LOC = "best"
LEGEND_BBOX = None
LEGEND_NCOL = None


def safe_name(x):
    """Turn an arbitrary label into a filesystem-safe token."""
    return str(x).replace("/", "-").replace(" ", "_").replace(".", "p")


# ssynth_{metric}_{dataset}_{m}_{l}_{rep}{temp}_f{logr}_e{evalr}_{seed}_{approach}
#        [_pi{target}] [_{trainsize}] [_realized] _{iter}.z
FILENAME_RE = re.compile(
    r"^ssynth_(?P<metric>[A-Za-z]+)_(?P<dataset>MSLR10k|MSLR|MQ2008|MQ2007|Deezer)_"
    r"(?P<m>-?\d+)_(?P<l>\d+)_(?P<rep>[rn])(?P<temp>[0-9.]+)_"
    r"f(?P<logr>[a-z]+)_e(?P<evalr>[a-z]+)_(?P<seed>\d+)_"
    r"(?P<approach>OnPolicy|IPS_SN|IPS|PI_SN|PI|DMc_lasso|DMc_ridge|DM_tree|DM_lasso|DM_ridge)"
    r"(?:_pi(?P<target>[A-Za-z]+))?"
    r"(?:_(?P<trainsize>\d+))?"
    r"(?:_(?P<realized>realized))?"
    r"_(?P<iter>\d+)\.z$"
)


def parse_filename(path):
    """Parse one result filename into its experiment-condition fields."""
    mobj = FILENAME_RE.match(os.path.basename(path))
    if mobj is None:
        return None
    g = mobj.groupdict()
    approach = g["approach"]
    if combine_ips and approach == "IPS_SN":
        approach = "IPS"                # pool IPS_SN into IPS
    return {
        "metric": g["metric"],
        "dataset": g["dataset"],
        "m": int(g["m"]),
        "l": int(g["l"]),
        "temp": float(g["temp"]),
        "approach": approach,
        "target": g["target"],
        "trainsize": int(g["trainsize"]) if g["trainsize"] else None,
        "realized": g["realized"] is not None,
        "iter": int(g["iter"]),
        "path": path,
    }


def condition_key(f):
    """Group key identifying the experiment condition a result file belongs to."""
    return (f["metric"], f["m"], f["l"], round(f["temp"], 4),
            f["approach"], f["target"], f["realized"])


def load_all(dirs):
    """Load and group every parseable .z result file under the given directories."""
    paths = []
    for d in dirs:
        paths.extend(glob.glob(os.path.join(d, "**", "*.z"), recursive=True))
    groups = {}
    seen = set()
    n_parsed = n_bad = 0
    for p in sorted(paths):
        f = parse_filename(p)
        if f is None:
            continue
        key = condition_key(f)
        if (key, f["iter"], os.path.basename(p)) in seen:
            continue                    # IPS/IPS_SN share a key once combined
        try:
            saveValues, saveMSEs, savePreds, target = joblib.load(p)
        except Exception as e:
            warnings.warn("could not load %s (%s)" % (p, e))
            n_bad += 1
            continue
        seen.add((key, f["iter"], os.path.basename(p)))
        f["n"] = numpy.asarray(saveValues, dtype=numpy.float64)
        f["mse"] = numpy.asarray(saveMSEs, dtype=numpy.float64)
        f["preds"] = numpy.asarray(savePreds, dtype=numpy.float64)
        f["true_value"] = float(target)
        groups.setdefault(key, []).append(f)
        n_parsed += 1
    print("[load] parsed %d files (%d unreadable), %d conditions"
          % (n_parsed, n_bad, len(groups)))
    return groups


def aggregate(groups):
    """Reduce each condition's per-run records to mean RMSE, SE band, and bias."""
    out = {}
    for key, recs in groups.items():
        L = min(len(r["n"]) for r in recs)
        n = recs[0]["n"][:L]
        mse = numpy.vstack([r["mse"][:L] for r in recs])      # (n_iter, L)
        preds = numpy.vstack([r["preds"][:L] for r in recs])
        target = recs[0]["true_value"]
        n_iter = mse.shape[0]
        rmse_runs = numpy.sqrt(numpy.clip(mse, 0.0, None))     # per-run RMSE
        rmse = rmse_runs.mean(axis=0)
        if n_iter > 1:
            se = rmse_runs.std(axis=0, ddof=1) / numpy.sqrt(n_iter)
        else:
            se = numpy.zeros_like(rmse)
        rmse_lo = numpy.maximum(rmse - se, 1e-12)
        rmse_hi = rmse + se
        bias = preds.mean(axis=0) - target

        f0 = recs[0]
        if f0["approach"].startswith("DM") and f0["trainsize"]:
            mask = n <= f0["trainsize"]
            for arr in (rmse, rmse_lo, rmse_hi):
                arr[mask] = numpy.nan

        out[key] = dict(n=n, rmse=rmse, rmse_lo=rmse_lo, rmse_hi=rmse_hi,
                        bias=bias, target=target, n_iter=n_iter,
                        metric=f0["metric"], m=f0["m"], l=f0["l"],
                        temp=round(f0["temp"], 4), approach=f0["approach"],
                        target_pol=f0["target"], realized=f0["realized"])
    return out


def select(agg, **filt):
    """Return {approach: cond} for conditions matching the given fields."""
    res = {}
    for cond in agg.values():
        if all(cond.get(k) == v for k, v in filt.items()):
            res[cond["approach"]] = cond
    return res


def rmse_at(cond, n_star):
    """Read off (rmse, n) at the checkpoint nearest to n_star (last if None)."""
    if n_star is None:
        idx = len(cond["n"]) - 1
    else:
        idx = int(numpy.argmin(numpy.abs(cond["n"] - n_star)))
    return cond["rmse"][idx], cond["n"][idx]


def setup_log_axes(fs=12, xlabel=r"logged samples $n$", ylabel=r"RMSE"):
    """Configure a log-log RMSE-vs-n axis with the project's grid style."""
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel, fontsize=fs + 1)
    plt.ylabel(ylabel, fontsize=fs + 3)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.minorticks_on()
    plt.grid(True, which="major", linewidth=0.55, alpha=0.45, linestyle="dashed")
    plt.grid(True, which="minor", linewidth=0.35, alpha=0.14, linestyle="dashed")


def save_pdf(fig, savepath):
    """Tight-layout, save, and close a figure as a PDF."""
    fig.tight_layout()
    fig.savefig(savepath, format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print("saved:", savepath)


def _draw_legend(fs, loc=None, bbox=None, ncol=None, fontsize=None):
    """Draw the legend using the global LEGEND_* config, with per-call overrides."""
    eff_loc = LEGEND_LOC if loc is None else loc
    eff_bbox = LEGEND_BBOX if bbox is None else bbox
    eff_ncol = ncol if ncol is not None else (LEGEND_NCOL if LEGEND_NCOL is not None else 1)
    kw = dict(frameon=False,
              fontsize=(fs + 1) if fontsize is None else fontsize,
              loc=eff_loc, ncol=eff_ncol)
    if eff_bbox is not None:
        kw["bbox_to_anchor"] = eff_bbox
    plt.legend(**kw)


def _curve(cond, fs, style, label, lw=1.7, band=True, ls=None, alpha=1.0):
    """Plot one RMSE-vs-n line, with an optional SE band, for a single condition."""
    x = cond["n"]
    y = numpy.array(cond["rmse"], dtype=numpy.float64); y[y <= 0] = numpy.nan
    plt.plot(x, y, linewidth=lw, linestyle=(ls or style["linestyle"]),
             color=style["color"], alpha=alpha, label=label)
    if band:
        lo = numpy.array(cond["rmse_lo"], dtype=numpy.float64); lo[lo <= 0] = numpy.nan
        hi = numpy.array(cond["rmse_hi"], dtype=numpy.float64); hi[hi <= 0] = numpy.nan
        plt.fill_between(x, lo, hi, color=style["band_color"],
                         alpha=style["band_alpha"], linewidth=0)


def plot_rmse_curve(conds, savepath, fs=12, ylim=None, show_bands=True,
                    legend_loc=None, legend_bbox=None, legend_ncol=None):
    """Save one RMSE-vs-n panel covering the estimators present in `conds`."""
    present = [e for e in plot_estimators if e in conds]
    if not present:
        print("rmse curve skipped:", savepath)
        return
    fig = plt.figure(figsize=FIGSIZE)
    for est in present:
        _curve(conds[est], fs, line_styles[est],
               estimator_labels.get(est, est), band=show_bands)
    setup_log_axes(fs=fs)
    if ylim is not None:
        plt.ylim(*ylim)
    _draw_legend(fs, loc=legend_loc, bbox=legend_bbox,
                 ncol=(2 if legend_ncol is None and LEGEND_NCOL is None else legend_ncol))
    save_pdf(fig, savepath)


def plot_final_rmse(conds, savepath, n_star, fs=12):
    """Save discrete final-RMSE points with SE caps, one estimator per x-tick."""
    present = [e for e in plot_estimators if e in conds]
    if not present:
        print("final rmse skipped:", savepath)
        return
    fig = plt.figure(figsize=FIGSIZE)
    x = numpy.arange(len(present))
    for i, est in enumerate(present):
        c = conds[est]
        idx = (len(c["n"]) - 1) if n_star is None else int(numpy.argmin(numpy.abs(c["n"] - n_star)))
        mean = c["rmse"][idx]
        err = max(c["rmse_hi"][idx] - mean, 0.0)
        plt.errorbar(x[i], mean, yerr=err, linewidth=0, capsize=3,
                     marker="o", markersize=4, color=line_styles[est]["color"])
    plt.yscale("log")
    plt.xticks(x, [estimator_labels.get(e, e) for e in present],
               fontsize=fs, rotation=35, ha="right")
    plt.yticks(fontsize=fs)
    plt.xlabel(r"estimator", fontsize=fs + 1)
    plt.ylabel(r"final RMSE", fontsize=fs + 3)
    plt.minorticks_on()
    plt.grid(True, which="major", axis="y", linewidth=0.55, alpha=0.45, linestyle="dashed")
    plt.grid(True, which="minor", axis="y", linewidth=0.35, alpha=0.14, linestyle="dashed")
    save_pdf(fig, savepath)


def plot_bias_floor(agg, savepath, m, l, target, fs=12):
    """Save a PI-only linear-vs-nonlinear-reward bias-floor comparison panel."""
    metric_styles = {
        "CarouselExpStreams": dict(color="darkblue", linestyle="solid",  label=r"linear reward"),
        "CarouselAnyStream":  dict(color="darkred",  linestyle="dashed", label=r"nonlinear reward"),
    }
    fig = plt.figure(figsize=FIGSIZE)
    drawn = False
    for metric_name, st in metric_styles.items():
        conds = select(agg, metric=metric_name, m=m, l=l, temp=0.0,
                       target_pol=target, realized=False)
        if "PI" not in conds:
            continue
        c = conds["PI"]
        x = c["n"]
        y = numpy.array(c["rmse"], dtype=numpy.float64); y[y <= 0] = numpy.nan
        lo = numpy.array(c["rmse_lo"], dtype=numpy.float64); lo[lo <= 0] = numpy.nan
        hi = numpy.array(c["rmse_hi"], dtype=numpy.float64); hi[hi <= 0] = numpy.nan
        plt.plot(x, y, linewidth=1.7, linestyle=st["linestyle"], color=st["color"], label=st["label"])
        plt.fill_between(x, lo, hi, color=st["color"], alpha=0.12, linewidth=0)
        drawn = True
    if not drawn:
        plt.close(fig); print("bias-floor skipped:", savepath); return
    setup_log_axes(fs=fs)
    _draw_legend(fs, ncol=1)
    save_pdf(fig, savepath)


def make_target_plots(agg, out_dir, m, l, metrics, targets, fs=12):
    """Save one RMSE curve per (target policy, metric) combination."""
    for tgt in targets:
        for metric_name in metrics:
            conds = select(agg, metric=metric_name, m=m, l=l, temp=0.0,
                           target_pol=tgt, realized=False)
            if not conds:
                print("target plot skipped:", tgt, metric_name); continue
            sp = os.path.join(out_dir, "10_target_%s_%s_m%d_l%d_t0p0.pdf"
                              % (safe_name(tgt), safe_name(metric_name), m, l))
            plot_rmse_curve(conds, sp, fs=fs)


def make_normalized_cdf(agg, out_dir, m, l, n_star, metrics, targets, fs=12):
    """Save the CDF of per-condition RMSE normalized to [0,1] across estimators."""
    per_est = {e: [] for e in plot_estimators if e != "OnPolicy"}
    n_cond = 0
    for metric_name in metrics:
        for tgt in targets:
            conds = select(agg, metric=metric_name, m=m, l=l, temp=0.0,
                           target_pol=tgt, realized=False)
            vals = {}
            for est in per_est:
                if est in conds:
                    r, _ = rmse_at(conds[est], n_star)
                    if numpy.isfinite(r):
                        vals[est] = r
            if len(vals) < 2:
                continue
            lo, hi = min(vals.values()), max(vals.values())
            n_cond += 1
            for est, r in vals.items():
                per_est[est].append(0.001 if hi == lo else 0.001 + (r - lo) / (hi - lo) * 0.999)
    if n_cond == 0:
        print("CDF skipped"); return
    fig = plt.figure(figsize=FIGSIZE)
    for est in per_est:
        v = numpy.sort(numpy.array(per_est[est]))
        if v.size == 0:
            continue
        y = numpy.arange(1, v.size + 1) / v.size
        st = line_styles[est]
        plt.step(numpy.concatenate([[1e-3], v]), numpy.concatenate([[0], y]),
                 where="post", color=st["color"], linestyle=st["linestyle"],
                 linewidth=1.7, label=estimator_labels.get(est, est))
    plt.xscale("log")
    plt.xlabel(r"normalized RMSE", fontsize=fs + 1)
    plt.ylabel(r"fraction of conditions", fontsize=fs + 3)
    plt.xticks(fontsize=fs); plt.yticks(fontsize=fs)
    plt.minorticks_on()
    plt.grid(True, which="major", linewidth=0.55, alpha=0.45, linestyle="dashed")
    plt.grid(True, which="minor", linewidth=0.35, alpha=0.14, linestyle="dashed")
    _draw_legend(fs, ncol=1)
    save_pdf(fig, os.path.join(out_dir, "20_cdf_normalized_rmse_m%d_l%d_t0p0.pdf" % (m, l)))


def _scaling_panel(agg, out_dir, fname, metric_name, sweep, fixed, vary, xlabel, n_star, fs=12):
    """Save final-RMSE vs. one swept slate-shape parameter, for OnPolicy/IPS/PI."""
    estimators = ["OnPolicy", "IPS", "PI"]
    fig = plt.figure(figsize=FIGSIZE)
    drawn = False
    all_x = []
    for est in estimators:
        xs, ys, yerr = [], [], []
        for val in sweep:
            filt = dict(metric=metric_name, temp=0.0, target_pol="Optimal", realized=False)
            filt[vary] = val
            filt.update(fixed)
            conds = select(agg, **filt)
            if est in conds:
                c = conds[est]
                idx = (len(c["n"]) - 1) if n_star is None else int(numpy.argmin(numpy.abs(c["n"] - n_star)))
                xs.append(val); ys.append(c["rmse"][idx])
                yerr.append(max(c["rmse_hi"][idx] - c["rmse"][idx], 0.0))
                all_x.append(val)
        if xs:
            st = line_styles[est]
            plt.errorbar(xs, ys, yerr=numpy.nan_to_num(yerr), linewidth=1.7, capsize=3,
                         linestyle=st["linestyle"], color=st["color"],
                         label=estimator_labels.get(est, est))
            drawn = True
    if not drawn:
        plt.close(fig); print("scaling skipped:", fname); return
    ax = plt.gca()
    plt.yscale("log")
    plt.xlabel(xlabel, fontsize=fs + 1)
    plt.ylabel(r"final RMSE", fontsize=fs + 3)
    valid_x = sorted(set(all_x))
    ax.set_xscale("linear")
    ax.set_xticks(valid_x)
    ax.set_xticklabels([str(int(v)) for v in valid_x], fontsize=fs)
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    plt.yticks(fontsize=fs)
    plt.minorticks_on()
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    plt.grid(True, which="major", linewidth=0.55, alpha=0.45, linestyle="dashed")
    plt.grid(True, which="minor", axis="y", linewidth=0.35, alpha=0.14, linestyle="dashed")
    _draw_legend(fs, ncol=2)
    save_pdf(fig, os.path.join(out_dir, fname))


def make_scaling_plots(agg, out_dir, n_star, metrics, m_sweep, l_sweep, fs=12):
    """Save the m-sweep and l-sweep scaling panels for each metric."""
    for metric_name in metrics:
        _scaling_panel(agg, out_dir,
                       "30_scaling_m_%s_l12_Optimal.pdf" % safe_name(metric_name),
                       metric_name, m_sweep, fixed=dict(l=12), vary="m",
                       xlabel=r"candidate set size $m$", n_star=n_star, fs=fs)
        _scaling_panel(agg, out_dir,
                       "31_scaling_l_%s_m100_Optimal.pdf" % safe_name(metric_name),
                       metric_name, l_sweep, fixed=dict(m=100), vary="l",
                       xlabel=r"slate length $l$", n_star=n_star, fs=fs)


def make_logging_plots(agg, out_dir, m, l, temps, metrics, target="Optimal", fs=12):
    """Save one RMSE curve per (metric, logging temperature) combination."""
    for metric_name in metrics:
        for tt in temps:
            conds = select(agg, metric=metric_name, m=m, l=l, temp=round(tt, 4),
                           target_pol=target, realized=False)
            if not conds:
                print("logging plot skipped:", metric_name, tt); continue
            sp = os.path.join(out_dir, "40_logging_%s_t%s_m%d_l%d_%s.pdf"
                              % (safe_name(metric_name), safe_name(tt), m, l, safe_name(target)))
            plot_rmse_curve(conds, sp, fs=fs)


def make_realized_overlay(agg, out_dir, m, l, metrics, target="Optimal",
                          estimators=("OnPolicy", "IPS", "PI"), fs=12):
    """Save deterministic-vs-realized overlay curves for each metric."""
    for metric_name in metrics:
        det = select(agg, metric=metric_name, m=m, l=l, temp=0.0, target_pol=target, realized=False)
        rel = select(agg, metric=metric_name, m=m, l=l, temp=0.0, target_pol=target, realized=True)
        if not any(e in rel for e in estimators):
            print("realized overlay skipped (no realized):", metric_name); continue
        fig = plt.figure(figsize=FIGSIZE)
        drawn = False
        for est in estimators:
            if est not in line_styles:
                continue
            st = line_styles[est]
            if est in det:
                _curve(det[est], fs, st, estimator_labels.get(est, est) + r" (det.)",
                       lw=1.5, band=True, ls="solid")
                drawn = True
            if est in rel:
                _curve(rel[est], fs, st, estimator_labels.get(est, est) + r" (real.)",
                       lw=1.5, band=True, ls="dashed", alpha=0.85)
                drawn = True
        if not drawn:
            plt.close(fig); continue
        setup_log_axes(fs=fs)
        _draw_legend(fs, ncol=2, fontsize=fs)
        save_pdf(fig, os.path.join(out_dir, "50_realized_overlay_%s_m%d_l%d_%s.pdf"
                                   % (safe_name(metric_name), m, l, safe_name(target))))


def dump_summary_csv(agg, out_dir, n_star):
    """Write one summary.csv row per condition with its RMSE and bias at n_star."""
    rows = ["metric,m,l,temp,approach,target,realized,n_iter,n_star,rmse,signed_bias"]
    for c in agg.values():
        r, ns = rmse_at(c, n_star)
        idx = -1 if n_star is None else int(numpy.argmin(numpy.abs(c["n"] - n_star)))
        rows.append("%s,%d,%d,%.3f,%s,%s,%d,%d,%g,%.6g,%.6g" % (
            c["metric"], c["m"], c["l"], c["temp"], c["approach"],
            c["target_pol"], int(c["realized"]), c["n_iter"], ns, r, c["bias"][idx]))
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print("[csv] wrote %s (%d conditions)" % (path, len(agg)))


def main():
    """Aggregate result files under --results and write the standard plot set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["./results"],
                    help="directories holding the .z files (searched recursively)")
    ap.add_argument("--figdir", default="./plots/selected")
    ap.add_argument("--n-star", type=int, default=None,
                    help="n at which to read RMSE for CDF/scaling/final (default: last checkpoint)")
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--l", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    groups = load_all(args.results)
    if not groups:
        print("No parseable .z files found under: %s" % args.results)
        return
    agg = aggregate(groups)

    for fld in ("metric", "m", "l", "temp", "approach", "target_pol", "realized"):
        print(fld, sorted(set(str(c[fld]) for c in agg.values())))

    metrics = ["CarouselExpStreams", "CarouselAnyStream", "SlateGroupExposure", "SlateAWRF", "SlateNDKL"]
    targets = ["Optimal", "Popularity", "Segment", "Random", "LearnedLogistic"]
    temps = [0.0, 0.5, 1.0]
    m_sweep = [20, 50, 100]
    l_sweep = [3, 5, 12]
    M, L = args.m, args.l

    for metric_name in metrics:                                # 00 - selected headline condition
        conds = select(agg, metric=metric_name, m=M, l=L, temp=0.0,
                       target_pol="Optimal", realized=False)
        base = "00_selected_%s_m%d_l%d_t0p0_Optimal" % (safe_name(metric_name), M, L)
        plot_rmse_curve(conds, os.path.join(args.figdir, base + "_rmse_curve.pdf"), fs=10)
        plot_final_rmse(conds, os.path.join(args.figdir, base + "_final_rmse.pdf"), args.n_star, fs=10)

    plot_bias_floor(agg, os.path.join(args.figdir, "02_bias_floor_pi_m%d_l%d_Optimal.pdf" % (M, L)),
                    m=M, l=L, target="Optimal")                # 02 - bias floor

    make_target_plots(agg, args.figdir, M, L, metrics, targets, fs=10)             # 10
    make_normalized_cdf(agg, args.figdir, M, L, args.n_star, metrics, targets, fs=10)  # 20
    make_scaling_plots(agg, args.figdir, args.n_star, metrics, m_sweep, l_sweep, fs=10)  # 30/31
    make_logging_plots(agg, args.figdir, M, L, temps, metrics, fs=10)               # 40
    make_realized_overlay(agg, args.figdir, M, L, metrics, fs=10)                   # 50

    dump_summary_csv(agg, args.figdir, args.n_star)
    print("Done. PDFs + summary.csv in", args.figdir)


if __name__ == "__main__":
    main()
