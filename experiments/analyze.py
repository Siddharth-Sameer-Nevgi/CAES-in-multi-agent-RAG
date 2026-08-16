"""Phase 5 analysis: the paper's table, figures, and headline number.

    python -m experiments.analyze

Reads results/{policy}_raw.jsonl and produces:
  1. Main table         -- mean cost, latency, EM, F1, mean iterations per policy
  2. Cost-vs-quality    -- the headline figure
  3. Iteration histogram-- CAES spread across 1-5 vs Fixed flat at N
  4. Coverage-vs-iteration -- empirical diminishing returns
  5. Headline number    -- % cost reduction vs Fixed at statistically
                           indistinguishable F1 (paired t-test + bootstrap CI)

Figures are light-mode only: they are destined for a printed paper, so a single
committed look is correct here rather than a theme-aware pair.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

import config

POLICIES = ["fixed", "oneshot", "caes"]

DISPLAY = {"fixed": "Fixed (N=3)", "oneshot": "One-shot routing", "caes": "CAES"}

# Categorical slots 1-3 of the validated palette, assigned to ENTITIES and never
# to rank, so a re-ordered table never repaints a series. Only the first three
# slots clear the all-pairs CVD floors, which is exactly the number of policies.
COLORS = {"caes": "#2a78d6", "fixed": "#eb6834", "oneshot": "#1baf7a"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_policy(policy: str) -> list[dict]:
    path = config.RESULTS_DIR / f"{policy}_raw.jsonl"
    if not path.exists():
        return []
    out, seen = [], set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec["query_id"] in seen:
                continue        # a resumed run can duplicate; keep the first
            seen.add(rec["query_id"])
            out.append(rec)
    return out


def load_all() -> dict[str, list[dict]]:
    return {p: load_policy(p) for p in POLICIES}


def paired(a: list[dict], b: list[dict], field: str):
    """Align two policies on shared query ids and return paired arrays."""
    ma = {r["query_id"]: r for r in a}
    mb = {r["query_id"]: r for r in b}
    ids = sorted(set(ma) & set(mb))
    return (np.array([ma[i][field] for i in ids], dtype=float),
            np.array([mb[i][field] for i in ids], dtype=float),
            ids)


# ---------------------------------------------------------------------------
# 1. Main table
# ---------------------------------------------------------------------------

def main_table(data: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for p in POLICIES:
        recs = data.get(p, [])
        if not recs:
            continue
        rows.append({
            "policy": DISPLAY[p],
            "n": len(recs),
            "mean_iterations": statistics.mean(r["iterations_used"] for r in recs),
            "mean_cost_usd": statistics.mean(r["total_usd"] for r in recs),
            "mean_latency_ms": statistics.mean(r["total_latency_ms"] for r in recs),
            "exact_match": statistics.mean(r["exact_match"] for r in recs),
            "f1": statistics.mean(r["f1"] for r in recs),
            "final_coverage": statistics.mean(r["final_coverage"] for r in recs),
            "abstention_rate": statistics.mean(r["abstained"] for r in recs),
        })
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results to tabulate.")
        return
    hdr = (f"{'policy':<20}{'n':>5}{'iters':>8}{'cost $':>10}{'lat ms':>9}"
           f"{'EM':>8}{'F1':>8}{'cov':>7}{'abst':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['policy']:<20}{r['n']:>5}{r['mean_iterations']:>8.2f}"
              f"{r['mean_cost_usd']:>10.5f}{r['mean_latency_ms']:>9.0f}"
              f"{r['exact_match']:>8.3f}{r['f1']:>8.3f}"
              f"{r['final_coverage']:>7.2f}{r['abstention_rate']:>7.2f}")


# ---------------------------------------------------------------------------
# 5. Headline number
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def headline(data: dict[str, list[dict]]) -> dict | None:
    """Cost reduction vs Fixed at statistically indistinguishable F1."""
    fixed, caes = data.get("fixed", []), data.get("caes", [])
    if not fixed or not caes:
        return None

    f1_fixed, f1_caes, ids = paired(fixed, caes, "f1")
    c_fixed, c_caes, _ = paired(fixed, caes, "total_usd")
    if len(ids) < 2:
        return None

    d_f1 = f1_caes - f1_fixed
    lo, hi = bootstrap_ci(d_f1)

    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(f1_caes, f1_fixed)
        t_stat, p_value = float(t_stat), float(p_value)
    except ImportError:
        t_stat = p_value = float("nan")
        print("(scipy not installed: reporting bootstrap CI only, no t-test)")

    cost_reduction = (1 - c_caes.mean() / c_fixed.mean()) * 100 if c_fixed.mean() else 0.0
    rel = (c_caes - c_fixed) / np.maximum(c_fixed, 1e-12) * 100
    r_lo, r_hi = bootstrap_ci(rel)

    return {
        "n_paired": len(ids),
        "f1_fixed": float(f1_fixed.mean()),
        "f1_caes": float(f1_caes.mean()),
        "f1_delta": float(d_f1.mean()),
        "f1_delta_ci95": (lo, hi),
        "f1_t_stat": t_stat,
        "f1_p_value": p_value,
        "cost_fixed": float(c_fixed.mean()),
        "cost_caes": float(c_caes.mean()),
        "cost_reduction_pct": float(cost_reduction),
        "cost_reduction_ci95": (-r_hi, -r_lo),
        "indistinguishable": bool(lo <= 0 <= hi),
    }


def print_headline(h: dict | None) -> None:
    print("\n" + "=" * 66)
    if h is None:
        print("Headline number needs both fixed_raw.jsonl and caes_raw.jsonl.")
        print("=" * 66)
        return
    print("HEADLINE")
    print(f"  paired queries    : {h['n_paired']}")
    print(f"  F1  Fixed -> CAES : {h['f1_fixed']:.4f} -> {h['f1_caes']:.4f} "
          f"(delta {h['f1_delta']:+.4f})")
    print(f"  F1 delta 95% CI   : [{h['f1_delta_ci95'][0]:+.4f}, "
          f"{h['f1_delta_ci95'][1]:+.4f}]")
    print(f"  paired t-test     : t={h['f1_t_stat']:.3f}  p={h['f1_p_value']:.4f}")
    print(f"  cost Fixed -> CAES: ${h['cost_fixed']:.5f} -> ${h['cost_caes']:.5f}")
    print(f"  COST REDUCTION    : {h['cost_reduction_pct']:.1f}%  "
          f"95% CI [{h['cost_reduction_ci95'][0]:.1f}%, "
          f"{h['cost_reduction_ci95'][1]:.1f}%]")
    if h["indistinguishable"]:
        print(f"\n  => CAES cuts cost {h['cost_reduction_pct']:.1f}% at F1 "
              f"statistically indistinguishable from Fixed\n"
              f"     (the 95% CI on the F1 difference contains zero).")
    else:
        print(f"\n  => F1 DIFFERS significantly (CI excludes zero). Report the "
              f"cost reduction\n     alongside the quality change; do not claim "
              f"parity.")
    print("=" * 66)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, pad=14, loc="left")
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)   # grid stays recessive, behind the marks


def _new_fig(size=(7.2, 4.6)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size, dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, name: str) -> Path:
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def fig_cost_vs_quality(data: dict[str, list[dict]]) -> None:
    """The headline figure. One point per policy, with 95% CIs on both axes."""
    fig, ax = _new_fig()
    points = []
    for p in POLICIES:
        recs = data.get(p, [])
        if not recs:
            continue
        cost = np.array([r["total_usd"] for r in recs])
        f1 = np.array([r["f1"] for r in recs])
        clo, chi = bootstrap_ci(cost)
        flo, fhi = bootstrap_ci(f1)
        x, y = cost.mean(), f1.mean()
        points.append((p, x, y))

        ax.errorbar(
            x, y,
            xerr=[[x - clo], [chi - x]], yerr=[[y - flo], [fhi - y]],
            fmt="o", markersize=11, color=COLORS[p], ecolor=COLORS[p],
            elinewidth=2, capsize=0, alpha=0.95,
            markeredgecolor=SURFACE, markeredgewidth=2,   # surface ring
            label=DISPLAY[p], zorder=3,
        )

    # Headroom so direct labels have somewhere to live.
    ax.margins(x=0.20, y=0.30)
    lo_x, hi_x = ax.get_xlim()
    midpoint = (lo_x + hi_x) / 2

    for p, x, y in points:
        # Direct label. Required relief for the palette's contrast WARN, and it
        # means identity never rests on colour alone. Labels on right-hand
        # points flip inward so they cannot run off the axes.
        right_half = x > midpoint
        ax.annotate(
            DISPLAY[p], (x, y), textcoords="offset points",
            xytext=(-13 if right_half else 13, 9),
            ha="right" if right_half else "left",
            color=INK, fontsize=10, zorder=4,
        )

    _style_axes(ax, "Mean cost per query (USD)", "Mean F1",
                "Cost vs quality: CAES against fixed-depth and one-shot routing")
    ax.legend(frameon=False, labelcolor=INK_MUTED, fontsize=9, loc="lower right")
    _save(fig, "fig1_cost_vs_quality.png")


def fig_iteration_histogram(data: dict[str, list[dict]]) -> None:
    """CAES spread across 1-N vs Fixed flat at N. The clearest visual argument."""
    fig, ax = _new_fig()
    bins = list(range(1, config.MAX_ITERATIONS + 1))
    present = [p for p in POLICIES if data.get(p)]
    if not present:
        return
    group_w = 0.8
    bar_w = group_w / len(present)

    for i, p in enumerate(present):
        recs = data[p]
        counts = [sum(1 for r in recs if r["iterations_used"] == b) for b in bins]
        share = [100 * c / len(recs) for c in counts]
        offs = [b - group_w / 2 + bar_w * (i + 0.5) for b in bins]
        ax.bar(offs, share, width=bar_w * 0.86, color=COLORS[p],
               label=DISPLAY[p], zorder=3,
               edgecolor=SURFACE, linewidth=1.2)   # surface gap between bars

    _style_axes(ax, "Retrieval iterations used", "Share of queries (%)",
                "Per-iteration granularity: where each policy actually stops")
    ax.set_xticks(bins)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 118)          # headroom so the legend never crowds a 100% bar
    ax.legend(frameon=False, labelcolor=INK_MUTED, fontsize=9,
              loc="upper right", ncols=len(present))
    _save(fig, "fig2_iteration_histogram.png")


def fig_coverage_vs_iteration(data: dict[str, list[dict]]) -> None:
    """Empirical diminishing returns -- the premise of the problem statement."""
    fig, ax = _new_fig()
    max_it = config.MAX_ITERATIONS
    drawn = []
    for p in POLICIES:
        recs = data.get(p, [])
        if not recs:
            continue
        xs, ys, los, his = [], [], [], []
        for it in range(1, max_it + 1):
            vals = np.array([r["coverage_history"][it - 1] for r in recs
                             if len(r.get("coverage_history", [])) >= it])
            if len(vals) < 3:      # too few queries reached this depth to plot
                continue
            lo, hi = bootstrap_ci(vals)
            xs.append(it); ys.append(vals.mean()); los.append(lo); his.append(hi)
        if not xs:
            continue
        ax.plot(xs, ys, color=COLORS[p], linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label=DISPLAY[p], zorder=3)
        ax.fill_between(xs, los, his, color=COLORS[p], alpha=0.12, linewidth=0,
                        zorder=2)
        drawn.append((p, xs[-1], ys[-1]))

    # Policies that stop at the same depth land on the same x, and their labels
    # would stack. Stagger vertically by draw order so they stay legible.
    ax.margins(x=0.16, y=0.18)
    for rank, (p, x, y) in enumerate(drawn):
        ax.annotate(DISPLAY[p], (x, y), textcoords="offset points",
                    xytext=(11, 10 - 13 * rank), ha="left",
                    color=INK, fontsize=9, zorder=4)

    _style_axes(ax, "Retrieval iteration", "Mean verifier coverage",
                "Diminishing returns: coverage gain per additional iteration")
    ax.set_xticks(range(1, max_it + 1))
    ax.legend(frameon=False, labelcolor=INK_MUTED, fontsize=9, loc="lower right")
    _save(fig, "fig3_coverage_vs_iteration.png")


def fig_lambda_sweep() -> None:
    """The tuning curve: cost/quality tradeoff as lambda varies."""
    import csv

    path = config.RESULTS_DIR / "lambda_sweep.csv"
    if not path.exists():
        print("  (skipping fig4: run tune_lambda.py first)")
        return
    with path.open(encoding="utf-8") as fh:
        rows = sorted((r for r in csv.DictReader(fh)),
                      key=lambda r: float(r["lambda"]))
    if not rows:
        return

    lams = [float(r["lambda"]) for r in rows]
    cost = [float(r["mean_usd"]) for r in rows]
    f1 = [float(r["f1"]) for r in rows]

    # One axis, never two: cost is plotted as the x position, not a second y.
    fig, ax = _new_fig()
    ax.plot(cost, f1, color=COLORS["caes"], linewidth=2, marker="o",
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    for lam, c, f in zip(lams, cost, f1):
        ax.annotate(f"λ={lam:g}", (c, f), textcoords="offset points",
                    xytext=(8, -4), color=INK_MUTED, fontsize=8, zorder=4)

    _style_axes(ax, "Mean cost per query (USD)", "Mean F1 (tuning split)",
                "λ sweep: the cost/quality tradeoff curve")
    _save(fig, "fig4_lambda_sweep.png")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    data = load_all()
    missing = [p for p in POLICIES if not data.get(p)]
    if missing:
        print(f"WARNING: no results for {', '.join(missing)}. "
              f"Run `python -m experiments.run --policy <name> --yes` first.\n")
    if not any(data.values()):
        print("Nothing to analyse.", file=sys.stderr)
        return 2

    rows = main_table(data)
    print("MAIN TABLE")
    print_table(rows)

    out_csv = config.RESULTS_DIR / "main_table.csv"
    import csv as _csv
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    # Stop-reason breakdown: evidence the gate is engaging, not just the cap.
    print("\nSTOP REASONS")
    for p in POLICIES:
        recs = data.get(p, [])
        if not recs:
            continue
        counts: dict[str, int] = {}
        for r in recs:
            counts[r["stop_reason"]] = counts.get(r["stop_reason"], 0) + 1
        spread = sorted({r["iterations_used"] for r in recs})
        print(f"  {DISPLAY[p]:<20} {counts}   iterations seen: {spread}")

    h = headline(data)
    print_headline(h)
    if h:
        (config.RESULTS_DIR / "headline.json").write_text(
            json.dumps(h, indent=2), encoding="utf-8")

    if not args.no_figures:
        print("\nFIGURES")
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            print("  matplotlib not installed; skipping figures.")
            return 0
        fig_cost_vs_quality(data)
        fig_iteration_histogram(data)
        fig_coverage_vs_iteration(data)
        fig_lambda_sweep()
    return 0


if __name__ == "__main__":
    sys.exit(main())
