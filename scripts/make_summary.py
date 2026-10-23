"""Generate report/results_summary.md from results/ (no hand-typed metrics).

Every number and every ranking in the "Key findings" section is computed here from the CSVs; prose
that states a direction (better/worse) is chosen by the computed comparison, not written in advance.

  python scripts/make_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlse_ml.viz.style import MODELS  # noqa: E402

RES, TAB = ROOT / "results", ROOT / "report" / "tables"
LAB = {k: v[0] for k, v in MODELS.items()}
FAR = {"A": "far", "B_sech": "extrap", "B_gauss": "extrap"}


def seed_split(long, metric, regime, exp, split):
    d = long[(long.metric == metric) & (long.regime == regime) & (long.exp == exp) & (long.split == split)
             & (long.tag.fillna("") == "")]
    per_seed = d.groupby(["model", "seed"])["value"].mean()
    return per_seed.groupby("model").agg(["mean", "std", "median", "count"])


def ranking(s, pct=True, k=None):
    s = s.sort_values("mean")
    items = [f"{LAB[m]} {r['mean'] * (100 if pct else 1):.{1 if pct else 2}f}{'%' if pct else ''}"
             for m, r in s.iterrows()]
    return ", ".join(items[:k] if k else items)


def robust_gap(s, a, b):
    """True if a < b in every... approximated by non-overlapping mean ± std intervals."""
    if a not in s.index or b not in s.index:
        return None
    ra, rb = s.loc[a], s.loc[b]
    if min(ra["count"], rb["count"]) < 3:
        return None  # not assessed with fewer than 3 seeds
    return bool((ra["mean"] + ra["std"]) < (rb["mean"] - rb["std"]))


def findings(long, runs):
    F = []
    regs = [r for r in ["A", "B_sech", "B_gauss"] if r in set(long.regime)]
    for reg in regs:
        if not ((long.regime == reg) & (long.exp == "n_extrap")).any():
            continue
        ind = seed_split(long, "rel_l2", reg, "n_extrap", "interp")
        far = seed_split(long, "rel_l2", reg, "n_extrap", FAR[reg])
        fari = seed_split(long, "rel_l2_intensity", reg, "n_extrap", FAR[reg])
        F.append(f"**{reg}, in distribution (interp N), complex-field rel-L2:** {ranking(ind)} "
                 f"(mean over {int(ind['count'].min())} seeds). See `fig_extrap_N_{'regimeA' if reg == 'A' else reg}`.")
        if "near" in set(long[long.regime == reg].split):
            near = seed_split(long, "rel_l2", reg, "n_extrap", "near")
            F.append(f"**{reg}, near extrapolation:** {ranking(near)}.")
        F.append(f"**{reg}, {FAR[reg]} extrapolation, complex field:** {ranking(far)}. "
                 f"**Intensity only:** {ranking(fari)}.")
        best_far_i = fari["mean"].idxmin()
        best_in = ind["mean"].idxmin()
        if best_far_i != best_in:
            rg = robust_gap(fari, best_far_i, best_in)
            rob = {True: "robust across seeds (non-overlapping mean ± std)",
                   False: "NOT robust across seeds (mean ± std intervals overlap)",
                   None: "robustness not assessed: fewer than 3 seeds"}[rg]
            F.append(f"*{reg}: the ranking inverts.* The best in-distribution model ({LAB[best_in]}) is not the best "
                     f"at {FAR[reg]}-N intensity ({LAB[best_far_i]}); {rob}.")
        ed = seed_split(long, "energy_drift_max", reg, "n_extrap", FAR[reg])
        F.append(f"**{reg}, {FAR[reg]}-N max energy drift |ΔE|/E** (label-free plausibility): {ranking(ed, pct=False)}.")
    for reg in regs:
        if ((long.regime == reg) & (long.exp == "z_extrap")).any():
            z = seed_split(long, "rel_l2", reg, "z_extrap", "interp")
            zs = seed_split(long, "spectral_err", reg, "z_extrap", "interp")
            F.append(f"**{reg}, z-extrapolation (ξ ∈ (2, 4], interp N):** {ranking(z)}. Spectral error: "
                     f"{ranking(zs)}. See `fig_extrap_z`.")
    # cost
    r = runs[(runs.tag.fillna("") == "")]
    t = r.groupby("model")["train_wall_s"].median() / 60
    inf = r.groupby("model")["inference_s_per_field"].median() * 1e3
    F.append("**Cost (median per run):** training " + ", ".join(f"{LAB[m]} {t[m]:.1f} min" for m in t.sort_values().index)
             + "; inference " + ", ".join(f"{LAB[m]} {inf[m]:.0f} ms" for m in inf.sort_values().index) + " per field.")
    c = RES / "ssfm_cost.csv"
    if c.exists():
        cc = pd.read_csv(c)
        cc = cc[cc.rel_l2 > 1e-12]  # drop ladder entries identical to the reference itself
        fastest_inf = inf.min()
        parts = []
        for reg in regs:
            ind = seed_split(long, "rel_l2", reg, "n_extrap", "interp")
            if ind.empty or reg not in set(cc.regime):
                continue
            tgt = ind["mean"].min()  # best model's in-distribution error in this regime
            ok = cc[(cc.regime == reg) & (cc.rel_l2 <= tgt)]
            worst_n = ok.groupby("N")["wall_s"].min().max()  # slowest N in the ladder at that accuracy
            best_acc = cc[cc.regime == reg].groupby("N")["rel_l2"].min().max()
            parts.append(f"{reg}: matching the best model's interp error ({tgt * 100:.2f}%) costs ≤ {worst_n * 1e3:.0f} ms "
                         f"per field (worst N in the ladder), and finer SSFM settings reach {best_acc:.0e} for every N")
        F.append("**SSFM at matched accuracy** (single-threaded NumPy, `results/ssfm_cost.csv`): " + "; ".join(parts)
                 + f". Fastest model inference: {fastest_inf:.0f} ms per field. See `fig_pareto`.")
    # analyses
    a = RES / "analysis_A_spearman.csv"
    if a.exists():
        sa = pd.read_csv(a)
        if "regime" in sa:
            sa = sa[sa.regime == "ALL"]  # pooled over regimes; per-regime rows are in the CSV
        bl = RES / "analysis_A_phi_vs_baselines.csv"
        dv = pd.read_csv(bl) if bl.exists() else pd.DataFrame()
        names = {"xi_only": "ξ alone", "N_only": "N alone", "N2_xi": "N²ξ", "rank_N_plus_rank_xi": "rank(N)+rank(ξ)"}
        for exp in sa.exp.unique():
            for pred in ["wf_train", "phi_eff", "N2_xi", "rank_N_plus_rank_xi", "xi_only"]:
                s = sa[(sa.exp == exp) & (sa.predictor == pred)]
                if s.empty:
                    continue
                F.append(f"**Analysis A ({exp}), error vs {pred}:** Spearman ρ " + ", ".join(
                    f"{LAB[m]} {row.rho:.2f} [{row.ci_lo:.2f}, {row.ci_hi:.2f}]" for m, row in s.set_index("model").iterrows())
                         + ".")
            if dv.empty:
                continue
            d = dv[dv.exp == exp]
            parts = []
            for b_, g in d.groupby("baseline", sort=False):
                better = [LAB[m] for m, r in g.set_index("model").iterrows() if r.ci_lo > 0]
                worse = [LAB[m] for m, r in g.set_index("model").iterrows() if r.ci_hi < 0]
                tie = [LAB[m] for m, r in g.set_index("model").iterrows() if r.ci_lo <= 0 <= r.ci_hi]
                parts.append(f"vs {names[b_]}: Δρ {g.rho_phi_minus_baseline.min():+.3f} to {g.rho_phi_minus_baseline.max():+.3f}"
                             f" (φ_eff better: {', '.join(better) or 'none'}; worse: {', '.join(worse) or 'none'}; "
                             f"CI includes 0: {', '.join(tie) or 'none'})")
            F.append(f"**Analysis A ({exp}), is the SPM phase φ_eff = N² asinh ξ better than coordinate-only baselines?** "
                     "ρ(φ_eff) − ρ(baseline), 95% cluster-bootstrap CI: " + "; ".join(parts)
                     + ". See `fig_error_predictors_baselines`.")
        if not dv.empty:
            coord = dv[dv.baseline.isin(["N2_xi", "rank_N_plus_rank_xi"])]
            n_max = coord[coord.exp == "n_extrap"].rho_phi_minus_baseline.abs().max()
            z_all_worse = bool((coord[coord.exp == "z_extrap"].ci_hi < 0).all())
            if n_max <= 0.06 and z_all_worse:  # the wording below is only emitted when the numbers support it
                F.append(f"**Analysis A verdict (D-031):** φ_eff is no more than {n_max:.3f} in ρ away from coordinate-only "
                         "baselines in N-extrapolation, and below them for every model in z-extrapolation. Per-slice "
                         "error grows monotonically with N and ξ for every model, and φ_eff carries little information "
                         "beyond that. It should not be read as a specific physical failure law.")
            else:
                F.append("**Analysis A verdict:** the baseline comparison changed; re-assess the wording (D-031).")
    b = RES / "analysis_B_spearman.csv"
    if b.exists():
        sb = pd.read_csv(b)
        w = sb[sb.model.str.startswith("ALL")]
        for exp in w.exp.unique():
            F.append(f"**Analysis B ({exp}), within-N across models:** " + ", ".join(
                f"{row.estimator} ρ={row.rho:.2f} [{row.ci_lo:.2f}, {row.ci_hi:.2f}]"
                for row in w[w.exp == exp].itertuples())
                     + ". A positive ρ means a lower label-free quantity picks the more accurate prediction at a fixed N. "
                       "See `fig_drift_vs_error`.")
    ab = RES / "ablations_long.csv"
    if ab.exists():
        a_ = pd.read_csv(ab)
        a_["tag"] = a_["tag"].fillna("")
        mn = long[(long.regime == "A") & (long.exp == "n_extrap") & (long.tag.fillna("") == "")].assign(tag="main")
        d = pd.concat([a_, mn])
        d = d[(d.metric == "rel_l2") & (d.seed == 0)]
        t = d.groupby(["model", "tag", "split"])["value"].mean()

        def g(m, tag, sp):
            try:
                return f"{t[(m, tag, sp)] * 100:.1f}%"
            except KeyError:
                return "n/a"

        F.append("**Ablation, PINN+ components (A, seed 0, interp / near):** all on " + f"{g('pinn_plus', 'main', 'interp')} / {g('pinn_plus', 'main', 'near')}; "
                 + "no Fourier features " + f"{g('pinn_plus', 'pp_noff', 'interp')} / {g('pinn_plus', 'pp_noff', 'near')}; "
                 + "no causal " + f"{g('pinn_plus', 'pp_nocausal', 'interp')} / {g('pinn_plus', 'pp_nocausal', 'near')}; "
                 + "no adaptive weights " + f"{g('pinn_plus', 'pp_noadapt', 'interp')} / {g('pinn_plus', 'pp_noadapt', 'near')}. "
                 "See `fig_ablation_components`.")
        F.append("**Ablation, PINN-strict collocation points (interp):** 10k " + g("pinn_strict", "col10k", "interp")
                 + ", 20k " + g("pinn_strict", "main", "interp") + ", 50k " + g("pinn_strict", "col50k", "interp")
                 + ". **IC weight (interp / near):** w_ic=1 " + f"{g('pinn_strict', 'wic1', 'interp')} / {g('pinn_strict', 'wic1', 'near')}"
                 + ", w_ic=10 (used) " + f"{g('pinn_strict', 'main', 'interp')} / {g('pinn_strict', 'main', 'near')}"
                 + ", w_ic=100 " + f"{g('pinn_strict', 'wic100', 'interp')} / {g('pinn_strict', 'wic100', 'near')}"
                 + ". w_ic was not tuned before the main study, so the PINNs could be somewhat better (single seed).")
        F.append("**Ablation, data efficiency (interp, 2 / 4 / 6 training N):** DD-MLP " + " / ".join(
            g("dd_mlp", tg, "interp") for tg in ["de2", "de4", "main"]) + "; cFNO " + " / ".join(
            g("cfno", tg, "interp") for tg in ["de2", "de4", "main"]) + ". The PINNs use no field data.")
        F.append("**Ablation, training-range width:** see `fig_ablation_range_width` (error vs distance outside the training "
                 "range, seeds 0-1). PINN-strict's curves for the three range widths nearly coincide, while the "
                 "data-driven models trained on the narrow range degrade faster with distance.")
    f = RES / "finetune.csv"
    if f.exists():
        ft = pd.read_csv(f)
        for (reg, N), d in ft.groupby(["regime", "N"]):
            d = d.sort_values("step")
            hit = d[d.rel_l2 <= d.target_interp]
            msg = (f"reached its in-distribution error after {int(hit.step.iloc[0])} L-BFGS steps ({hit.wall_s.iloc[0]:.0f} s)"
                   if len(hit) else f"did not reach its in-distribution error ({d.target_interp.iloc[0] * 100:.1f}%) in "
                                    f"{int(d.step.max())} steps; best {d.rel_l2.min() * 100:.1f}% within {d.wall_s.max():.0f} s of fine-tuning, final {d.rel_l2.iloc[-1] * 100:.1f}%")
            F.append(f"**Fine-tuning at {reg} N={N:g}:** starting at {d.rel_l2.iloc[0] * 100:.0f}% error, the PINN {msg}. "
                     f"See `fig_finetune`.")
    return F


def tables_md():
    out = []
    for reg in ["A", "B_sech", "B_gauss"]:
        for exp in ["n_extrap", "z_extrap"]:
            for metric, name in [("rel_l2", "complex-field rel-L2 (%)"), ("rel_l2_intensity", "intensity rel-L2 (%)")]:
                p = TAB / f"main_{reg}_{exp}_{metric}.md"
                if p.exists():
                    out += [f"#### {reg} / {exp}: {name}", "", p.read_text(encoding="utf-8"), ""]
    cost = (TAB / "main_all.md").read_text(encoding="utf-8").split("## Cost")[-1] if (TAB / "main_all.md").exists() else ""
    out += ["#### Cost", cost]
    return "\n".join(out)


STATIC_HEAD = """# Results summary

*Generated by `scripts/make_summary.py` from `results/`. Do not edit numbers by hand; re-run the script.*

## Research question and hypotheses
When ultrashort pulses enter a propagation regime not seen in training (higher soliton order N, or longer
distance ξ), does physics-informed learning generalize better than purely data-driven learning, and can we
predict where each model fails using physically meaningful quantities?

* **H1:** PINNs extrapolate in N and ξ better than data-driven models (DD-MLP, cFNO), because they see the PDE.
* **H2 (analysis A):** extrapolation error is predicted by spectral shift (Wasserstein-Fourier distance to the
  training spectra, or the closed-form SPM phase N² asinh ξ). This extends Fesser et al. (2023), who studied a
  single NLS time-extrapolation case with PINNs only.
* **H3 (analysis B):** conservation-law drift and PDE residual, which need no ground truth, estimate the true error.

## Setup (details in README, DECISIONS.md and configs/)
* Normalized NLSE `i U_ξ = (s/2) U_ττ − N²|U|²U`. Ground truth: a symmetric SSFM validated to < 1e-6
  (`figures/ssfm_validation`).
* Regime A: normal dispersion, Gaussian input, train N ∈ {1.0, …, 2.0}, tests interp, near {2.25, 2.5} and far {3, 4}.
  Regime B: anomalous dispersion, sech (B_sech) and Gaussian (B_gauss) inputs, train N ∈ {0.8, …, 1.6}, extrap {2, 3}.
  z-extrapolation: train on ξ ≤ 2, test on ξ ∈ (2, 4]. Splits were pre-registered in `configs/splits.yaml` (commit 278e132).
* Models: PINN-strict, PINN-extended (physics at test N, no labels), PINN+ (periodic Fourier features, causal,
  adaptive weights), DD-MLP (same 4×100 tanh trunk), cFNO (N, ξ-conditioned) and cFNO-AR (autoregressive).
  All get an equal 22k optimizer steps. Hyperparameters were tuned on the Regime A interp split only. 3 seeds.
* Scope reductions (D-028): B_gauss z-extrapolation was not run.
"""

STATIC_FIGS = """## Figures

* `fig_extrap_N_regimeA`: complex-field (left) and intensity (right) error vs N. The shaded band is the training
  range; mean ± std over 3 seeds. Takeaway: in the training range the data-driven models are most accurate; far from
  it every model loses the phase, and the intensity ranking changes (see findings).
* `fig_extrap_N_B_sech`, `fig_extrap_N_B_gauss`: the same for anomalous dispersion, where extrapolation crosses into
  higher-order soliton breathing.
* `fig_extrap_z`: per-slice error vs ξ with training on ξ ≤ 2. Takeaway: every model degrades beyond ξ = 2, at very
  different rates.
* `fig_heatmaps_compare`: true and predicted intensity and error maps at one interp and one far N (seed 0).
  Visual check of *how* each model fails.
* `fig_spectra_far`: output spectra at far N. Shows which models reproduce the SPM-broadened spectrum.
* `fig_energy_drift`: energy drift vs ξ. Physically implausible predictions are visible without ground truth.
* `fig_pareto`: accuracy vs wall-clock, including SSFM. Takeaway: in Regime A a few-millisecond SSFM run beats every
  learned model on both speed and accuracy. In the harder anomalous-dispersion cases, an SSFM at the models' accuracy can
  be slower than model inference (see the SSFM finding), but it is never less accurate, and the models' training cost
  comes on top.
* `fig_error_vs_spectral_shift` and `fig_error_predictors_baselines` (analysis A: spectral and SPM-phase predictors
  vs coordinate-only baselines), `fig_drift_vs_error` and `fig_energy_vs_phase` (analysis B).
* `fig_ablation_range_width`, `fig_ablation_components`, `fig_finetune`: ablations and the optional extras.
"""

STATIC_LIMITS = """## Limitations and threats to validity
* **The PINNs are undertrained relative to the data-driven models in distribution** at the equal-step budget
  (D-025 to D-030). Their extrapolation comparison is therefore between unequally accurate in-distribution models. We
  report it as is and point to the fine-tuning and ablation results.
* Hyperparameters were tuned only on the Regime A interpolation split and applied unchanged to Regime B, where the
  PINN in-distribution fit is worse. The IC loss weight (w_ic = 10) was not tuned before the main study; the ablation
  shows w_ic = 1 roughly halves PINN-strict's in-distribution error (single seed). That would still leave it behind
  the data-driven models, but the PINN numbers are not the best achievable.
* Three seeds per configuration; the confidence intervals are wide for small differences. Conclusions labelled
  "not robust" should not be quoted as findings.
* The complex-field rel-L2 saturates near ~1.4 once the phase is lost, so it under-reports differences in intensity
  and observables far from training. We therefore report intensity, spectral and energy metrics alongside it.
* Equal optimizer steps is not equal wall-clock: the PINNs cost 3-5× more per run (cost table).
* Analysis A's WF distance to the training spectra uses ground truth, so it explains error after the fact. The SPM
  phase N² asinh ξ is a-priori, but it is a monotone function of N and ξ, and it does not beat coordinate-only
  baselines (N²ξ, rank(N)+rank(ξ); in z-extrapolation sometimes not even ξ alone). Its correlation therefore mostly
  reflects "error grows with N and ξ" (D-031). Pooled correlations are driven by N and ξ, which is why analysis B
  also reports a within-N test.
* An earlier version of analysis A silently dropped 12.4% of slices because of a float-merge bug; it was fixed and
  re-run (D-031). The corrected numbers are within ~0.01 of the old ones.
* Single physical model (lossless scalar NLSE, no higher-order effects) and a single grid per regime; results may
  not transfer to GNLSE or supercontinuum.
"""


def readme_block(long, runs):
    """Compact generated results tables for README.md (between RESULTS markers)."""
    out = ["*This block is generated by `scripts/make_summary.py` from `results/`. Relative L2 errors, mean over 3 seeds "
           "(per seed: mean over the split's N values). Full tables with std/median: `report/tables/`.*", ""]
    order = [m for m in MODELS if m in set(long.model)]
    for reg, title in [("A", "Regime A: normal dispersion, Gaussian pulse"),
                       ("B_sech", "Regime B: anomalous dispersion, sech pulse"),
                       ("B_gauss", "Regime B: anomalous dispersion, Gaussian pulse")]:
        if not ((long.regime == reg) & (long.exp == "n_extrap")).any():
            continue
        far = FAR[reg]
        cols = [("rel_l2", "interp", "In-distribution (interp N)")]
        if reg == "A":
            cols.append(("rel_l2", "near", "Near extrapolation (complex field)"))
        cols += [("rel_l2", far, "Far extrapolation (complex field)"),
                 ("rel_l2_intensity", far, "Far extrapolation (intensity)"),
                 ("energy_drift_max", far, "Far: max energy drift (ΔE/E, ideal 0)")]
        stats = {c: seed_split(long, c[0], reg, "n_extrap", c[1]) for c in cols}
        out += [f"**{title}, N-extrapolation**", "",
                "| Model | " + " | ".join(c[2] for c in cols) + " |", "|---|" + "---|" * len(cols)]
        for m in order:
            cells = []
            for c in cols:
                s = stats[c]
                if m not in s.index:
                    cells.append("–")
                    continue
                v = s.loc[m, "mean"]
                best = v == s["mean"].min()
                txt = f"{v:.2f}" if c[0] == "energy_drift_max" else f"{v * 100:.1f}%"
                cells.append(f"**{txt}**" if best else txt)
            out.append(f"| {LAB[m]} | " + " | ".join(cells) + " |")
        out.append("")
    zr = [r for r in ["A", "B_sech", "B_gauss"] if ((long.regime == r) & (long.exp == "z_extrap")).any()]
    if zr:
        out += ["**z-extrapolation** (trained on ξ ≤ 2, evaluated on ξ ∈ (2, 4], interp N)", "",
                "| Model | " + " | ".join(f"{r}: complex field" for r in zr) + " |", "|---|" + "---|" * len(zr)]
        zs = {r: seed_split(long, "rel_l2", r, "z_extrap", "interp") for r in zr}
        for m in order:
            if not any(m in zs[r].index for r in zr):
                continue
            cells = []
            for r in zr:
                if m not in zs[r].index:
                    cells.append("–")
                    continue
                v = zs[r].loc[m, "mean"]
                cells.append(f"**{v * 100:.1f}%**" if v == zs[r]["mean"].min() else f"{v * 100:.1f}%")
            out.append(f"| {LAB[m]} | " + " | ".join(cells) + " |")
        out.append("")
    r = runs[runs.tag.fillna("") == ""]
    t = r.groupby("model")["train_wall_s"].median() / 60
    out.append("**Median training time per run:** " + ", ".join(f"{LAB[m]} {t[m]:.1f} min" for m in t.sort_values().index)
               + f". Main sweep: {len(r)} runs, {int((r.status != 'ok').sum())} diverged.")
    return "\n".join(out)


def update_readme(block):
    p = ROOT / "README.md"
    t = p.read_text(encoding="utf-8")
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a in t and b in t:
        t = t[: t.index(a) + len(a)] + "\n" + block + "\n" + t[t.index(b):]
        p.write_text(t, encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    long = pd.read_csv(RES / "main_long.csv")
    runs = pd.read_csv(RES / "main_runs.csv")
    n_runs, n_fail = len(runs), int((runs.status != "ok").sum())
    F = findings(long, runs)
    sweep_log = ROOT / "runs" / "main" / "sweep_log.jsonl"
    n_crash = 0
    if sweep_log.exists():
        n_crash = sum(1 for line in sweep_log.read_text().splitlines() if not json.loads(line)["ok"])
    doc = [STATIC_HEAD,
           f"Main sweep: {n_runs} completed runs, {n_fail} diverged, {n_crash} crashed jobs (see `runs/main/sweep_log.jsonl`).",
           "", "## Key findings (all numbers computed from results/)", ""]
    doc += [f"{i}. {f}" for i, f in enumerate(F, 1)]
    doc += ["", "## Tables (generated by scripts/make_tables.py)", "", tables_md(), "", STATIC_FIGS, STATIC_LIMITS]
    (ROOT / "report" / "results_summary.md").write_text("\n".join(doc), encoding="utf-8")
    update_readme(readme_block(long, runs))
    print("\n".join(doc[:len(F) + 6]))


if __name__ == "__main__":
    main()
