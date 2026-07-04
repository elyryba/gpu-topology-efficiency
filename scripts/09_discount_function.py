"""
09_discount_function.py

CLAUDE.md open-work item 5: a continuous, monotone topology discount
function, workload-conditional, validated on a held-out temporal window.

Domain feature: the evidence-augmented true_domain from
08_refit_with_mined_domains.py (mined high-confidence NVLink domain where
available, categorical min(total_gpus,72) cap otherwise) -- build_mined_domain
below mirrors 08's function of the same name so this script stays
self-contained (same convention as COMM_BOUND being duplicated, not
imported, between 05_deep_checks.py and 06_continuous_bandwidth_model.py).

Model: log_time ~ log_gpus + log_domain_mined + domain_mined_x_comm +
model FE + gen FE, org-clustered SEs. COMM_BOUND is copied verbatim from
06_continuous_bandwidth_model.py. This gives two log-linear curves (a
single coefficient each -- comm-light: log_domain_mined; comm-bound:
log_domain_mined + domain_mined_x_comm), and a log-linear power-law curve
is monotonic by construction as long as its slope has a consistent sign.
Monotonicity is still explicitly ENFORCED (not just assumed): if either
fitted slope came out positive (larger domain predicting MORE time, which
would contradict every prior finding in this repo and violate the
physical prior that more NVLink fabric never hurts), it is clipped to 0
before being used to build the discount table, and the clip is reported
loudly rather than silently applied.

Validation: the same temporal holdout as 05_deep_checks.py section D --
fit on v3.1-v5.0, predict v5.1-v6.0, restricted to models/gens present in
both windows so fixed effects transfer.

Output: results/discount_function.txt, with a table of discount
multipliers (relative to the domain=4 reference point) for domain sizes
4/8/16/36/72, org-clustered uncertainty bands, and an explicit
interpolation/extrapolation flag per cell based on the observed domain
range within each workload class.

Referee follow-up: 11_measured_comm_intensity.py found flux1 is in
COMM_BOUND but shows zero measured TP/PP parallelism anywhere in the
corpus -- its hand-label disagrees with the measured evidence. This
script includes a robustness check: rebuild the whole discount table with
flux1 moved to comm-light and report how much every cell moves. If that
move were material, the corrected table would replace this one as the
headline; empirically it doesn't (see ROBUSTNESS section), so the
COMM_BOUND-based table above remains authoritative and the check is
reported as evidence of insensitivity, not a correction.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from topology_common import load_clean

DATASET_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                          "mlperf_topology_dataset.csv")
MINED_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                        "nvl_config_mined.csv")
OUT_TXT = os.path.join(os.path.dirname(__file__), "..", "results",
                      "discount_function.txt")

# Copied verbatim from 06_continuous_bandwidth_model.py (same set used by
# 05_deep_checks.py section C).
COMM_BOUND = {"gpt3", "llama31_405b", "deepseekv3_671b", "llama2_70b_lora",
              "llama31_8b", "gpt_oss_20b", "flux1"}

DOMAIN_SIZES = [4, 8, 16, 36, 72]
DOMAIN_REF = 4  # smallest table entry; multiplier(DOMAIN_REF) == 1.0 by construction

EARLY_REPOS = ["training_results_v3.1", "training_results_v4.0",
              "training_results_v4.1", "training_results_v5.0"]
LATE_REPOS = ["training_results_v5.1", "training_results_v6.0"]


def build_mined_domain(df):
    """Mirrors 08_refit_with_mined_domains.py's function of the same name:
    override true_domain with mined high-confidence evidence where
    available, categorical cap otherwise."""
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    merged = df.merge(high, on=["repo", "org", "system_id"], how="left")
    merged["true_domain_mined"] = merged["inferred_domain"].fillna(merged["true_domain"])
    merged["log_domain_mined"] = np.log(merged["true_domain_mined"])
    return merged


def fit(df, extra_cols=()):
    """log_time ~ log_gpus + log_domain_mined + extra_cols + model FE + gen
    FE, org-clustered SEs."""
    cols = ["log_gpus", "log_domain_mined"] + list(extra_cols)
    X = pd.concat([df[cols],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["org"]})


def heavy_slope_and_se(r):
    """Coefficient and SE of the comm-bound curve's total slope
    (log_domain_mined + domain_mined_x_comm), propagating the full
    clustered covariance (not just summing SEs -- these two coefficients
    are correlated)."""
    b = r.params["log_domain_mined"] + r.params["domain_mined_x_comm"]
    cov = r.cov_params()
    var = (cov.loc["log_domain_mined", "log_domain_mined"]
          + cov.loc["domain_mined_x_comm", "domain_mined_x_comm"]
          + 2 * cov.loc["log_domain_mined", "domain_mined_x_comm"])
    return b, float(np.sqrt(var))


def enforce_nonpositive(b, label, notes):
    """Monotonicity enforcement: a discount function must never predict
    that a larger NVLink domain increases training time. Clip to 0 (flat,
    no discount) if the fit ever disagrees, and say so loudly."""
    if b > 0:
        notes.append(f"MONOTONICITY CLIP: fitted {label} slope was "
                     f"{b:+.4f} (positive) -- clipped to 0.0000 so the "
                     f"discount table never predicts a larger domain "
                     f"hurting throughput.")
        return 0.0
    return b


def multiplier_band(b_hat, se, domain, ref=DOMAIN_REF, z=1.96):
    """Discount multiplier (time-to-train relative to the domain=ref
    baseline) and its [lo, hi] band from the log-linear coefficient's
    sampling distribution (delta method: multiplier is a monotonic
    transform of b, so the band's endpoints map directly)."""
    logratio = np.log(domain / ref)
    point = float(np.exp(b_hat * logratio))
    lo = float(np.exp((b_hat - z * se) * logratio))
    hi = float(np.exp((b_hat + z * se) * logratio))
    return point, min(lo, hi), max(lo, hi)


def fit_curves(df, comm_set):
    """Fit the two-curve spec for a given comm-bound model set and return
    the monotonicity-enforced slopes/SEs used to build a discount table
    (silently swallows monotonicity-clip notes -- the caller decides
    whether to surface them)."""
    d = df.copy()
    d["comm"] = d["model"].isin(comm_set).astype(float)
    d["domain_mined_x_comm"] = d["log_domain_mined"] * d["comm"]
    r = fit(d, extra_cols=["domain_mined_x_comm"])
    b_light = r.params["log_domain_mined"]
    se_light = r.bse["log_domain_mined"]
    b_heavy, se_heavy = heavy_slope_and_se(r)
    notes = []
    b_light_used = enforce_nonpositive(b_light, "comm-light", notes)
    b_heavy_used = enforce_nonpositive(b_heavy, "comm-bound", notes)
    return b_light_used, se_light, b_heavy_used, se_heavy, notes


def main():
    df = load_clean(DATASET_CSV)
    df = build_mined_domain(df)
    df["comm"] = df["model"].isin(COMM_BOUND).astype(float)
    df["domain_mined_x_comm"] = df["log_domain_mined"] * df["comm"]

    lines = ["TOPOLOGY DISCOUNT FUNCTION (workload-conditional, "
            "evidence-augmented domain feature)"]
    lines.append(f"n={len(df)} | comm-bound rows: {int(df['comm'].sum())} | "
                f"comm-light rows: {int((1 - df['comm']).sum())}")
    lines.append("")

    r = fit(df, extra_cols=["domain_mined_x_comm"])
    b_light = r.params["log_domain_mined"]
    se_light = r.bse["log_domain_mined"]
    b_heavy, se_heavy = heavy_slope_and_se(r)

    lines += [
        "FITTED CURVES (log_time ~ log_gpus + log_domain_mined "
        "+ domain_mined_x_comm + model FE + gen FE, org-clustered SEs)",
        f"  comm-light slope  coef={b_light:+.4f}  se={se_light:.4f}  "
        f"p={r.pvalues['log_domain_mined']:.3g}",
        f"  comm-bound slope  coef={b_heavy:+.4f}  se={se_heavy:.4f}  "
        f"(= log_domain_mined + domain_mined_x_comm, covariance-propagated SE)",
        f"  interaction term  coef={r.params['domain_mined_x_comm']:+.4f}  "
        f"se={r.bse['domain_mined_x_comm']:.4f}  "
        f"p={r.pvalues['domain_mined_x_comm']:.3g}",
        f"  R^2: {r.rsquared:.4f}",
        "",
    ]

    notes = []
    b_light_used = enforce_nonpositive(b_light, "comm-light", notes)
    b_heavy_used = enforce_nonpositive(b_heavy, "comm-bound", notes)
    lines.append("MONOTONICITY ENFORCEMENT")
    if notes:
        lines += [f"  {n}" for n in notes]
    else:
        lines.append("  Both fitted slopes are already <= 0 (larger domain "
                     "predicts less or equal time-to-train) -- no clipping "
                     "needed. Enforcement is active but did not trigger.")
    lines.append("")

    # ---------- temporal holdout validation (mirrors 05's section D) ----------
    early = df[df["repo"].isin(EARLY_REPOS)]
    late = df[df["repo"].isin(LATE_REPOS)]
    shared_m = set(early["model"]) & set(late["model"])
    shared_g = set(early["gen"]) & set(late["gen"])
    e = early[early["model"].isin(shared_m) & early["gen"].isin(shared_g)]
    l = late[late["model"].isin(shared_m) & late["gen"].isin(shared_g)]

    lines.append("TEMPORAL HOLDOUT — fit v3.1-v5.0, predict v5.1-v6.0 "
                "(shared models/gens only, so FEs transfer)")
    if len(e) > 50 and len(l) > 50:
        re_ = fit(e, extra_cols=["domain_mined_x_comm"])
        rl = fit(l, extra_cols=["domain_mined_x_comm"])
        b_light_e = re_.params["log_domain_mined"]
        b_heavy_e, _ = heavy_slope_and_se(re_)
        b_light_l = rl.params["log_domain_mined"]
        b_heavy_l, _ = heavy_slope_and_se(rl)
        lines.append(f"  early n={len(e)}: comm-light={b_light_e:+.4f}  "
                     f"comm-bound={b_heavy_e:+.4f}")
        lines.append(f"  late  n={len(l)}: comm-light={b_light_l:+.4f}  "
                     f"comm-bound={b_heavy_l:+.4f}")

        cols = ["log_gpus", "log_domain_mined", "domain_mined_x_comm"]
        Xl = pd.concat([l[cols],
                       pd.get_dummies(l["model"], prefix="model", drop_first=True),
                       pd.get_dummies(l["gen"], prefix="gen", drop_first=True)],
                      axis=1)
        Xl = sm.add_constant(Xl.astype(float)).reindex(
            columns=re_.params.index, fill_value=0.0)
        pred = Xl @ re_.params
        resid = l["log_time"].astype(float) - pred
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((l["log_time"] - l["log_time"].mean()) ** 2).sum())
        oos_r2 = 1 - ss_res / ss_tot
        median_err = np.exp(np.median(np.abs(resid))) - 1
        lines.append(f"  out-of-sample R^2 on late rounds (early-fit "
                     f"coefficients): {oos_r2:.3f}   "
                     f"median |error| = {median_err:.1%} of time-to-train")
    else:
        lines.append(f"  insufficient overlap (early n={len(e)}, late n={len(l)})")
    lines.append("")

    # ---------- discount table ----------
    light_domains = df.loc[df["comm"] == 0, "true_domain_mined"]
    heavy_domains = df.loc[df["comm"] == 1, "true_domain_mined"]
    light_range = (light_domains.min(), light_domains.max())
    heavy_range = (heavy_domains.min(), heavy_domains.max())

    lines.append("DISCOUNT TABLE (time-to-train multiplier relative to "
                f"domain={DOMAIN_REF}; <1.0 = less time = a discount; "
                "org-clustered 95% band)")
    lines.append(f"  observed comm-light domain range: "
                f"[{light_range[0]:.0f}, {light_range[1]:.0f}]")
    lines.append(f"  observed comm-bound domain range: "
                f"[{heavy_range[0]:.0f}, {heavy_range[1]:.0f}]")
    lines.append("")
    band_w = 24  # width of a "0.xxx [0.xxx, 0.xxx]" cell
    lines.append(f"{'domain':>8s}  {'comm-light mult [95% band]':<{band_w}s}  "
                f"{'status':<13s}  {'comm-bound mult [95% band]':<{band_w}s}  {'status':<13s}")
    for d in DOMAIN_SIZES:
        lp, llo, lhi = multiplier_band(b_light_used, se_light, d)
        hp, hlo, hhi = multiplier_band(b_heavy_used, se_heavy, d)
        light_status = ("interpolation" if light_range[0] <= d <= light_range[1]
                        else "extrapolation")
        heavy_status = ("interpolation" if heavy_range[0] <= d <= heavy_range[1]
                        else "extrapolation")
        light_cell = f"{lp:.3f} [{llo:.3f}, {lhi:.3f}]"
        heavy_cell = f"{hp:.3f} [{hlo:.3f}, {hhi:.3f}]"
        lines.append(f"{d:8d}  {light_cell:<{band_w}s}  {light_status:<13s}  "
                     f"{heavy_cell:<{band_w}s}  {heavy_status:<13s}")
    lines.append("")
    lines.append("Interpolation = domain size falls within the observed "
                "range for that workload class in this dataset. "
                "Extrapolation = outside it -- treat those cells as a "
                "model-implied projection, not an empirical finding.")
    lines.append("")

    # ---------- robustness: flux1 label sensitivity ----------
    # 11_measured_comm_intensity.py found flux1 (in COMM_BOUND) shows zero
    # measured TP/PP parallelism anywhere in the corpus -- its hand-label
    # disagrees with the measured evidence. Check how much the table
    # actually depends on that one label.
    comm_bound_noflux = COMM_BOUND - {"flux1"}
    bl_nf, se_l_nf, bh_nf, se_h_nf, notes_nf = fit_curves(df, comm_bound_noflux)

    lines.append("ROBUSTNESS: flux1 moved from comm-bound to comm-light "
                "(per 11_measured_comm_intensity.py's finding that flux1 "
                "has zero measured TP/PP parallelism anywhere in the "
                "corpus)")
    lines.append(f"  comm-light slope  coef={bl_nf:+.4f}  "
                f"(headline: {b_light_used:+.4f})")
    lines.append(f"  comm-bound slope  coef={bh_nf:+.4f}  "
                f"(headline: {b_heavy_used:+.4f})")
    lines.append("")
    lines.append(f"{'domain':>8s}  {'light (headline)':>17s}  "
                f"{'light (no-flux1)':>17s}  {'delta':>8s}  "
                f"{'heavy (headline)':>17s}  {'heavy (no-flux1)':>17s}  "
                f"{'delta':>8s}")
    max_abs_delta = 0.0
    for d in DOMAIN_SIZES:
        lp0, _, _ = multiplier_band(b_light_used, se_light, d)
        hp0, _, _ = multiplier_band(b_heavy_used, se_heavy, d)
        lp1, _, _ = multiplier_band(bl_nf, se_l_nf, d)
        hp1, _, _ = multiplier_band(bh_nf, se_h_nf, d)
        dl, dh = lp1 - lp0, hp1 - hp0
        max_abs_delta = max(max_abs_delta, abs(dl), abs(dh))
        lines.append(f"{d:8d}  {lp0:17.4f}  {lp1:17.4f}  {dl:+8.4f}  "
                     f"{hp0:17.4f}  {hp1:17.4f}  {dh:+8.4f}")
    lines.append("")

    band_widths = []
    for d in DOMAIN_SIZES:
        _, llo, lhi = multiplier_band(b_light_used, se_light, d)
        _, hlo, hhi = multiplier_band(b_heavy_used, se_heavy, d)
        band_widths += [lhi - llo, hhi - hlo]
    max_band_width = max(w for w in band_widths if w > 0) if any(
        w > 0 for w in band_widths) else 1.0

    if max_abs_delta > 0.25 * max_band_width:
        lines.append(f"VERDICT: MATERIAL. Largest cell movement "
                     f"({max_abs_delta:.4f}) exceeds 25% of the largest "
                     f"95% band width ({max_band_width:.4f}) -- the "
                     f"no-flux1 table should be treated as the headline "
                     f"and the table above as superseded.")
    else:
        lines.append(f"VERDICT: NOT MATERIAL. Largest cell movement "
                     f"({max_abs_delta:.4f}) is a small fraction of the "
                     f"largest 95% band width ({max_band_width:.4f}) -- "
                     f"the discount table above is insensitive to "
                     f"flux1's COMM_BOUND label and remains the "
                     f"headline. This is a robustness check, not a "
                     f"correction.")

    out = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(out)
    print(out)
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
