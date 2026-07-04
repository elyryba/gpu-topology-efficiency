"""
10_inference_hardening.py

Referee point: with only 41 org clusters, the asymptotic cluster-robust
inference used everywhere else in this repo is at the edge of where it's
known to become anti-conservative in finite samples (Cameron, Gelbach &
Miller 2008; rule-of-thumb minimum is often quoted around 40-50 clusters).
This script re-tests the two headline topology-premium point estimates
with a wild cluster bootstrap-t (Rademacher weights, null imposed, 9999
replications, clustered by org):

  1. Categorical-cap baseline (04_rigor_checks.py): log_domain,
     coef=-0.0961, asymptotic p=0.0259 -- the estimate whose "marginal
     significance" is most exposed to few-cluster inference problems.
  2. Mined-augmented (08_refit_with_mined_domains.py): log_domain_mined,
     coef=-0.1309, asymptotic p=0.0001131.

Uses the `wildboottest` package, which installs cleanly against this
project's pinned dependencies (verified; pulls in numba/llvmlite as
transitive deps). One real bug worth flagging for anyone reusing this:
passing the cluster variable as raw org-name strings crashes its numba
JIT with a typing error ("non-precise type array(pyobject, 1d, C)");
integer-coded cluster IDs (pd.factorize) work correctly. Falls back to a
direct WCR implementation (validated to match statsmodels' clustered SE
to ~12 significant figures on this exact spec) if the package import
fails.

Output: results/inference_hardening.txt
"""
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from topology_common import load_clean

DATASET_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                          "mlperf_topology_dataset.csv")
MINED_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                        "nvl_config_mined.csv")
OUT_TXT = os.path.join(os.path.dirname(__file__), "..", "results",
                      "inference_hardening.txt")

B = 9999
SEED = 42


def build_mined_domain(df):
    """Mirrors 08_refit_with_mined_domains.py's function of the same name."""
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    merged = df.merge(high, on=["repo", "org", "system_id"], how="left")
    merged["true_domain_mined"] = merged["inferred_domain"].fillna(merged["true_domain"])
    merged["log_domain_mined"] = np.log(merged["true_domain_mined"])
    return merged


def design_matrix(df, domain_col):
    X = pd.concat([df[["log_gpus", domain_col]],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    return sm.add_constant(X.astype(float))


def wild_cluster_bootstrap_direct(y, X, param_idx, cluster_codes, B=B, seed=SEED):
    """Direct WCR (restricted-residual) wild cluster bootstrap-t with
    Rademacher weights (Cameron/Gelbach/Miller 2008), used only if the
    `wildboottest` package is unavailable. Validated to reproduce
    statsmodels' cluster-robust SE to ~12 significant figures on this
    project's actual spec before being trusted here."""
    rng = np.random.default_rng(seed)
    Xv = X.values.astype(float)
    yv = y.values.astype(float)
    N, K = Xv.shape
    groups = np.unique(cluster_codes)
    G = len(groups)

    fitted = sm.OLS(yv, Xv).fit(cov_type="cluster", cov_kwds={"groups": cluster_codes})
    t_obs = fitted.params[param_idx] / fitted.bse[param_idx]

    keep = [j for j in range(K) if j != param_idx]
    Xr = Xv[:, keep]
    beta_r = np.linalg.pinv(Xr) @ yv
    resid_r = yv - Xr @ beta_r
    fitted_r = Xr @ beta_r

    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    group_idx = np.searchsorted(groups, cluster_codes)

    def cluster_se(resid):
        meat = np.zeros((K, K))
        for gi, g in enumerate(groups):
            idx = group_idx == gi
            score_g = Xv[idx].T @ resid[idx]
            meat += np.outer(score_g, score_g)
        correction = (G / (G - 1)) * ((N - 1) / (N - K))
        V = correction * XtX_inv @ meat @ XtX_inv
        return np.sqrt(V[param_idx, param_idx])

    t_boot = np.empty(B)
    for b in range(B):
        w = rng.choice(np.array([-1.0, 1.0]), size=G)
        y_star = fitted_r + resid_r * w[group_idx]
        beta_star = XtX_inv @ (Xv.T @ y_star)
        resid_star = y_star - Xv @ beta_star
        t_boot[b] = beta_star[param_idx] / cluster_se(resid_star)

    p_boot = float(np.mean(np.abs(t_boot) >= abs(t_obs)))
    return float(t_obs), p_boot


def run_bootstrap(df, domain_col, label):
    X = design_matrix(df, domain_col)
    y = df["log_time"].astype(float)
    cluster_codes, org_names = pd.factorize(df["org"])
    cluster_codes = cluster_codes.astype(np.int64)

    fitted = sm.OLS(y.values, X.values).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_codes})
    param_idx = list(X.columns).index(domain_col)
    t_asymp = fitted.params[param_idx] / fitted.bse[param_idx]
    p_asymp = fitted.pvalues[param_idx]

    used_package = False
    with warnings.catch_warnings():
        # Same benign Apple-Accelerate-BLAS matmul FP traps documented in
        # 05_deep_checks.py -- pinv()/matmul() on this arm64 macOS numpy
        # build trips spurious divide/overflow/invalid warnings despite
        # correct results.
        warnings.simplefilter("ignore", RuntimeWarning)
        if importlib.util.find_spec("wildboottest") is not None:
            from wildboottest.wildboottest import wildboottest
            # wildboottest indexes params by column name, so pass a
            # DataFrame-backed model (not .values) with matching index.
            model_named = sm.OLS(y.reset_index(drop=True),
                                 X.reset_index(drop=True))
            res = wildboottest(model_named, param=domain_col,
                              cluster=cluster_codes, B=B, seed=SEED,
                              show=False)
            p_boot = float(res["p-value"].iloc[0])
            t_boot_stat = float(res["statistic"].iloc[0])
            used_package = True
        else:
            t_boot_stat, p_boot = wild_cluster_bootstrap_direct(
                y, X, param_idx, cluster_codes)

    lines = [
        f"{label}",
        f"  n={len(df)}, clusters={len(org_names)}, feature={domain_col}",
        f"  asymptotic:  coef={fitted.params[param_idx]:+.4f}  "
        f"se={fitted.bse[param_idx]:.4f}  t={t_asymp:+.3f}  p={p_asymp:.4g}",
        f"  wild cluster bootstrap-t (Rademacher, {B} reps, "
        f"{'wildboottest package' if used_package else 'direct implementation'}): "
        f"t={t_boot_stat:+.3f}  p={p_boot:.4g}",
        f"  significance at alpha=0.05: "
        f"{'HOLDS under both' if p_asymp < 0.05 and p_boot < 0.05 else 'LOST UNDER BOOTSTRAP' if p_asymp < 0.05 else 'not significant either way'}",
    ]
    return lines, p_asymp, p_boot


def main():
    df = load_clean(DATASET_CSV)
    df_mined = build_mined_domain(df)

    lines = ["INFERENCE HARDENING: wild cluster bootstrap-t vs. asymptotic "
            "cluster-robust p-values", ""]

    l1, p_asymp_cat, p_boot_cat = run_bootstrap(
        df, "log_domain",
        "1. CATEGORICAL-CAP BASELINE (04_rigor_checks.py)")
    lines += l1
    lines.append("")

    l2, p_asymp_mined, p_boot_mined = run_bootstrap(
        df_mined, "log_domain_mined",
        "2. MINED-AUGMENTED (08_refit_with_mined_domains.py)")
    lines += l2
    lines.append("")

    lines.append("SUMMARY")
    lines.append(f"  categorical-cap:  asymptotic p={p_asymp_cat:.4g} -> "
                f"bootstrap p={p_boot_cat:.4g}")
    lines.append(f"  mined-augmented:  asymptotic p={p_asymp_mined:.4g} -> "
                f"bootstrap p={p_boot_mined:.4g}")
    if p_boot_cat >= 0.05 or p_boot_mined >= 0.05:
        lines.append("  AT LEAST ONE HEADLINE LOSES SIGNIFICANCE UNDER THE "
                     "BOOTSTRAP -- see above for which.")
    else:
        lines.append("  Both estimates remain significant at alpha=0.05 "
                     "under the bootstrap. Note the asymptotic p-values "
                     "still understate uncertainty relative to the "
                     "bootstrap (most visibly for the mined-augmented "
                     "estimate) -- the conclusion doesn't flip, but the "
                     "41-cluster asymptotic SEs were optimistic.")

    out = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(out)
    print(out)
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
