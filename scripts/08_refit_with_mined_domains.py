"""
08_refit_with_mined_domains.py

Rebuilds the true_domain feature using mined high-confidence NVLink domain
evidence (data/nvl_config_mined.csv, see 07_mine_nvl_configs.py) where
available, falling back to the existing categorical min(total_gpus, 72) cap
otherwise. Medium/low-confidence mined values are never used in the
headline spec -- see 07's docstring for why (Nebius GB300 "NVL4" tray
trap, gen-inconsistent filename fallbacks).

Refits the topology-premium spec from 04_rigor_checks.py:
    log_time ~ log_total_gpus + log_domain_size + model FE + gen FE
with org-clustered SEs, and reports:
  - the new premium next to the pooled v2 baseline (-0.0961, p=0.0259,
    04_rigor_checks.py / CLAUDE.md)
  - leave-one-generation-out robustness on the new feature
  - a sensitivity fit dropping every row without high-confidence evidence
    (cap-only rows: 38 of 92 sub-rack rows as of the reviewed mining pass,
    not the earlier pre-fix count of 51), to show how much the headline
    result depends on the categorical assumption vs. direct evidence

Does not modify topology_common.py, 03/04, or the dataset CSV -- this is
an independent refit for review.

Input:  data/mlperf_topology_dataset.csv, data/nvl_config_mined.csv
Output: results/refit_mined_domains.txt
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
                      "refit_mined_domains.txt")


def premium_fit(df, domain_col, cluster=True):
    """log_time ~ log_gpus + domain_col + model FE + gen FE (identical
    spec to 04_rigor_checks.py's premium_fit)."""
    X = pd.concat([df[["log_gpus", domain_col]],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    m = sm.OLS(y, X)
    if cluster:
        return m.fit(cov_type="cluster", cov_kwds={"groups": df["org"]})
    return m.fit()


def build_mined_domain(df):
    """Override true_domain with mined high-confidence evidence where
    available; fall back to the existing categorical-cap true_domain
    otherwise. Only affects GB200/GB300 rows -- mining only covers those
    generations, and only 'high' confidence rows ever carry a value in
    inferred_domain (medium/low/none are always blank by construction in
    07_mine_nvl_configs.py)."""
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    merged = df.merge(high, on=["repo", "org", "system_id"], how="left")
    merged["true_domain_mined"] = merged["inferred_domain"].fillna(merged["true_domain"])
    merged["n_domains_mined"] = np.maximum(
        merged["total_gpus"] / merged["true_domain_mined"], 1.0)
    merged["log_domain_mined"] = np.log(merged["true_domain_mined"])
    merged["log_n_domains_mined"] = np.log(merged["n_domains_mined"])
    return merged


def main():
    df = load_clean(DATASET_CSV)
    df = build_mined_domain(df)
    n_overridden = int(df["inferred_domain"].notna().sum())

    lines = [
        "REFIT WITH MINED NVLINK DOMAINS",
        f"n={len(df)} | rows with high-confidence mined domain override: {n_overridden}",
        "",
    ]

    old = premium_fit(df, "log_domain")          # existing categorical-cap feature
    new = premium_fit(df, "log_domain_mined")    # mined-augmented feature

    lines += [
        "TOPOLOGY PREMIUM: categorical cap vs. mined-augmented (org-clustered SEs)",
        f"  categorical cap (baseline)  coef={old.params['log_domain']:+.4f}  "
        f"se={old.bse['log_domain']:.4f}  p={old.pvalues['log_domain']:.4g}  "
        f"(reference: -0.0961, p=0.0259 per CLAUDE.md / 04_rigor_checks.py)",
        f"  mined-augmented             coef={new.params['log_domain_mined']:+.4f}  "
        f"se={new.bse['log_domain_mined']:.4f}  p={new.pvalues['log_domain_mined']:.4g}",
        f"  R^2 (mined-augmented): {new.rsquared:.4f}",
        "",
    ]

    lines.append("LEAVE-ONE-GENERATION-OUT (mined-augmented feature)")
    for g in sorted(df["gen"].unique()):
        d = df[df["gen"] != g]
        if d["gen"].nunique() < 3:
            continue
        r = premium_fit(d, "log_domain_mined")
        lines.append(f"  drop {g:10s} n={len(d):4d}  "
                     f"premium={r.params['log_domain_mined']:+.4f} "
                     f"(p={r.pvalues['log_domain_mined']:.4g})")
    lines.append("")

    is_subrack = df["gen"].isin(["GB200", "GB300"]) & (df["total_gpus"] < 72)
    is_cap_only = is_subrack & df["inferred_domain"].isna()
    n_cap_only = int(is_cap_only.sum())
    df_dropped = df[~is_cap_only]
    dropped = premium_fit(df_dropped, "log_domain_mined")

    lines += [
        f"SENSITIVITY: drop the {n_cap_only} cap-only sub-rack rows entirely "
        f"(no high-confidence evidence -- pure categorical assumption)",
        f"  n={len(df_dropped)}  premium={dropped.params['log_domain_mined']:+.4f}  "
        f"se={dropped.bse['log_domain_mined']:.4f}  "
        f"p={dropped.pvalues['log_domain_mined']:.4g}",
        "  (shows how much the headline premium depends on rows where we still "
        "have to trust min(total_gpus,72) rather than direct submission evidence)",
    ]

    out = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(out)
    print(out)
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
