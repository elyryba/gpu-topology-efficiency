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
    not the earlier pre-fix count of 51 -- see the ACCOUNTING section below
    for the full trace), to show how much the headline result depends on
    the categorical assumption vs. direct evidence
  - an ACCOUNTING section tracing all 92 sub-rack rows end to end (every
    row matches a mining entry; high + cap-only = 92 with no remainder)
  - a LEVERAGE AUDIT on the cap-only-dropped sensitivity fit: the top 10
    Cook's-distance rows, with their mined evidence_string where
    applicable, so an outsized -0.338 doesn't get taken at face value
    without checking what's actually driving it

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


def accounting_lines(df, mined):
    """Trace every one of the 92 GB200/GB300 sub-rack rows (total_gpus<72)
    end to end: does it match a mining entry, and at what confidence.
    high + cap_only must equal the sub-rack row count with no remainder."""
    subrack = df[df["gen"].isin(["GB200", "GB300"]) & (df["total_gpus"] < 72)]
    merged = subrack.merge(mined[["repo", "org", "system_id", "confidence"]],
                          on=["repo", "org", "system_id"], how="left", indicator=True)
    n_total = len(subrack)
    n_unmatched = int((merged["_merge"] == "left_only").sum())
    counts = merged["confidence"].value_counts(dropna=False)
    n_high = int(counts.get("high", 0))
    n_medium = int(counts.get("medium", 0))
    n_low = int(counts.get("low", 0))
    n_cap_only = n_total - n_high

    return [
        "ACCOUNTING: all 92 GB200/GB300 sub-rack rows (total_gpus < 72), traced end to end",
        f"  sub-rack rows in this analysis sample: {n_total}",
        f"  matched to a mining entry: {n_total - n_unmatched}  "
        f"(unmatched -- no mining entry at all: {n_unmatched})",
        f"  high confidence:   {n_high}",
        f"  medium confidence: {n_medium}",
        f"  low confidence:    {n_low}",
        f"  cap-only (medium + low + unmatched, i.e. everything non-high): {n_cap_only}",
        f"  check: high + cap-only = {n_high} + {n_cap_only} = {n_high + n_cap_only} "
        f"(must equal {n_total}: {'OK' if n_high + n_cap_only == n_total else 'MISMATCH'})",
        "",
        "Note: an earlier mining pass (commit cbccc13, before the field-coverage and",
        "gen-consistency fixes in commit 4681925) reported 41 high / 51 cap-only for",
        "this same 92-row set. That number is superseded by the fix -- do not cite it;",
        "the counts above are the current, reviewed state.",
        "",
    ], n_high, n_cap_only


def main():
    df = load_clean(DATASET_CSV)
    mined_full = pd.read_csv(MINED_CSV)
    df = build_mined_domain(df)
    n_overridden = int(df["inferred_domain"].notna().sum())

    lines = [
        "REFIT WITH MINED NVLINK DOMAINS",
        f"n={len(df)} | rows with high-confidence mined domain override: {n_overridden}",
        "",
    ]

    acct_lines, acct_high, acct_cap_only = accounting_lines(df, mined_full)
    lines += acct_lines

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
    assert n_cap_only == acct_cap_only, (
        f"cap-only count mismatch: sensitivity-fit computation gives {n_cap_only}, "
        f"accounting section gives {acct_cap_only}")
    df_dropped = df[~is_cap_only].reset_index(drop=True)
    dropped = premium_fit(df_dropped, "log_domain_mined")

    lines += [
        f"SENSITIVITY: drop the {n_cap_only} cap-only sub-rack rows entirely "
        f"(no high-confidence evidence -- pure categorical assumption)",
        f"  n={len(df_dropped)}  premium={dropped.params['log_domain_mined']:+.4f}  "
        f"se={dropped.bse['log_domain_mined']:.4f}  "
        f"p={dropped.pvalues['log_domain_mined']:.4g}",
        "  (shows how much the headline premium depends on rows where we still "
        "have to trust min(total_gpus,72) rather than direct submission evidence)",
        "",
    ]

    # LEVERAGE AUDIT: is the sensitivity number actually driven by mined
    # evidence, or by a handful of unrelated high-leverage rows? Cook's
    # distance depends only on the design matrix/residuals (not on the SE
    # covariance type), so compute it off a plain (non-clustered) OLS fit
    # on the same spec, matching topology_common.robustness()'s convention.
    base_dropped = premium_fit(df_dropped, "log_domain_mined", cluster=False)
    cooks = base_dropped.get_influence().cooks_distance[0]
    df_dropped = df_dropped.copy()
    df_dropped["cooks_d"] = cooks
    top10 = df_dropped.sort_values("cooks_d", ascending=False).head(10)

    mined_lookup = {(r.repo, r.org, r.system_id): r
                    for r in mined_full.itertuples()}

    lines.append(f"LEVERAGE AUDIT: top 10 Cook's-distance rows in the "
                f"cap-only-dropped fit (n={len(df_dropped)})")
    lines.append("(checks whether the -0.338 sensitivity number is actually "
                "driven by mined evidence, or by unrelated outliers)")
    n_mined_in_top10 = 0
    for r in top10.itertuples():
        key = (r.repo, r.org, r.system_id)
        ev = mined_lookup.get(key)
        if ev is not None and r.gen in ("GB200", "GB300"):
            n_mined_in_top10 += 1
            ev_str = (f"[mined, confidence={ev.confidence}, "
                     f"inferred_domain={ev.inferred_domain}] {ev.evidence_string}")
        else:
            ev_str = ("(not a mined GB200/GB300 evidence row -- generic "
                     "FE/scale leverage point, unrelated to domain mining)")
        lines.append(f"  cooks_d={r.cooks_d:.4f}  {r.org}/{r.system_id} "
                     f"({r.gen}, total_gpus={r.total_gpus}, model={r.model})")
        lines.append(f"    evidence: {ev_str}")
    lines += [
        "",
        f"CAVEAT: only {n_mined_in_top10} of the top 10 highest-leverage rows is even a "
        "mined GB200/GB300 evidence row (HPE GB300 ngpu72, and its mined domain "
        "[72] matches what the categorical cap already assumed at total_gpus=72 "
        "-- not new information). The other 9 are generic thin-generation "
        "leverage points (single-system tinycorp/Dell/TTA/JuniperNetworks/"
        "Ailiverse/Fujitsu/NVIDIA submissions in MI300X, RTX, A100, L40S) with "
        "no connection to NVLink domain mining at all.",
        "The -0.338 sensitivity figure should NOT be read as 'strengthened "
        "because we now trust the evidence more' -- dropping the 38 cap-only "
        "rows changes which thin generations dominate the fixed effects, and "
        "that recomposition, not the mined domain values, is what is mostly "
        "moving this number. Treat -0.338 as a sensitivity bound, not a "
        "second confirmation of the mined-augmented headline (-0.131).",
    ]

    out = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(out)
    print(out)
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
