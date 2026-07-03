"""
04_rigor_checks.py

Two upgrades the original analysis needs before anyone cites it:

(1) THE DIRECT HYPOTHESIS TEST. The claim "topology matters at fixed
    total scale" is NOT the log_gpn coefficient itself — it is the
    DIFFERENCE between the domain-size and domain-count elasticities.
    Since log(total_gpus) = log(domain) + log(n_domains), reparametrize:

        log_time ~ b_scale*log(total_gpus) + b_topo*log(domain) + FEs

    b_topo is then exactly the topology premium: the extra speedup from
    reaching the SAME total GPU count via larger NVLink domains. Its
    p-value is the test the README implies but never actually runs
    (comparing -1.02 vs -0.69 by eye is not a test).

(2) CLUSTER-ROBUST STANDARD ERRORS. Submissions from the same
    organization share tuning teams, software stacks, and system
    configs — their errors are not independent. OLS "nonrobust" SEs
    (what the original summary reports) overstate precision. Refit with
    errors clustered by submitting org (42 clusters).

Input:  data/mlperf_topology_dataset.csv
Output: results/rigor_checks.txt
"""
import os
import pandas as pd
import statsmodels.api as sm
from topology_common import load_clean

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                       "mlperf_topology_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def build_X(df, cols):
    X = pd.concat([df[list(cols)],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    return sm.add_constant(X.astype(float))


def premium_fit(df, domain_col, cov=None, groups=None):
    X = build_X(df, ["log_gpus", domain_col])
    y = df["log_time"].astype(float)
    m = sm.OLS(y, X)
    if cov == "cluster":
        return m.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return m.fit()


def main():
    df = load_clean(IN_PATH)
    lines = []

    for label, dcol in [("v1 proxy (gpus_per_node)", "log_gpn"),
                        ("v2 corrected (NVLink domain)", "log_domain")]:
        lines.append(f"\n===== TOPOLOGY PREMIUM — {label} =====")
        lines.append("Spec: log_time ~ log_total_gpus + log_domain_size + model FE + gen FE")
        lines.append("The domain-size coefficient here IS the topology effect at fixed total scale.\n")

        naive = premium_fit(df, dcol)
        lines.append(f"[nonrobust SE]      "
                     f"scale={naive.params['log_gpus']:.4f} "
                     f"(se={naive.bse['log_gpus']:.4f})   "
                     f"topology premium={naive.params[dcol]:.4f} "
                     f"(se={naive.bse[dcol]:.4f}, p={naive.pvalues[dcol]:.2e})")

        clus = premium_fit(df, dcol, cov="cluster", groups=df["org"])
        lines.append(f"[clustered by org]  "
                     f"scale={clus.params['log_gpus']:.4f} "
                     f"(se={clus.bse['log_gpus']:.4f})   "
                     f"topology premium={clus.params[dcol]:.4f} "
                     f"(se={clus.bse[dcol]:.4f}, p={clus.pvalues[dcol]:.2e})")
        lines.append(f"n={len(df)}, clusters={df['org'].nunique()}, R2={naive.rsquared:.4f}")

    lines.append("\nInterpretation: the topology premium is the additional elasticity")
    lines.append("of training time with respect to NVLink-domain size, HOLDING TOTAL")
    lines.append("GPU COUNT FIXED. A premium of -0.10 means doubling domain size at")
    lines.append("constant cluster size cuts time-to-train by ~2^0.10 - 1 = ~7%.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "rigor_checks.txt"), "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
