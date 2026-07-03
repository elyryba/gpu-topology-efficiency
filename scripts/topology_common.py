"""
topology_common.py

Shared data-cleaning and model-fitting logic used by both the baseline
(02) and corrected (03) topology-efficiency models. Keeping this in one
place guarantees the v1/v2 comparison is apples-to-apples: identical
sample, identical filters, identical fixed effects — the ONLY thing
that differs between the two models is the domain-size feature.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

MIN_OBS_PER_GROUP = 8    # min submissions to keep a model/generation category
MAX_RUN_TO_RUN_CV = 0.5  # drop submissions whose repeated runs disagree > 50%

# Order matters: more specific tags first (e.g. GB300 before B300,
# MI355X before MI35... substring collisions).
GEN_TAGS = ["GB300", "B300", "GB200", "B200", "H200", "H100", "A100",
            "MI355X", "MI350X", "MI325X", "MI300X",
            "TPU-v5p", "TPU-trillium", "GH200", "L40S", "A10G", "RTX"]

# Generations whose NVLink fabric spans the whole rack (72-GPU domain),
# not the physical server. MLPerf's "gpus_per_node" schema reports the
# server boundary (uniformly 4 for these), which understates the true
# scale-up domain. Source: NVIDIA GB200/GB300 NVL72 official specs and
# DGX GB Rack Scale Systems User Guide ("a 72-GPU NVIDIA NVLink domain
# that acts as a single, massive GPU"), corroborated by HPE / Lenovo /
# Supermicro OEM datasheets.
RACK_SCALE_GENS = {"GB200", "GB300"}
NVL72_DOMAIN = 72


def gen_bucket(accel_name):
    if pd.isna(accel_name):
        return "unknown"
    s = str(accel_name)
    for tag in GEN_TAGS:
        if tag in s:
            return tag
    return "other"


def load_clean(path):
    """Load the extracted MLPerf dataset and apply the data-quality
    filters described in the README. Returns the analysis sample."""
    df = pd.read_csv(path)

    df = df.dropna(subset=["total_gpus", "time_to_train_s",
                           "gpus_per_node", "nodes"])
    df = df[(df["total_gpus"] > 0) & (df["gpus_per_node"] > 0)
            & (df["nodes"] > 0) & (df["time_to_train_s"] > 0)]

    # Drop noisy submissions: repeated runs disagreeing by >= 50% reflect
    # configuration/measurement problems, not hardware differences.
    df = df[(df["run_to_run_cv"].isna())
            | (df["run_to_run_cv"] < MAX_RUN_TO_RUN_CV)]

    df["gen"] = df["accelerator"].apply(gen_bucket)
    df = df[~df["gen"].isin(["other", "unknown"])]

    # Thin categories make fixed-effect coefficients unreliable.
    model_counts = df["model"].value_counts()
    df = df[df["model"].isin(model_counts[model_counts >= MIN_OBS_PER_GROUP].index)]
    gen_counts = df["gen"].value_counts()
    df = df[df["gen"].isin(gen_counts[gen_counts >= MIN_OBS_PER_GROUP].index)]

    df["log_nodes"] = np.log(df["nodes"])
    df["log_gpus"] = np.log(df["total_gpus"])
    df["log_time"] = np.log(df["time_to_train_s"])
    df["log_gpn"] = np.log(df["gpus_per_node"])  # v1 proxy

    # v2 corrected feature: documented NVLink domain size.
    # For rack-scale GB200/GB300, the true scale-up domain is the NVL72
    # rack fabric, capped by how many GPUs the submission actually used.
    # For every other generation, NVLink genuinely stops at the server
    # boundary, so gpus_per_node is already correct.
    df["true_domain"] = np.where(
        df["gen"].isin(RACK_SCALE_GENS),
        np.minimum(df["total_gpus"], NVL72_DOMAIN),
        df["gpus_per_node"],
    )
    # Number of domains that must talk over the slower fabric.
    df["n_domains"] = np.maximum(df["total_gpus"] / df["true_domain"], 1.0)
    df["log_domain"] = np.log(df["true_domain"])
    df["log_n_domains"] = np.log(df["n_domains"])

    return df.reset_index(drop=True)


def fit_model(df, size_col="log_gpn", count_col="log_nodes"):
    """OLS: log(time) ~ domain-size + domain-count + model FE + gen FE."""
    X = df[[size_col, count_col]].copy()
    X = pd.concat([X,
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    return sm.OLS(y, X).fit()


def vif_report(df, cols):
    Xv = sm.add_constant(df[list(cols)].astype(float))
    return {c: round(variance_inflation_factor(Xv.values, i), 3)
            for i, c in enumerate(Xv.columns) if c != "const"}


def robustness(df, size_col, count_col):
    """Leave-one-model-out, leave-one-generation-out, Cook's-distance
    outlier check. Returns report lines."""
    lines = ["LEAVE-ONE-MODEL-OUT"]
    for m in sorted(df["model"].unique()):
        d = df[df["model"] != m]
        if d["model"].nunique() < 3:
            continue
        r = fit_model(d, size_col, count_col)
        lines.append(f"drop {m:20s} n={len(d):4d}  "
                     f"{size_col}={r.params[size_col]:.3f} (p={r.pvalues[size_col]:.4f})  "
                     f"{count_col}={r.params[count_col]:.3f} (p={r.pvalues[count_col]:.4f})")

    lines.append("\nLEAVE-ONE-GENERATION-OUT")
    for g in sorted(df["gen"].unique()):
        d = df[df["gen"] != g]
        if d["gen"].nunique() < 3:
            continue
        r = fit_model(d, size_col, count_col)
        lines.append(f"drop {g:10s} n={len(d):4d}  "
                     f"{size_col}={r.params[size_col]:.3f} (p={r.pvalues[size_col]:.4f})  "
                     f"{count_col}={r.params[count_col]:.3f} (p={r.pvalues[count_col]:.4f})")

    base = fit_model(df, size_col, count_col)
    infl = base.get_influence()
    cooks = infl.cooks_distance[0]
    cutoff = np.quantile(cooks, 0.99)
    d2 = df[cooks <= cutoff]
    m2 = fit_model(d2, size_col, count_col)
    lines.append(f"\nOUTLIER CHECK: after dropping top 1% Cook's-distance points (n={len(d2)})")
    lines.append(f"{size_col}={m2.params[size_col]:.3f} (p={m2.pvalues[size_col]:.5f})  "
                 f"{count_col}={m2.params[count_col]:.3f} (p={m2.pvalues[count_col]:.5f})  "
                 f"R2={m2.rsquared:.3f}")
    return lines
