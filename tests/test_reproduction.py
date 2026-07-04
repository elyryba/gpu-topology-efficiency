"""
test_reproduction.py

Pins down the headline numbers quoted in CLAUDE.md so a dependency bump,
data change, or code edit that silently shifts the analysis sample or the
core coefficients gets caught immediately, rather than being noticed only
when someone eyeballs results/*.txt.

Run: pytest tests/test_reproduction.py
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from topology_common import fit_model, load_clean

DATASET_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                          "mlperf_topology_dataset.csv")
MINED_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                        "nvl_config_mined.csv")
COMM_INTENSITY_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                                 "comm_intensity_mined.csv")
MARKET_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                        "market_prices_snapshot.csv")

# Copied verbatim from 06_continuous_bandwidth_model.py / 09_discount_function.py
COMM_BOUND = {"gpt3", "llama31_405b", "deepseekv3_671b", "llama2_70b_lora",
              "llama31_8b", "gpt_oss_20b", "flux1"}


def _premium_fit(df, domain_col="log_domain"):
    """log_time ~ log_gpus + domain_col + model FE + gen FE, org-clustered
    SEs -- the topology-premium spec from 04_rigor_checks.py."""
    X = pd.concat([df[["log_gpus", domain_col]],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["org"]})


def test_clean_sample_shape():
    df = load_clean(DATASET_CSV)
    assert len(df) == 969
    assert df["model"].nunique() == 15
    assert df["gen"].nunique() == 14


def test_v1_baseline_coefficients():
    df = load_clean(DATASET_CSV)
    v1 = fit_model(df, "log_gpn", "log_nodes")
    assert round(v1.params["log_gpn"], 3) == -1.022
    assert round(v1.params["log_nodes"], 3) == -0.690


def test_v2_corrected_domain_coefficient():
    df = load_clean(DATASET_CSV)
    v2 = fit_model(df, "log_domain", "log_n_domains")
    assert round(v2.params["log_domain"], 3) == -0.781


def test_pooled_topology_premium():
    df = load_clean(DATASET_CSV)
    r = _premium_fit(df)
    assert round(r.params["log_domain"], 2) == -0.10


def test_discount_function_headline_multipliers():
    """Locks the 09_discount_function.py discount-table multipliers at
    domain=72 (relative to the domain=4 baseline) for both workload
    curves, using the evidence-augmented (mined + categorical-cap)
    domain feature."""
    df = load_clean(DATASET_CSV)
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    df = df.merge(high, on=["repo", "org", "system_id"], how="left")
    df["true_domain_mined"] = df["inferred_domain"].fillna(df["true_domain"])
    df["log_domain_mined"] = np.log(df["true_domain_mined"])
    df["comm"] = df["model"].isin(COMM_BOUND).astype(float)
    df["domain_mined_x_comm"] = df["log_domain_mined"] * df["comm"]

    X = pd.concat([df[["log_gpus", "log_domain_mined", "domain_mined_x_comm"]],
                  pd.get_dummies(df["model"], prefix="model", drop_first=True),
                  pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                 axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["org"]})

    b_light = r.params["log_domain_mined"]
    b_heavy = b_light + r.params["domain_mined_x_comm"]

    light_mult_72 = np.exp(b_light * np.log(72 / 4))
    heavy_mult_72 = np.exp(b_heavy * np.log(72 / 4))

    assert round(light_mult_72, 3) == 0.784
    assert round(heavy_mult_72, 3) == 0.627


def test_wild_cluster_bootstrap_holds_significance():
    """Locks the 10_inference_hardening.py wild cluster bootstrap-t
    p-values (Rademacher weights, 9999 reps, seed=42, clustered by org)
    for both headline topology-premium estimates. If this ever fails,
    the referee-point-3 finding -- neither estimate loses significance
    under the bootstrap, though the asymptotic p-values understate
    uncertainty -- has changed and needs re-reporting, not a quiet test
    patch."""
    import warnings

    from wildboottest.wildboottest import wildboottest

    df = load_clean(DATASET_CSV)
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    df_mined = df.merge(high, on=["repo", "org", "system_id"], how="left")
    df_mined["true_domain_mined"] = df_mined["inferred_domain"].fillna(
        df_mined["true_domain"])
    df_mined["log_domain_mined"] = np.log(df_mined["true_domain_mined"])

    def boot_p(d, domain_col):
        X = pd.concat([d[["log_gpus", domain_col]],
                       pd.get_dummies(d["model"], prefix="model", drop_first=True),
                       pd.get_dummies(d["gen"], prefix="gen", drop_first=True)],
                      axis=1)
        X = sm.add_constant(X.astype(float)).reset_index(drop=True)
        y = d["log_time"].astype(float).reset_index(drop=True)
        cluster_codes, _ = pd.factorize(d["org"])
        cluster_codes = cluster_codes.astype(np.int64)
        model = sm.OLS(y, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = wildboottest(model, param=domain_col, cluster=cluster_codes,
                              B=9999, seed=42, show=False)
        return float(res["p-value"].iloc[0])

    p_cat = boot_p(df, "log_domain")
    p_mined = boot_p(df_mined, "log_domain_mined")

    assert round(p_cat, 4) == 0.0284
    assert round(p_mined, 4) == 0.0067
    assert p_cat < 0.05
    assert p_mined < 0.05


def test_measured_comm_intensity_coverage_and_sharpening():
    """Locks 11_measured_comm_intensity.py's coverage breakdown and the
    binary-vs-continuous interaction p-value comparison, using the
    committed data/comm_intensity_mined.csv (that CSV is this script's
    mined-data artifact -- rebuilding it from scratch needs $MLPERF_ROOT,
    same constraint as data/nvl_config_mined.csv/07, so this test doesn't
    re-scan the raw repos)."""
    df = load_clean(DATASET_CSV)
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    df = df.merge(high, on=["repo", "org", "system_id"], how="left")
    df["true_domain_mined"] = df["inferred_domain"].fillna(df["true_domain"])
    df["log_domain_mined"] = np.log(df["true_domain_mined"])

    ci = pd.read_csv(COMM_INTENSITY_CSV)
    df = df.merge(ci[["repo", "org", "system_id", "model", "comm_intensity",
                      "comm_intensity_reason"]],
                 on=["repo", "org", "system_id", "model"], how="left")

    counts = df["comm_intensity_reason"].value_counts()
    assert int(counts["measured"]) == 303
    assert int(counts["assumed_dp_default"]) == 240
    assert int(counts["unmeasured_no_scale_match"]) == 65
    assert int(counts["repo_unavailable"]) == 361

    restricted = df[df["comm_intensity"].notna()].copy()
    assert len(restricted) == 543

    restricted["comm"] = restricted["model"].isin(COMM_BOUND).astype(float)
    restricted["domain_mined_x_comm"] = (restricted["log_domain_mined"]
                                        * restricted["comm"])
    restricted["domain_mined_x_comm_intensity"] = (
        restricted["log_domain_mined"] * restricted["comm_intensity"])

    def fit(d, extra_cols):
        cols = ["log_gpus", "log_domain_mined"] + list(extra_cols)
        X = pd.concat([d[cols],
                       pd.get_dummies(d["model"], prefix="model", drop_first=True),
                       pd.get_dummies(d["gen"], prefix="gen", drop_first=True)],
                      axis=1)
        X = sm.add_constant(X.astype(float))
        y = d["log_time"].astype(float)
        return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["org"]})

    r_binary = fit(restricted, ["domain_mined_x_comm"])
    r_cont = fit(restricted, ["comm_intensity", "domain_mined_x_comm_intensity"])

    p_bin = r_binary.pvalues["domain_mined_x_comm"]
    p_cont = r_cont.pvalues["domain_mined_x_comm_intensity"]

    assert round(p_bin, 4) == 0.0229
    assert round(p_cont, 4) == 0.0040
    assert p_cont < p_bin  # measured continuous proxy sharpens the interaction


def test_discount_function_insensitive_to_flux1_label():
    """Locks 09_discount_function.py's flux1-label robustness check:
    flux1 sits in COMM_BOUND but 11_measured_comm_intensity.py found it
    has zero measured TP/PP parallelism anywhere in the corpus. Moving it
    to comm-light should move the domain=72 discount cells by only a
    small fraction of their confidence bands. If this ever fails, the
    finding has become material and the headline table needs
    re-evaluating, not a quiet threshold bump."""
    df = load_clean(DATASET_CSV)
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    df = df.merge(high, on=["repo", "org", "system_id"], how="left")
    df["true_domain_mined"] = df["inferred_domain"].fillna(df["true_domain"])
    df["log_domain_mined"] = np.log(df["true_domain_mined"])

    def fit_curve(comm_set):
        d = df.copy()
        d["comm"] = d["model"].isin(comm_set).astype(float)
        d["domain_mined_x_comm"] = d["log_domain_mined"] * d["comm"]
        X = pd.concat([d[["log_gpus", "log_domain_mined", "domain_mined_x_comm"]],
                      pd.get_dummies(d["model"], prefix="model", drop_first=True),
                      pd.get_dummies(d["gen"], prefix="gen", drop_first=True)],
                     axis=1)
        X = sm.add_constant(X.astype(float))
        y = d["log_time"].astype(float)
        r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["org"]})
        b_light = r.params["log_domain_mined"]
        b_heavy = b_light + r.params["domain_mined_x_comm"]
        return b_light, b_heavy

    bl0, bh0 = fit_curve(COMM_BOUND)
    bl1, bh1 = fit_curve(COMM_BOUND - {"flux1"})

    delta_light_72 = (np.exp(bl1 * np.log(72 / 4))
                      - np.exp(bl0 * np.log(72 / 4)))
    delta_heavy_72 = (np.exp(bh1 * np.log(72 / 4))
                      - np.exp(bh0 * np.log(72 / 4)))

    assert round(delta_light_72, 3) == 0.005
    assert round(delta_heavy_72, 3) == -0.017
    assert abs(delta_light_72) < 0.02
    assert abs(delta_heavy_72) < 0.02


def test_gen_adjusted_hedonic_premium():
    """Locks 14_gen_adjusted_hedonic.py's headline gen-adjusted price
    premium (pooled log-log slope, on the n=8 subset whose GPU model has
    a measured MLPerf generation: A100/H100/B200/GB200/GB300 only -- T4,
    L4, A10/A10G are excluded, not guessed). If this ever fails, either
    the underlying market snapshot or the mined-augmented regression has
    changed and the finding needs re-reporting, not a quiet threshold
    bump."""
    df = load_clean(DATASET_CSV)
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    df = df.merge(high, on=["repo", "org", "system_id"], how="left")
    df["true_domain_mined"] = df["inferred_domain"].fillna(df["true_domain"])
    df["log_domain_mined"] = np.log(df["true_domain_mined"])

    X = pd.concat([df[["log_gpus", "log_domain_mined"]],
                  pd.get_dummies(df["model"], prefix="model", drop_first=True),
                  pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                 axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["org"]})

    speed_mult = {"A100": 1.0}
    for g in ("H100", "B200", "GB200", "GB300"):
        speed_mult[g] = float(np.exp(-r.params[f"gen_{g}"]))

    market = pd.read_csv(MARKET_CSV)
    covered = {"A100", "H100", "B200", "GB200", "GB300"}
    kept = market[market["gpu_model"].isin(covered)].copy()
    assert len(kept) == 8  # 5 of 13 excluded: T4 x2, L4 x1, A10/A10G x2

    kept["speed_multiplier"] = kept["gpu_model"].map(speed_mult)
    kept["price_eff"] = kept["price_per_gpu_hour_usd"] / kept["speed_multiplier"]

    def slope(m, col):
        m = m.copy()
        m["log_price"] = np.log(m[col])
        m["log_domain"] = np.log(m["domain_size"])
        X = sm.add_constant(m[["log_domain"]].astype(float))
        y = m["log_price"].astype(float)
        return sm.OLS(y, X).fit().params["log_domain"]

    raw_slope = slope(kept, "price_per_gpu_hour_usd")
    adj_slope = slope(kept, "price_eff")
    raw_mult = float(np.exp(raw_slope * np.log(2)))
    adj_mult = float(np.exp(adj_slope * np.log(2)))

    assert round(adj_mult, 3) == 1.243
    assert adj_mult < raw_mult  # gen-adjustment shrinks the raw premium
    assert adj_mult > 1.0       # but a premium survives -- doesn't flip sign
