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
