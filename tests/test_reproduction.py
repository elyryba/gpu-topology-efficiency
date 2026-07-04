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

import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from topology_common import fit_model, load_clean

DATASET_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                          "mlperf_topology_dataset.csv")


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
