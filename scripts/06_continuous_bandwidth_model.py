"""
06_continuous_bandwidth_model.py

Roadmap item #1: replace/augment the categorical domain feature with a
CONTINUOUS communication-capability feature built from the interconnect
bandwidth fields already extracted (but never used) by the pipeline:

    accel_interconnect_gbps  (intra-domain, GB/s aggregate per GPU)
    host_network_gbps        (inter-node fabric, GB/s aggregate per NODE)

Design decisions (each is a modeling choice, stated explicitly):

1. The inter-node fabric only matters when it is actually crossed, so the
   continuous-bandwidth test runs on MULTI-NODE submissions (nodes > 1).
   Single-node runs never touch the host network.

2. The physically meaningful quantity is inter-node bandwidth PER GPU:
       ibw_per_gpu = host_network_gbps / gpus_per_node
   Eight GPUs sharing 4x400Gb/s NICs see half the per-GPU egress of four
   GPUs sharing the same NICs.

3. Coverage is partial (host bw parseable for ~36% of rows; free-text
   entries like "InfiniBand" carry no number). The model is therefore fit
   on the parseable subsample, and a selection check compares that
   subsample's topology premium to the full sample's, so we know whether
   "has parseable bandwidth" is itself a biased filter.

Specs fit (all with model + generation FEs, org-clustered SEs):
   S0  log_time ~ log_gpus + log_domain                     [baseline, same subsample]
   S1  log_time ~ log_gpus + log_ibw_per_gpu                [bandwidth replaces domain]
   S2  log_time ~ log_gpus + log_domain + log_ibw_per_gpu   [horse race]
   S3  S2 + ibw x comm-bound interaction                    [mechanism: fabric should
                                                             matter more for comm-bound jobs]

Output: results/continuous_bandwidth.txt
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from topology_common import load_clean

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                       "mlperf_topology_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

COMM_BOUND = {"gpt3", "llama31_405b", "deepseekv3_671b", "llama2_70b_lora",
              "llama31_8b", "gpt_oss_20b", "flux1"}
MIN_GROUP = 8


def fit(df, cols):
    X = pd.concat([df[list(cols)],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["org"]})


def line(r, col):
    return (f"{col:20s} coef={r.params[col]:+.4f}  se={r.bse[col]:.4f}  "
            f"p={r.pvalues[col]:.3g}")


def main():
    df = load_clean(IN_PATH)
    L = []

    # ---- build the continuous feature on the multi-node subsample ----
    mn = df[(df["nodes"] > 1) & df["host_network_gbps"].notna()
            & (df["host_network_gbps"] > 0)].copy()
    mn["ibw_per_gpu"] = mn["host_network_gbps"] / mn["gpus_per_node"]
    mn["log_ibw"] = np.log(mn["ibw_per_gpu"])
    mn["comm"] = mn["model"].isin(COMM_BOUND).astype(float)
    mn["ibw_x_comm"] = mn["log_ibw"] * mn["comm"]

    # re-apply thin-category floors on the subsample so FEs are estimable
    for col in ("model", "gen"):
        mn = mn[mn.groupby(col)[col].transform("size") >= MIN_GROUP]
    mn = mn.reset_index(drop=True)

    L.append("CONTINUOUS INTER-NODE BANDWIDTH MODEL")
    L.append(f"Multi-node rows with parseable host bandwidth: n={len(mn)} "
             f"({len(mn)}/{(df['nodes'] > 1).sum()} of all multi-node rows)")
    L.append(f"models={mn['model'].nunique()}  gens={mn['gen'].nunique()}  "
             f"orgs={mn['org'].nunique()}")
    L.append(f"ibw_per_gpu (GB/s): min={mn['ibw_per_gpu'].min():.1f}  "
             f"median={mn['ibw_per_gpu'].median():.1f}  "
             f"max={mn['ibw_per_gpu'].max():.1f}")

    # ---- selection check ----
    L.append("\nSELECTION CHECK — is 'has parseable bandwidth' a biased filter?")
    allmn = df[df["nodes"] > 1].copy()
    for col in ("model", "gen"):
        allmn = allmn[allmn.groupby(col)[col].transform("size") >= MIN_GROUP]
    r_all = fit(allmn, ["log_gpus", "log_domain"])
    r_sub = fit(mn, ["log_gpus", "log_domain"])
    L.append(f"all multi-node rows (n={len(allmn)}):        " + line(r_all, "log_domain"))
    L.append(f"parseable-bw subsample (n={len(mn)}):  " + line(r_sub, "log_domain"))

    # ---- S1: bandwidth replaces domain ----
    L.append("\nS1 — bandwidth as the sole topology feature")
    r1 = fit(mn, ["log_gpus", "log_ibw"])
    L.append(line(r1, "log_ibw"))
    L.append(f"R2={r1.rsquared:.4f}")

    # ---- S2: horse race ----
    L.append("\nS2 — horse race: domain size vs inter-node bandwidth, jointly")
    r2 = fit(mn, ["log_gpus", "log_domain", "log_ibw"])
    L.append(line(r2, "log_domain"))
    L.append(line(r2, "log_ibw"))
    corr = mn[["log_domain", "log_ibw"]].corr().iloc[0, 1]
    L.append(f"corr(log_domain, log_ibw)={corr:+.3f}   R2={r2.rsquared:.4f}")

    # ---- S3: mechanism interaction ----
    L.append("\nS3 — does inter-node bandwidth matter more for comm-bound jobs?")
    r3 = fit(mn, ["log_gpus", "log_domain", "log_ibw", "ibw_x_comm"])
    L.append(line(r3, "log_ibw"))
    L.append(line(r3, "ibw_x_comm"))
    tot = r3.params["log_ibw"] + r3.params["ibw_x_comm"]
    L.append(f"implied bandwidth elasticity for comm-bound workloads: {tot:+.4f}")
    L.append("(Physics predicts ibw coefficients NEGATIVE — more egress "
             "bandwidth per GPU, less training time — and the interaction "
             "more negative still.)")

    L.append("\nINTERPRETATION GUIDE")
    L.append("A log_ibw coefficient of -0.05 means doubling per-GPU inter-node")
    L.append("bandwidth at fixed scale and domain size cuts time-to-train ~3.4%.")
    L.append("If S2 shows bandwidth absorbing the domain coefficient, the")
    L.append("continuous feature subsumes the categorical one — the property a")
    L.append("generalizable discount function needs.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = "\n".join(L)
    with open(os.path.join(OUT_DIR, "continuous_bandwidth.txt"), "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
