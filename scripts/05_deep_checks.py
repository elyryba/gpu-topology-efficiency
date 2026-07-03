"""
05_deep_checks.py

Six attacks a skeptical reviewer (or a counterparty pricing against this
analysis) would mount, each implemented and answered with the data:

A. IDENTIFICATION AUDIT — where does the v2 domain coefficient actually
   get its variation? If log_domain is nearly constant within each
   hardware generation, the gen fixed effects absorb it and the
   "effect" is identified from a sliver of the data.

B. CLEAN-ID SUBSAMPLE — Hopper-and-earlier only, where NVLink genuinely
   stops at the server and gpus_per_node truly varies (2/4/8/16).
   No rack-scale correction needed; the least model-dependent estimate.

C. MECHANISM TEST — if the premium is real physics (inter-node fabric
   is slower than NVLink), it must be LARGER for communication-bound
   workloads (big LLM pretraining/finetuning) than for small
   single-node-friendly vision models. Interaction term tests this.
   A premium that's flat across workload types would suggest
   confounding, not physics.

D. TEMPORAL HOLDOUT — fit on rounds v3.1–v5.0, test on v5.1–v6.0.
   Both coefficient stability and out-of-sample prediction quality.

E. ESTIMATOR SENSITIVITY — kurtosis ~15 says OLS point estimates may be
   dragged by tails. Refit the premium spec with Huber robust
   regression and median (quantile) regression.

F. MEASUREMENT SENSITIVITY — the pipeline uses best (min) run time,
   which favors orgs submitting many runs. Refit on median run time.
   Also: within H100, PCIe vs SXM variants confound "small domain"
   with "slower intra-node link" — refit excluding PCIe/NVL variants.

Output: results/deep_checks.txt
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from topology_common import load_clean

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                       "mlperf_topology_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Communication-bound benchmarks: large-model pretraining/finetuning where
# tensor/pipeline-parallel traffic dominates. Everything else fits in (or
# nearly in) one node and is data-parallel dominated.
COMM_BOUND = {"gpt3", "llama31_405b", "deepseekv3_671b", "llama2_70b_lora",
              "llama31_8b", "gpt_oss_20b", "flux1"}

PRE_BLACKWELL = {"A100", "H100", "H200", "L40S", "RTX",
                 "MI300X", "MI325X", "TPU-v5p"}


def premium_fit(df, domain_col="log_domain", ycol="log_time",
                extra_cols=(), cluster=True):
    cols = ["log_gpus", domain_col] + list(extra_cols)
    X = pd.concat([df[cols],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    X = sm.add_constant(X.astype(float))
    y = df[ycol].astype(float)
    m = sm.OLS(y, X)
    if cluster:
        return m.fit(cov_type="cluster", cov_kwds={"groups": df["org"]})
    return m.fit()


def fmt(r, col):
    return (f"coef={r.params[col]:+.4f}  se={r.bse[col]:.4f}  "
            f"p={r.pvalues[col]:.3g}")


def main():
    df = load_clean(IN_PATH)
    L = []

    # ---------- A. identification audit ----------
    L.append("A. IDENTIFICATION AUDIT — within-generation variation in log_domain")
    L.append(f"{'gen':10s} {'n':>5s} {'sd(log_domain)':>15s} {'sd(log_gpn)':>12s}")
    for g, d in df.groupby("gen"):
        L.append(f"{g:10s} {len(d):5d} {d['log_domain'].std():15.3f} "
                 f"{d['log_gpn'].std():12.3f}")
    within = df.groupby("gen")["log_domain"].transform(lambda s: s - s.mean())
    total_sd = df["log_domain"].std()
    L.append(f"Within-gen share of log_domain variance: "
             f"{(within.var() / df['log_domain'].var()) * 100:.1f}% "
             f"(the part NOT absorbed by gen fixed effects)")
    gb = df[df["gen"].isin(["GB200", "GB300"])]
    L.append(f"GB200/GB300 rows with total_gpus < 72 (sub-rack, where the "
             f"correction creates variation): {(gb['total_gpus'] < 72).sum()} "
             f"of {len(gb)}")

    # ---------- B. clean-ID subsample ----------
    L.append("\nB. CLEAN-ID SUBSAMPLE — pre-Blackwell generations only "
             "(NVLink truly = server boundary)")
    sub = df[df["gen"].isin(PRE_BLACKWELL)].copy()
    sub = sub[sub.groupby("gen")["gen"].transform("size") >= 8]
    sub = sub[sub.groupby("model")["model"].transform("size") >= 8]
    r = premium_fit(sub)
    L.append(f"n={len(sub)}, gens={sorted(sub['gen'].unique())}")
    L.append(f"topology premium: {fmt(r, 'log_domain')}")

    # ---------- C. mechanism test ----------
    L.append("\nC. MECHANISM TEST — premium x communication-boundedness")
    df["comm"] = df["model"].isin(COMM_BOUND).astype(float)
    df["domain_x_comm"] = df["log_domain"] * df["comm"]
    r = premium_fit(df, extra_cols=["domain_x_comm"])
    L.append(f"base premium (comm-light workloads):  {fmt(r, 'log_domain')}")
    L.append(f"extra premium for comm-bound LLMs:    {fmt(r, 'domain_x_comm')}")
    tot = r.params["log_domain"] + r.params["domain_x_comm"]
    L.append(f"implied total premium for comm-bound workloads: {tot:+.4f}")
    L.append("(Physics predicts the interaction should be negative — "
             "larger domains should help communication-bound jobs more.)")

    # ---------- D. temporal holdout ----------
    L.append("\nD. TEMPORAL HOLDOUT — fit v3.1–v5.0, test v5.1–v6.0")
    early = df[df["repo"].isin(["training_results_v3.1", "training_results_v4.0",
                                "training_results_v4.1", "training_results_v5.0"])]
    late = df[df["repo"].isin(["training_results_v5.1", "training_results_v6.0"])]
    # restrict both to models/gens present in both windows so FEs transfer
    shared_m = set(early["model"]) & set(late["model"])
    shared_g = set(early["gen"]) & set(late["gen"])
    e = early[early["model"].isin(shared_m) & early["gen"].isin(shared_g)]
    l = late[late["model"].isin(shared_m) & late["gen"].isin(shared_g)]
    if len(e) > 50 and len(l) > 50:
        re_ = premium_fit(e)
        rl = premium_fit(l)
        L.append(f"early n={len(e)}: premium {fmt(re_, 'log_domain')}")
        L.append(f"late  n={len(l)}: premium {fmt(rl, 'log_domain')}")
        # out-of-sample prediction using early coefficients
        cols = ["log_gpus", "log_domain"]
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
        L.append(f"out-of-sample R^2 on late rounds (early-fit coefficients): "
                 f"{1 - ss_res / ss_tot:.3f}   "
                 f"median |error| = {np.exp(np.median(np.abs(resid))) - 1:.1%} "
                 f"of time-to-train")
    else:
        L.append(f"insufficient overlap (early n={len(e)}, late n={len(l)})")

    # ---------- E. estimator sensitivity ----------
    L.append("\nE. ESTIMATOR SENSITIVITY (heavy-tailed residuals, kurtosis ~15)")
    cols = ["log_gpus", "log_domain"]
    X = pd.concat([df[cols],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    # RLM/QuantReg call np.linalg.pinv() on this (full-rank, cond#~143) design
    # matrix. On numpy>=2.0 + Apple's Accelerate BLAS (arm64 macOS), pinv's
    # internal SVD reconstruction trips spurious divide/overflow/invalid FP
    # traps inside matmul even though the result is correct: verified
    # pinv(X) @ X reconstructs the identity to ~1e-15 with no NaN/Inf, and
    # the coefficients below reproduce results/deep_checks.txt exactly.
    # Scoped to this block only so a genuine numerical problem elsewhere
    # still surfaces.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        hub = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        qr = sm.QuantReg(y, X).fit(q=0.5)
    L.append(f"Huber robust regression:  premium coef={hub.params['log_domain']:+.4f} "
             f"se={hub.bse['log_domain']:.4f}")
    L.append(f"Median (quantile) regr.:  premium coef={qr.params['log_domain']:+.4f} "
             f"se={qr.bse['log_domain']:.4f}")

    # ---------- F. measurement sensitivity ----------
    L.append("\nF. MEASUREMENT SENSITIVITY")
    dfm = df[df["median_time_to_train_s"] > 0].copy()
    dfm["log_time_med"] = np.log(dfm["median_time_to_train_s"])
    r = premium_fit(dfm, ycol="log_time_med")
    L.append(f"median run time instead of best: {fmt(r, 'log_domain')}  (n={len(dfm)})")

    sxm = df[~df["accelerator"].str.contains("PCIe|NVL-", case=False,
                                             na=False, regex=True)]
    r = premium_fit(sxm)
    L.append(f"excluding PCIe / H200-NVL variants: {fmt(r, 'log_domain')}  "
             f"(n={len(sxm)})")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = "\n".join(L)
    with open(os.path.join(OUT_DIR, "deep_checks.txt"), "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
