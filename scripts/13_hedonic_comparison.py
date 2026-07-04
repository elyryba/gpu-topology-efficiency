"""
13_hedonic_comparison.py

Compares the market's price premium per NVLink-domain doubling (from
data/market_prices_snapshot.csv, collected by 12) against the measured
throughput premium per domain doubling (09's comm-bound discount curve):
does the market already price in the performance benefit, or is its price
spread larger/smaller than the measured performance spread?

CRITICAL CAVEAT, stated up front because it changes how to read every
number below: unlike 09's throughput regression -- which isolates domain
size while holding model and hardware generation fixed via fixed effects
-- this market snapshot has ZERO within-GPU-model domain-size variation.
Across the 13 collected SKUs, no provider sells the same GPU model at more
than one topology class (no single-GPU H100 next to an 8x-NVLink H100,
for instance). Every domain-size comparison here is therefore confounded
with GPU generation, memory, and everything else that differs between a
T4/A10 and an H100/GB200. The "market premium" computed below describes
how LIST PRICE scales with deployment class, not a topology-isolated
hedonic estimate the way 09's number is model/gen-isolated. This is a
first-order, descriptive comparison, not a matched causal estimate.

Reported anyway, honestly, because a first-order comparison is still
informative: it answers "in the same ballpark, larger, or smaller" even
without clean identification, as long as the confound is stated plainly
each time a number is used.

Combines both curves into one economically meaningful number: the
total-cost-to-train multiplier per domain doubling = price multiplier x
time multiplier. If <1, bigger domains are a net cost saving despite a
higher sticker price (the performance gain outweighs the price premium --
market underprices the benefit). If >1, bigger domains cost more overall
despite finishing faster (market's price spread exceeds the measured
performance spread).

Output: results/hedonic_comparison.txt
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from topology_common import load_clean

MARKET_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                         "market_prices_snapshot.csv")
DATASET_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                          "mlperf_topology_dataset.csv")
MINED_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                        "nvl_config_mined.csv")
OUT_TXT = os.path.join(os.path.dirname(__file__), "..", "results",
                      "hedonic_comparison.txt")

# Copied verbatim from 06/09 (same set throughout this repo).
COMM_BOUND = {"gpt3", "llama31_405b", "deepseekv3_671b", "llama2_70b_lora",
              "llama31_8b", "gpt_oss_20b", "flux1"}


def build_mined_domain(df):
    """Mirrors 08/09/10/11's function of the same name."""
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    merged = df.merge(high, on=["repo", "org", "system_id"], how="left")
    merged["true_domain_mined"] = merged["inferred_domain"].fillna(merged["true_domain"])
    merged["log_domain_mined"] = np.log(merged["true_domain_mined"])
    return merged


def fit_perf(df, extra_cols=()):
    cols = ["log_gpus", "log_domain_mined"] + list(extra_cols)
    X = pd.concat([df[cols],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["org"]})


def heavy_slope_and_se(r):
    """Mirrors 09's function of the same name: comm-bound curve's total
    slope with covariance-propagated SE."""
    b = r.params["log_domain_mined"] + r.params["domain_mined_x_comm"]
    cov = r.cov_params()
    var = (cov.loc["log_domain_mined", "log_domain_mined"]
          + cov.loc["domain_mined_x_comm", "domain_mined_x_comm"]
          + 2 * cov.loc["log_domain_mined", "domain_mined_x_comm"])
    return b, float(np.sqrt(var))


def main():
    market = pd.read_csv(MARKET_CSV)
    lines = ["HEDONIC COMPARISON: market price premium vs. measured "
            "throughput premium, per NVLink-domain doubling", ""]

    lines.append("CAVEAT (read before trusting any number below): every "
                "domain-size comparison in this market snapshot is "
                "confounded with GPU generation/model -- no provider in "
                "the collected data sells the same GPU model at more "
                "than one topology class. Unlike 09's throughput "
                "regression (model/gen fixed effects), this is a "
                "descriptive, first-order comparison, not a matched "
                "causal estimate.")
    lines.append("")

    # ---------- market side: descriptive summary ----------
    lines.append(f"MARKET SNAPSHOT (n={len(market)} SKUs, "
                f"{market['provider'].nunique()} providers, "
                f"collected {market['collected_at'].iloc[0]})")
    by_domain = market.groupby("domain_size")["price_per_gpu_hour_usd"].agg(
        ["count", "mean", "median", "min", "max"])
    for dom, row in by_domain.iterrows():
        lines.append(f"  domain={int(dom):3d}  n={int(row['count'])}  "
                    f"mean=${row['mean']:.3f}/GPU-hr  "
                    f"median=${row['median']:.3f}/GPU-hr  "
                    f"range=[${row['min']:.3f}, ${row['max']:.3f}]")
    lines.append("")

    # ---------- market side: log-log slope (descriptive) ----------
    market["log_price"] = np.log(market["price_per_gpu_hour_usd"])
    market["log_domain"] = np.log(market["domain_size"])

    X_pooled = sm.add_constant(market[["log_domain"]].astype(float))
    y = market["log_price"].astype(float)
    r_pooled = sm.OLS(y, X_pooled).fit()
    price_slope_pooled = r_pooled.params["log_domain"]

    X_fe = pd.concat([market[["log_domain"]],
                      pd.get_dummies(market["provider"], prefix="provider",
                                    drop_first=True)], axis=1)
    X_fe = sm.add_constant(X_fe.astype(float))
    r_fe = sm.OLS(y, X_fe).fit()
    price_slope_fe = r_fe.params["log_domain"]

    price_mult_pooled = float(np.exp(price_slope_pooled * np.log(2)))
    price_mult_fe = float(np.exp(price_slope_fe * np.log(2)))

    lines += [
        "MARKET PRICE SLOPE (descriptive OLS, n=13, no meaningful "
        "clustering with 3 providers -- these are NOT inferential p-values)",
        f"  pooled (no provider FE):     slope={price_slope_pooled:+.4f}  "
        f"-> price x{price_mult_pooled:.3f} per domain doubling  "
        f"R^2={r_pooled.rsquared:.3f}",
        f"  with provider FE:            slope={price_slope_fe:+.4f}  "
        f"-> price x{price_mult_fe:.3f} per domain doubling  "
        f"R^2={r_fe.rsquared:.3f}",
        "",
    ]

    # ---------- performance side: 09's comm-bound curve, recomputed ----------
    df = load_clean(DATASET_CSV)
    df = build_mined_domain(df)
    df["comm"] = df["model"].isin(COMM_BOUND).astype(float)
    df["domain_mined_x_comm"] = df["log_domain_mined"] * df["comm"]
    r_perf = fit_perf(df, extra_cols=["domain_mined_x_comm"])
    b_heavy, se_heavy = heavy_slope_and_se(r_perf)
    time_mult_per_doubling = float(np.exp(b_heavy * np.log(2)))

    lines += [
        "MEASURED THROUGHPUT PREMIUM (09_discount_function.py comm-bound "
        "curve, recomputed fresh here)",
        f"  comm-bound slope: coef={b_heavy:+.4f}  se={se_heavy:.4f}  "
        f"-> time x{time_mult_per_doubling:.3f} per domain doubling "
        f"(i.e. {(time_mult_per_doubling - 1):+.1%} change in "
        f"time-to-train)",
        "",
    ]

    # ---------- combined: total-cost-to-train multiplier ----------
    cost_mult_pooled = price_mult_pooled * time_mult_per_doubling
    cost_mult_fe = price_mult_fe * time_mult_per_doubling

    def verdict(cost_mult):
        if cost_mult < 0.97:
            return ("NET SAVING: bigger domains cost LESS overall despite "
                    "the higher sticker price -- the measured performance "
                    "gain outweighs the market's price premium. The market "
                    "underprices the topology benefit relative to what "
                    "09 measures.")
        elif cost_mult > 1.03:
            return ("NET COST: bigger domains cost MORE overall even "
                    "after finishing faster -- the market's price spread "
                    "exceeds the measured performance spread.")
        else:
            return ("ROUGHLY BREAK-EVEN: the price premium and the "
                    "performance gain are close to offsetting.")

    lines += [
        "COMBINED: total-cost-to-train multiplier per domain doubling "
        "(price multiplier x time multiplier)",
        f"  using pooled market slope:       "
        f"{price_mult_pooled:.3f} x {time_mult_per_doubling:.3f} = "
        f"{cost_mult_pooled:.3f}",
        f"    -> {verdict(cost_mult_pooled)}",
        f"  using provider-FE market slope:  "
        f"{price_mult_fe:.3f} x {time_mult_per_doubling:.3f} = "
        f"{cost_mult_fe:.3f}",
        f"    -> {verdict(cost_mult_fe)}",
        "",
        "Read this as a first-order, confounded comparison (see caveat "
        "at top), not a precise valuation -- with n=13 market SKUs and no "
        "within-GPU-model domain variation, treat the direction as the "
        "finding, not the exact multiplier.",
    ]

    out = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(out)
    print(out)
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
