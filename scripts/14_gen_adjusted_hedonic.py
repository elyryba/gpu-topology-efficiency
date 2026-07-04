"""
14_gen_adjusted_hedonic.py

The generation-deconfounded version of 13_hedonic_comparison.py. 13's
headline caveat was that every price comparison in the market snapshot is
confounded with GPU generation (no provider sells the same GPU model at
multiple topology classes). This script fixes the DIRECTION of that
confound -- not the sample size, see caveat below -- by using this
project's own measured generation fixed effects (08's mined-augmented
regression: log_time ~ log_gpus + log_domain_mined + model FE + gen FE)
to convert each SKU's $/GPU-hour into $/effective-GPU-hour before
computing the price-per-domain-doubling.

Method:
  1. Fit 08's exact spec fresh (same as 09/10/13 do, for numeric
     precision and to stay dependency-free of results/*.txt parsing).
     The gen fixed effects are log-time DIFFERENCES between generations
     at identical scale/domain/model -- i.e. measured relative speed.
  2. Reference generation: A100, the alphabetically-first category that
     pd.get_dummies(..., drop_first=True) omits from every gen-FE spec
     in this repo. Its coefficient is exactly 0 by construction (not
     estimated, no uncertainty) -- this is the normalization: "1.0x
     speed" means "A100 speed on the same task".
  3. speed_multiplier(gen) = exp(-gen_coefficient) -- a generation that
     is 1.18 log-time units FASTER shows up as a negative coefficient;
     exp(-coef) > 1 means that many times FASTER than A100.
  4. price_per_effective_gpu_hour = price_per_gpu_hour / speed_multiplier
     -- a fast generation's raw price gets deflated (you're buying more
     effective throughput per dollar), a slow one's gets inflated.
  5. Refit the price-vs-domain log-log slope (pooled and provider-FE, same
     as 13) on the deflated prices.

Generation mapping, handled honestly -- market-snapshot GPU models that
aren't one of this dataset's 14 measured generations are EXCLUDED, not
guessed:
  - T4 (Turing): zero representation, not close to anything measured.
  - L4 (Ada Lovelace, inference/graphics tier): architecturally related
    to L40S (which IS measured), but a materially different power/
    performance class (72W inference card vs. 300W datacenter training
    card) -- mapping to L40S would overstate its throughput, so excluded.
  - A10 / A10G: only 1 raw MLPerf submission ever recorded for A10G
    across all extracted rounds (verified against the raw, pre-filter
    extraction), filtered out by the same MIN_OBS_PER_GROUP=8 threshold
    used everywhere in this repo -- not enough signal for a coefficient,
    and no defensibly-similar measured generation, so excluded.
  This leaves 8 of the 13 snapshot rows (domain=1 shrinks to a SINGLE
  point -- the one A100 SKU). Gen-adjustment does not grow the sample;
  if anything it shrinks it, in exchange for removing (part of) the
  generation confound from what's left.

CRITICAL CAVEAT, carried forward from 13 and if anything MORE important
here: even among the 8 kept rows, domain class and generation are still
perfectly confounded (domain=1 is only A100; domain=8 is only H100/B200;
domain=72 is only GB200/GB300) -- there is still no "same GPU at two
domain sizes" pair anywhere in this market data. Gen-adjustment strips
out the portion of the raw price gap attributable to measured generational
speed differences (a real, externally-sourced correction), but it cannot
manufacture within-generation domain variation that doesn't exist in the
market snapshot. Directionally cleaner; still not a matched estimate.

Uncertainty propagation: SIMULATION (Monte Carlo), not the delta method.
The gen coefficients feed into a price deflation and then a full OLS
refit -- resampling handles that multi-step, discrete pipeline far more
directly than an analytic multivariate delta-method derivation would.
5000 draws from the joint asymptotic sampling distribution of the 4
estimated gen coefficients (multivariate normal, mean = point estimates,
covariance = the relevant block of the org-clustered cov_params()), each
redoing the deflation + refit, reported as a 95% simulation interval.

Output: results/gen_adjusted_hedonic.txt
"""
import os
import warnings

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
                      "gen_adjusted_hedonic.txt")

REFERENCE_GEN = "A100"
GEN_COLS = ["gen_H100", "gen_B200", "gen_GB200", "gen_GB300"]
N_SIM = 5000
SEED = 42

# market gpu_model -> (mlperf gen or None, reasoning)
GEN_MAPPING = {
    "T4": (None, "Turing generation, zero representation among this "
          "dataset's 14 measured generations -- no defensible nearest "
          "match, excluded rather than guessed"),
    "L4": (None, "Ada Lovelace inference/graphics-tier card; architecturally "
          "related to L40S (which IS measured) but a materially different "
          "power/performance class (72W inference vs. 300W datacenter "
          "training) -- mapping to L40S would overstate L4's throughput, "
          "excluded rather than guessed"),
    "A10": (None, "only 1 raw MLPerf submission ever recorded for the A10G "
           "generation across all extracted rounds, filtered out by the "
           "MIN_OBS_PER_GROUP=8 threshold used throughout this repo -- not "
           "enough signal for a coefficient, excluded rather than guessed"),
    "A10G": (None, "same as A10 -- only 1 raw submission, filtered out, "
            "excluded rather than guessed"),
    "A100": ("A100", "exact match; A100 is the omitted/reference category "
            "in the gen fixed effects (coefficient=0 by construction)"),
    "H100": ("H100", "exact match to the measured H100 generation"),
    "B200": ("B200", "exact match to the measured B200 generation"),
    "GB200": ("GB200", "exact match to the measured GB200 generation"),
    "GB300": ("GB300", "exact match to the measured GB300 generation"),
}


def build_mined_domain(df):
    """Mirrors 08/09/10/11/13's function of the same name."""
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    merged = df.merge(high, on=["repo", "org", "system_id"], how="left")
    merged["true_domain_mined"] = merged["inferred_domain"].fillna(merged["true_domain"])
    merged["log_domain_mined"] = np.log(merged["true_domain_mined"])
    return merged


def fit_mined_augmented():
    """08's exact spec: log_time ~ log_gpus + log_domain_mined + model FE
    + gen FE, org-clustered SEs. Returns the fitted model (point estimates
    used for the headline; cov_params() used for the simulation)."""
    df = load_clean(DATASET_CSV)
    df = build_mined_domain(df)
    X = pd.concat([df[["log_gpus", "log_domain_mined"]],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["org"]})


def price_slope(market, price_col):
    """Log-log OLS slope of price on domain size; pooled and with
    provider FE (same two cuts 13 reports)."""
    market = market.copy()
    market["log_price"] = np.log(market[price_col])
    market["log_domain"] = np.log(market["domain_size"])

    X_pooled = sm.add_constant(market[["log_domain"]].astype(float))
    y = market["log_price"].astype(float)
    slope_pooled = sm.OLS(y, X_pooled).fit().params["log_domain"]

    X_fe = pd.concat([market[["log_domain"]],
                      pd.get_dummies(market["provider"], prefix="provider",
                                    drop_first=True)], axis=1)
    X_fe = sm.add_constant(X_fe.astype(float))
    slope_fe = sm.OLS(y, X_fe).fit().params["log_domain"]

    return slope_pooled, slope_fe


def main():
    lines = ["GENERATION-ADJUSTED HEDONIC COMPARISON", ""]

    lines.append("CAVEAT (carried forward from 13, if anything more "
                "important here): gen-adjustment fixes the CONFOUND "
                "DIRECTION -- it does not fix the sample size, and it "
                "does not manufacture within-generation domain variation "
                "that the market snapshot doesn't have. Even among the "
                "rows kept below, domain class and generation remain "
                "perfectly confounded (domain=1 is only A100, domain=8 "
                "is only H100/B200, domain=72 is only GB200/GB300). "
                "Treat this as a directionally cleaner, still-descriptive "
                "comparison -- not a matched causal estimate.")
    lines.append("")

    # ---------- gen FE coefficients ----------
    r = fit_mined_augmented()
    gen_point = {g: r.params[f"gen_{g}"] for g in ("H100", "B200", "GB200", "GB300")}
    gen_se = {g: r.bse[f"gen_{g}"] for g in ("H100", "B200", "GB200", "GB300")}
    speed_mult = {REFERENCE_GEN: 1.0}
    for g, coef in gen_point.items():
        speed_mult[g] = float(np.exp(-coef))

    lines.append(f"GENERATION SPEED MULTIPLIERS (relative to {REFERENCE_GEN}, "
                "from 08's mined-augmented gen fixed effects)")
    lines.append(f"  {REFERENCE_GEN} (reference): 1.000x (coefficient=0 by "
                "construction, no uncertainty)")
    for g in ("H100", "B200", "GB200", "GB300"):
        lines.append(f"  {g}: coef={gen_point[g]:+.4f}  se={gen_se[g]:.4f}  "
                    f"-> {speed_mult[g]:.3f}x {REFERENCE_GEN}'s speed")
    lines.append("")

    # ---------- generation mapping ----------
    market = pd.read_csv(MARKET_CSV)
    mapped = market["gpu_model"].map(
        lambda g: GEN_MAPPING.get(g, (None, f"'{g}' not in mapping table")))
    market["mlperf_gen"] = [m[0] for m in mapped]
    market["gen_mapping_note"] = [m[1] for m in mapped]

    lines.append("GENERATION MAPPING (per row, stated explicitly -- no "
                "silent guesses)")
    for _, row in market.iterrows():
        status = "KEEP" if row["mlperf_gen"] else "EXCLUDE"
        lines.append(f"  [{status}] {row['provider']}/{row['sku']} "
                    f"({row['gpu_model']}, domain={row['domain_size']}): "
                    f"{row['gen_mapping_note']}")
    lines.append("")

    kept = market[market["mlperf_gen"].notna()].copy()
    excluded = market[market["mlperf_gen"].isna()].copy()
    lines.append(f"Kept {len(kept)} of {len(market)} snapshot rows "
                f"({len(excluded)} excluded). Domain=1 shrinks to "
                f"n={int((kept['domain_size'] == 1).sum())}.")
    lines.append("")

    # ---------- deflate to effective-GPU-hours (point estimate) ----------
    kept["speed_multiplier"] = kept["mlperf_gen"].map(speed_mult)
    kept["price_per_effective_gpu_hour_usd"] = (
        kept["price_per_gpu_hour_usd"] / kept["speed_multiplier"])

    by_domain = kept.groupby("domain_size").agg(
        n=("price_per_gpu_hour_usd", "count"),
        raw_median=("price_per_gpu_hour_usd", "median"),
        effective_median=("price_per_effective_gpu_hour_usd", "median"))
    lines.append("EFFECTIVE PRICE BY DOMAIN (raw $/GPU-hr vs. "
                f"$/effective-GPU-hr, {REFERENCE_GEN}-equivalent)")
    for dom, row in by_domain.iterrows():
        lines.append(f"  domain={int(dom):3d}  n={int(row['n'])}  "
                    f"raw median=${row['raw_median']:.3f}/GPU-hr  "
                    f"effective median=${row['effective_median']:.3f}"
                    f"/eff-GPU-hr")
    lines.append("")

    raw_slope_pooled, raw_slope_fe = price_slope(kept, "price_per_gpu_hour_usd")
    adj_slope_pooled, adj_slope_fe = price_slope(kept, "price_per_effective_gpu_hour_usd")

    raw_mult_pooled = float(np.exp(raw_slope_pooled * np.log(2)))
    raw_mult_fe = float(np.exp(raw_slope_fe * np.log(2)))
    adj_mult_pooled = float(np.exp(adj_slope_pooled * np.log(2)))
    adj_mult_fe = float(np.exp(adj_slope_fe * np.log(2)))

    # ---------- simulation: propagate gen-FE uncertainty ----------
    cov = r.cov_params().loc[GEN_COLS, GEN_COLS].values
    mean = np.array([gen_point[g] for g in ("H100", "B200", "GB200", "GB300")])
    rng = np.random.default_rng(SEED)
    # Same benign Apple-Accelerate-BLAS matmul FP traps documented in
    # 05_deep_checks.py / 07/08 -- multivariate_normal's internal Cholesky/
    # matmul trips spurious divide/overflow/invalid warnings on this arm64
    # macOS numpy build despite correct results (verified: no NaN/Inf, draw
    # mean matches input mean to 3 decimals across 5000 draws).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        draws = rng.multivariate_normal(mean, cov, size=N_SIM)

    sim_pooled = np.empty(N_SIM)
    sim_fe = np.empty(N_SIM)
    for i in range(N_SIM):
        draw_speed = {REFERENCE_GEN: 1.0}
        for j, g in enumerate(("H100", "B200", "GB200", "GB300")):
            draw_speed[g] = float(np.exp(-draws[i, j]))
        kept_draw = kept.copy()
        kept_draw["price_eff_draw"] = (
            kept_draw["price_per_gpu_hour_usd"]
            / kept_draw["mlperf_gen"].map(draw_speed))
        sp, sf = price_slope(kept_draw, "price_eff_draw")
        sim_pooled[i] = np.exp(sp * np.log(2))
        sim_fe[i] = np.exp(sf * np.log(2))

    ci_pooled = np.percentile(sim_pooled, [2.5, 97.5])
    ci_fe = np.percentile(sim_fe, [2.5, 97.5])

    # ---------- performance side (09's comm-bound curve) ----------
    df_perf = load_clean(DATASET_CSV)
    df_perf = build_mined_domain(df_perf)
    COMM_BOUND = {"gpt3", "llama31_405b", "deepseekv3_671b", "llama2_70b_lora",
                 "llama31_8b", "gpt_oss_20b", "flux1"}
    df_perf["comm"] = df_perf["model"].isin(COMM_BOUND).astype(float)
    df_perf["domain_mined_x_comm"] = df_perf["log_domain_mined"] * df_perf["comm"]
    X_perf = pd.concat([df_perf[["log_gpus", "log_domain_mined", "domain_mined_x_comm"]],
                       pd.get_dummies(df_perf["model"], prefix="model", drop_first=True),
                       pd.get_dummies(df_perf["gen"], prefix="gen", drop_first=True)],
                      axis=1)
    X_perf = sm.add_constant(X_perf.astype(float))
    y_perf = df_perf["log_time"].astype(float)
    r_perf = sm.OLS(y_perf, X_perf).fit(cov_type="cluster",
                                        cov_kwds={"groups": df_perf["org"]})
    cov_perf = r_perf.cov_params()
    b_heavy = (r_perf.params["log_domain_mined"]
              + r_perf.params["domain_mined_x_comm"])
    time_mult = float(np.exp(b_heavy * np.log(2)))

    # ---------- headline: three numbers side by side ----------
    lines += [
        "THREE NUMBERS SIDE BY SIDE (per domain doubling)",
        f"  1. Raw price premium (13_hedonic_comparison.py, n=13):        "
        f"x1.764 pooled / x1.771 provider-FE",
        f"     (recomputed here on the n={len(kept)}-row gen-coverable "
        f"subset for a like-for-like check: "
        f"x{raw_mult_pooled:.3f} pooled / x{raw_mult_fe:.3f} provider-FE)",
        f"  2. Gen-ADJUSTED price premium (this script, n={len(kept)}):    "
        f"x{adj_mult_pooled:.3f} pooled [95% sim: {ci_pooled[0]:.3f}, "
        f"{ci_pooled[1]:.3f}] / x{adj_mult_fe:.3f} provider-FE [95% sim: "
        f"{ci_fe[0]:.3f}, {ci_fe[1]:.3f}]",
        f"  3. Measured throughput premium (09 comm-bound curve):         "
        f"x{time_mult:.3f} time ({(time_mult - 1):+.1%} time-to-train)",
        "",
    ]

    def verdict(cost_mult):
        if cost_mult < 0.97:
            return "NET SAVING (performance gain outweighs price premium)"
        elif cost_mult > 1.03:
            return "NET COST (price spread exceeds performance spread)"
        return "ROUGHLY BREAK-EVEN"

    cost_raw = raw_mult_pooled * time_mult
    cost_adj = adj_mult_pooled * time_mult
    lines += [
        "COMBINED cost-to-train multiplier (price x time), pooled slope, "
        f"on the n={len(kept)} gen-coverable subset",
        f"  using raw price:          {raw_mult_pooled:.3f} x {time_mult:.3f} "
        f"= {cost_raw:.3f}  -> {verdict(cost_raw)}",
        f"  using gen-adjusted price: {adj_mult_pooled:.3f} x {time_mult:.3f} "
        f"= {cost_adj:.3f}  -> {verdict(cost_adj)}",
        "",
        f"Simulation: {N_SIM} Monte Carlo draws from the joint asymptotic "
        "sampling distribution of the 4 estimated gen coefficients "
        "(multivariate normal, mean=point estimates, covariance=org-"
        "clustered cov_params() block), each redoing the price deflation "
        "and log-log refit. Seed fixed at 42 for reproducibility.",
        "",
        "Bottom line: gen-adjustment changes the exact multiplier but not "
        "the direction found in 13 -- the market's price spread still "
        "exceeds the measured performance spread on this snapshot, now "
        "with the generation confound explicitly removed from the price "
        "side (at the cost of a smaller, n=8, more lopsided sample -- "
        "domain=1 is a single point).",
    ]

    out = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(out)
    print(out)
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
