"""
15_export_model_params.py

Exports every parameter the public calculator (site/index.html) needs
into site/model_params.json. Everything here is RECOMPUTED fresh from
the same fitting code as 08/09/13/14 -- nothing is hand-copied from
results/*.txt -- so the calculator can never silently drift from the
actual fitted models. tests/test_reproduction.py regenerates this file
in-memory and asserts it matches the committed copy byte-for-byte,
enforcing that.

Exports:
  - Discount curve slopes + SEs (09's two-curve spec, evidence-augmented
    domain feature): comm-bound and comm-light.
  - Observed domain ranges per workload class (09), for the site's own
    interpolation/extrapolation flag -- computed the same way 09 does.
  - Gen FE speed multipliers vs. A100 (08's mined-augmented spec) --
    informational context on the site, not used by the core discount math.
  - Market premium numbers: raw (13) and gen-adjusted (14, with its
    simulated 95% interval).
  - Metadata: latest repo version tag, dataset n, export timestamp.

Output: site/model_params.json
"""
import datetime
import json
import os
import subprocess

import numpy as np
import pandas as pd
import statsmodels.api as sm

from topology_common import load_clean

DATASET_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                          "mlperf_topology_dataset.csv")
MINED_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                        "nvl_config_mined.csv")
MARKET_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                         "market_prices_snapshot.csv")
OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "site",
                       "model_params.json")

COMM_BOUND = {"gpt3", "llama31_405b", "deepseekv3_671b", "llama2_70b_lora",
              "llama31_8b", "gpt_oss_20b", "flux1"}
DOMAIN_REF = 4
Z = 1.96
N_SIM = 5000
SIM_SEED = 42


def build_mined_domain(df):
    """Mirrors 08/09/10/11/13/14's function of the same name."""
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    merged = df.merge(high, on=["repo", "org", "system_id"], how="left")
    merged["true_domain_mined"] = merged["inferred_domain"].fillna(merged["true_domain"])
    merged["log_domain_mined"] = np.log(merged["true_domain_mined"])
    return merged


def fit(df, extra_cols=()):
    cols = ["log_gpus", "log_domain_mined"] + list(extra_cols)
    X = pd.concat([df[cols],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["org"]})


def heavy_slope_and_se(r):
    """Mirrors 09's function of the same name."""
    b = r.params["log_domain_mined"] + r.params["domain_mined_x_comm"]
    cov = r.cov_params()
    var = (cov.loc["log_domain_mined", "log_domain_mined"]
          + cov.loc["domain_mined_x_comm", "domain_mined_x_comm"]
          + 2 * cov.loc["log_domain_mined", "domain_mined_x_comm"])
    return b, float(np.sqrt(var))


def price_slope(market, price_col):
    """Mirrors 13/14's function of the same name."""
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


def latest_tag():
    try:
        return subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=os.path.dirname(__file__), capture_output=True, text=True,
            check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_params():
    """Pure computation, no I/O -- returns the params dict. Split out from
    main() so tests can call this directly without writing to (or reading
    a stale copy of) site/model_params.json."""
    df = load_clean(DATASET_CSV)
    df = build_mined_domain(df)
    df["comm"] = df["model"].isin(COMM_BOUND).astype(float)
    df["domain_mined_x_comm"] = df["log_domain_mined"] * df["comm"]

    r = fit(df, extra_cols=["domain_mined_x_comm"])
    b_light = float(r.params["log_domain_mined"])
    se_light = float(r.bse["log_domain_mined"])
    b_heavy, se_heavy = heavy_slope_and_se(r)

    light_domains = df.loc[df["comm"] == 0, "true_domain_mined"]
    heavy_domains = df.loc[df["comm"] == 1, "true_domain_mined"]

    # ---------- gen FE speed multipliers (08's spec -- NOT 09's) ----------
    # 08's spec has no domain_mined_x_comm interaction term; refitting it
    # here rather than reusing `r` above (09's two-curve spec) matters --
    # the two specs give measurably different gen coefficients since the
    # interaction term soaks up some of what would otherwise load onto
    # the gen dummies. Confirmed against 14_gen_adjusted_hedonic.py's
    # already-reported values (H100=3.260x, B200=6.886x, GB200=5.922x,
    # GB300=8.446x) before trusting this.
    r08 = fit(df)
    gen_point = {g: float(r08.params[f"gen_{g}"])
                for g in ("H100", "B200", "GB200", "GB300")}
    gen_speed = {"A100": 1.0}
    for g, coef in gen_point.items():
        gen_speed[g] = float(np.exp(-coef))

    # ---------- market premiums (13 raw, 14 gen-adjusted) ----------
    market = pd.read_csv(MARKET_CSV)
    raw_slope_pooled, raw_slope_fe = price_slope(market, "price_per_gpu_hour_usd")
    raw_mult_pooled = float(np.exp(raw_slope_pooled * np.log(2)))
    raw_mult_fe = float(np.exp(raw_slope_fe * np.log(2)))

    covered = {"A100", "H100", "B200", "GB200", "GB300"}
    kept = market[market["gpu_model"].isin(covered)].copy()
    kept["speed_multiplier"] = kept["gpu_model"].map(gen_speed)
    kept["price_eff"] = kept["price_per_gpu_hour_usd"] / kept["speed_multiplier"]
    adj_slope_pooled, adj_slope_fe = price_slope(kept, "price_eff")
    adj_mult_pooled = float(np.exp(adj_slope_pooled * np.log(2)))
    adj_mult_fe = float(np.exp(adj_slope_fe * np.log(2)))

    gen_cols = ["gen_H100", "gen_B200", "gen_GB200", "gen_GB300"]
    cov = r08.cov_params().loc[gen_cols, gen_cols].values
    mean = np.array([gen_point[g] for g in ("H100", "B200", "GB200", "GB300")])
    rng = np.random.default_rng(SIM_SEED)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        draws = rng.multivariate_normal(mean, cov, size=N_SIM)
    sim_pooled = np.empty(N_SIM)
    sim_fe = np.empty(N_SIM)
    for i in range(N_SIM):
        draw_speed = {"A100": 1.0}
        for j, g in enumerate(("H100", "B200", "GB200", "GB300")):
            draw_speed[g] = float(np.exp(-draws[i, j]))
        kd = kept.copy()
        kd["price_eff_draw"] = kd["price_per_gpu_hour_usd"] / kd["gpu_model"].map(draw_speed)
        sp, sf = price_slope(kd, "price_eff_draw")
        sim_pooled[i] = np.exp(sp * np.log(2))
        sim_fe[i] = np.exp(sf * np.log(2))
    ci_pooled = np.percentile(sim_pooled, [2.5, 97.5]).tolist()
    ci_fe = np.percentile(sim_fe, [2.5, 97.5]).tolist()

    params = {
        "metadata": {
            "repo_url": "https://github.com/elyryba/gpu-topology-efficiency",
            "repo_version": latest_tag(),
            "dataset_n": int(len(df)),
            "generated_at": datetime.datetime.now(datetime.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "discount_curves": {
            "spec": ("log_time ~ log_gpus + log_domain_mined + "
                    "domain_mined_x_comm + model FE + gen FE, "
                    "org-clustered SEs (09_discount_function.py)"),
            "domain_ref": DOMAIN_REF,
            "z": Z,
            "comm_bound": {
                "slope": round(b_heavy, 6),
                "se": round(se_heavy, 6),
                "observed_domain_range": [float(heavy_domains.min()),
                                         float(heavy_domains.max())],
            },
            "comm_light": {
                "slope": round(b_light, 6),
                "se": round(se_light, 6),
                "observed_domain_range": [float(light_domains.min()),
                                         float(light_domains.max())],
            },
        },
        "gen_speed_multipliers": {
            "reference": "A100",
            "source": "08_refit_with_mined_domains.py gen fixed effects",
            "multipliers": {g: round(v, 4) for g, v in gen_speed.items()},
        },
        "market_premium": {
            "note": ("Descriptive, confounded with GPU generation (13); "
                    "gen-adjusted removes that confound but shrinks the "
                    "sample to n=8 (14). See README Market pricing "
                    "comparison section."),
            "raw": {
                "n": int(len(market)),
                "pooled": round(raw_mult_pooled, 4),
                "provider_fe": round(raw_mult_fe, 4),
            },
            "gen_adjusted": {
                "n": int(len(kept)),
                "pooled": round(adj_mult_pooled, 4),
                "pooled_ci95": [round(x, 4) for x in ci_pooled],
                "provider_fe": round(adj_mult_fe, 4),
                "provider_fe_ci95": [round(x, 4) for x in ci_fe],
            },
        },
    }
    return params


def main():
    params = build_params()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(params, indent=2, sort_keys=True))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
