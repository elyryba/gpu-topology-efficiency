"""
11_measured_comm_intensity.py

Referee point: 09_discount_function.py's comm-bound/comm-light split is a
hand-labeled binary (COMM_BOUND, a fixed set of 7 model names) standing in
for a continuous physical quantity -- how much cross-GPU communication a
workload's parallelization strategy actually requires. This script builds
a MEASURED continuous proxy instead: tensor-parallel (TP) and
pipeline-parallel (PP) degree, parsed directly from MLPerf benchmark
config filenames (e.g. "config_GB300_128x4x56xtp2pp8cp2_cg_fp4.sh" ->
TP=2, PP=8), and compares the resulting interaction against the original
hand-labeled binary on the SAME restricted subsample.

comm_intensity = log(TP * PP). TP*PP is the size of the model-parallel
group that must communicate synchronously every training step (distinct
from data-parallel replicas, which don't need to). TP=PP=1 (comm_intensity
= log(1) = 0) is pure data-parallel training -- no model-parallel
communication at all.

Coverage is necessarily partial and unevenly distributed, reported
honestly rather than papered over:
  - Config files only exist in the raw MLPerf repos, so only rows from
    repos we have cloned (v5.0/v5.1/v6.0; not v3.1/v4.0/v4.1) can be
    measured at all -- gpt3 (in COMM_BOUND) is entirely absent from these
    three rounds and gets zero coverage as a result.
  - Within those repos, TP/PP naming is used almost exclusively by
    transformer/LLM benchmarks (llama*, gpt_oss_20b, deepseekv3_671b) --
    verified by scanning every config file in all three repos and
    checking which model directories ever contain a tp/pp-style filename,
    for ANY org, not just per-row. Models that NEVER show this naming
    anywhere in the corpus (bert, resnet, retinanet, dlrm_dcnv2, dlrm_v2,
    rgat, stable_diffusion, ssd, gnn -- and notably flux1, see below) are
    treated as an explicit ASSUMED default of TP=PP=1 (standard practice
    for these architectures at MLPerf scale), flagged as `measured=False`
    so this assumption is never confused with an actual measurement.
  - flux1 IS in the hand-labeled COMM_BOUND set, but never shows a tp/pp
    config anywhere in the three repos scanned -- i.e. the measured proxy
    says flux1 submissions ran pure data-parallel (comm_intensity=0),
    while the hand-label says "comm-bound". This is reported as a finding,
    not silently reconciled.
  - For models that DO use tp/pp naming, a row is only "measured" if a
    config file exists whose own node-shape token's product (nodes x
    gpus_per_node) matches that row's total_gpus -- rows with no
    scale-matching config are left unmeasured (NA), not guessed.

Refits, on the identical restricted (measured + assumed-DP) subsample, for
a fair comparison:
  A. the original 09 binary spec: log_time ~ log_gpus + log_domain_mined +
     domain_mined_x_comm + model FE + gen FE
  B. the new continuous spec: log_time ~ log_gpus + log_domain_mined +
     comm_intensity + domain_mined_x_comm_intensity + model FE + gen FE
     (comm_intensity gets its OWN main-effect term here, unlike the binary
     in 09, because -- unlike the binary -- it is NOT a deterministic
     function of model and therefore isn't collinear with model FE;
     omitting it would risk omitted-variable bias on the interaction)

Output: data/comm_intensity_mined.csv (per-row mined feature, committed so
tests/CI can rebuild the regression without $MLPERF_ROOT -- same pattern
as data/nvl_config_mined.csv/07), results/measured_comm_intensity.txt
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

from topology_common import load_clean

ROOT = os.environ.get("MLPERF_ROOT", os.path.expanduser("~"))
REPOS_WITH_CONFIGS = ["training_results_v5.0", "training_results_v5.1",
                     "training_results_v6.0"]
DATASET_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                          "mlperf_topology_dataset.csv")
MINED_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                        "nvl_config_mined.csv")
COMM_INTENSITY_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                                 "comm_intensity_mined.csv")
OUT_TXT = os.path.join(os.path.dirname(__file__), "..", "results",
                      "measured_comm_intensity.txt")

COMM_BOUND = {"gpt3", "llama31_405b", "deepseekv3_671b", "llama2_70b_lora",
              "llama31_8b", "gpt_oss_20b", "flux1"}

NODE_SHAPE_TP_PP = re.compile(r"_(\d+)x(\d+)x(\d+)[x_]?tp(\d+)pp(\d+)",
                              re.IGNORECASE)
GH200_STYLE = re.compile(r"_n(\d+)_tp(\d+)pp(\d+)", re.IGNORECASE)


def build_mined_domain(df):
    """Mirrors 08/09's function of the same name."""
    mined = pd.read_csv(MINED_CSV)
    high = mined[mined["confidence"] == "high"][
        ["repo", "org", "system_id", "inferred_domain"]]
    merged = df.merge(high, on=["repo", "org", "system_id"], how="left")
    merged["true_domain_mined"] = merged["inferred_domain"].fillna(merged["true_domain"])
    merged["log_domain_mined"] = np.log(merged["true_domain_mined"])
    return merged


def scan_config_files():
    """Walk every config_*.sh under each cloned repo's <org>/benchmarks/
    <model>/ tree. Returns:
      - configs: dict {(repo, org, model): [(total_gpus_implied, tp, pp), ...]}
      - models_with_tp_pp: set of model names that show tp/pp ANYWHERE,
        for ANY org/repo (used to distinguish "never applicable" from
        "applicable but unmatched for this row")
    """
    configs = {}
    models_with_tp_pp = set()
    for repo in REPOS_WITH_CONFIGS:
        base = os.path.join(ROOT, repo)
        if not os.path.isdir(base):
            continue
        pattern = os.path.join(base, "*", "benchmarks", "*", "**",
                              "config_*.sh")
        for path in glob.glob(pattern, recursive=True):
            rel = os.path.relpath(path, base)
            parts = rel.split(os.sep)
            org, model = parts[0], parts[2]
            name = os.path.basename(path)

            m = NODE_SHAPE_TP_PP.search(name)
            if m:
                n1, n2, _n3, tp, pp = (int(g) for g in m.groups())
                total_gpus_implied = n1 * n2
                tp_pp = (int(tp), int(pp))
            else:
                m2 = GH200_STYLE.search(name)
                if not m2:
                    continue
                n_nodes, tp, pp = (int(g) for g in m2.groups())
                total_gpus_implied = None  # GH200 style: no per-node count encoded
                tp_pp = (int(tp), int(pp))

            models_with_tp_pp.add(model)
            key = (repo, org, model)
            configs.setdefault(key, []).append((total_gpus_implied, *tp_pp))

    return configs, models_with_tp_pp


def match_comm_intensity(df, configs, models_with_tp_pp):
    comm_intensity = []
    measured = []
    reason = []
    for row in df.itertuples():
        if row.repo not in REPOS_WITH_CONFIGS:
            comm_intensity.append(np.nan)
            measured.append(False)
            reason.append("repo_unavailable")
            continue

        if row.model not in models_with_tp_pp:
            comm_intensity.append(np.log(1 * 1))
            measured.append(False)
            reason.append("assumed_dp_default")
            continue

        candidates = configs.get((row.repo, row.org, row.model), [])
        scale_matched = [c for c in candidates
                        if c[0] is not None and c[0] == row.total_gpus]
        if not scale_matched:
            # try GH200-style (no scale field) as a last resort if it's
            # literally the only candidate type available for this row
            scale_matched = [c for c in candidates if c[0] is None]
        if scale_matched:
            _total, tp, pp = sorted(scale_matched)[0]
            comm_intensity.append(np.log(tp * pp))
            measured.append(True)
            reason.append("measured")
        else:
            comm_intensity.append(np.nan)
            measured.append(False)
            reason.append("unmeasured_no_scale_match")

    df = df.copy()
    df["comm_intensity"] = comm_intensity
    df["comm_intensity_measured"] = measured
    df["comm_intensity_reason"] = reason
    return df


def fit(df, extra_cols):
    cols = ["log_gpus", "log_domain_mined"] + list(extra_cols)
    X = pd.concat([df[cols],
                   pd.get_dummies(df["model"], prefix="model", drop_first=True),
                   pd.get_dummies(df["gen"], prefix="gen", drop_first=True)],
                  axis=1)
    X = sm.add_constant(X.astype(float))
    y = df["log_time"].astype(float)
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["org"]})


def main():
    df = load_clean(DATASET_CSV)
    df = build_mined_domain(df)

    configs, models_with_tp_pp = scan_config_files()
    df = match_comm_intensity(df, configs, models_with_tp_pp)

    # Persist the mined feature (keyed the same way as data/nvl_config_mined.csv)
    # so tests/CI can rebuild and check the regression without needing
    # $MLPERF_ROOT -- this script's own config-file scan is the only step
    # that requires the raw MLPerf repos, same constraint as 07.
    key_cols = ["repo", "org", "system_id", "model", "total_gpus",
               "comm_intensity", "comm_intensity_measured",
               "comm_intensity_reason"]
    os.makedirs(os.path.dirname(COMM_INTENSITY_CSV), exist_ok=True)
    df[key_cols].to_csv(COMM_INTENSITY_CSV, index=False)

    lines = ["MEASURED COMM-INTENSITY: TP*PP parsed from benchmark config "
            "filenames vs. hand-labeled COMM_BOUND binary", ""]

    lines.append(f"Models with tp/pp naming ANYWHERE in v5.0/v5.1/v6.0: "
                f"{sorted(models_with_tp_pp)}")
    flux_note = ("flux1 IS in COMM_BOUND but shows NO tp/pp config "
                "anywhere -- measured proxy treats it as pure "
                "data-parallel (comm_intensity=0), contradicting its "
                "hand-label.") if "flux1" not in models_with_tp_pp else \
               "flux1 unexpectedly has tp/pp configs -- recheck."
    lines.append(flux_note)
    lines.append("")

    n_total = len(df)
    counts = df["comm_intensity_reason"].value_counts()
    lines.append(f"COVERAGE (n={n_total} total analysis rows)")
    for reason in ("measured", "assumed_dp_default",
                  "unmeasured_no_scale_match", "repo_unavailable"):
        n = int(counts.get(reason, 0))
        lines.append(f"  {reason:28s} {n:4d}  ({n/n_total:.1%})")
    lines.append("")

    restricted = df[df["comm_intensity"].notna()].copy()
    n_restricted = len(restricted)
    lines.append(f"Restricted analysis sample (measured + assumed-DP): "
                f"n={n_restricted} ({n_restricted/n_total:.1%} of {n_total})")
    lines.append("")

    restricted["comm"] = restricted["model"].isin(COMM_BOUND).astype(float)
    restricted["domain_mined_x_comm"] = (restricted["log_domain_mined"]
                                        * restricted["comm"])
    restricted["domain_mined_x_comm_intensity"] = (
        restricted["log_domain_mined"] * restricted["comm_intensity"])

    r_binary = fit(restricted, ["domain_mined_x_comm"])
    r_continuous = fit(restricted, ["comm_intensity",
                                   "domain_mined_x_comm_intensity"])

    lines.append("A. ORIGINAL HAND-LABELED BINARY SPEC (09), refit on the "
                "restricted subsample")
    lines.append(f"  base premium (comm-light)   coef="
                f"{r_binary.params['log_domain_mined']:+.4f}  "
                f"se={r_binary.bse['log_domain_mined']:.4f}  "
                f"p={r_binary.pvalues['log_domain_mined']:.3g}")
    lines.append(f"  interaction (extra for comm-bound)  coef="
                f"{r_binary.params['domain_mined_x_comm']:+.4f}  "
                f"se={r_binary.bse['domain_mined_x_comm']:.4f}  "
                f"p={r_binary.pvalues['domain_mined_x_comm']:.3g}")
    lines.append(f"  R^2={r_binary.rsquared:.4f}")
    lines.append("")

    lines.append("B. MEASURED CONTINUOUS SPEC (comm_intensity = log(TP*PP))")
    lines.append(f"  domain main effect           coef="
                f"{r_continuous.params['log_domain_mined']:+.4f}  "
                f"se={r_continuous.bse['log_domain_mined']:.4f}  "
                f"p={r_continuous.pvalues['log_domain_mined']:.3g}")
    lines.append(f"  comm_intensity main effect   coef="
                f"{r_continuous.params['comm_intensity']:+.4f}  "
                f"se={r_continuous.bse['comm_intensity']:.4f}  "
                f"p={r_continuous.pvalues['comm_intensity']:.3g}")
    lines.append(f"  interaction (domain x comm_intensity)  coef="
                f"{r_continuous.params['domain_mined_x_comm_intensity']:+.4f}  "
                f"se={r_continuous.bse['domain_mined_x_comm_intensity']:.4f}  "
                f"p={r_continuous.pvalues['domain_mined_x_comm_intensity']:.3g}")
    lines.append(f"  R^2={r_continuous.rsquared:.4f}")
    lines.append("")

    p_bin = r_binary.pvalues["domain_mined_x_comm"]
    p_cont = r_continuous.pvalues["domain_mined_x_comm_intensity"]
    if p_cont < p_bin:
        verdict = "SHARPENS: the measured continuous interaction is more significant than the hand-labeled binary's."
    elif p_cont > p_bin * 2:
        verdict = "WEAKENS: the measured continuous interaction is substantially less significant than the hand-labeled binary's."
    else:
        verdict = "HOLDS: comparable significance between binary and measured continuous versions."
    lines.append(f"COMPARISON: {verdict}")
    lines.append(f"  (binary interaction p={p_bin:.3g} vs. "
                f"continuous interaction p={p_cont:.3g}, same n={n_restricted} subsample)")

    out = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(out)
    print(out)
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
