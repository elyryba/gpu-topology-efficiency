# GPU Cluster Topology → Training Efficiency (full rebuild)

[![Reproduce](https://github.com/elyryba/gpu-topology-efficiency/actions/workflows/reproduce.yml/badge.svg)](https://github.com/elyryba/gpu-topology-efficiency/actions/workflows/reproduce.yml)

Independent, from-scratch rebuild and extension of the MLPerf® topology-efficiency
analysis. Everything below was re-derived by running the code in this repo
against `data/mlperf_topology_dataset.csv` (1,041 extracted MLPerf® Training
submissions, v3.1–v6.0, Closed and Open divisions; 969 after quality filtering).

MLPerf® is a registered trademark of the MLCommons® Association. Source data
is MLCommons MLPerf® Training results (Apache License 2.0) -- see
[DATA_LICENSE](DATA_LICENSE) for the full citation and attribution terms.
This repo's own code is [MIT-licensed](LICENSE). **This project is an
independent, third-party analysis: it is not endorsed by, sponsored by, or
affiliated with MLCommons, and none of its findings have been verified by
the MLCommons Association.**

## Pipeline

```
scripts/
  topology_common.py               shared cleaning + fitting (guarantees v1/v2 use identical samples)
  01_extract_mlperf_data.py        raw MLPerf repos -> dataset CSV (needs repos cloned; CSV included)
  02_topology_efficiency_model.py  v1 baseline (gpus_per_node proxy)
  03_corrected_topology_model.py   v2 corrected (documented NVL72 domain for GB200/GB300)
  04_rigor_checks.py               direct topology-premium test + org-clustered standard errors
  05_deep_checks.py                six reviewer-grade robustness attacks (identification, holdout, estimators, ...)
  06_continuous_bandwidth_model.py continuous inter-node bandwidth vs. categorical domain size
  07_mine_nvl_configs.py           mines per-submission NVLink domain evidence from raw MLPerf repos
  08_refit_with_mined_domains.py   refits the topology premium using mined evidence where available
  09_discount_function.py          workload-conditional discount function, validated on a temporal holdout
  10_inference_hardening.py        wild cluster bootstrap-t on both headline premiums (few-cluster robustness)
  11_measured_comm_intensity.py    TP/PP degree mined from config files -> continuous comm-intensity proxy
```

Run: `bash setup.sh` (creates a venv, installs the pinned `requirements.txt`,
runs scripts 02–06 end to end). See **How to reproduce** below for the full
picture, including 07/08 and tests.

## Reproduction status

Every headline number in the original write-up reproduces exactly from the
included dataset:

| Claim | Original | This rebuild |
|---|---|---|
| Analysis sample | 969 rows, 15 models, 14 gens | 969 / 15 / 14 ✓ |
| v1 log_gpn | -1.0221 (p<1e-80) | -1.0221 ✓ |
| v1 log_nodes | -0.6905 | -0.6905 ✓ |
| v1 R² | 0.957 | 0.9574 ✓ |
| v1 VIF | 1.17 | 1.173 ✓ |
| v2 domain coef | -0.78 | -0.7806 ✓ |
| v2 R² | 0.956 | 0.9562 ✓ |
| v2 VIF | 1.00 | 1.002 ✓ |

One discrepancy: the original claimed v2 robustness coefficients stay in
-0.77 to -0.80 across all 29 refits. This rebuild finds **-0.75 to -0.83**
(extremes: drop-GB300 → -0.832, drop-H100 → -0.754). Still uniformly
p < 0.0001, but the stated range was slightly too tight.

## The correction that matters most (new in this rebuild)

The original compared the domain-size coefficient (-0.78) to the domain-count
coefficient (-0.68) by eye and called the gap "the topology effect." That
comparison is never formally tested. Because
log(total_gpus) = log(domain) + log(n_domains), the model can be
reparametrized so the topology effect gets its own coefficient and p-value:

    log_time ~ b_scale · log(total_gpus) + b_topo · log(domain_size) + FEs

b_topo is exactly: the extra speedup from reaching the *same* total GPU count
via larger NVLink domains. Results (`results/rigor_checks.txt`):

| | v1 proxy | v2 corrected |
|---|---|---|
| Topology premium (b_topo) | -0.332 | **-0.096** |
| p, nonrobust SE | 4e-11 | 5e-5 |
| p, clustered by org (41 clusters) | 2e-4 | **0.026** |

Read plainly: under the architecturally honest domain feature, doubling
NVLink-domain size at fixed total cluster size cuts time-to-train by about
**7%** (2^0.096 − 1), and the evidence for even that is only *marginally*
significant once you stop pretending submissions from the same org are
independent. The original's "-0.78 vs -0.68, both p<0.0001" framing is
technically true and materially misleading about effect size: the correction
shrank the topology premium by ~70% (from 0.33 to 0.10), not the ~24%
implied by "-1.02 → -0.78."

*(41 org clusters is right at the edge where asymptotic cluster-robust
inference is known to get optimistic. `10_inference_hardening.py` checked
this p=0.026 with a 9999-rep wild cluster bootstrap: it holds, at
p=0.028 — see that section below before treating "marginal" as settled.)*

## Standing limitations (inherited + new)

1. The GB200/GB300 domain correction is categorical (documented NVL72 spec),
   not per-submission ground truth; GH200 (NVL32-capable) rows remain on the
   uncorrected proxy.
2. MLPerf submitters co-optimize software, parallelism strategy, and topology
   — this is observational data with non-random topology assignment. The
   premium is an association, not a causal cluster-design elasticity.
3. Residual diagnostics show heavy tails (kurtosis ≈ 15) and mild
   autocorrelation; org-clustered SEs (04) partially address this, but
   quantile or robust regression would be a sensible sensitivity check.
4. Thin generations (TPU-v5p, A100, RTX, L40S) sit near the 8-obs floor.
5. This is not a pricing formula. At a ~7% marginally-significant premium per
   domain doubling, a topology discount function priced off this dataset
   alone would be built on weak footing — the honest conclusion is that
   topology matters *directionally* but MLPerf cannot yet size it precisely
   for rack-scale hardware.

## Deep validation round (05_deep_checks.py)

Six reviewer-grade attacks, all implemented in `scripts/05_deep_checks.py`,
output in `results/deep_checks.txt`:

| Check | Result |
|---|---|
| A. Identification audit | 37% of domain variance survives gen FEs; B200/B300 and all AMD gens contribute zero (constant domains). Premium is identified from GB200/GB300 sub-rack rows (92), H100 variants, and thin older gens. |
| B. Clean-ID subsample (pre-Blackwell, n=506) | Premium **-0.30** (p=0.0006, org-clustered) — larger and stronger where the proxy is architecturally exact. |
| C. Mechanism test | Comm-bound LLM workloads show an extra -0.06 premium on top of -0.06 base; correct sign, individually underpowered (p=0.17). |
| D. Temporal holdout | Early-rounds fit predicts late rounds with out-of-sample R²=0.886; premium persists (early -0.12, late -0.31), same sign both windows. |
| E. Estimator sensitivity | Huber -0.084, median regression -0.090 — the OLS -0.096 is not tail-driven. |
| F. Measurement sensitivity | Median-run-time: -0.087; excluding PCIe/NVL variants: -0.083 (p=0.038). Stable. |

Synthesis: the topology premium's **sign is unambiguous** across every cut
(29 leave-one-out refits, 2 time windows, 3 estimators, 2 outcome
definitions). Its **magnitude is regime-dependent**: ~-0.30 (≈23% per
domain doubling) on pre-Blackwell hardware where identification is clean,
~-0.10 (≈7%) in the pooled corrected sample where Blackwell-era domain
variation is thin. The honest one-line summary: topology effects are real,
directionally certain, and not yet precisely sized for rack-scale hardware
from MLPerf alone.

## Mined NVLink domain evidence (07_mine_nvl_configs.py)

The categorical `min(total_gpus, 72)` cap used above for GB200/GB300 rows is
an architecture-level assumption, not per-submission ground truth. To check
it, `07_mine_nvl_configs.py` independently text-mines each GB200/GB300
submission's raw JSON (`system_name`, `hw_notes`, `sw_notes`,
`accelerator_interconnect_topology`) and benchmark config filenames for
direct domain-size evidence, classified `high` / `medium` / `low` / `none`
confidence (output: `data/nvl_config_mined.csv`, reconciliation report:
`results/nvl_mining_reconciliation.txt`).

Of the 92 GB200/GB300 sub-rack rows (`total_gpus < 72`, the rows that
actually carry Blackwell-era identification):

| | Dataset rows (n=92) | Unique submissions (n=47) |
|---|---|---|
| Direct high-confidence evidence (confirms the cap) | 54 | 31 |
| Conflicting evidence (mined domain ≠ 72) | 0 | 0 |
| Cap-only (no resolving high-confidence evidence) | 38 | 16 |

Zero conflicting rows: every piece of high-confidence evidence found is
consistent with the existing NVL72 cap. One real trap did surface and was
caught before it reached the model: Nebius's GB300 submissions literally
say "NVL4" in `system_name`, but that number equals the submission's own
`accelerators_per_node` in all 3 occurrences — a compute-tray label, not
the rack fabric — so those rows are `medium` confidence, not `high`, and
never override the cap.

## Refit with mined domains (08_refit_with_mined_domains.py)

Rebuilding `true_domain` with the 137 high-confidence mined values
(falling back to the categorical cap everywhere else) and refitting the
same topology-premium spec as 04 (`results/refit_mined_domains.txt`):

| | Topology premium (org-clustered) |
|---|---|
| Categorical cap (04, baseline) | -0.096 (p=0.026) |
| Mined-augmented | **-0.131 (p=0.0001)** |
| Sensitivity: drop the 38 cap-only rows entirely | -0.338 (p=0.0002, n=931) — see caveat below |

Direct evidence points the same direction as the categorical assumption,
and more strongly: the premium gets *larger* in magnitude and *more*
significant once mined evidence replaces the flat cap. Leave-one-
generation-out on the mined-augmented feature holds sign and significance
across all 14 generations (range: -0.10 to -0.19).

**Audited caveat on the -0.338 figure:** a Cook's-distance leverage check
on that sensitivity fit (`results/refit_mined_domains.txt`, LEVERAGE AUDIT
section) found only 1 of its top 10 highest-leverage rows is even a mined
GB200/GB300 evidence row — and its mined domain (72) matches what the
categorical cap already assumed, so it isn't new information. The other 9
are unrelated thin-generation single-system leverage points (tinycorp,
Dell, TTA, JuniperNetworks, Ailiverse, Fujitsu, NVIDIA). **-0.338 is a
sensitivity bound, not a second confirmation of -0.131** — it's mostly an
artifact of which thin generations dominate the fixed effects once the
38 cap-only rows are dropped, not a mined-evidence effect. Also note: the
92/47-row mining reconciliation numbers above (54 high / 38 cap-only)
supersede an earlier pre-classifier-fix pass that reported 41/51 for the
same rows — that number is retired, don't cite it.

## Discount function (09_discount_function.py)

CLAUDE.md open-work item 5: a continuous, monotone, workload-conditional
discount function, built on the evidence-augmented domain feature from 08
(mined high-confidence domain, categorical cap otherwise), with separate
curves for comm-bound and comm-light workloads (`COMM_BOUND`, same set as
05/06) and validated on the same v3.1–v5.0 → v5.1–v6.0 temporal holdout as
05 (out-of-sample R²=0.892, comparable to 05's 0.886). Monotonicity is
explicitly enforced (either slope would be clipped to 0 if it ever came
out positive — it didn't; see `results/discount_function.txt`).

Discount multiplier (time-to-train relative to domain=4, org-clustered 95%
band), for the domain sizes actually observed in this dataset:

| domain | comm-light | comm-bound |
|---|---|---|
| 4 | 1.000 (ref) | 1.000 (ref) |
| 8 | 0.943 [0.888, 1.002] | 0.894 [0.850, 0.940] |
| 16 | 0.890 [0.789, 1.004] | 0.799 [0.723, 0.884] |
| 36 | 0.831 [0.687, 1.007] | 0.701 [0.598, 0.822] |
| 72 | 0.784 [0.610, 1.009] | 0.627 [0.508, 0.773] |

All five points are **interpolation** within the observed domain range for
both workload classes (comm-light: [2, 72], comm-bound: [4, 72]) — these
happen to be the canonical tray/rack sizes present throughout the dataset,
not a coincidence of the table choice. Note the comm-light band at
domain=72 crosses 1.0 (consistent with its p=0.058, only marginally
significant); the comm-bound curve is comfortably significant throughout.

## Inference hardening (10_inference_hardening.py)

Referee point: with only 41 org clusters, asymptotic cluster-robust SEs
(used for every headline number above) are at the edge of where they're
known to become anti-conservative in finite samples. This script re-tests
both headline topology-premium estimates with a wild cluster bootstrap-t
(Rademacher weights, null imposed, 9999 replications, clustered by org --
`results/inference_hardening.txt`):

| | asymptotic p | bootstrap p |
|---|---|---|
| Categorical cap (04) | 0.026 | **0.028** |
| Mined-augmented (08) | 0.0001 | **0.0067** |

Neither headline loses significance. But the gap is worth sitting with:
the mined-augmented estimate's asymptotic p-value understated the true
uncertainty by roughly 67x (0.0001 → 0.0067). The bootstrap doesn't flip
any conclusion in this repo, but it does mean the asymptotic SEs
everywhere else were optimistic, not just for the one headline this
happened to be checked on.

## Measured comm-intensity (11_measured_comm_intensity.py)

Referee point: 09's comm-bound/comm-light split is a hand-labeled binary
(`COMM_BOUND`, 7 fixed model names) standing in for a continuous physical
quantity. This script mines actual tensor-parallel x pipeline-parallel
(TP×PP) degree from MLPerf benchmark config filenames (e.g.
`config_GB300_128x4x56xtp2pp8cp2_cg_fp4.sh` → TP=2, PP=8) as a measured
comm-intensity proxy (`comm_intensity = log(TP×PP)`), and refits 09's
interaction using it instead of the binary, on an identical subsample so
the comparison is apples-to-apples (`results/measured_comm_intensity.txt`,
mined feature: `data/comm_intensity_mined.csv`).

Coverage is partial and unevenly distributed -- reported honestly, not
smoothed over. Of 969 analysis rows: 361 (37%) are from MLPerf rounds we
don't have cloned (v3.1/v4.0/v4.1) and can't be checked at all; of the
rest, 303 (31%) have a directly measured TP/PP value, 240 (25%) are
models that never use TP/PP naming anywhere in the corpus and are
assumed pure data-parallel (TP=PP=1, flagged explicitly, not asserted as
measured), and 65 (7%) use TP/PP-style models but had no scale-matching
config found for that specific row. The restricted analysis sample
(measured + assumed-DP) is 543 rows (56% of 969).

One direct discrepancy surfaced: **flux1 is in the hand-labeled
COMM_BOUND set, but shows zero TP/PP configuration anywhere in the three
repos scanned** -- the measured proxy would call it pure data-parallel,
contradicting its hand-label.

On the 543-row restricted sample, refitting the same interaction spec
with the measured continuous proxy instead of the binary:

| | interaction p-value |
|---|---|
| Original hand-labeled binary (09), same subsample | 0.0229 |
| Measured continuous proxy | **0.0040** |

**The split sharpens** — about 5.7x more significant with the measured
proxy, on the identical rows. The continuous spec also separates two
effects the binary couldn't: `comm_intensity`'s own main effect is
*positive* (higher TP×PP alone tends to be slower, p=0.13, not
significant on its own) while its interaction with domain size is
negative and significant (p=0.004) -- a coherent mechanistic story
(communication-heavy parallelism is costly, but larger NVLink domains
measurably offset that cost) that the binary version couldn't surface.

## How to reproduce

```bash
bash setup.sh                       # venv + pinned deps + runs scripts 02-06
source .venv/bin/activate
pytest tests/test_reproduction.py   # pins the headline numbers above
```

07 and 11 additionally require the raw MLPerf repos cloned locally (08/09/10
only need the committed CSVs, same as 02-06 -- 11's mined feature is also
committed as `data/comm_intensity_mined.csv`, so only re-mining it from
scratch needs the clones):

```bash
mkdir -p ~/mlperf && cd ~/mlperf
git clone --depth 1 https://github.com/mlcommons/training_results_v5.0
git clone --depth 1 https://github.com/mlcommons/training_results_v5.1
git clone --depth 1 https://github.com/mlcommons/training_results_v6.0
cd -
MLPERF_ROOT=~/mlperf python3 scripts/07_mine_nvl_configs.py
python3 scripts/08_refit_with_mined_domains.py
python3 scripts/09_discount_function.py
python3 scripts/10_inference_hardening.py
MLPERF_ROOT=~/mlperf python3 scripts/11_measured_comm_intensity.py
```

CI (`.github/workflows/reproduce.yml`, badge above) runs the pinned
`requirements.txt` install, scripts 02-06, and the pytest suite on every
push/PR and on a monthly schedule, so a dependency drift or a new MLPerf
round dropped into `data/` later gets caught automatically.
