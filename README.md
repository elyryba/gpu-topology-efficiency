# GPU Cluster Topology → Training Efficiency (full rebuild)

Independent, from-scratch rebuild and extension of the MLPerf topology-efficiency
analysis. Everything below was re-derived by running the code in this repo
against `data/mlperf_topology_dataset.csv` (1,041 extracted MLPerf Training
submissions, v3.1–v6.0; 969 after quality filtering).

## Pipeline

```
scripts/
  topology_common.py              shared cleaning + fitting (guarantees v1/v2 use identical samples)
  01_extract_mlperf_data.py       raw MLPerf repos -> dataset CSV (needs repos cloned; CSV included)
  02_topology_efficiency_model.py v1 baseline (gpus_per_node proxy)
  03_corrected_topology_model.py  v2 corrected (documented NVL72 domain for GB200/GB300)
  04_rigor_checks.py              direct topology-premium test + org-clustered standard errors
```

Run:
```bash
pip install pandas numpy statsmodels
cd scripts
python3 02_topology_efficiency_model.py
python3 03_corrected_topology_model.py
python3 04_rigor_checks.py
```

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
