# Project: GPU Cluster Topology → Training Efficiency

## What this is
An empirical test of whether GPU cluster topology (NVLink domain size vs.
number of networked domains) measurably affects realized training throughput,
using public MLPerf Training results (v3.1–v6.0, 969 clean observations).
Goal: evolve from validated research finding into a topology-adjusted
compute-pricing discount function.

## Repo layout
- `data/mlperf_topology_dataset.csv` — extracted dataset (included; regressions reproduce without cloning MLPerf repos)
- `scripts/topology_common.py` — shared cleaning + fitting; ALL model scripts import from here so samples stay identical
- `scripts/01_extract_mlperf_data.py` — raw MLPerf repos → CSV (needs repos cloned; set $MLPERF_ROOT)
- `scripts/02..06_*.py` — models, in order: baseline, corrected NVL72 domain, rigor checks, deep checks, continuous bandwidth
- `results/*.txt` — all regression outputs

## Commands
- Setup: `bash setup.sh` (creates venv, installs deps)
- Run everything: `cd scripts && for s in 02 03 04 05 06; do python3 ${s}_*.py; done`
- Deps: pandas, numpy, statsmodels (pinned in requirements.txt)

## Key findings so far (do not regress these)
- Topology premium (reparametrized: log_time ~ log_total_gpus + log_domain + FEs):
  v2 corrected = -0.096 (p=0.026 org-clustered); pre-Blackwell clean subsample = -0.30 (p=0.0006)
- Sign survives 29 leave-one-out refits, temporal holdout (OOS R²=0.886), Huber/quantile estimators
- Continuous inter-node bandwidth (06): does NOT replace domain size (S1/S2 fail);
  but bandwidth×comm-bound interaction is -0.176 (p=1e-4) — strongest mechanism evidence
- Identification caveat: B200/B300 and AMD gens have ZERO within-gen domain variation;
  premium identified from GB200/GB300 sub-rack rows, H100 variants, thin older gens

## Conventions
- Always use org-clustered standard errors for headline claims; nonrobust only for comparison
- Always report n, and which subsample, next to every coefficient
- New model scripts must import load_clean/fit helpers from topology_common.py
- New results go in results/ as plain text; update README.md tables when findings change
- Never present the -0.30 clean-subsample number without the pooled -0.096 alongside it
- Environment is pinned exactly in requirements.txt (pandas==2.3.3, numpy==2.0.2,
  statsmodels==0.14.6), not floor-constrained — regression outputs (esp. deep_checks.txt
  section E, RLM/QuantReg) are sensitive to numpy/BLAS backend. Re-`pip freeze` and update
  the pins deliberately (not silently) if a dependency needs to move

## Open work (priority order)
1. Mine per-submission NVL config from system-description free text for GB200/GB300 (92 sub-rack rows carry most Blackwell-era identification)
2. Source GH200/NVL32 deployment domain sizes; currently on uncorrected proxy
3. Hedonic pricing regression on GPU rental market data (e.g. vast.ai price distributions) — compare performance premium vs price premium
4. Check MLCommons results-usage/trademark policy before any commercial use
5. Continuous monotone discount function, validated on held-out MLPerf rounds
