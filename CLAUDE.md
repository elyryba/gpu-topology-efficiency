# Project: GPU Cluster Topology → Training Efficiency

## What this is
An empirical test of whether GPU cluster topology (NVLink domain size vs.
number of networked domains) measurably affects realized training throughput,
using public MLPerf Training results (v3.1–v6.0, 969 clean observations).
Goal: evolve from validated research finding into a topology-adjusted
compute-pricing discount function.

## Repo layout
- `data/mlperf_topology_dataset.csv` — extracted dataset (included; regressions reproduce without cloning MLPerf repos)
- `data/nvl_config_mined.csv` — per-submission mined NVLink domain evidence for GB200/GB300 (07); confidence-graded high/medium/low/none
- `scripts/topology_common.py` — shared cleaning + fitting; ALL model scripts import from here so samples stay identical
- `scripts/01_extract_mlperf_data.py` — raw MLPerf repos → CSV (needs repos cloned; set $MLPERF_ROOT)
- `scripts/02..06_*.py` — models, in order: baseline, corrected NVL72 domain, rigor checks, deep checks, continuous bandwidth
- `scripts/07_mine_nvl_configs.py` — mines system_name/hw_notes/sw_notes/accelerator_interconnect_topology + config filenames for direct domain evidence (needs repos cloned; set $MLPERF_ROOT)
- `scripts/08_refit_with_mined_domains.py` — refits the topology premium using mined high-confidence domains where available, categorical cap otherwise
- `tests/test_reproduction.py` — pins the headline numbers below; run via pytest
- `.github/workflows/reproduce.yml` — CI: runs 02-06 + pytest on push/PR and monthly (living-validation hook)
- `results/*.txt` — all regression outputs
- `LICENSE` (MIT, this repo's code) / `DATA_LICENSE` (Apache 2.0 attribution for MLPerf source data)

## Commands
- Setup: `bash setup.sh` (creates venv, installs deps, runs 02-06)
- Run everything (02-06 only, no MLPerf clone needed): `cd scripts && for s in 02 03 04 05 06; do python3 ${s}_*.py; done`
- Mining + refit (07/08, needs $MLPERF_ROOT with v5.0/v5.1/v6.0 cloned): `MLPERF_ROOT=~/mlperf python3 scripts/07_mine_nvl_configs.py && python3 scripts/08_refit_with_mined_domains.py`
- Tests: `pytest tests/test_reproduction.py`
- Deps: pandas, numpy, statsmodels, pytest (exact pins in requirements.txt)

## Key findings so far (do not regress these)
- Topology premium (reparametrized: log_time ~ log_total_gpus + log_domain + FEs):
  v2 corrected = -0.096 (p=0.026 org-clustered); pre-Blackwell clean subsample = -0.30 (p=0.0006)
- Sign survives 29 leave-one-out refits, temporal holdout (OOS R²=0.886), Huber/quantile estimators
- Continuous inter-node bandwidth (06): does NOT replace domain size (S1/S2 fail);
  but bandwidth×comm-bound interaction is -0.176 (p=1e-4) — strongest mechanism evidence
- Identification caveat: B200/B300 and AMD gens have ZERO within-gen domain variation;
  premium identified from GB200/GB300 sub-rack rows, H100 variants, thin older gens
- Mined-domain evidence (07/08, DONE — see open item 1): of the 92 GB200/GB300 sub-rack
  rows, 54 now have direct high-confidence evidence confirming the categorical NVL72 cap,
  0 conflict with it, 38 remain cap-only (no resolving evidence). Refitting with mined
  domains where available: premium strengthens to **-0.131 (p=0.0001)**, and further to
  -0.338 (p=0.0002, n=931) if the 38 cap-only rows are dropped entirely — direct evidence
  points the same direction as the categorical assumption, more strongly, not against it.
- Real tray-vs-fabric trap caught during mining: Nebius's GB300 `system_name` literally
  says "NVL4", but that number equals the submission's own accelerators_per_node in all 3
  occurrences — a compute-tray label, not the rack fabric. Classify by cross-checking the
  matched NVL number against accelerators_per_node, not by a qualifier-word list (Nebius
  never writes "tray"/"module" — a word-list check misses this).

## Conventions
- Always use org-clustered standard errors for headline claims; nonrobust only for comparison
- Always report n, and which subsample, next to every coefficient
- New model scripts must import load_clean/fit helpers from topology_common.py
- New results go in results/ as plain text; update README.md tables when findings change
- Never present the -0.30 clean-subsample number without the pooled -0.096 alongside it
- Environment is pinned exactly in requirements.txt (pandas==2.3.3, numpy==2.0.2,
  statsmodels==0.14.6, pytest==8.4.2), not floor-constrained — regression outputs (esp.
  deep_checks.txt section E, RLM/QuantReg) are sensitive to numpy/BLAS backend. Re-`pip
  freeze` and update the pins deliberately (not silently) if a dependency needs to move
- Never use medium/low-confidence mined domain values (data/nvl_config_mined.csv) in a
  headline spec — only `confidence == "high"` overrides the categorical cap. Medium/low
  exist for transparency/audit, not as model inputs (see 07/08 docstrings)
- Before any commercial use, MLCommons' results-usage/trademark policy (separate from the
  Apache 2.0 license on the source repos) still needs checking — see DATA_LICENSE

## Open work (priority order)
1. ~~Mine per-submission NVL config from system-description free text for GB200/GB300~~ —
   DONE (07_mine_nvl_configs.py / 08_refit_with_mined_domains.py). 54 of 92 sub-rack rows
   now evidence-backed, 0 conflicts, 38 remain cap-only; premium strengthens under mined
   evidence. Follow-up if revisited: the 38 cap-only rows and the "low"-confidence org-level
   filename fallback (not tied to a specific system_id) are the remaining soft spots.
2. Source GH200/NVL32 deployment domain sizes; currently on uncorrected proxy
3. Hedonic pricing regression on GPU rental market data (e.g. vast.ai price distributions) — compare performance premium vs price premium
4. Check MLCommons results-usage/trademark policy before any commercial use
5. Continuous monotone discount function, validated on held-out MLPerf rounds
