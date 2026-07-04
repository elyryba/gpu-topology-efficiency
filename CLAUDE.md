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
- `scripts/09_discount_function.py` — workload-conditional (comm-bound/comm-light) monotone discount function on the evidence-augmented domain feature, validated on the v5.1-v6.0 temporal holdout
- `scripts/10_inference_hardening.py` — wild cluster bootstrap-t (wildboottest package, Rademacher, 9999 reps) on both headline premiums, no MLPERF_ROOT needed
- `scripts/11_measured_comm_intensity.py` — mines TP/PP degree from benchmark config filenames -> continuous comm-intensity proxy; needs $MLPERF_ROOT to re-mine, but its output (`data/comm_intensity_mined.csv`) is committed
- `data/comm_intensity_mined.csv` — per-row mined TP*PP comm-intensity feature (11); `comm_intensity_reason` column distinguishes measured / assumed_dp_default / unmeasured_no_scale_match / repo_unavailable
- `docs/market_data_recon.md` — reconnaissance (no data collected) of GPU rental pricing sources; flags vast.ai (needs written permission) and GPUPerHour (needs paid license) as off-limits, clears AWS/GCP/Azure/OCI's official pricing APIs
- `scripts/12_market_price_collection.py` — collects current on-demand pricing from AWS/Azure/OCI (the cleared sources only; never touches vast.ai/GPUPerHour); manually annotates topology per SKU from provider docs, same discipline as 07
- `scripts/13_hedonic_comparison.py` — market price premium vs. measured throughput premium (09's comm-bound curve), per domain doubling
- `data/market_prices_snapshot.csv` — point-in-time GPU pricing snapshot (12); has a `collected_at` column since none of the three APIs expose historical pricing
- `tests/test_reproduction.py` — pins the headline numbers below; run via pytest
- `.github/workflows/reproduce.yml` — CI: runs 02-06 + pytest on push/PR and monthly (living-validation hook)
- `results/*.txt` — all regression outputs
- `LICENSE` (MIT, this repo's code) / `DATA_LICENSE` (Apache 2.0 attribution for MLPerf source data)

## Commands
- Setup: `bash setup.sh` (creates venv, installs deps, runs 02-06)
- Run everything (02-06 only, no MLPerf clone needed): `cd scripts && for s in 02 03 04 05 06; do python3 ${s}_*.py; done`
- Mining (07, needs $MLPERF_ROOT with v5.0/v5.1/v6.0 cloned): `MLPERF_ROOT=~/mlperf python3 scripts/07_mine_nvl_configs.py`
- Refit + discount function (08/09, no MLPerf clone needed, just the committed CSVs): `python3 scripts/08_refit_with_mined_domains.py && python3 scripts/09_discount_function.py`
- Inference hardening (10, no MLPerf clone needed): `python3 scripts/10_inference_hardening.py`
- Measured comm-intensity (11, needs $MLPERF_ROOT to re-mine; skip and just use the committed data/comm_intensity_mined.csv otherwise): `MLPERF_ROOT=~/mlperf python3 scripts/11_measured_comm_intensity.py`
- Market pricing (12/13, 12 hits live AWS/Azure/OCI APIs -- never run in CI, re-run manually for a fresh snapshot; 13 only needs the committed CSV): `python3 scripts/12_market_price_collection.py && python3 scripts/13_hedonic_comparison.py`
- Tests: `pytest tests/test_reproduction.py`
- Deps: pandas, numpy, statsmodels, pytest, wildboottest (exact pins in requirements.txt)

## Key findings so far (do not regress these)
- Topology premium (reparametrized: log_time ~ log_total_gpus + log_domain + FEs):
  v2 corrected = -0.096 (p=0.026 org-clustered; wild cluster bootstrap p=0.028, holds --
  see item 10 below); pre-Blackwell clean subsample = -0.30 (p=0.0006)
- Sign survives 29 leave-one-out refits, temporal holdout (OOS R²=0.886), Huber/quantile estimators
- Continuous inter-node bandwidth (06): does NOT replace domain size (S1/S2 fail);
  but bandwidth×comm-bound interaction is -0.176 (p=1e-4) — strongest mechanism evidence
- Identification caveat: B200/B300 and AMD gens have ZERO within-gen domain variation;
  premium identified from GB200/GB300 sub-rack rows, H100 variants, thin older gens
- Mined-domain evidence (07/08, DONE — see open item 1): of the 92 GB200/GB300 sub-rack
  rows, all 92 match a mining entry (0 unmatched); 54 are high confidence (confirms the
  categorical NVL72 cap), 3 medium + 35 low = 38 cap-only. 54 + 38 = 92, verified exactly
  (results/refit_mined_domains.txt ACCOUNTING section). An earlier mining pass (commit
  cbccc13, before two classifier fixes in 4681925) reported 41/51 for this same 92-row
  set — superseded, do not cite it. Refitting with mined domains where available: premium
  strengthens to **-0.131 (p=0.0001)**.
- Sensitivity check (drop the 38 cap-only rows entirely) gives -0.338 (p=0.0002, n=931),
  but a Cook's-distance leverage audit (results/refit_mined_domains.txt LEVERAGE AUDIT)
  shows only 1 of the top 10 highest-leverage rows in that fit is even a mined GB200/GB300
  row (HPE GB300 ngpu72, and its mined domain [72] matches what the cap already assumed —
  not new information); the other 9 are unrelated thin-generation single-system leverage
  points (tinycorp, Dell, TTA, JuniperNetworks, Ailiverse, Fujitsu, NVIDIA — MI300X/RTX/
  A100/L40S). **Do not cite -0.338 as confirmation that mined evidence strengthens the
  premium** — it's mostly an artifact of which thin-generation rows dominate the fixed
  effects once the cap-only rows are dropped, not the mined domain values themselves.
  Treat it as a sensitivity bound, not a second estimate of the effect.
- Real tray-vs-fabric trap caught during mining: Nebius's GB300 `system_name` literally
  says "NVL4", but that number equals the submission's own accelerators_per_node in all 3
  occurrences — a compute-tray label, not the rack fabric. Classify by cross-checking the
  matched NVL number against accelerators_per_node, not by a qualifier-word list (Nebius
  never writes "tray"/"module" — a word-list check misses this).
- Discount function (09, DONE — see open item 5): workload-conditional, evidence-augmented
  domain feature (from 08), two log-linear curves (comm-light slope=-0.084 p=0.058;
  comm-bound slope=-0.162, i.e. base + a -0.078 interaction). Monotonicity enforcement is
  active (clips either slope to 0 if it ever comes out positive) but did not trigger — both
  slopes were already <=0. Temporal holdout (v3.1-v5.0 -> v5.1-v6.0): OOS R²=0.892,
  comparable to 05's 0.886. Discount table at domain={4,8,16,36,72} (relative to domain=4):
  comm-light down to 0.784 [0.610,1.009] at domain=72; comm-bound down to 0.627
  [0.508,0.773]. All 5 requested domain sizes are interpolation (within observed range) for
  both workload classes — not a coincidence, they're the canonical tray/rack sizes present
  throughout the dataset. Comm-light's band crosses 1.0 at domain=72 (p=0.058, marginal);
  comm-bound is comfortably significant throughout.
- Discount function flux1-label robustness (09, DONE — follow-up to referee point 4):
  11 found flux1 sits in COMM_BOUND but has zero measured TP/PP parallelism anywhere in the
  corpus. Moving it to comm-light and rebuilding the whole discount table moves every cell
  by at most 0.017 (comm-bound at domain=72: 0.627 -> 0.610) — a small fraction of the
  ~0.27-0.40-wide confidence bands. NOT MATERIAL: the COMM_BOUND-based table remains the
  headline (results/discount_function.txt ROBUSTNESS section has the full comparison and
  the verdict logic: material iff largest cell delta > 25% of the largest band width).
- Inference hardening (10, DONE — referee point 3): wild cluster bootstrap-t (Rademacher,
  9999 reps, clustered by org, wildboottest package) on both headline premiums. Categorical
  cap: asymptotic p=0.026 -> bootstrap p=0.028 (holds). Mined-augmented: asymptotic p=0.0001
  -> bootstrap p=0.0067 (holds, but the asymptotic p understated uncertainty by ~67x).
  Neither headline flips, but treat every asymptotic p in this repo as a lower bound on
  the true p, not the true p, given only 41 clusters. wildboottest gotcha: pass cluster as
  integer-coded IDs (pd.factorize), not raw org-name strings -- strings crash its numba JIT
  with a typing error.
- Measured comm-intensity (11, DONE — referee point 4): mined TP*PP degree from benchmark
  config filenames as a continuous comm-intensity proxy (log(TP*PP)), replacing the
  hand-labeled COMM_BOUND binary in 09's interaction. Coverage: of 969 rows, 361 (37%) have
  no available raw repo to check (v3.1/v4.0/v4.1 not cloned) and are excluded; of the rest,
  303 (31% of 969) measured directly, 240 (25%) assumed pure-data-parallel (model never
  shows tp/pp naming anywhere in the corpus -- explicit assumption, not measurement), 65 (7%)
  unmeasured (tp/pp-style model but no scale-matching config found). Restricted sample
  (measured+assumed-DP) n=543. On that identical subsample: hand-labeled binary interaction
  p=0.0229 vs. measured continuous interaction p=0.0040 -- SHARPENS, ~5.7x more significant.
  comm_intensity's own main effect is positive (p=0.13, not significant) while its
  interaction with domain is negative and significant -- a coherent mechanism story the
  binary couldn't separate out. Also surfaced: flux1 is in COMM_BOUND but has ZERO tp/pp
  configs anywhere in the corpus -- the measured proxy disagrees with its hand-label.
- Hedonic price comparison (12/13, DONE — see open item 3): market snapshot (n=13 SKUs,
  AWS+Azure+OCI only, per docs/market_data_recon.md's clearance) shows price ~x1.76-1.77
  per domain doubling (pooled vs. provider-FE OLS), vs. the measured throughput premium of
  x0.894 time per doubling (09 comm-bound curve). Combined cost-to-train multiplier ~x1.58:
  the market's price spread is LARGER than the measured performance spread -- paying for a
  bigger domain costs more overall even after finishing faster, in this snapshot. CRITICAL:
  zero within-GPU-model domain variation in the collected data (no provider sells the same
  GPU at multiple topology classes) -- every number here is confounded with GPU generation,
  unlike 09's model/gen-FE-isolated regression. Treat as a first-order, descriptive
  comparison (direction, not precise multiplier). vast.ai (richest topology schema found)
  and GPUPerHour remain untouched pending permission/license -- see docs/market_data_recon.md.

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
- Before citing any "drop rows and refit" sensitivity number as confirmation, check Cook's
  distance on that specific fit — a sensitivity result can look like it's driven by the
  rows you cared about when it's actually a handful of unrelated thin-generation leverage
  points (see the -0.338 caveat above; 07/08's leverage audit is the template)
- Report asymptotic clustered p-values as a lower bound, not the final word, given only 41
  org clusters — wild cluster bootstrap them (10_inference_hardening.py is the template)
  before treating a "marginal" result as settled either way
- Never conflate `comm_intensity_reason == "assumed_dp_default"` with an actual measurement
  (data/comm_intensity_mined.csv, 11) — only "measured" rows come from a parsed config file;
  the assumed-DP default is a documented modeling choice for models that never show tp/pp
  naming anywhere in the corpus, not evidence

## Open work (priority order)
1. ~~Mine per-submission NVL config from system-description free text for GB200/GB300~~ —
   DONE (07_mine_nvl_configs.py / 08_refit_with_mined_domains.py). 54 of 92 sub-rack rows
   now evidence-backed, 0 conflicts, 38 remain cap-only; premium strengthens under mined
   evidence. Follow-up if revisited: the 38 cap-only rows and the "low"-confidence org-level
   filename fallback (not tied to a specific system_id) are the remaining soft spots.
2. Source GH200/NVL32 deployment domain sizes; currently on uncorrected proxy
3. ~~Hedonic pricing regression on GPU rental market data — compare performance premium vs
   price premium~~ — DONE with AWS/Azure/OCI only (12_market_price_collection.py /
   13_hedonic_comparison.py); market price spread (~x1.76-1.77/doubling) exceeds the
   measured throughput spread (x0.894/doubling), combined cost multiplier ~x1.58. Follow-up
   if revisited: this used only the 3 sources docs/market_data_recon.md cleared with zero
   ToS friction, at n=13 SKUs with no within-GPU-model domain variation (confounded with
   GPU generation) -- vast.ai (richest topology schema: bw_nvlink, num_gpus per listing,
   real transaction-like prices) would let this be redone properly, but needs written
   permission from vast.ai first per their ToS (see docs/market_data_recon.md). GCP was
   also skipped (needs an API key not configured here) -- revisit if credentials appear.
4. Check MLCommons results-usage/trademark policy before any commercial use
5. ~~Continuous monotone discount function, validated on held-out MLPerf rounds~~ — DONE
   (09_discount_function.py). Workload-conditional (comm-bound/comm-light), built on the
   evidence-augmented domain feature, monotonicity explicitly enforced (didn't need to
   trigger), OOS R²=0.892 on the v5.1-v6.0 holdout. Follow-up if revisited: the comm-light
   curve is only marginally significant (p=0.058) and its band crosses 1.0 at domain=72 —
   don't present that curve's discount as precisely sized, same caveat as the pooled
   premium's marginal significance (item 1 above).
