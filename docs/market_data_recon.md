# Market pricing data — reconnaissance (CLAUDE.md open-work item 3)

**Status: reconnaissance only. No data collected, no scraping performed, no
API keys used.** This document researches candidate sources for a future
hedonic price regression testing whether GPU rental markets already price
the topology premium measured in this repo (comm-bound: 0.627
[0.508, 0.773] at domain=72 vs. domain=4; comm-light: 0.784 [0.610, 1.009],
marginal — `results/discount_function.txt`). All findings below are from
reading public documentation and terms-of-service pages, current as of
July 2026.

**Two sources below have real ToS friction. Flagging up front, not
proceeding:**
- **vast.ai**: richest topology schema of anything researched, but its
  general Terms of Service prohibit "systematic retrieval of data... to
  create or compile... a collection, compilation, database or directory
  without written permission" and separately prohibit "web scraping, web
  harvesting, web data extraction" — and does not carve out the official
  API from these restrictions. Building a research dataset via their API
  is exactly the kind of "compilation" this clause describes. **Do not
  proceed without emailing vast.ai for explicit written permission first.**
- **GPUPerHour**: explicitly states "any programmatic access to pricing
  data requires a commercial license agreement" and prohibits scraping
  outright, with IP-blocking enforcement. **Not usable without purchasing
  a commercial license.**

Everything else below has no discovered ToS obstacle for the kind of
public-pricing-API use this project would make of it.

---

## 1. vast.ai

**(a) Topology attributes visible per listing:** Yes, and it's the richest
of anything surveyed. The `search offers` API endpoint
(`docs.vast.ai/api-reference/search/search-offers`) returns per-listing
fields including `bw_nvlink` (NVLink bandwidth, GB/s), `num_gpus`,
`compute_cap`, `bundle_id`, plus 20+ other filterable fields. The console
additionally exposes a dedicated NVLink-bandwidth filter, and vast.ai
separately offers "dedicated multi-node GPU clusters with InfiniBand
networking" as a distinct product line. In principle this lets us build a
`(num_gpus, bw_nvlink)`-based domain-size proxy per listing without any
text-mining — a structural advantage over every other source below, all
of which require joining topology from separate documentation.

**(b) Historical vs. snapshot:** Unconfirmed from documentation alone —
the API appears to expose current live offers (it's a real-time
marketplace, prices float with supply/demand). No historical price-history
endpoint was found in the docs reviewed. Would need to be resolved directly
with vast.ai (see caveat below) before assuming point-in-time snapshots
are the only option; some marketplaces of this kind expose historical
completed-rental data separately.

**(c) Terms of programmatic access:** Requires a free account and an API
key (Bearer token). **But: see the flag at the top of this document.**
Exact clauses (fetched from `vast.ai/terms`):
> "Systematic retrieval of data or other content from the Website to
> create or compile, directly or indirectly, a collection, compilation,
> database or directory without written permission from Company"

> "Use web scraping, web harvesting, web data extraction or any other
> method to extract data from the Website or Services"

The ToS does not distinguish "Website" from "Services" (which includes the
API) — both are covered by the same prohibited-conduct list. A hedonic
regression dataset is, definitionally, a "collection, compilation,
database" built from systematically retrieved listings. **Recommendation:
email vast.ai (or use whatever contact channel they designate for
data/research requests) for explicit written permission before pulling
anything, even via the official API with a valid key.**

**(d) Topology-distinguishable listings:** Likely the largest of any
source here — vast.ai's marketplace typically lists on the order of
thousands of individual host machines at any given time, each carrying
its own `num_gpus`/`bw_nvlink` values, i.e. every listing is
topology-distinguishable by construction. Exact current count not
retrieved (would require an actual API call, out of scope for this
recon).

---

## 2. Cloud provider official pricing APIs

These four are grouped because they share a pattern: official, free,
publicly documented APIs meant exactly for this kind of programmatic
consumption, with **no discovered ToS friction** — but **none expose
NVLink/interconnect topology as a structured field**. Topology has to be
manually joined from each provider's own instance-family documentation
(the same text-sourcing pattern already used in `07_mine_nvl_configs.py`
for MLPerf `system_name`/`hw_notes` fields), keyed on instance-type name.

### AWS (EC2 Price List API + Spot Price History API)
- **(a) Topology:** No. Price List API attributes include `gpu`,
  `gpuMemory`, `networkPerformance`, `instanceType`, `physicalProcessor`,
  etc., but nothing that structurally identifies an NVLink domain or the
  new P6e-GB200/GB300 "UltraServer" multi-node grouping. That has to be
  read off AWS's own instance-type pages (e.g. the P6e-GB200 announcement
  explicitly describing "GB200 NVL72") and joined by `instanceType` string.
- **(b) Historical:** Two different answers for two different APIs. The
  Price List (Bulk/Query) API gives only current effective prices, no
  history. Separately, `DescribeSpotPriceHistory` gives genuine historical
  spot-price time series, but only a **90-day rolling window**, and only
  for instance types actually offered on the Spot market — capacity-
  constrained rack-scale GPU types (P6e-GB200/GB300) may have thin-to-zero
  Spot availability, which would limit this API's usefulness for exactly
  the hardware we care about most.
- **(c) Terms:** Official, documented AWS APIs under the standard AWS
  Customer Agreement / Service Terms; explicitly built for programmatic
  price consumption. No additional restriction found. `DescribeSpotPriceHistory`
  requires an AWS account with IAM credentials (not a public/anonymous
  endpoint); the Price List API can be queried unauthenticated for the
  bulk JSON files.
- **(d) Count:** Small number of distinct GPU instance *types* (roughly a
  dozen across P3/P4/P5/P6/P6e/G4/G5/G6 families), multiplied by regions —
  hundreds of priced SKU-region rows, but only a handful of genuinely
  distinct *topology classes* (single-node 8-GPU NVLink, GB200/GB300 NVL72
  multi-node, non-NVLink G-series).

### Google Cloud (Cloud Billing Catalog / Pricing API)
- **(a) Topology:** No, same limitation as AWS — SKU/pricing schema
  doesn't carry NVLink domain info. A3/A4 machine-family docs describe
  topology in prose (e.g. "A3 Mega: 8x H100 Mega, NVLink"), joined
  manually by machine-type name.
- **(b) Historical:** Current pricing only from what was found; no
  history endpoint documented.
- **(c) Terms:** Official Google API, explicitly free ("All use of the
  Cloud Billing APIs is free of charge"), no restriction found.
- **(d) Count:** Similarly small — a handful of accelerator-optimized
  machine families (A2, A3, A3 Mega, A4), each a single topology class.

### Microsoft Azure (Retail Prices API)
- **(a) Topology:** No. Schema (`armSkuName`, `meterName`, `productName`,
  `serviceFamily`, etc. — full field list retrieved and reviewed) has
  nothing NVLink/interconnect-specific. Would need to join ND-series /
  NDv5 documentation pages by `armSkuName`.
- **(b) Historical:** Ambiguous. Each price record carries an
  `effectiveStartDate`, which hints at some change-tracking, but no
  explicit "price history" query capability is documented — appears to
  return the currently-effective meter set, not a full historical series.
- **(c) Terms:** Explicitly **unauthenticated** ("This API gives you an
  unauthenticated experience to get retail rates"), no API key needed, no
  restriction found; returns max 1,000 records/page with pagination.
- **(d) Count:** Similar to GCP/AWS — a handful of GPU-carrying VM series
  (NC, ND, NDv5 families), few distinct topology classes.

### Oracle Cloud Infrastructure (undocumented-but-public price API)
- **(a) Topology:** No structural field, same limitation. Oracle's own
  GB300 NVL72 documentation (already used as ground truth in
  `07_mine_nvl_configs.py` — the OCI `hw_notes`/`sw_notes` fields that
  fixed our mining classifier) would again need to be joined manually.
- **(b) Historical:** Current pricing only.
- **(c) Terms:** A public JSON endpoint exists
  (`apexapps.oracle.com/pls/apex/cetools/api/v1/products/`), no
  authentication required, documented by Oracle itself as the intended way
  to access list pricing programmatically (Pay-As-You-Go only — no
  committed-use, government, or negotiated pricing). Lower confidence than
  AWS/GCP/Azure since this was confirmed via a third-party blog and a
  community-built MCP server rather than a prominent first-party API
  reference page, but Oracle's own docs do point at it.
- **(d) Count:** Small — OCI's Blackwell/Hopper GPU shapes (BM.GPU.*
  families), few topology classes, though directly relevant since Oracle
  is already a submitter in our MLPerf dataset.

---

## 3. Third-party GPU price index / aggregator sites

- **GPUPerHour** (28 providers, ~17,888 tracked configurations, 60-second
  refresh, NVLink filter in the UI) — **blocked**: ToS requires a
  commercial license for any programmatic access and explicitly prohibits
  scraping (see flag at top). Otherwise would have been an attractive
  volume source.
- **AI Multiple "Cloud GPU Rental Price Index"** — genuine historical
  depth (24 monthly snapshots, July 2024–June 2026), but its own stated
  methodology **medians PCIe/SXM/NVL interconnect variants together under
  one GPU name** — i.e. it explicitly throws away the exact topology
  distinction this project needs. Static published report, no API. Also
  worth noting: the site discloses a customer/vendor relationship with
  Ionos (an AI-hardware customer), a potential objectivity conflict for a
  price index. Not usable for a topology term regardless of ToS.
- **GetDeploying** — tracks ~155+ H200 listings across 33+ providers, has
  some historical trend claims (e.g. "+8% since June 2025" in prose) and a
  distinct "GPU Prices API" product mentioned in-footer — but that API
  appears to be a separate (likely paid/licensed) commercial product, not
  confirmed free/open, and per-listing topology granularity wasn't
  confirmed either. Needs direct inquiry before use, not usable as-is.

## 4. Direct provider pages with no public pricing API

**CoreWeave** and **Lambda Labs** — both already submitters in our MLPerf
dataset, both describe topology richly in prose per instance family
(CoreWeave: "H100 HGX ... fully connected with NVLink ... 400G NDR
InfiniBand fabric"; Lambda: "HGX B200 SXM6 nodes and Quantum-2 InfiniBand
networking") — exactly the kind of language `07_mine_nvl_configs.py`
already knows how to parse. But neither exposes a public pricing API; the
only source is their marketing pricing page, which would require scraping
(out of scope this phase) and hasn't been ToS-cleared. Flagging as
"interesting, blocked" rather than dropping — these are literally the same
organizations already in our dataset, so a future direct partnership /
manual data-sharing conversation could be worth more than automated
collection here.

---

## Recommendation

**No source is unconditionally ready to collect from today.** The
single best-fit source on data quality — vast.ai, real market-clearing
prices with structured per-listing topology fields — needs written
permission from vast.ai first, given how its ToS is worded. Cloud
provider APIs (AWS/GCP/Azure/OCI) are immediately usable with zero ToS
risk, but give only a handful of distinct topology classes each — enough
for a coarse, provider-list-price version of the regression, not a rich
one.

**Proposed path once cleared:**
1. Contact vast.ai for written permission to compile a research dataset
   via their public API for this analysis. If granted, this is the
   primary source: it has both real transaction-like prices and
   structured topology fields (`num_gpus`, `bw_nvlink`) at real volume.
2. In parallel (no permission needed): pull AWS/GCP/Azure/OCI's official
   pricing APIs as a smaller, authoritative cross-check — manually
   annotate each SKU's topology class from provider documentation, the
   same way `07_mine_nvl_configs.py` annotates MLPerf submissions, so the
   two datasets use a consistent domain-size definition.
3. Do not use GPUPerHour without a commercial license. Do not use AI
   Multiple's index (defeats the purpose by design). Revisit GetDeploying
   only after confirming its API's actual terms and per-listing topology
   granularity directly with them.

**Proposed schema** (shared across whichever sources get cleared, so they
can be pooled or compared like-for-like):

| field | type | notes |
|---|---|---|
| `source` | string | `vast_ai`, `aws`, `gcp`, `azure`, `oci` |
| `listing_id` / `sku_id` | string | source's native identifier |
| `collected_at` | timestamp | when this row was retrieved (since most sources are snapshot-only) |
| `gpu_model` | string | e.g. H100, GB200, GB300 |
| `num_gpus` | int | GPUs in this listing/instance |
| `domain_size` | int | NVLink domain size — direct from `bw_nvlink`/`num_gpus` for vast.ai; manually annotated from provider docs otherwise, same method as `true_domain` in `topology_common.py` |
| `interconnect_type` | categorical | NVLink / InfiniBand / PCIe / none |
| `multi_node` | bool | whether the listing spans multiple physical nodes |
| `price_type` | categorical | on-demand, spot, reserved, market (vast.ai) |
| `price_per_gpu_hour` | float | normalized to per-GPU for cross-source comparability |
| `region` | string | |
| `provider` | string | distinct from `source` where source is an aggregator (e.g. vast.ai host org) |

This mirrors `data/mlperf_topology_dataset.csv`'s existing shape closely
enough that the same `topology_common.py`-style cleaning conventions
(explicit domain-size derivation, org-level clustering for SEs — many
providers, few observations each, so cluster by provider) should transfer
directly to a future `10_hedonic_price_regression.py`.
