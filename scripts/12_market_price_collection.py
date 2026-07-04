"""
12_market_price_collection.py

Collects current on-demand GPU rental pricing from the cloud-provider
official pricing APIs docs/market_data_recon.md cleared as unrestricted
(no discovered ToS friction, publicly documented for programmatic use):
AWS, Azure, and OCI. Does NOT touch vast.ai or GPUPerHour -- the recon
flagged both (vast.ai's ToS needs written permission before any
compiled-dataset use; GPUPerHour requires a paid commercial license).
GCP is skipped: its Cloud Billing Catalog API needs an API key tied to a
GCP project, which isn't configured in this environment -- that's a
documented gap, not a silent workaround.

None of these three APIs expose NVLink/interconnect topology as a
structured field (confirmed in the recon). Topology (num_gpus_in_sku,
domain_size, interconnect_type, multi_node) is therefore a manually
curated lookup table sourced from each provider's own instance-family
documentation -- the same annotation methodology
07_mine_nvl_configs.py uses for MLPerf system_name/hw_notes fields,
applied here to price-sheet SKUs instead of benchmark submissions. Every
lookup entry cites its documentation source inline.

Three topology classes, matching the user's framing:
  - low-interconnect: single GPU, no NVLink (T4/A10/A100-single/L4 VMs)
  - 8-GPU NVLink node: single-node HGX-class, domain=8
  - rack-scale: GB200/GB300 NVL72-class, domain=72 (this repo's
    established true_domain convention, topology_common.py)

How each source is fetched:
  - AWS: EC2 Price List Bulk API (unauthenticated, public JSON), the
    us-east-1 region file. That file is ~480MB -- a real, one-time
    bandwidth/parse cost, which is exactly why this script is excluded
    from CI's run set (see .github/workflows/reproduce.yml).
  - Azure: Retail Prices API (unauthenticated REST), filtered per SKU,
    eastus region, most recent effectiveStartDate, non-Windows product
    line (Consumption price type).
  - OCI: the public (unauthenticated, documented by Oracle as the
    intended access path per the recon) apex JSON endpoint, full product
    catalog fetched once and filtered client-side by displayName. OCI
    prices per-GPU directly ("GPU Per Hour" metric) rather than per-VM --
    unlike AWS/Azure, no per-VM-to-per-GPU normalization is needed.

This is a point-in-time snapshot, not a historical series -- none of the
three APIs expose historical on-demand pricing (confirmed in recon); a
`collected_at` column records exactly when.

Output: data/market_prices_snapshot.csv
"""
import csv
import datetime
import json
import os
import re
import time
import urllib.error
import urllib.request

OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                      "market_prices_snapshot.csv")

COLLECTED_AT = datetime.datetime.now(datetime.timezone.utc).strftime(
    "%Y-%m-%dT%H:%M:%SZ")

AWS_BULK_URL = ("https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/"
               "AmazonEC2/current/us-east-1/index.json")
AZURE_API = "https://prices.azure.com/api/retail/prices"
OCI_API = "https://apexapps.oracle.com/pls/apex/cetools/api/v1/products/"

# ---------------------------------------------------------------------
# Manually-curated topology annotations. None of the three APIs expose
# these fields programmatically -- every entry cites its documentation
# source, same discipline as 07_mine_nvl_configs.py's evidence_string.
# ---------------------------------------------------------------------

# instance_type, gpu_model, num_gpus_in_sku, domain_size, interconnect, multi_node, source
AWS_CATALOG = [
    ("g4dn.xlarge", "T4", 1, 1, "none", False,
     "aws.amazon.com/ec2/instance-types/g4 -- single GPU, no NVLink"),
    ("g5.xlarge", "A10G", 1, 1, "none", False,
     "aws.amazon.com/ec2/instance-types/g5 -- single GPU, no NVLink"),
    ("g6.xlarge", "L4", 1, 1, "none", False,
     "aws.amazon.com/ec2/instance-types/g6 -- single GPU, no NVLink"),
    ("p5.48xlarge", "H100", 8, 8, "NVLink", False,
     "aws.amazon.com/ec2/instance-types/p5 -- 8x H100 NVLink/NVSwitch, single node"),
    ("p6-b200.48xlarge", "B200", 8, 8, "NVLink", False,
     "aws.amazon.com/ec2/instance-types/p6 -- 8x B200 NVLink/NVSwitch, single node"),
]

# armSkuName, gpu_model, num_gpus_in_sku, domain_size, interconnect, multi_node, source
AZURE_CATALOG = [
    ("Standard_NC4as_T4_v3", "T4", 1, 1, "none", False,
     "learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/"
     "ncast4v3-series -- single GPU, no NVLink"),
    ("Standard_NC24ads_A100_v4", "A100", 1, 1, "none", False,
     "learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/"
     "ncads-a100-v4-series -- single GPU tier of this family, PCIe, no NVLink"),
    ("Standard_ND96isr_H100_v5", "H100", 8, 8, "NVLink", False,
     "learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/"
     "nd-h100-v5-series -- 8x H100 NVLink/NVSwitch, single node"),
    ("Standard_ND128isr_NDR_GB200_v6", "GB200", 4, 72, "NVLink+InfiniBand", True,
     "learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/"
     "nd-gb200-v6-series -- 4 GPUs/VM; rack-scale domain is 72 GPUs "
     "across 18 VMs (confirmed via Microsoft Learn docs)"),
]

# gpu_token (matched as a whole word in displayName -- NOT substring, see
# below), gpu_model, num_gpus_in_sku, domain_size, interconnect, multi_node,
# source. OCI's "GPU Per Hour" metric already prices per single GPU
# regardless of instance shape, so num_gpus_in_sku=1 throughout (see 13's
# discussion of what this itself implies about OCI's list pricing).
OCI_CATALOG = [
    ("A10", "A10", 1, 1, "none", False,
     "docs.oracle.com/iaas/Content/Compute/References/computeshapes.htm -- "
     "VM.GPU.A10.x shapes, PCIe, no NVLink"),
    ("H100", "H100", 1, 8, "NVLink", False,
     "docs.oracle.com/iaas/Content/Compute/References/computeshapes.htm -- "
     "BM.GPU.H100.8, 8x NVLink/NVSwitch single node"),
    ("GB200", "GB200", 1, 72, "NVLink+InfiniBand", True,
     "docs.oracle.com/iaas/Content/Compute/References/computeshapes.htm -- "
     "GB200 NVL72 rack-scale domain (same source used to fix "
     "07_mine_nvl_configs.py's classifier)"),
    ("GB300", "GB300", 1, 72, "NVLink+InfiniBand", True,
     "docs.oracle.com/iaas/Content/Compute/References/computeshapes.htm -- "
     "GB300 NVL72 rack-scale domain"),
]


def fetch_json(url, retries=4, backoff_s=3):
    """GET + parse JSON, with retry-with-backoff on HTTP 429 (Azure's
    Retail Prices API rate-limits back-to-back requests; four sequential
    SKU queries can trip it)."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(backoff_s * (attempt + 1))
                continue
            raise


def collect_aws(rows, log):
    log.append("AWS: downloading EC2 Price List Bulk API (us-east-1, ~480MB) ...")
    try:
        data = fetch_json(AWS_BULK_URL)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.append(f"AWS: FAILED to download bulk price list ({e}) -- skipping AWS entirely")
        return
    products = data["products"]
    terms = data["terms"]["OnDemand"]

    for instance_type, gpu_model, n_gpus, domain, interconnect, multi_node, source in AWS_CATALOG:
        found = False
        for sku, p in products.items():
            attrs = p.get("attributes", {})
            if (attrs.get("instanceType") == instance_type
                    and attrs.get("operatingSystem") == "Linux"
                    and attrs.get("tenancy") == "Shared"
                    and attrs.get("preInstalledSw", "NA") == "NA"
                    and attrs.get("capacitystatus") == "Used"):
                od = terms.get(sku, {})
                for term_val in od.values():
                    for dim in term_val["priceDimensions"].values():
                        price = dim["pricePerUnit"].get("USD")
                        if price and float(price) > 0:
                            price = float(price)
                            rows.append({
                                "provider": "aws", "sku": instance_type,
                                "region": "us-east-1", "gpu_model": gpu_model,
                                "num_gpus_in_sku": n_gpus, "domain_size": domain,
                                "interconnect_type": interconnect,
                                "multi_node": multi_node,
                                "price_type": "on-demand",
                                "price_per_hour_usd": round(price, 6),
                                "price_per_gpu_hour_usd": round(price / n_gpus, 6),
                                "collected_at": COLLECTED_AT,
                                "topology_source": source,
                            })
                            log.append(f"AWS: {instance_type} -> ${price:.4f}/hr "
                                     f"(${price/n_gpus:.4f}/GPU-hr)")
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        if not found:
            log.append(f"AWS: FAILED to find on-demand price for {instance_type}")


def collect_azure(rows, log):
    log.append("Azure: querying Retail Prices API ...")
    for i, (sku, gpu_model, n_gpus, domain, interconnect, multi_node, source) in enumerate(AZURE_CATALOG):
        if i > 0:
            time.sleep(2)  # space out requests -- back-to-back queries trip 429s
        url = (f"{AZURE_API}?api-version=2023-01-01-preview&$filter="
              f"armSkuName eq '{sku}' and priceType eq 'Consumption'"
              ).replace(" ", "%20").replace("'", "%27")
        try:
            data = fetch_json(url)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log.append(f"Azure: FAILED to query {sku} ({e})")
            continue
        candidates = [i for i in data.get("Items", [])
                     if i.get("isPrimaryMeterRegion")
                     and i.get("armRegionName") == "eastus"
                     and "Windows" not in i.get("productName", "")]
        if not candidates:
            log.append(f"Azure: no matching eastus/non-Windows/primary-meter "
                     f"record found for {sku}")
            continue
        best = max(candidates, key=lambda i: i["effectiveStartDate"])
        price = float(best["retailPrice"])
        rows.append({
            "provider": "azure", "sku": sku, "region": "eastus",
            "gpu_model": gpu_model, "num_gpus_in_sku": n_gpus,
            "domain_size": domain, "interconnect_type": interconnect,
            "multi_node": multi_node, "price_type": "on-demand",
            "price_per_hour_usd": round(price, 6),
            "price_per_gpu_hour_usd": round(price / n_gpus, 6),
            "collected_at": COLLECTED_AT, "topology_source": source,
        })
        log.append(f"Azure: {sku} -> ${price:.4f}/hr (${price/n_gpus:.4f}/GPU-hr, "
                 f"effective {best['effectiveStartDate']})")


OCI_EXCLUDE = ("ai enterprise", "vmware", "roving edge", "cloud@customer")


def collect_oci(rows, log):
    log.append("OCI: fetching public product catalog ...")
    try:
        data = fetch_json(OCI_API)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.append(f"OCI: FAILED to fetch product catalog ({e}) -- skipping OCI entirely")
        return
    items = data["items"]

    for gpu_token, gpu_model, n_gpus, domain, interconnect, multi_node, source in OCI_CATALOG:
        # Word-boundary match, not substring -- "A10" must not match "A100",
        # and excludes AI-Enterprise/VMware/Cloud@Customer bundled SKUs so
        # we get the base compute price, not a software-bundled variant.
        pattern = re.compile(r"\b" + re.escape(gpu_token) + r"\b", re.IGNORECASE)
        candidates = [i for i in items
                     if pattern.search(i["displayName"])
                     and i.get("metricName", "").lower().startswith("gpu")
                     and not any(x in i["displayName"].lower() for x in OCI_EXCLUDE)]
        if len(candidates) > 1:
            log.append(f"OCI: {len(candidates)} candidates matched '{gpu_token}' "
                     f"({[c['partNumber'] for c in candidates]}), using the first")
        match = candidates[0] if candidates else None
        if match is None:
            log.append(f"OCI: FAILED to find product matching '{gpu_token}'")
            continue
        usd = next((p["prices"][0]["value"]
                   for p in match["currencyCodeLocalizations"]
                   if p["currencyCode"] == "USD"
                   for pr in [p["prices"]] if pr and pr[0]["model"] == "PAY_AS_YOU_GO"),
                  None)
        if usd is None:
            log.append(f"OCI: no USD PAY_AS_YOU_GO price found for {match['displayName']}")
            continue
        price = float(usd)
        rows.append({
            "provider": "oci", "sku": match["partNumber"], "region": "global-list-price",
            "gpu_model": gpu_model, "num_gpus_in_sku": n_gpus,
            "domain_size": domain, "interconnect_type": interconnect,
            "multi_node": multi_node, "price_type": "on-demand",
            "price_per_hour_usd": round(price * n_gpus, 6),
            "price_per_gpu_hour_usd": round(price, 6),
            "collected_at": COLLECTED_AT, "topology_source": source,
        })
        log.append(f"OCI: {match['displayName']} ({match['partNumber']}) -> "
                 f"${price:.4f}/GPU-hr")


def main():
    rows = []
    log = []
    collect_aws(rows, log)
    collect_azure(rows, log)
    collect_oci(rows, log)

    print("\n".join(log))
    print(f"\nCollected {len(rows)} priced SKUs across "
         f"{len(set(r['provider'] for r in rows))} providers")

    if not rows:
        raise SystemExit("No rows collected from any source -- check network "
                        "access and API availability before trusting an "
                        "empty snapshot.")

    fieldnames = ["provider", "sku", "region", "gpu_model", "num_gpus_in_sku",
                 "domain_size", "interconnect_type", "multi_node",
                 "price_type", "price_per_hour_usd", "price_per_gpu_hour_usd",
                 "collected_at", "topology_source"]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
