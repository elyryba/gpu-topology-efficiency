"""
07_mine_nvl_configs.py

Independent text-mining pass over the raw MLPerf submission repos to look
for per-submission evidence of actual NVLink domain size for GB200/GB300
rows, as a check on the flat true_domain = min(total_gpus, 72) cap applied
by topology_common.py (see 03_corrected_topology_model.py's docstring for
why a naive metadata detector was rejected there: submitter naming is
inconsistent — a GB300 compute tray labeled "NVL4" describes the tray
module, not the rack fabric).

Evidence sources, in preference order:
1. `system_name` field in <org>/systems/<id>.json. This is the only place
   a literal NVL72/36/16/8/4 token was found in practice, usually inside
   an unambiguous rack product name (e.g. "8x NVIDIA GB200 NVL72"). BUT:
   Nebius's GB300 submissions say e.g. "8x NVIDIA GB300 NVL4" — no
   "tray"/"module" qualifier word appears anywhere in the string, yet
   "NVL4" here names the compute tray, not the rack fabric (confirmed:
   the matched number equals that submission's own accelerators_per_node
   in all 3 occurrences found in v6.0). A word-list check for qualifier
   language does NOT catch this — the real tell is that the mention
   collapses to the tray-level GPU count. So: any NVL match whose number
   equals accelerators_per_node is downgraded to `low` confidence
   (tray/module label) regardless of org or wording; only a match that
   does NOT coincide with accelerators_per_node is trusted as `high`
   (genuine rack-fabric-scale evidence, e.g. NVL72 on an 8-GPU-per-node
   system).
2. Benchmark config filenames under <org>/benchmarks/*/implementations/*/.
   Node-shape tokens like "config_GB300_2x4x1xtp1pp1cp1_fp4.sh" encode
   accelerators-per-node (a tray-level count), not domain size. Recorded
   at `low` confidence with `inferred_domain` left blank so this signal
   can never silently masquerade as fabric evidence.
No match from either source -> `none` (cap remains the only assumption).

This script does not modify topology_common.py or any model script — it
only produces an independent evidence trail for review.

Input:  $MLPERF_ROOT/training_results_{v5.0,v5.1,v6.0}
Output: data/nvl_config_mined.csv, results/nvl_mining_reconciliation.txt
"""
import glob
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from topology_common import gen_bucket, load_clean

ROOT = os.environ.get("MLPERF_ROOT", os.path.expanduser("~"))
REPOS = ["training_results_v5.0", "training_results_v5.1", "training_results_v6.0"]
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                      "nvl_config_mined.csv")
OUT_TXT = os.path.join(os.path.dirname(__file__), "..", "results",
                      "nvl_mining_reconciliation.txt")
DATASET_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                          "mlperf_topology_dataset.csv")

NVL_RE = re.compile(r"NVL[\s\-]?(72|36|16|8|4)\b", re.IGNORECASE)
NODE_SHAPE_RE = re.compile(r"(GB200|GB300).{0,40}?x(\d+)x", re.IGNORECASE)


def int_field(v):
    try:
        digits = re.sub(r"\D", "", str(v))
        return int(digits) if digits else None
    except Exception:
        return None


def mine_system_name(system_name, accel_pn):
    """Return (confidence, evidence_string, inferred_domain) or None.

    An NVL token whose number equals accelerators_per_node is a tray/module
    label, not rack-fabric evidence (the Nebius GB300 "NVL4" trap) — it
    gets `medium` confidence with no inferred_domain. A token that doesn't
    coincide with the per-node count is trusted as `high`.
    """
    m = NVL_RE.search(system_name or "")
    if not m:
        return None
    domain = int(m.group(1))
    if accel_pn is not None and domain == accel_pn:
        note = (f"{system_name} (NVL{domain} == accelerators_per_node "
                f"{accel_pn} -> tray/module label, not rack fabric)")
        return ("medium", note, None)
    return ("high", system_name, domain)


def mine_config_filenames(base, org, system_id):
    """Fallback low-confidence signal: node-shape tokens in benchmark config
    filenames. Prefers files whose name contains system_id; else any
    GB200/GB300 node-shape match under the org's benchmarks tree."""
    pattern = os.path.join(base, org, "benchmarks", "*", "implementations", "*", "*")
    candidates = glob.glob(pattern)
    scoped = [c for c in candidates if system_id.lower() in os.path.basename(c).lower()]
    pool = scoped or candidates
    for path in pool:
        name = os.path.basename(path)
        m = NODE_SHAPE_RE.search(name)
        if m:
            tray_n = m.group(2)
            note = (f"{name} (node-shape token 'x{tray_n}x' = "
                    f"accelerators-per-node, NOT rack-fabric domain size)")
            return ("low", note, None)
    return None


def mine_all():
    rows = []
    for repo in REPOS:
        base = os.path.join(ROOT, repo)
        if not os.path.isdir(base):
            print(f"skip (not found): {base}")
            continue
        for sf in glob.glob(os.path.join(base, "*", "systems", "*.json")):
            rel = os.path.relpath(sf, base)
            org = rel.split(os.sep)[0]
            system_id = os.path.splitext(os.path.basename(sf))[0]
            try:
                with open(sf) as f:
                    sysinfo = json.load(f)
            except Exception:
                continue

            gen = gen_bucket(sysinfo.get("accelerator_model_name", ""))
            if gen not in ("GB200", "GB300"):
                continue

            nodes_n = int_field(sysinfo.get("number_of_nodes", ""))
            accel_pn = int_field(sysinfo.get("accelerators_per_node", ""))
            total_gpus = nodes_n * accel_pn if (nodes_n and accel_pn) else None

            evidence = mine_system_name(sysinfo.get("system_name", ""), accel_pn)
            if evidence is None:
                evidence = mine_config_filenames(base, org, system_id)
            if evidence is None:
                confidence, evidence_string, inferred_domain = "none", "", None
            else:
                confidence, evidence_string, inferred_domain = evidence

            rows.append({
                "repo": repo, "org": org, "system_id": system_id, "gen": gen,
                "total_gpus": total_gpus, "evidence_string": evidence_string,
                "inferred_domain": inferred_domain, "confidence": confidence,
            })

    return pd.DataFrame(rows, columns=["repo", "org", "system_id", "gen",
                                       "total_gpus", "evidence_string",
                                       "inferred_domain", "confidence"])


def reconcile(mined):
    """Join mined evidence against the GB200/GB300 sub-rack rows
    (total_gpus < 72) in the cleaned analysis sample and report coverage.

    Reported at two grains: dataset ROWS (one per system x model benchmark
    -- this is the "92" figure quoted in CLAUDE.md/03_corrected_topology_
    model.py) and unique SUBMISSIONS (one per system, since evidence is
    mined at the system level and applies identically to every model a
    given system ran).
    """
    clean = load_clean(DATASET_CSV)
    subrack = clean[clean["gen"].isin(["GB200", "GB300"]) & (clean["total_gpus"] < 72)]
    subrack_keys = subrack[["repo", "org", "system_id"]].drop_duplicates()

    joined_rows = subrack.merge(mined, on=["repo", "org", "system_id"],
                               how="left", suffixes=("", "_mined"))
    joined_rows["confidence"] = joined_rows["confidence"].fillna("none")
    joined = subrack_keys.merge(mined, on=["repo", "org", "system_id"],
                               how="left", suffixes=("", "_mined"))
    joined["confidence"] = joined["confidence"].fillna("none")

    def split(df):
        is_high = df["confidence"] == "high"
        is_conflicting = is_high & df["inferred_domain"].notna() & (df["inferred_domain"] != 72)
        return df[is_high & ~is_conflicting], df[is_conflicting], df[~is_high]

    high, conflicting, cap_only = split(joined)
    high_r, conflicting_r, cap_only_r = split(joined_rows)

    lines = [
        "NVL CONFIG MINING — RECONCILIATION vs. GB200/GB300 sub-rack rows",
        "(total_gpus < 72, from the load_clean() analysis sample)",
        "",
        f"Dataset rows (system x model benchmark, matches CLAUDE.md's '92' figure): {len(subrack)}",
        f"  direct high-confidence evidence (confirms cap):   {len(high_r)}",
        f"  conflicting evidence (mined domain != 72):        {len(conflicting_r)}",
        f"  cap-only (no resolving high-confidence evidence): {len(cap_only_r)}",
        "",
        f"Unique sub-rack submissions (system-level, evidence's real grain): {len(subrack_keys)}",
        f"  direct high-confidence evidence (confirms cap):   {len(high)}",
        f"  conflicting evidence (mined domain != 72):        {len(conflicting)}",
        f"  cap-only (no resolving high-confidence evidence): {len(cap_only)}",
        "",
    ]
    if len(conflicting):
        lines.append("Conflicting rows:")
        for _, r in conflicting.iterrows():
            lines.append(f"  {r['repo']}/{r['org']}/{r['system_id']} -> "
                         f"inferred_domain={r['inferred_domain']} "
                         f"evidence={r['evidence_string']!r}")
        lines.append("")

    for conf in ("high", "medium", "low", "none"):
        subset = mined[mined["confidence"] == conf]
        lines.append(f"Examples ({conf}, n={len(subset)}):")
        examples = subset["evidence_string"].head(5).tolist()
        if examples:
            for e in examples:
                lines.append(f"  - {e!r}")
        else:
            lines.append("  (none found)")
        lines.append("")

    return "\n".join(lines)


def main():
    mined = mine_all()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    mined.to_csv(OUT_CSV, index=False)
    print(f"Mined {len(mined)} GB200/GB300 system rows -> {OUT_CSV}")
    for c in ("high", "medium", "low", "none"):
        print(f"  confidence={c}: {(mined['confidence'] == c).sum()}")

    report = reconcile(mined)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
