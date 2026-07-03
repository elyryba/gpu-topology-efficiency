"""
01_extract_mlperf_data.py

Extract structured (topology, scale, model, time-to-train) data from
MLCommons MLPerf Training result repos.

Requires the repos cloned locally, e.g.:
    git clone --depth 1 https://github.com/mlcommons/training_results_v5.1

Output: data/mlperf_topology_dataset.csv

Notes on two bugs this version guards against:
1. Bandwidth parsing: strings like "4x ConnectX-7 IB NDR 400Gb/s" have a
   leading link-count multiplier NOT adjacent to the bandwidth number.
   An earlier parser missed the x4 and produced a spurious, backwards-
   signed regression result. parse_bw() handles per-clause multipliers.
2. Org extraction: the previous version used sf.split(os.sep)[4], which
   silently returns the wrong path component if ROOT's depth changes.
   This version derives org relative to the repo base.
"""
import csv
import glob
import json
import os
import re

ROOT = os.environ.get("MLPERF_ROOT", os.path.expanduser("~"))
REPOS = ["training_results_v3.1", "training_results_v4.0",
         "training_results_v4.1", "training_results_v5.0",
         "training_results_v5.1", "training_results_v6.0"]
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                        "mlperf_topology_dataset.csv")


def parse_bw(s):
    """Aggregate bandwidth (GB/s) from free-text interconnect strings.
    Handles leading 'Nx' link multipliers per comma/semicolon clause:
    '4x ConnectX-7 IB NDR 400Gb/s' -> 4 * 400Gb/s = 1600Gb/s = 200GB/s."""
    if not s:
        return None
    best = None
    for clause in re.split(r"[;,]", str(s)):
        mult_match = re.match(r"\s*(\d+)\s*x\b", clause)
        mult = int(mult_match.group(1)) if mult_match else 1
        for val_str, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(GB/s|Gb/s)", clause):
            val = float(val_str) * mult
            gbs = val if unit == "GB/s" else val / 8.0
            if best is None or gbs > best:
                best = gbs
    return best


def int_field(v):
    try:
        digits = re.sub(r"\D", "", str(v))
        return int(digits) if digits else None
    except Exception:
        return None


def extract_times(result_files):
    """Realized time-to-train from raw run_start/run_stop log timestamps,
    keeping only runs whose log reports success."""
    times = []
    for rf in result_files:
        try:
            with open(rf, errors="ignore") as f:
                content = f.read()
        except Exception:
            continue
        starts = re.findall(r'"time_ms":\s*(\d+).*?"key":\s*"run_start"', content)
        stops = re.findall(r'"time_ms":\s*(\d+).*?"key":\s*"run_stop"', content)
        if not (starts and stops and 'status": "success"' in content):
            continue
        dur_s = (int(stops[-1]) - int(starts[0])) / 1000.0
        if dur_s > 0:
            times.append(dur_s)
    return times


def main():
    rows = []
    for repo in REPOS:
        base = os.path.join(ROOT, repo)
        if not os.path.isdir(base):
            print(f"skip (not found): {base}")
            continue
        for sf in glob.glob(os.path.join(base, "*", "systems", "*.json")):
            rel = os.path.relpath(sf, base)          # org/systems/<sys>.json
            org = rel.split(os.sep)[0]
            sys_id = os.path.splitext(os.path.basename(sf))[0]
            try:
                with open(sf) as f:
                    sysinfo = json.load(f)
            except Exception:
                continue

            results_dir = os.path.join(base, org, "results", sys_id)
            if not os.path.isdir(results_dir):
                continue

            nodes_n = int_field(sysinfo.get("number_of_nodes", ""))
            accel_pn = int_field(sysinfo.get("accelerators_per_node", ""))
            total = nodes_n * accel_pn if (nodes_n and accel_pn) else None
            accel_ic = sysinfo.get("accelerator_interconnect", "")
            host_net = sysinfo.get("host_networking", "")

            for model_dir in glob.glob(os.path.join(results_dir, "*")):
                if not os.path.isdir(model_dir):
                    continue
                times = extract_times(
                    glob.glob(os.path.join(model_dir, "result_*.txt")))
                if not times:
                    continue
                best = min(times)
                median = sorted(times)[len(times) // 2]
                cv = (max(times) - min(times)) / median if median else None
                rows.append({
                    "repo": repo, "org": org, "system_id": sys_id,
                    "model": os.path.basename(model_dir),
                    "nodes": nodes_n, "gpus_per_node": accel_pn,
                    "total_gpus": total,
                    "accelerator": sysinfo.get("accelerator_model_name", ""),
                    "accel_interconnect_raw": accel_ic,
                    "accel_interconnect_gbps": parse_bw(accel_ic),
                    "host_network_raw": host_net,
                    "host_network_gbps": parse_bw(host_net),
                    "time_to_train_s": round(best, 2),
                    "median_time_to_train_s": round(median, 2),
                    "run_to_run_cv": round(cv, 3) if cv is not None else None,
                    "n_runs_found": len(times),
                })

    if not rows:
        raise SystemExit("No rows extracted — are the MLPerf repos cloned "
                         "under $MLPERF_ROOT?")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Extracted {len(rows)} rows -> {OUT_PATH}")
    print(f"Models: {sorted(set(r['model'] for r in rows))}")
    print(f"Orgs:   {len(set(r['org'] for r in rows))}")


if __name__ == "__main__":
    main()
