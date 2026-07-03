"""
02_topology_efficiency_model.py  (v1 — baseline "gpus_per_node" proxy)

Tests whether GPU cluster topology (NVLink domain size vs. number of
networked domains) has a statistically detectable effect on realized
training throughput, using public MLPerf Training benchmark results,
after controlling for total scale, model, and hardware generation.

Input:  data/mlperf_topology_dataset.csv
Output: results/regression_summary.txt, results/robustness_checks.txt
"""
import os
from topology_common import load_clean, fit_model, vif_report, robustness

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                       "mlperf_topology_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    df = load_clean(IN_PATH)
    print(f"Clean analysis sample: {len(df)} rows | "
          f"{df['model'].nunique()} models | {df['gen'].nunique()} hardware generations")

    vifs = vif_report(df, ["log_gpn", "log_nodes"])
    print(f"VIF check (should be well under 5): {vifs}")

    base = fit_model(df, "log_gpn", "log_nodes")
    print("\n=== PRIMARY MODEL (v1) ===")
    print(f"log_gpn (NVLink domain size)    coef={base.params['log_gpn']:.4f}  "
          f"p={base.pvalues['log_gpn']:.2e}")
    print(f"log_nodes (# networked domains) coef={base.params['log_nodes']:.4f}  "
          f"p={base.pvalues['log_nodes']:.2e}")
    print(f"R^2 = {base.rsquared:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "regression_summary.txt"), "w") as f:
        f.write(str(base.summary()))

    lines = robustness(df, "log_gpn", "log_nodes")
    with open(os.path.join(OUT_DIR, "robustness_checks.txt"), "w") as f:
        f.write("\n".join(lines))

    print("\nWrote results/regression_summary.txt and results/robustness_checks.txt")


if __name__ == "__main__":
    main()
