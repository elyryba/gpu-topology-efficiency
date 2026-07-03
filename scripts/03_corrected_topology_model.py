"""
03_corrected_topology_model.py  (v2 — documented NVLink-domain correction)

Identical sample and specification to 02, with ONE change: the
domain-size feature. MLPerf's schema reports physical-server GPU counts,
which is correct for every generation where NVLink stops at the server
boundary — but wrong for rack-scale GB200/GB300 NVL72 systems, where the
NVLink fabric spans the whole rack (72-GPU domain across 18 servers).

Correction (documented-architecture, not a guess):
    true_domain = min(total_gpus, 72)   for GB200 / GB300
    true_domain = gpus_per_node         for everything else
    n_domains   = total_gpus / true_domain

Source: NVIDIA GB200/GB300 NVL72 specs + DGX GB Rack Scale Systems User
Guide; corroborated by HPE/Lenovo/Supermicro OEM datasheets. A
metadata-based detector (grepping "NVL72" in system names) was tested
and rejected — submitter naming is inconsistent (e.g. a GB300 compute
tray labeled "NVL4" describes the tray module, not the rack fabric).

Known remaining gap: GH200-family (NVL32-capable) submissions are left
on the uncorrected proxy pending clearer sourcing on typical deployment
domain size.

Input:  data/mlperf_topology_dataset.csv
Output: results/regression_summary_corrected.txt,
        results/robustness_checks_corrected.txt,
        results/v1_v2_comparison.txt
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

    n_corrected = int(df["gen"].isin(["GB200", "GB300"]).sum())
    changed = df[df["true_domain"] != df["gpus_per_node"]]
    print(f"Rack-scale rows (GB200/GB300): {n_corrected} | "
          f"rows whose domain feature actually changed: {len(changed)}")

    vifs = vif_report(df, ["log_domain", "log_n_domains"])
    print(f"VIF check: {vifs}")

    v1 = fit_model(df, "log_gpn", "log_nodes")
    v2 = fit_model(df, "log_domain", "log_n_domains")

    print("\n=== CORRECTED MODEL (v2) ===")
    print(f"log_domain (true NVLink domain)   coef={v2.params['log_domain']:.4f}  "
          f"p={v2.pvalues['log_domain']:.2e}")
    print(f"log_n_domains (# domains)         coef={v2.params['log_n_domains']:.4f}  "
          f"p={v2.pvalues['log_n_domains']:.2e}")
    print(f"R^2 = {v2.rsquared:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "regression_summary_corrected.txt"), "w") as f:
        f.write(str(v2.summary()))

    lines = robustness(df, "log_domain", "log_n_domains")
    with open(os.path.join(OUT_DIR, "robustness_checks_corrected.txt"), "w") as f:
        f.write("\n".join(lines))

    comp = [
        "V1 (gpus_per_node proxy)  vs  V2 (documented NVLink domain)",
        "-" * 62,
        f"{'':32s}{'v1':>12s}{'v2':>12s}",
        f"{'domain-size coefficient':32s}{v1.params['log_gpn']:>12.4f}{v2.params['log_domain']:>12.4f}",
        f"{'domain-size p-value':32s}{v1.pvalues['log_gpn']:>12.2e}{v2.pvalues['log_domain']:>12.2e}",
        f"{'domain-count coefficient':32s}{v1.params['log_nodes']:>12.4f}{v2.params['log_n_domains']:>12.4f}",
        f"{'R^2':32s}{v1.rsquared:>12.4f}{v2.rsquared:>12.4f}",
        f"{'AIC':32s}{v1.aic:>12.1f}{v2.aic:>12.1f}",
        f"{'BIC':32s}{v1.bic:>12.1f}{v2.bic:>12.1f}",
    ]
    with open(os.path.join(OUT_DIR, "v1_v2_comparison.txt"), "w") as f:
        f.write("\n".join(comp))
    print("\n" + "\n".join(comp))

    print("\nWrote results/regression_summary_corrected.txt, "
          "results/robustness_checks_corrected.txt, results/v1_v2_comparison.txt")


if __name__ == "__main__":
    main()
