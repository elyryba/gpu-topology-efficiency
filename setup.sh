#!/usr/bin/env bash
# One-shot local setup for the topology-rebuild project (macOS/Linux).
set -euo pipefail
cd "$(dirname "$0")"

# 1. Python venv (uses system python3; macOS 13+ ships 3.9+, any 3.9+ works)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✓ environment ready (python: $(python3 --version))"

# 2. Run the full analysis pipeline (skips 01, which needs MLPerf repos cloned)
cd scripts
for s in 02_topology_efficiency_model 03_corrected_topology_model \
         04_rigor_checks 05_deep_checks 06_continuous_bandwidth_model; do
  echo ""
  echo "===== running ${s}.py ====="
  python3 "${s}.py"
done

echo ""
echo "✓ all done — outputs in results/"
