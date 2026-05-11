#!/usr/bin/env bash
# One-command full benchmark = stage (a) inference, then stage (b) leaderboard,
# writing the result CSVs into results/. Both stages read benchmark/config.json;
# any run_inference.py flags pass through (e.g. --num-test N, --rose-dir,
# --stead-dir, --eval-dir, --config, --skip-rose, --skip-stead).
#
# Equivalent to:
#     python benchmark/run_inference.py     [flags...]
#     python benchmark/build_leaderboard.py [flags...] --update-results
#
# To test WITHOUT overwriting results/, run the two stages yourself and drop
# --update-results from the second (it then writes only under eval/).
#
# Run with a Python env that has seisbench + torch + tensorflow. The full pools
# take ~hours on CPU; pass --num-test N for a quick subset (N traces per pool).
set -euo pipefail
cd "$(dirname "$0")/.."

python benchmark/run_inference.py "$@"
python benchmark/build_leaderboard.py "$@" --update-results
