#!/usr/bin/env bash
#
# Rebuild the magnitude calibration tables from the working tree.
#
# Nothing here is needed to use the released magnitudes. They are in the
# catalog, and `rose.magnitudes` reads the published calibration tables from
# this checkout. This script is for checking that the tables follow from their
# inputs, or for recomputing them after a change.
#
# Usage:
#   ROMANIA_ROOT=/path/to/romania magnitudes/regenerate_magnitudes.sh [stage ...]
#
# Stages, in order. With no argument the two cheap ones run, because the
# others need the waveform archive and hours to days of compute:
#
#   amplitudes   Wood-Anderson amplitudes from the waveforms       (hours, needs SAC + StationXML)
#   observations amplitude table with path geometry and QC         (minutes)
#   bundles      origin and picks per event, for SourceSpec        (minutes)
#   sourcespec   S-wave spectral inversion, one run per event      (days, needs SourceSpec 1.8)
#   station_fits collect the per station spectral fits             (minutes)
#   mw           Mw catalog from the fits                          (seconds)
#   fit          -log A0 shapes and station terms, unanchored      (~1 h, sparse inversion)
#   anchor       baseline shift onto Mw, writes the ML catalog     (minutes)   [default]
#   conversion   the ML to Mw relation and its validity range      (minutes)   [default]
#   plots        the ML diagnostic figures                         (minutes)
#   release      merge both magnitudes into the release catalog    (seconds)
#   tables       copy the result into magnitudes/calibration/      (seconds)
#
# `sourcespec` reads `bundles`' output, `anchor` reads `fit`'s and `conversion`
# reads `anchor`'s. Running a later stage without its input stops with the
# missing file named.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

usage() {
    # the header comment of this file, which is the documentation
    sed -n '3,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
    case "$arg" in
    -h | --help) usage; exit 0 ;;
    esac
done

if [[ -z "${ROMANIA_ROOT:-}" ]]; then
    echo "ROMANIA_ROOT is not set. It must point at the working tree that holds" >&2
    echo "romania_ml/ and romania_mw/. See magnitudes/README.md." >&2
    exit 2
fi
if [[ ! -d "$ROMANIA_ROOT" ]]; then
    echo "ROMANIA_ROOT=$ROMANIA_ROOT is not a directory." >&2
    exit 2
fi
export ROMANIA_ROOT

run() {
    echo
    echo "=== $1"
    shift
    ( cd "$REPO" && python "$@" )
}

stage() {
    case "$1" in
    amplitudes)   run "Wood-Anderson amplitudes"      magnitudes/ml/extract_wa_amplitudes.py ;;
    observations) run "amplitude table with QC"       magnitudes/ml/build_observations.py ;;
    bundles)      run "origin and picks per event"    magnitudes/mw/sourcespec/extract_all_events.py ;;
    sourcespec)   run "spectral inversion"            magnitudes/mw/sourcespec/run_sourcespec.py ;;
    station_fits) run "per station spectral fits"     magnitudes/mw/build_station_table.py ;;
    mw)           run "Mw catalog"                    magnitudes/mw/build_mw_catalog.py ;;
    fit)          run "-log A0 and station terms"     magnitudes/ml/fit_ml.py ;;
    anchor)       run "baseline shift onto Mw"        magnitudes/ml/anchor_ml_to_mw.py ;;
    conversion)   run "ML to Mw relation"             magnitudes/ml/fit_ml_to_mw_conversion.py ;;
    plots)        run "ML diagnostic figures"         magnitudes/ml/plot_ml.py ;;
    release)      run "release catalog"               magnitudes/mw/build_release_catalog.py ;;
    tables)       copy_tables ;;
    *)            echo "unknown stage: $1" >&2; exit 2 ;;
    esac
}

# The published tables are copies of driver outputs. Keeping the copy in one
# place is what stops the repository and the working tree drifting apart.
copy_tables() {
    echo
    echo "=== calibration tables -> magnitudes/calibration/"
    local ml="$ROMANIA_ROOT/romania_ml/outputs"
    local mw="$ROMANIA_ROOT/romania_mw/outputs"
    local dst="$HERE/calibration"
    local pair
    for pair in \
        "$ml/fit/parameter_table.csv:parameter_table.csv" \
        "$ml/fit/station_terms.csv:station_terms.csv" \
        "$ml/fit/anchor_events.csv:anchor_events.csv" \
        "$ml/fit/ml_minus_mw_by_bin.csv:ml_minus_mw_by_bin.csv" \
        "$ml/fit/clipped_fraction_by_magnitude.csv:clipped_fraction_by_magnitude.csv" \
        "$ml/fit/report.json:fit_report.json" \
        "$ml/fit/validation_extra.json:validation_extra.json" \
        "$ml/conversion/conversion_coefficients.csv:conversion_coefficients.csv" \
        "$ml/conversion/conversion_residuals_by_ml_bin.csv:conversion_residuals_by_ml_bin.csv" \
        "$ml/conversion/report.json:conversion_report.json" \
        "$mw/Mw_validation.yaml:Mw_validation.yaml" \
        "$mw/usgs_cross_validation.csv:usgs_cross_validation.csv" \
    ; do
        src="${pair%%:*}"
        name="${pair##*:}"
        if [[ -f "$src" ]]; then
            cp -- "$src" "$dst/$name"
            echo "  $name"
        else
            echo "  SKIPPED $name (no $src)" >&2
        fi
    done
    echo
    echo "Now run the tests, which check the tables against the released catalog:"
    echo "  python -m pytest tests/test_magnitudes.py"
}

if [[ $# -eq 0 ]]; then
    set -- anchor conversion
fi
for s in "$@"; do
    stage "$s"
done
echo
echo "done"
