#!/usr/bin/env bash
#
# Produce and check the δHBV2.0 MTS validation artifacts.
#
#   ./scripts/run_validation.sh              # all legs, then report
#   ./scripts/run_validation.sh standalone   # BMI driven directly by Python
#   ./scripts/run_validation.sh ngen         # BMI driven by NextGen
#   ./scripts/run_validation.sh troute       # NextGen + Muskingum-Cunge routing
#   ./scripts/run_validation.sh csv          # NextGen driven by CSV forcing
#   ./scripts/run_validation.sh check        # compare existing artifacts only
#
# The ngen and troute legs need a built NextGen image (see docs/5-run_ngen.md):
#
#   NGEN_IMAGE=localbuild/ngen:latest        # override to use another tag
#   MOUNT_LOCAL_SRC=1                        # run the image against this
#                                            # checkout's dhbv2/hydrodl2/dmg
#                                            # instead of the installed copies
#
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NGEN_IMAGE="${NGEN_IMAGE:-localbuild/ngen:latest}"
MOUNT_LOCAL_SRC="${MOUNT_LOCAL_SRC:-0}"

DATA_DIR="${PKG_ROOT}/ngen_resources/data"
GEO="data/geo/camels_subset_hf2_2.gpkg"
CAT="cat-2453"
NEX="nex-2454"

OUT_NGEN="${PKG_ROOT}/output/validation"
OUT_TROUTE="${PKG_ROOT}/output/validation_troute"
OUT_CSV="${PKG_ROOT}/output/validation_csv"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# Mount this checkout's sources over the image's installed packages, so the
# ngen legs exercise the code you are editing rather than what was baked in.
src_mounts() {
    [[ "${MOUNT_LOCAL_SRC}" == "1" ]] || return 0
    local sp=/ngen/.venv/lib/python3.9/site-packages
    local repos="${PKG_ROOT}/.."
    for pkg in dhbv2:dhbv2_mts hydrodl2:hydrodl2 dmg:dmg; do
        local name="${pkg%%:*}" dir="${pkg##*:}"
        local path="${repos}/${dir}/src/${name}"
        [[ -d "${path}" ]] && printf -- '-v\n%s:%s/%s:ro\n' "${path}" "${sp}" "${name}"
    done
}

run_ngen() {
    local realization="$1" outdir="$2"
    mkdir -p "${outdir}"
    # Clear only ngen's own artifacts: the standalone leg writes its .npy into
    # this same directory and must survive.
    rm -rf "${outdir}/stream_output"
    rm -f "${outdir}"/cat-*.csv "${outdir}"/nex-*
    mapfile -t mounts < <(src_mounts)
    docker run --rm \
        -v "${DATA_DIR}:/ngen/data" \
        -v "${outdir}:/ngen/output" \
        ${mounts[@]+"${mounts[@]}"} \
        -w /ngen \
        "${NGEN_IMAGE}" \
        ngen "${GEO}" "${CAT}" "${GEO}" "${NEX}" \
        "data/dhbv_2_mts/realizations/${realization}"
}

leg_standalone() {
    say "Leg 1/4: standalone BMI (~1 min, 26088 hourly steps)"
    mkdir -p "${OUT_NGEN}"
    DHBV2_MTS_OUTPUT="${OUT_NGEN}/standalone_${CAT}.npy" \
        python "${PKG_ROOT}/scripts/mts_forward_example.py"
}

leg_ngen() {
    say "Leg 2/4: NextGen, full window (~1 min)"
    run_ngen realization_validation_cat-2453.json "${OUT_NGEN}"
}

leg_troute() {
    say "Leg 3/4: NextGen + t-route, full window (~1.5 min)"
    run_ngen realization_troute_cat-2453.json "${OUT_TROUTE}"
}

leg_csv() {
    say "Leg 4/4: NextGen with CsvPerFeature forcing, example window (~25 s)"
    # Forcing CSVs are named for the period they cover, so match by prefix
    # rather than an exact filename.
    local existing=()
    shopt -s nullglob
    existing=("${DATA_DIR}/forcing/${CAT}"*.csv)
    shopt -u nullglob
    (( ${#existing[@]} )) || python "${PKG_ROOT}/scripts/make_csv_forcing.py" "${CAT}"
    run_ngen realization_cat-2453.json "${OUT_CSV}"
}

leg_check() {
    say "Comparing artifacts against the committed benchmark"
    cd "${PKG_ROOT}"
    python -m pytest tests/test_validation.py -v
}

case "${1:-all}" in
    standalone) leg_standalone ;;
    ngen)       leg_ngen ;;
    troute)     leg_troute ;;
    csv)        leg_csv ;;
    check)      leg_check ;;
    all)        leg_standalone; leg_ngen; leg_troute; leg_csv; leg_check ;;
    *)          echo "Unknown leg '$1'. Use: standalone | ngen | troute | csv | check | all" >&2
                exit 2 ;;
esac
