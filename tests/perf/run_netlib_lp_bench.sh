#!/usr/bin/env bash
# Run mipx-solve over Netlib .mps.gz files and emit median runtime CSV.
#
# Usage:
#   ./tests/perf/run_netlib_lp_bench.sh \
#       --binary ./build/mipx-solve \
#       --netlib-dir ./tests/data/netlib \
#       --output /tmp/results.csv \
#       [--repeats 3]

set -euo pipefail

BIN=""
NETLIB_DIR=""
OUT=""
REPEATS=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            BIN="$2"
            shift 2
            ;;
        --netlib-dir)
            NETLIB_DIR="$2"
            shift 2
            ;;
        --output)
            OUT="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${BIN}" || -z "${NETLIB_DIR}" || -z "${OUT}" ]]; then
    echo "Usage: $0 --binary <path> --netlib-dir <dir> --output <csv> [--repeats N]" >&2
    exit 1
fi

if [[ ! -x "${BIN}" ]]; then
    echo "Binary not executable: ${BIN}" >&2
    exit 1
fi

shopt -s nullglob
instances=("${NETLIB_DIR}"/*.mps.gz)
shopt -u nullglob
if [[ ${#instances[@]} -eq 0 ]]; then
    echo "No .mps.gz instances found in ${NETLIB_DIR}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT}")"
echo "instance,time_seconds,status" > "${OUT}"

for inst in "${instances[@]}"; do
    name="$(basename "${inst}")"
    name="${name%.mps.gz}"

    times=()
    status="ok"
    for ((r = 0; r < REPEATS; ++r)); do
        start_ns="$(date +%s%N)"
        if ! "${BIN}" "${inst}" >/dev/null 2>&1; then
            status="solve_error"
            break
        fi
        end_ns="$(date +%s%N)"
        dt_ns=$((end_ns - start_ns))
        t="$(awk -v ns="${dt_ns}" 'BEGIN { printf "%.6f", ns / 1000000000.0 }')"
        times+=("${t}")
    done

    if [[ "${status}" != "ok" ]]; then
        echo "${name},,${status}" >> "${OUT}"
        continue
    fi

    median="$(
        printf "%s\n" "${times[@]}" \
            | sort -n \
            | awk '{
                a[NR] = $1
            } END {
                n = NR
                if (n == 0) exit 1
                if (n % 2 == 1) {
                    printf "%.6f", a[(n + 1) / 2]
                } else {
                    printf "%.6f", (a[n / 2] + a[n / 2 + 1]) / 2.0
                }
            }'
    )"

    echo "${name},${median},ok" >> "${OUT}"
done

echo "Wrote benchmark CSV: ${OUT}"
