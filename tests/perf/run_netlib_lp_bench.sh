#!/usr/bin/env bash
# Run mipx-solve over Netlib .mps.gz files and emit median benchmark CSV.
#
# Usage:
#   ./tests/perf/run_netlib_lp_bench.sh \
#       --binary ./build/mipx-solve \
#       --netlib-dir ./tests/data/netlib \
#       --output /tmp/results.csv \
#       [--repeats 3] \
#       [--solver-arg --quiet]

set -euo pipefail

BIN=""
NETLIB_DIR=""
OUT=""
REPEATS=3
EXTRA_ARGS=()

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
        --solver-arg)
            EXTRA_ARGS+=("$2")
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${BIN}" || -z "${NETLIB_DIR}" || -z "${OUT}" ]]; then
    echo "Usage: $0 --binary <path> --netlib-dir <dir> --output <csv> [--repeats N] [--solver-arg ARG]" >&2
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
echo "instance,time_seconds,work_units,status" > "${OUT}"

median_value() {
    printf "%s\n" "$@" \
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
}

for inst in "${instances[@]}"; do
    name="$(basename "${inst}")"
    name="${name%.mps.gz}"

    times=()
    works=()
    status="ok"

    for ((r = 0; r < REPEATS; ++r)); do
        log_file="$(mktemp)"
        start_ns="$(date +%s%N)"
        if ! "${BIN}" "${inst}" "${EXTRA_ARGS[@]}" >"${log_file}" 2>&1; then
            status="solve_error"
            rm -f "${log_file}"
            break
        fi
        end_ns="$(date +%s%N)"
        dt_ns=$((end_ns - start_ns))
        t="$(awk -v ns="${dt_ns}" 'BEGIN { printf "%.6f", ns / 1000000000.0 }')"
        w="$(
            awk -F: '
                /^Work units:/ || /^Work:/ {
                    v = $2
                    gsub(/^[ \t]+|[ \t]+$/, "", v)
                    print v
                    exit
                }
            ' "${log_file}"
        )"
        run_status="$(
            awk -F: '
                /^Status:/ {
                    v = $2
                    gsub(/^[ \t]+|[ \t]+$/, "", v)
                    print tolower(v)
                    exit
                }
            ' "${log_file}"
        )"
        rm -f "${log_file}"

        if [[ -z "${w}" ]]; then
            status="parse_error"
            break
        fi
        if [[ -z "${run_status}" ]]; then
            run_status="unknown"
        fi
        run_status="${run_status// /_}"
        if [[ "${status}" == "ok" ]]; then
            status="${run_status}"
        elif [[ "${status}" != "${run_status}" ]]; then
            status="mixed_status"
        fi

        times+=("${t}")
        works+=("${w}")
    done

    if [[ "${status}" == "solve_error" || "${status}" == "parse_error" ]]; then
        echo "${name},,,${status}" >> "${OUT}"
        continue
    fi

    median_time="$(median_value "${times[@]}")"
    median_work="$(median_value "${works[@]}")"

    echo "${name},${median_time},${median_work},${status}" >> "${OUT}"
done

echo "Wrote benchmark CSV: ${OUT}"
