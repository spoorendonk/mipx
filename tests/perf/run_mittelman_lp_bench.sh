#!/usr/bin/env bash
# Run mipx-solve LP benchmark matching Mittelman's LPopt configuration.
#
# Mittelman LPopt: find optimal basic solution using simplex.
# Reference: https://plato.asu.edu/ftp/lpopt.html
#
# Mittelman parameters:
#   - Time limit: 15000 seconds (per instance)
#   - Threads: 1 (single-threaded simplex)
#   - Instances: ~65 LP instances from multiple sources
#
# This script runs the LP benchmark on Mittelman LP instances and/or Netlib
# instances and produces a CSV suitable for regression comparison.
#
# Usage:
#   ./tests/perf/run_mittelman_lp_bench.sh \
#       --binary ./build/mipx-solve \
#       --output /tmp/mittelman_lp.csv \
#       [--mittelman-dir ./tests/data/mittelman_lp] \
#       [--netlib-dir ./tests/data/netlib] \
#       [--repeats 3] \
#       [--time-limit 15000] \
#       [--solver-arg --quiet]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BIN=""
MITTELMAN_DIR=""
NETLIB_DIR=""
OUT=""
REPEATS=3
TIME_LIMIT=15000
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            BIN="$2"
            shift 2
            ;;
        --mittelman-dir)
            MITTELMAN_DIR="$2"
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
        --time-limit)
            TIME_LIMIT="$2"
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

# Defaults
if [[ -z "${MITTELMAN_DIR}" ]]; then
    MITTELMAN_DIR="${ROOT_DIR}/tests/data/mittelman_lp"
fi
if [[ -z "${NETLIB_DIR}" ]]; then
    NETLIB_DIR="${ROOT_DIR}/tests/data/netlib"
fi

if [[ -z "${BIN}" || -z "${OUT}" ]]; then
    echo "Usage: $0 --binary <path> --output <csv> [--mittelman-dir <dir>] [--netlib-dir <dir>] [--repeats N] [--time-limit S] [--solver-arg ARG]" >&2
    exit 1
fi

if [[ ! -x "${BIN}" ]]; then
    echo "Binary not executable: ${BIN}" >&2
    exit 1
fi

# Collect all .mps.gz instances from both directories (dedup by name).
declare -A instance_map
shopt -s nullglob

if [[ -d "${MITTELMAN_DIR}" ]]; then
    for f in "${MITTELMAN_DIR}"/*.mps.gz; do
        name="$(basename "${f}" .mps.gz)"
        instance_map["${name}"]="${f}"
    done
fi

if [[ -d "${NETLIB_DIR}" ]]; then
    for f in "${NETLIB_DIR}"/*.mps.gz; do
        name="$(basename "${f}" .mps.gz)"
        # Don't overwrite if already present from Mittelman dir
        if [[ -z "${instance_map["${name}"]:-}" ]]; then
            instance_map["${name}"]="${f}"
        fi
    done
fi

shopt -u nullglob

if [[ ${#instance_map[@]} -eq 0 ]]; then
    echo "No .mps.gz instances found in ${MITTELMAN_DIR} or ${NETLIB_DIR}" >&2
    echo "Download first: ./tests/data/download_mittelman_lp.sh" >&2
    exit 1
fi

echo "[mittelman-lp] Found ${#instance_map[@]} LP instances"

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

# Sort instance names for deterministic output
mapfile -t sorted_names < <(printf '%s\n' "${!instance_map[@]}" | sort)

for name in "${sorted_names[@]}"; do
    inst="${instance_map["${name}"]}"

    times=()
    works=()
    status="ok"

    for ((r = 0; r < REPEATS; ++r)); do
        log_file="$(mktemp)"
        start_ns="$(date +%s%N)"
        if ! timeout "${TIME_LIMIT}" "${BIN}" "${inst}" "${EXTRA_ARGS[@]}" >"${log_file}" 2>&1; then
            exit_code=$?
            if [[ ${exit_code} -eq 124 ]]; then
                status="time_limit"
            else
                status="solve_error"
            fi
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

    if [[ "${status}" == "solve_error" || "${status}" == "parse_error" || "${status}" == "time_limit" ]]; then
        echo "${name},,,${status}" >> "${OUT}"
        echo "  ${name}: ${status}"
        continue
    fi

    median_time="$(median_value "${times[@]}")"
    median_work="$(median_value "${works[@]}")"

    echo "${name},${median_time},${median_work},${status}" >> "${OUT}"
    echo "  ${name}: ${median_time}s, ${median_work} wu, ${status}"
done

echo "Wrote Mittelman LP benchmark CSV: ${OUT}"
