#!/usr/bin/env bash
# Run mipx-solve over MIPLIB .mps.gz files and emit median benchmark CSV.
#
# Usage:
#   ./tests/perf/run_miplib_mip_bench.sh \
#       --binary ./build/mipx-solve \
#       --miplib-dir ./tests/data/miplib \
#       --output /tmp/results.csv \
#       [--repeats 1] [--threads 1] [--time-limit 60] [--node-limit 100000] \
#       [--gap-tol 1e-4] [--solver-arg --no-cuts]

set -euo pipefail

BIN=""
MIPLIB_DIR=""
OUT=""
REPEATS=1
THREADS=1
TIME_LIMIT=60
NODE_LIMIT=100000
GAP_TOL=1e-4
MAX_INSTANCES=0
INSTANCE_FILTER=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            BIN="$2"
            shift 2
            ;;
        --miplib-dir)
            MIPLIB_DIR="$2"
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
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --node-limit)
            NODE_LIMIT="$2"
            shift 2
            ;;
        --gap-tol)
            GAP_TOL="$2"
            shift 2
            ;;
        --max-instances)
            MAX_INSTANCES="$2"
            shift 2
            ;;
        --instances)
            INSTANCE_FILTER="$2"
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

if [[ -z "${BIN}" || -z "${MIPLIB_DIR}" || -z "${OUT}" ]]; then
    echo "Usage: $0 --binary <path> --miplib-dir <dir> --output <csv> [--repeats N] [--threads N] [--time-limit S] [--node-limit N] [--gap-tol G] [--max-instances N] [--instances a,b,c] [--solver-arg ARG]" >&2
    exit 1
fi

if [[ ! -x "${BIN}" ]]; then
    echo "Binary not executable: ${BIN}" >&2
    exit 1
fi

shopt -s nullglob
instances=("${MIPLIB_DIR}"/*.mps.gz)
shopt -u nullglob
if [[ ${#instances[@]} -eq 0 ]]; then
    echo "No .mps.gz instances found in ${MIPLIB_DIR}" >&2
    exit 1
fi

if [[ -n "${INSTANCE_FILTER}" ]]; then
    filtered=()
    IFS=',' read -r -a names <<< "${INSTANCE_FILTER}"
    for n in "${names[@]}"; do
        n="${n// /}"
        [[ -z "${n}" ]] && continue
        p="${MIPLIB_DIR}/${n}.mps.gz"
        if [[ -f "${p}" ]]; then
            filtered+=("${p}")
        else
            echo "Warning: instance not found: ${n}" >&2
        fi
    done
    instances=("${filtered[@]}")
fi

if [[ "${MAX_INSTANCES}" -gt 0 && ${#instances[@]} -gt "${MAX_INSTANCES}" ]]; then
    instances=("${instances[@]:0:${MAX_INSTANCES}}")
fi

if [[ ${#instances[@]} -eq 0 ]]; then
    echo "No instances selected for benchmarking." >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT}")"
echo "instance,time_seconds,work_units,nodes,lp_iterations,status" > "${OUT}"

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
    nodes=()
    lp_iters=()
    status="ok"

    for ((r = 0; r < REPEATS; ++r)); do
        log_file="$(mktemp)"
        start_ns="$(date +%s%N)"
        if ! "${BIN}" "${inst}" \
            --threads "${THREADS}" \
            --time-limit "${TIME_LIMIT}" \
            --node-limit "${NODE_LIMIT}" \
            --gap-tol "${GAP_TOL}" \
            "${EXTRA_ARGS[@]}" >"${log_file}" 2>&1; then
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
        n="$(
            awk -F: '
                /^Nodes:/ {
                    v = $2
                    gsub(/^[ \t]+|[ \t]+$/, "", v)
                    print v
                    exit
                }
            ' "${log_file}"
        )"
        li="$(
            awk -F: '
                /^LP iterations:/ {
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

        if [[ -z "${w}" || -z "${n}" || -z "${li}" ]]; then
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
        nodes+=("${n}")
        lp_iters+=("${li}")
    done

    if [[ "${status}" == "solve_error" || "${status}" == "parse_error" ]]; then
        echo "${name},,,,,${status}" >> "${OUT}"
        continue
    fi

    median_time="$(median_value "${times[@]}")"
    median_work="$(median_value "${works[@]}")"
    median_nodes="$(median_value "${nodes[@]}")"
    median_lpiters="$(median_value "${lp_iters[@]}")"

    echo "${name},${median_time},${median_work},${median_nodes},${median_lpiters},${status}" >> "${OUT}"
done

echo "Wrote benchmark CSV: ${OUT}"
