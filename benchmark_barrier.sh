#!/bin/bash
# Benchmark: mipx barrier vs HiGHS IPX vs cuOpt barrier
# Usage: ./benchmark_barrier.sh [--netlib|--miplib|--all] [--time-limit N]
set -euo pipefail

cd "$(dirname "$0")"

MIPX=./build/mipx-solve
MODE="--all"
TLIMIT="120"
RTOL=1e-4  # relative tolerance for objective comparison

usage() {
    echo "Usage: ./benchmark_barrier.sh [--netlib|--miplib|--all] [--time-limit N]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --netlib|--miplib|--all)
            MODE="$1"
            shift
            ;;
        --time-limit)
            if [[ $# -lt 2 ]]; then
                echo "error: --time-limit requires a value" >&2
                usage >&2
                exit 2
            fi
            TLIMIT="$2"
            shift 2
            ;;
        --time-limit=*)
            TLIMIT="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$TLIMIT" =~ ^[0-9]+$ ]] || [[ "$TLIMIT" -le 0 ]]; then
    echo "error: --time-limit must be a positive integer (seconds)" >&2
    exit 2
fi

# Parse solu file into associative array
declare -A SOLU_VALS
load_solu() {
    local f="$1"
    [[ -f "$f" ]] || return
    while IFS= read -r line; do
        if [[ "$line" =~ ^=opt=\ +([^ ]+)\ +([^ ]+) ]]; then
            SOLU_VALS["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
        fi
    done < "$f"
}

# Run mipx barrier (force GPU with min thresholds at 0)
run_mipx() {
    local mps="$1"
    local out rc=0
    out=$(timeout "$TLIMIT" $MIPX "$mps" --barrier --no-presolve --relax-integrality --gpu --gpu-min-rows 0 --gpu-min-nnz 0 --quiet 2>&1) || rc=$?
    local obj status iters time_s
    obj=$(echo "$out" | grep "^Objective:" | awk '{print $2}')
    status=$(echo "$out" | grep "^Status:" | awk '{print $2}')
    iters=$(echo "$out" | grep "^Iterations:" | awk '{print $2}')
    time_s=$(echo "$out" | grep "^Time:" | awk '{print $2}' | sed 's/s//')
    if [[ -z "${status:-}" ]]; then
        if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
            status="timeout"
        elif [[ "$rc" -ne 0 ]]; then
            status="error"
        else
            status="timeout"
        fi
    fi
    echo "${status:-timeout}|${obj:-NA}|${iters:-NA}|${time_s:-NA}"
}

# Run HiGHS IPM (no presolve). Only works for pure LP instances.
# For MIP instances, pass relax=1 to skip HiGHS (it has no --relax-integrality CLI flag).
run_highs() {
    local mps="$1" relax="${2:-0}"
    if [[ "$relax" == "1" ]]; then
        echo "skip|NA|NA|NA"
        return
    fi
    local out
    out=$(timeout "$TLIMIT" highs "$mps" --solver=ipm --presolve=off 2>&1) || true
    local obj status iters time_s
    obj=$(echo "$out" | grep "^Objective value" | tail -1 | awk '{print $NF}')
    status=$(echo "$out" | grep "^Model status" | awk -F': ' '{print $2}')
    iters=$(echo "$out" | grep "^IPM       iterations:" | awk '{print $NF}')
    time_s=$(echo "$out" | grep "^HiGHS run time" | awk '{print $NF}')
    echo "${status:-timeout}|${obj:-NA}|${iters:-NA}|${time_s:-NA}"
}

# Run cuOpt barrier (method 3)
run_cuopt() {
    local mps="$1"
    local out
    out=$(timeout "$TLIMIT" cuopt_cli "$mps" --relaxation --method 3 2>&1) || true
    local obj status iters time_s
    # Parse objective from "Objective <value>" line (not the iteration log)
    obj=$(echo "$out" | grep -P "^Objective\s+[+-]" | tail -1 | awk '{print $2}')
    # Status
    if echo "$out" | grep -q "Optimal solution found"; then
        status="Optimal"
    elif echo "$out" | grep -q "Problem is infeasible"; then
        status="Infeasible"
    else
        status="timeout"
    fi
    # Iterations from "Optimal solution found in N iterations"
    iters=$(echo "$out" | grep -oP "found in \K\d+" | head -1)
    # Time from "Barrier finished in X.XXs seconds"
    time_s=$(echo "$out" | grep -oP "Barrier finished in \K[\d.]+" | head -1)
    echo "${status:-timeout}|${obj:-NA}|${iters:-NA}|${time_s:-NA}"
}

# Compare objectives
compare_obj() {
    python3 -c "
import sys
a, b = '$1', '$2'
if a == 'NA' or b == 'NA':
    sys.exit(1)
a, b = float(a), float(b)
denom = max(abs(b), 1.0)
rel = abs(a - b) / denom
sys.exit(0 if rel < $RTOL else 1)
" 2>/dev/null
}

print_header() {
    printf "\033[1m%-18s | %-24s | %-24s | %-24s | %s\033[0m\n" \
        "Instance" "mipx barrier" "HiGHS IPX" "cuOpt barrier" "Match"
    printf "%-18s-+-%-24s-+-%-24s-+-%-24s-+-%s\n" \
        "------------------" "------------------------" "------------------------" "------------------------" "------"
}

run_set() {
    local dir="$1" solu_file="$2" is_mip="${3:-0}"
    load_solu "$solu_file"
    print_header

    local mipx_pass=0 mipx_fail=0 mipx_faster_h=0 mipx_faster_c=0 total=0 solved_h=0 solved_c=0
    local mipx_total_time=0 highs_total_time=0 cuopt_total_time=0

    for mps in "$dir"/*.mps.gz; do
        [[ -f "$mps" ]] || continue
        local name
        name=$(basename "$mps" .mps.gz)
        total=$((total + 1))

        # Run all three solvers
        local m_status m_obj m_iters m_time
        local h_status h_obj h_iters h_time
        local c_status c_obj c_iters c_time

        IFS='|' read -r m_status m_obj m_iters m_time <<< "$(run_mipx "$mps")"
        IFS='|' read -r h_status h_obj h_iters h_time <<< "$(run_highs "$mps" "$is_mip")"
        IFS='|' read -r c_status c_obj c_iters c_time <<< "$(run_cuopt "$mps")"

        # Format timing strings
        local m_str h_str c_str
        if [[ "$m_status" == "Optimal" ]]; then
            m_str="${m_time}s / ${m_iters}it"
        else
            m_str="$m_status"
        fi
        if [[ "$h_status" == "Optimal" ]]; then
            h_str="${h_time}s / ${h_iters}it"
        else
            h_str="$h_status"
        fi
        if [[ "$c_status" == "Optimal" ]]; then
            c_str="${c_time}s / ${c_iters}it"
        else
            c_str="$c_status"
        fi

        # Objective match check (compare against HiGHS or cuOpt as reference)
        local match="ok"
        if [[ "$m_status" != "Optimal" ]]; then
            match="mipx:$m_status"
        elif [[ "$h_status" == "Optimal" ]]; then
            if ! compare_obj "$m_obj" "$h_obj"; then
                match="MISMATCH(h)"
            fi
        elif [[ "$c_status" == "Optimal" ]]; then
            if ! compare_obj "$m_obj" "$c_obj"; then
                match="MISMATCH(c)"
            fi
        fi

        # Check against solu (only for pure LP, not LP relaxation of MIPs)
        if [[ "$is_mip" != "1" ]]; then
            local expected="${SOLU_VALS[$name]:-}"
            if [[ -n "$expected" && "$m_status" == "Optimal" ]]; then
                if ! compare_obj "$m_obj" "$expected"; then
                    match="${match}+solu"
                fi
            fi
        fi

        # Color coding
        local color="\033[0m"
        if [[ "$match" == "ok" ]]; then
            color="\033[32m"
            mipx_pass=$((mipx_pass + 1))
        else
            color="\033[31m"
            mipx_fail=$((mipx_fail + 1))
        fi

        # Track fastest vs HiGHS
        if [[ "$m_status" == "Optimal" && "$h_status" == "Optimal" && "$m_time" != "NA" && "$h_time" != "NA" && -n "$m_time" && -n "$h_time" ]]; then
            solved_h=$((solved_h + 1))
            python3 -c "exit(0 if float('$m_time') < float('$h_time') else 1)" 2>/dev/null && mipx_faster_h=$((mipx_faster_h + 1))
        fi
        # Track fastest vs cuOpt
        if [[ "$m_status" == "Optimal" && "$c_status" == "Optimal" && "$m_time" != "NA" && "$c_time" != "NA" && -n "$m_time" && -n "$c_time" ]]; then
            solved_c=$((solved_c + 1))
            python3 -c "exit(0 if float('$m_time') < float('$c_time') else 1)" 2>/dev/null && mipx_faster_c=$((mipx_faster_c + 1))
        fi

        # Accumulate times
        [[ "$m_time" != "NA" && -n "$m_time" ]] && mipx_total_time=$(python3 -c "print(f'{$mipx_total_time + float(\"$m_time\"):.2f}')")
        [[ "$h_time" != "NA" && -n "$h_time" ]] && highs_total_time=$(python3 -c "print(f'{$highs_total_time + float(\"$h_time\"):.2f}')")
        [[ "$c_time" != "NA" && -n "$c_time" ]] && cuopt_total_time=$(python3 -c "print(f'{$cuopt_total_time + float(\"$c_time\"):.2f}')")

        printf "${color}%-18s | %-24s | %-24s | %-24s | %s\033[0m\n" \
            "$name" "$m_str" "$h_str" "$c_str" "$match"
    done

    printf "%-18s-+-%-24s-+-%-24s-+-%-24s-+-%s\n" \
        "------------------" "------------------------" "------------------------" "------------------------" "------"
    printf "%-18s | %-24s | %-24s | %-24s |\n" \
        "TOTAL TIME" "${mipx_total_time}s" "${highs_total_time}s" "${cuopt_total_time}s"
    echo ""
    echo "Correctness: $mipx_pass/$total objectives match"
    echo "mipx faster than HiGHS:  $mipx_faster_h/$solved_h"
    echo "mipx faster than cuOpt:  $mipx_faster_c/$solved_c"
    if [[ "$solved_h" -gt 0 ]]; then
        local ratio
        ratio=$(python3 -c "print(f'{$highs_total_time / max($mipx_total_time, 0.01):.2f}')")
        echo "Overall speedup vs HiGHS: ${ratio}x"
    fi
    if [[ "$solved_c" -gt 0 ]] && python3 -c "exit(0 if $cuopt_total_time > 0 else 1)" 2>/dev/null; then
        local ratio
        ratio=$(python3 -c "print(f'{$cuopt_total_time / max($mipx_total_time, 0.01):.2f}')")
        echo "Overall speedup vs cuOpt: ${ratio}x"
    fi
    echo ""
}

if [[ "$MODE" == "--netlib" || "$MODE" == "--all" ]]; then
    echo ""
    echo "=========================================="
    echo "  NETLIB LP Benchmark (no presolve)"
    echo "=========================================="
    echo ""
    run_set "tests/data/netlib" "tests/data/netlib/netlib.solu"
fi

if [[ "$MODE" == "--miplib" || "$MODE" == "--all" ]]; then
    echo ""
    echo "=========================================="
    echo "  MIPLIB LP Relaxation Benchmark"
    echo "=========================================="
    echo ""
    run_set "tests/data/miplib" "tests/data/miplib/miplib.solu" 1
fi
